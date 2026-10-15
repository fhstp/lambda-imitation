"""Five-seed POPGym benchmark: recurrent SAC with/without lambda discrepancy.

See examples/POPGYM.md for installation, weekend commands and interpretation.
The supervisor never imports JAX; each task/method/seed runs in a fresh process.
Supported tasks use compiled POPJym rollouts; other tasks use Gymnasium on host.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def write_json(path, data):
    """Atomic replacement: interrupted writes cannot corrupt a saved result."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("popgym_output"))
    p.add_argument("--envs", nargs="+", help="exact registered IDs; default: all discrete tasks")
    p.add_argument("--methods", nargs="+", choices=("lambda", "sac"), default=["lambda", "sac"])
    p.add_argument("--backend", choices=("auto", "gymnasium", "jax"), default="auto",
                   help="auto: audited POPJym ports with explicit Gymnasium fallbacks; jax: require all selected tasks to be supported")
    p.add_argument("--jax-train-steps", type=int, default=256, help="interactions per compiled JAX training call; carries persist")
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=250_000, help="training interactions, excluding prefill")
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--final-episodes", type=int, default=100)
    p.add_argument("--max-episode-steps", type=int, default=10_000, help="safety guard; exceeding it fails the run")
    p.add_argument("--prefill-steps", type=int, default=0, help="0: batch size × sample window")
    p.add_argument("--collect-steps", type=int, default=16, help="collect this many steps, then run the matching update block")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128, help="replay sequences per update")
    p.add_argument("--online-buffer-size", type=int, default=100_000)
    p.add_argument("--burn-in-length", type=int, default=10)
    p.add_argument("--sequence-length", type=int, default=40)
    p.add_argument("--lambda-truncation", type=int, default=50)
    p.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
    p.add_argument("--memory-hidden-dim", type=int, default=512)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--fe-lr", type=float, default=1e-5)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--lambda-coef", type=float, default=0.1)
    p.add_argument("--lambda1", type=float, default=0.05)
    p.add_argument("--lambda2", type=float, default=0.85)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--use-sac", action="store_true", help="include entropy in critic targets (default: reward-only, like the Battleship study)")
    p.add_argument("--critic-layer-norm", action="store_true")
    p.add_argument("--gpus", nargs="+", help="CUDA device IDs, e.g. --gpus 0 1 2")
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--workers", type=int, default=1, help="concurrency without an explicit GPU list")
    p.add_argument("--resume", action="store_true", help="resume unfinished checkpoints; skip matching completed runs")
    p.add_argument("--list", action="store_true", help="preflight spaces, print included/excluded tasks, then exit")
    p.add_argument("--summarize", action="store_true", help="rebuild summaries without training")
    p.add_argument("--plot", action="store_true", help="write per-task learning curves after the suite or with --summarize (requires matplotlib)")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="lambda-popgym")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--task-file", type=Path, help=argparse.SUPPRESS)
    return p


def validate_args(args):
    positive = ("num_seeds", "steps", "eval_every", "eval_episodes", "final_episodes",
                "max_episode_steps", "collect_steps", "updates_per_step", "batch_size",
                "online_buffer_size", "sequence_length", "lambda_truncation",
                "memory_hidden_dim", "head_dim", "workers", "workers_per_gpu", "jax_train_steps")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.burn_in_length < 0 or args.prefill_steps < 0 or args.projection_dim < 0:
        raise ValueError("Burn-in, prefill and projection sizes cannot be negative")
    window = args.burn_in_length + args.sequence_length + args.lambda_truncation
    warm = args.prefill_steps or args.batch_size * window
    if args.online_buffer_size <= window + 1 or warm < window + 2:
        raise ValueError("Replay and prefill must fit a complete sample window plus successors")
    if not 0 <= args.seed < 2**32 or args.seed + args.num_seeds > 2**32:
        raise ValueError("Seeds must fit uint32")
    if not 0 <= args.lambda1 <= args.lambda2 <= 1 or not 0 <= args.gamma <= 1:
        raise ValueError("Require 0 <= lambda1 <= lambda2 <= 1 and gamma in [0, 1]")
    if not 0 < args.tau <= 1 or args.alpha < 0 or args.lambda_coef < 0:
        raise ValueError("Invalid tau, alpha or discrepancy coefficient")
    if any(getattr(args, n) <= 0 for n in ("fe_lr", "actor_lr", "critic_lr")):
        raise ValueError("Learning rates must be positive")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("Duplicate methods")


def make_env(env_id):
    """One-hot flatten partial observations; ravel MultiDiscrete actions only.

    No get_state(), action masks derived from hidden state, Markovian wrapper,
    reward rescaling, or extra observations are used.
    """
    import gymnasium as gym
    import popgym  # noqa: F401 -- registration
    return wrap_env(gym.make(env_id))


def wrap_env(env):
    import gymnasium as gym
    import numpy as np

    if not isinstance(env.action_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        env.close()
        raise ValueError("Only Discrete/MultiDiscrete actions are supported")

    class CategoricalAction(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.original_space = env.action_space
            n = int(np.prod(self.original_space.nvec)) if isinstance(self.original_space, gym.spaces.MultiDiscrete) else int(self.original_space.n)
            self.action_space = gym.spaces.Discrete(n)

        def action(self, action):
            if isinstance(self.original_space, gym.spaces.MultiDiscrete):
                coordinates = np.asarray(np.unravel_index(int(action), self.original_space.nvec.ravel()))
                return coordinates.reshape(self.original_space.shape) + self.original_space.start
            return int(action) + self.original_space.start

    return gym.wrappers.FlattenObservation(CategoricalAction(env))


def discover(env_ids=None):
    import gymnasium as gym
    import numpy as np
    import popgym.envs

    if env_ids is None and not popgym.envs.ALL_NAVIGATION:
        raise RuntimeError("Full suite needs maze tasks: pip install 'popgym[navigation]' (or select --envs explicitly)")
    registered = {item["id"] for item in popgym.envs.ALL.values()}
    selected = sorted(registered if env_ids is None else set(env_ids))
    unknown = set(selected) - registered
    if unknown:
        raise ValueError(f"Unknown suite environment IDs: {sorted(unknown)}")
    included, excluded = [], []
    for env_id in selected:
        env = gym.make(env_id)
        try:
            space = env.action_space
            if isinstance(space, gym.spaces.Box):
                excluded.append({"env_id": env_id, "reason": "continuous actions (not implemented for Retrace lambda)"})
                continue
            original = str(space)
        finally:
            env.close()
        env = make_env(env_id)
        try:
            obs, _ = env.reset(seed=0)
            if not np.all(np.isfinite(obs)):
                raise ValueError(f"{env_id}: nonfinite initial observation")
            # Fail before scheduling 420 runs if an adapter cannot actually step.
            env.step(0)
            included.append({"env_id": env_id, "obs_shape": list(env.observation_space.shape),
                             "num_actions": int(env.action_space.n), "original_action_space": original})
        finally:
            env.close()
    if not included:
        raise ValueError("No discrete tasks selected")
    return included, excluded


def run_config(args, task):
    """Scheduling/logging knobs do not change a resumable experiment."""
    excluded = {"output_dir", "envs", "methods", "num_seeds", "seed", "steps", "gpus",
                "workers", "workers_per_gpu", "resume", "list", "summarize", "plot", "task_file",
                "wandb", "wandb_project", "wandb_entity"}
    config = {k: v for k, v in vars(args).items() if k not in excluded}
    return {**config, **task, "retrace": True, "fake_onpolicy_loss": False,
            "stop_actor_fe": True, "use_prev_action": True, "stored_carry": False,
            "lambda_td_loss": "mse", "discrepancy_gradient": "feature_extractor_only"}


def source_fingerprint():
    import importlib.util

    # Locate the installed package without importing it/initializing JAX in the
    # supervisor. Freeze one fingerprint for the entire planned suite.
    package = Path(importlib.util.find_spec("lambda_imitation").origin).parent
    paths = [Path(__file__), Path(__file__).with_name("popgym_jax.py"), package / "iqlearn.py", package / "buffer.py", package / "utils.py"]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def return_targets(rewards, entropies, gamma, alpha=0.0):
    """Executed-trajectory MC; entropy belongs at the NEXT state, not current."""
    import numpy as np

    result = np.zeros(len(rewards), dtype=np.float64)
    value = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_entropy = entropies[t + 1] if t + 1 < len(rewards) else 0.0
        value = rewards[t] + gamma * (value + alpha * next_entropy)
        result[t] = value
    return result


def make_insert(buffer_fns, buffer_size):
    """Insert a chronological host batch without allowing newest→oldest windows."""
    import jax

    @jax.jit
    def insert(buffer, transitions):
        def body(buffer, transition):
            return buffer_fns.add(buffer, transition, transition["terminated"].astype(bool)), None
        buffer = jax.lax.scan(body, buffer, transitions)[0]
        newest = (buffer.pos - 1) % buffer_size
        return buffer._replace(sampling_ok=buffer.sampling_ok.at[newest].set(False))
    return insert


def evaluate(env, state, fns, debug, seed, episode_count, mode, carry_dim, args):
    """Frozen online policy, independent environment and reproducible episode keys.

    Q errors are measured ONLY under stochastic actor rollouts: greedy returns
    are not the stochastic policy's Q target. These are noisy on-policy MC
    residuals, not posterior-averaged estimation errors or memory probes.
    POPGym time limits define finite episodes here; truncations stop bootstrap.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rows, errors, gaps = [], {k: [] for k in ("all", "early", "middle", "late")}, []
    action_dim = env.action_space.n
    policy_key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    for episode in range(episode_count):
        # Identical environment seeds across methods and policy modes.
        obs, _ = env.reset(seed=(seed + episode) % 2**32)
        carry = jnp.zeros(carry_dim, jnp.float32)
        prev = jnp.zeros(action_dim, jnp.float32)
        rewards, qs, entropies, ld = [], [], [], []
        for step in range(args.max_episode_steps):
            if mode == "random":
                action = int(rng.integers(action_dim))
            else:
                values, carry = debug.predict_values(state, jnp.asarray(obs, jnp.float32), carry, prev)
                values = jax.device_get(values)
                probs = values["probs"]
                policy_key, key = jax.random.split(policy_key)
                if mode == "critic":
                    action = int(np.argmax(values["q"].min(-1)))
                elif mode == "greedy":
                    action = int(np.argmax(probs))
                else:
                    action = int(jax.random.categorical(key, jnp.log(jnp.asarray(probs) + 1e-30)))
                if mode == "stochastic":
                    qs.append(float(values["q"][action].min()))
                    entropies.append(float(-np.sum(probs * np.log(np.maximum(probs, 1e-30)))))
                    if "lambda1_q" in values:
                        ld.append(float(values["lambda1_q"][action].min() - values["lambda2_q"][action].min()))
                prev = jax.nn.one_hot(action, action_dim)
            obs, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(reward) or not np.all(np.isfinite(obs)):
                raise FloatingPointError("Nonfinite evaluation reward/observation")
            rewards.append(float(reward))
            if terminated or truncated:
                break
        else:
            raise RuntimeError(f"Evaluation exceeded --max-episode-steps={args.max_episode_steps}")
        row = {"episode": episode, "return": sum(rewards), "length": len(rewards),
               "terminated": bool(terminated), "truncated": bool(truncated)}
        if qs:
            target = return_targets(rewards, entropies, args.gamma, args.alpha if args.use_sac else 0.0)
            residual = np.asarray(qs) - target
            errors["all"].extend(residual.tolist())
            # Relative episode thirds allow comparisons across diverse horizons.
            for stage, part in zip(("early", "middle", "late"), np.array_split(residual, 3)):
                errors[stage].extend(part.tolist())
            gaps.extend(ld)
            row.update(q_mc_bias=float(residual.mean()), q_mc_rmse=float(np.sqrt(np.mean(residual**2))))
        rows.append(row)
    rets = np.asarray([r["return"] for r in rows])
    metrics = {"return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.0,
               "length_mean": float(np.mean([r["length"] for r in rows])),
               "truncated_fraction": float(np.mean([r["truncated"] for r in rows]))}
    for stage, values in errors.items():
        if values:
            e = np.asarray(values)
            metrics[f"q_mc_rmse/{stage}"] = float(np.sqrt(np.mean(e**2)))
            metrics[f"q_mc_bias/{stage}"] = float(e.mean())
    if gaps:
        metrics["lambda_discrepancy_rms"] = float(np.sqrt(np.mean(np.asarray(gaps)**2)))
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite evaluation metrics")
    return {"metrics": metrics, "episodes": rows}


def train_one(args, task):
    import cloudpickle
    import importlib.metadata
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lambda_imitation.buffer import create_buffer
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnasium

    out = args.output_dir / task["env_id"] / task["method"] / f"seed{task['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    config = run_config(args, task)
    fingerprints = source_fingerprint()
    versions = {name: importlib.metadata.version(name) for name in ("popgym", "gymnasium", "jax", "flax", "optax", "cloudpickle")}
    checkpoint = out / "checkpoint.pkl"
    status = out / "status.json"
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        if old["experiment"] != config or old["source_sha256"] != fingerprints or old["versions"] != versions:
            raise ValueError("Existing run has different settings, source or package versions; choose a new --output-dir")
        if not args.resume:
            raise FileExistsError("Run already exists; use --resume or a new output directory")
    else:
        write_json(out / "config.json", {"experiment": config, "source_sha256": fingerprints,
                                        "versions": versions, "devices": [str(d) for d in jax.devices()]})
    env = make_env(task["env_id"])
    eval_env = make_env(task["env_id"])
    spec = env_spec_from_gymnasium(env)
    hp = Hyperparameters(batch_size=1, online_batch_size=args.batch_size,
                         online_buffer_size=args.online_buffer_size, fe_lr=args.fe_lr,
                         actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                         lambda_critic_lr=args.critic_lr, alpha=args.alpha, gamma=args.gamma,
                         tau=args.tau, burn_in_length=args.burn_in_length,
                         sequence_length=args.sequence_length, lambda_truncation=args.lambda_truncation,
                         lambda1=args.lambda1, lambda2=args.lambda2, lambda_coef=args.lambda_coef,
                         fake_onpolicy_loss=False, retrace=True, stop_actor_fe=True)
    state, fns, debug = create_iqlearn_from_env(
        spec, {"observations": jnp.zeros((1, *spec.obs_shape)), "actions": jnp.zeros((1, 1))},
        buffer_size=1, hp=hp, projection=args.projection_dim or None,
        memory_type=args.memory_type, memory_hidden_dim=args.memory_hidden_dim,
        use_prev_action=True, actor_dims=(args.head_dim,), critic_dims=(args.head_dim,),
        lambda1_critic_dims=(args.head_dim,), lambda2_critic_dims=(args.head_dim,),
        approximate_lambda=task["method"] == "lambda", critic_layer_norm=args.critic_layer_norm,
        debug=True, seed=task["seed"], use_sac=args.use_sac)
    carry_dim = 0 if args.memory_type == "identity" else args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1)
    zero_carry, zero_prev = jnp.zeros(carry_dim, jnp.float32), jnp.zeros(spec.action_dim, jnp.float32)
    shapes = {k: v.shape[1:] for k, v in state.online_buffer.info.items()}
    _, buffer_fns = create_buffer(shapes, args.online_buffer_size, 1, [], [])

    insert = make_insert(buffer_fns, args.online_buffer_size)

    @jax.jit
    def update(state, keys):
        def body(state, key):
            return fns.update(state, key)
        state, metrics = jax.lax.scan(body, state, keys)
        return state, jax.tree.map(lambda x: x.mean(), metrics)

    key = jax.random.PRNGKey(task["seed"])
    rng = np.random.default_rng(task["seed"])
    obs, _ = env.reset(seed=task["seed"])
    carry, prev = zero_carry, zero_prev
    steps, interactions, episode_steps, history = 0, 0, 0, []
    if args.resume and checkpoint.exists():
        # Trust only local checkpoints: cloudpickle is executable, not a safe input format.
        saved = cloudpickle.loads(checkpoint.read_bytes())
        if saved["config"] != config:
            raise ValueError("Checkpoint configuration mismatch")
        state = jax.tree.map(jnp.asarray, saved["state"])
        env.close()
        env, obs = saved["env"], saved["obs"]
        key, carry, prev = map(jnp.asarray, (saved["key"], saved["carry"], saved["prev"]))
        rng.bit_generator.state = saved["numpy_rng"]
        steps, interactions, episode_steps, history = (saved[n] for n in ("steps", "interactions", "episode_steps", "history"))
        write_json(out / "history.json", history)
        if steps >= args.steps and json.loads(status.read_text()).get("state") == "complete":
            print(f"Already complete: {task}", flush=True)
            env.close(); eval_env.close()
            return
        print(f"Resuming {task} at {steps} steps", flush=True)
    start = time.monotonic()
    starting_steps = steps
    wandb_run = None
    if args.wandb:
        import wandb
        run_id = hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id,
                               resume="allow", group=task["env_id"], name=f"{task['env_id']}-{task['method']}-{task['seed']}",
                               config=config, dir=str(out))
        wandb_run.define_metric("steps")
        wandb_run.define_metric("train/*", step_metric="steps")
        wandb_run.define_metric("eval/*", step_metric="steps")
        wandb_run.define_metric("final/*", step_metric="steps")

    def save():
        data = {"config": config, "state": jax.device_get(state), "env": env, "obs": obs,
                "key": np.asarray(key), "carry": np.asarray(carry), "prev": np.asarray(prev),
                "numpy_rng": rng.bit_generator.state, "steps": steps, "interactions": interactions,
                "episode_steps": episode_steps, "history": history}
        temporary = checkpoint.with_suffix(".pkl.tmp")
        with temporary.open("wb") as file:
            cloudpickle.dump(data, file)
        temporary.replace(checkpoint)
        write_json(out / "history.json", history)

    def collect(count, random=False):
        nonlocal state, key, obs, carry, prev, interactions, episode_steps
        transitions = []
        for _ in range(count):
            if random:
                action = int(rng.integers(spec.action_dim))
                prob = 1.0 / spec.action_dim
            else:
                key, act_key = jax.random.split(key)
                raw, carry, prob = fns.predict(state, jnp.asarray(obs, jnp.float32), carry, act_key,
                                               return_prob=True, prev_action=prev)
                action, prob = int(raw), float(prob)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            if not np.isfinite(reward) or not np.all(np.isfinite(next_obs)):
                raise FloatingPointError("Nonfinite training reward/observation")
            done = terminated or truncated
            transition = {"observations": np.asarray(obs, np.float32), "actions": np.asarray([action], np.float32),
                          "rewards": np.float32(reward), "terminated": np.float32(done)}
            if "behaviour_weight" in shapes:
                transition["behaviour_weight"] = np.float32(prob)
            transitions.append(transition)
            interactions += 1
            episode_steps += 1
            if done:
                next_obs, _ = env.reset()
                carry, prev, episode_steps = zero_carry, zero_prev, 0
            else:
                if episode_steps >= args.max_episode_steps:
                    raise RuntimeError("Training episode exceeded --max-episode-steps")
                prev = jax.nn.one_hot(action, spec.action_dim)
            obs = next_obs
        batch = {k: jnp.asarray(np.stack([t[k] for t in transitions])) for k in shapes}
        state = state._replace(online_buffer=insert(state.online_buffer, batch))

    def evaluation(final=False):
        # Intermediate cases are fixed validation seeds; final cases are disjoint.
        eval_seed = (task["seed"] + (2_000_000 if final else 1_000_000)) % 2**32
        count = args.final_episodes if final else args.eval_episodes
        results = {mode: evaluate(eval_env, state, fns, debug, eval_seed, count, mode, carry_dim, args)
                   for mode in ("greedy", "stochastic", "critic")}
        if final:
            results["random"] = evaluate(eval_env, state, fns, debug, eval_seed, count, "random", carry_dim, args)
        row = {"steps": steps, "interactions_including_prefill": interactions, "final": final,
               "elapsed_seconds_this_process": time.monotonic() - start,
               "steps_this_process": steps - starting_steps,
               "eval": {mode: value["metrics"] for mode, value in results.items()}}
        write_json(out / f"episodes_{steps}{'_final' if final else ''}.json", results)
        history.append(row)
        if wandb_run:
            prefix = "final" if final else "eval"
            wandb_run.log({"steps": steps, **{f"{prefix}/{mode}/{k}": v for mode, value in results.items() for k, v in value["metrics"].items()}})
        save()
        write_json(status, {"state": "running", "steps": steps, "target_steps": args.steps})
        print(json.dumps({**task, "steps": steps, "final": final,
                          "training_steps_per_second_including_eval": (steps - starting_steps) / max(time.monotonic() - start, 1e-6),
                          "returns": {m: r["metrics"]["return_mean"] for m, r in results.items()}}), flush=True)

    try:
        write_json(status, {"state": "running", "steps": steps, "target_steps": args.steps})
        if interactions == 0:
            window = args.burn_in_length + args.sequence_length + args.lambda_truncation
            remaining = args.prefill_steps or args.batch_size * window
            while remaining:
                count = min(args.collect_steps, remaining)
                collect(count, random=True)
                remaining -= count
            # Finish the random episode naturally: training starts with zero
            # memory without inserting an artificial terminal into the data.
            while episode_steps:
                collect(1, random=True)
            evaluation()
        while steps < args.steps:
            boundary = min(args.steps, (steps // args.eval_every + 1) * args.eval_every)
            count = min(args.collect_steps, boundary - steps)
            collect(count)
            key, update_key = jax.random.split(key)
            keys = jax.random.split(update_key, count * args.updates_per_step)
            state, metrics = update(state, keys)
            metrics = jax.device_get(metrics)
            if not all(np.all(np.isfinite(v)) for v in jax.tree.leaves(metrics)):
                raise FloatingPointError("Nonfinite training metrics")
            steps += count
            if steps == boundary:
                if wandb_run:
                    wandb_run.log({"steps": steps, **{f"train/{k}": float(v) for k, v in metrics.items()}})
                evaluation()
        if not history or not history[-1]["final"] or history[-1]["steps"] != steps:
            evaluation(final=True)
        write_json(status, {"state": "complete", "steps": steps, "target_steps": args.steps,
                            "interactions_including_prefill": interactions})
    except BaseException:
        write_json(status, {"state": "failed", "steps": steps, "error": traceback.format_exc()})
        raise
    finally:
        env.close(); eval_env.close()
        if wandb_run:
            wandb_run.finish()


def aggregate(values):
    import numpy as np

    a = np.asarray(values, dtype=float)
    if not a.size:
        return {"n": 0, "mean": None, "std": None, "se": None}
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"n": len(a), "mean": float(a.mean()), "std": sd, "se": sd / len(a)**0.5}


def summarize(root, tasks):
    """Across-seed uncertainty, explicit coverage and paired method differences."""
    groups, completed, pending, failed = {}, {}, [], []
    for task in tasks:
        folder = root / task["env_id"] / task["method"] / f"seed{task['seed']}"
        status_file = folder / "status.json"
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        if status.get("state") == "failed":
            failed.append(task)
        if status.get("state") != "complete" or status.get("steps", 0) < task.get("target_steps", 0):
            pending.append(task)
            continue
        history = json.loads((folder / "history.json").read_text())
        final = history[-1]
        completed[(task["env_id"], task["method"], task["seed"])] = final
        groups.setdefault((task["env_id"], task["method"]), []).append(final)
    result = {"planned_runs": len(tasks), "completed_runs": len(completed), "pending": pending,
              "failed": failed, "tasks": {}, "paired_lambda_minus_sac": {}}
    table = []
    for (env_id, method), rows in sorted(groups.items()):
        metrics = {}
        for mode in rows[0]["eval"]:
            keys = {key for r in rows for key in r["eval"][mode]}
            for key in sorted(keys):
                metrics[f"{mode}/{key}"] = aggregate([r["eval"][mode][key] for r in rows if key in r["eval"][mode]])
        result["tasks"].setdefault(env_id, {})[method] = metrics
        backend = next(t.get("backend", "gymnasium") for t in tasks if t["env_id"] == env_id and t["method"] == method)
        row = {"env_id": env_id, "backend": backend, "method": method, "completed_seeds": len(rows)}
        for mode in ("greedy", "stochastic", "critic", "random"):
            for stat in ("mean", "std", "se"):
                row[f"{mode}_return_{stat}"] = metrics[f"{mode}/return_mean"][stat]
        row["stochastic_q_mc_rmse"] = metrics["stochastic/q_mc_rmse/all"]["mean"]
        table.append(row)
    for env_id in result["tasks"]:
        seeds = sorted({seed for env, method, seed in completed if env == env_id and method == "lambda"}
                       & {seed for env, method, seed in completed if env == env_id and method == "sac"})
        if seeds:
            result["paired_lambda_minus_sac"][env_id] = {
                mode: {**aggregate([completed[(env_id, "lambda", s)]["eval"][mode]["return_mean"]
                                    - completed[(env_id, "sac", s)]["eval"][mode]["return_mean"] for s in seeds]),
                       "seeds": seeds}
                for mode in ("greedy", "stochastic", "critic")}
    write_json(root / "summary.json", result)
    if table:
        temporary = root / "summary.csv.tmp"
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
        temporary.replace(root / "summary.csv")
    lines = ["# POPGym paired benchmark", "",
             f"Completed **{len(completed)}/{len(tasks)} runs**; {len(failed)} failed.", "",
             "Continuous-action Pendulum tasks are excluded. Means and SD below are across trained seeds, not pooled episodes.",
             "Unfinished/failing seeds are not imputed; partial coverage must not be treated as a full five-seed result.", "",
             "| Environment | Backend | Method | Seeds | Greedy return | Stochastic return | Critic-greedy return | Random return |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in table:
        cells = [f"{row[m+'_return_mean']:.3f} ± {row[m+'_return_std']:.3f}" for m in ("greedy", "stochastic", "critic", "random")]
        lines.append(f"| {row['env_id']} | {row['backend']} | {row['method']} | {row['completed_seeds']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired lambda minus SAC (return)", "",
              "| Environment | Paired seeds | Greedy Δ mean ± SD | Stochastic Δ mean ± SD |",
              "|---|---:|---:|---:|"]
    for env_id, modes in sorted(result["paired_lambda_minus_sac"].items()):
        greedy, stochastic = modes["greedy"], modes["stochastic"]
        lines.append(f"| {env_id} | {greedy['n']} | {greedy['mean']:+.3f} ± {greedy['std']:.3f} | {stochastic['mean']:+.3f} ± {stochastic['std']:.3f} |")
    lines += ["", "Per-task histories contain validation learning curves. Final returns use disjoint held-out cases.",
              "Q-MC residuals in summary.json use stochastic trajectories; their RMSE includes return variance.",
              "No common-score average across unrelated task reward scales is reported."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return result


def plot_curves(root, tasks):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = root / "plots"
    out.mkdir(exist_ok=True)
    histories = {}
    for task in tasks:
        path = root / task["env_id"] / task["method"] / f"seed{task['seed']}" / "history.json"
        if path.exists():
            histories.setdefault(task["env_id"], {}).setdefault(task["method"], []).append(json.loads(path.read_text()))
    for env_id, methods in sorted(histories.items()):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        for ax, mode in zip(axes, ("greedy", "stochastic", "critic")):
            for method, seeds in sorted(methods.items()):
                at_steps = {}
                for history in seeds:
                    for row in history:
                        if not row["final"]:
                            at_steps.setdefault(row["steps"], []).append(row["eval"][mode]["return_mean"])
                # Keep incomplete curve coverage visible in the legend.
                steps = sorted(at_steps)
                means = np.array([np.mean(at_steps[s]) for s in steps])
                sd = np.array([np.std(at_steps[s], ddof=1) if len(at_steps[s]) > 1 else 0 for s in steps])
                ax.plot(steps, means, label=f"{method} (up to {len(seeds)} seeds)")
                ax.fill_between(steps, means - sd, means + sd, alpha=.15)
            ax.set_title(mode); ax.set_xlabel("Training interactions (prefill excluded)")
            ax.set_ylabel("Validation episode return"); ax.grid(alpha=.2); ax.legend()
        fig.suptitle(env_id + " — mean ± sample SD; final holdout not plotted")
        fig.savefig(out / f"{env_id}.png", dpi=130)
        plt.close(fig)


def supervise(args):
    included, excluded = discover(args.envs)
    from popgym_jax import assign_backends
    included = assign_backends(included, args.backend)
    if args.list:
        print(json.dumps({"included": included, "excluded": excluded}, indent=2))
        return
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fingerprints = source_fingerprint()
    tasks = [{"env_id": env["env_id"], "backend": env["backend"], "port_id": env.get("port_id"),
              "backend_note": env["backend_note"], "expected_source_sha256": fingerprints,
              "method": method, "seed": args.seed + s, "target_steps": args.steps}
             for s in range(args.num_seeds) for env in included for method in args.methods]
    manifest = {"included": included, "excluded": excluded, "tasks": tasks,
                "note": "Discrete Easy/Medium/Hard only; base aliases not duplicated; no hidden-state observations"}
    if (root / "manifest.json").exists():
        old = json.loads((root / "manifest.json").read_text())
        if old != manifest:
            raise ValueError("Output directory belongs to a different suite; use a new --output-dir")
        if not args.resume:
            raise FileExistsError("Suite already exists; use --resume")
    write_json(root / "manifest.json", manifest)
    task_dir = root / "tasks"
    task_dir.mkdir(exist_ok=True)
    command = sys.argv[1:]
    slots = [gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)] if args.gpus else [None] * args.workers
    queue, active, failures = list(tasks), [], 0
    last_progress = time.monotonic()
    print(f"{len(included)} tasks × {len(args.methods)} methods × {args.num_seeds} seeds = {len(tasks)} runs; {len(slots)} workers", flush=True)
    print(f"Excluded {len(excluded)} continuous tasks; JAX={sum(e['backend']=='jax' for e in included)}, CPU={sum(e['backend']=='gymnasium' for e in included)}. Output: {root}", flush=True)
    try:
        while queue or active:
            for gpu in list(slots):
                if not queue:
                    break
                task = queue.pop(0)
                name = f"{task['env_id']}_{task['method']}_{task['seed']}"
                task_file = task_dir / f"{name}.json"
                write_json(task_file, task)
                folder = root / task["env_id"] / task["method"] / f"seed{task['seed']}"
                folder.mkdir(parents=True, exist_ok=True)
                log = (folder / "run.log").open("a")
                environment = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
                if gpu is not None:
                    environment.update(CUDA_VISIBLE_DEVICES=gpu, JAX_PLATFORMS="cuda")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *command,
                                            "--task-file", str(task_file)], env=environment,
                                           stdout=log, stderr=subprocess.STDOUT)
                active.append((process, gpu, log, task, folder))
                slots.remove(gpu)
            for item in list(active):
                process, gpu, log, task, folder = item
                code = process.poll()
                if code is None:
                    continue
                log.close(); active.remove(item); slots.append(gpu)
                if code:
                    failures += 1
                    write_json(folder / "status.json", {"state": "failed", "exit_code": code,
                                                        "error": "See run.log (checkpoint retained)"})
                print(f"{'FAILED' if code else 'DONE'} {task['env_id']} {task['method']} seed={task['seed']}; {len(queue)} queued", flush=True)
                summarize(root, tasks)
            if active:
                if time.monotonic() - last_progress >= 60:
                    progress = []
                    for _, gpu, _, task, folder in active:
                        path = folder / "status.json"
                        status = json.loads(path.read_text()) if path.exists() else {}
                        progress.append(f"{task['env_id']}/{task['method']}/s{task['seed']}={status.get('steps', 0)}/{args.steps}")
                    print("ACTIVE " + "; ".join(progress) + f"; {len(queue)} queued", flush=True)
                    last_progress = time.monotonic()
                time.sleep(1)
    except KeyboardInterrupt:
        for process, _, log, _, _ in active:
            process.terminate()
        for process, _, log, _, _ in active:
            process.wait(); log.close()
        print("Stopped workers. --resume restores the last atomic checkpoint.", flush=True)
        raise
    finally:
        summarize(root, tasks)
    if args.plot:
        plot_curves(root, tasks)
    if failures:
        raise SystemExit(f"{failures} runs failed; other runs continued. Inspect summary.json and run.log.")


def main():
    args = parser().parse_args()
    validate_args(args)
    if args.summarize:
        manifest = json.loads((args.output_dir / "manifest.json").read_text())
        result = summarize(args.output_dir, manifest["tasks"])
        if args.plot:
            plot_curves(args.output_dir, manifest["tasks"])
        print(f"{result['completed_runs']}/{result['planned_runs']} complete, {len(result['failed'])} failed")
    elif args.task_file:
        task = json.loads(args.task_file.read_text())
        task = {k: v for k, v in task.items() if k != "target_steps"}
        if task.get("expected_source_sha256", source_fingerprint()) != source_fingerprint():
            raise ValueError("Suite source changed after scheduling; refusing to mix implementations")
        if task.get("backend") == "jax":
            from popgym_jax import train_one_jax
            train_one_jax(args, task)
        else:
            train_one(args, task)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
