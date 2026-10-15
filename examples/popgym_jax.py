"""Audited POPJym backend for popgym_suite.py (no JAX import in supervisor).

Ports are explicitly labelled, not assumed bitwise-equivalent to NumPy POPGym.
Pinned upstream revision: 2cd2ee4114579e25577c34a90695cebfc80f714d.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


PORTS = {name: name for name in ("Battleship", "Concentration", "HigherLower", "RepeatFirst", "RepeatPrevious")}
PORTS.update(PositionOnlyCartPole="StatelessCartPole", NoisyPositionOnlyCartPole="NoisyStatelessCartPole",
             MultiarmedBandit="MultiArmedBandit")
FALLBACKS = {
    "Autoencode": "Unaudited/broken port: watch/play observation reversed and phase observation missing",
    "CountRecall": "Broken port: action space is Discrete(2), not the required count range; count indexing differs",
    "MineSweeper": "Broken port: observation width depends on num_mines and drops valid neighbour counts",
}


def port_for(env_id):
    name = env_id.removeprefix("popgym-").removesuffix("-v0")
    for difficulty in ("Easy", "Medium", "Hard"):
        if name.endswith(difficulty):
            family = name.removesuffix(difficulty)
            if family in PORTS:
                return PORTS[family] + difficulty, "Audited POPJym variant; partial observations only; card horizon/index fixes where applicable"
            return None, FALLBACKS.get(family, "No corresponding POPJym port")
    return None, "Not an Easy/Medium/Hard suite task"


def assign_backends(included, backend):
    """Run JAX preflight in a CPU subprocess, never initialize JAX in supervisor."""
    result, ids = [], []
    for row in included:
        port, note = port_for(row["env_id"])
        use_jax = backend != "gymnasium" and port is not None
        if backend == "jax" and not use_jax:
            raise ValueError(f"{row['env_id']}: {note}; use --backend auto for explicit CPU fallback")
        result.append({**row, "backend": "jax" if use_jax else "gymnasium",
                       "port_id": port if use_jax else None,
                       "backend_note": note if backend != "gymnasium" else "Original NumPy/Gymnasium POPGym requested"})
        if use_jax:
            ids.append(row["env_id"])
    if ids:
        if importlib.util.find_spec("popjym") is None:
            raise ImportError("Install the pinned JAX extra: pip install -e '.[popgym,popgym-jax]' (or --backend gymnasium)")
        import os
        output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), "--preflight", *ids],
                                         env={**os.environ, "JAX_PLATFORMS": "cpu"}, text=True)
        details = json.loads(output)
        for row in result:
            if row["backend"] == "jax":
                row.update(details[row["env_id"]])
    return result


class PortEnv:
    """Use the observation returned by step, including noisy/action-dependent obs.

    Do not re-query get_obs: some ports require a noise key, others cannot
    reconstruct the last result from state. Reset only when done (not every
    step as the gymnax base implementation does). No hidden-state input/mask.
    """

    def __init__(self, env_id):
        import popjym
        self.env_id = env_id
        self.port_id, _ = port_for(env_id)
        if self.port_id is None:
            raise ValueError(f"No audited port for {env_id}")
        self.env, self.params = popjym.make(self.port_id)
        self.family = env_id.removeprefix("popgym-").removesuffix("-v0")
        if self.family.startswith(("RepeatFirst", "RepeatPrevious", "HigherLower")):
            self.horizon = int(self.env.num_decks * self.env.decksize - 1)
        elif hasattr(self.env, "episode_length"):
            self.horizon = int(self.env.episode_length)
        elif hasattr(self.env, "max_steps_in_episode"):
            self.horizon = int(self.env.max_steps_in_episode)
        else:
            self.horizon = int(self.env.max_episode_length)
        self.action_dim = int(self.env.action_space(self.params).n)
        self.obs_shape = tuple(self.env.observation_space(self.params).shape)

    def reset(self, key):
        return self.env.reset_env(key, self.params)

    def transition(self, key, state, action):
        import jax
        import jax.numpy as jnp

        obs, next_state, reward, done, _ = self.env.step_env(key, state, action, self.params)
        if self.family.startswith(("RepeatFirst", "RepeatPrevious", "HigherLower")):
            # Original POPGym deals the first card at reset and executes N-1
            # actions. Upstream executes N actions and clamps the last gather.
            done = next_state.timestep >= self.horizon
        if self.family.startswith("RepeatPrevious"):
            # At t=0 original hand has one card; the kth-last card is queried
            # once hand length >= k, hence t-k+1 (not upstream t-k).
            scale = 1.0 / (self.horizon + 1 - self.env.k)
            reward = jnp.where(state.cards[state.timestep - self.env.k + 1] == action, scale, -scale)
            reward = jnp.where(state.timestep < self.env.k - 1, 0., reward)
        return obs.astype(jnp.float32), jax.tree.map(jax.lax.stop_gradient, next_state), reward, done

    def step(self, key, state, action):
        import jax
        sk, rk = jax.random.split(key)
        obs, state, reward, done = self.transition(sk, state, action)
        obs, state = jax.lax.cond(done, lambda _: self.reset(rk), lambda _: (obs, state), None)
        return obs, state, reward, done


def port_fingerprint():
    import popjym
    root = Path(popjym.__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


class JaxRunner:
    """Compiled collect/update blocks with explicit persistent rollout state."""

    def __init__(self, args, port, fns, debug, buffer_fns, carry_dim, approximate_lambda):
        import jax
        import jax.numpy as jnp
        from functools import partial

        self.port, self.carry_dim = port, carry_dim
        self.zero_carry = jnp.zeros(carry_dim, jnp.float32)
        self.zero_prev = jnp.zeros(port.action_dim, jnp.float32)

        def collect(sac, rollout, random):
            key, obs, env_state, carry, prev, overflow, age = rollout
            key, ak, ek = jax.random.split(key, 3)
            if random:
                action = jax.random.randint(ak, (), 0, port.action_dim)
                prob = jnp.float32(1.0 / port.action_dim)
                nc = carry
            else:
                action, nc, prob = fns.predict(sac, obs, carry, ak, return_prob=True, prev_action=prev)
                action = action.astype(jnp.int32)
            nobs, ns, reward, done = port.step(ek, env_state, action)
            t = {"observations": obs, "actions": jnp.atleast_1d(action).astype(jnp.float32),
                 "rewards": jnp.asarray(reward, jnp.float32), "terminated": done.astype(jnp.float32)}
            if approximate_lambda:
                t["behaviour_weight"] = prob
            buffer = buffer_fns.add(sac.online_buffer, t, done)
            nc = jnp.where(done, self.zero_carry, nc)
            np = jnp.where(done, self.zero_prev, fns.encode_action(jnp.atleast_1d(action)))
            overflow = overflow | ((age + 1 >= args.max_episode_steps) & ~done)
            age = jnp.where(done, 0, age + 1)
            return sac._replace(online_buffer=buffer), (key, nobs, ns, nc, np, overflow, age)

        def mark_boundary(sac):
            b = sac.online_buffer
            newest = (b.pos - 1) % args.online_buffer_size
            return sac._replace(online_buffer=b._replace(sampling_ok=b.sampling_ok.at[newest].set(False)))

        @partial(jax.jit, static_argnames=("n_steps",))
        def prefill(sac, rollout, n_steps):
            def body(carry, _):
                sac, rollout = collect(*carry, random=True)
                return (sac, rollout), None
            (sac, rollout), _ = jax.lax.scan(body, (sac, rollout), None, length=n_steps)
            # End random prefill naturally, bounded by the known port horizon.
            def finish(c):
                sac, r, extra = c
                sac, r = collect(sac, r, random=True)
                return sac, r, extra + 1
            sac, rollout, extra = jax.lax.while_loop(lambda c: (c[1][-1] != 0) & (c[2] < port.horizon),
                                                    finish, (sac, rollout, jnp.int32(0)))
            # Random collection doesn't update memory; at this natural boundary
            # both carry and prev-action are zero, matching replay episode starts.
            return mark_boundary(sac), rollout, n_steps + extra

        def block(sac, rollout, count):
            def collect_body(c, _):
                return collect(*c, random=False), None
            (sac, rollout), _ = jax.lax.scan(collect_body, (sac, rollout), None, length=count)
            sac = mark_boundary(sac)
            key, *rest = rollout
            key, uk = jax.random.split(key)
            keys = jax.random.split(uk, count * args.updates_per_step)
            def update_body(sac, key):
                return fns.update(sac, key)
            sac, metrics = jax.lax.scan(update_body, sac, keys)
            metrics = jax.tree.map(lambda x: x.mean(), metrics)
            return sac, (key, *rest), metrics

        @partial(jax.jit, static_argnames=("n_steps",))
        def train(sac, rollout, n_steps):
            full, remainder = divmod(n_steps, args.collect_steps)
            def body(c, _):
                ns, nr, metrics = block(*c, args.collect_steps)
                return (ns, nr), metrics
            if full:
                (sac, rollout), metrics = jax.lax.scan(body, (sac, rollout), None, length=full)
                metrics = jax.tree.map(lambda x: x.mean(), metrics)
            if remainder:
                sac, rollout, tail = block(sac, rollout, remainder)
                metrics = jax.tree.map(lambda a, b: (a * (n_steps - remainder) + b * remainder) / n_steps,
                                       metrics, tail) if full else tail
            return sac, rollout, metrics

        @partial(jax.jit, static_argnames=("mode", "n_episodes"))
        def evaluate(sac, seed, mode, n_episodes):
            horizon = min(port.horizon, args.max_episode_steps)
            def episode(index):
                key = jax.random.PRNGKey(seed + index)
                key, rk = jax.random.split(key)
                obs, es = port.reset(rk)
                def body(c, _):
                    key, obs, es, carry, prev, active = c
                    key, ak, ek = jax.random.split(key, 3)
                    values, nc = debug.predict_values(sac, obs, carry, prev)
                    probs, qs = values["probs"], values["q"].min(-1)
                    if mode == "random":
                        action = jax.random.randint(ak, (), 0, port.action_dim)
                    elif mode == "critic":
                        action = jnp.argmax(qs)
                    elif mode == "greedy":
                        action = jnp.argmax(probs)
                    else:
                        action = jax.random.categorical(ak, jnp.log(probs + 1e-30))
                    nobs, nes, reward, done = port.transition(ek, es, action)
                    entropy = -(probs * jnp.log(probs + 1e-30)).sum()
                    ld = (values["lambda1_q"][action].min() - values["lambda2_q"][action].min()) if approximate_lambda else jnp.float32(0.)
                    outputs = (jnp.where(active, reward, 0.), active, active & done,
                               jnp.where(active, qs[action], 0.), jnp.where(active, entropy, 0.), jnp.where(active, ld, 0.))
                    return (key, nobs, nes, nc, fns.encode_action(jnp.atleast_1d(action)), active & ~done), outputs
                _, outputs = jax.lax.scan(body, (key, obs, es, self.zero_carry, self.zero_prev, jnp.bool_(True)), None, length=horizon)
                return outputs
            return jax.vmap(episode)(jnp.arange(n_episodes, dtype=jnp.uint32))

        self.prefill, self.train, self.evaluate = prefill, train, evaluate

    def initial_rollout(self, seed):
        import jax
        import jax.numpy as jnp
        key, rk = jax.random.split(jax.random.PRNGKey(seed))
        obs, state = self.port.reset(rk)
        return (key, obs, state, self.zero_carry, self.zero_prev, jnp.bool_(False), jnp.int32(0))


def evaluation_result(outputs, mode, args, approximate_lambda):
    """Same artifact schema as host evaluation; only stochastic MC is calibrated."""
    import numpy as np
    from popgym_suite import return_targets

    rewards, active, done, qs, entropies, ld = map(np.asarray, outputs)
    rows, errors, gaps = [], {s: [] for s in ("all", "early", "middle", "late")}, []
    for i in range(rewards.shape[0]):
        length = int(active[i].sum())
        if not bool(done[i].any()):
            raise RuntimeError("JAX evaluation did not terminate within --max-episode-steps")
        row = {"episode": i, "return": float(rewards[i].sum()), "length": length,
               "done": True, "termination_kind": "combined_done (POPJym does not distinguish truncation)"}
        if mode == "stochastic":
            target = return_targets(rewards[i, :length], entropies[i, :length], args.gamma, args.alpha if args.use_sac else 0.)
            residual = qs[i, :length] - target
            errors["all"].extend(residual.tolist())
            for stage, values in zip(("early", "middle", "late"), np.array_split(residual, 3)):
                errors[stage].extend(values.tolist())
            row.update(q_mc_bias=float(residual.mean()), q_mc_rmse=float(np.sqrt(np.mean(residual**2))))
            if approximate_lambda:
                gaps.extend(ld[i, :length].tolist())
        rows.append(row)
    rets = np.asarray([r["return"] for r in rows])
    metrics = {"return_mean": float(rets.mean()), "return_std": float(rets.std(ddof=1)) if len(rets) > 1 else 0.,
               "length_mean": float(np.mean([r["length"] for r in rows]))}
    for stage, values in errors.items():
        if values:
            e = np.asarray(values)
            metrics[f"q_mc_rmse/{stage}"] = float(np.sqrt(np.mean(e**2)))
            metrics[f"q_mc_bias/{stage}"] = float(e.mean())
    if gaps:
        metrics["lambda_discrepancy_rms"] = float(np.sqrt(np.mean(np.asarray(gaps)**2)))
    if not all(np.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite JAX evaluation metrics")
    return {"metrics": metrics, "episodes": rows}


def train_one_jax(args, task):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lambda_imitation.buffer import create_buffer
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import EnvSpec, create_iqlearn_from_env
    from popgym_suite import run_config, source_fingerprint, write_json

    port = PortEnv(task["env_id"])
    out = args.output_dir / task["env_id"] / task["method"] / f"seed{task['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    config = run_config(args, task)
    config.update(actual_obs_shape=list(port.obs_shape), actual_action_dim=port.action_dim,
                  episode_horizon=port.horizon, port_source_sha256=port_fingerprint())
    versions = {n: importlib.metadata.version(n) for n in ("popjym", "gymnax", "jax", "flax", "optax", "cloudpickle")}
    provenance = {"experiment": config, "source_sha256": source_fingerprint(), "versions": versions}
    if (out / "config.json").exists():
        old = json.loads((out / "config.json").read_text())
        if any(old[k] != v for k, v in provenance.items()):
            raise ValueError("Existing JAX run has different settings/source/packages; choose a new output directory")
        if not args.resume:
            raise FileExistsError("JAX run already exists; use --resume")
    else:
        write_json(out / "config.json", {**provenance, "devices": [str(d) for d in jax.devices()]})
    hp = Hyperparameters(batch_size=1, online_batch_size=args.batch_size, online_buffer_size=args.online_buffer_size,
                         fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr, lambda_critic_lr=args.critic_lr,
                         alpha=args.alpha, gamma=args.gamma, tau=args.tau, burn_in_length=args.burn_in_length,
                         sequence_length=args.sequence_length, lambda_truncation=args.lambda_truncation,
                         lambda1=args.lambda1, lambda2=args.lambda2, lambda_coef=args.lambda_coef,
                         fake_onpolicy_loss=False, retrace=True, stop_actor_fe=True)
    approximate_lambda = task["method"] == "lambda"
    state, fns, debug = create_iqlearn_from_env(
        EnvSpec(port.obs_shape, port.action_dim, True),
        {"observations": jnp.zeros((1, *port.obs_shape)), "actions": jnp.zeros((1, 1))},
        buffer_size=1, hp=hp, projection=args.projection_dim or None, memory_type=args.memory_type,
        memory_hidden_dim=args.memory_hidden_dim, use_prev_action=True, actor_dims=(args.head_dim,),
        critic_dims=(args.head_dim,), lambda1_critic_dims=(args.head_dim,), lambda2_critic_dims=(args.head_dim,),
        approximate_lambda=approximate_lambda, critic_layer_norm=args.critic_layer_norm,
        debug=True, seed=task["seed"], use_sac=args.use_sac)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
    carry_dim = 0 if args.memory_type == "identity" else args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1)
    runner = JaxRunner(args, port, fns, debug, bfns, carry_dim, approximate_lambda)
    rollout = runner.initial_rollout(task["seed"])
    steps, interactions, history = 0, 0, []
    checkpoint, status = out / "checkpoint.pkl", out / "status.json"
    if args.resume and checkpoint.exists():
        saved = cloudpickle.loads(checkpoint.read_bytes())
        if saved["config"] != config:
            raise ValueError("JAX checkpoint config mismatch")
        state, rollout = jax.tree.map(jnp.asarray, (saved["state"], saved["rollout"]))
        steps, interactions, history = saved["steps"], saved["interactions"], saved["history"]
        write_json(out / "history.json", history)
        if steps >= args.steps and status.exists() and json.loads(status.read_text()).get("state") == "complete":
            print(f"Already complete: {task}", flush=True)
            return
    start, start_steps = time.monotonic(), steps
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         id=hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16], resume="allow",
                         group=task["env_id"], name=f"{task['env_id']}-{task['method']}-{task['seed']}",
                         config=config, dir=str(out))
        run.define_metric("steps")
        for prefix in ("train", "eval", "final"):
            run.define_metric(f"{prefix}/*", step_metric="steps")
    def save():
        data = {"config": config, "state": jax.device_get(state), "rollout": jax.device_get(rollout),
                "steps": steps, "interactions": interactions, "history": history}
        tmp = checkpoint.with_suffix(".pkl.tmp")
        with tmp.open("wb") as file:
            cloudpickle.dump(data, file)
        tmp.replace(checkpoint)
        write_json(out / "history.json", history)
    def evaluate(final=False):
        seed = np.uint32((task["seed"] + (2_000_000 if final else 1_000_000)) % 2**32)
        n = args.final_episodes if final else args.eval_episodes
        modes = ("greedy", "stochastic", "critic", "random") if final else ("greedy", "stochastic", "critic")
        results = {m: evaluation_result(jax.device_get(runner.evaluate(state, seed, m, n)), m, args, approximate_lambda) for m in modes}
        elapsed = time.monotonic() - start
        history.append({"steps": steps, "interactions_including_prefill": interactions, "final": final,
                        "backend": "jax", "elapsed_seconds_this_process": elapsed, "steps_this_process": steps - start_steps,
                        "eval": {m: r["metrics"] for m, r in results.items()}})
        write_json(out / f"episodes_{steps}{'_final' if final else ''}.json", results)
        if run:
            prefix = "final" if final else "eval"
            run.log({"steps": steps, **{f"{prefix}/{m}/{k}": v for m, r in results.items() for k, v in r["metrics"].items()}})
        save()
        write_json(status, {"state": "running", "steps": steps, "target_steps": args.steps})
        print(json.dumps({**task, "steps": steps, "final": final,
                          "training_steps_per_second_including_eval": (steps - start_steps) / max(elapsed, 1e-6),
                          "returns": {m: r["metrics"]["return_mean"] for m, r in results.items()}}), flush=True)
    try:
        write_json(status, {"state": "running", "steps": steps, "target_steps": args.steps})
        if interactions == 0:
            n = args.prefill_steps or args.batch_size * (args.burn_in_length + args.sequence_length + args.lambda_truncation)
            state, rollout, count = runner.prefill(state, rollout, n)
            interactions = int(count)
            if int(rollout[-1]) != 0 or bool(rollout[-2]):
                raise RuntimeError("Random prefill did not reach a natural episode boundary")
            evaluate()
        while steps < args.steps:
            boundary = min(args.steps, (steps // args.eval_every + 1) * args.eval_every)
            count = min(args.jax_train_steps, boundary - steps)
            state, rollout, metrics = runner.train(state, rollout, count)
            metrics = jax.device_get(metrics)
            if not all(np.isfinite(v).all() for v in jax.tree.leaves(metrics)):
                raise FloatingPointError("Nonfinite JAX training metrics")
            if bool(rollout[-2]):
                raise RuntimeError("JAX training episode exceeded safety horizon")
            steps += count; interactions += count
            if steps == boundary:
                if run:
                    run.log({"steps": steps, **{f"train/{k}": float(v) for k, v in metrics.items()}})
                evaluate()
        if not history or not history[-1]["final"] or history[-1]["steps"] != steps:
            evaluate(final=True)
        write_json(status, {"state": "complete", "steps": steps, "target_steps": args.steps,
                            "interactions_including_prefill": interactions})
    except BaseException:
        write_json(status, {"state": "failed", "steps": steps, "error": traceback.format_exc()})
        raise
    finally:
        if run:
            run.finish()


def preflight(ids):
    import jax
    import jax.numpy as jnp
    import numpy as np
    result = {}
    for env_id in ids:
        port = PortEnv(env_id)
        obs, state = jax.jit(port.reset)(jax.random.PRNGKey(0))
        obs, state, reward, done = jax.jit(port.step)(jax.random.PRNGKey(1), state, jnp.int32(0))
        if not np.isfinite(obs).all() or not np.isfinite(reward).all() or tuple(obs.shape) != port.obs_shape:
            raise ValueError(f"Invalid JAX port: {env_id}")
        result[env_id] = {"jax_obs_shape": list(port.obs_shape), "jax_num_actions": port.action_dim,
                          "jax_horizon": port.horizon}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", nargs="+", required=True)
    print(json.dumps(preflight(p.parse_args().preflight)))
