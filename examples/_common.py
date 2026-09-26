"""Shared online experiment protocol: CLI, evaluation, logging and checkpoints.

Environment entry points supply dimensions, architecture and optional diagnostic
callbacks. Imports of JAX and plotting libraries are delayed until execution so
``--help`` works without the experiment extras installed.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import NamedTuple


class Experiment(NamedTuple):
    env: object
    projection: object
    actor_dims: tuple
    critic_dims: tuple
    max_steps: int
    obs_fn: object = lambda x: x
    mask_fn: object = None
    reference: object = None
    probe: object = None
    # Battleship resets rollout memory at host-round boundaries in the
    # reported runs. Minesweeper retains it. Both reset at episode boundaries.
    reset_memory_each_round: bool = True


def build_parser(name, defaults=None):
    p = argparse.ArgumentParser(
        description=f"{name}: recurrent actor–critic with Retrace discrepancy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("training")
    g.add_argument("--method", choices=("ac", "critics", "ld"), default="ld",
                   help="task critic only, auxiliary regression, or regression + discrepancy")
    g.add_argument("--rounds", type=int, default=50, help="total rounds, including on resume")
    g.add_argument("--train-steps", type=int, default=10_000, help="interactions and updates per round")
    g.add_argument("--seed", type=int, default=42, help="first seed; subsequent seeds increment by one")
    g.add_argument("--num-seeds", type=int, default=1, help="seeds run together with vmap")
    g.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
    g.add_argument("--memory-hidden-dim", type=int, default=512)
    g.add_argument("--fe-lr", type=float, default=1e-5)
    g.add_argument("--actor-lr", type=float, default=1e-4)
    g.add_argument("--critic-lr", type=float, default=1e-4, help="task and auxiliary critic learning rate")
    g.add_argument("--alpha", type=float, default=0.1, help="fixed actor entropy coefficient")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--tau", type=float, default=0.005)
    g.add_argument("--critic-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--sequence-length", type=int, default=40)
    g.add_argument("--burn-in-length", type=int, default=10)
    g.add_argument("--lambda-truncation", type=int, default=50, help="look-ahead tail excluded from auxiliary regression")
    g.add_argument("--lambda1", type=float, default=0.05)
    g.add_argument("--lambda2", type=float, default=0.85)
    g.add_argument("--lambda-coef", type=float, default=0.01)
    g.add_argument("--online-buffer-size", type=int, default=100_000)
    g.add_argument("--prefill-steps", type=int, default=0,
                   help="random prefill; 0 means batch size times replay window length")
    g = p.add_argument_group("evaluation and output")
    g.add_argument("--eval-every", type=int, default=1, help="evaluation interval in rounds")
    g.add_argument("--eval-episodes", type=int, default=50)
    g.add_argument("--final-return-window", type=int, default=5,
                   help="final evaluation points averaged within each seed")
    g.add_argument("--checkpoint-every", type=int, default=10,
                   help="periodic checkpoint interval; 0 disables periodic saves; final is always saved")
    g.add_argument("--resume-from", type=Path)
    g.add_argument("--output-dir", type=Path,
                   default=Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", "outputs")) / name.lower())
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb-project", default=f"retrace-{name.lower()}")
    g.add_argument("--wandb-run-name")
    g = p.add_argument_group("optional memory probes")
    g.add_argument("--probe", action="store_true", help="fit, score and visualise a final memory probe")
    g.add_argument("--probe-every", type=int, default=0, help="also probe every N rounds; 0 disables")
    g.add_argument("--probe-collect-steps", type=int, default=20_000, help="transitions per independent train/test rollout")
    g.add_argument("--probe-steps", type=int, default=10_000)
    g.add_argument("--probe-hidden-dim", type=int, default=256)
    g.add_argument("--probe-batch-size", type=int, default=128)
    g.add_argument("--probe-lr", type=float, default=1e-4)
    p.set_defaults(**(defaults or {}))
    return p


def parse_args(parser, argv=None):
    """Accept W&B underscore spelling while rejecting removed/unknown flags."""
    known = {option for a in parser._actions for option in a.option_strings}
    normalized = []
    for token in sys.argv[1:] if argv is None else argv:
        flag, sep, value = token.partition("=")
        canonical = flag.replace("_", "-")
        normalized.append(canonical + sep + value if canonical in known else token)
    return parser.parse_args(normalized)


def prepare_tracking(parser, args):
    """Resolve sweep values before building the environment or the agent."""
    if not (args.wandb or os.environ.get("WANDB_SWEEP_ID")):
        return None
    import wandb

    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    if os.environ.get("WANDB_SWEEP_ID"):
        actions = {a.dest: a for a in parser._actions}
        for name, value in dict(run.config).items():
            if name == "_wandb":
                continue
            if name not in actions:
                parser.error(f"unknown sweep parameter {name!r}")
            action = actions[name]
            if isinstance(value, str) and value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif action.type is not None:
                value = action.type(value)
            if action.choices is not None and value not in action.choices:
                parser.error(f"invalid sweep value {value!r} for {name!r}")
            setattr(args, name, value)
        args.output_dir = Path(args.output_dir) / f"run_{run.id}"
        args.wandb = True
        if args.resume_from:
            parser.error("sweep trials must start fresh")
    run.define_metric("env_interactions")
    for prefix in ("eval", "train", "reference", "probe", "timing"):
        run.define_metric(f"{prefix}/*", step_metric="env_interactions")
    return run


def validate_args(parser, args):
    positive = ("train_steps", "num_seeds", "memory_hidden_dim", "batch_size",
                "sequence_length", "lambda_truncation", "online_buffer_size",
                "eval_every", "eval_episodes", "final_return_window",
                "probe_collect_steps", "probe_steps", "probe_hidden_dim", "probe_batch_size")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("rounds", "burn_in_length", "checkpoint_every", "prefill_steps", "probe_every"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in ("fe_lr", "actor_lr", "critic_lr", "alpha", "lambda_coef", "probe_lr"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if not 0 <= args.lambda1 < args.lambda2 <= 1:
        parser.error("require 0 <= lambda1 < lambda2 <= 1")
    if not 0 < args.gamma < 1 or not 0 < args.tau <= 1:
        parser.error("require 0 < gamma < 1 and 0 < tau <= 1")
    if args.method != "ld":
        args.lambda_coef = 0.0
    window = args.burn_in_length + args.sequence_length + args.lambda_truncation
    if not args.prefill_steps:
        args.prefill_steps = args.batch_size * window
    if not window < args.prefill_steps <= args.online_buffer_size:
        parser.error("prefill-steps must exceed the replay window and fit the buffer")
    if args.memory_type == "identity" and (args.probe or args.probe_every):
        parser.error("memory probes require a recurrent memory")
    args.output_dir = Path(args.output_dir)


def config_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


RESUME_MUTABLE = {
    "rounds", "eval_every", "eval_episodes", "final_return_window", "checkpoint_every",
    "resume_from", "output_dir", "wandb", "wandb_project", "wandb_run_name",
    "probe", "probe_every", "probe_collect_steps", "probe_steps", "probe_hidden_dim",
    "probe_batch_size", "probe_lr",
}


def save_checkpoint(path, config, training, rnd, history):
    """Atomically save replay, optimizers, RNG and in-flight rollout history."""
    import jax

    snapshot = {"version": 1, "config": config, "training": jax.device_get(training),
                "round": rnd, "history": history}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(snapshot, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path, config):
    import jax
    import jax.numpy as jnp

    with Path(path).open("rb") as stream:
        saved = pickle.load(stream)
    if saved.get("version") != 1:
        raise ValueError("unsupported checkpoint version")
    mismatches = {k: (saved["config"].get(k), v) for k, v in config.items()
                  if k not in RESUME_MUTABLE and saved["config"].get(k) != v}
    if mismatches:
        raise ValueError(f"resume requires the same training configuration: {mismatches}")
    if saved["round"] > config["rounds"]:
        raise ValueError("rounds is the total budget and precedes the checkpoint")
    training = jax.tree.map(jnp.asarray, saved["training"])
    return training, saved["round"], saved["history"]


def make_evaluator(experiment, fns, carry_dim, action_dim, episodes, mode):
    import jax
    import jax.numpy as jnp

    env, params = experiment.env, experiment.env.default_params

    def evaluate(agent, key):
        def episode(key):
            key, rk = jax.random.split(key)
            obs, es = env.reset(rk, params)

            def step(carry, _):
                obs, es, memory, pa, key, active, totals = carry
                key, ak, ek = jax.random.split(key, 3)
                if mode == "reference":
                    action = experiment.reference(obs, es, ak)
                elif mode == "random":
                    if experiment.mask_fn is None:
                        action = jax.random.randint(ak, (), 0, action_dim)
                    else:
                        legal = experiment.mask_fn(obs).astype(bool)
                        action = jax.random.categorical(ak, jnp.where(legal, 0.0, -1e9))
                else:
                    action, memory = fns.predict(agent, obs, memory, ak,
                                                 deterministic=mode == "greedy", prev_action=pa)
                action = action.astype(jnp.int32)
                obs, es, reward, done, info = env.step(ek, es, action, params)
                totals += active * jnp.array([
                    reward, 1.0, done,
                    info.get("known_safe_opportunity", 0.0),
                    info.get("known_safe_taken", 0.0), info.get("repeat_action", 0.0)], jnp.float32)
                memory = jnp.where(done, jnp.zeros_like(memory), memory)
                pa = jnp.where(done, jnp.zeros_like(pa), fns.encode_action(action))
                return (obs, es, memory, pa, key, active & ~done, totals), None

            initial = (obs, es, jnp.zeros(carry_dim), jnp.zeros(action_dim), key,
                       jnp.bool_(True), jnp.zeros(6))
            final, _ = jax.lax.scan(step, initial, length=experiment.max_steps)
            return final[-1]

        results = jax.vmap(episode)(jax.random.split(key, episodes))
        mean = results.mean(0)
        totals = results.sum(0)
        return {"return": mean[0], "steps": mean[1], "completed": mean[2],
                "known_safe_choice": jnp.where(totals[3] > 0, totals[4] / jnp.maximum(totals[3], 1), jnp.nan),
                "repeat_fraction": totals[5] / jnp.maximum(totals[1], 1)}

    return jax.jit(jax.vmap(evaluate))


def summarize(values, prefix):
    import numpy as np

    result = {}
    for name, array in values.items():
        a = np.asarray(array, dtype=float)
        for seed, value in enumerate(a):
            result[f"{prefix}/{name}/seed_{seed}"] = float(value)
        finite = a[np.isfinite(a)]
        result[f"{prefix}/{name}/mean"] = float(finite.mean()) if finite.size else None
        result[f"{prefix}/{name}/std"] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    # JSON null represents diagnostics with no opportunities, not NaN literals.
    return {k: v if v is None or np.isfinite(v) else None for k, v in result.items()}


def run(parser, args, build_experiment):
    tracking = prepare_tracking(parser, args)
    validate_args(parser, args)
    experiment = build_experiment(args)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lambda_imitation import Hyperparameters, create_actor_critic_from_env, env_spec_from_gymnax

    if (args.probe or args.probe_every) and experiment.probe is None:
        parser.error("this environment has no probe specification")
    env, params = experiment.env, experiment.env.default_params
    spec = env_spec_from_gymnax(env, params)
    hp = Hyperparameters(**{k: getattr(args, k) for k in Hyperparameters._fields
                            if k != "lambda_critic_lr"}, lambda_critic_lr=args.critic_lr)
    config = {**config_dict(args), "environment": type(env).__name__,
              "lambda_critic_loss": "huber", "huber_delta": 1.0,
              "reset_memory_each_round": experiment.reset_memory_each_round}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists() and not args.resume_from:
        raise ValueError(f"{args.output_dir} already contains a run; use a fresh output-dir or resume-from")

    def build(seed):
        return create_actor_critic_from_env(
            spec, hp=hp, projection=experiment.projection, memory_type=args.memory_type,
            memory_hidden_dim=args.memory_hidden_dim, use_prev_action=True,
            actor_dims=experiment.actor_dims, critic_dims=experiment.critic_dims,
            lambda1_critic_dims=experiment.critic_dims, lambda2_critic_dims=experiment.critic_dims,
            train_steps=args.train_steps, approximate_lambda=args.method != "ac",
            critic_layer_norm=args.critic_layer_norm, obs_fn=experiment.obs_fn,
            mask_fn=experiment.mask_fn, seed=seed)

    first, fns = build(args.seed)
    carry_dim = (0 if args.memory_type == "identity" else
                 args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1))
    train = jax.jit(jax.vmap(lambda s, es, c, pa, k:
        fns.train_unrolled(s, env, params, es, c, pa, k)))
    evaluate = {mode: make_evaluator(experiment, fns, carry_dim, spec.action_dim,
                                     args.eval_episodes, mode) for mode in ("greedy", "sampled", "random")}
    if experiment.reference is not None:
        evaluate["reference"] = make_evaluator(experiment, fns, carry_dim, spec.action_dim,
                                                args.eval_episodes, "reference")
    seeds = [args.seed + i for i in range(args.num_seeds)]

    def eval_keys(rnd, stream):
        return jnp.stack([jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s), stream), rnd)
                          for s in seeds])

    if args.resume_from:
        training, start_round, history = load_checkpoint(args.resume_from, config)
    else:
        states = [first, *(build(s)[0] for s in seeds[1:])]
        agents = jax.tree.map(lambda *x: jnp.stack(x), *states)
        keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
        split = jax.vmap(lambda k: jax.random.split(k, 3))(keys)
        _, env_states = jax.vmap(lambda k: env.reset(k, params))(split[:, 1])
        prefill = jax.jit(jax.vmap(lambda s, es, k:
            fns.prefill_buffer(s, env, params, es, args.prefill_steps, k)))
        agents, _ = prefill(agents, env_states, split[:, 2])
        # Prefill ends with an artificial terminal. Start a fresh real episode.
        _, env_states = jax.vmap(lambda k: env.reset(k, params))(eval_keys(0, 1))
        training = (agents, env_states, jnp.zeros((args.num_seeds, carry_dim)),
                    jnp.zeros((args.num_seeds, spec.action_dim)), split[:, 0])
        start_round, history = 0, []
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if tracking:
        tracking.config.update(config, allow_val_change=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    # On resume, the checkpoint is authoritative, including its history.
    metrics_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in history))

    def log(row):
        history.append(row)
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if tracking:
            tracking.log(row)

    reference = {"env_interactions": args.prefill_steps + start_round * args.train_steps}
    for mode in ("random", "reference"):
        if mode in evaluate:
            reference.update(summarize(evaluate[mode](training[0], eval_keys(0, 2)), f"reference/{mode}"))
    log(reference)
    compiled = False
    for rnd in range(start_round + 1, args.rounds + 1):
        agents, env_states, carries, prev_actions, keys = training
        if experiment.reset_memory_each_round:
            carries, prev_actions = jnp.zeros_like(carries), jnp.zeros_like(prev_actions)
        split = jax.vmap(jax.random.split)(keys)
        start = time.perf_counter()
        agents, env_states, carries, prev_actions, metrics = train(
            agents, env_states, carries, prev_actions, split[:, 1])
        jax.block_until_ready(agents.update_step)
        elapsed = time.perf_counter() - start
        training = (agents, env_states, carries, prev_actions, split[:, 0])
        row = {"round": rnd, "train_env_steps": rnd * args.train_steps,
               "env_interactions": args.prefill_steps + rnd * args.train_steps,
               "timing/train_seconds": elapsed,
               "timing/includes_compilation": not compiled,
               **summarize(metrics, "train")}
        compiled = True
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            for mode in ("greedy", "sampled"):
                row.update(summarize(evaluate[mode](agents, eval_keys(rnd, 3)), f"eval/{mode}"))
        if args.probe_every and rnd % args.probe_every == 0:
            from _probes import run_probes
            row.update(run_probes(args, experiment, fns, agents, carry_dim, rnd))
        log(row)
        ret = row.get("eval/greedy/return/mean")
        print(f"round {rnd}/{args.rounds}  interactions={row['env_interactions']}  "
              f"train={elapsed:.2f}s" + (f"  return={ret:.3f}" if ret is not None else ""), flush=True)
        if args.checkpoint_every and rnd % args.checkpoint_every == 0:
            save_checkpoint(args.output_dir / f"checkpoint_{rnd}.pkl", config, training, rnd, history)

    if args.probe and not (args.probe_every and args.rounds % args.probe_every == 0
                          and args.rounds > start_round):
        from _probes import run_probes
        log({"env_interactions": args.prefill_steps + args.rounds * args.train_steps,
             **run_probes(args, experiment, fns, training[0], carry_dim, args.rounds)})
    summary = {"env_interactions": args.prefill_steps + args.rounds * args.train_steps}
    evaluations = [row for row in history if "eval/greedy/return/mean" in row]
    if evaluations:
        final = np.array([[row[f"eval/greedy/return/seed_{i}"] for i in range(args.num_seeds)]
                          for row in evaluations[-args.final_return_window:]]).mean(0)
        summary.update(summarize({"return_smoothed": final}, "final"))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    save_checkpoint(args.output_dir / "checkpoint_final.pkl", config, training, args.rounds, history)
    if tracking:
        tracking.summary.update(summary)
        tracking.finish()
    return summary
