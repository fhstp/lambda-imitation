"""Concentration / Minesweeper pilots with matched SAC + Retrace ablations.

Requires the existing lambda-imitation[gymnax] environment; no new env package.
Examples::

    python examples/lambda-envs/memory_games_sac.py --env concentration --method ld
    python examples/lambda-envs/memory_games_sac.py --env minesweeper --method critics
    python examples/lambda-envs/memory_games_sac.py --env minesweeper --method sac \
        --remember --memory-type identity

``sac`` disables the lambda branches; ``critics`` retains their regression losses
with discrepancy weight zero; ``ld`` adds the discrepancy. All use the SAME
replay windows, architecture, initialization seeds and evaluation boards.

Every round prints synchronized training time and projected 100k / 1M-step
runtime. Compilation is timed separately. Checkpoints include replay, optimizers,
RNG and in-flight episode/memory state; --rounds is the TOTAL budget on resume.
Metrics are also appended to metrics.jsonl without requiring W&B. The interaction
axis includes random prefill; train_env_steps counts the subsequent updates.

Previous actions are appended to observations by a stateful env adapter, before
the usual LinearProjection. This is equivalent to use_prev_action's concatenation
but persists across train_unrolled calls (that API starts its internal previous
action at zero on each call). The recurrent carry itself is passed between rounds.
Both histories reset only at actual episode boundaries. Core learner code is
unchanged. Full-episode burn-in is the default to reconstruct sampled histories.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces

import _probe_common as common
from _concentration_env import Concentration
from _minesweeper_env import MineSweeper
from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


class ActionHistoryState(NamedTuple):
    inner: object
    previous_action: jax.Array


class ActionHistoryEnv:
    """Keep the executed previous action available across host training rounds."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_obs(self, state, params=None):
        return jnp.concatenate((self.env.get_obs(state.inner, params),
                                state.previous_action))

    def reset(self, key, params=None):
        params = self.default_params if params is None else params
        _, inner = self.env.reset(key, params)
        state = ActionHistoryState(inner, jnp.zeros(self.num_actions, jnp.float32))
        return self.get_obs(state, params), state

    def step(self, key, state, action, params=None):
        params = self.default_params if params is None else params
        _, inner, reward, done, info = self.env.step(key, state.inner, action, params)
        pa = jax.nn.one_hot(action, self.num_actions, dtype=jnp.float32)
        state = ActionHistoryState(inner, jnp.where(done, jnp.zeros_like(pa), pa))
        return self.get_obs(state, params), state, reward, done, info

    def observation_space(self, params=None):
        width = self.env.observation_space(params).shape[0] + self.num_actions
        return spaces.Box(0.0, 1.0, (width,), jnp.float32)

    def diagnostics(self, state, action):
        return self.env.diagnostics(state.inner, action)


def history_action(env, state, key):
    """Simple history-only reference player, never reading unseen labels/mines.

    Concentration: take known pairs, otherwise explore unseen cards.
    Minesweeper: take unseen neighbors of known zeros, otherwise unseen cells.
    These are useful reference policies, not optimal-policy upper bounds.
    """
    s = state.inner
    if isinstance(env.env, Concentration):
        available = ~s.face_up & ~s.in_play
        known = s.seen & ~s.face_up
        same = ((s.last_seen[:, None] == s.last_seen[None, :])
                & known[:, None] & known[None, :]
                & ~jnp.eye(env.num_actions, dtype=bool))
        first = jnp.argmax(s.in_play)
        wanted = jnp.where(jnp.any(s.in_play),
                           known & available & (s.last_seen == s.last_seen[first]),
                           jnp.any(same, axis=1))
        unknown = available & ~s.seen
        fallback = jnp.where(jnp.any(unknown), unknown, available)
    else:
        known_zero = s.viewed & (s.neighbor_counts == 0)
        neighbors = jax.lax.reduce_window(
            known_zero.astype(jnp.int32), jnp.int32(0), jax.lax.add,
            (3, 3), (1, 1), "SAME")
        wanted = ((neighbors > 0) & ~s.viewed).reshape(-1)
        fallback = (~s.viewed).reshape(-1)
    candidates = jnp.where(jnp.any(wanted), wanted, fallback)
    return jax.random.categorical(key, jnp.where(candidates, 0.0, -1e9))


def make_evaluator(fns, env, carry_dim, n_episodes, mode):
    """Fresh episodes, with denominators pooled across episodes for diagnostics."""
    params = env.default_params
    concentration = isinstance(env.env, Concentration)
    opportunity_key = "known_match_opportunity" if concentration else "known_safe_opportunity"
    taken_key = "known_match_taken" if concentration else "known_safe_taken"
    invalid_key = "invalid_action" if concentration else "repeat_action"
    goal_count = env.num_cards // 2 if concentration else env.episode_length

    def evaluate(agent, key):
        def episode(key):
            key, rk = jax.random.split(key)
            obs, es = env.reset(rk, params)
            initial = (obs, es, jnp.zeros(carry_dim, jnp.float32), key,
                       jnp.bool_(True), jnp.zeros(6, jnp.float32))

            def step(carry, _):
                obs, es, memory, key, active, totals = carry
                key, ak, ek = jax.random.split(key, 3)
                if mode == "random":
                    action = jax.random.randint(ak, (), 0, env.num_actions)
                    next_memory = memory
                elif mode == "history":
                    action = history_action(env, es, ak)
                    next_memory = memory
                else:
                    action, next_memory = fns.predict(
                        agent, obs, memory, ak, deterministic=mode == "greedy")
                    action = jnp.round(action).astype(jnp.int32)
                diagnostics = env.diagnostics(es, action)
                obs, es, reward, done, _ = env.step(ek, es, action, params)
                totals += active * jnp.asarray([
                    reward, 1.0, reward > 0, diagnostics[opportunity_key],
                    diagnostics[taken_key], diagnostics[invalid_key]], jnp.float32)
                next_memory = jnp.where(done, jnp.zeros_like(next_memory), next_memory)
                return (obs, es, next_memory, key, active & ~done, totals), None

            final, _ = jax.lax.scan(step, initial, length=env.episode_length)
            return final[-1]

        totals = jax.vmap(episode)(jax.random.split(key, n_episodes))
        mean = totals.mean(axis=0)
        opportunities = totals[:, 3].sum()
        return {
            "return": mean[0], "steps": mean[1],
            "progress": mean[2] / goal_count,
            "success": (totals[:, 2] >= goal_count).mean(),
            "opportunities": mean[3],
            "known_choice_rate": jnp.where(
                opportunities > 0, totals[:, 4].sum() / jnp.maximum(opportunities, 1),
                jnp.nan),
            "invalid_fraction": totals[:, 5].sum() / jnp.maximum(totals[:, 1].sum(), 1),
        }

    return jax.jit(jax.vmap(evaluate))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=("concentration", "minesweeper"), required=True)
    p.add_argument("--method", choices=("sac", "critics", "ld"), default="ld")
    p.add_argument("--cards", type=int, default=52)
    p.add_argument("--types", type=int, default=13)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--mines", type=int, default=6)
    p.add_argument("--remember", action="store_true", help="expose only accumulated observed history")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--train-steps", type=int, default=5000)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
    p.add_argument("--memory-hidden-dim", type=int, default=128)
    p.add_argument("--projection-dim", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--sequence-length", type=int, default=32)
    p.add_argument("--burn-in-length", type=int, default=None,
                   help="default: full episode length, so replay contains all past clues")
    p.add_argument("--lambda-truncation", type=int, default=32)
    p.add_argument("--online-buffer-size", type=int, default=100000)
    p.add_argument("--prefill-steps", type=int, default=0, help="0 = batch_size * replay window length")
    p.add_argument("--fe-lr", type=float, default=1e-4)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lambda1", type=float, default=0.0)
    p.add_argument("--lambda2", type=float, default=0.95)
    p.add_argument("--lambda-coef", type=float, default=0.01)
    p.add_argument("--use-sac", action="store_true", help="also put entropy in the main critic backup")
    p.add_argument("--uncorrected", action="store_true", help="ablation: pin importance ratios to 1")
    p.add_argument("--actor-fe-grad", action="store_true", help="allow actor gradients into memory")
    p.add_argument("--no-critic-layer-norm", dest="critic_layer_norm", action="store_false")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--checkpoint-every", type=int, default=10, help="rounds; 0 disables all checkpoints")
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="offline-lambda-memory-games-results")
    p.add_argument("--wandb-run-name", default=None)
    return p


def resolve_args(p, args):
    for field in ("train_steps", "num_seeds", "memory_hidden_dim", "projection_dim",
                  "head_dim", "batch_size", "sequence_length", "lambda_truncation",
                  "online_buffer_size", "eval_every", "eval_episodes"):
        if getattr(args, field) <= 0:
            p.error(f"--{field.replace('_', '-')} must be positive")
    if args.rounds < 0 or args.checkpoint_every < 0 or args.prefill_steps < 0:
        p.error("rounds, checkpoint-every and prefill-steps must be nonnegative")
    if not 0 <= args.lambda1 < args.lambda2 <= 1:
        p.error("require 0 <= lambda1 < lambda2 <= 1")
    if not 0 < args.gamma < 1 or args.alpha < 0 or args.lambda_coef < 0:
        p.error("require 0 < gamma < 1 and nonnegative alpha / lambda-coef")
    if args.method != "ld":
        args.lambda_coef = 0.0
    base = (Concentration(args.cards, args.types, remember=args.remember)
            if args.env == "concentration" else
            MineSweeper(args.rows, args.cols, args.mines, remember=args.remember))
    if args.burn_in_length is None:
        args.burn_in_length = base.episode_length
    if args.burn_in_length < 0:
        p.error("burn-in-length must be nonnegative")
    window = args.burn_in_length + args.sequence_length + args.lambda_truncation
    if args.online_buffer_size < max(window + 1, args.batch_size):
        p.error("online-buffer-size must accommodate the replay window and batch")
    if not args.prefill_steps:
        args.prefill_steps = args.batch_size * window
    if args.prefill_steps > args.online_buffer_size or args.prefill_steps < args.batch_size:
        p.error("prefill-steps must be between batch-size and online-buffer-size")
    if args.output_dir is None:
        suffix = f"ld{args.lambda_coef:g}" if args.method == "ld" else args.method
        args.output_dir = Path("memory_games_output") / args.env / suffix
    return ActionHistoryEnv(base)


# Only these settings may change when continuing a saved training run.
RESUME_MUTABLE = {"rounds", "eval_every", "eval_episodes", "checkpoint_every",
                  "resume_from", "output_dir", "wandb", "wandb_project", "wandb_run_name"}


def config_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def save_checkpoint(path, args, agents, env_states, carries, keys, rnd, history):
    leaves, tree = jax.tree.flatten((agents, env_states, carries, keys))
    snapshot = {"version": 1, "config": config_dict(args), "round": rnd,
                "tree": tree, "leaves": jax.device_get(leaves), "history": history}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as f:
        pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def load_checkpoint(path, args):
    with path.open("rb") as f:
        saved = pickle.load(f)
    if saved.get("version") != 1:
        raise ValueError("Unsupported memory-games checkpoint version")
    current = config_dict(args)
    mismatches = {k: (saved["config"].get(k), v) for k, v in current.items()
                  if k not in RESUME_MUTABLE and saved["config"].get(k) != v}
    if mismatches:
        raise ValueError(f"Resume requires the same training configuration: {mismatches}")
    if saved["round"] > args.rounds:
        raise ValueError("--rounds is the TOTAL target and is smaller than the saved round")
    restored = saved["tree"].unflatten([jnp.asarray(x) for x in saved["leaves"]])
    return (*restored, saved["round"], saved["history"])


def init_tracking(args, config):
    """Attach to an existing W&B ID without resetting its configuration/history.

    WANDB_RUN_ID + WANDB_RESUME=must are set per process by the resume launcher.
    Attach first, then update mutable metadata (e.g. the larger round budget)
    through the shared helper's allow_val_change path. The checkpoint, not W&B,
    restores the actual training state.
    """
    attached_run = None
    if args.wandb and os.environ.get("WANDB_RUN_ID") and os.environ.get("WANDB_RESUME"):
        import wandb
        attached_run = wandb.init(project=args.wandb_project)
    tracking = common.init_wandb(args, config, attached_run)
    if tracking is not None:
        for prefix in ("agg", "eval", "reference", "timing"):
            tracking.define_metric(f"{prefix}/*", step_metric="env_interactions")
    return tracking


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    env = resolve_args(p, args)
    config = config_dict(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume_from:
        p.error(f"{metrics_path} already exists; use a new output-dir or --resume-from")
    spec = env_spec_from_gymnax(env, env.default_params)
    carry_dim = (0 if args.memory_type == "identity" else
                 args.memory_hidden_dim * (2 if args.memory_type == "lstm" else 1))
    hp = Hyperparameters(
        batch_size=args.batch_size, online_buffer_size=args.online_buffer_size,
        fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        lambda_critic_lr=args.critic_lr, alpha=args.alpha, autotune_alpha=False,
        gamma=args.gamma, tau=args.tau, lambda1=args.lambda1, lambda2=args.lambda2,
        lambda_coef=args.lambda_coef, lambda_truncation=args.lambda_truncation,
        sequence_length=args.sequence_length, burn_in_length=args.burn_in_length,
        stop_actor_fe=not args.actor_fe_grad, retrace=True,
        fake_onpolicy_loss=args.uncorrected,
    )
    expert = {"observations": jnp.zeros((1, *spec.obs_shape), jnp.float32),
              "actions": jnp.zeros((1, 1), jnp.float32)}

    def build(seed):
        return create_iqlearn_from_env(
            spec, expert, buffer_size=1, hp=hp, projection=args.projection_dim,
            memory_type=args.memory_type, memory_hidden_dim=args.memory_hidden_dim,
            use_prev_action=False, actor_dims=(args.head_dim,),
            critic_dims=(args.head_dim,), lambda1_critic_dims=(args.head_dim,),
            lambda2_critic_dims=(args.head_dim,), train_steps=args.train_steps,
            approximate_lambda=args.method != "sac", seed=seed,
            critic_layer_norm=args.critic_layer_norm, use_sac=args.use_sac)

    print(f"{args.env} {args.method}, weight={args.lambda_coef:g}, "
          f"obs={spec.obs_shape}, actions={env.num_actions}, device={jax.devices()[0]}", flush=True)
    print(f"{args.num_seeds} vmapped seeds, {args.rounds} x {args.train_steps} updates, "
          f"prefill={args.prefill_steps}; replay window="
          f"{args.burn_in_length}+{args.sequence_length}+{args.lambda_truncation}", flush=True)
    state, fns = build(args.seed)
    if args.resume_from:
        agents, env_states, carries, keys, start, history = load_checkpoint(args.resume_from, args)
        print(f"Resumed round {start} from {args.resume_from}", flush=True)
    else:
        agents = common.stack_states([state] + [build(args.seed + i)[0]
                                                for i in range(1, args.num_seeds)])
        keys = jnp.stack([jax.random.PRNGKey(args.seed + i) for i in range(args.num_seeds)])
        keys, rk = common.split_each(keys)
        _, env_states = jax.vmap(lambda k: env.reset(k, env.default_params))(rk)
        keys, pk = common.split_each(keys)
        prefill = jax.jit(jax.vmap(lambda s, es, k: fns.prefill_buffer(
            s, env, env.default_params, es, args.prefill_steps, k)), donate_argnums=(0, 1))
        t0 = time.perf_counter()
        agents, env_states = prefill(agents, env_states, pk)
        jax.block_until_ready(agents)
        print(f"Prefill + compile: {time.perf_counter() - t0:.1f}s", flush=True)
        # Prefill marks its final transition done; begin collection at a real reset.
        keys, rk = common.split_each(keys)
        _, env_states = jax.vmap(lambda k: env.reset(k, env.default_params))(rk)
        carries = jnp.zeros((args.num_seeds, carry_dim), jnp.float32)
        start, history = 0, []

    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
    wandb = init_tracking(args, config)

    def emit(payload):
        # None makes undefined opportunity rates valid JSON rather than NaN.
        payload = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                   for k, v in payload.items()}
        with metrics_path.open("a") as f:
            f.write(json.dumps(payload) + "\n")
        if wandb is not None:
            wandb.log({k: v for k, v in payload.items() if v is not None})

    evaluators = {mode: make_evaluator(fns, env, carry_dim, args.eval_episodes, mode)
                  for mode in ("greedy", "sampled", "random", "history")}

    def eval_keys(rnd):
        return jnp.stack([jax.random.fold_in(jax.random.PRNGKey(args.seed + i + 1000000), rnd)
                          for i in range(args.num_seeds)])

    def evaluate_round(rnd, reference=False):
        payload = {"round": rnd, "train_env_steps": rnd * args.train_steps,
                   "env_interactions": args.prefill_steps + rnd * args.train_steps}
        for mode in (("random", "history") if reference else ("greedy", "sampled")):
            values = jax.device_get(evaluators[mode](agents, eval_keys(rnd)))
            for name, values_per_seed in values.items():
                prefix = f"reference/{mode}" if reference else f"eval/{mode}"
                payload.update(common.agg(values_per_seed, f"{prefix}/{name}"))
                if mode == "greedy":
                    payload.update(common.agg(values_per_seed, f"agg/{name}"))
                for i, value in enumerate(values_per_seed):
                    payload[f"seed_{i}/{prefix}/{name}"] = float(value)
                    if mode == "greedy" and name == "return":
                        payload[f"seed_{i}/agent/mean_return"] = float(value)
            print(f"  {mode:7s}: return={float(np.mean(values['return'])):+.4f} "
                  f"progress={float(np.mean(values['progress'])):.3f} "
                  f"opportunities/ep={float(np.mean(values['opportunities'])):.2f}", flush=True)
        return payload

    if not args.resume_from:
        emit(evaluate_round(0, reference=True))
        emit(evaluate_round(0))
    train = jax.jit(jax.vmap(lambda s, es, ec, k: fns.train_unrolled(
        s, env, env.default_params, es, ec, k)), donate_argnums=(0, 1, 2))
    training_times = []
    compile_s = 0.0
    if start < args.rounds:
        _, compile_keys = common.split_each(keys)
        t0 = time.perf_counter()
        compiled = train.lower(agents, env_states, carries, compile_keys).compile()
        compile_s = time.perf_counter() - t0
        print(f"Training compilation: {compile_s:.1f}s", flush=True)
    for rnd in range(start + 1, args.rounds + 1):
        keys, tk = common.split_each(keys)
        t0 = time.perf_counter()
        agents, env_states, carries, metrics = compiled(agents, env_states, carries, tk)
        jax.block_until_ready(agents)
        duration = time.perf_counter() - t0
        training_times.append(duration)
        speed = args.train_steps / np.mean(training_times)
        print(f"Round {rnd}/{args.rounds}: {duration:.2f}s, "
              f"{speed:.1f} updates/s/seed ({args.num_seeds} seeds together); "
              f"100k={100000 / speed / 3600:.2f}h, 1M={1000000 / speed / 3600:.2f}h", flush=True)
        payload = {"round": rnd, "train_env_steps": rnd * args.train_steps,
                   "env_interactions": args.prefill_steps + rnd * args.train_steps,
                   "timing/train_s": duration, "timing/compile_s": compile_s,
                   "timing/updates_per_second_per_seed": float(speed),
                   "timing/projected_100k_hours": float(100000 / speed / 3600),
                   "timing/projected_1M_hours": float(1000000 / speed / 3600)}
        for name, value in jax.device_get(metrics).items():
            value = np.asarray(value)
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"Non-finite training metric {name} at round {rnd}: {value}")
            payload.update(common.agg(value, f"agg/{name}"))
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            t0 = time.perf_counter()
            payload.update(evaluate_round(rnd))
            payload["timing/eval_s"] = time.perf_counter() - t0
            history.append({"round": rnd, "returns": [payload[f"seed_{i}/agent/mean_return"]
                                                      for i in range(args.num_seeds)]})
        if args.checkpoint_every and (rnd % args.checkpoint_every == 0 or rnd == args.rounds):
            t0 = time.perf_counter()
            save_checkpoint(args.output_dir / "checkpoint.pkl", args, agents,
                            env_states, carries, keys, rnd, history)
            payload["timing/checkpoint_s"] = time.perf_counter() - t0
        emit(payload)

    summary = {"round": args.rounds, "train_env_steps": args.rounds * args.train_steps,
               "env_interactions": args.prefill_steps + args.rounds * args.train_steps}
    if history:
        last = np.asarray([row["returns"] for row in history[-5:]])
        summary.update(common.agg(last.mean(axis=0), "final/return_smoothed"))
        summary["final/evaluations_averaged"] = len(last)
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    if wandb is not None:
        wandb.summary.update(summary)
        wandb.finish()
    print(f"Done: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
