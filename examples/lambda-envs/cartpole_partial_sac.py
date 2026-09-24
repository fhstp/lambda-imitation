"""Partially-observable CartPole on the shared runner: SAC + λ-discrepancy.

CartPole-v1 with the two velocity components masked out, so the observation is
only ``[cart position, pole angle]``.  Velocity has to be integrated from the
observation–action history, which is what the recurrent memory is for — the same
question the Battleship and PocMan scripts ask, in the cheapest environment that
asks it.

Unlike those two there is **no probe**: the hidden state here is a pair of
continuous velocities, not a set of cells, so there is nothing discrete to decode
and the return curve is the whole story.  The script therefore stops after
training and evaluation, and `_probe_common.add_common_args(include_probe=False)`
keeps the probe flags off the CLI rather than accepting and ignoring them.

Everything else matches the other runners: multi-seed vmapped training, the same
flag names, offline / expert-prefill / PER / Retrace, per-round evaluation and
the same W&B keys (`agg/return/*` against `env_interactions`), so its curves
overlay the others directly.

Usage:
    python cartpole_partial_sac.py                          # partial, 1 seed
    python cartpole_partial_sac.py --num-seeds 5 --wandb
    python cartpole_partial_sac.py --full-obs               # the MDP control
    python cartpole_partial_sac.py --memory-type identity    # memoryless control
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _probe_common as common

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Partially-observable CartPole.")

g = parser.add_argument_group("environment")
g.add_argument("--full-obs", dest="partial", action="store_false",
               help="keep the full 4-dim observation (the MDP control: with the "
                    "velocities visible, no memory is needed)")
g.add_argument("--partial", dest="partial", action="store_true",
               help="mask cart velocity and pole angular velocity (default)")
parser.set_defaults(partial=True)
g.add_argument("--eval-max-steps", type=int, default=500,
               help="episode cap for the evaluation scans (default 500 = the "
                    "CartPole-v1 time limit, which is also the maximum return)")

common.add_common_args(
    parser, output_dir_default="./cartpole_partial_output",
    wandb_project_default="offline-lambda-cartpole-results",
    include_probe=False)

# CartPole is small and fast; these defaults make a single-GPU run cheap.
parser.set_defaults(
    rounds=20, train_steps=5_000, gamma=0.99, tau=0.005,
    memory_type="gru", memory_hidden_dim=128, projection_dim=64,
    batch_size=64, sequence_length=20, burn_in_length=5,
    online_buffer_size=100_000, alpha=0.2, autotune_alpha=True,
)

args = common.parse_args(parser, extra_flags_env="CARTPOLE_EXTRA_FLAGS")
_SWEEP_RUN = common.apply_sweep_config(parser, args)

# ── always-needed imports ────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

# ── wandb (optional) ─────────────────────────────────────────────────────────

_wandb = common.init_wandb(args, {
    "env": "CartPole-v1",
    "experiment": "partial_cartpole",
    "partial": args.partial,
    "algo": ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else ""),
    "eval_max_steps": args.eval_max_steps,
    **common.common_wandb_config(args),
}, _SWEEP_RUN)

# ── setup ────────────────────────────────────────────────────────────────────

import optax
from tqdm.rich import tqdm

try:
    import gymnax
except ImportError:
    sys.exit("gymnax required.  pip install 'lambda-imitation[gymnax]'")

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


class PartialObsEnv:
    """Expose only cart position and pole angle (drop both velocities).

    Same masking as ``examples/cartpole_sac_mc.py`` and as ``mask_dims=[0, 2]``
    in lambda-envs' ``get_gymnax_env``: obs_shape (4,) -> (2,).
    """

    _KEEP = jnp.array([0, 2])

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def _mask(self, obs):
        return obs[self._KEEP]

    def get_obs(self, state, params=None):
        return self._mask(self._wrapped.get_obs(state, params))

    def reset(self, key, params):
        obs, state = self._wrapped.reset(key, params)
        return self._mask(obs), state

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._wrapped.step(
            key, state, action, params)
        return self._mask(obs), new_state, reward, done, info

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


_raw_env, env_params = gymnax.make("CartPole-v1")
spec = env_spec_from_gymnax(_raw_env, env_params)   # CartPole-v1: (4,), 2 actions
if args.partial:
    # The wrapper forwards observation_space to the inner env, so the spec has to
    # be narrowed by hand -- otherwise the buffer is built for 4-dim rows and the
    # 2-dim masked observation fails to broadcast into it.
    env = PartialObsEnv(_raw_env)
    spec = spec._replace(obs_shape=(int(PartialObsEnv._KEEP.shape[0]),))
else:
    env = _raw_env
NUM_ACTIONS = int(spec.action_dim)

if args.memory_type == "identity" and args.partial:
    print("[control] identity memory on the partial env: the velocities are "
          "unobservable and cannot be integrated, so this is the memoryless "
          "floor, not a bug.")

os.makedirs(args.output_dir, exist_ok=True)

CARRY_DIM = (0 if args.memory_type == "identity"
             else 2 * args.memory_hidden_dim if args.memory_type == "lstm"
             else args.memory_hidden_dim)


def zero_carry():
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


def zero_prev_action():
    return jnp.zeros((NUM_ACTIONS,), dtype=jnp.float32)


hp = Hyperparameters(
    batch_size=args.batch_size,
    online_buffer_size=args.online_buffer_size,
    target_entropy=args.target_entropy,
    fe_lr=args.fe_lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
    lambda_critic_lr=args.critic_lr, alpha_lr=1e-4,
    alpha=args.alpha, autotune_alpha=args.autotune_alpha,
    gamma=args.gamma, tau=args.tau,
    lambda1=0.05, lambda2=0.85,
    c_bar=1.0, rho_bar=1.0, lambda_truncation=20,
    sequence_length=args.sequence_length,
    burn_in_length=args.burn_in_length,
    lambda_coef=args.lambda_coef, fake_onpolicy_loss=False,
    actor_critic=args.actor_critic,
    gvd_coef=args.gvd_coef, gvd_lambda1=args.gvd_lambda1,
    gvd_lambda2=args.gvd_lambda2, gvd_sf_lr=args.gvd_sf_lr,
    gvd_stop_fe=args.gvd_stop_fe,
    stop_actor_fe=args.stop_actor_fe, stop_critic_fe=args.stop_critic_fe,
    ld_center=args.ld_center, retrace=args.retrace,
    per_alpha=args.per_alpha, per_beta=args.per_beta,
    per_ratio_floor=args.per_ratio_floor, per_window=args.per_window,
)

_MAX_STEPS = min(int(env_params.max_steps_in_episode), args.eval_max_steps)
expert_data = {"observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
               "actions": jnp.zeros((1, 1), dtype=jnp.float32)}


def _build_agent(seed_val):
    return create_iqlearn_from_env(
        spec, expert_data, buffer_size=1, hp=hp,
        projection=args.projection_dim if args.projection_dim > 0 else None,
        memory_type=args.memory_type, memory_hidden_dim=args.memory_hidden_dim,
        actor_dims=(256, 256), critic_dims=(256, 256),
        lambda1_critic_dims=(256, 256), lambda2_critic_dims=(256, 256),
        train_steps=args.train_steps, approximate_lambda=args.approximate_lambda,
        use_prev_action=True, critic_layer_norm=args.critic_layer_norm,
        burn_in_from_stored_carry=args.burn_in_from_stored_carry,
        use_gvd=args.gvd, gvd_sf_dims=(256, 256),
        debug=True, seed=seed_val, use_sac=args.use_sac)


tag = ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else "")
print(f"Building {tag} agent for CartPole-v1 "
      f"({'partial: position + angle' if args.partial else 'full observation'}, "
      f"memory={args.memory_type}, hidden={args.memory_hidden_dim})…")
state, fns, debug_fns = _build_agent(args.seed)

_transplant_fe, _save_fe = common.make_fe_checkpointer(args.init_fe, args.output_dir)
_eval_kwargs = dict(zero_carry=zero_carry, zero_prev_action=zero_prev_action,
                    max_steps=_MAX_STEPS)
evaluate = common.make_evaluate(fns, env, env_params, **_eval_kwargs)
evaluate_critic = (
    common.make_evaluate_critic(debug_fns, env, env_params,
                                num_actions=NUM_ACTIONS, **_eval_kwargs)
    if (args.critic_greedy_eval
        and getattr(debug_fns, "predict_qpi", None) is not None)
    else None)

# ── multi-seed training ──────────────────────────────────────────────────────

CONCURRENT = args.concurrent_seeds or args.num_seeds
if CONCURRENT < 1:
    sys.exit("--concurrent-seeds must be >= 1")
if args.num_seeds % CONCURRENT != 0:
    sys.exit(f"--num-seeds ({args.num_seeds}) must be divisible by "
             f"--concurrent-seeds ({CONCURRENT}).")
seeds = [args.seed + i for i in range(args.num_seeds)]
n_groups = args.num_seeds // CONCURRENT
print(f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT} "
      f"trained concurrently (vmap); {args.rounds} rounds × {args.train_steps} steps.")

PREFILL_STEPS = hp.batch_size * (
    hp.lambda_truncation + hp.sequence_length + hp.burn_in_length)
_PREFILL_N = max(PREFILL_STEPS, args.expert_prefill_steps)
_reset_v = jax.jit(jax.vmap(lambda k: env.reset(k, env_params)))
_prefill_v = jax.jit(
    jax.vmap(lambda s, es, k: fns.prefill_buffer(s, env, env_params, es,
                                                 _PREFILL_N, k),
             in_axes=(0, 0, 0)),
    donate_argnums=(0, 1))
if args.offline:
    def _offline_round(s, es, ec, k):
        s, m = fns.update_only(s, args.train_steps, k)
        return s, es, ec, m

    _train_v = jax.jit(jax.vmap(_offline_round, in_axes=(0, 0, 0, 0)),
                       donate_argnums=(0,))
else:
    _train_v = jax.jit(
        jax.vmap(lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
                 in_axes=(0, 0, 0, 0)),
        donate_argnums=(0, 1))


def _evaluate_v(states_b, keys, n):
    return jax.vmap(lambda s, k: evaluate(s, k, n_episodes=n))(states_b, keys)


def _evaluate_critic_v(states_b, keys, n):
    return jax.vmap(lambda s, k: evaluate_critic(s, k, n_episodes=n))(states_b, keys)


return_hist = {gi: [] for gi in range(args.num_seeds)}


def run_group(group_idx, gidxs):
    svals = [seeds[gi] for gi in gidxs]
    print(f"\n{'=' * 60}\nGroup {group_idx + 1}/{n_groups}  seeds={svals}\n{'=' * 60}")
    states = [_transplant_fe(state if gi == 0 else _build_agent(seeds[gi])[0])
              for gi in gidxs]
    batched = common.stack_states(states)
    keys = jnp.stack([jax.random.key(sv) for sv in svals])
    keys, reset_keys = common.split_each(keys)
    _obs, env_state = _reset_v(reset_keys)
    keys, prefill_keys = common.split_each(keys)
    print(f"  prefilling {_PREFILL_N} steps/seed…")
    batched, env_state = _prefill_v(batched, env_state, prefill_keys)
    zero_carry_b = jnp.zeros((len(gidxs), CARRY_DIM), dtype=jnp.float32)

    def round_eval(keys, batched):
        keys, eval_keys = common.split_each(keys)
        returns, steps, done = _evaluate_v(batched, eval_keys, 20)
        evals = {"return": np.array(returns), "steps": np.array(steps),
                 "done_frac": np.array(done), "cg": None}
        if evaluate_critic is not None:
            keys, cg_keys = common.split_each(keys)
            cgr, cgs, cgd, qsp, qrg = _evaluate_critic_v(batched, cg_keys, 10)
            evals["cg"] = (np.array(cgr), np.array(cgs), np.array(cgd),
                           np.array(qsp), np.array(qrg))
        return keys, evals

    def on_round(rnd, step, evals, metrics):
        returns, steps, cg = evals["return"], evals["steps"], evals["cg"]
        for j, gi in enumerate(gidxs):
            return_hist[gi].append(float(returns[j]))
        print(f"Round {rnd:4d}/{args.rounds}  "
              f"return={float(returns.mean()):7.1f}±{float(returns.std()):.1f}  "
              f"steps={float(steps.mean()):6.1f}")
        if _wandb is None:
            return
        payload = {"round": rnd, "env_interactions": step}
        payload.update(common.agg(returns, "agg/return"))
        payload.update(common.agg(steps, "agg/steps"))
        if cg is not None:
            payload.update(common.agg(cg[0], "agg/critic_greedy_return"))
            payload.update(common.agg(cg[3], "agg/q_action_spread"))
        for mk, mv in metrics.items():
            payload.update(common.agg(np.array(mv), f"agg/{mk}"))
        for j, gi in enumerate(gidxs):
            payload[f"seed_{gi}/agent/mean_return"] = float(returns[j])
            for mk, mv in metrics.items():
                payload[f"seed_{gi}/agent/{mk}"] = float(np.array(mv)[j])
        _wandb.log(payload)

    batched, keys = common.run_seed_group(
        rounds=args.rounds, train_steps=args.train_steps,
        keys=keys, batched=batched, env_state=env_state,
        zero_carry_b=zero_carry_b, train_v=_train_v,
        round_eval=round_eval, on_round=on_round,
        tqdm=tqdm, desc=f"Group {group_idx + 1}")
    return batched


indexed = list(range(args.num_seeds))
groups = [indexed[g * CONCURRENT:(g + 1) * CONCURRENT] for g in range(n_groups)]
seed_states = [None] * args.num_seeds
if not args.skip_train:
    for g, gidxs in enumerate(groups):
        batched = run_group(g, gidxs)
        for j, gi in enumerate(gidxs):
            seed_states[gi] = common.unstack_state(batched, j)
            leaves, treedef = jax.tree.flatten(seed_states[gi])
            with open(os.path.join(args.output_dir, f"agent_seed{gi}.pkl"), "wb") as f:
                pickle.dump({"leaves": [np.array(l) for l in leaves],
                             "treedef": treedef}, f)
    print(f"Saved {args.num_seeds} per-seed agents → {args.output_dir}/agent_seed*.pkl")

    W = max(1, args.final_return_window)
    fin = np.array([np.mean(return_hist[gi][-W:]) for gi in range(args.num_seeds)])
    agg_final = common.agg(fin, "final/return_smoothed")
    print(f"\n{'=' * 60}\nAggregated over {args.num_seeds} seed(s) "
          f"(smoothed over final {W} round(s)):\n"
          f"  final/return_smoothed = {agg_final['final/return_smoothed/mean']:.1f} "
          f"± {agg_final['final/return_smoothed/sterr']:.1f} (sterr)\n{'=' * 60}")
    if _wandb is not None:
        _wandb.log(agg_final)
        _wandb.summary.update(agg_final)
else:
    for gi in range(args.num_seeds):
        with open(os.path.join(args.output_dir, f"agent_seed{gi}.pkl"), "rb") as f:
            saved = pickle.load(f)
        _, treedef = jax.tree.flatten(state)
        seed_states[gi] = treedef.unflatten([jnp.array(l) for l in saved["leaves"]])

if _wandb is not None:
    _wandb.finish()
print("Done.")
