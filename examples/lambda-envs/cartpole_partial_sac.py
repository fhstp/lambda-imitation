"""Partially-observable CartPole on the shared runner: SAC + λ-discrepancy.

CartPole-v1 with the two velocity components masked out, so the observation is
only ``[cart position, pole angle]``.  Velocity has to be integrated from the
observation–action history, which is what the recurrent memory is for — the same
question the Battleship and PocMan scripts ask, in the cheapest environment that
asks it.

The probe is a **regression** one, unlike Battleship's and PocMan's: the hidden
state here is a pair of continuous velocities, not a set of cells, so it decodes
the full 4-dim state `[x, x_dot, theta, theta_dot]` from the carry under MSE and
scores it with per-dimension R2.  `x` and `theta` are in the observation and
should come out near 1 — they are the wiring check; `x_dot` and `theta_dot` are
the measurement, since nothing but the observation-action history carries them.
Only the periodic (every `--probe-eval-interval` rounds) probe exists, so the
collect/visualise flags of the other two scripts stay off the CLI.

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
from functools import partial
from typing import NamedTuple

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
g.add_argument("--obs-noise", type=float, default=0.0,
               help="std of Gaussian observation noise, as a FRACTION of each "
                    "dimension's range (position +-4.8, angle +-0.42 rad), so one "
                    "value means the same for both.  0 = clean (default).  Noise "
                    "is what turns this from a 2-step inference problem into one "
                    "where the memory has to average over a window: the velocity "
                    "is still recoverable, but no longer from two observations.")
g.add_argument("--head-dim", type=int, default=64,
               help="width of every actor/critic hidden layer (2 layers)")
g.add_argument("--eval-max-steps", type=int, default=500,
               help="episode cap for the evaluation scans (default 500 = the "
                    "CartPole-v1 time limit, which is also the maximum return)")

g = parser.add_argument_group("full-state probe")
g.add_argument("--probe-eval-interval", type=int, default=10, metavar="ROUNDS",
               help="train and score a probe every N rounds (0 disables)")
g.add_argument("--probe-eval-steps", type=int, default=20_000,
               help="SGD steps per probe eval (default 20 k)")
g.add_argument("--probe-eval-collect-steps", type=int, default=4_000,
               help="env steps per probe dataset (a train and a test set)")
g.add_argument("--probe-hidden-dim", type=int, default=256)
g.add_argument("--probe-lr", type=float, default=1e-3)
g.add_argument("--probe-batch-size", type=int, default=256)

common.add_common_args(
    parser, output_dir_default="./cartpole_partial_output",
    wandb_project_default="offline-lambda-cartpole-results",
    include_probe=False)

# Sized for speed, not for headroom: CartPole's state is four numbers and its
# observation two, so a 32-unit GRU and 64-wide heads are not the bottleneck --
# a 512/256 network here mostly buys compile time.  The stability settings (the
# learning rates, alpha, stop-actor-fe, layer norm) are still the ones from the
# Battleship 5x5 run that held together (W&B 6d5cfugb); only the sizes shrank.
#
# A round is 1000 steps, so the return curve and the probe both report early and
# often: --probe-eval-interval 10 is a probe every 10k env steps.  Sequences are
# short (5 with a 3-step burn-in) because velocity is recoverable from a couple
# of observations; lambda_truncation stays at 30, which is what sets how far the
# lambda-discrepancy target looks ahead.
parser.set_defaults(
    rounds=200, train_steps=1_000, gamma=0.99, tau=0.005,
    memory_type="gru", memory_hidden_dim=32, projection_dim=32,
    batch_size=64, sequence_length=5, burn_in_length=3,
    online_buffer_size=100_000,
    alpha=0.1, autotune_alpha=False, target_entropy=0.0,
    fe_lr=1e-5, actor_lr=1e-4, critic_lr=1e-4,
    lambda_coef=0.01, retrace=True, use_sac=False, stop_actor_fe=True,
    critic_layer_norm=True, actor_critic="sac",
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
    "obs_noise": args.obs_noise,
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


class _NoisyState(NamedTuple):
    """Inner env state plus the key that generated the CURRENT observation.

    The noise has to live in the state: the library calls ``get_obs(state)`` in
    places where it has no PRNG key (the buffer prefill, for one), and if that
    returned a differently-noised observation than the one ``step`` returned for
    the same state, the stored transitions would disagree with the rollout.
    """

    inner: object
    noise_key: jax.Array


class PartialObsEnv:
    """Expose only cart position and pole angle, optionally with noise.

    Masking is the same as ``examples/cartpole_sac_mc.py`` and ``mask_dims=[0,
    2]`` in lambda-envs' ``get_gymnax_env``: obs_shape (4,) -> (2,).

    ``noise_std`` is a fraction of each dimension's own range, because position
    (±4.8) and angle (±0.42 rad) differ by more than 10x and a single absolute
    sigma would drown the angle while barely touching the position.  Velocity
    dimensions have infinite bounds, so they take a scale of 1.0 (their actual
    spread is order 1) -- only relevant with --full-obs.
    """

    _KEEP = jnp.array([0, 2])

    def __init__(self, wrapped, noise_std=0.0, partial=True):
        self._wrapped = wrapped
        self.noise_std = float(noise_std)
        self.partial = partial
        high = jnp.asarray(wrapped.observation_space(wrapped.default_params).high)
        scale = jnp.where(jnp.isfinite(high), high, 1.0)
        self._scale = scale[self._KEEP] if partial else scale

    def _observe(self, obs, noise_key):
        if self.partial:
            obs = obs[self._KEEP]
        if self.noise_std > 0.0:
            obs = obs + self.noise_std * self._scale * jax.random.normal(
                noise_key, obs.shape)
        return obs

    def get_obs(self, state, params=None):
        return self._observe(self._wrapped.get_obs(state.inner, params),
                             state.noise_key)

    def reset(self, key, params):
        env_key, noise_key = jax.random.split(key)
        obs, inner = self._wrapped.reset(env_key, params)
        return self._observe(obs, noise_key), _NoisyState(inner, noise_key)

    def step(self, key, state, action, params):
        env_key, noise_key = jax.random.split(key)
        obs, inner, reward, done, info = self._wrapped.step(
            env_key, state.inner, action, params)
        return (self._observe(obs, noise_key), _NoisyState(inner, noise_key),
                reward, done, info)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


_raw_env, env_params = gymnax.make("CartPole-v1")
spec = env_spec_from_gymnax(_raw_env, env_params)   # CartPole-v1: (4,), 2 actions
# Always wrapped, so the noise key is carried in the state even when the noise is
# off (noise_std=0 reproduces the clean env exactly).
env = PartialObsEnv(_raw_env, noise_std=args.obs_noise, partial=args.partial)
if args.partial:
    # The wrapper forwards observation_space to the inner env, so the spec has to
    # be narrowed by hand -- otherwise the buffer is built for 4-dim rows and the
    # 2-dim masked observation fails to broadcast into it.
    spec = spec._replace(obs_shape=(int(PartialObsEnv._KEEP.shape[0]),))
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
    c_bar=1.0, rho_bar=1.0, lambda_truncation=30,
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


_HEAD = (args.head_dim, args.head_dim)


def _build_agent(seed_val):
    return create_iqlearn_from_env(
        spec, expert_data, buffer_size=1, hp=hp,
        projection=args.projection_dim if args.projection_dim > 0 else None,
        memory_type=args.memory_type, memory_hidden_dim=args.memory_hidden_dim,
        actor_dims=_HEAD, critic_dims=_HEAD,
        lambda1_critic_dims=_HEAD, lambda2_critic_dims=_HEAD,
        train_steps=args.train_steps, approximate_lambda=args.approximate_lambda,
        use_prev_action=True, critic_layer_norm=args.critic_layer_norm,
        burn_in_from_stored_carry=args.burn_in_from_stored_carry,
        use_gvd=args.gvd, gvd_sf_dims=_HEAD,
        debug=True, seed=seed_val, use_sac=args.use_sac)


tag = ("SAC+LD" if args.approximate_lambda else "SAC") + ("+GVD" if args.gvd else "")
print(f"Building {tag} agent for CartPole-v1 "
      f"({'partial: position + angle' if args.partial else 'full observation'}"
      f"{f', obs noise {args.obs_noise}' if args.obs_noise else ''}, "
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

# ── full-state probe ─────────────────────────────────────────────────────────
#
# What the mask removes is velocity, so the probe regresses the whole 4-dim
# CartPole state out of the carry.  x and theta are in the observation and
# should decode near R2 1 -- they are the wiring check, not the result.  x_dot
# and theta_dot are the result: nothing but the observation-action history
# carries them, so their R2 IS "how much velocity does this memory hold".
#
# Targets are z-scored with the training set's mean/std before the MSE, because
# theta (+-0.21 rad) is an order of magnitude smaller than x_dot and an
# unnormalised MSE would simply ignore it.  R2 is invariant to that rescaling.

STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")
PROBE_ON = (args.probe_eval_interval > 0 and CARRY_DIM > 0
            and not args.skip_train)


@partial(jax.jit, static_argnames=["n_steps"])
def _collect_probe_data(agent_state, key, n_steps):
    """One on-policy stream: (carry after obs_t, true state at t) per step."""
    key, rk = jax.random.split(key)
    obs, env_st = env.reset(rk, env_params)

    def step_fn(s, _):
        obs, env_st, carry, pa, key = s
        key, sk, ek = jax.random.split(key, 3)
        # sampled, not greedy: the probe wants the states the agent visits
        # while still exploring, not a single deterministic trajectory.
        raw, nc = fns.predict(agent_state, obs, carry, sk, deterministic=False,
                              prev_action=pa)
        inner = env_st.inner
        truth = jnp.stack([inner.x, inner.x_dot, inner.theta, inner.theta_dot])
        action = jnp.round(raw).astype(jnp.int32)
        nobs, nst, _rew, d, _ = env.step(ek, env_st, action, env_params)
        npa = fns.encode_action(jnp.atleast_1d(raw))
        # gymnax auto-resets the env; only the memory has to be reset by hand.
        # nc is recorded BEFORE this, so the pair is (history through t, state t).
        return (nobs, nst,
                jnp.where(d, jnp.zeros_like(nc), nc),
                jnp.where(d, jnp.zeros_like(npa), npa), key), (nc, truth)

    init = (obs, env_st, zero_carry(), zero_prev_action(), key)
    _, (carries, truths) = jax.lax.scan(step_fn, init, length=n_steps)
    return carries, truths


if PROBE_ON:
    _collect_v = jax.jit(jax.vmap(
        lambda s, k: _collect_probe_data(s, k, args.probe_eval_collect_steps)))
    _probe_train_v = common.make_probe_trainer(
        optax.adam(args.probe_lr), carry_dim=CARRY_DIM, n_out=len(STATE_NAMES),
        hidden=args.probe_hidden_dim, batch_size=args.probe_batch_size,
        loss="mse")[1]


def make_probe_hook(gidxs):
    """A `common.Hooks` that trains a fresh probe on the current memory."""
    n = len(gidxs)
    rng = jax.random.PRNGKey(args.seed + 9973)

    def compute(batched, rnd):
        nonlocal rng
        rng, tr_k, te_k, init_k, train_k = jax.random.split(rng, 5)
        c_tr, t_tr = _collect_v(batched, jax.random.split(tr_k, n))
        c_te, t_te = _collect_v(batched, jax.random.split(te_k, n))
        mu = t_tr.mean(axis=1, keepdims=True)
        sd = t_tr.std(axis=1, keepdims=True) + 1e-6
        params = _probe_train_v(c_tr, (t_tr - mu) / sd,
                                jax.random.split(init_k, n),
                                jax.random.split(train_k, n),
                                args.probe_eval_steps)
        preds = jax.vmap(common.probe_forward)(params, c_te)
        return rnd, np.array(preds), np.array((t_te - mu) / sd)

    def render(host):
        rnd, preds, targets = host
        per_seed = [common.probe_metrics_regression(preds[j], targets[j],
                                                    STATE_NAMES)
                    for j in range(n)]
        r2 = {nm: np.array([m[f"r2_{nm}"] for m in per_seed]) for nm in STATE_NAMES}
        print("  probe R2  " + "  ".join(
            f"{nm}={float(np.nanmean(v)):+.3f}" for nm, v in r2.items()))
        if _wandb is None:
            return
        payload = {"round": rnd, "env_interactions": rnd * args.train_steps}
        for k in per_seed[0]:
            payload.update(common.agg(
                np.array([m[k] for m in per_seed]), f"probe_eval/{k}"))
        for j, gi in enumerate(gidxs):
            for k, v in per_seed[j].items():
                payload[f"seed_{gi}/probe/{k}"] = float(v)
        _wandb.log(payload)

    return common.Hooks(compute, render)


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
        probe=make_probe_hook(gidxs) if PROBE_ON else None,
        probe_interval=args.probe_eval_interval if PROBE_ON else 0,
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
