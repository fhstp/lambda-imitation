"""SAC + λ-discrepancy online training demo: PocMan (discrete, lambda-envs).

Pure online reinforcement learning with SAC and twin λ-critic branches — no
expert data or imitation learning is used.  The placeholder expert buffer
required by ``create_iqlearn_from_env`` is never sampled
(``fns.train()`` is never called).

PocMan is a partially-observable PacMan variant (built on Jumanji's PacMan):
the agent observes only walls in the four cardinal directions, a "smell"
indicator for nearby pellets, an "audible ghost" indicator, line-of-sight
ghost flags in each direction, and a powerpill flag — 11 binary features
total.  Action space is the four cardinal moves.

The feature extractor is configurable via ``--memory-type`` (identity, rnn,
gru, lstm), ``--memory-hidden-dim`` and ``--projection-dim``; layout is
``obs -> Linear(projection_dim) -> memory cell``.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install "lambda-envs[pocman]"   # the [pocman] extra pulls in jumanji

Usage
-----
    python pocman_sac_mc.py                              # defaults
    python pocman_sac_mc.py --rounds 200                 # more training
    python pocman_sac_mc.py --memory-type lstm            # LSTM memory
    python pocman_sac_mc.py --wandb                      # enable W&B logging
    python pocman_sac_mc.py --num-seeds 5                # 5 seeds, aggregate
    python pocman_sac_mc.py --num-seeds 8 --concurrent-seeds 4  # 2×4 vmapped
    python pocman_sac_mc.py --help                       # full option list
"""

import argparse
import sys
from functools import partial

from tqdm.rich import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="pocman_sac_mc.py",
    description="SAC + λ-discrepancy on PocMan (discrete, lambda-envs).",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=20,
    metavar="N",
    help="number of training rounds (default: 20)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=10000,
    metavar="N",
    help="env steps and gradient updates per round (default: 1000)",
)
parser.add_argument(
    "--lambda1",
    type=float,
    default=0.05,
    metavar="LC",
    help="lambda1 head (default: 0.05)",
)
parser.add_argument(
    "--lambda2",
    type=float,
    default=0.75,
    metavar="LC",
    help="lambda2 head (default: 0.75)",
)
parser.add_argument(
    "--lambda-coef",
    type=float,
    default=1.0,
    metavar="LC",
    help="lambda discrepancy coefficient (default: 1.0)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    metavar="S",
    help="JAX random seed (default: random from /dev/urandom)",
)
parser.add_argument(
    "--num-seeds",
    type=int,
    default=1,
    metavar="N",
    help="number of seeds per parameter set (default: 1; sweep default: 5)",
)
parser.add_argument(
    "--concurrent-seeds",
    type=int,
    default=10,
    metavar="N",
    help=(
        "number of seeds trained concurrently in a single vmapped+jitted "
        "kernel (default: 1 = sequential). --num-seeds must be divisible by "
        "this. Higher values trade GPU memory for throughput; the sweet spot "
        "is GPU-dependent (the workload is often latency/occupancy bound, so "
        "vmapping seeds is a near-free speedup until memory runs out)"
    ),
)
parser.add_argument(
    "--memory-type",
    choices=("identity", "rnn", "gru", "lstm"),
    default="gru",
    help="recurrent cell after the linear projection (default: gru)",
)
parser.add_argument(
    "--memory-hidden-dim",
    type=int,
    default=750,
    metavar="N",
    help="hidden-state width of the recurrent cell (default: 750)",
)
parser.add_argument(
    "--projection-dim",
    type=int,
    default=128,
    metavar="N",
    help=(
        "width of the linear obs embedding before the memory cell "
        "(default: 128; pass 0 to disable the projection and feed raw obs)"
    ),
)
parser.add_argument(
    "--use-prev-action",
    action="store_true",
    help=(
        "feed the previous action (one-hot) into the feature extractor "
        "alongside the observation, mirroring action_concat=True in the "
        "original lambda-discrepancy pocman experiments"
    ),
)
parser.add_argument(
    "--gvd",
    dest="gvd",
    action="store_true",
    help="enable the General Value Discrepancy branches (two successor-"
    "feature V-heads on observable-feature cumulants + squared discrepancy "
    "regulariser on the FE; reward-free memory pressure)",
)
parser.add_argument(
    "--no-gvd",
    dest="gvd",
    action="store_false",
    help="disable GVD (default)",
)
parser.set_defaults(gvd=False)
parser.add_argument(
    "--gvd-coef",
    type=float,
    default=1.0,
    metavar="GC",
    help="GVD discrepancy coefficient (default: 1.0)",
)
parser.add_argument(
    "--gvd-features",
    type=int,
    default=0,
    metavar="N",
    help="width of the random obs projection in the GVD feature map; 0 "
    "(default) uses the identity over the 11 binary obs features",
)
parser.add_argument(
    "--gvd-lambda1",
    type=float,
    default=0.0,
    metavar="L",
    help="λ of the first GVD successor-feature head (default: 0.0)",
)
parser.add_argument(
    "--gvd-lambda2",
    type=float,
    default=1.0,
    metavar="L",
    help="λ of the second GVD successor-feature head (default: 1.0)",
)
parser.add_argument(
    "--gvd-sf-lr",
    type=float,
    default=1.8e-4,
    help="GVD successor-feature head learning rate (default: 1.8e-4)",
)
parser.add_argument(
    "--approximate-lambda",
    dest="approximate_lambda",
    action="store_true",
    help="enable the twin λ-critic branches (default)",
)
parser.add_argument(
    "--no-approximate-lambda",
    dest="approximate_lambda",
    action="store_false",
    help="disable the λ-critic branches and run pure SAC",
)
parser.set_defaults(approximate_lambda=True)
parser.add_argument(
    "--grad-clip-norm",
    type=float,
    default=0.5,
    metavar="G",
    help="global-norm gradient clip (0.0 disables; default: 0.5)",
)
parser.add_argument(
    "--actor-lr",
    type=float,
    default=6e-5,
    help="actor learning rate (default: 6e-5)",
)
parser.add_argument(
    "--critic-lr",
    type=float,
    default=2e-4,
    help="critic learning rate (default: 2e-4)",
)
parser.add_argument(
    "--lambda-critic-lr",
    type=float,
    default=1.8e-4,
    help="λ-critic learning rate (default: 1.8e-4)",
)
parser.add_argument(
    "--fe-lr",
    type=float,
    default=7e-5,
    help="feature extractor learning rate (default: 7e-5)",
)
parser.add_argument(
    "--alpha-lr",
    type=float,
    default=1e-4,
    help="entropy coef learning rate (default: 1e-4)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.1,
    help="initial entropy coefficient (default: 0.1)",
)
parser.add_argument(
    "--autotune-alpha",
    dest="autotune_alpha",
    action="store_true",
    help="enable automatic entropy tuning",
)
parser.add_argument(
    "--no-autotune-alpha",
    dest="autotune_alpha",
    action="store_false",
    help="disable automatic entropy tuning (default)",
)
parser.set_defaults(autotune_alpha=False)
parser.add_argument(
    "--tau",
    type=float,
    default=0.006,
    help="target network EMA rate (default: 0.006)",
)
parser.add_argument(
    "--target-entropy",
    type=float,
    default=0.0,
    help="target entropy (default: 0.0)",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512,
    help="training batch size (default: 512)",
)
parser.add_argument(
    "--online-buffer-size",
    type=int,
    default=200_000,
    help="replay buffer capacity (default: 200000)",
)
parser.add_argument(
    "--sequence-length",
    type=int,
    default=20,
    help="sequence length for R2D2-style sampling (default: 20)",
)
parser.add_argument(
    "--burn-in-length",
    type=int,
    default=32,
    help="burn-in steps for recurrent state (default: 32)",
)
parser.add_argument(
    "--burn-in-from-stored-carry",
    dest="burn_in_from_stored_carry",
    action="store_true",
    help=(
        "store the online recurrent carry per transition and initialise the "
        "training burn-in from it instead of zeros (R2D2 stored-state; "
        "enables a much shorter --burn-in-length).  Costs carry_dim x "
        "buffer_size x 4 bytes extra memory per seed"
    ),
)
parser.add_argument(
    "--no-burn-in-from-stored-carry",
    dest="burn_in_from_stored_carry",
    action="store_false",
    help="initialise the training burn-in from zeros (default)",
)
parser.set_defaults(burn_in_from_stored_carry=False)
parser.add_argument(
    "--c-bar",
    type=float,
    default=1.17,
    help="V-trace IS clipping c̄ (default: 1.17)",
)
parser.add_argument(
    "--rho-bar",
    type=float,
    default=1.15,
    help="V-trace IS clipping ρ̄ (default: 1.15)",
)
parser.add_argument(
    "--fake-onpolicy-loss",
    dest="fake_onpolicy_loss",
    action="store_true",
    help="use fake on-policy loss",
)
parser.add_argument(
    "--no-fake-onpolicy-loss",
    dest="fake_onpolicy_loss",
    action="store_false",
    help="disable fake on-policy loss (default)",
)
parser.set_defaults(fake_onpolicy_loss=False)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="enable Weights & Biases logging",
)
parser.add_argument(
    "--wandb-sweep",
    action="store_true",
    help="run as a W&B sweep agent (implies --wandb); hyperparameters are read from wandb.config",
)
parser.add_argument(
    "--wandb-project",
    default="offline-lambda-pocman",
    metavar="PROJECT",
    help="W&B project name (default: offline-lambda-pocman)",
)
parser.add_argument(
    "--wandb-run-name",
    default=None,
    metavar="NAME",
    help="W&B run name (default: auto-generated)",
)
args, _ = parser.parse_known_args()
if args.seed is None:
    import os

    args.seed = int.from_bytes(os.urandom(4), "little")
    print(f"No --seed given, using random seed: {args.seed}")
if not args.wandb_sweep and "WANDB_SWEEP_ID" in __import__("os").environ:
    args.wandb_sweep = True
if args.wandb_sweep:
    args.wandb = True

# ── imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp

try:
    from lambda_envs.envs.pocman import PocMan
except ImportError:
    sys.exit(
        "lambda-envs[pocman] is required.  Install with:  "
        "pip install 'lambda-envs[pocman]'"
    )

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

# ── env adapter: add params kwarg to get_obs ──────────────────────────────────
#
# lambda-imitation calls ``env.get_obs(state, params)`` (gymnax convention),
# but PocMan.get_obs() only accepts ``(state)``.  This wrapper drops the extra
# arg.  No env behaviour changes: PocMan's get_obs never reads params, and
# reset/step pass through unchanged via __getattr__.


class _LambdaEnvAdapter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_obs(self, state, params=None):
        return self._wrapped.get_obs(state)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# ── environment setup ─────────────────────────────────────────────────────────

env = _LambdaEnvAdapter(PocMan())
env_params = env.default_params
spec = env_spec_from_gymnax(env, env_params)
# PocMan: obs_shape=(11,), action_dim=4, is_discrete=True

# ── placeholder expert buffer (never sampled — SAC only) ──────────────────────

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── wandb sweep (early init) ─────────────────────────────────────────────────

_wandb = None
if args.wandb:
    try:
        import wandb as _wandb_mod

        _wandb = _wandb_mod
    except ImportError:
        sys.exit("wandb not installed.  Install with:  pip install wandb")

if args.wandb_sweep:
    _wandb.init(
        entity="fhstp-data-intelligence-research-group",
        project=args.wandb_project,
    )
    sc = _wandb.config

    def _sweep_bool(val):
        if isinstance(val, bool):
            return val
        return str(val).lower() == "true"

    args.memory_hidden_dim = sc.get("memory_hidden_dim", args.memory_hidden_dim)
    args.lambda_coef = sc.get("lambda_coef", args.lambda_coef)
    args.num_seeds = sc.get("num_seeds", 10)
    args.concurrent_seeds = sc.get("concurrent_seeds", args.concurrent_seeds)
    args.burn_in_from_stored_carry = _sweep_bool(
        sc.get("burn_in_from_stored_carry", args.burn_in_from_stored_carry)
    )
    args.gvd = _sweep_bool(sc.get("gvd", args.gvd))
    args.gvd_coef = sc.get("gvd_coef", args.gvd_coef)

    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=sc.get("online_buffer_size", args.online_buffer_size),
        target_entropy=sc.get("target_entropy", args.target_entropy),
        fe_lr=sc.get("fe_lr", args.fe_lr),
        actor_lr=sc.get("actor_lr", args.actor_lr),
        critic_lr=sc.get("critic_lr", args.critic_lr),
        lambda_critic_lr=sc.get("lambda_critic_lr", args.lambda_critic_lr),
        alpha_lr=sc.get("alpha_lr", args.alpha_lr),
        alpha=sc.get("alpha", args.alpha),
        autotune_alpha=_sweep_bool(sc.get("autotune_alpha", args.autotune_alpha)),
        batch_size=sc.get("batch_size", args.batch_size),
        gamma=0.95,
        tau=sc.get("tau", args.tau),
        lambda1=sc.get("lambda1", args.lambda1),
        lambda2=sc.get("lambda2", args.lambda2),
        c_bar=sc.get("c_bar", args.c_bar),
        rho_bar=sc.get("rho_bar", args.rho_bar),
        lambda_truncation=17,
        sequence_length=sc.get("sequence_length", args.sequence_length),
        burn_in_length=sc.get("burn_in_length", args.burn_in_length),
        lambda_coef=args.lambda_coef,
        fake_onpolicy_loss=_sweep_bool(
            sc.get("fake_onpolicy_loss", args.fake_onpolicy_loss)
        ),
        gvd_coef=args.gvd_coef,
        gvd_lambda1=sc.get("gvd_lambda1", args.gvd_lambda1),
        gvd_lambda2=sc.get("gvd_lambda2", args.gvd_lambda2),
        gvd_sf_lr=sc.get("gvd_sf_lr", args.gvd_sf_lr),
    )
else:
    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=args.online_buffer_size,
        target_entropy=args.target_entropy,
        fe_lr=args.fe_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        lambda_critic_lr=args.lambda_critic_lr,
        alpha_lr=args.alpha_lr,
        alpha=args.alpha,
        autotune_alpha=args.autotune_alpha,
        batch_size=args.batch_size,
        gamma=0.95,
        tau=args.tau,
        lambda1=0.05,
        lambda2=0.8,
        c_bar=args.c_bar,
        rho_bar=args.rho_bar,
        lambda_truncation=17,
        sequence_length=args.sequence_length,
        burn_in_length=args.burn_in_length,
        lambda_coef=args.lambda_coef,
        fake_onpolicy_loss=args.fake_onpolicy_loss,
        gvd_coef=args.gvd_coef,
        gvd_lambda1=args.gvd_lambda1,
        gvd_lambda2=args.gvd_lambda2,
        gvd_sf_lr=args.gvd_sf_lr,
    )

# GVD feature map φ over the raw 11-binary-feature observation.  Identity by
# default (the obs is already a compact feature vector; its differences are
# an informative cumulant); ``--gvd-features N>0`` swaps in a fixed random
# projection (seed independent of --seed so it is one constant shared across
# all vmapped seeds).  Built after the sweep branch so a sweep-set ``gvd``
# flag is honoured.
if args.gvd:
    if args.gvd_features > 0:
        _GVD_P = jax.random.normal(
            jax.random.key(0), (spec.obs_shape[0], args.gvd_features)
        ) / jnp.sqrt(spec.obs_shape[0])

        # Pocman GVD probes the raw obs directly — it has no battleship-style
        # hit bit to localise, so the prev-action arg is accepted and ignored.
        def gvd_feature_fn(o, a_prev):
            return o @ _GVD_P
    else:
        gvd_feature_fn = lambda o, a_prev: o
else:
    gvd_feature_fn = None

# ── carry helper ──────────────────────────────────────────────────────────────

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim
# The carry is the memory state only; the prev-action is threaded separately
# (not packed into the carry).
PREV_ACTION_DIM = spec.action_dim if args.use_prev_action else 0

projection_dim = args.projection_dim if args.projection_dim > 0 else None


def zero_carry() -> jax.Array:
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


def zero_prev_action() -> jax.Array:
    return jnp.zeros((PREV_ACTION_DIM,), dtype=jnp.float32)


# ── evaluation helper ─────────────────────────────────────────────────────────


_MAX_STEPS = int(env_params.max_steps_in_episode)


def _make_evaluate(fns):
    """Build a JIT-compiled evaluator that uses lax.scan + vmap."""

    @partial(jax.jit, static_argnames=["n_episodes"])
    def _evaluate(agent_state, rng_key, n_episodes=10):
        def run_episode(key):
            key, rk = jax.random.split(key)
            obs, env_st = env.reset(rk, env_params)
            carry = zero_carry()
            prev_action = zero_prev_action()

            def step_fn(state, _):
                obs, env_st, carry, prev_action, key, ret, done = state
                key, sk, ek = jax.random.split(key, 3)
                raw, new_carry = fns.predict(
                    agent_state, obs, carry, sk, deterministic=True,
                    prev_action=prev_action,
                )
                action = jnp.round(raw).astype(jnp.int32)
                next_obs, next_st, reward, d, _ = env.step(
                    ek, env_st, action, env_params
                )
                new_prev_action = (
                    fns.encode_action(jnp.atleast_1d(raw))
                    if args.use_prev_action
                    else prev_action
                )
                ret = ret + reward * (1.0 - done)
                done = jnp.maximum(done, d.astype(jnp.float32))
                new_prev_action = jnp.where(
                    done > 0, jnp.zeros_like(new_prev_action), new_prev_action
                )
                return (
                    next_obs, next_st, new_carry, new_prev_action, key, ret, done
                ), None

            init = (
                obs, env_st, carry, prev_action, key,
                jnp.float32(0.0), jnp.float32(0.0),
            )
            (_, _, _, _, _, ep_return, _), _ = jax.lax.scan(
                step_fn, init, length=_MAX_STEPS
            )
            return ep_return

        keys = jax.random.split(rng_key, n_episodes)
        returns = jax.vmap(run_episode)(keys)
        return jnp.mean(returns)

    return _evaluate


# ── agent factory (shared across seeds) ─────────────────────────────────────

_AGENT_KWARGS = dict(
    buffer_size=1,
    hp=hp,
    projection=projection_dim,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    use_prev_action=args.use_prev_action,
    critic_dims=(128,),
    lambda1_critic_dims=(128,),
    lambda2_critic_dims=(128,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    burn_in_from_stored_carry=args.burn_in_from_stored_carry,
    use_gvd=args.gvd,
    gvd_feature_fn=gvd_feature_fn,
    gvd_sf_dims=(128,),
    debug=True,
)


def _build_agent(seed_val: int):
    return create_iqlearn_from_env(spec, expert_data, **_AGENT_KWARGS, seed=seed_val)


# ── vmapped multi-seed training ───────────────────────────────────────────────
#
# Seeds are trained in groups of ``--concurrent-seeds``: a group's agent states /
# env states / carries / keys are stacked along a leading axis and the
# (env-rollout + grad-update) scan runs under a single ``jax.vmap`` + ``jax.jit``.
# When the workload is latency / occupancy bound on the GPU this trains N seeds
# in roughly the wall-time of one.  ``fns.train_unrolled`` (not ``fns.train``) is
# used because the latter's host-side buffer-warmth check is not vmap-safe; the
# buffer is instead pre-filled explicitly per seed before the round loop.

# Transitions collected before training, matching fns.train's auto-prefill size.
PREFILL_STEPS = hp.online_batch_size * (
    hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
)


def _stack_states(states):
    """Stack a list of per-seed pytrees along a new leading axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def _split_each(keys):
    """Split a batch of PRNG keys, returning two batches (carry, fresh)."""
    out = jax.vmap(lambda k: jax.random.split(k))(keys)
    return out[:, 0], out[:, 1]


# ── wandb (non-sweep init) ───────────────────────────────────────────────────

if _wandb is not None and not args.wandb_sweep:
    _wandb.init(
        entity="fhstp-data-intelligence-research-group",
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "env": "PocMan",
            "algo": "SAC"
            + ("+lambda" if args.approximate_lambda else "")
            + ("+gvd" if args.gvd else ""),
            "action_space": "discrete",
            "approximate_lambda": args.approximate_lambda,
            "use_gvd": args.gvd,
            "gvd_features": args.gvd_features,
            "memory_type": args.memory_type,
            "memory_hidden_dim": args.memory_hidden_dim,
            "projection_dim": projection_dim,
            "use_prev_action": args.use_prev_action,
            "burn_in_from_stored_carry": args.burn_in_from_stored_carry,
            "rounds": args.rounds,
            "train_steps": args.train_steps,
            "num_seeds": args.num_seeds,
            "concurrent_seeds": args.concurrent_seeds,
            "seed": args.seed,
            **hp._asdict(),
        },
    )

# ── wandb metric setup ───────────────────────────────────────────────────────

if _wandb is not None and args.num_seeds > 1:
    _wandb.define_metric("env_interactions")
    for i in range(args.num_seeds):
        _wandb.define_metric(f"seed_{i}/*", step_metric="env_interactions")

# ── main: run seeds (grouped, vmapped) and aggregate ──────────────────────────

CONCURRENT = args.concurrent_seeds
if CONCURRENT < 1:
    sys.exit("--concurrent-seeds must be >= 1")
if args.num_seeds % CONCURRENT != 0:
    sys.exit(
        f"--num-seeds ({args.num_seeds}) must be divisible by "
        f"--concurrent-seeds ({CONCURRENT})."
    )

seeds = [args.seed + i for i in range(args.num_seeds)]
n_groups = args.num_seeds // CONCURRENT

print(
    f"Building SAC + λ-discrepancy agent for PocMan "
    f"(discrete, lambda-envs)  memory={args.memory_type} "
    f"hidden={args.memory_hidden_dim} projection={projection_dim}…"
)
print(
    f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT} trained "
    f"concurrently (vmap); {args.rounds} rounds × {args.train_steps} steps each."
)
state_0, fns, debug_fns = _build_agent(seeds[0])
evaluate = _make_evaluate(fns)

# vmapped + jitted env reset / prefill / train over the leading seed axis.
# env / env_params are captured as closure constants (static), not vmapped.
# donate_argnums hands the (large: replay buffer) input agent state and env
# state buffers to XLA for in-place reuse — the caller rebinds both to the
# outputs every call.  The zero carry (arg 2 of _train_v) is reused across
# rounds: NOT donated.
_reset_v = jax.jit(jax.vmap(lambda k: env.reset(k, env_params)))
_prefill_v = jax.jit(
    jax.vmap(
        lambda s, es, k: fns.prefill_buffer(s, env, env_params, es, PREFILL_STEPS, k),
        in_axes=(0, 0, 0),
    ),
    donate_argnums=(0, 1),
)
_train_v = jax.jit(
    jax.vmap(
        lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
        in_axes=(0, 0, 0, 0),
    ),
    donate_argnums=(0, 1),
)


def _evaluate_v(states, keys, n_episodes):
    return jax.vmap(lambda s, k: evaluate(s, k, n_episodes=n_episodes))(states, keys)


def run_group(group_idx: int, group: list) -> list:
    """Train one group of ``CONCURRENT`` seeds concurrently; return finals.

    ``group`` is a list of ``(global_seed_idx, seed_val)`` pairs.  Returns a
    list of ``(global_seed_idx, final_return)``.
    """
    idxs = [gi for gi, _ in group]
    svals = [sv for _, sv in group]
    print(
        f"\n{'=' * 60}\n"
        f"Group {group_idx + 1}/{n_groups}  seeds={svals} "
        f"(global idx {idxs[0]}–{idxs[-1]})\n"
        f"{'=' * 60}"
    )

    # One agent per seed, stacked into a single batched state.  Reuse the
    # already-built agent for the very first seed of the first group.
    states = []
    for j, sv in enumerate(svals):
        if group_idx == 0 and j == 0:
            states.append(state_0)
        else:
            st, _, _ = _build_agent(sv)
            states.append(st)
    batched = _stack_states(states)

    keys = jnp.stack([jax.random.key(sv) for sv in svals])
    keys, reset_keys = _split_each(keys)
    _obs, env_state = _reset_v(reset_keys)
    keys, prefill_keys = _split_each(keys)
    batched, env_state = _prefill_v(batched, env_state, prefill_keys)

    # Fresh zero carry each round, matching fns.train's per-call carry reset.
    zero_carry_b = jnp.zeros((CONCURRENT, CARRY_DIM), dtype=jnp.float32)

    for rnd in tqdm(range(1, args.rounds + 1), desc=f"Group {group_idx + 1}"):
        keys, train_keys = _split_each(keys)
        batched, env_state, _carry, metrics = _train_v(
            batched, env_state, zero_carry_b, train_keys
        )

        keys, eval_keys = _split_each(keys)
        returns = _evaluate_v(batched, eval_keys, 10)  # (CONCURRENT,)

        def _gm(name):  # group mean of a batched metric, for the console line
            v = metrics.get(name)
            return float(jnp.mean(v)) if v is not None else float("nan")

        print(
            f"Round {rnd:4d}/{args.rounds}  "
            f"mean_return={float(jnp.mean(returns)):7.1f}  "
            f"alpha={_gm('alpha'):.4f}  "
            f"critic_loss={_gm('critic_loss'):.4f}  "
            f"entropy={_gm('entropy'):.4f}  "
            f"q={_gm('q'):7.3f}"
        )

        if _wandb is not None:
            if args.num_seeds == 1:
                _wandb.log(
                    {
                        "round": rnd,
                        "env_interactions": rnd * args.train_steps,
                        "mean_return": float(returns[0]),
                        **{k: float(v[0]) for k, v in metrics.items()},
                    },
                    step=rnd * args.train_steps,
                )
            else:
                for j, gi in enumerate(idxs):
                    prefix = f"seed_{gi}"
                    _wandb.log(
                        {
                            "env_interactions": rnd * args.train_steps,
                            f"{prefix}/mean_return": float(returns[j]),
                            **{
                                f"{prefix}/{k}": float(v[j])
                                for k, v in metrics.items()
                            },
                        }
                    )

    keys, eval_keys = _split_each(keys)
    final_returns_b = _evaluate_v(batched, eval_keys, 20)  # (CONCURRENT,)
    for j, gi in enumerate(idxs):
        print(
            f"Seed {gi + 1} (seed={svals[j]}) final evaluation (20 episodes): "
            f"mean_return={float(final_returns_b[j]):.1f}"
        )
    return [(gi, float(final_returns_b[j])) for j, gi in enumerate(idxs)]


indexed_seeds = list(enumerate(seeds))
groups = [
    indexed_seeds[g * CONCURRENT : (g + 1) * CONCURRENT] for g in range(n_groups)
]

final_returns = [None] * args.num_seeds
for g, group in enumerate(groups):
    for gi, ret in run_group(g, group):
        final_returns[gi] = ret

returns_arr = jnp.array(final_returns)
mean_ret = float(jnp.mean(returns_arr))
std_ret = float(jnp.std(returns_arr))

print(
    f"\n{'=' * 60}\n"
    f"Aggregated over {args.num_seeds} seed(s):\n"
    f"  final_mean_return = {mean_ret:.1f}\n"
    f"  final_std_return  = {std_ret:.1f}\n"
    f"{'=' * 60}"
)

if _wandb is not None:
    _wandb.log(
        {
            "mean_final_return": mean_ret,
            "std_final_return": std_ret,
        }
    )
    for i, ret in enumerate(final_returns):
        _wandb.log({f"final_return_seed_{i}": ret})
    _wandb.finish()
