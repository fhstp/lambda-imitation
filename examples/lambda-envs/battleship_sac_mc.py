"""SAC + λ-discrepancy online training demo: Battleship (discrete, action-masked).

Pure online reinforcement learning with SAC and twin λ-critic branches — no
expert data or imitation learning is used.  The placeholder expert buffer
required by ``create_iqlearn_from_env`` is never sampled
(``fns.train()`` is never called).

Battleship is a partially-observable hidden-board game.  Each step the agent
fires at one of ``rows*cols`` cells; the only feedback is a single bit — did the
last shot hit a ship.  Most actions are illegal at any step (you cannot fire on
a cell you have already hit), so the environment ships a *legal-action mask* and
this demo wires it into the policy via ``mask_fn``: invalid actions never
receive probability mass.

``lambda_envs.envs.battleship.Battleship`` packs both signals into one flat
observation: ``obs = [last_hit_miss (1) | valid_action_mask (rows*cols)]``.  We
split it at the agent boundary:

* ``obs_fn = obs[..., :1]``   → the only thing the feature extractor sees (the
  board history is *not* given to the network; the recurrent memory must
  reconstruct it — a strongly partially-observed task, matching the repo's
  POMDP / memory / λ-discrepancy focus).
* ``mask_fn = obs[..., 1:]``  → the legal-action mask applied to the policy.

The feature extractor is configurable via ``--memory-type`` (identity, rnn,
gru, lstm), ``--memory-hidden-dim`` and ``--projection-dim``; layout is
``[obs_fn(obs) | prev-action one-hot] -> Linear(projection_dim) -> memory
cell``.  The prev-action input is ON by default (mirroring
``action_concat=True`` in the original lambda-discrepancy battleship
experiments); disable with ``--no-use-prev-action``.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install lambda-envs

Usage
-----
    python battleship_sac_mc.py                              # defaults (10x10)
    python battleship_sac_mc.py --rounds 200                 # more training
    python battleship_sac_mc.py --rows 5 --cols 5            # smaller board
    python battleship_sac_mc.py --memory-type lstm           # LSTM memory
    python battleship_sac_mc.py --wandb                      # enable W&B logging
    python battleship_sac_mc.py --num-seeds 10 --concurrent-seeds 10 \
        --wandb --wandb-per-seed                             # 10 concurrent → 10 W&B runs
    python battleship_sac_mc.py --num-seeds 5                # 5 seeds, aggregate
    python battleship_sac_mc.py --help                       # full option list
    wandb agent <entity>/<project>/<sweep_id>                # hyperparams come
                                                             # from the sweep
                                                             # config (see
                                                             # "wandb sweep
                                                             # support" below)
"""

import argparse
import os
import sys
from functools import partial

from tqdm import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="battleship_sac_mc.py",
    description="SAC + λ-discrepancy on Battleship (discrete, action-masked).",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=100,
    metavar="N",
    help="number of training rounds (default: 20)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=10000,
    metavar="N",
    help="env steps and gradient updates per round (default: 10000)",
)
parser.add_argument(
    "--rows",
    type=int,
    default=10,
    metavar="N",
    help="board rows (default: 10)",
)
parser.add_argument(
    "--cols",
    type=int,
    default=10,
    metavar="N",
    help="board cols (default: 10)",
)
parser.add_argument(
    "--dense-reward",
    dest="dense_reward",
    action="store_true",
    help="reward every hit (default: sparse — terminal reward only)",
)
parser.set_defaults(dense_reward=False)
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
    default=10,
    metavar="N",
    help="number of seeds per parameter set (default: 1)",
)
parser.add_argument(
    "--concurrent-seeds",
    type=int,
    default=10,
    metavar="N",
    help=(
        "number of seeds trained concurrently in a single vmapped+jitted "
        "kernel (default: 10). --num-seeds must be divisible by this."
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
    default=512,
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
    dest="use_prev_action",
    action="store_true",
    help=(
        "feed the previous action (one-hot) into the feature extractor "
        "alongside the observation, mirroring action_concat=True in the "
        "original lambda-discrepancy battleship experiments (default)"
    ),
)
parser.add_argument(
    "--no-use-prev-action",
    dest="use_prev_action",
    action="store_false",
    help="disable the prev-action input to the feature extractor",
)
parser.set_defaults(use_prev_action=True)
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
parser.set_defaults(gvd=True)
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
    default=16,
    metavar="N",
    help="width of the random obs projection in the GVD feature map; total "
    "feature dim is 1 (explicit hit bit) + N (default: 16)",
)
parser.add_argument(
    "--gvd-lambda1",
    type=float,
    default=0.05,
    metavar="L",
    help="λ of the first GVD successor-feature head (default: 0.0)",
)
parser.add_argument(
    "--gvd-lambda2",
    type=float,
    default=0.75,
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
parser.set_defaults(approximate_lambda=False)
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
    default=1e-4,
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
    default=1e-4,
    help="λ-critic learning rate (default: 1.8e-4)",
)
parser.add_argument(
    "--fe-lr",
    type=float,
    default=1e-4,
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
    default=0.005,
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
    default=128,
    help="training batch size (default: 512)",
)
parser.add_argument(
    "--online-batch-size",
    type=int,
    default=128,
    help="online batch size (default: 128)",
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
    default=40,
    help="sequence length for R2D2-style sampling (default: 40)",
)
parser.add_argument(
    "--burn-in-length",
    type=int,
    default=5,
    help="burn-in steps for recurrent state (default: 5)",
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
parser.set_defaults(burn_in_from_stored_carry=True)
parser.add_argument(
    "--c-bar",
    type=float,
    default=1.05,
    help="V-trace IS clipping c̄ (default: 1.17)",
)
parser.add_argument(
    "--rho-bar",
    type=float,
    default=1.05,
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
    "--wandb-project",
    default="offline-lambda-battleship",
    metavar="PROJECT",
    help="W&B project name (default: offline-lambda-battleship)",
)
parser.add_argument(
    "--wandb-run-name",
    default=None,
    metavar="NAME",
    help="W&B run-name prefix; each seed becomes '<name>-seed<i>' (default: auto)",
)
parser.add_argument(
    "--wandb-per-seed",
    dest="wandb_per_seed",
    action="store_true",
    help="log each seed as its OWN W&B run (10 concurrent runs → 10 runs in W&B), "
    "instead of one aggregated run with seed_i/ prefixes (default)",
)
parser.set_defaults(wandb_per_seed=False)
parser.add_argument(
    "--wandb-group",
    default=None,
    metavar="GROUP",
    help="W&B group tying the per-seed runs together in --wandb-per-seed mode "
    "(default: the run-name prefix, else auto)",
)
args, _ = parser.parse_known_args()

# ── wandb sweep support ───────────────────────────────────────────────────────
#
# Under ``wandb agent`` the hyperparameters arrive via the sweep config, not
# the CLI: the agent passes underscore-style ``--name=value`` args that match
# neither the dashed flags nor the store_true/store_false pairs above (they
# are dropped by parse_known_args).  Attach to the run the agent pre-created
# and override args from its config instead.

_SWEEP_RUN = None
if os.environ.get("WANDB_SWEEP_ID"):
    import wandb as _wandb_mod

    _SWEEP_RUN = _wandb_mod.init()
    _SWEEP_ALIASES = {"use_gvd": "gvd"}
    _SWEEP_IGNORED = {"action_masked", "buffer_size"}  # not script knobs
    _ARG_TYPES = {a.dest: a.type for a in parser._actions if callable(a.type)}
    for _k, _v in dict(_SWEEP_RUN.config).items():
        _key = _SWEEP_ALIASES.get(_k, _k)
        if _key in _SWEEP_IGNORED:
            continue
        if not hasattr(args, _key):
            print(f"sweep config: ignoring unknown parameter {_k!r}")
            continue
        if isinstance(_v, str) and _v.lower() in ("true", "false"):
            _v = _v.lower() == "true"
        elif _key in _ARG_TYPES:
            _v = _ARG_TYPES[_key](_v)
        setattr(args, _key, _v)
    # The agent owns exactly one run; log everything into it.
    args.wandb = True
    args.wandb_per_seed = False

if args.seed is None:
    args.seed = int.from_bytes(os.urandom(4), "little")
    print(f"No --seed given, using random seed: {args.seed}")

# ── imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp

try:
    from lambda_envs.envs.battleship import Battleship
except ImportError:
    sys.exit("lambda-envs is required.  Install with:  pip install lambda-envs")

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

# ── environment setup ─────────────────────────────────────────────────────────
#
# Battleship subclasses the gymnax Environment base, so reset/step (with
# auto-reset on done) and get_obs(state, params) work directly — no adapter
# needed.  The observation packs the legal-action mask into its tail; we split
# it for the agent below.

env = Battleship(rows=args.rows, cols=args.cols, dense_reward=args.dense_reward)
env_params = env.default_params
spec = env_spec_from_gymnax(env, env_params)
# Battleship: obs_shape=(1 + rows*cols,), action_dim=rows*cols, is_discrete=True

# obs_fn: what the feature extractor sees (only the last-shot-result bit).
# mask_fn: the legal-action mask applied to the policy everywhere.
obs_fn = lambda o: o[..., :1]
mask_fn = lambda o: o[..., 1:]

# GVD feature map φ over the FULL raw observation (hit bit + action mask):
# an explicit hit-bit feature (1 of 1+rows*cols dims — pure random mixing
# would dilute the only stochastic-but-predictable signal) plus a fixed
# random projection.  Built once at module level with a seed independent of
# --seed, so the projection is one constant shared across all vmapped seeds
# and identical between training and probing runs.  The library diffs φ, so
# the SF cumulant is [Δhit | P·Δobs].
if args.gvd:
    _RAW_OBS_DIM = 1 + args.rows * args.cols
    _GVD_P = jax.random.normal(
        jax.random.key(0), (_RAW_OBS_DIM, args.gvd_features)
    ) / jnp.sqrt(_RAW_OBS_DIM)

    def gvd_feature_fn(o):
        return jnp.concatenate([o[..., :1], o @ _GVD_P], axis=-1)

else:
    gvd_feature_fn = None

# ── placeholder expert buffer (never sampled — SAC only) ──────────────────────

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── wandb (optional) ─────────────────────────────────────────────────────────

_wandb = None
if args.wandb:
    try:
        import wandb as _wandb_mod

        _wandb = _wandb_mod
    except ImportError:
        sys.exit("wandb not installed.  Install with:  pip install wandb")

hp = Hyperparameters(
    online_batch_size=args.online_batch_size,
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
    gamma=0.99,
    tau=args.tau,
    lambda1=args.lambda1,
    lambda2=args.lambda2,
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

# ── carry helper ──────────────────────────────────────────────────────────────

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim
if args.use_prev_action:
    # The carry tail holds the prev-action one-hot (see
    # RecurrentFeatureExtractor.prev_action_dim).
    CARRY_DIM += spec.action_dim

projection_dim = args.projection_dim if args.projection_dim > 0 else None


def zero_carry() -> jax.Array:
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


# ── evaluation helper ─────────────────────────────────────────────────────────


# Battleship episodes end after at most rows*cols shots — the legal-action
# mask exhausts the board and the last ship cell must be hit by then — far
# below max_steps_in_episode (1000).  Scanning further wastes ~10x eval time
# on done-frozen steps.
_MAX_STEPS = min(int(env_params.max_steps_in_episode), args.rows * args.cols + 1)


def _make_evaluate(fns):
    """Build a JIT-compiled evaluator that uses lax.scan + vmap."""

    @partial(jax.jit, static_argnames=["n_episodes"])
    def _evaluate(agent_state, rng_key, n_episodes=10):
        def run_episode(key):
            key, rk = jax.random.split(key)
            obs, env_st = env.reset(rk, env_params)
            carry = zero_carry()

            def step_fn(state, _):
                obs, env_st, carry, key, ret, done = state
                key, sk, ek = jax.random.split(key, 3)
                # predict masks invalid actions internally (via mask_fn).
                raw, new_carry = fns.predict(
                    agent_state, obs, carry, sk, deterministic=True
                )
                action = jnp.round(raw).astype(jnp.int32)
                next_obs, next_st, reward, d, _ = env.step(
                    ek, env_st, action, env_params
                )
                ret = ret + reward * (1.0 - done)
                done = jnp.maximum(done, d.astype(jnp.float32))
                return (next_obs, next_st, new_carry, key, ret, done), None

            init = (obs, env_st, carry, key, jnp.float32(0.0), jnp.float32(0.0))
            (_, _, _, _, ep_return, _), _ = jax.lax.scan(
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
    projection_dim=projection_dim,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    use_prev_action=args.use_prev_action,
    critic_dims=(128,),
    lambda1_critic_dims=(128,),
    lambda2_critic_dims=(128,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    critic_layer_norm=True,
    obs_fn=obs_fn,
    mask_fn=mask_fn,
    burn_in_from_stored_carry=args.burn_in_from_stored_carry,
    use_gvd=args.gvd,
    gvd_feature_fn=gvd_feature_fn,
    gvd_sf_dims=(128,),
    debug=True,
)


def _build_agent(seed_val: int):
    return create_iqlearn_from_env(spec, expert_data, **_AGENT_KWARGS, seed=seed_val)


# ── vmapped multi-seed training ───────────────────────────────────────────────

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


# ── wandb init ───────────────────────────────────────────────────────────────
#
# Two logging modes:
#   * default              — one aggregated run, seeds under ``seed_i/`` prefixes
#   * --wandb-per-seed     — one W&B run PER seed (see _make_wandb_runs below);
#                            seeds still train concurrently (vmapped), but each
#                            logs as its own run.  WANDB_RUNS is filled later
#                            (needs the seed list).

_WANDB_BASE_CFG = {
    "env": "Battleship",
    "algo": "SAC"
    + ("+lambda" if args.approximate_lambda else "")
    + ("+gvd" if args.gvd else ""),
    "action_space": "discrete",
    "action_masked": True,
    "use_gvd": args.gvd,
    "gvd_features": args.gvd_features,
    "rows": args.rows,
    "cols": args.cols,
    "dense_reward": args.dense_reward,
    "burn_in_from_stored_carry": args.burn_in_from_stored_carry,
    "approximate_lambda": args.approximate_lambda,
    "memory_type": args.memory_type,
    "memory_hidden_dim": args.memory_hidden_dim,
    "projection_dim": projection_dim,
    "use_prev_action": args.use_prev_action,
    "rounds": args.rounds,
    "train_steps": args.train_steps,
    "num_seeds": args.num_seeds,
    "concurrent_seeds": args.concurrent_seeds,
    **hp._asdict(),
}

WANDB_RUNS = {}  # gi -> run handle, populated in --wandb-per-seed mode

if _wandb is not None and not args.wandb_per_seed:
    if _SWEEP_RUN is None:
        _wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**_WANDB_BASE_CFG, "seed": args.seed},
        )
    else:
        # Already attached to the agent's run; only add the descriptive keys
        # the sweep doesn't control (sweep keys are locked and can't change).
        _SWEEP_RUN.config.update(
            {
                k: v
                for k, v in {**_WANDB_BASE_CFG, "seed": args.seed}.items()
                if k not in _SWEEP_RUN.config
            }
        )
    if args.num_seeds > 1:
        _wandb.define_metric("env_interactions")
        for i in range(args.num_seeds):
            _wandb.define_metric(f"seed_{i}/*", step_metric="env_interactions")


def _make_wandb_runs(seeds):
    """--wandb-per-seed: one W&B run handle per seed (index gi -> run).

    ``reinit="create_new"`` keeps several runs open at once in a single process
    (plain ``reinit=True`` would finish the previous run on each init); a shared
    ``group`` ties them together in the dashboard.
    """
    group = args.wandb_group or args.wandb_run_name
    for gi, sv in enumerate(seeds):
        name = f"{args.wandb_run_name}-seed{gi}" if args.wandb_run_name else f"seed{gi}"
        r = _wandb.init(
            project=args.wandb_project,
            name=name,
            group=group,
            reinit="create_new",
            config={**_WANDB_BASE_CFG, "seed": sv, "seed_idx": gi},
        )
        r.define_metric("env_interactions")
        r.define_metric("*", step_metric="env_interactions")
        WANDB_RUNS[gi] = r


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

if _wandb is not None and args.wandb_per_seed:
    _make_wandb_runs(seeds)

print(
    f"Building SAC + λ-discrepancy agent for Battleship "
    f"({args.rows}x{args.cols}, discrete, action-masked)  "
    f"memory={args.memory_type} hidden={args.memory_hidden_dim} "
    f"projection={projection_dim}…"
)
print(
    f"{args.num_seeds} seed(s) in {n_groups} group(s) of {CONCURRENT} trained "
    f"concurrently (vmap); {args.rounds} rounds × {args.train_steps} steps each."
)
state_0, fns, debug_fns = _build_agent(seeds[0])
evaluate = _make_evaluate(fns)

# vmapped + jitted env reset / prefill / train over the leading seed axis.
# donate_argnums hands the (large: replay buffer ~100MB/seed) input agent
# state and env state buffers to XLA for in-place reuse — the caller rebinds
# both to the outputs every call, so the stale inputs are never read again.
# The zero carry (arg 2 of _train_v) is reused across rounds: NOT donated.
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
    """Train one group of ``CONCURRENT`` seeds concurrently; return finals."""
    idxs = [gi for gi, _ in group]
    svals = [sv for _, sv in group]
    print(
        f"\n{'=' * 60}\n"
        f"Group {group_idx + 1}/{n_groups}  seeds={svals} "
        f"(global idx {idxs[0]}–{idxs[-1]})\n"
        f"{'=' * 60}"
    )

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
        if rnd == 2:
            jax.profiler.start_trace("./jax-trace")
        keys, train_keys = _split_each(keys)
        batched, env_state, _carry, metrics = _train_v(
            batched, env_state, zero_carry_b, train_keys
        )
        _carry.block_until_ready()

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
            step = rnd * args.train_steps
            if args.wandb_per_seed:
                # each seed → its own run handle
                for j, gi in enumerate(idxs):
                    WANDB_RUNS[gi].log(
                        {
                            "round": rnd,
                            "env_interactions": step,
                            "mean_return": float(returns[j]),
                            **{k: float(v[j]) for k, v in metrics.items()},
                        },
                        step=step,
                    )
            elif args.num_seeds == 1:
                _wandb.log(
                    {
                        "round": rnd,
                        "env_interactions": step,
                        "mean_return": float(returns[0]),
                        **{k: float(v[0]) for k, v in metrics.items()},
                    },
                    step=step,
                )
            else:
                for j, gi in enumerate(idxs):
                    prefix = f"seed_{gi}"
                    _wandb.log(
                        {
                            "env_interactions": step,
                            f"{prefix}/mean_return": float(returns[j]),
                            **{
                                f"{prefix}/{k}": float(v[j]) for k, v in metrics.items()
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
    jax.profiler.stop_trace()
    return [(gi, float(final_returns_b[j])) for j, gi in enumerate(idxs)]


indexed_seeds = list(enumerate(seeds))
groups = [indexed_seeds[g * CONCURRENT : (g + 1) * CONCURRENT] for g in range(n_groups)]

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
    if args.wandb_per_seed:
        # one summary value per run, then close each run
        for gi, ret in enumerate(final_returns):
            run = WANDB_RUNS[gi]
            run.summary["final_return"] = ret
            run.summary["group_mean_final_return"] = mean_ret
            run.summary["group_std_final_return"] = std_ret
            run.finish()
    else:
        _wandb.log(
            {
                "mean_final_return": mean_ret,
                "std_final_return": std_ret,
            }
        )
        for i, ret in enumerate(final_returns):
            _wandb.log({f"final_return_seed_{i}": ret})
        _wandb.finish()
