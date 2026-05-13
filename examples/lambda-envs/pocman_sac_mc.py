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
    python pocman_sac_mc.py --help                       # full option list
"""

import argparse
import sys

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
    help="number of training rounds (default: 200)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=10000,
    metavar="N",
    help="env steps and gradient updates per round (default: 1000)",
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
    help="hidden-state width of the recurrent cell (default: 512)",
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

    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=sc.get("online_buffer_size", 100_000),
        target_entropy=sc.get("target_entropy", 0.3),
        fe_lr=sc.get("fe_lr", 1e-4),
        actor_lr=sc.get("actor_lr", 1e-4),
        critic_lr=sc.get("critic_lr", 1e-4),
        lambda_critic_lr=sc.get("lambda_critic_lr", 1e-4),
        alpha_lr=sc.get("alpha_lr", 1e-4),
        alpha=sc.get("alpha", 0.2),
        autotune_alpha=_sweep_bool(sc.get("autotune_alpha", False)),
        batch_size=sc.get("batch_size", 256),
        gamma=0.95,
        tau=sc.get("tau", 0.005),
        lambda1=sc.get("lambda1", 0.05),
        lambda2=sc.get("lambda2", 0.8),
        c_bar=sc.get("c_bar", 1.0),
        rho_bar=sc.get("rho_bar", 1.0),
        lambda_truncation=17,
        sequence_length=sc.get("sequence_length", 20),
        burn_in_length=sc.get("burn_in_length", 20),
        lambda_coef=args.lambda_coef,
        fake_onpolicy_loss=_sweep_bool(sc.get("fake_onpolicy_loss", True)),
    )
else:
    hp = Hyperparameters(
        online_batch_size=128,
        online_buffer_size=100_000,
        target_entropy=0.3,
        actor_lr=1e-4,
        critic_lr=1e-4,
        lambda_critic_lr=1e-4,
        alpha_lr=1e-4,
        alpha=0.2,
        autotune_alpha=False,
        batch_size=256,
        gamma=0.95,
        tau=0.005,
        lambda1=0.05,
        lambda2=0.8,
        lambda_truncation=17,
        sequence_length=20,
        burn_in_length=20,
        lambda_coef=args.lambda_coef,
    )

# ── build agent ───────────────────────────────────────────────────────────────

projection_dim = args.projection_dim if args.projection_dim > 0 else None

print(
    f"Building SAC + λ-discrepancy agent for PocMan "
    f"(discrete, lambda-envs)  memory={args.memory_type} "
    f"hidden={args.memory_hidden_dim} projection={projection_dim}…"
)
state, fns, debug_fns = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,
    hp=hp,
    projection_dim=projection_dim,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    critic_dims=(128,),
    lambda1_critic_dims=(128,),
    lambda2_critic_dims=(128,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    debug=True,
    seed=args.seed,
)

# ── carry helper ──────────────────────────────────────────────────────────────

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim


def zero_carry() -> jax.Array:
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


# ── initial environment reset ─────────────────────────────────────────────────

key = jax.random.key(args.seed)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)

# ── evaluation helper ─────────────────────────────────────────────────────────


def evaluate(agent_state, rng_key, n_episodes: int = 10) -> float:
    total = 0.0
    max_steps = int(env_params.max_steps_in_episode)
    for _ in range(n_episodes):
        rng_key, rk = jax.random.split(rng_key)
        ep_obs, ep_state = env.reset(rk, env_params)
        ep_carry = zero_carry()
        ep_return = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            rng_key, sk = jax.random.split(rng_key)
            raw, ep_carry = fns.predict(
                agent_state, ep_obs, ep_carry, sk, deterministic=True
            )
            action = jnp.round(raw).astype(jnp.int32)
            rng_key, ek = jax.random.split(rng_key)
            ep_obs, ep_state, reward, done, _ = env.step(
                ek, ep_state, action, env_params
            )
            ep_return += float(reward)
            done = bool(done)
            step += 1
        total += ep_return
    return total / n_episodes


# ── wandb (non-sweep init) ───────────────────────────────────────────────────

if _wandb is not None and not args.wandb_sweep:
    _wandb.init(
        entity="fhstp-data-intelligence-research-group",
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "env": "PocMan",
            "algo": "SAC+lambda" if args.approximate_lambda else "SAC",
            "action_space": "discrete",
            "approximate_lambda": args.approximate_lambda,
            "memory_type": args.memory_type,
            "memory_hidden_dim": args.memory_hidden_dim,
            "projection_dim": projection_dim,
            "rounds": args.rounds,
            "train_steps": args.train_steps,
            "seed": args.seed,
            **hp._asdict(),
        },
    )

# ── training loop ─────────────────────────────────────────────────────────────

total_steps = args.rounds * args.train_steps
print(
    f"Training for {args.rounds} rounds × {args.train_steps} steps "
    f"= {total_steps} total env steps."
)

for rnd in tqdm(range(1, args.rounds + 1)):
    key, train_key = jax.random.split(key)
    state, env_state, metrics = fns.train(state, env, env_params, env_state, train_key)

    key, eval_key = jax.random.split(key)
    mean_return = evaluate(state, eval_key, n_episodes=10)

    print(
        f"Round {rnd:4d}/{args.rounds}  "
        f"mean_return={mean_return:7.1f}  "
        f"alpha={float(metrics.get('alpha', state.alpha)):.4f}  "
        f"critic_loss={float(metrics.get('critic_loss', jnp.nan)):.4f}  "
        f"entropy={float(metrics.get('entropy', jnp.nan)):.4f}  "
        f"q={float(metrics.get('q', jnp.nan)):7.3f}"
    )

    if _wandb is not None:
        _wandb.log(
            {
                "round": rnd,
                "step": rnd * args.train_steps,
                "mean_return": mean_return,
                **{k: float(v) for k, v in metrics.items()},
            },
            step=rnd * args.train_steps,
        )

print("\nTraining complete.")
key, eval_key = jax.random.split(key)
final_return = evaluate(state, eval_key, n_episodes=20)
print(f"Final evaluation (20 episodes): mean_return={final_return:.1f}")

if _wandb is not None:
    _wandb.log({"final_mean_return": final_return})
    _wandb.finish()
