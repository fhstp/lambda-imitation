"""SAC + λ-discrepancy online training demo: CartPole-v1 (gymnax, discrete).

Pure online reinforcement learning with SAC and twin λ-critic branches — no
expert data or imitation learning is used.  The placeholder expert buffer
required by ``create_iqlearn_from_env`` is never sampled
(``fns.train()`` is never called).

The feature extractor is configurable via ``--memory-type`` (identity, rnn,
gru, lstm), ``--memory-hidden-dim`` and ``--projection-dim``; layout is
``obs -> Linear(projection_dim) -> memory cell``.

After training the script opens a gymnasium render window so you can watch the
trained agent.  Press Enter in the terminal after each episode; Ctrl-C to quit.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install "gymnasium[classic-control]"   # for visualisation

Usage
-----
    python cartpole_sac_mc.py                                # defaults
    python cartpole_sac_mc.py --memory-type gru              # GRU memory
    python cartpole_sac_mc.py --memory-type lstm --memory-hidden-dim 64
    python cartpole_sac_mc.py --partial --memory-type lstm   # POMDP setting
    python cartpole_sac_mc.py --help                         # full option list
"""

import argparse
import sys

from tqdm.rich import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="cartpole_sac.py",
    description="SAC online training on CartPole-v1 (discrete, gymnax).",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=100,
    metavar="N",
    help="number of training rounds (default: 100)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=1000,
    metavar="N",
    help="env steps and gradient updates per round (default: 1000)",
)
parser.add_argument(
    "--lambda-coef",
    type=float,
    default=0.5,
    metavar="N",
    help="lambda discrepancy coefficient (default: 0.5)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="JAX random seed (default: 0)",
)
parser.add_argument(
    "--partial",
    action="store_true",
    help=(
        "make the environment partially observable: hide cart velocity (index 1) "
        "and pole angular velocity (index 3), leaving only cart position and pole angle"
    ),
)
parser.add_argument(
    "--memory-type",
    choices=("identity", "rnn", "gru", "lstm"),
    default="gru",
    help="recurrent cell after the linear projection (default: identity)",
)
parser.add_argument(
    "--memory-hidden-dim",
    type=int,
    default=256,
    metavar="N",
    help="hidden-state width of the recurrent cell (default: 64)",
)
parser.add_argument(
    "--projection-dim",
    type=int,
    default=128,
    metavar="N",
    help=(
        "width of the linear obs embedding before the memory cell "
        "(default: 64; pass 0 to disable the projection and feed raw obs)"
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
    "--wandb",
    action="store_true",
    help="enable Weights & Biases logging",
)
parser.add_argument(
    "--wandb-project",
    default="lambda-imitation",
    metavar="PROJECT",
    help="W&B project name (default: lambda-imitation)",
)
parser.add_argument(
    "--wandb-run-name",
    default=None,
    metavar="NAME",
    help="W&B run name (default: auto-generated)",
)
args = parser.parse_args()

# ── imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp

try:
    import gymnax
except ImportError:
    sys.exit(
        "gymnax is required.  Install with:  pip install gymnax\n"
        "or:  pip install 'lambda-imitation[gymnax]'"
    )

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

# ── partial observability wrapper ─────────────────────────────────────────────


class _PartialObsEnv:
    """Wraps a gymnax CartPole-v1 env to expose only cart position and pole angle.

    Drops cart velocity (index 1) and pole angular velocity (index 3) from
    every observation, reducing obs_shape from (4,) to (2,).
    """

    _KEEP = jnp.array([0, 2])

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def _mask(self, obs):
        return obs[self._KEEP]

    def get_obs(self, state, params):
        return self._mask(self._wrapped.get_obs(state, params))

    def reset(self, key, params):
        obs, state = self._wrapped.reset(key, params)
        return self._mask(obs), state

    def step(self, key, state, action, params):
        obs, new_state, reward, done, info = self._wrapped.step(
            key, state, action, params
        )
        return self._mask(obs), new_state, reward, done, info

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# ── environment setup ─────────────────────────────────────────────────────────

env, env_params = gymnax.make("CartPole-v1")
spec = env_spec_from_gymnax(env, env_params)
# CartPole-v1: obs_shape=(4,), action_dim=2, is_discrete=True

if args.partial:
    # Wrap the env so every observation only contains cart position and pole
    # angle (indices 0 and 2).  Patch the spec to reflect the reduced shape.
    env = _PartialObsEnv(env)
    spec = spec._replace(obs_shape=(2,))

# ── placeholder expert buffer (never sampled — SAC only) ──────────────────────
#
# create_iqlearn_from_env requires a structurally valid expert buffer even when
# only fns.train() is used.  A single all-zeros transition with the minimum
# buffer_size=1 and batch_size=1 satisfies the structural constraint without
# allocating significant memory.

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── build agent ───────────────────────────────────────────────────────────────

hp = Hyperparameters(
    batch_size=1,  # expert buffer sampling size; never used
    alpha=0.05,
    online_batch_size=32,
    online_buffer_size=10_000,
    sequence_length=20,
    lambda1=0.05,
    lambda2=0.8,
    lambda_truncation=30,
    lambda_coef = args.lambda_coef,
)

projection_dim = args.projection_dim if args.projection_dim > 0 else None

print(
    f"Building SAC + λ-discrepancy agent for CartPole-v1 "
    f"(discrete, gymnax)  memory={args.memory_type} "
    f"hidden={args.memory_hidden_dim} projection={projection_dim}…"
)
state, fns, debug_fns = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,  # expert buffer capacity; minimum valid size
    hp=hp,
    projection_dim=projection_dim,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    critic_dims=(64,),
    lambda1_critic_dims=(64,),
    lambda2_critic_dims=(64,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    debug=True,
)

# ── carry helper ──────────────────────────────────────────────────────────────
#
# RecurrentFeatureExtractor uses a flat carry of width:
#   identity   -> 0
#   rnn / gru  -> memory_hidden_dim
#   lstm       -> 2 * memory_hidden_dim   (concatenated [c, h])

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim


def zero_carry() -> jax.Array:
    """Zero carry shaped ``(carry_dim,)`` for a single-observation predict()."""
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


# ── initial environment reset ─────────────────────────────────────────────────

key = jax.random.key(args.seed)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)

# ── evaluation helper ─────────────────────────────────────────────────────────


def evaluate(agent_state, rng_key, n_episodes: int = 10) -> float:
    """Run ``n_episodes`` deterministic episodes; return mean episode return."""
    total = 0.0
    for _ in range(n_episodes):
        rng_key, rk = jax.random.split(rng_key)
        ep_obs, ep_state = env.reset(rk, env_params)
        ep_carry = zero_carry()
        ep_return = 0.0
        done = False
        while not done:
            rng_key, sk = jax.random.split(rng_key)
            # predict returns (action_index_float32, new_carry); gymnax expects int32
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
        total += ep_return
    return total / n_episodes


# ── wandb (optional) ─────────────────────────────────────────────────────────

_wandb = None
if args.wandb:
    try:
        import wandb as _wandb_mod

        _wandb = _wandb_mod
    except ImportError:
        print("wandb not installed — skipping.  Install with:  pip install wandb")
    else:
        _wandb.init(
            entity="fhstp-data-intelligence-research-group",
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "env": "CartPole-v1",
                "algo": "SAC+lambda" if args.approximate_lambda else "SAC",
                "action_space": "discrete",
                "approximate_lambda": args.approximate_lambda,
                "memory_type": args.memory_type,
                "memory_hidden_dim": args.memory_hidden_dim,
                "projection_dim": projection_dim,
                "partial_obs": args.partial,
                "rounds": args.rounds,
                "train_steps": args.train_steps,
                "seed": args.seed,
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
            }
        )

print("\nTraining complete.")
key, eval_key = jax.random.split(key)
final_return = evaluate(state, eval_key, n_episodes=20)
print(f"Final evaluation (20 episodes): mean_return={final_return:.1f}")

if _wandb is not None:
    _wandb.log({"final_mean_return": final_return})
    _wandb.finish()

# ── visualise ─────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
except ImportError:
    print(
        "\ngymnasium not installed — skipping visualisation.\n"
        "Install with:  pip install 'gymnasium[classic-control]'"
    )
    sys.exit(0)

print("\nVisualising.  Press Enter in this terminal after each episode to continue.")
print("Ctrl-C to quit.\n")

vis_env = gym.make("CartPole-v1", render_mode="human")
obs, _ = vis_env.reset()
key = jax.random.key(args.seed + 1)
vis_carry = zero_carry()

while True:
    key, subkey = jax.random.split(key)
    # If partial, strip velocity dims from the gymnasium observation before
    # passing to the agent (which was trained on 2-dim observations).
    vis_obs = jnp.array(obs)
    if args.partial:
        vis_obs = vis_obs[jnp.array([0, 2])]
    # predict returns (action_float32, new_carry); gymnasium CartPole expects int
    raw, vis_carry = fns.predict(state, vis_obs, vis_carry, subkey, deterministic=True)
    obs, _, terminated, truncated, _ = vis_env.step(int(jnp.round(raw)))
    if terminated or truncated:
        obs, _ = vis_env.reset()
        vis_carry = zero_carry()
        input("Episode done — press Enter to continue…")
