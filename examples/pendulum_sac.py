"""SAC online training demo: Pendulum-v1 (continuous action space, gymnax).

Pure online reinforcement learning with SAC — no expert data or imitation
learning is used.  The placeholder expert buffer required by
``create_iqlearn_from_env`` is never sampled (``fns.train()`` is never called).

After training the script opens a gymnasium render window so you can watch the
trained agent.  Press Enter in the terminal after each episode; Ctrl-C to quit.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install "gymnasium[classic-control]"   # for visualisation

Usage
-----
    python pendulum_sac.py                        # default 200 rounds × 50 steps
    python pendulum_sac.py --rounds 400           # more training
    python pendulum_sac.py --train-steps 100      # longer rounds
    python pendulum_sac.py --seed 42              # different random seed
    python pendulum_sac.py --help                 # full option list
"""

import argparse
import sys

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="pendulum_sac.py",
    description="SAC online training on Pendulum-v1 (continuous, gymnax).",
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
    default=500,
    metavar="N",
    help="env steps and gradient updates per round (default: 50)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="JAX random seed (default: 0)",
)
args = parser.parse_args()

# ── imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gymnax
except ImportError:
    sys.exit(
        "gymnax is required.  Install with:  pip install gymnax\n"
        "or:  pip install 'lambda-imitation[gymnax]'"
    )

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

# ── environment setup ─────────────────────────────────────────────────────────

env, env_params = gymnax.make("Pendulum-v1")
spec = env_spec_from_gymnax(env, env_params)
# Pendulum-v1: obs_shape=(3,), action_dim=1, is_discrete=False
# action bounds: low=-2.0, high=2.0  →  action_scale=2.0, action_bias=0.0

# ── placeholder expert buffer (never sampled — SAC only) ──────────────────────
#
# create_iqlearn_from_env requires a structurally valid expert buffer even when
# only fns.train_sac() is used.  A single all-zeros transition with the minimum
# buffer_size=1 and batch_size=1 satisfies the structural constraint without
# allocating significant memory.

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── build agent ───────────────────────────────────────────────────────────────

hp = Hyperparameters(
    batch_size=1,  # expert buffer sampling size; never used
    online_batch_size=64,
    online_buffer_size=10_000,
    target_entropy=-1.0,  # -action_dim (standard SAC heuristic)
)

print("Building SAC agent for Pendulum-v1 (continuous, gymnax)…")
state, fns, _ = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,  # expert buffer capacity; minimum valid size
    hp=hp,
    fe_hidden_dims=(128, 128),
    critic_head_dims=(64,),
    train_steps=args.train_steps,
)

# ── initial environment reset ─────────────────────────────────────────────────

key = jax.random.key(args.seed)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)

# ── evaluation helper ─────────────────────────────────────────────────────────


def evaluate(agent_state, rng_key, n_episodes: int = 5) -> float:
    """Run ``n_episodes`` deterministic episodes; return mean episode return."""
    total = 0.0
    for _ in range(n_episodes):
        rng_key, rk = jax.random.split(rng_key)
        ep_obs, ep_state = env.reset(rk, env_params)
        ep_return = 0.0
        done = False
        while not done:
            rng_key, sk = jax.random.split(rng_key)
            # predict returns float32 shape (1,); gymnax Pendulum expects shape (1,)
            action = fns.predict(agent_state, ep_obs, sk, deterministic=True)
            rng_key, ek = jax.random.split(rng_key)
            ep_obs, ep_state, reward, done, _ = env.step(
                ek, ep_state, action, env_params
            )
            ep_return += float(reward)
            done = bool(done)
        total += ep_return
    return total / n_episodes


# ── training loop ─────────────────────────────────────────────────────────────

total_steps = args.rounds * args.train_steps
print(
    f"Training for {args.rounds} rounds × {args.train_steps} steps "
    f"= {total_steps} total env steps."
)
print("(First round JIT-compiles the update loop — expect a short delay.)\n")

for rnd in range(1, args.rounds + 1):
    key, train_key = jax.random.split(key)
    state, env_state, metrics = fns.train_sac(
        state, env, env_params, env_state, train_key
    )

    key, eval_key = jax.random.split(key)
    mean_return = evaluate(state, eval_key, n_episodes=5)
    print(
        f"Round {rnd:4d}/{args.rounds}  "
        f"mean_return={mean_return:8.1f}  "
        f"alpha={float(metrics['alpha']):.4f}  "
        f"critic_loss={float(metrics['critic_loss']):.4f}"
    )

print("\nTraining complete.")
key, eval_key = jax.random.split(key)
final_return = evaluate(state, eval_key, n_episodes=10)
print(f"Final evaluation (10 episodes): mean_return={final_return:.1f}")

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

vis_env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = vis_env.reset()
key = jax.random.key(args.seed + 1)

while True:
    key, subkey = jax.random.split(key)
    # predict returns float32 shape (1,); env.step expects a numpy array
    action = fns.predict(state, jnp.array(obs), subkey, deterministic=False)
    obs, _, terminated, truncated, _ = vis_env.step(np.array(action))
    if terminated or truncated:
        obs, _ = vis_env.reset()
        input("Episode done — press Enter to continue…")
