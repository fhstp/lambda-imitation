"""SAC online training demo: CartPole-v1 (discrete action space, gymnax).

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
    python cartpole_sac.py                        # default 100 rounds × 50 steps
    python cartpole_sac.py --rounds 200           # more training
    python cartpole_sac.py --train-steps 100      # longer rounds
    python cartpole_sac.py --seed 42              # different random seed
    python cartpole_sac.py --help                 # full option list
"""

import argparse
import math
import sys

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="cartpole_sac.py",
    description="SAC online training on CartPole-v1 (discrete, gymnax).",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=10,
    metavar="N",
    help="number of training rounds (default: 10)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=500,
    metavar="N",
    help="env steps and gradient updates per round (default: 500)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="JAX random seed (default: 0)",
)
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

# ── environment setup ─────────────────────────────────────────────────────────

env, env_params = gymnax.make("CartPole-v1")
spec = env_spec_from_gymnax(env, env_params)
# CartPole-v1: obs_shape=(4,), action_dim=2, is_discrete=True

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
    online_batch_size=256,
    online_buffer_size=10_000,
    # Christodoulou 2019 discrete target entropy: 0.98 * log(num_actions)
    target_entropy=float(0.98 * math.log(spec.action_dim)),
)

print("Building SAC agent for CartPole-v1 (discrete, gymnax)…")
state, fns, _ = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,  # expert buffer capacity; minimum valid size
    hp=hp,
    fe_hidden_dims=(64, 64),
    critic_head_dims=(64,),
    train_steps=args.train_steps,
    seed=args.seed,
)

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
        ep_return = 0.0
        done = False
        while not done:
            rng_key, sk = jax.random.split(rng_key)
            # predict returns a float32 scalar action index; gymnax expects int32
            raw = fns.predict(agent_state, ep_obs, sk, deterministic=True)
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
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "env": "CartPole-v1",
                "algo": "SAC",
                "action_space": "discrete",
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

for rnd in range(1, args.rounds + 1):
    key, train_key = jax.random.split(key)
    state, env_state, metrics = fns.train_sac(
        state, env, env_params, env_state, train_key
    )

    key, eval_key = jax.random.split(key)
    mean_return = evaluate(state, eval_key, n_episodes=10)
    print(
        f"Round {rnd:4d}/{args.rounds}  "
        f"mean_return={mean_return:7.1f}  "
        f"alpha={float(metrics['alpha']):.4f}  "
        f"critic_loss={float(metrics['critic_loss']):.4f}"
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

while True:
    key, subkey = jax.random.split(key)
    # predict returns a float32 scalar; gymnasium CartPole expects a plain int
    raw = fns.predict(state, jnp.array(obs), subkey, deterministic=True)
    obs, _, terminated, truncated, _ = vis_env.step(int(jnp.round(raw)))
    if terminated or truncated:
        obs, _ = vis_env.reset()
        input("Episode done — press Enter to continue…")
