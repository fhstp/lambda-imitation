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

from tqdm.rich import tqdm

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
    default=1000,
    metavar="N",
    help="env steps and gradient updates per round (default: 1000)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="JAX random seed (default: 0)",
)
parser.add_argument(
    "--lambda-coef",
    type=float,
    default=0.6,
    metavar="LC",
    help="lambda discrepancy coefficient (default: 0.6)",
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

from lambda_imitation.iqlearn import Hyperparameters, LSTMMemory
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
# only fns.train_sac() is used.  A single all-zeros transition with the minimum
# buffer_size=1 and batch_size=1 satisfies the structural constraint without
# allocating significant memory.

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── build agent ───────────────────────────────────────────────────────────────

hp = Hyperparameters(
    online_batch_size=32,
    online_buffer_size=10_000,
    target_entropy=0.3,  # float(0.98 * math.log(spec.action_dim)),
    actor_lr=1e-4,
    critic_lr=1e-4,
    mc_critic_lr=1e-3,
    alpha_lr=1e-4,
    alpha=1.0,
    autotune_alpha=True,
    batch_size=256,
    gamma=0.99,
    regularizer_coef=1 / 40,
    tau=0.005,
    lam=0.5,
    lambda_truncation=15,
    sequence_length=5,
    burn_in_length=20,
    n_step=1,
    burn_in_from_stored_carry=True,
    value_rescaling=False,
    value_rescaling_eps=1e-3,
    lambda_discrepancy_coef=args.lambda_coef,
    lambda_discrepancy_delta=1.0,
    refresh_stored_carries=True,
)

print("Building SAC agent for CartPole-v1 (discrete, gymnax)…")
state, fns, _, debug_fns = create_iqlearn_from_env(
    spec,
    expert_data,
    buffer_size=1,  # expert buffer capacity; minimum valid size
    hp=hp,
    fe_hidden_dims=(64, 64),
    critic_head_dims=(64,),
    train_steps=args.train_steps,
    approximate_mc=True,
    debug=True,
    memory_factory=lambda f, r: LSTMMemory(f, 64, rngs=r),
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
        carry = jnp.zeros_like(agent_state.actor_online_carry)
        ep_return = 0.0
        done = False
        while not done:
            rng_key, sk = jax.random.split(rng_key)
            # predict returns a float32 scalar action index; gymnax expects int32
            raw, carry = fns.predict(agent_state, ep_obs, carry, sk, deterministic=True)
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
                "algo": "SAC",
                "action_space": "discrete",
                "approximate_mc": True,
                "partial_obs": args.partial,
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
    state, env_state, metrics = fns.train_sac(
        state, env, env_params, env_state, train_key
    )

    key, eval_key = jax.random.split(key)
    mean_return = evaluate(state, eval_key, n_episodes=10)

    # Q comparison: SAC critic_target vs MC critic_target on a random batch
    key, cmp_key, entropy_key = jax.random.split(key, 3)
    buf_size = int(state.online_buffer.size)
    # Only sample from slots that are marked sampling_ok (populated and have a
    # valid successor).
    sampling_ok_f = state.online_buffer.sampling_ok.astype(jnp.float32)
    sampling_ok_probs = sampling_ok_f / sampling_ok_f.sum()
    cmp_idx = jax.random.choice(
        cmp_key, buf_size, (hp.online_batch_size,), replace=False, p=sampling_ok_probs
    )
    cmp_obs = state.online_buffer.info["observations"][cmp_idx]
    cmp_actions = state.online_buffer.info["actions"][cmp_idx]
    entropy = debug_fns.get_entropy(state.actor, cmp_obs, entropy_key)
    q_sac = (
        debug_fns.get_q(state.critic_target, cmp_obs, cmp_actions, False)
        + state.alpha * entropy
    )
    q_mc = debug_fns.get_q(state.mc_critic_target, cmp_obs, cmp_actions, True)

    disc_loss = float(metrics.get("lambda_discrepancy_loss", 0.0))
    print(
        f"Round {rnd:4d}/{args.rounds}  "
        f"mean_return={mean_return:7.1f}  "
        f"alpha={float(metrics['alpha']):.4f}  "
        f"mc_critic_loss={float(metrics['mc_critic_loss']):.4f}  "
        f"disc_loss={disc_loss:.4f}  "
        f"q_sac={q_sac.mean():7.3f}  q_mc={q_mc.mean():7.3f}  |Δq|={jnp.abs(q_sac - q_mc).mean():.4f}"
    )

    if _wandb is not None:
        _wandb.log(
            {
                "round": rnd,
                "step": rnd * args.train_steps,
                "mean_return": mean_return,
                "q_sac": q_sac.mean(),
                "q_mc": q_mc.mean(),
                "q_delta_abs": jnp.abs(q_sac - q_mc).mean(),
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
vis_carry = jnp.zeros_like(state.actor_online_carry)

while True:
    key, subkey = jax.random.split(key)
    # If partial, strip velocity dims from the gymnasium observation before
    # passing to the agent (which was trained on 2-dim observations).
    vis_obs = jnp.array(obs)
    if args.partial:
        vis_obs = vis_obs[jnp.array([0, 2])]
    # predict returns a float32 scalar; gymnasium CartPole expects a plain int
    raw, vis_carry = fns.predict(state, vis_obs, vis_carry, subkey, deterministic=True)
    obs, _, terminated, truncated, _ = vis_env.step(int(jnp.round(raw)))
    if terminated or truncated:
        obs, _ = vis_env.reset()
        vis_carry = jnp.zeros_like(state.actor_online_carry)
        input("Episode done — press Enter to continue…")
