"""SAC online training demo: RockSample (discrete action space, lambda-envs).

Pure online reinforcement learning with SAC + Monte Carlo critic — no expert
data or imitation learning is used.  The placeholder expert buffer required by
``create_iqlearn_from_env`` is never sampled (``fns.train()`` is never called).

RockSample is a POMDP exploration-exploitation task: the agent navigates a grid,
sensing nearby rocks with noisy distance-dependent observations, and must decide
whether to sample each rock (good/bad) or exit the map.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install lambda-envs

Usage
-----
    python rocksample_sac_mc.py                  # 7×8 variant (default)
    python rocksample_sac_mc.py --size 5         # smallest 5×5 variant
    python rocksample_sac_mc.py --size 11        # large 11×11 variant
    python rocksample_sac_mc.py --size 15        # largest 15×15 variant
    python rocksample_sac_mc.py --rounds 200     # more training
    python rocksample_sac_mc.py --wandb          # enable W&B logging
    python rocksample_sac_mc.py --help           # full option list
"""

import argparse
import sys
from pathlib import Path

from tqdm.rich import tqdm

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    prog="rocksample_sac_mc.py",
    description="SAC + MC critic on RockSample (discrete, lambda-envs).",
)
parser.add_argument(
    "--size",
    type=int,
    default=7,
    choices=[5, 7, 11, 15],
    metavar="{5,7,11,15}",
    help="grid size variant: 5 (5×5/5 rocks), 7 (7×8/8 rocks), 11 (11×11), 15 (15×15) (default: 7)",
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
    "--seed",
    type=int,
    default=0,
    metavar="S",
    help="JAX random seed (default: 0)",
)
parser.add_argument(
    "--lambda-coef",
    type=float,
    default=1.0,
    metavar="LC",
    help="lambda discrepancy coefficient (default: 1.0)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="TAU",
    help="smoothing coefficient for target network (default: 0.005)",
)
parser.add_argument(
    "--burn-in-len",
    type=int,
    default=20,
    help="burn-in length for recurrent policy (default: 20)",
)
parser.add_argument(
    "--burn-in-from-stored-carry",
    action="store",
    default=True,
    help="set stored carry for burn-in (default: True)",
)
parser.add_argument(
    "--refresh-stored-carries",
    action="store",
    default=True,
    help="refresh stored carries each round (default: True)",
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
    from lambda_envs.envs.rocksample import RockSample
except ImportError:
    sys.exit(
        "lambda-envs is required.  Install with:  pip install lambda-envs"
    )

from lambda_imitation.iqlearn import Hyperparameters, LSTMMemory
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


# ── env adapter: add params kwarg to get_obs ──────────────────────────────────
#
# lambda-imitation calls ``env.get_obs(state, params)`` (gymnax convention),
# but RockSample.get_obs() only accepts ``(state)``.  This wrapper drops the
# extra arg.  No env behaviour changes: RockSample's get_obs never reads
# params, and reset/step pass through unchanged via __getattr__.
# Trajectories remain bit-identical to lambda_discrepancy.

class _LambdaEnvAdapter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_obs(self, state, params=None):
        return self._wrapped.get_obs(state)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


# ── environment setup ─────────────────────────────────────────────────────────

# Map --size to the correct config filename.
# Size 7 uses the 7×8 (7 rows, 8 rocks) config; all others use square grids.
_CONFIG_NAMES = {
    5:  "rocksample_5_5_config.json",
    7:  "rocksample_7_8_config.json",
    11: "rocksample_11_11_config.json",
    15: "rocksample_15_15_config.json",
}
_configs_dir = Path(__file__).parent.parent.parent.parent / "lambda-envs" / "src" / "lambda_envs" / "configs"
config_path = _configs_dir / _CONFIG_NAMES[args.size]

if not config_path.exists():
    # Fallback: try to locate configs via the installed package
    import importlib.resources as _ir
    try:
        import lambda_envs
        _pkg_root = Path(lambda_envs.__file__).parent
        config_path = _pkg_root / "configs" / _CONFIG_NAMES[args.size]
    except Exception:
        pass

if not config_path.exists():
    sys.exit(
        f"Cannot find RockSample config at {config_path}.\n"
        "Make sure lambda-envs is installed or run from the repo root."
    )

init_key = jax.random.key(args.seed)
env = _LambdaEnvAdapter(RockSample(init_key, config_path=config_path))
env_params = env.default_params
spec = env_spec_from_gymnax(env, env_params)
# obs_shape=(2*size + k,), action_dim=k+5, is_discrete=True

env_label = f"rocksample_{_CONFIG_NAMES[args.size].replace('_config.json', '')}"

# ── placeholder expert buffer (never sampled — SAC only) ──────────────────────

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

# ── build agent ───────────────────────────────────────────────────────────────

hp = Hyperparameters(
    online_batch_size=32,
    online_buffer_size=10_000,
    target_entropy=0.3,
    actor_lr=1e-4,
    critic_lr=1e-4,
    mc_critic_lr=1e-4,
    alpha_lr=1e-4,
    alpha=1.0,
    autotune_alpha=True,
    batch_size=256,
    gamma=0.99,
    regularizer_coef=1 / 40,
    tau=args.tau,
    lam=0.5,
    lambda_truncation=15,
    sequence_length=5,
    burn_in_length=args.burn_in_len,
    n_step=1,
    burn_in_from_stored_carry=args.burn_in_from_stored_carry,
    value_rescaling=False,
    value_rescaling_eps=1e-3,
    lambda_discrepancy_coef=args.lambda_coef,
    lambda_discrepancy_delta=1.0,
    refresh_stored_carries=args.refresh_stored_carries,
)

# ── initial environment reset ─────────────────────────────────────────────────

key = jax.random.key(args.seed)
key, reset_key = jax.random.split(key)
obs, env_state = env.reset(reset_key, env_params)

key, key_env = jax.random.split(key)
print(f"Building SAC agent for {env_label} (obs_shape={spec.obs_shape}, action_dim={spec.action_dim})…")
state, fns, _, debug_fns = create_iqlearn_from_env(
    spec,
    expert_data,
    key_env,
    buffer_size=1,
    hp=hp,
    fe_hidden_dims=(64, 64),
    actor_head_dims=(64, 64),
    critic_head_dims=(64, 64),
    train_steps=args.train_steps,
    approximate_mc=True,
    debug=True,
    memory_factory=lambda f, r: LSTMMemory(f, 256, rngs=r),
)

# ── evaluation helper ─────────────────────────────────────────────────────────


def evaluate(agent_state, rng_key, n_episodes: int = 10) -> float:
    total = 0.0
    max_steps = int(env_params.max_steps_in_episode)
    for _ in range(n_episodes):
        rng_key, rk = jax.random.split(rng_key)
        ep_obs, ep_state = env.reset(rk, env_params)
        carry = jnp.zeros_like(agent_state.actor_online_carry)
        ep_return = 0.0
        done = False
        step = 0
        while not done and step < max_steps:
            rng_key, sk = jax.random.split(rng_key)
            raw, carry = fns.predict(agent_state, ep_obs, carry, sk, deterministic=True)
            action = jnp.round(raw).astype(jnp.int32)
            rng_key, ek = jax.random.split(rng_key)
            ep_obs, ep_state, reward, done, _ = env.step(ek, ep_state, action, env_params)
            ep_return += float(reward)
            done = bool(done)
            step += 1
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
                "env": env_label,
                "algo": "SAC",
                "action_space": "discrete",
                "approximate_mc": True,
                "size": args.size,
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

    key, cmp_key, entropy_key = jax.random.split(key, 3)
    buf_size = int(state.online_buffer.size)
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
