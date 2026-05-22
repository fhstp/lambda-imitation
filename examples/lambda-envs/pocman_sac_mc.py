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
    )

# ── carry helper ──────────────────────────────────────────────────────────────

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim

projection_dim = args.projection_dim if args.projection_dim > 0 else None


def zero_carry() -> jax.Array:
    return jnp.zeros((CARRY_DIM,), dtype=jnp.float32)


# ── evaluation helper ─────────────────────────────────────────────────────────


def evaluate(fns, agent_state, rng_key, n_episodes: int = 10) -> float:
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


# ── single seed run ──────────────────────────────────────────────────────────


def run_seed(seed_val: int, seed_idx: int) -> float:
    print(
        f"\n{'=' * 60}\n"
        f"Seed {seed_idx + 1}/{args.num_seeds}  (seed={seed_val})\n"
        f"{'=' * 60}"
    )
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
        seed=seed_val,
    )

    key = jax.random.key(seed_val)
    key, reset_key = jax.random.split(key)
    _obs, env_state = env.reset(reset_key, env_params)

    total_steps = args.rounds * args.train_steps
    print(
        f"Training for {args.rounds} rounds × {args.train_steps} steps "
        f"= {total_steps} total env steps."
    )

    for rnd in tqdm(range(1, args.rounds + 1), desc=f"Seed {seed_idx + 1}"):
        key, train_key = jax.random.split(key)
        state, env_state, metrics = fns.train(
            state, env, env_params, env_state, train_key
        )

        key, eval_key = jax.random.split(key)
        mean_return = evaluate(fns, state, eval_key, n_episodes=10)

        print(
            f"Round {rnd:4d}/{args.rounds}  "
            f"mean_return={mean_return:7.1f}  "
            f"alpha={float(metrics.get('alpha', state.alpha)):.4f}  "
            f"critic_loss={float(metrics.get('critic_loss', jnp.nan)):.4f}  "
            f"entropy={float(metrics.get('entropy', jnp.nan)):.4f}  "
            f"q={float(metrics.get('q', jnp.nan)):7.3f}"
        )

        if _wandb is not None:
            if args.num_seeds == 1:
                _wandb.log(
                    {
                        "round": rnd,
                        "env_interactions": rnd * args.train_steps,
                        "mean_return": mean_return,
                        **{k: float(v) for k, v in metrics.items()},
                    },
                    step=rnd * args.train_steps,
                )
            else:
                prefix = f"seed_{seed_idx}"
                _wandb.log(
                    {
                        "env_interactions": rnd * args.train_steps,
                        f"{prefix}/mean_return": mean_return,
                        **{f"{prefix}/{k}": float(v) for k, v in metrics.items()},
                    }
                )

    key, eval_key = jax.random.split(key)
    final_return = evaluate(fns, state, eval_key, n_episodes=20)
    print(
        f"Seed {seed_idx + 1} final evaluation (20 episodes): "
        f"mean_return={final_return:.1f}"
    )
    return final_return


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
            "num_seeds": args.num_seeds,
            "seed": args.seed,
            **hp._asdict(),
        },
    )

# ── wandb metric setup ───────────────────────────────────────────────────────

if _wandb is not None and args.num_seeds > 1:
    _wandb.define_metric("env_interactions")
    for i in range(args.num_seeds):
        _wandb.define_metric(f"seed_{i}/*", step_metric="env_interactions")

# ── main: run seeds and aggregate ────────────────────────────────────────────

seeds = [args.seed + i for i in range(args.num_seeds)]
final_returns = []

for i, s in enumerate(seeds):
    ret = run_seed(s, i)
    final_returns.append(ret)

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
