"""Benchmark: does vmap-over-seeds speed up pocman training?

Measures per-round wall-time for training N seeds **concurrently** in one
vmapped+jitted call, for several N.  The decisive number is the per-round time
*per seed* (``round_s / N``): if it stays roughly flat as N grows, the GPU is
latency / occupancy bound and vmapping seeds is a near-free speedup; if it
grows ~linearly, the workload is genuinely compute bound and vmap won't help.

This does NOT train to convergence — it times a short scan to compare scaling.
Buffer size is kept small (does not affect the scaling ratio) so prefill is
quick.

Usage
-----
    python bench_vmap_seeds.py                       # 1,4,16 seeds, bf16+fp32 carry
    python bench_vmap_seeds.py --seeds-list 1,2,4,8,16,32
    python bench_vmap_seeds.py --train-steps 2000 --rounds 3
    python bench_vmap_seeds.py --dtype f32           # baseline precision
"""

import argparse
import sys
import time

parser = argparse.ArgumentParser(description="vmap-over-seeds scaling benchmark (pocman).")
parser.add_argument("--seeds-list", default="1,4,16",
                    help="comma-separated seed counts to benchmark (default: 1,4,16)")
parser.add_argument("--train-steps", type=int, default=1000,
                    help="env-step+grad-update iterations per timed round (default: 1000)")
parser.add_argument("--rounds", type=int, default=3,
                    help="timed rounds to average over, after one warmup/compile (default: 3)")
parser.add_argument("--online-batch-size", type=int, default=128)
parser.add_argument("--online-buffer-size", type=int, default=50_000,
                    help="kept small for fast prefill; does not affect the scaling ratio")
parser.add_argument("--memory-hidden-dim", type=int, default=750)
parser.add_argument("--projection-dim", type=int, default=128)
parser.add_argument("--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru")
parser.add_argument("--sequence-length", type=int, default=20)
parser.add_argument("--burn-in-length", type=int, default=32)
parser.add_argument("--dtype", default="bf16")
parser.add_argument("--param-dtype", default=None)
parser.add_argument("--carry-dtype", default="f32")
parser.add_argument("--no-approximate-lambda", dest="approximate_lambda",
                    action="store_false", help="disable λ-critic branches")
parser.set_defaults(approximate_lambda=True)
args = parser.parse_args()

import jax
import jax.numpy as jnp

try:
    from lambda_envs.envs.pocman import PocMan
except ImportError:
    sys.exit("lambda-envs[pocman] required: pip install 'lambda-envs[pocman]'")

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import (
    create_iqlearn_from_env,
    env_spec_from_gymnax,
    resolve_dtype,
)

DTYPE = resolve_dtype(args.dtype)
PARAM_DTYPE = resolve_dtype(args.param_dtype)
CARRY_DTYPE = resolve_dtype(args.carry_dtype) or DTYPE
print(f"device={jax.devices()[0]}  compute={jnp.dtype(DTYPE).name} "
      f"params={jnp.dtype(PARAM_DTYPE if PARAM_DTYPE is not None else DTYPE).name} "
      f"carry={jnp.dtype(CARRY_DTYPE).name}  approx_lambda={args.approximate_lambda}")


class _LambdaEnvAdapter:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def get_obs(self, state, params=None):
        return self._wrapped.get_obs(state)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


env = _LambdaEnvAdapter(PocMan())
env_params = env.default_params
spec = env_spec_from_gymnax(env, env_params)

expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

hp = Hyperparameters(
    online_batch_size=args.online_batch_size,
    online_buffer_size=args.online_buffer_size,
    target_entropy=0.0,
    alpha=0.1,
    gamma=0.95,
    lambda1=0.05,
    lambda2=0.8,
    lambda_truncation=17,
    sequence_length=args.sequence_length,
    burn_in_length=args.burn_in_length,
)

AGENT_KWARGS = dict(
    buffer_size=1,
    hp=hp,
    projection_dim=args.projection_dim if args.projection_dim > 0 else None,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    critic_dims=(128,),
    lambda1_critic_dims=(128,),
    lambda2_critic_dims=(128,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    debug=True,
    dtype=DTYPE,
    param_dtype=PARAM_DTYPE,
    carry_dtype=CARRY_DTYPE,
)

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim

# Prefill needs at least one full sampleable window per slot.
PREFILL_STEPS = args.online_batch_size * (
    args.burn_in_length + args.sequence_length + hp.lambda_truncation
)


def build_one(seed):
    """Build a single agent; returns (state, fns)."""
    state, fns, _ = create_iqlearn_from_env(spec, expert_data, seed=seed, **AGENT_KWARGS)
    return state, fns


def stack_states(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def bench(n_seeds):
    seeds = list(range(n_seeds))
    builds = [build_one(s) for s in seeds]
    batched_state = stack_states([b[0] for b in builds])
    fns = builds[0][1]  # graphdefs identical across seeds

    key = jax.random.key(0)
    reset_keys = jax.random.split(jax.random.split(key)[0], n_seeds)
    _, env_states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)

    # Prefill all seeds (vmapped); env / env_params / n_steps broadcast.
    pf_keys = jax.random.split(jax.random.split(key)[1], n_seeds)
    prefill_v = jax.jit(jax.vmap(
        lambda s, es, k: fns.prefill_buffer(s, env, env_params, es, PREFILL_STEPS, k),
        in_axes=(0, 0, 0),
    ))
    batched_state, env_states = prefill_v(batched_state, env_states, pf_keys)
    jax.block_until_ready(batched_state)

    env_carry = jnp.zeros((n_seeds, CARRY_DIM), dtype=CARRY_DTYPE)

    train_v = jax.jit(jax.vmap(
        lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
        in_axes=(0, 0, 0, 0),
    ))

    # Warmup / compile (not timed).
    t0 = time.perf_counter()
    rk = jax.random.split(key, n_seeds)
    out = train_v(batched_state, env_states, env_carry, rk)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    state_b, env_states, env_carry, _ = out

    # Timed rounds.
    t0 = time.perf_counter()
    for r in range(args.rounds):
        rk = jax.random.split(jax.random.key(1000 + r), n_seeds)
        out = train_v(state_b, env_states, env_carry, rk)
        state_b, env_states, env_carry, _ = out
    jax.block_until_ready(out)
    round_s = (time.perf_counter() - t0) / args.rounds
    return compile_s, round_s


print(f"\ntrain_steps={args.train_steps}  prefill_steps={PREFILL_STEPS}  "
      f"rounds_averaged={args.rounds}\n")
print(f"{'seeds':>6} {'compile_s':>10} {'round_s':>9} {'round_s/seed':>13} {'speedup_vs_1':>13}")
base = None
for n in [int(x) for x in args.seeds_list.split(",")]:
    c, r = bench(n)
    per_seed = r / n
    if base is None:
        base = per_seed
    print(f"{n:>6} {c:>10.2f} {r:>9.3f} {per_seed:>13.4f} {base / per_seed:>12.2f}x")

print("\nflat round_s/seed -> latency-bound -> vmap wins.  linear growth -> compute-bound.")
