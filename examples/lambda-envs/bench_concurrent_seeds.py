"""Tune ``--concurrent-seeds`` for ``pocman_sac_mc.py`` on a new GPU.

Trains N PocMan seeds **concurrently** in one vmapped+jitted kernel for several
N and times the per-round wall-clock.  The decisive number is the per-round
time *per seed* (``round_s / N``):

    * stays roughly flat as N grows  -> GPU is latency / occupancy bound,
      vmapping seeds is a near-free speedup; push N until memory runs out.
    * grows ~linearly with N         -> workload is genuinely compute bound,
      vmap buys little; keep N small.

The script reports throughput (seeds/s) and prints a suggested
``--concurrent-seeds`` (the N with the highest measured throughput that fit in
memory).  It stops early on the first out-of-memory N.

This does NOT train to convergence — it times a short scan to compare scaling.
The hyperparameters mirror ``pocman_sac_mc.py`` defaults so the numbers transfer
directly to the sweep; override them to match a specific sweep config.

Requirements
------------
    pip install "lambda-imitation[gymnax]"
    pip install "lambda-envs[pocman]"

Usage
-----
    python bench_concurrent_seeds.py                          # 1,2,4,8,16
    python bench_concurrent_seeds.py --concurrent-list 1,4,16,32,64
    python bench_concurrent_seeds.py --train-steps 10000 --rounds 3
    python bench_concurrent_seeds.py --memory-hidden-dim 256  # smaller net
"""

import argparse
import sys
import time

parser = argparse.ArgumentParser(
    prog="bench_concurrent_seeds.py",
    description="vmap-over-seeds scaling benchmark for PocMan (tune --concurrent-seeds).",
)
parser.add_argument(
    "--concurrent-list",
    default="1,2,4,8,16",
    metavar="N,N,…",
    help="comma-separated concurrent-seed counts to benchmark (default: 1,2,4,8,16)",
)
parser.add_argument(
    "--train-steps",
    type=int,
    default=10000,
    metavar="N",
    help="env-step + grad-update iterations per timed round (default: 10000, "
    "matching pocman_sac_mc.py). Lower it for a quicker probe.",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=3,
    metavar="N",
    help="timed rounds to average over, after one warmup/compile (default: 3)",
)
# ── hyperparameters mirrored from pocman_sac_mc.py (override to match a sweep) ──
parser.add_argument("--online-batch-size", type=int, default=128)
parser.add_argument("--online-buffer-size", type=int, default=200_000)
parser.add_argument("--memory-hidden-dim", type=int, default=750)
parser.add_argument("--projection-dim", type=int, default=128)
parser.add_argument(
    "--memory-type", choices=("identity", "rnn", "gru", "lstm"), default="gru"
)
parser.add_argument("--sequence-length", type=int, default=20)
parser.add_argument("--burn-in-length", type=int, default=32)
parser.add_argument("--lambda-truncation", type=int, default=17)
parser.add_argument(
    "--no-approximate-lambda",
    dest="approximate_lambda",
    action="store_false",
    help="disable the λ-critic branches (default: enabled, matching the sweep)",
)
parser.set_defaults(approximate_lambda=True)
args = parser.parse_args()

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

print(f"device={jax.devices()[0]}  approx_lambda={args.approximate_lambda}")


# ── env adapter (PocMan.get_obs takes no params; see pocman_sac_mc.py) ─────────
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

# placeholder expert buffer (never sampled — online SAC only)
expert_data = {
    "observations": jnp.zeros((1, *spec.obs_shape), dtype=jnp.float32),
    "actions": jnp.zeros((1, 1), dtype=jnp.float32),
}

hp = Hyperparameters(
    online_batch_size=args.online_batch_size,
    online_buffer_size=args.online_buffer_size,
    gamma=0.95,
    lambda1=0.05,
    lambda2=0.8,
    lambda_truncation=args.lambda_truncation,
    sequence_length=args.sequence_length,
    burn_in_length=args.burn_in_length,
)

projection_dim = args.projection_dim if args.projection_dim > 0 else None

AGENT_KWARGS = dict(
    buffer_size=1,
    hp=hp,
    projection=projection_dim,
    memory_type=args.memory_type,
    memory_hidden_dim=args.memory_hidden_dim,
    critic_dims=(128,),
    lambda1_critic_dims=(128,),
    lambda2_critic_dims=(128,),
    train_steps=args.train_steps,
    approximate_lambda=args.approximate_lambda,
    debug=True,
)

if args.memory_type == "identity":
    CARRY_DIM = 0
elif args.memory_type == "lstm":
    CARRY_DIM = 2 * args.memory_hidden_dim
else:
    CARRY_DIM = args.memory_hidden_dim

# transitions collected before training, matching pocman_sac_mc.py / fns.train
PREFILL_STEPS = hp.online_batch_size * (
    hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
)


def _build(seed):
    state, fns, _ = create_iqlearn_from_env(
        spec, expert_data, seed=seed, **AGENT_KWARGS
    )
    return state, fns


def _stack(states):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def bench(n):
    """Compile + time ``args.rounds`` vmapped train rounds for n concurrent seeds.

    Returns ``(compile_s, round_s)``.  Raises on OOM (propagated to caller).
    """
    builds = [_build(s) for s in range(n)]
    batched = _stack([b[0] for b in builds])
    fns = builds[0][1]  # graphdefs identical across seeds

    key = jax.random.key(0)
    reset_keys = jax.random.split(jax.random.split(key)[0], n)
    _, env_states = jax.vmap(lambda k: env.reset(k, env_params))(reset_keys)

    # Prefill all seeds (vmapped); env / env_params / n_steps broadcast.
    pf_keys = jax.random.split(jax.random.split(key)[1], n)
    prefill_v = jax.jit(
        jax.vmap(
            lambda s, es, k: fns.prefill_buffer(s, env, env_params, es, PREFILL_STEPS, k),
            in_axes=(0, 0, 0),
        )
    )
    batched, env_states = prefill_v(batched, env_states, pf_keys)
    jax.block_until_ready(batched)

    env_carry = jnp.zeros((n, CARRY_DIM), dtype=jnp.float32)
    train_v = jax.jit(
        jax.vmap(
            lambda s, es, ec, k: fns.train_unrolled(s, env, env_params, es, ec, k),
            in_axes=(0, 0, 0, 0),
        )
    )

    # Warmup / compile (not timed).
    t0 = time.perf_counter()
    rk = jax.random.split(key, n)
    out = train_v(batched, env_states, env_carry, rk)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0
    state_b, env_states, env_carry, _ = out

    # Timed rounds.
    t0 = time.perf_counter()
    for r in range(args.rounds):
        rk = jax.random.split(jax.random.key(1000 + r), n)
        out = train_v(state_b, env_states, env_carry, rk)
        state_b, env_states, env_carry, _ = out
    jax.block_until_ready(out)
    round_s = (time.perf_counter() - t0) / args.rounds
    return compile_s, round_s


concurrent = [int(x) for x in args.concurrent_list.split(",")]

print(
    f"\ntrain_steps={args.train_steps}  prefill_steps={PREFILL_STEPS}  "
    f"rounds_averaged={args.rounds}  memory={args.memory_type}"
    f"(hidden={args.memory_hidden_dim}, proj={projection_dim})\n"
)
header = f"{'concurrent':>10} {'compile_s':>10} {'round_s':>9} {'round_s/seed':>13} {'seeds/s':>9} {'speedup':>8}"
print(header)
print("-" * len(header))

base_per_seed = None
best = None  # (throughput, n, round_s)
for n in concurrent:
    try:
        c, r = bench(n)
    except Exception as e:  # noqa: BLE001 — OOM / resource exhaustion ends the sweep
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"{n:>10} {'OOM / failed':>34}  ({msg[:40]})")
        break
    per_seed = r / n
    throughput = n / r
    if base_per_seed is None:
        base_per_seed = per_seed
    speedup = base_per_seed / per_seed
    print(
        f"{n:>10} {c:>10.2f} {r:>9.3f} {per_seed:>13.4f} {throughput:>9.2f} {speedup:>7.2f}x"
    )
    if best is None or throughput > best[0]:
        best = (throughput, n, r)

print(
    "\nflat round_s/seed -> latency-bound -> vmap wins.  linear growth -> compute-bound."
)
if best is not None:
    print(
        f"\nSuggested --concurrent-seeds {best[1]}  "
        f"(peak throughput {best[0]:.2f} seeds/s, {best[2]:.3f}s/round).\n"
        f"Pick a value that also divides your --num-seeds; the next-lower divisor\n"
        f"keeps most of the speedup if {best[1]} doesn't divide it."
    )
