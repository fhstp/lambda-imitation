"""Prioritised sequence sampling: draw the windows whose traces survive.

With a near-deterministic target policy and expert behaviour data, ρ = min(1,
π/b) is 0 on ~97 % of transitions (measured), so every V-trace / Retrace trace
is cut at the first step and λ1 / λ2 receive identical targets — the
λ-discrepancy has nothing to measure on a uniformly drawn window.  Prioritising
by the window's trace mass samples the minority of windows where the policies do
agree, which are the only ones where the discrepancy carries signal.

Score is the log-space geometric mean of the clipped ratios over the window,
which preserves the ranking of the raw product Π ρ without underflowing (a
strict product over 50 steps is 0 for essentially every window).

``priority_alpha = 0`` must reproduce uniform sampling exactly, so every earlier
result stays reproducible.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lambda_imitation.buffer import (
    Buffer,
    create_prioritised_sequence_sample,
    create_sequence_sample,
    trace_priority,
    update_priorities,
)

SIZE, SEQ, BATCH = 64, 4, 256


def _buffer(priorities=None):
    info = {"x": jnp.arange(SIZE, dtype=jnp.float32)[:, None]}
    return Buffer(
        info,
        sampling_ok=jnp.ones((SIZE,), dtype=jnp.bool),
        pos=SIZE,
        size=SIZE,
        priorities=jnp.ones((SIZE,)) if priorities is None else priorities,
    )


def test_alpha_zero_is_identical_to_uniform_sampling():
    buf = _buffer(priorities=jax.random.uniform(jax.random.key(1), (SIZE,)))
    uniform = create_sequence_sample(SIZE, BATCH, SEQ, ["x"])
    prioritised = create_prioritised_sequence_sample(SIZE, BATCH, SEQ, ["x"], alpha=0.0)
    key = jax.random.key(0)
    (_s_u, idx_u) = uniform(buf, key)
    (_s_p, idx_p, probs) = prioritised(buf, key)
    assert np.array_equal(np.asarray(idx_u), np.asarray(idx_p))
    # uniform over the 64 valid windows
    assert np.allclose(np.asarray(probs), 1.0 / SIZE, atol=1e-6)


def test_high_priority_windows_dominate_the_draw():
    p = jnp.full((SIZE,), 0.01).at[7].set(10.0)
    buf = _buffer(priorities=p)
    prioritised = create_prioritised_sequence_sample(SIZE, BATCH, SEQ, ["x"], alpha=1.0)
    _s, idx, _probs = prioritised(buf, jax.random.key(0))
    starts = np.asarray(idx)[:, 0]
    frac = (starts == 7).mean()
    assert frac > 0.5, f"slot 7 drawn only {frac:.2%} of the time"


def test_invalid_windows_are_never_drawn():
    ok = jnp.ones((SIZE,), dtype=jnp.bool).at[20].set(False)
    buf = _buffer()._replace(sampling_ok=ok)
    prioritised = create_prioritised_sequence_sample(SIZE, BATCH, SEQ, ["x"], alpha=1.0)
    _s, idx, _probs = prioritised(buf, jax.random.key(0))
    starts = set(np.asarray(idx)[:, 0].tolist())
    # a window starting in [17, 20] would span the invalid slot
    assert starts.isdisjoint({17, 18, 19, 20}), sorted(starts & {17, 18, 19, 20})


def test_update_priorities_round_trips():
    buf = _buffer()
    idx = jnp.array([3, 9, 9])
    vals = jnp.array([0.5, 0.25, 0.75])
    out = update_priorities(buf, idx, vals)
    assert float(out.priorities[3]) == 0.5
    assert float(out.priorities[9]) in (0.25, 0.75)   # duplicate index: last wins
    assert float(out.priorities[0]) == 1.0            # untouched


def test_trace_priority_ranks_like_the_raw_product_without_underflowing():
    # three windows: fully grounded, half grounded, fully cut
    ratios = jnp.array([[1.0] * 8, [1.0] * 4 + [0.0] * 4, [0.0] * 8])
    pri = trace_priority(ratios, floor=1e-3)
    assert pri[0] > pri[1] > pri[2] > 0.0, pri
    # same ordering as the product of clipped ratios, computed in float64
    prod = np.prod(np.clip(np.asarray(ratios, dtype=np.float64), 1e-3, 1.0), axis=-1)
    assert np.argsort(np.asarray(pri)).tolist() == np.argsort(prod).tolist()
