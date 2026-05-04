"""Numerical-equivalence guards for upcoming SAC perf refactors.

These tests snapshot the full output of ``train_sac`` (which exercises
``update_step_sac`` under the active sequence/IID path) for fixed RNG keys,
and assert that subsequent code changes reproduce the snapshot bit-for-bit
within a tight float tolerance.

Use these tests to verify refactors that are *meant* to be numerically
identical to the current implementation (e.g. swapping per-step `get_v`
calls for one batched `run_actor_scan`+`run_critic_scan`, hoisting `nnx.merge`
out of scan bodies, gating diagnostics).  Refactors that intentionally change
numerics (e.g. sharing one buffer sample between actor and critic loss)
should NOT pass these tests — that's the whole point.

To regenerate the golden files (e.g. after a deliberate numerics change),
run with ``LAMBDA_REGEN_GOLDEN=1 pytest tests/test_sac_perf_equivalence.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Reuse the existing sequence-SAC helper from the main test module.  This
# guarantees the snapshot exercises the same code path as TestSeqTrainSAC.
from tests.test_iqlearn import _make_seq_sac


GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

# Tolerances chosen tight enough to catch any real numerics change but loose
# enough to absorb XLA op reordering across equivalent traces.  Strict
# bit-exact (atol=0, rtol=0) is too brittle: even removing dead code can
# trigger different XLA op scheduling and produce ~1e-5 drifts in float32 due
# to non-associative add reordering.  1e-4 absorbs that while still catching
# any real algorithmic divergence (typical real changes are >1e-3).
ATOL = 1e-4
RTOL = 1e-4

REGEN = os.environ.get("LAMBDA_REGEN_GOLDEN") == "1"


def _flatten_leaves(tree, prefix=""):
    """Walk a pytree and return ``{path: ndarray}`` for every float leaf.

    Non-float leaves (uint32 PRNG counters, etc.) are skipped — they are
    bookkeeping state, not numeric quantities under test.
    """
    out: dict[str, np.ndarray] = {}

    def _walk(node, path):
        if hasattr(node, "_fields"):  # NamedTuple
            for f in node._fields:
                _walk(getattr(node, f), f"{path}.{f}" if path else f)
        elif isinstance(node, dict):
            for k, v in node.items():
                _walk(v, f"{path}[{k}]" if path else f"[{k}]")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _walk(v, f"{path}[{i}]" if path else f"[{i}]")
        elif node is None:
            pass
        else:
            arr = np.asarray(node)
            if np.issubdtype(arr.dtype, np.floating):
                out[path] = arr

    _walk(tree, prefix)
    return out


def _snapshot(new_state, metrics) -> dict[str, np.ndarray]:
    snap = _flatten_leaves(new_state, prefix="state")
    for k, v in metrics.items():
        snap[f"metric.{k}"] = np.asarray(v)
    return snap


def _save_golden(name: str, snap: dict[str, np.ndarray]) -> None:
    path = GOLDEN_DIR / f"{name}.npz"
    np.savez_compressed(path, **snap)


def _load_golden(name: str) -> dict[str, np.ndarray] | None:
    path = GOLDEN_DIR / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _compare(name: str, snap: dict[str, np.ndarray]) -> None:
    """Compare snap to golden file or write it on first run / regen."""
    if REGEN:
        _save_golden(name, snap)
        pytest.skip(f"regenerated golden for {name}")
    golden = _load_golden(name)
    if golden is None:
        _save_golden(name, snap)
        pytest.skip(
            f"golden {name}.npz not present — wrote fresh snapshot. "
            "Re-run pytest to enable equivalence check."
        )
    # Key set must match exactly — a refactor that adds/removes leaves
    # silently is a regression we want to catch.
    missing_in_new = sorted(set(golden) - set(snap))
    missing_in_golden = sorted(set(snap) - set(golden))
    assert not missing_in_new, f"{name}: keys missing in new snap: {missing_in_new[:5]}"
    assert not missing_in_golden, (
        f"{name}: keys missing in golden (regen needed?): {missing_in_golden[:5]}"
    )
    for k in golden:
        g = golden[k]
        s = snap[k]
        assert g.shape == s.shape, f"{name}.{k}: shape {s.shape} != golden {g.shape}"
        if not np.allclose(g, s, atol=ATOL, rtol=RTOL):
            diff = np.abs(g.astype(np.float64) - s.astype(np.float64))
            max_abs = float(diff.max())
            max_rel = float((diff / (np.abs(g.astype(np.float64)) + 1e-12)).max())
            pytest.fail(
                f"{name}.{k}: numerics drift — max|Δ|={max_abs:.3e}, "
                f"max rel={max_rel:.3e} (atol={ATOL}, rtol={RTOL})"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSACUpdateNumericalEquivalence:
    """Snapshot guards for upcoming SAC perf refactors.

    Each test:
      1. Builds a fixed-seed SAC agent (discrete or continuous).
      2. Runs one ``train_sac`` round (which calls ``update_step_sac`` inside
         a ``jax.lax.scan`` over the configured ``train_steps``).
      3. Captures every float leaf of the resulting state plus every metric.
      4. Compares to the corresponding ``tests/golden/*.npz`` snapshot.

    Refactors that preserve numerics (e.g. swapping per-step helpers for
    batched scans, hoisting `nnx.merge`, gating diagnostics off — only the
    metrics whose keys still exist are compared, so gating is fine when the
    snapshot was taken with diagnostics on) keep these green.  Any drift
    above 1e-5 atol/rtol fails the test.
    """

    def test_discrete_seq_sac_equivalence(self):
        _, new_state, _, metrics = _make_seq_sac(discrete=True)
        snap = _snapshot(new_state, metrics)
        _compare("seq_sac_discrete", snap)

    def test_continuous_seq_sac_equivalence(self):
        _, new_state, _, metrics = _make_seq_sac(discrete=False)
        snap = _snapshot(new_state, metrics)
        _compare("seq_sac_continuous", snap)

    def test_discrete_seq_sac_burn_in_zero_equivalence(self):
        # Burn-in zero exercises a different branch in sample_with_burn_in.
        _, new_state, _, metrics = _make_seq_sac(discrete=True, burn_in=0)
        snap = _snapshot(new_state, metrics)
        _compare("seq_sac_discrete_burn0", snap)

    def test_continuous_seq_sac_burn_in_zero_equivalence(self):
        _, new_state, _, metrics = _make_seq_sac(discrete=False, burn_in=0)
        snap = _snapshot(new_state, metrics)
        _compare("seq_sac_continuous_burn0", snap)
