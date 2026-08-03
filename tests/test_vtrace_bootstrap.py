"""The V-trace tail bootstrap must not inject bias into the kept window.

The backward recursion is

    delta_t = r_t + (1-d_t) γ V_{t+1} - V_t
    v_t     = V_t + ρ_t delta_t + (1-d_t) γ c_t (v_{t+1} - V_{t+1})

so the terminal carry only enters through `(v_{T'} - V_{T'})`. Initialising that
difference at zero asserts nothing; initialising the carry at `(0, 0)` in ABSOLUTE form
instead asserts `V(s_{T'}) = 0` and injects an error of `-V(s_{T'})` which decays only as
`(γλ)^k`. With `lambda_truncation=20` that leaves 29 % of the error at λ=0.95 and 82 % at
λ=1 — and the V-trace actor uses λ=1. See differences_report.md §1.2.

The test pins the fixed-point property: on a steady state where `V = -1/(1-γ)` is exactly
the true value, every λ-return target must equal `V` regardless of λ.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.iqlearn import sf_vtrace_targets

GAMMA = 0.99
V_FIX = -1.0 / (1.0 - GAMMA)          # -100: true value of "-1 per step, never ends"
T, B, NF = 130, 3, 2
TRUNC = 20                            # what the probe hardcodes


def _steady_state(n_features=NF):
    V = jnp.full((T, B, n_features), V_FIX, jnp.float32)
    cumulants = jnp.full((T, B, n_features), -1.0, jnp.float32)
    dones = jnp.zeros((T, B), jnp.float32)
    ratios = jnp.ones((T, B), jnp.float32)
    return V, cumulants, dones, ratios


@pytest.mark.parametrize("lam", [0.0, 0.5, 0.75, 0.95, 0.99, 1.0])
def test_targets_equal_the_fixed_point_for_every_lambda(lam):
    """At the value fixed point the λ-return must be the fixed point, for all λ.

    This is the property the (0, 0) init violated: it returned -71 at λ=0.95 and
    -19 at λ=1 for the last kept step, against a true -100.
    """
    V, cumulants, dones, ratios = _steady_state()
    targets = np.asarray(sf_vtrace_targets(
        V, cumulants, dones, ratios, GAMMA, lam, 1.0, 1.0))
    kept = targets[:-TRUNC]
    err = np.abs(kept - V_FIX).max()
    assert err < 1e-2, (
        f"λ={lam}: max |target - {V_FIX}| = {err:.3f} over the kept window; "
        "a non-zero tail correction is leaking backward")


def test_worst_case_lambda_one_is_unbiased_at_the_last_kept_step():
    """λ=1 is the actor's setting and the worst case for tail bias."""
    V, cumulants, dones, ratios = _steady_state()
    targets = np.asarray(sf_vtrace_targets(
        V, cumulants, dones, ratios, GAMMA, 1.0, 1.0, 1.0))
    last_kept = targets[-(TRUNC + 1)].mean()
    assert abs(last_kept - V_FIX) < 1e-2, (
        f"last kept target {last_kept:.2f} vs true {V_FIX}; the (0,0) init gave -19.0")


def test_terminations_still_cut_the_trace():
    """A done at step t must stop the bootstrap: v_t = V_t + ρ(r_t - V_t)."""
    V, cumulants, dones, ratios = _steady_state(n_features=1)
    t_done = 60
    dones = dones.at[t_done].set(1.0)
    targets = np.asarray(sf_vtrace_targets(
        V, cumulants, dones, ratios, GAMMA, 0.95, 1.0, 1.0))
    expected = V_FIX + (-1.0 - V_FIX)          # r + 0 - V, added to V
    np.testing.assert_allclose(targets[t_done], expected, atol=1e-3)


def test_reward_scale_propagates():
    """Sanity: a different constant reward moves the fixed point accordingly."""
    r = -2.0
    v_fix = r / (1.0 - GAMMA)
    V = jnp.full((T, B, 1), v_fix, jnp.float32)
    targets = np.asarray(sf_vtrace_targets(
        V, jnp.full((T, B, 1), r, jnp.float32), jnp.zeros((T, B), jnp.float32),
        jnp.ones((T, B), jnp.float32), GAMMA, 0.95, 1.0, 1.0))
    assert abs(targets[:-TRUNC].mean() - v_fix) < 1e-2
