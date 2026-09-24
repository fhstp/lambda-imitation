"""The λ-critics must train BOTH twin branches, like the SAC critic does.

``get_q`` returns ``min(q1, q2)``, so regressing it gives gradient only to
whichever branch is currently lower.  The branches then drift apart, and since
every bootstrap uses ``V̄ = Σ_a π(a)·min(q1,q2)``, a spread pair makes that min
systematically pessimistic — a bias applied at every backup, compounding over
the 1/(1-γ) = 100-step effective horizon.  Measured on a saved agent: mean
|q1-q2| of the output biases was 0.004 for the SAC critic (both branches
trained) versus 0.209 / 0.225 for λ1 / λ2, alongside a monotone value drift of
-8.6 → -79.2 while the SAC critic sat at +80 on the same data.

The SAC critic's recipe is the fix: regress each branch against the shared
target, keeping ``min`` for target construction only.
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

def _agent(seed=0):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=8,
        online_buffer_size=512, burn_in_length=2, sequence_length=4,
        lambda_truncation=2, critic_lr=3e-3, lambda_critic_lr=3e-3,
        lambda1=0.05, lambda2=0.85,
    )
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns = create_iqlearn_from_env(
        spec, expert_data, buffer_size=4, hp=hp, projection=16,
        memory_type="gru", memory_hidden_dim=8, use_prev_action=True,
        critic_dims=(16,), train_steps=4, approximate_lambda=True, seed=seed,
    )
    key = jax.random.key(seed)
    key, reset_key, prefill_key = jax.random.split(key, 3)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key)
    return state, fns, key

def test_twin_gap_is_reported_for_each_lambda_critic():
    state, fns, key = _agent()
    _state, metrics = fns.update_only(state, 1, key)
    assert "lambda0.05_twin_gap:" in metrics, sorted(metrics)
    assert "lambda0.85_twin_gap:" in metrics, sorted(metrics)

def test_lambda_branches_do_not_drift_apart_while_training():
    """Both branches are fit to the same target, so the pair must not spread —
    the SAC critic's twin gap is the reference for what 'trained' looks like."""
    state, fns, key = _agent()
    state, early = fns.update_only(state, 5, key)
    state, late = fns.update_only(state, 150, jax.random.split(key)[0])
    for lam in ("0.05", "0.85"):
        g0 = float(early[f"lambda{lam}_twin_gap:"])
        g1 = float(late[f"lambda{lam}_twin_gap:"])
        assert g1 <= g0 * 1.5, (
            f"λ={lam} twin branches drifted apart: {g0:.4f} -> {g1:.4f}; "
            "only the min branch is being trained"
        )
