"""Centred λ-discrepancy: ``loss_ld`` minus its state-independent bias.

Diagnosed on the 10x10 offline runs: ``ld_loss = mean((Q1-Q2)^2)`` was driven
almost entirely by a *constant* offset between the two λ-critics — in every
logged round ``sqrt(ld_loss)`` matched ``|E[Q1]-E[Q2]|`` to two decimals, i.e.
``var(Q1-Q2) ~ 0``.  A uniform offset carries no information about the hidden
state, so that gradient cannot teach the recurrent memory anything; it only
grows with the λ-critics' value drift (to 1e4 while the SAC critic sat at 1e2).

``ld_center`` subtracts the batch mean before squaring, so the term becomes the
variance of the discrepancy — the state-dependent part the λ-discrepancy is
meant to measure.  The two components are logged either way as ``ld_mean`` and
``ld_std``.
"""

import gymnax
import jax
import jax.numpy as jnp

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

def _metrics_of_one_update(**hp_overrides):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=4,
        online_buffer_size=256, burn_in_length=2, sequence_length=4,
        lambda_truncation=2, **hp_overrides,
    )
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns = create_iqlearn_from_env(
        spec, expert_data, buffer_size=4, hp=hp, projection=16,
        memory_type="gru", memory_hidden_dim=8, use_prev_action=True,
        critic_dims=(16,), train_steps=4, approximate_lambda=True, seed=0,
    )
    key = jax.random.key(0)
    key, reset_key, prefill_key, update_key = jax.random.split(key, 4)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key)
    # exactly one update: the metrics are then that step's, not an average
    _state, metrics = fns.update_only(state, 1, update_key)
    return {k: float(v) for k, v in metrics.items()}

def test_discrepancy_components_are_logged():
    m = _metrics_of_one_update()
    assert "ld_mean" in m and "ld_std" in m, sorted(m)

def test_uncentred_loss_is_mean_squared_plus_variance():
    m = _metrics_of_one_update(ld_center=False)
    assert jnp.allclose(m["ld_loss"], m["ld_mean"] ** 2 + m["ld_std"] ** 2,
                        rtol=1e-4, atol=1e-7), m

def test_centred_loss_drops_the_bias_and_keeps_the_variance():
    m = _metrics_of_one_update(ld_center=True)
    assert jnp.allclose(m["ld_loss"], m["ld_std"] ** 2, rtol=1e-4, atol=1e-7), m
