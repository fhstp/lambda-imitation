"""Prioritised replay end-to-end: sampling, IS weights, priority write-back.

``per_alpha = 0`` (the default) must leave everything exactly as it was — same
draws, unit IS weights, and no ``priorities`` array in the state at all, so
agents checkpointed before this feature still load.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


def _agent(**hp_overrides):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=8, online_batch_size=8,
        online_buffer_size=512, burn_in_length=2, sequence_length=4,
        lambda_truncation=2, lambda1=0.05, lambda2=0.85, retrace=True,
        # real ratios: fake_onpolicy_loss=True pins them to 1, which makes
        # every window priority identical (and is rejected outright when
        # per_alpha > 0)
        **{"fake_onpolicy_loss": False, **hp_overrides},
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
    key, reset_key, prefill_key = jax.random.split(key, 3)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    # A scripted behaviour policy reporting b(a|s) = 1, so the ratios
    # pi/b = pi(a|s) < 1 and the trace priority is genuinely below the
    # max-priority initialisation (mirrors the expert-data setting; a uniform
    # random prefill gives ratios >= 1, which clip to 1 and are then
    # indistinguishable from the init).
    def deterministic_behaviour(obs, env_st, k):
        return jnp.float32(0.0), jnp.float32(1.0)

    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key,
        behaviour_fn=deterministic_behaviour)
    return state, fns, key


def test_default_keeps_no_priorities_so_old_checkpoints_load():
    state, fns, key = _agent()
    assert state.online_buffer.priorities is None
    n_before = len(jax.tree.leaves(state))
    state, _metrics = fns.update_only(state, 1, key)
    assert state.online_buffer.priorities is None
    assert len(jax.tree.leaves(state)) == n_before


def test_prioritised_run_reports_its_diagnostics():
    state, fns, key = _agent(per_alpha=0.6, per_beta=0.4)
    assert state.online_buffer.priorities is not None
    _state, metrics = fns.update_only(state, 1, key)
    assert "per_ess" in metrics and "per_priority" in metrics, sorted(metrics)
    # ESS is a fraction of the batch and must be positive
    assert 0.0 < float(metrics["per_ess"]) <= 1.0, metrics["per_ess"]


def test_priorities_are_written_back_from_the_traces():
    state, fns, key = _agent(per_alpha=0.6, per_beta=0.4)
    before = np.asarray(state.online_buffer.priorities)
    state, _metrics = fns.update_only(state, 3, key)
    after = np.asarray(state.online_buffer.priorities)
    assert not np.allclose(before, after), "no priority was updated"
    assert (after > 0).all(), "priorities must stay strictly positive"


def test_prioritised_training_is_finite():
    state, fns, key = _agent(per_alpha=0.6, per_beta=0.4)
    _state, metrics = fns.update_only(state, 5, key)
    assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics)), metrics


def test_prioritisation_with_pinned_ratios_is_rejected():
    """fake_onpolicy_loss=True makes every priority identical — fail loudly
    rather than silently running uniform sampling for 50k updates."""
    import pytest
    with pytest.raises(ValueError, match="fake_onpolicy_loss"):
        _agent(per_alpha=0.6, fake_onpolicy_loss=True)
