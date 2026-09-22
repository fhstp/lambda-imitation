"""Which critic the actor maximises.

The SAC critic and the λ-critics end up on very different value scales on the
offline Battleship data: after the Retrace / twin / λ_coef fixes the λ-critics
settle at +47, between the analytic 1-step fixed point (+35.6) and the buffer's
own MC return (+55.5), while the SAC critic sits at +106 — above the +100
ceiling, i.e. overestimating on actions the data never took.  ``actor_critic``
lets the actor be extracted from a λ-critic instead, which also couples the
policy directly to the λ-discrepancy objective.
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


def _agent(approximate_lambda=True, **hp_overrides):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=8, online_batch_size=8,
        online_buffer_size=512, burn_in_length=2, sequence_length=4,
        lambda_truncation=2, lambda1=0.05, lambda2=0.85, retrace=True,
        actor_lr=3e-3, **hp_overrides,
    )
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns = create_iqlearn_from_env(
        spec, expert_data, buffer_size=4, hp=hp, projection=16,
        memory_type="gru", memory_hidden_dim=8, use_prev_action=True,
        critic_dims=(16,), train_steps=4,
        approximate_lambda=approximate_lambda, seed=0,
    )
    key = jax.random.key(0)
    key, reset_key, prefill_key, update_key = jax.random.split(key, 4)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key)
    return state, fns, update_key


def _actor_after_one_update(**hp_overrides):
    state, fns, key = _agent(**hp_overrides)
    state, metrics = fns.update_only(state, 1, key)
    return jax.tree.leaves(state.actor), metrics


def test_extracting_from_a_lambda_critic_changes_the_actor_update():
    sac_actor, _ = _actor_after_one_update(actor_critic="sac")
    lam_actor, _ = _actor_after_one_update(actor_critic="lambda1")
    assert any(not jnp.allclose(a, b) for a, b in zip(sac_actor, lam_actor)), \
        "actor update identical — it is still being extracted from the SAC critic"


def test_lambda1_and_lambda2_are_distinct_sources():
    a1, _ = _actor_after_one_update(actor_critic="lambda1")
    a2, _ = _actor_after_one_update(actor_critic="lambda2")
    assert any(not jnp.allclose(a, b) for a, b in zip(a1, a2))


def test_lambda_source_without_lambda_critics_is_rejected():
    with pytest.raises(ValueError, match="actor_critic"):
        _agent(approximate_lambda=False, actor_critic="lambda1")


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="actor_critic"):
        _agent(actor_critic="banana")


def test_training_from_a_lambda_critic_stays_finite():
    _leaves, metrics = _actor_after_one_update(actor_critic="lambda2")
    assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics)), metrics
