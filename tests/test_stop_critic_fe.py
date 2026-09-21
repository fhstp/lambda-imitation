"""``stop_critic_fe``: detach the **SAC** twin-Q critic from the shared encoder.

The 10x10 offline runs showed retention peaking at 10k updates and then decaying
while the SAC critic loss grew ~10x — value-magnitude inflation being pushed into
the shared recurrent memory.  ``stop_critic_fe`` cuts exactly that path: the main
twin critic learns on detached features, while the λ-critics *and* their
discrepancy (``loss_ld``) keep the live latent and go on training the FE — they
are the terms that are supposed to shape memory.

(The dense-value-loss branch's knob of the same name detaches the λ-critics and
``loss_ld`` as well.  These tests pin the narrower semantics used here.)
"""

import gymnax
import jax
import jax.numpy as jnp

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


def _agent(approximate_lambda=True, **hp_overrides):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=4, online_batch_size=4,
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
        critic_dims=(16,), train_steps=4,
        approximate_lambda=approximate_lambda, seed=0,
    )
    return env, env_params, hp, state, fns


def _fe_after_updates(approximate_lambda=True, **hp_overrides):
    """FE leaves before and after 3 offline updates on a random-policy buffer."""
    env, env_params, hp, state, fns = _agent(approximate_lambda, **hp_overrides)
    key = jax.random.key(0)
    key, reset_key, prefill_key, update_key = jax.random.split(key, 4)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key)
    before = jax.tree.leaves(state.feature_extractor)
    state, _metrics = fns.update_only(state, 3, update_key)
    return before, jax.tree.leaves(state.feature_extractor)


def _changed(before, after):
    return any(not jnp.allclose(a, b) for a, b in zip(before, after))


def test_lambda_side_still_trains_the_fe_when_the_sac_critic_is_detached():
    before, after = _fe_after_updates(
        approximate_lambda=True, stop_critic_fe=True, stop_actor_fe=True,
        lambda_coef=1.0)
    assert _changed(before, after), \
        "FE frozen although the λ-critics and their discrepancy should train it"


def test_sac_critic_path_is_really_cut():
    """No λ side, actor detached: the SAC critic is the only remaining path,
    and stop_critic_fe must leave the FE untouched."""
    before, after = _fe_after_updates(
        approximate_lambda=False, stop_critic_fe=True, stop_actor_fe=True)
    assert not _changed(before, after), \
        "FE changed although the SAC critic was the only live path"


def test_default_keeps_the_sac_critic_training_the_fe():
    before, after = _fe_after_updates(
        approximate_lambda=False, stop_critic_fe=False, stop_actor_fe=True)
    assert _changed(before, after), \
        "with stop_critic_fe=False the SAC critic loss must reach the FE"
