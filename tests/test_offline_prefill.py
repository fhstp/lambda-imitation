"""Tests for the offline-training hooks: a scripted pre-fill behaviour policy
and env-free gradient updates.

Both exist so an agent can be trained on data from a *scripted* policy with no
environment interaction at all (see
``examples/lambda-envs/battleship_offline_bayes.py``):

* ``fns.prefill_buffer(..., behaviour_fn=...)`` writes transitions from a
  caller-supplied policy instead of the uniform-random default, storing that
  policy's ``b(a|s)`` so the V-trace importance ratios stay correct,
* ``fns.update_only(state, n_steps, key)`` runs ``n_steps`` gradient updates
  off the buffer alone — no env, no new transitions.
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest

from lambda_imitation.iqlearn import Hyperparameters, behaviour_key

action_key = "actions"   # create_iqlearn default
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


def _tiny_agent(seed=0):
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2,
        batch_size=4,
        online_batch_size=4,
        online_buffer_size=256,
        burn_in_length=2,
        sequence_length=4,
        lambda_truncation=2,
    )
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns = create_iqlearn_from_env(
        spec, expert_data, buffer_size=4, hp=hp,
        projection=16, memory_type="gru", memory_hidden_dim=8,
        use_prev_action=True, critic_dims=(16,), train_steps=4,
        approximate_lambda=True, seed=seed,
    )
    return env, env_params, hp, state, fns


@pytest.fixture(scope="module")
def agent():
    return _tiny_agent()


def test_prefill_behaviour_fn_stores_its_actions_and_probabilities(agent):
    env, env_params, hp, state, fns = agent
    key = jax.random.key(0)
    key, reset_key, prefill_key = jax.random.split(key, 3)
    _obs, env_state = env.reset(reset_key, env_params)

    # A scripted policy: always action 1, which it reports taking w.p. 0.7.
    def always_right(obs, env_st, k):
        return jnp.float32(1.0), jnp.float32(0.7)

    n_steps = 16
    state, _env_state = fns.prefill_buffer(
        state, env, env_params, env_state, n_steps, prefill_key,
        behaviour_fn=always_right,
    )

    written = slice(0, n_steps)
    actions = state.online_buffer.info[action_key][written]
    probs = state.online_buffer.info[behaviour_key][written]
    assert jnp.all(actions == 1.0), actions
    assert jnp.allclose(probs, 0.7), probs


def test_prefill_without_behaviour_fn_is_still_uniform_random(agent):
    """The default path must be untouched: uniform over 2 CartPole actions."""
    env, env_params, hp, state, fns = agent
    key = jax.random.key(1)
    key, reset_key, prefill_key = jax.random.split(key, 3)
    _obs, env_state = env.reset(reset_key, env_params)

    n_steps = 32
    state, _env_state = fns.prefill_buffer(
        state, env, env_params, env_state, n_steps, prefill_key
    )
    probs = state.online_buffer.info[behaviour_key][:n_steps]
    actions = state.online_buffer.info[action_key][:n_steps]
    assert jnp.allclose(probs, 0.5), probs
    assert set(jnp.unique(actions).tolist()) <= {0.0, 1.0}


def test_update_only_trains_from_the_buffer_with_no_environment(agent):
    env, env_params, hp, state, fns = agent
    key = jax.random.key(2)
    key, reset_key, prefill_key, update_key = jax.random.split(key, 4)
    _obs, env_state = env.reset(reset_key, env_params)

    prefill = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _env_state = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key
    )
    before = jax.tree.leaves(state.actor)
    pos_before = int(state.online_buffer.pos)

    state, metrics = fns.update_only(state, 3, update_key)

    after = jax.tree.leaves(state.actor)
    assert any(not jnp.allclose(a, b) for a, b in zip(before, after)), \
        "update_only did not change the actor parameters"
    # No environment interaction: the buffer must not have grown.
    assert int(state.online_buffer.pos) == pos_before
    assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics)), metrics
