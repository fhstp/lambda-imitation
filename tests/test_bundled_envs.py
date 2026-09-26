"""Bundled environment observations, reset semantics and reference policy."""

import jax
import jax.numpy as jnp
import numpy as np

from lambda_imitation.envs.battleship import Battleship


def test_battleship_placement_mask_and_return():
    env = Battleship()
    params = env.default_params
    obs, state = env.reset(jax.random.key(0), params)
    assert int(state.board.sum()) == 5
    np.testing.assert_array_equal(obs, np.r_[0, np.ones(25)])
    actions = list(np.flatnonzero(np.asarray(state.board).ravel() == 0))
    actions += list(np.flatnonzero(np.asarray(state.board).ravel()))
    total = 0
    for i, action in enumerate(actions):
        obs, state, reward, done, _ = env.step_env(jax.random.key(i), state, action, params)
        total += float(reward)
        assert obs[action + 1] == 0
        assert bool(obs[0]) == bool(state.board.ravel()[action])
        assert bool(done) == (i == 24)
    assert total == 26 - len(actions)


def test_battleship_public_step_auto_resets():
    env = Battleship()
    _, state = env.reset(jax.random.key(0))
    last = jnp.argmax(state.board.reshape(-1))
    shots = (state.board * 2).reshape(-1).at[last].set(0).reshape(5, 5)
    state = state.replace(hits_misses=shots)
    obs, reset, reward, done, _ = env.step(jax.random.key(1), state, last)
    assert done and reward == 25
    np.testing.assert_array_equal(obs, env.get_obs(reset))
    assert not reset.hits_misses.any()
