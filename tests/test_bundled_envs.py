"""Bundled environment observations, reset semantics and reference policy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_pocman_wall_bits_match_every_walkable_cell():
    pytest.importorskip("jumanji")
    from lambda_imitation.envs.pocman import PocMan
    from jumanji.environments.routing.pac_man.types import Position

    env = PocMan()
    _, state = env.reset(jax.random.key(0), env.default_params)
    grid = np.asarray(state.grid)
    get_obs = jax.jit(env.get_obs)
    for row, col in np.argwhere(grid == 1):
        at_cell = state.replace(player_locations=Position(x=jnp.int32(row), y=jnp.int32(col)))
        obs = get_obs(at_cell)
        expected = [grid[max(row - 1, 0), col], grid[row, min(col + 1, grid.shape[1] - 1)],
                    grid[min(row + 1, grid.shape[0] - 1), col], grid[row, max(col - 1, 0)]]
        np.testing.assert_array_equal(obs[:4], expected)
