"""POPGym rule checks against deterministic boards, without importing POPGym."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("gymnax")

_path = Path(__file__).resolve().parents[1] / "examples/lambda-envs/_minesweeper_env.py"
_spec = importlib.util.spec_from_file_location("minesweeper_env", _path)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def _numpy_counts(mines):
    """Independent clipped-slice oracle, including rectangular/thin boards."""
    rows, cols = mines.shape
    counts = np.zeros((rows, cols), dtype=np.int32)
    for row in range(rows):
        for col in range(cols):
            block = mines[max(0, row-1):row+2, max(0, col-1):col+2]
            counts[row, col] = int(block.sum()) - int(mines[row, col])
    return counts


def _board_state(env, mine_locations=((0, 0), (3, 3))):
    mines = np.zeros((env.rows, env.cols), dtype=bool)
    for location in mine_locations:
        mines[location] = True
    assert mines.sum() == env.num_mines
    return ms.MineSweeperState(
        mines=jnp.asarray(mines),
        neighbor_counts=jnp.asarray(_numpy_counts(mines)),
        viewed=jnp.zeros_like(jnp.asarray(mines)),
        timestep=jnp.int32(0),
        last_value=jnp.int32(-1),
        hit_mine=jnp.bool_(False),
    )


def _step(env, state, action, params=None):
    if params is None:
        params = env.default_params
    return env.step_env(jax.random.key(0), state, action, params)


def _assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def env():
    return ms.MineSweeper(rows=4, cols=4, num_mines=2)


@pytest.mark.parametrize("rows,cols,mines,partial_dim,history_dim", [
    (6, 6, 6, 8, 289), (4, 4, 2, 4, 65), (8, 8, 10, 10, 641),
])
@pytest.mark.parametrize("remember", [False, True])
def test_public_contract_and_observation_spaces(rows, cols, mines, partial_dim,
                                                history_dim, remember):
    env = ms.MineSweeper(rows, cols, mines, remember=remember)
    assert isinstance(env, ms.environment.Environment)
    assert type(env.num_actions) is int and env.num_actions == rows * cols
    assert type(env.episode_length) is int and env.episode_length == rows * cols - mines
    assert env.default_params.max_steps_in_episode == env.episode_length
    assert type(env.default_params.max_steps_in_episode) is int
    assert env.obs_requires_prev_action == (not remember)
    assert env.action_space().n == rows * cols
    assert env.action_space().shape == ()
    assert env.action_space().contains(rows * cols - 1)
    assert not env.action_space().contains(rows * cols)
    obs, state = env.reset(jax.random.key(0))
    space = env.observation_space()
    assert space.shape == obs.shape == ((history_dim if remember else partial_dim),)
    assert obs.dtype == space.dtype == jnp.float32
    assert space.contains(obs)
    assert env.state_space().contains(state)
    assert not env.is_terminal(state)
    np.testing.assert_array_equal(obs, env.get_obs(state))
    assert ms.MineSweeper().episode_length == 30


@pytest.mark.parametrize("kwargs,match", [
    ({"rows": 0}, "positive"),
    ({"cols": -1}, "positive"),
    ({"rows": 4.5}, "integers"),
    ({"num_mines": -1}, "num_mines"),
    ({"rows": 2, "cols": 2, "num_mines": 4}, "num_mines"),
    ({"rows": 2, "cols": 2, "num_mines": 2}, "three safe"),
    ({"rows": 2, "cols": 2, "num_mines": 3}, "three safe"),
])
def test_invalid_configs_do_not_silently_change_reference_rewards(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ms.MineSweeper(**kwargs)


@pytest.mark.parametrize("board,expected", [
    ([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1]],
     [[1, 2, 2, 0], [2, 1, 3, 2], [1, 1, 2, 0]]),
    ([[1, 0, 0, 1, 0]], [[0, 1, 1, 0, 1]]),
    ([[1], [0], [0], [1], [0]], [[0], [1], [1], [0], [1]]),
])
def test_eight_neighbor_counts_corners_edges_and_no_wrap(board, expected):
    actual = jax.jit(ms._count_neighbors)(jnp.asarray(board, dtype=jnp.bool_))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jnp.int32


def test_reset_exact_mine_count_independent_counts_and_immutable_state(env):
    keys = jax.random.split(jax.random.key(31), 16)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    np.testing.assert_array_equal(states.mines.sum(axis=(1, 2)), 2)
    assert states.mines.dtype == states.viewed.dtype == jnp.bool_
    assert states.neighbor_counts.dtype == states.timestep.dtype == jnp.int32
    assert not np.asarray(states.viewed).any()
    assert not np.asarray(states.hit_mine).any()
    np.testing.assert_array_equal(states.timestep, 0)
    np.testing.assert_array_equal(states.last_value, -1)
    np.testing.assert_array_equal(obs, np.tile([0, 0, 0, 1], (16, 1)))
    for mines, counts in zip(np.asarray(states.mines), np.asarray(states.neighbor_counts)):
        np.testing.assert_array_equal(counts, _numpy_counts(mines))
    assert len(np.unique(np.asarray(states.mines).reshape(16, -1), axis=0)) > 1
    first = jax.tree.map(lambda x: x[0], states)
    _assert_tree_equal(env.reset(keys[0]), (obs[0], first))
    with pytest.raises(AttributeError):
        first.timestep = jnp.int32(1)


@pytest.mark.parametrize("num_mines,count", [(n, n) for n in range(9)] + [(10, 8)])
@pytest.mark.parametrize("remember", [False, True])
def test_every_clue_including_num_mines_and_eight_has_a_bin(num_mines, count, remember):
    env = ms.MineSweeper(4, 4, num_mines, remember=remember)
    neighbors = [(0, 0), (0, 1), (0, 2), (1, 0),
                 (1, 2), (2, 0), (2, 1), (2, 2)]
    locations = neighbors[:count] + [(3, c) for c in range(num_mines - count)]
    initial = _board_state(env, locations)
    np.testing.assert_array_equal(ms._count_neighbors(initial.mines), initial.neighbor_counts)
    obs, state, reward, done, _ = _step(env, initial, 5)  # center (1, 1)
    width = min(8, num_mines) + 2
    clue_obs = obs[:-1].reshape(16, width)[5] if remember else obs
    np.testing.assert_array_equal(clue_obs, np.eye(width)[count])
    assert int(state.last_value) == count
    assert float(reward) == pytest.approx(1 / (16 - num_mines))
    assert not done
    assert env.observation_space().contains(obs)
    np.testing.assert_array_equal(obs, env.get_obs(state))
    assert not np.array_equal(obs, env.get_obs(initial))


def test_safe_reveal_repeat_penalty_and_no_automatic_zero_expansion(env):
    initial = _board_state(env)
    obs, first, reward, done, info = _step(env, initial, jnp.float32(3))
    assert not done
    assert float(reward) == pytest.approx(1 / 14)
    assert int(first.last_value) == 0
    np.testing.assert_array_equal(obs, [1, 0, 0, 0])
    np.testing.assert_array_equal(np.flatnonzero(first.viewed), [3])
    assert not initial.viewed.any()  # functional update, no state mutation
    assert float(info["safe_cells_revealed"]) == 1
    assert float(info["known_safe_opportunity"]) == 0  # zero was not seen before action
    obs2, second, reward2, done2, info2 = _step(env, first, 3)
    assert not done2
    assert float(reward2) == pytest.approx(-0.5 / (14 - 2))
    assert int(second.timestep) == 2
    np.testing.assert_array_equal(second.viewed, first.viewed)
    np.testing.assert_array_equal(obs2, obs)
    assert float(info2["repeat_action"]) == 1
    assert float(info2["known_safe_opportunity"]) == 1
    assert float(info2["known_safe_taken"]) == 0
    assert float(info2["safe_cells_revealed"]) == 1
    assert float(info2["discount"]) == 1
    assert not info2["terminated"] and not info2["truncated"]


def test_first_click_can_hit_bomb_and_mines_never_count_as_revealed(env):
    initial = _board_state(env)
    obs, state, reward, done, info = _step(env, initial, 0)
    assert done and state.hit_mine and env.is_terminal(state)
    assert float(reward) == pytest.approx(-0.5 - 1 / 14)
    assert int(state.timestep) == 1
    assert not state.viewed.any()
    np.testing.assert_array_equal(state.mines, initial.mines)
    np.testing.assert_array_equal(obs, env.get_obs(state))
    assert info["terminated"] and not info["truncated"]
    assert float(info["success"]) == float(info["safe_cells_revealed"]) == 0
    assert float(info["discount"]) == float(env.discount(state, env.default_params)) == 0


def test_win_needs_only_safe_cells_and_total_reward_is_one(env):
    state = _board_state(env)
    actions = np.flatnonzero(~np.asarray(state.mines))
    rewards = []
    for step, action in enumerate(actions, 1):
        obs, state, reward, done, info = _step(env, state, int(action))
        rewards.append(float(reward))
        assert bool(done) == (step == 14)
        assert float(info["safe_cells_revealed"]) == step
        np.testing.assert_array_equal(obs, env.get_obs(state))
    assert sum(rewards) == pytest.approx(1.0)
    np.testing.assert_array_equal(state.viewed, ~state.mines)
    assert not state.viewed.all()  # unclicked mines must not block a win
    assert int(state.timestep) == 14 and env.is_terminal(state)
    assert float(info["success"]) == 1
    assert info["terminated"] and not info["truncated"]  # win takes precedence


@pytest.mark.parametrize("limit", [14, 4])
def test_timeout_counts_repeats_and_occurs_on_exact_limit(env, limit):
    params = env.default_params._replace(max_steps_in_episode=limit)
    state = _board_state(env)
    for step in range(1, limit + 1):
        _, state, reward, done, info = _step(env, state, 3, params)
        assert bool(done) == (step == limit)
        assert int(state.timestep) == step
        expected = 1 / 14 if step == 1 else -0.5 / 12
        assert float(reward) == pytest.approx(expected)
    assert state.viewed.sum() == 1
    assert env.is_terminal(state, params)
    assert not info["terminated"] and info["truncated"]
    assert float(info["success"]) == float(info["discount"]) == 0


def test_bomb_on_final_step_precedes_timeout_and_worst_return_is_minus_one(env):
    state = _board_state(env)
    rewards = []
    # One safe reveal, S-2 repeats, then a mine: the reference's worst return.
    for action in [3] * 13 + [0]:
        _, state, reward, done, info = _step(env, state, action)
        rewards.append(float(reward))
        assert bool(done) == (len(rewards) == 14)
    assert sum(rewards) == pytest.approx(-1.0)
    assert info["terminated"] and not info["truncated"]
    assert float(info["safe_cells_revealed"]) == 1
    assert float(info["success"]) == 0


def test_diagnostics_use_previous_zero_clues_and_do_not_wrap_at_edges(env):
    initial = _board_state(env)
    assert all(float(x) == 0 for x in env.diagnostics(initial, 3).values())
    _, state, _, _, _ = _step(env, initial, 3)  # top-right zero
    diagnostics = jax.jit(jax.vmap(env.diagnostics, in_axes=(None, 0)))(
        state, jnp.arange(16))
    assert set(diagnostics) == {"repeat_action", "known_safe_opportunity",
                                "known_safe_taken", "safe_cells_revealed"}
    np.testing.assert_array_equal(diagnostics["known_safe_opportunity"], 1)
    np.testing.assert_array_equal(diagnostics["safe_cells_revealed"], 1)
    np.testing.assert_array_equal(np.flatnonzero(diagnostics["repeat_action"]), [3])
    np.testing.assert_array_equal(np.flatnonzero(diagnostics["known_safe_taken"]), [2, 6, 7])
    for value in env.diagnostics(state, 2).values():
        assert value.shape == () and value.dtype == jnp.float32
    _, _, _, _, info = _step(env, state, 2)
    assert float(info["known_safe_opportunity"]) == float(info["known_safe_taken"]) == 1
    assert float(info["safe_cells_revealed"]) == 2


def test_exhausted_zero_opportunity_does_not_use_unseen_safe_cells(env):
    state = _board_state(env, ((0, 2), (2, 0)))
    # The corner is zero; its three neighbors have nonzero clues. After these
    # four queries many safe squares remain, but none is a neighbor of a seen zero.
    for action in [0, 1, 4, 5]:
        _, state, _, done, _ = _step(env, state, action)
        assert not done
    diagnostic = env.diagnostics(state, 15)
    assert float(diagnostic["known_safe_opportunity"]) == 0
    assert float(diagnostic["known_safe_taken"]) == 0
    assert float(diagnostic["safe_cells_revealed"]) == 4


@pytest.mark.parametrize("remember", [False, True])
def test_observations_and_diagnostics_do_not_leak_unobserved_board(remember):
    env = ms.MineSweeper(4, 4, 2, remember=remember)
    first = _board_state(env, ((0, 0), (3, 3)))
    second = _board_state(env, ((0, 0), (3, 2)))
    np.testing.assert_array_equal(env.get_obs(first), env.get_obs(second))
    for action in [3, 5, 3]:  # identical observable histories on distinct boards
        obs1, first, r1, d1, _ = _step(env, first, action)
        obs2, second, r2, d2, _ = _step(env, second, action)
        np.testing.assert_array_equal(obs1, obs2)
        assert r1 == r2 and d1 == d2
    for action in range(16):
        _assert_tree_equal(env.diagnostics(first, action), env.diagnostics(second, action))

    # Isolation check: even adversarial changes to privileged data must not
    # affect observation/history metrics. Preserve only actually seen clues.
    hidden_changed = first._replace(
        mines=~first.mines,
        neighbor_counts=jnp.where(first.viewed, first.neighbor_counts, 0),
    )
    np.testing.assert_array_equal(env.get_obs(first), env.get_obs(hidden_changed))
    for action in range(16):
        _assert_tree_equal(env.diagnostics(first, action),
                           env.diagnostics(hidden_changed, action))


def test_partial_observation_omits_location_but_history_control_retains_it(env):
    initial = _board_state(env)
    obs1, first, _, _, _ = _step(env, initial, 3)
    obs2, second, _, _, _ = _step(env, initial, 12)
    np.testing.assert_array_equal(obs1, obs2)  # both zero, different queried cells
    history_env = ms.MineSweeper(4, 4, 2, remember=True)
    assert not np.array_equal(history_env.get_obs(first), history_env.get_obs(second))


def test_remember_layout_unknowns_zero_clues_and_repeat_time():
    env = ms.MineSweeper(4, 4, 2, remember=True)
    state = _board_state(env)
    initial = env.get_obs(state)
    np.testing.assert_array_equal(initial[:-1].reshape(16, 4), np.tile([0, 0, 0, 1], (16, 1)))
    assert float(initial[-1]) == 0
    for action in [3, 5, 3]:
        obs, state, _, _, _ = _step(env, state, action)
    expected = np.full(16, 3)  # slot 3 is unknown, including both mine locations
    expected[3], expected[5] = 0, 1
    np.testing.assert_array_equal(obs[:-1].reshape(16, 4), np.eye(4)[expected])
    assert float(obs[-1]) == pytest.approx(3 / 14)
    assert state.viewed.sum() == 2
    np.testing.assert_array_equal(obs, env.get_obs(state))
    # Optional timeout params do not change observation reconstruction.
    np.testing.assert_array_equal(obs, env.get_obs(state, env.default_params._replace(
        max_steps_in_episode=4)))


@pytest.mark.parametrize("remember", [False, True])
@pytest.mark.parametrize("ending", ["bomb", "win", "timeout"])
def test_base_auto_reset_returns_consistent_obs_state_and_terminal_info(remember, ending):
    env = ms.MineSweeper(4, 4, 2, remember=remember)
    state = _board_state(env)
    action = 0
    if ending == "win":
        state = state._replace(
            viewed=(~state.mines).at[0, 3].set(False),
            timestep=jnp.int32(13), last_value=jnp.int32(1),
        )
        action = 3
    elif ending == "timeout":
        _, state, _, _, _ = _step(env, state, 3)
        state = state._replace(timestep=jnp.int32(13))
        action = 3
    key = jax.random.key(78)
    key_step, key_reset = jax.random.split(key)
    raw_obs, terminal, expected_reward, expected_done, expected_info = env.step_env(
        key_step, state, action, env.default_params)
    np.testing.assert_array_equal(raw_obs, env.get_obs(terminal))
    obs, reset, reward, done, info = env.step(key, state, action)
    assert done and expected_done
    _assert_tree_equal((reward, info), (expected_reward, expected_info))
    _assert_tree_equal((obs, reset), env.reset(key_reset))
    np.testing.assert_array_equal(obs, env.get_obs(reset))
    assert int(reset.timestep) == 0 and int(reset.last_value) == -1
    assert not reset.viewed.any() and not reset.hit_mine
    assert float(info["success"]) == (ending == "win")
    assert bool(info["truncated"]) == (ending == "timeout")
    assert float(info["safe_cells_revealed"]) == {"bomb": 0, "win": 14, "timeout": 1}[ending]


@pytest.mark.parametrize("remember", [False, True])
def test_jit_vmap_scan_mixed_live_and_auto_reset_episodes(remember):
    env = ms.MineSweeper(4, 4, 2, remember=remember)
    batch, steps = 8, 32
    keys = jax.random.split(jax.random.key(51), batch)
    _, initial = jax.jit(jax.vmap(env.reset))(keys)
    actions = jax.random.randint(jax.random.key(52), (steps, batch), 0, 16)
    board = initial.mines.reshape(batch, 16)
    # Force a mixed live/terminal first batch; later steps also exercise repeats.
    actions = actions.at[0].set(jnp.where(
        jnp.arange(batch) % 2 == 0, jnp.argmin(board, axis=1), jnp.argmax(board, axis=1)))
    step_keys = jax.random.split(jax.random.key(53), steps * batch).reshape(steps, batch)

    @jax.jit
    def rollout(state):
        def tick(state, inputs):
            key, action = inputs
            obs, state, reward, done, info = jax.vmap(env.step)(key, state, action)
            reconstructed = jax.vmap(env.get_obs)(state)
            consistent = jnp.all(obs == reconstructed, axis=-1)
            return state, (obs, reward, done, info, consistent, state.timestep, state.last_value)
        return jax.lax.scan(tick, state, (step_keys, actions))

    final, (obs, reward, done, info, consistent, timestep, last_value) = rollout(initial)
    assert obs.shape == (steps, batch, *env.observation_space().shape)
    assert reward.shape == done.shape == (steps, batch)
    np.testing.assert_array_equal(done[0], np.arange(batch) % 2 != 0)
    assert np.asarray(consistent).all()
    for leaf in jax.tree.leaves((obs, reward, info)):
        assert np.isfinite(leaf).all()
    for value in info.values():
        assert value.shape == (steps, batch)
    np.testing.assert_array_equal(np.asarray(timestep)[done], 0)
    np.testing.assert_array_equal(np.asarray(last_value)[done], -1)
    assert (np.asarray(timestep)[~np.asarray(done)] > 0).all()
    assert (np.asarray(timestep) < env.episode_length).all()
    np.testing.assert_array_equal(final.mines.sum(axis=(1, 2)), 2)
