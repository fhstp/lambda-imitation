"""POPGym rule parity, observation privacy and native gymnax/JAX integration.

The small list-based oracle below independently enumerates flip attempts,
including duplicates; no POPGym installation is needed for these tests.
"""

import importlib.util
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gymnax.environments import environment, spaces

from lambda_imitation.utils import env_spec_from_gymnax


_path = Path(__file__).resolve().parents[1] / "examples/lambda-envs/_concentration_env.py"
_spec = importlib.util.spec_from_file_location("concentration_env", _path)
concentration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(concentration)
Concentration = concentration.Concentration

LAYOUT = [0, 1, 0, 2, 1, 2]
KEY = jax.random.key(0)


@pytest.fixture(scope="module", params=[False, True], ids=["partial", "history"])
def env(request):
    return Concentration(num_cards=6, num_types=3, remember=request.param)


def layout_state(env, cards=LAYOUT):
    _, state = env.reset(KEY)
    return state._replace(cards=jnp.asarray(cards, dtype=jnp.int32))


def advance(env, state, actions, params=None):
    params = env.default_params if params is None else params
    for action in actions:
        obs, state, reward, done, info = env.step_env(KEY, state, action, params)
    return obs, state, reward, done, info


def assert_visible(env, obs, state, labels):
    width = env.num_cards * (env.num_types + 1)
    expected = np.eye(env.num_types + 1, dtype=np.float32)[labels].reshape(-1)
    np.testing.assert_array_equal(obs[:width], expected)
    np.testing.assert_array_equal(state.visible, labels)
    np.testing.assert_array_equal(obs, env.get_obs(state))
    np.testing.assert_array_equal(obs, env.get_obs(state, env.default_params))
    assert obs.shape == env.observation_space().shape
    assert obs.dtype == jnp.float32
    assert env.observation_space().contains(obs)


def assert_same_tree(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("remember, width", [(False, 728), (True, 1560)])
def test_default_api_spaces_and_immutable_state(remember, width):
    env = Concentration(remember=remember)
    assert isinstance(env, environment.Environment)
    assert Concentration.step is environment.Environment.step
    assert Concentration.reset is environment.Environment.reset
    assert type(env.episode_length) is int
    assert env.episode_length == env.default_params.max_steps_in_episode == 104
    assert env.num_actions == 52
    assert env.num_types == 13
    assert isinstance(env.action_space(), spaces.Discrete)
    assert env.action_space().n == 52
    assert isinstance(env.observation_space(), spaces.Box)
    assert env.observation_space().shape == (width,)
    spec = env_spec_from_gymnax(env, env.default_params)
    assert spec.is_discrete and spec.action_dim == 52 and spec.obs_shape == (width,)

    obs, state = env.reset(KEY)
    assert isinstance(state, concentration.ConcentrationState)
    assert_visible(env, obs, state, [13] * 52)
    assert env.state_space().contains(state)
    assert state.cards.dtype == state.visible.dtype == state.last_seen.dtype == jnp.int32
    assert state.face_up.dtype == state.in_play.dtype == state.seen.dtype == jnp.bool_
    assert state.timestep.shape == () and state.timestep.dtype == jnp.int32
    with pytest.raises(AttributeError):
        state.timestep = jnp.int32(1)


@pytest.mark.parametrize("num_cards, num_types", [
    (0, 1), (-2, 1), (3, 1), (6, 2), (8, 3), (4, 0), (4, -1), (4, 4),
    (4.5, 2), (4, 2.0),
])
def test_rejects_decks_that_cannot_be_balanced_even_pairs(num_cards, num_types):
    with pytest.raises(ValueError):
        Concentration(num_cards=num_cards, num_types=num_types)


@pytest.mark.parametrize("num_cards, num_types", [(52, 13), (8, 2), (6, 3), (2, 1)])
def test_resets_are_shuffled_balanced_and_reproducible(num_cards, num_types):
    env = Concentration(num_cards, num_types)
    keys = jax.random.split(KEY, 16)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    expected = np.repeat(np.arange(num_types), num_cards // num_types)
    np.testing.assert_array_equal(np.sort(states.cards, axis=1),
                                  np.broadcast_to(expected, states.cards.shape))
    if num_types > 1:
        assert np.unique(np.asarray(states.cards), axis=0).shape[0] > 1
    assert not np.any(states.face_up)
    assert not np.any(states.in_play)
    assert not np.any(states.seen)
    assert np.all(states.timestep == 0)
    assert np.all(states.last_seen == num_types)
    assert np.all(states.visible == num_types)
    expected_obs = np.eye(num_types + 1, dtype=np.float32)[[num_types] * num_cards]
    np.testing.assert_array_equal(obs, np.broadcast_to(expected_obs.reshape(-1), obs.shape))
    assert_same_tree(env.reset(keys[0]), (obs[0], jax.tree.map(lambda x: x[0], states)))


def test_first_flip_and_mismatch_reveal_survives_get_obs_until_next_action(env):
    start = layout_state(env)
    obs, first, reward, done, info = advance(env, start, [0])
    assert float(reward) == 0.0 and not done and float(info["discount"]) == 1.0
    assert_visible(env, obs, first, [0, 3, 3, 3, 3, 3])
    np.testing.assert_array_equal(first.in_play, [True, False, False, False, False, False])

    obs, mismatch, reward, done, _ = advance(env, first, [1])
    assert float(reward) == pytest.approx(-2 / 12)
    assert not done and not np.any(mismatch.in_play) and not np.any(mismatch.face_up)
    assert int(mismatch.timestep) == 2
    assert_visible(env, obs, mismatch, [0, 1, 3, 3, 3, 3])

    obs, third, reward, done, _ = advance(env, mismatch, [3])
    assert float(reward) == 0 and not done
    assert_visible(env, obs, third, [3, 3, 3, 2, 3, 3])
    np.testing.assert_array_equal(third.seen, [True, True, False, True, False, False])
    np.testing.assert_array_equal(third.last_seen, [0, 1, 3, 2, 3, 3])
    assert not np.any(start.seen) and not np.any(start.in_play)
    np.testing.assert_array_equal(start.cards, LAYOUT)


def test_matching_pair_is_permanent_and_reward_is_normalized(env):
    obs, state, reward, done, _ = advance(env, layout_state(env), [0, 2])
    assert float(reward) == pytest.approx(1 / 3) and not done
    np.testing.assert_array_equal(state.face_up, [True, False, True, False, False, False])
    assert not np.any(state.in_play)
    assert_visible(env, obs, state, [0, 3, 0, 3, 3, 3])

    obs, state, reward, done, _ = advance(env, state, [3])
    assert float(reward) == 0 and not done
    assert_visible(env, obs, state, [0, 3, 0, 2, 3, 3])
    assert int(env.diagnostics(state, 5)["pairs_matched"]) == 1


def test_repeating_the_first_card_counts_two_attempted_flips(env):
    _, first, _, _, _ = advance(env, layout_state(env), [0])
    assert env.diagnostics(first, 0)["invalid_action"]
    obs, state, reward, done, _ = advance(env, first, [0])
    assert float(reward) == pytest.approx(-2 / 12) and not done
    assert not np.any(state.face_up) and not np.any(state.in_play)
    assert_visible(env, obs, state, [0, 3, 3, 3, 3, 3])
    obs, state, reward, _, _ = advance(env, state, [1])
    assert float(reward) == 0
    assert_visible(env, obs, state, [3, 1, 3, 3, 3, 3])


@pytest.mark.parametrize("second", [False, True], ids=["first-selection", "second-selection"])
@pytest.mark.parametrize("matched_action", [0, 2])
def test_selecting_matched_card_penalizes_attempts_and_clears_phase(env, second, matched_action):
    _, state, _, _, _ = advance(env, layout_state(env), [0, 2] + ([1] if second else []))
    assert env.diagnostics(state, matched_action)["invalid_action"]
    obs, state, reward, done, _ = advance(env, state, [matched_action])
    assert float(reward) == pytest.approx(-(2 if second else 1) / 12)
    assert not done and not np.any(state.in_play)
    np.testing.assert_array_equal(state.face_up, [True, False, True, False, False, False])
    assert_visible(env, obs, state, [0, 1 if second else 3, 0, 3, 3, 3])
    # The invalid action finished the attempt; an unmatched card now starts anew.
    obs, state, reward, _, _ = advance(env, state, [1])
    assert float(reward) == 0 and state.in_play[1]
    assert_visible(env, obs, state, [0, 1, 0, 3, 3, 3])


def test_perfect_play_finishes_with_unit_return(env):
    state = layout_state(env)
    total = 0.0
    for t, action in enumerate([0, 2, 1, 4, 3, 5], start=1):
        obs, state, reward, done, info = advance(env, state, [action])
        total += float(reward)
        assert bool(done) == (t == 6)
        assert bool(env.is_terminal(state)) == (t == 6)
        assert float(info["discount"]) == (0.0 if t == 6 else 1.0)
    assert total == pytest.approx(1.0)
    assert np.all(state.face_up) and not np.any(state.in_play)
    assert int(env.diagnostics(state, 0)["pairs_matched"]) == 3
    assert_visible(env, obs, state, LAYOUT)


@pytest.mark.parametrize("termination", ["win", "timeout"])
def test_terminal_step_uses_gymnax_autoreset_and_preserves_reward(env, termination):
    actions = [0, 2, 1, 4, 3, 5] if termination == "win" else [0] * 12
    state = layout_state(env)
    for action in actions[:-1]:
        _, state, _, done, _ = advance(env, state, [action])
        assert not done
    obs_terminal, terminal, reward, done, info = advance(env, state, actions[-1:])
    assert done and float(info["discount"]) == 0.0
    assert int(terminal.timestep) == len(actions)
    assert_visible(env, obs_terminal, terminal,
                   LAYOUT if termination == "win" else [0, 3, 3, 3, 3, 3])
    expected_reward = 1 / 3 if termination == "win" else -2 / 12
    assert float(reward) == pytest.approx(expected_reward)

    key = jax.random.key(123)
    obs, reset, reward_auto, done_auto, info_auto = env.step(key, state, actions[-1])
    assert done_auto and float(reward_auto) == float(reward)
    assert float(info_auto["discount"]) == 0.0
    _, reset_key = jax.random.split(key)
    assert_same_tree((obs, reset), env.reset(reset_key))
    assert_visible(env, obs, reset, [3] * 6)
    assert int(reset.timestep) == 0
    assert not np.any(reset.seen) and not np.any(reset.face_up) and not np.any(reset.in_play)
    assert np.all(reset.last_seen == 3)


def test_custom_gymnax_time_limit_does_not_change_reward_scale(env):
    params = env.default_params._replace(max_steps_in_episode=3)
    _, state, reward, done, _ = advance(env, layout_state(env), [0, 0], params)
    assert not done and float(reward) == pytest.approx(-2 / 12)
    obs, state, reward, done, _ = env.step_env(KEY, state, 1, params)
    assert done and float(reward) == 0 and int(state.timestep) == 3
    assert_visible(env, obs, state, [3, 1, 3, 3, 3, 3])


def test_history_control_keeps_visible_board_and_only_adds_recorded_information():
    partial = Concentration(6, 3)
    history = Concentration(6, 3, remember=True)
    obs, state, _, _, _ = advance(partial, layout_state(partial), [0, 1, 3])
    remembered = history.get_obs(state)
    width = 6 * 4
    np.testing.assert_array_equal(remembered[:width], obs)
    np.testing.assert_array_equal(
        remembered[width:2 * width],
        np.eye(4, dtype=np.float32)[[0, 1, 3, 2, 3, 3]].reshape(-1))
    np.testing.assert_array_equal(remembered[2 * width:2 * width + 6], np.zeros(6))
    np.testing.assert_array_equal(remembered[2 * width + 6:], [0, 0, 0, 1, 0, 0])
    assert_visible(partial, obs, state, [3, 3, 3, 2, 3, 3])

    # Matched positions and the pending first selection are separate public data.
    _, state, _, _, _ = advance(partial, layout_state(partial), [0, 2, 1])
    remembered = history.get_obs(state)
    np.testing.assert_array_equal(remembered[-12:-6], [1, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(remembered[-6:], [0, 1, 0, 0, 0, 0])
    # A failed pair clears the phase even though both ranks remain in the table.
    _, state, _, _, _ = advance(partial, state, [3])
    assert not np.any(history.get_obs(state)[-6:])


def test_unseen_rank_permutation_cannot_change_observations_or_diagnostics(env):
    left = layout_state(env)
    right = layout_state(env, [0, 1, 1, 2, 0, 2])  # swap only unrevealed positions 2, 4
    for action in [0, 1, 3]:
        obs_left, left, reward_left, done_left, _ = advance(env, left, [action])
        obs_right, right, reward_right, done_right, _ = advance(env, right, [action])
        np.testing.assert_array_equal(obs_left, obs_right)
        assert reward_left == reward_right and done_left == done_right
        for proposed in range(env.num_actions):
            d_left = env.diagnostics(left, proposed)
            assert_same_tree(d_left, env.diagnostics(right, proposed))
            # All potential partners remain unseen on these histories.
            assert not d_left["known_match_opportunity"]
            assert not d_left["known_match_taken"]
    # Observation and diagnostics depend on cached visible history, not cards.
    poisoned = left._replace(cards=jnp.full_like(left.cards, -999))
    np.testing.assert_array_equal(env.get_obs(left), env.get_obs(poisoned))
    assert_same_tree(env.diagnostics(left, 5), env.diagnostics(poisoned, 5))


def test_known_match_diagnostics_use_history_exclude_self_matched_and_unseen():
    env = Concentration(8, 2)
    start = layout_state(env, [0, 1, 0, 1, 0, 1, 0, 1])
    # Two mismatches reveal matching ranks, but there is no pending first card.
    _, no_first, _, _, _ = advance(env, start, [0, 1, 2, 3])
    assert not env.diagnostics(no_first, 0)["known_match_opportunity"]

    _, state, _, _, _ = advance(env, start, [0, 1, 2])
    metrics = jax.jit(jax.vmap(lambda a: env.diagnostics(state, a)))(jnp.arange(8))
    assert set(metrics) == {"known_match_opportunity", "known_match_taken",
                            "invalid_action", "pairs_matched"}
    np.testing.assert_array_equal(metrics["known_match_opportunity"], [True] * 8)
    np.testing.assert_array_equal(metrics["known_match_taken"],
                                  [True, False, False, False, False, False, False, False])
    np.testing.assert_array_equal(metrics["invalid_action"],
                                  [False, False, True, False, False, False, False, False])
    assert np.all(metrics["pairs_matched"] == 0)
    assert metrics["pairs_matched"].dtype == jnp.int32
    for key in ("known_match_opportunity", "known_match_taken", "invalid_action"):
        assert metrics[key].dtype == jnp.bool_
    for value in env.diagnostics(state, 0).values():
        assert isinstance(value, jax.Array) and value.shape == ()
    # Positions 4/6 really match, but were unseen: taking them is not a known match.
    assert not metrics["known_match_taken"][4] and not metrics["known_match_taken"][6]

    _, state, _, _, _ = advance(env, state, [0, 4])
    # Recorded matching ranks at 0/2 are now permanent matches and unavailable.
    assert not env.diagnostics(state, 6)["known_match_opportunity"]
    assert env.diagnostics(state, 0)["invalid_action"]
    assert int(env.diagnostics(state, 6)["pairs_matched"]) == 1
    _, state, _, _, _ = advance(env, state, [6])
    assert int(env.diagnostics(state, 0)["pairs_matched"]) == 2


def test_jitted_vmapped_step_handles_mixed_phases_and_terminal_autoreset(env):
    start = layout_state(env)
    first = advance(env, start, [0])[1]
    matched = advance(env, start, [0, 2])[1]
    almost_won = advance(env, start, [0, 2, 1, 4, 3])[1]
    states = jax.tree.map(lambda *xs: jnp.stack(xs), start, first, first, matched, almost_won)
    keys = jax.random.split(KEY, 5)
    # Scalar float32 action indices are used by the package's discrete actor.
    actions = jnp.array([1, 2, 0, 0, 5], dtype=jnp.float32)
    obs, next_states, rewards, dones, info = jax.jit(jax.vmap(env.step))(keys, states, actions)
    np.testing.assert_allclose(rewards, [0, 1 / 3, -2 / 12, -1 / 12, 1 / 3])
    np.testing.assert_array_equal(dones, [False, False, False, False, True])
    np.testing.assert_array_equal(info["discount"], [1, 1, 1, 1, 0])
    np.testing.assert_array_equal(next_states.timestep, [1, 2, 2, 3, 0])
    np.testing.assert_array_equal(obs, jax.jit(jax.vmap(env.get_obs))(next_states))
    assert not np.any(next_states.seen[-1])
    assert not np.any(next_states.face_up[-1])


def reference_trace(cards, actions, num_types):
    """List-based POPGym flip/resolve rules, independent of JAX mask arithmetic."""
    face_up, in_play, seen = set(), [], {}
    visible_trace, reward_trace, done_trace, up_trace, play_trace, history_trace = [], [], [], [], [], []
    for t, action in enumerate(actions, start=1):
        in_play.append(action)
        visible = [num_types] * len(cards)
        for position in list(face_up) + in_play:
            visible[position] = cards[position]
            seen[position] = cards[position]
        reward = 0.0
        if any(position in face_up for position in in_play):
            reward = -len(in_play) / (2 * len(cards))
            in_play = []
        elif len(in_play) == 2:
            first, second = in_play
            if first != second and cards[first] == cards[second]:
                reward = 1 / (len(cards) // 2)
                face_up.update(in_play)
            else:
                reward = -2 / (2 * len(cards))
            in_play = []
        visible_trace.append(visible)
        reward_trace.append(reward)
        done_trace.append(len(face_up) == len(cards) or t >= 2 * len(cards))
        up_trace.append([i in face_up for i in range(len(cards))])
        play_trace.append([i in in_play for i in range(len(cards))])
        history_trace.append([seen.get(i, num_types) for i in range(len(cards))])
    return (visible_trace, reward_trace, done_trace, up_trace, play_trace, history_trace)


@pytest.mark.parametrize("remember", [False, True])
def test_all_four_flip_histories_match_independent_python_rules(remember):
    env = Concentration(4, 2, remember=remember)
    cards = [0, 1, 0, 1]
    start = layout_state(env, cards)
    # 256 full histories include duplicate flips, mismatches, permanent pairs,
    # matched-card penalties in both phases, and all perfect-play terminations.
    actions = np.array(list(itertools.product(range(4), repeat=4)), dtype=np.int32)

    def rollout(sequence):
        def step(state, action):
            obs, state, reward, done, _ = env.step_env(KEY, state, action, env.default_params)
            return state, (obs, state, reward, done)
        return jax.lax.scan(step, start, sequence)[1]

    obs, states, rewards, dones = jax.jit(jax.vmap(rollout))(jnp.asarray(actions))
    expected = [reference_trace(cards, sequence, 2) for sequence in actions.tolist()]
    visible, reward, done, face_up, in_play, history = map(np.array, zip(*expected))
    np.testing.assert_array_equal(states.visible, visible)
    np.testing.assert_allclose(rewards, reward)
    np.testing.assert_array_equal(dones, done)
    np.testing.assert_array_equal(states.face_up, face_up)
    np.testing.assert_array_equal(states.in_play, in_play)
    np.testing.assert_array_equal(states.last_seen, history)
    np.testing.assert_array_equal(states.seen, history != 2)
    expected_obs = np.eye(3, dtype=np.float32)[visible].reshape(len(actions), 4, -1)
    if remember:
        expected_history = np.eye(3, dtype=np.float32)[history].reshape(len(actions), 4, -1)
        expected_obs = np.concatenate((expected_obs, expected_history, face_up, in_play), axis=-1)
    np.testing.assert_array_equal(obs, expected_obs)
    np.testing.assert_array_equal(obs, jax.jit(jax.vmap(jax.vmap(env.get_obs)))(states))
