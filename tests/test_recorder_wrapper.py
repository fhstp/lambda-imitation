from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Dict, Discrete

from lambda_imitation import RecorderWrapper


# assertion helpers
def assert_dicts_equal_one_sided(dict1, dict2):
    for key in dict1:
        assert key in dict2
        if type(dict1[key]) == OrderedDict or type(dict1[key]) == dict:
            assert type(dict2[key]) == OrderedDict or type(dict2[key]) == dict
            assert_dicts_equal(dict1[key], dict2[key])
        elif type(dict1[key]) == np.ndarray:
            assert type(dict2[key]) == np.ndarray
            assert (dict1[key] == dict2[key]).all()
        else:
            assert dict1[key] == dict2[key]


def assert_dicts_equal(dict1, dict2):
    assert_dicts_equal_one_sided(dict1, dict2)
    assert_dicts_equal_one_sided(dict2, dict1)


# nested dict env for testing dict envs
class NestedDictEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Dict(
            {
                "box": Box(-1, 1, (2, 2)),
                "discrete": Discrete(12),
                "another_discrete": Discrete(11),
                "nested_dict": Dict(
                    {"nested_box": Box(0, 1, (1, 3)), "nested_discrete": Discrete(2)}
                ),
            }
        )
        self.action_space = self.observation_space

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), np.random.rand(1)[0], False, False, {}


def test_nested_dict_env_records():
    # assemble
    env = RecorderWrapper(NestedDictEnvironment(), 0.99, 10000)

    # act
    obs1, _ = env.reset()
    action1 = env.action_space.sample()
    obs2, reward1, terminated1, truncated1, _ = env.step(action1)
    action2 = env.action_space.sample()
    obs3, reward2, terminated2, truncated2, _ = env.step(action2)

    # assert
    assert_dicts_equal(obs1, env.get_observation_at(0))
    assert_dicts_equal(env.actions[0], env.get_action_at(0))
    assert env.rewards[0] == pytest.approx(reward1)
    assert env.terminated[0] == terminated1
    assert env.truncated[0] == truncated1
    assert_dicts_equal(obs2, env.get_observation_at(1))
    assert_dicts_equal(env.actions[1], env.get_action_at(1))
    assert env.rewards[1] == pytest.approx(reward2)
    assert env.terminated[1] == terminated2
    assert env.truncated[1] == truncated2
    assert_dicts_equal(obs3, env.get_observation_at(2))


def test_mountaincar_records():
    # assemble
    env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 10000)

    # act
    obs1, _ = env.reset()
    action1 = env.action_space.sample()
    obs2, reward1, terminated1, truncated1, _ = env.step(action1)
    action2 = env.action_space.sample()
    obs3, reward2, terminated2, truncated2, _ = env.step(action2)

    # assert
    assert (obs1 == env.observations[0]).all()
    assert env.actions[0] == action1
    assert env.rewards[0] == reward1
    assert env.terminated[0] == terminated1
    assert env.truncated[0] == truncated1
    assert (obs2 == env.observations[1]).all()
    assert env.actions[1] == action2
    assert env.rewards[1] == reward2
    assert env.terminated[1] == terminated2
    assert env.truncated[1] == truncated2
    assert env.actions[1] == action2
    assert (obs3 == env.observations[2]).all()


def test_pendulum_records():
    # assemble
    env = RecorderWrapper(gym.make("Pendulum-v1"), 0.99, 10000)

    # act
    obs1, _ = env.reset()
    action1 = env.action_space.sample()
    obs2, reward1, terminated1, truncated1, _ = env.step(action1)
    action2 = env.action_space.sample()
    obs3, reward2, terminated2, truncated2, _ = env.step(action2)

    # assert
    assert (obs1 == env.observations[0]).all()
    assert env.actions[0] == action1
    assert env.rewards[0] == pytest.approx(reward1)
    assert env.terminated[0] == terminated1
    assert env.truncated[0] == truncated1
    assert (obs2 == env.observations[1]).all()
    assert env.actions[1] == action2
    assert env.rewards[1] == pytest.approx(reward2)
    assert env.terminated[1] == terminated2
    assert env.truncated[1] == truncated2
    assert env.actions[1] == action2
    assert (obs3 == env.observations[2]).all()


def test_circularity():
    # assemble
    env = RecorderWrapper(gym.make("Pendulum-v1"), 0.99, 5)

    # act
    obs, _ = env.reset()
    for i in range(7):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

    # assert
    assert env.pos % 5 == 2
    assert (obs == env.observations[2]).all()
    assert reward == pytest.approx(env.rewards[1])
    assert (action == env.actions[1]).all()
    assert terminated == env.terminated[1]
    assert truncated == env.truncated[1]


def test_reset_overwrite():
    # assemble
    env = RecorderWrapper(gym.make("Pendulum-v1"), 0.99, 5)

    # act
    obs, _ = env.reset()
    for i in range(4):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
    obs, _ = env.reset()
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

    # assert
    assert (obs == env.observations[2]).all()
    assert terminated == env.terminated[1]
    assert truncated == env.truncated[1]

    # act again
    new_obs, _ = env.reset()
    while (obs == new_obs).any():
        new_obs, _ = env.reset()

    # assert difference
    assert (new_obs == env.observations[2]).all()
    assert (obs != env.observations[2]).any()


def test_save_load(tmpdir):
    # assemble
    env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 10000)
    obs1, _ = env.reset()
    action1 = env.action_space.sample()
    obs2, reward1, terminated1, truncated1, _ = env.step(action1)
    action2 = env.action_space.sample()
    obs3, reward2, terminated2, truncated2, _ = env.step(action2)

    # act
    filename = tmpdir.join("tmp")
    env.save_buffer(filename)
    other_env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 1)
    other_env.load_buffer(filename)

    # assert
    assert (obs1 == other_env.observations[0]).all()
    assert other_env.actions[0] == action1
    assert other_env.rewards[0] == reward1
    assert other_env.terminated[0] == terminated1
    assert other_env.truncated[0] == truncated1
    assert (obs2 == other_env.observations[1]).all()
    assert other_env.actions[1] == action2
    assert other_env.rewards[1] == reward2
    assert other_env.terminated[1] == terminated2
    assert other_env.truncated[1] == truncated2
    assert other_env.actions[1] == action2
    assert (obs3 == other_env.observations[2]).all()


def test_return_calc_mountaincar():
    # assemble
    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 10000)

    # act
    env.reset()
    _, reward1, _, _, _ = env.step(env.action_space.sample())
    _, reward2, _, _, _ = env.step(env.action_space.sample())
    _, reward3, _, _, _ = env.step(env.action_space.sample())
    _, reward4, _, _, _ = env.step(env.action_space.sample())
    env.reset()  # triggers return calc

    # assert
    assert env.returns[0] == -1 - gamma - gamma**2 - gamma**3
    assert env.returns[1] == -1 - gamma - gamma**2
    assert env.returns[2] == -1 - gamma
    assert env.returns[3] == -1


def test_buffer_exactly_long_enough_for_return():
    # assemble
    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 3)

    # act
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())

    # assert
    env.reset()  # must not raise error


def test_buffer_too_short_for_return():
    # assemble
    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 3)

    # act
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())

    # assert
    with pytest.raises(AssertionError):
        env.reset()


def test_hidden_state_calculation():
    # assemble
    def hidden_state_net(state, action, hidden_state):
        hidden_state = hidden_state.copy()
        hidden_state[0] += 1
        return hidden_state

    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 100, 2, hidden_state_net)

    # act
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())

    # assert
    assert env.hidden_states[0][0] == 1
    assert env.hidden_states[0][1] == 0
    assert env.hidden_states[1][0] == 2
    assert env.hidden_states[1][1] == 0
    assert env.hidden_states[2][0] == 0  # not set yet
    assert env.hidden_states[2][1] == 0

    # act again after reset
    env.reset()
    env.step(env.action_space.sample())

    # assert nothing gets overwritten
    assert env.hidden_states[0][0] == 1
    assert env.hidden_states[0][1] == 0
    assert env.hidden_states[1][0] == 2
    assert env.hidden_states[1][1] == 0
    assert env.hidden_states[2][0] == 1
    assert env.hidden_states[2][1] == 0


def test_hidden_state_recalculation():
    # assemble
    def hidden_state_net(state, action, hidden_state):
        hidden_state = hidden_state.copy()
        hidden_state[0] += 1
        return hidden_state

    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 100)

    # act
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())

    # assert no hidden states
    assert env.hidden_states.shape == (100, 0)

    # act again - recalculate
    env.setup_hidden_states(2, hidden_state_net)
    env.recalculate_hidden_states()

    # assert hidden states as above when directly calculating
    assert env.hidden_states[0][0] == 1
    assert env.hidden_states[0][1] == 0
    assert env.hidden_states[1][0] == 2
    assert env.hidden_states[1][1] == 0
    assert env.hidden_states[2][0] == 0
    assert env.hidden_states[2][1] == 0


def test_hidden_state_recalculation_two_episodes():
    # assemble
    def hidden_state_net(state, action, hidden_state):
        hidden_state = hidden_state.copy()
        hidden_state[0] += 1
        return hidden_state

    gamma = 0.99
    env = RecorderWrapper(gym.make("MountainCar-v0"), gamma, 100)

    # act
    env.reset()
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    env.truncated[1] = True  # mock truncation with reset
    env.reset()
    env.step(env.action_space.sample())

    # assert no hidden states
    assert env.hidden_states.shape == (100, 0)

    # act again - recalculate
    env.setup_hidden_states(2, hidden_state_net)
    env.recalculate_hidden_states()

    # assert hidden states as above when directly calculating
    assert env.hidden_states[0][0] == 1
    assert env.hidden_states[0][1] == 0
    assert env.hidden_states[1][0] == 2
    assert env.hidden_states[1][1] == 0
    assert env.hidden_states[2][0] == 1
    assert env.hidden_states[2][1] == 0


def test_sample_generate_indices_not_full():
    # assemble
    env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 7)
    env.reset()
    for i in range(5):
        env.step(env.action_space.sample())

    # act
    inds = env._generate_indices(10)

    # assert
    assert env.pos == 5
    assert inds.shape[0] == 10
    assert np.max(inds) < 5


def test_sample_generate_indices_exactly_full():
    # assemble
    env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 7)
    env.reset()
    for i in range(6):
        env.step(env.action_space.sample())

    # act assert in loop for probabilistic testing
    for _ in range(100):
        # act
        inds = env._generate_indices(10)

        # assert
        assert np.max(inds) < 6


def test_sample_generate_indices_over_full():
    # assemble
    env = RecorderWrapper(gym.make("MountainCar-v0"), 0.99, 7)
    env.reset()
    for i in range(9):
        env.step(env.action_space.sample())

    # act assert in loop for probabilistic testing
    for _ in range(100):
        # act
        env.step(env.action_space.sample())
        inds = env._generate_indices(10)

        # assert
        assert (env.pos % env.buffer_size) not in inds

class SimpleGridWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.Discrete(2)
        self.pos = 0

    def reset(self, **kwargs):
        self.pos = 0
        return self.pos, {}

    def step(self, action):
        if action == 0:
            self.pos -= 1
        else:
            self.pos += 1

        if self.pos < 0:
            self.pos = 0
        if self.pos > 2:
            self.pos = 2

        terminated = self.pos == 2
        truncated = False
        reward = 1 if self.pos == 2 else 0

        return self.pos, reward, terminated, truncated, {}

def test_sample():
    # assemble
    gamma = 0.99
    env = RecorderWrapper(SimpleGridWorld(), gamma, 1000)
    
    # act
    obs0, _ = env.reset()
    action0 = 1
    obs1, reward0, terminated0, truncated0, _ = env.step(action0)
    action1 = 0
    obs2, reward1, terminated1, truncated1, _ = env.step(action1)
    action2 = 1
    obs3, reward2, terminated2, truncated2, _ = env.step(action2)
    action3 = 1
    obs4, reward3, terminated3, truncated3, _ = env.step(action3)
    action4 = 1
    observations, next_observations, actions, rewards, returns, terminated, truncated, hidden_states = env._sample(np.array([1,0,2,2, 3]))

    # assert
    assert observations[0] == obs1
    assert next_observations[0] == obs2
    assert actions[0] == action1
    assert rewards[0] == reward1
    assert returns[0] == pytest.approx(gamma**2)
    assert terminated[0] == terminated1
    assert truncated[0] == truncated1

    assert observations[1] == obs0
    assert next_observations[1] == obs1
    assert actions[1] == action0
    assert rewards[1] == reward0
    assert returns[1] == pytest.approx(gamma**3)
    assert terminated[1] == terminated0
    assert truncated[1] == truncated0

    assert observations[2] == obs2
    assert next_observations[2] == obs3
    assert actions[2] == action2
    assert rewards[2] == reward2
    assert returns[2] == pytest.approx(gamma**1)
    assert terminated[2] == terminated2
    assert truncated[2] == truncated2

    assert observations[3] == obs2
    assert next_observations[3] == obs3
    assert actions[3] == action2
    assert rewards[3] == reward2
    assert returns[3] == pytest.approx(gamma**1)
    assert terminated[3] == terminated2
    assert truncated[3] == truncated2

    assert observations[4] == obs3
    assert actions[4] == action3
    assert rewards[4] == reward3
    assert returns[4] == reward3
    assert terminated[4] == terminated3
    assert truncated[4] == truncated3

