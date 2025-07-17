## no tests in classical unit test sense, since RL algos are hard to test that way...

from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete

from lambda_imitation import IQLearn


class SimpleGridWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([2]))
        self.action_space = gym.spaces.Discrete(2)
        self.pos = 0

    def reset(self, **kwargs):
        self.pos = 0
        return np.array([self.pos], dtype=np.float32), {}

    def step(self, action):
        if action == 0:
            self.pos -= 1
        else:
            self.pos += 1

        if self.pos > 2:
            self.pos = 2

        terminated = self.pos == 2 or self.pos == -1
        truncated = False
        reward = 1 if self.pos == 2 else (-1 if self.pos == -1 else 0)

        if self.pos < 0:
            self.pos = 0

        return np.array([self.pos], dtype=np.float32), reward, terminated, truncated, {}


class SimpleGridWorldContinuous(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([2]))
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]))
        self.pos = 0

    def reset(self, **kwargs):
        self.pos = 0
        return np.array([self.pos], dtype=np.float32), {}

    def step(self, action):
        if action[0] < 0:
            self.pos -= 1
        else:
            self.pos += 1

        if self.pos > 2:
            self.pos = 2

        terminated = self.pos == 2 or self.pos == -1
        truncated = False
        reward = 1 if self.pos == 2 else (-1 if self.pos == -1 else 0)

        if self.pos < 0:
            self.pos = 0

        return np.array([self.pos], dtype=np.float32), reward, terminated, truncated, {}


@pytest.mark.slow
def test_sac_simple_continuous_gridworld():
    # assemble
    env = SimpleGridWorldContinuous()
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "policy_lr": 0.1,
            "q_lr": 0.1,
            "tensorboard_dir": None,
        },
    )

    # act
    iqlearn.sac_learn(10, progress="none")

    # assert
    steps = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
        if steps > 10:
            assert False

    assert reward == 1
    assert steps == 2


@pytest.mark.slow
def test_sac_simple_gridworld():
    # assemble
    env = SimpleGridWorld()
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "policy_lr": 0.1,
            "q_lr": 0.1,
            "tensorboard_dir": None,
        },
    )

    # act
    iqlearn.sac_learn(100, progress="none")

    # assert
    steps = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
        if steps > 10:
            assert False

    assert reward == 1
    assert steps == 2
