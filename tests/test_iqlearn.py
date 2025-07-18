## no tests in classical unit test sense, since RL algos are hard to test that way...

import warnings
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN, PPO

from lambda_imitation import IQLearn
from lambda_imitation.recorder_wrapper import RecorderWrapper


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
            "use_targets": False,
            "policy_lr": 0.1,
            "q_lr": 0.1,
            "target_entropy": 0.2,
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


@pytest.mark.slow
def test_sac_cartpole():
    # assemble
    env = gym.make("CartPole-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    steps = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), True)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 150


@pytest.mark.slow
def test_sac_pendulum():
    # assemble
    env = gym.make("Pendulum-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    undiscounted_return = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        undiscounted_return += reward  # type: ignore
        if terminated or truncated:
            break

    print(undiscounted_return)
    assert undiscounted_return > -500


@pytest.mark.slow
def test_iqlearn_mountaincar():
    # assemble
    checkpoint = load_from_hub(
        repo_id="sb3/dqn-MountainCar-v0",
        filename="dqn-MountainCar-v0.zip",
    )
    env = gym.make("MountainCar-v0")
    with warnings.catch_warnings(
        action="ignore"
    ):  # UserWarning warning of loading from an old version of SB3
        model = DQN.load(
            checkpoint,
            env=env,
            device="cpu",
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "learning_rate": 0.0,
                "lr_schedule": None,
                "exploration_schedule": None,
                "verbose": 0,
            },
        )
    recorder_env = RecorderWrapper(env, 0.99, 1000, (0, 0, 0))
    for _ in range(5):
        obs, _ = recorder_env.reset()
        while True:
            obs, _, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                break

    iqlearn = IQLearn(env, sac_args={"device": "cpu", "tensorboard_dir": None})
    iqlearn.set_demonstration_buffer(recorder_env)
    assert (
        iqlearn.demonstration_buffer.pos < 550  # type: ignore
    )  # safety assert, if this one throws, something is wrong with expert

    # act
    iqlearn.learn(500, progress="none")

    # assert
    steps = []
    for _ in range(5):
        step = 0
        obs, _ = env.reset()
        while True:
            step += 1
            obs, _, terminated, truncated, _ = env.step(
                model.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                steps.append(step)
                break

    assert np.mean(steps) < 120


@pytest.mark.slow
def test_iqlearn_pendulum():
    # assemble
    checkpoint = load_from_hub(
        repo_id="sb3/ppo-Pendulum-v1",
        filename="ppo-Pendulum-v1.zip",
    )
    env = gym.make("Pendulum-v1")
    with warnings.catch_warnings(
        action="ignore"
    ):  # UserWarning warning of loading from an old version of SB3
        model = PPO.load(
            checkpoint,
            env=env,
            device="cpu",
            custom_objects={
                "observation_space": env.observation_space,
                "action_space": env.action_space,
                "learning_rate": 0.0,
                "lr_schedule": None,
                "exploration_schedule": None,
                "clip_range": 0.2,
                "verbose": 0,
            },
        )
    recorder_env = RecorderWrapper(env, 0.99, 5000, (0, 0, 0))
    for _ in range(5):
        obs, _ = recorder_env.reset()
        while True:
            obs, _, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                break

    iqlearn = IQLearn(env, sac_args={"device": "cpu", "tensorboard_dir": None})
    iqlearn.set_demonstration_buffer(recorder_env)

    # act
    iqlearn.learn(1000, progress="none")

    # assert
    returns = []
    for _ in range(5):
        undisc_return = 0
        obs, _ = env.reset()
        while True:
            obs, reward, terminated, truncated, _ = env.step(
                model.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break

    assert np.mean(returns) > -300
