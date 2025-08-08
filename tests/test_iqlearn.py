## no tests in classical unit test sense, since RL algos are hard to test that way...

import warnings
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN, PPO, SAC

from lambda_imitation import IQLearn
from lambda_imitation.recorder_wrapper import RecorderWrapper


class SimpleGridWorld(gym.Env):
    def __init__(self, max_right=2):
        super().__init__()
        self.max_right = max_right
        self.observation_space = gym.spaces.Box(
            np.array([0]), np.array([self.max_right])
        )
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

        if self.pos > self.max_right:
            self.pos = self.max_right

        terminated = self.pos == self.max_right or self.pos == -1
        truncated = False
        reward = 1 if self.pos == self.max_right else (-1 if self.pos == -1 else 0)

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
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
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
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
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
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
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
            "tensorboard_dir": None,
        },
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    undiscounted_return = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        undiscounted_return += reward  # type: ignore
        if terminated or truncated:
            break

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

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 4,
        sac_args={"device": "cpu", "target_entropy": 0.2, "tensorboard_dir": None},
    )
    iqlearn.set_demonstration_buffer(recorder_env)
    assert (
        iqlearn.demonstration_buffer.pos < 550  # type: ignore
    )  # safety assert, if this one throws, something is wrong with expert

    # act
    iqlearn.learn(10000, progress="none")

    # assert
    env = gym.make("MountainCar-v0")
    steps = []
    for _ in range(5):
        step = 0
        obs, _ = env.reset()
        while True:
            step += 1
            obs, _, terminated, truncated, _ = env.step(
                iqlearn.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                steps.append(step)
                break

    assert np.mean(steps) < 140


@pytest.mark.slow
def test_iqlearn_mountaincar_continuous():
    # assemble
    checkpoint = load_from_hub(
        repo_id="sb3/sac-MountainCarContinuous-v0",
        filename="sac-MountainCarContinuous-v0.zip",
    )
    env = gym.make("MountainCarContinuous-v0")
    with warnings.catch_warnings(
        action="ignore"
    ):  # UserWarning warning of loading from an old version of SB3
        model = SAC.load(
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
    returns = []
    for _ in range(20):
        undisc_return = 0
        obs, _ = recorder_env.reset()
        while True:
            obs, reward, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break
    assert np.mean(returns) > -400

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 10,
        sac_args={"device": "cpu", "tensorboard_dir": None},
    )
    iqlearn.set_demonstration_buffer(recorder_env)

    # act
    iqlearn.learn(3000, progress="none")

    # env = gym.make("MountainCarContinuous-v0", render_mode="human")
    # assert
    returns = []
    for _ in range(5):
        undisc_return = 0
        obs, _ = env.reset()
        while True:
            obs, reward, terminated, truncated, _ = env.step(
                iqlearn.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break

    assert np.mean(returns) > -700


@pytest.mark.slow
def test_sac_cartpole_q_lstm():
    # assemble
    env = gym.make("CartPole-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(0, 10, 10),
    )

    # act
    iqlearn.sac_learn(20000, progress="none")

    # assert
    steps = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert iqlearn.env.hidden_states[0][32].shape == (0,)
    assert iqlearn.env.hidden_states[1][32].shape == (10,)
    assert iqlearn.env.hidden_states[2][32].shape == (10,)
    assert np.nonzero(iqlearn.env.hidden_states[1][32])
    assert np.nonzero(iqlearn.env.hidden_states[2][32])
    assert steps > 150


@pytest.mark.slow
def test_sac_pendulum_q_lstm():
    # assemble
    env = gym.make("Pendulum-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(0, 10, 10),
    )

    # act
    iqlearn.sac_learn(20000, progress="none")

    # assert
    undiscounted_return = 0
    obs, info = env.reset()
    while True:
        action = iqlearn.predict(torch.tensor(obs), deterministic=True)[0]
        obs, reward, terminated, truncated, _ = env.step(action)
        undiscounted_return += reward  # type: ignore
        if terminated or truncated:
            break

    assert iqlearn.env.hidden_states[0][32].shape == (0,)
    assert iqlearn.env.hidden_states[1][32].shape == (10,)
    assert iqlearn.env.hidden_states[2][32].shape == (10,)
    assert np.nonzero(iqlearn.env.hidden_states[1][32])
    assert np.nonzero(iqlearn.env.hidden_states[2][32])
    assert undiscounted_return > -600


@pytest.mark.slow
def test_sac_cartpole_actor_lstm():
    # assemble
    env = gym.make("CartPole-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(10, 0, 0),
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    steps = 0
    obs, info = env.reset()
    hidden_state = np.zeros(10, dtype=np.float32)
    while True:
        action, hidden_state = iqlearn.predict(obs, hidden_state, True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert np.nonzero(iqlearn.env.hidden_states[0][32])
    assert np.nonzero(iqlearn.env.hidden_states[1][32])
    assert steps > 150


@pytest.mark.slow
def test_sac_pendulum_actor_lstm():
    # assemble
    env = gym.make("Pendulum-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(10, 0, 0),
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    undiscounted_return = 0
    obs, info = env.reset()
    hidden_state = np.zeros(10, dtype=np.float32)
    while True:
        action, hidden_state = iqlearn.predict(obs, hidden_state, True)
        obs, reward, terminated, truncated, _ = env.step(action)
        undiscounted_return += reward  # type: ignore
        if terminated or truncated:
            break

    assert undiscounted_return > -500


@pytest.mark.slow
def test_iqlearn_mountaincar_lstm_wrong_buffer_dims():
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

    iqlearn = IQLearn(
        env,
        sac_args={"device": "cpu", "tensorboard_dir": None},
        hidden_state_dims=(0, 10, 10),
    )

    # act % assert
    with pytest.raises(AssertionError):
        iqlearn.set_demonstration_buffer(recorder_env)


@pytest.mark.slow
def test_iqlearn_mountaincar_q_lstm():
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
    recorder_env = RecorderWrapper(env, 0.99, 1000, (0, 10, 10))
    for _ in range(5):
        obs, _ = recorder_env.reset()
        while True:
            obs, _, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                break

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 4,
        sac_args={"device": "cpu", "target_entropy": 0.2, "tensorboard_dir": None},
        hidden_state_dims=(0, 10, 10),
    )
    iqlearn.set_demonstration_buffer(recorder_env)
    assert (
        iqlearn.demonstration_buffer.pos < 550  # type: ignore
    )  # safety assert, if this one throws, something is wrong with expert

    # act
    iqlearn.learn(15000, progress="none")

    # assert
    steps = []
    for _ in range(5):
        step = 0
        obs, _ = env.reset()
        while True:
            step += 1
            obs, _, terminated, truncated, _ = env.step(
                iqlearn.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                steps.append(step)
                break

    print(np.mean(steps))
    assert np.mean(steps) < 140


@pytest.mark.slow
def test_iqlearn_mountain_car_continuous_q_lstm():
    # assemble
    checkpoint = load_from_hub(
        repo_id="sb3/sac-MountainCarContinuous-v0",
        filename="sac-MountainCarContinuous-v0.zip",
    )
    env = gym.make("MountainCarContinuous-v0")
    with warnings.catch_warnings(
        action="ignore"
    ):  # UserWarning warning of loading from an old version of SB3
        model = SAC.load(
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
    recorder_env = RecorderWrapper(env, 0.99, 5000, (0, 10, 10))
    returns = []
    for _ in range(20):
        undisc_return = 0
        obs, _ = recorder_env.reset()
        while True:
            obs, reward, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break
    assert np.mean(returns) > -400

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 10,
        sac_args={"device": "cpu", "tensorboard_dir": None},
        hidden_state_dims=(0, 10, 10),
    )
    iqlearn.set_demonstration_buffer(recorder_env)

    # act
    iqlearn.learn(5000, progress="none")

    # assert
    returns = []
    for _ in range(5):
        undisc_return = 0
        obs, _ = env.reset()
        while True:
            obs, reward, terminated, truncated, _ = env.step(
                iqlearn.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break

    assert np.mean(returns) > -700


@pytest.mark.slow
def test_iqlearn_mountaincar_actor_lstm():
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
    recorder_env = RecorderWrapper(env, 0.99, 1000, (10, 0, 0))
    for _ in range(5):
        obs, _ = recorder_env.reset()
        while True:
            obs, _, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            if terminated or truncated:
                break

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 4,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "buffer_size": 1000,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(10, 0, 0),
    )
    iqlearn.set_demonstration_buffer(recorder_env)
    assert (
        iqlearn.demonstration_buffer.pos < 550  # type: ignore
    )  # safety assert, if this one throws, something is wrong with expert

    # act
    iqlearn.learn(30000, progress="none")

    # assert
    steps = []
    for _ in range(5):
        step = 0
        obs, _ = env.reset()
        hidden_state = np.zeros(10, dtype=np.float32)
        while True:
            action, hidden_state = iqlearn.predict(
                obs, hidden_state, deterministic=True
            )
            step += 1
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                steps.append(step)
                break

    assert np.mean(steps) < 130


@pytest.mark.slow
def test_iqlearn_mountain_car_continuous_actor_lstm():
    # assemble
    checkpoint = load_from_hub(
        repo_id="sb3/sac-MountainCarContinuous-v0",
        filename="sac-MountainCarContinuous-v0.zip",
    )
    env = gym.make("MountainCarContinuous-v0")
    with warnings.catch_warnings(
        action="ignore"
    ):  # UserWarning warning of loading from an old version of SB3
        model = SAC.load(
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
    recorder_env = RecorderWrapper(env, 0.99, 5000, (10, 0, 0))
    returns = []
    for _ in range(20):
        undisc_return = 0
        obs, _ = recorder_env.reset()
        while True:
            obs, reward, terminated, truncated, _ = recorder_env.step(
                model.predict(obs, deterministic=True)[0]
            )
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break
    assert np.mean(returns) > -400

    iqlearn = IQLearn(
        env,
        regularizer=lambda x: x**2 / 10,
        sac_args={"device": "cpu", "tensorboard_dir": None},
        hidden_state_dims=(10, 0, 0),
    )
    iqlearn.set_demonstration_buffer(recorder_env)

    # act
    iqlearn.learn(5000, progress="none")

    # assert
    returns = []
    for _ in range(5):
        undisc_return = 0
        obs, _ = env.reset()
        hidden_state = np.zeros(10, dtype=np.float32)
        while True:
            action, hidden_state = iqlearn.predict(
                obs, hidden_state, deterministic=True
            )
            obs, reward, terminated, truncated, _ = env.step(action)
            undisc_return += reward  # type: ignore
            if terminated or truncated:
                returns.append(undisc_return)
                break

    assert np.mean(returns) > -700


# @pytest.mark.slow
# def test_lambda_discrepancy_sac_cartpole():
#     # assemble
#     env = gym.wrappers.TimeLimit(SimpleGridWorld(10), 100)
#     iqlearn = IQLearn(
#         env,
#         sac_args={
#             "device": "cpu",
#             "target_entropy": 0.2,
#             # "tensorboard_dir": None,
#             "use_lambda_discrepancy": True,
#             "buffer_size": 1000,
#             "learning_starts": 100,
#             "use_targets": False,
#         },
#         hidden_state_dims=(0, 0, 0),
#     )
#
#     # act
#     iqlearn.sac_learn(150000)
#
#     # assert
#     steps = 0
#     obs, info = env.reset()
#     while True:
#         action, _ = iqlearn.predict(obs, deterministic=True)
#         obs, _, terminated, truncated, _ = env.step(action)
#         steps += 1
#         if terminated or truncated:
#             break
#
#     assert steps < 11
#
#
# class PartialGridworldWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.original_obs = None
#         self.observation_space = gym.spaces.Box(
#             np.array([0], dtype=np.float32),
#             np.array([1], dtype=np.float32),
#         )
#
#     def _obs(self, obs):
#         self.original_obs = obs
#         return np.array([0])
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         obs = self._obs(obs)
#         return obs, info
#
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         obs = self._obs(obs)
#
#         return obs, reward, terminated, truncated, info
#
#
# @pytest.mark.slow
# def test_lambda_discrepancy_sac_po_cartpole():
#     # assemble
#     env = PartialGridworldWrapper(gym.wrappers.TimeLimit(SimpleGridWorld(10), 100))
#     iqlearn = IQLearn(
#         env,
#         sac_args={
#             "device": "cpu",
#             "target_entropy": 0.2,
#             # "tensorboard_dir": None,
#             "use_lambda_discrepancy": True,
#             "buffer_size": 1000,
#             "learning_starts": 100,
#         },
#         hidden_state_dims=(0, 0, 0),
#     )
#
#     # act
#     iqlearn.sac_learn(150000)
#
#     # assert
#     steps = 0
#     obs, info = env.reset()
#     while True:
#         action, _ = iqlearn.predict(obs, deterministic=True)
#         obs, _, terminated, truncated, _ = env.step(action)
#         steps += 1
#         if terminated or truncated:
#             break
#
#     assert steps < 11


# @pytest.mark.slow
# def test_iqlearn_mountaincar_shared_lstm():
#     # assemble
#     checkpoint = load_from_hub(
#         repo_id="sb3/dqn-MountainCar-v0",
#         filename="dqn-MountainCar-v0.zip",
#     )
#     env = gym.make("MountainCar-v0")
#     with warnings.catch_warnings(
#         action="ignore"
#     ):  # UserWarning warning of loading from an old version of SB3
#         model = DQN.load(
#             checkpoint,
#             env=env,
#             device="cpu",
#             custom_objects={
#                 "observation_space": env.observation_space,
#                 "action_space": env.action_space,
#                 "learning_rate": 0.0,
#                 "lr_schedule": None,
#                 "exploration_schedule": None,
#                 "verbose": 0,
#             },
#         )
#     recorder_env = RecorderWrapper(env, 0.99, 1000, (10,))
#     for _ in range(5):
#         obs, _ = recorder_env.reset()
#         while True:
#             obs, _, terminated, truncated, _ = recorder_env.step(
#                 model.predict(obs, deterministic=True)[0]
#             )
#             if terminated or truncated:
#                 break
#
#     iqlearn = IQLearn(
#         env,
#         regularizer=lambda x: x**2 / 4,
#         sac_args={
#             "device": "cpu",
#             "target_entropy": 0.2,
#             "buffer_size": 1000,
#             # "tensorboard_dir": None,
#             "use_lambda_discrepancy": True,
#             "recalculate_hidden_states_in_update": True,
#         },
#         hidden_state_dims=(10,),
#     )
#     iqlearn.set_demonstration_buffer(recorder_env)
#     assert (
#         iqlearn.demonstration_buffer.pos < 550  # type: ignore
#     )  # safety assert, if this one throws, something is wrong with expert
#
#     # act
#     iqlearn.learn(30000)
#
#     # assert
#     steps = []
#     for _ in range(5):
#         step = 0
#         obs, _ = env.reset()
#         hidden_state = np.zeros(10, dtype=np.float32)
#         while True:
#             action, hidden_state = iqlearn.predict(
#                 obs, hidden_state, deterministic=True
#             )
#             step += 1
#             obs, _, terminated, truncated, _ = env.step(action)
#             if terminated or truncated:
#                 steps.append(step)
#                 break
#
#     assert np.mean(steps) < 130
#
#
# @pytest.mark.slow
# def test_iqlearn_mountain_car_continuous_shared_lstm():
#     # assemble
#     checkpoint = load_from_hub(
#         repo_id="sb3/sac-MountainCarContinuous-v0",
#         filename="sac-MountainCarContinuous-v0.zip",
#     )
#     env = gym.make("MountainCarContinuous-v0")
#     with warnings.catch_warnings(
#         action="ignore"
#     ):  # UserWarning warning of loading from an old version of SB3
#         model = SAC.load(
#             checkpoint,
#             env=env,
#             device="cpu",
#             custom_objects={
#                 "observation_space": env.observation_space,
#                 "action_space": env.action_space,
#                 "learning_rate": 0.0,
#                 "lr_schedule": None,
#                 "exploration_schedule": None,
#                 "clip_range": 0.2,
#                 "verbose": 0,
#             },
#         )
#     recorder_env = RecorderWrapper(env, 0.99, 5000, (10,))
#     returns = []
#     for _ in range(20):
#         undisc_return = 0
#         obs, _ = recorder_env.reset()
#         while True:
#             obs, reward, terminated, truncated, _ = recorder_env.step(
#                 model.predict(obs, deterministic=True)[0]
#             )
#             undisc_return += reward  # type: ignore
#             if terminated or truncated:
#                 returns.append(undisc_return)
#                 break
#     assert np.mean(returns) > -400
#
#     iqlearn = IQLearn(
#         env,
#         regularizer=lambda x: x**2 / 10,
#         sac_args={"device": "cpu", "tensorboard_dir": None},
#         hidden_state_dims=(10,),
#     )
#     iqlearn.set_demonstration_buffer(recorder_env)
#
#     # act
#     iqlearn.learn(5000, progress="none")
#
#     # assert
#     returns = []
#     for _ in range(5):
#         undisc_return = 0
#         obs, _ = env.reset()
#         hidden_state = np.zeros(10, dtype=np.float32)
#         while True:
#             action, hidden_state = iqlearn.predict(
#                 obs, hidden_state, deterministic=True
#             )
#             obs, reward, terminated, truncated, _ = env.step(action)
#             undisc_return += reward  # type: ignore
#             if terminated or truncated:
#                 returns.append(undisc_return)
#                 break
#
#     assert np.mean(returns) > -700


class PartialCartpoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_obs = None
        self.observation_space = gym.spaces.Box(
            np.array([-4.8, -0.41887903], dtype=np.float32),
            np.array([4.8, 0.41887903], dtype=np.float32),
        )

    def _obs(self, obs):
        self.original_obs = obs
        return obs[[0, 2]]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._obs(obs)

        return obs, reward, terminated, truncated, info


@pytest.mark.slow
def test_sac_cartpole_shared_lstm():
    # assemble
    env = PartialCartpoleWrapper(gym.make("CartPole-v1"))
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cuda",
            "target_entropy": 0.2,
            # "tensorboard_dir": None,
            "use_lambda_discrepancy": True,
            "recalculate_hidden_states_in_update": True,
            "hidden_state_recalculation_interval": 1,
            "buffer_size": 10000,
            "learning_starts": 500,
            # "q_lr": 0.0001,
            # "policy_lr": 0.0001,
            # "tau": 0.0005,
        },
        hidden_state_dim=100,
    )

    # act
    iqlearn.sac_learn(1500000)

    # assert
    steps = 0
    obs, info = env.reset()
    hidden_state = np.zeros(100, dtype=np.float32)
    while True:
        action, hidden_state = iqlearn.predict(obs, hidden_state, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    assert steps > 150


@pytest.mark.slow
def test_sac_pendulum_actor_lstm():
    # assemble
    env = gym.make("Pendulum-v1")
    iqlearn = IQLearn(
        env,
        sac_args={
            "device": "cpu",
            "target_entropy": 0.2,
            "tensorboard_dir": None,
        },
        hidden_state_dims=(10,),
    )

    # act
    iqlearn.sac_learn(15000, progress="none")

    # assert
    undiscounted_return = 0
    obs, info = env.reset()
    hidden_state = np.zeros(10, dtype=np.float32)
    while True:
        action, hidden_state = iqlearn.predict(obs, hidden_state, True)
        obs, reward, terminated, truncated, _ = env.step(action)
        undiscounted_return += reward  # type: ignore
        if terminated or truncated:
            break

    assert undiscounted_return > -500


test_sac_cartpole_shared_lstm()
