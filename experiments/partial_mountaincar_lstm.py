import os
import sys
import time
import warnings
from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Dict, Discrete
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN, PPO, SAC
from tqdm.rich import tqdm

from lambda_imitation import IQLearn
from lambda_imitation.recorder_wrapper import RecorderWrapper


class PartialWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.original_obs = None
        self.observation_space = gym.spaces.Box(
            np.array([-1.2], dtype=np.float32), np.array([0.6], dtype=np.float32)
        )

    def _obs(self, obs):
        self.original_obs = obs
        return obs[:1]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._obs(obs)

        return obs, reward, terminated, truncated, info


# assemble
checkpoint = load_from_hub(
    repo_id="sb3/dqn-MountainCar-v0",
    filename="dqn-MountainCar-v0.zip",
)
env = gym.make("MountainCar-v0")
partial_env = PartialWrapper(env)
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
recorder_env = RecorderWrapper(partial_env, 0.99, 1000, (10, 10, 10))
for _ in range(5):
    obs, _ = recorder_env.reset()
    obs = partial_env.original_obs
    while True:
        obs, _, terminated, truncated, _ = recorder_env.step(
            model.predict(obs, deterministic=True)[0]
        )
        obs = partial_env.original_obs
        if terminated or truncated:
            break

iqlearn = IQLearn(
    partial_env,
    regularizer=lambda x: x**2 / 4,
    sac_args={
        "device": "cpu",
        "target_entropy": 0.2,
        "buffer_size": 1000,
        "hidden_state_recalculation_interval": int(sys.argv[1]),
        "recalculate_hidden_states_in_update": bool(sys.argv[2]),
    },
    hidden_state_dims=(10, 10, 10),
)
iqlearn.set_demonstration_buffer(recorder_env)
assert (
    iqlearn.demonstration_buffer.pos < 550  # type: ignore
)  # safety assert, if this one throws, something is wrong with expert

eval_env = PartialWrapper(gym.make("MountainCar-v0"))
steps = []
for _ in tqdm(range(30)):
    iqlearn.learn(1000, progress="none")
    # eval
    eval_steps = []
    for _ in range(10):
        step = 0
        obs, _ = eval_env.reset()
        hidden_state = np.zeros(10, dtype=np.float32)
        while True:
            action, hidden_state = iqlearn.predict(
                obs, hidden_state, deterministic=True
            )
            step += 1
            obs, _, terminated, truncated, _ = eval_env.step(action)
            if terminated or truncated:
                eval_steps.append(step)
                break
    steps.append(np.mean(eval_steps))

results_path = f"experiments/results/{iqlearn.args.hidden_state_recalculation_interval}_{iqlearn.args.recalculate_hidden_states_in_update}"

os.makedirs(results_path, exist_ok=True)
np.save(f"{results_path}/{time.time()}.npy", np.array(steps))

# env = gym.make("MountainCar-v0", render_mode="human")
# partial_env = PartialWrapper(env)
# steps = []
# for _ in range(5):
#     step = 0
#     obs, _ = partial_env.reset()
#     hidden_state = np.zeros(10, dtype=np.float32)
#     while True:
#         action, hidden_state = iqlearn.predict(obs, hidden_state, deterministic=True)
#         step += 1
#         obs, _, terminated, truncated, _ = partial_env.step(action)
#         if terminated or truncated:
#             steps.append(step)
#             break
#
# print(np.mean(steps))
