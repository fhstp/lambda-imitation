import sys
import time

import gymnasium as gym
import numpy as np
import yaml

from lambda_imitation import IQLearn

device = "cpu"
eval_interval = 1_000
train_steps = (
    10_000_000  # should be divisible by eval_interval, otherwise will not be exact
)


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


with open(sys.argv[1]) as file:
    sac_args = yaml.safe_load(file.read())

## override some args
np.random.seed(None)
seed = np.random.randint(200000)
sac_args["seed"] = seed
sac_args["device"] = device

env = PartialCartpoleWrapper(gym.make("CartPole-v1"))
iqlearn = IQLearn(
    env,
    sac_args=sac_args,
)

eval_env = PartialCartpoleWrapper(gym.make("CartPole-v1"))
for i in range(train_steps // eval_interval):
    print(f"running training {i}/{train_steps//eval_interval}")
    iqlearn.sac_learn(eval_interval)
    print("evaluating...")

    steps_runs = []
    for _ in range(10):
        steps = 0
        obs, info = eval_env.reset()
        hidden_state = np.zeros(iqlearn.args.hidden_state_dim, dtype=np.float32)
        while True:
            action, hidden_state = iqlearn.predict(
                obs, hidden_state, deterministic=True
            )
            obs, _, terminated, truncated, _ = eval_env.step(action)
            steps += 1
            if terminated or truncated:
                break
        steps_runs.append(steps)

    iqlearn.writer.add_scalar(
        "charts/eval_return",
        np.mean(steps_runs),
        iqlearn.n_updates,
    )
