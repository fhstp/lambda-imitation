"""T-maze adapted from Allen et al., brownirl/lambda_discrepancy (Apache-2.0).

Vendored from lambda-envs dbd9dfc069430b3c4c1694bf18b8e3b14eda18df.
Publication changes: deterministic paper transitions only (unused slip option
removed), optional get_obs params, terminal predicate aligned with termination
after choosing a branch. See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams


@chex.dataclass
class TMazeState:
    grid_idx: int
    goal_dir: int


class TMaze(Environment):
    """Initial cue, aliased corridor and junction; actions N, S, E, W."""

    def __init__(self, hallway_length=5, good_reward=4.0, bad_reward=-0.1):
        if hallway_length < 1:
            raise ValueError("hallway_length must be positive")
        self.hallway_length = hallway_length
        self.good_reward, self.bad_reward = good_reward, bad_reward

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=1000)

    @property
    def num_actions(self):
        return 4

    def observation_space(self, params=None):
        return spaces.Box(0, 1, (5,))

    def action_space(self, params=None):
        return spaces.Discrete(4)

    def get_obs(self, state, params=None):
        start = state.grid_idx == 0
        junction = state.grid_idx == self.hallway_length + 1
        obs = jnp.zeros(5)
        return (start * obs.at[state.goal_dir].set(1)
                + junction * obs.at[3].set(1)
                + (1 - start) * (1 - junction) * obs.at[2].set(1))

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params):
        state = TMazeState(grid_idx=0, goal_dir=jax.random.bernoulli(key).astype(int))
        return self.get_obs(state), state

    def is_terminal(self, state, params=None):
        return state.grid_idx > self.hallway_length + 1

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, action, params):
        junction = state.grid_idx == self.hallway_length + 1
        done = junction & (action < 2)
        correct = action == state.goal_dir
        reward = jnp.where(done, jnp.where(correct, self.good_reward, self.bad_reward), 0.0)
        horizontal = jnp.where(action == 2,
                               jnp.minimum(state.grid_idx + 1, self.hallway_length + 1),
                               jnp.maximum(state.grid_idx - 1, 0))
        idx = jnp.where(action < 2, state.grid_idx + done, horizontal)
        state = TMazeState(grid_idx=idx, goal_dir=state.goal_dir)
        return self.get_obs(state), state, reward, done, {}
