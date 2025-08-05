import os
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict, Discrete


class RecorderSample(NamedTuple):
    observations: Any
    next_observations: Any
    actions: Any
    rewards: Any
    returns: Any
    importance_factors: Any
    terminated: Any
    truncated: Any
    hidden_states: Any
    next_hidden_states: Any


class RecorderWrapper(gym.Wrapper):

    def __init__(
        self, env, gamma, buffer_size, hidden_state_dims=(0,), hidden_state_net=None
    ):
        """
        Gymnasium Wrapper for automatically logging episodes in a circular buffer. Buffer size has to be big enough for single-episode rollouts! (for return calculation)

        :param env: The environment to wrap.
        :param gamma: the discount factor to be used in the environment.
        :param buffer_size: integer, the amount of transitions the buffer can save.
        :param hidden_state_dims: optional, tuple of integers, default (0,), for saving hidden states.
        :param hidden_state_net: optional, if given, hidden state from the network is saved, should be a callable taking state, action and hidden states tuple and returning a tuple of length `len(hidden_state_dims)`, with each entry an nd-array of respective dimension in `hidden_state_dims`.
        """
        super().__init__(env)
        assert type(env.observation_space) in [
            Box,
            Discrete,
            Dict,
        ], "Only Box, Discrete and Dict observation spaces are supported right now!"
        assert type(env.action_space) in [
            Box,
            Discrete,
            Dict,
        ], "Only Box, Discrete and Dict action spaces are supported right now!"

        self.gamma = gamma
        self.buffer_size = buffer_size
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.observations = _generate_collection(
            self.observation_space, self.buffer_size
        )
        self.actions = _generate_collection(self.action_space, self.buffer_size)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.behavior_probabilities = np.ones(self.buffer_size, dtype=np.float32)
        self.policy_probabilities = np.zeros(self.buffer_size, dtype=np.float32)
        self.importance_factors = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminated = np.zeros(self.buffer_size, dtype=np.bool_)
        self.truncated = np.zeros(self.buffer_size, dtype=np.bool_)
        self.setup_hidden_states(hidden_state_dims, hidden_state_net)

        self.pos = 0
        self.last_return_calculation = 0
        self.last_sampled_indices = None

    def setup_hidden_states(self, hidden_state_dims, hidden_state_net):
        self.hidden_states = tuple(
            np.zeros((self.buffer_size, hidden_state_dim), dtype=np.float32)
            for hidden_state_dim in hidden_state_dims
        )
        self.hidden_state_dims = hidden_state_dims
        self.hidden_state_net = hidden_state_net

    def recalculate_hidden_states(self):
        assert (
            self.pos < self.buffer_size
        ), "hidden state recalculation currently only supported for non-full buffers!"
        assert (
            self.hidden_state_net is not None
        ), "hidden state recalculation only possible for given hidden_state_net!"
        hidden_states = tuple(
            np.zeros((hidden_state_dim), dtype=np.float32)
            for hidden_state_dim in self.hidden_state_dims
        )
        for i in range(self.pos):
            obs = self.get_observation_at(i)
            action = self.get_action_at(i)
            hidden_states = self.hidden_state_net(obs, action, hidden_states)
            for k, hidden_state in enumerate(hidden_states):
                self.hidden_states[k][i + 1] = hidden_state
            if self.terminated[i] or self.truncated[i]:
                hidden_states = tuple(
                    np.zeros((hidden_state_dim), dtype=np.float32)
                    for hidden_state_dim in self.hidden_state_dims
                )
                for k, hidden_state in enumerate(hidden_states):
                    self.hidden_states[k][i + 1] = hidden_state

    def reset(self, **kwargs):
        """Resets the environment to an initial internal state, returning an initial observation and info and recording the state..

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        self._calculate_returns()
        obs, info = super().reset(**kwargs)
        self.last_obs = obs
        self.last_hidden_states = tuple(
            np.zeros((hidden_state_dim), dtype=np.float32)
            for hidden_state_dim in self.hidden_state_dims
        )
        mod_pos = self.pos % self.buffer_size
        _add_collection_entry(self.observation_space, self.observations, obs, mod_pos)
        for k, hidden_state_dim in enumerate(self.hidden_state_dims):
            self.hidden_states[k][mod_pos] = np.zeros(
                (hidden_state_dim), dtype=np.float32
            )
        return obs, info

    def recalculate_episodes(self):
        ret = 0
        factor = 1
        if self.pos >= self.buffer_size:
            indices = reversed(range(self.pos + 1, self.pos + self.buffer_size))
        else:
            indices = reversed(range(self.pos))
        for i in indices:
            ret = self.rewards[i % self.buffer_size] + self.gamma * ret
            self.returns[i % self.buffer_size] = ret
            factor *= (
                self.policy_probabilities[i % self.buffer_size]
                / self.behavior_probabilities[i % self.buffer_size]
            )
            self.importance_factors[i % self.buffer_size] = factor
            if (
                self.terminated[(i - 1) % self.buffer_size]
                or self.truncated[(i - 1) % self.buffer_size]
            ):
                ret = 0
                factor = 1

    def _calculate_returns(self):
        if self.pos > self.last_return_calculation:
            assert (
                self.pos - self.last_return_calculation < self.buffer_size
            ), f"Episode was longer than buffer size, return calculation not possible, {self.pos=}, {self.last_return_calculation=}, {self.pos-self.last_return_calculation}, {self.buffer_size}"
            ret = 0
            factor = 1
            for i in reversed(range(self.last_return_calculation, self.pos)):
                ret = self.rewards[i % self.buffer_size] + self.gamma * ret
                self.returns[i % self.buffer_size] = ret
            self.last_return_calculation = self.pos

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions and record the step. Note: if the environment is
        reset directly afterwards, the recorded observation is not saved, so keep this in mind when loading from the recorded trajectory!

        When the end of an episode is reached (``terminated or truncated``), it is necessary to call :meth:`reset` to
        reset this environment's state for the next episode.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                An example is a numpy array containing the positions and velocities of the pole in CartPole.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. An example is reaching the goal state or moving into the lava from
                the Sutton and Barton, Gridworld. If true, the user needs to call :meth:`reset`.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is reached.
                If true, the user needs to call :meth:`reset`.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                In OpenAI Gym <v26, it contains "TimeLimit.truncated" to distinguish truncation and termination,
                however this is deprecated in favour of returning terminated and truncated variables.
            done (bool): (Deprecated) A boolean value for if the episode has ended, in which case further :meth:`step` calls will
                return undefined results. This was removed in OpenAI Gym v26 in favor of terminated and truncated attributes.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        mod_pos = self.pos % self.buffer_size
        self.actions[mod_pos] = action
        _add_collection_entry(self.action_space, self.actions, action, mod_pos)
        self.rewards[mod_pos] = reward
        self.terminated[mod_pos] = terminated
        self.truncated[mod_pos] = truncated
        self.pos += 1
        if self.hidden_state_net is not None:
            self.last_hidden_states = self.hidden_state_net(
                self.last_obs, action, self.last_hidden_states
            )
            for k, last_hidden_state in enumerate(self.last_hidden_states):
                self.hidden_states[k][mod_pos] = last_hidden_state

        if terminated or truncated:
            self._calculate_returns()
        self.last_obs = obs
        _add_collection_entry(
            self.observation_space, self.observations, obs, self.pos % self.buffer_size
        )

        return obs, reward, terminated, truncated, info

    def set_probabilities_of_last_action(self, prob):
        self.behavior_probabilities[(self.pos - 1) % self.buffer_size] = prob
        self.policy_probabilities[(self.pos - 1) % self.buffer_size] = prob

    def get_observation_at(self, pos):
        obs = _get_collection_entry(self.observation_space, self.observations, pos)
        return obs

    def get_action_at(self, pos):
        action = _get_collection_entry(self.action_space, self.actions, pos)
        return action

    def save_buffer(self, filename):
        filename = str(filename)
        if not filename.endswith(".npy"):
            filename = filename + ".npy"
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "wb") as f:
            np.save(f, np.array([self.pos, self.buffer_size]))
            np.save(f, self.observations)
            np.save(f, self.rewards)
            np.save(f, self.returns)
            np.save(f, self.actions)
            np.save(f, self.terminated)
            np.save(f, self.truncated)
            np.save(f, self.hidden_states)

    def load_buffer(self, filename):
        filename = str(filename)
        if not filename.endswith(".npy"):
            filename = filename + ".npy"
        with open(filename, "rb") as f:
            pos_buffer_tmp = np.load(f)
            self.pos = pos_buffer_tmp[0]
            self.buffer_size = pos_buffer_tmp[1]
            self.observations = np.load(f)
            if isinstance(self.env.observation_space, Dict):
                self.observations = (
                    self.observations.item()
                )  # saved as array(dict()), so use .item() to retrieve the original dict
            self.rewards = np.load(f)
            self.returns = np.load(f)
            self.actions = np.load(f)
            self.terminated = np.load(f)
            self.truncated = np.load(f)
            self.hidden_states = np.load(f)

    def sample(self, batch_size, mode="numpy", full_episodes_only=False):
        """
        Sample `batch_size` items from the buffer, mode can be either "numpy" or a torch device, e.g. "cpu", "cuda", "auto"
        """
        return self._sample(
            self._generate_indices(batch_size, full_episodes_only), mode
        )

    def override_next_hidden_states_last_sample(self, hidden_states):
        ignore_mask = ~(
            self.terminated[self.last_sampled_indices]
            | self.truncated[self.last_sampled_indices]
        )  # ignore states where current state was terminal
        next_inds = (self.last_sampled_indices + 1) % self.buffer_size
        for k, hidden_state in enumerate(hidden_states):
            self.hidden_states[k][next_inds[ignore_mask]] = hidden_state[ignore_mask]

    def override_policy_probabilities_last_sample(self, prob):
        self.policy_probabilities[self.last_sampled_indices] = prob
        term_trunc = (
            self.terminated[self.last_sampled_indices]
            | self.truncated[self.last_sampled_indices]
        )
        self.importance_factors[self.last_sampled_indices] = (
            self.importance_factors[(self.last_sampled_indices + 1) % self.buffer_size]
            * (1 - term_trunc)
            + term_trunc
        ) * (
            self.policy_probabilities[self.last_sampled_indices]
            / self.behavior_probabilities[self.last_sampled_indices]
        )
        # print(f"{self.policy_probabilities[self.last_sampled_indices]=}")
        # print(f"{self.behavior_probabilities[self.last_sampled_indices]=}")
        # print(
        #     f"{self.policy_probabilities[self.last_sampled_indices]/self.behavior_probabilities[self.last_sampled_indices]=}"
        # )

    def _generate_indices(self, batch_size, full_episodes_only=False):
        if self.pos >= self.buffer_size:
            upper_exclusion = self.pos
            lower_exclusion = (
                self.last_return_calculation - 1 if full_episodes_only else self.pos - 1
            )
            exclusion_len = upper_exclusion - lower_exclusion
            assert (
                exclusion_len < self.buffer_size
            ), "Last/Current episode was longer than buffer size!"
            upper_bound = self.buffer_size - exclusion_len
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)

            upper_exclusion_ind = upper_exclusion % self.buffer_size
            lower_exclusion_ind = lower_exclusion % self.buffer_size
            if upper_exclusion_ind > lower_exclusion_ind:
                # print("upper > lower")
                # print(upper_exclusion_ind)
                # print(lower_exclusion_ind)
                # print(exclusion_len)
                batch_inds[batch_inds > lower_exclusion_ind] += exclusion_len
            else:
                # print("upper < lower")
                # print(upper_exclusion_ind)
                # print(lower_exclusion_ind)
                # print(exclusion_len)
                batch_inds += upper_exclusion_ind + 1
        else:
            upper_bound = (
                self.last_return_calculation if full_episodes_only else self.pos
            )
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return batch_inds

    def _sample(self, batch_inds, mode="numpy"):
        self.last_sampled_indices = batch_inds
        observations = self.get_observation_at(batch_inds)
        next_observations = self.get_observation_at((batch_inds + 1) % self.buffer_size)
        actions = self.get_action_at(batch_inds)
        rewards = self.rewards[batch_inds]
        returns = self.returns[batch_inds]
        importance_factors = self.importance_factors[batch_inds]
        terminated = self.terminated[batch_inds]
        truncated = self.truncated[batch_inds]
        hidden_states = tuple(
            hidden_state[batch_inds] for hidden_state in self.hidden_states
        )
        next_hidden_states = tuple(
            hidden_state[(batch_inds + 1) % self.buffer_size]
            for hidden_state in self.hidden_states
        )

        if mode != "numpy":
            observations = torch.tensor(observations).to(mode)
            next_observations = torch.tensor(next_observations).to(mode)
            actions = torch.tensor(actions).to(mode)
            rewards = torch.tensor(rewards).to(mode)
            returns = torch.tensor(returns).to(mode)
            importance_factors = torch.tensor(importance_factors).to(mode)
            terminated = torch.tensor(terminated).to(mode)
            truncated = torch.tensor(truncated).to(mode)
            hidden_states = tuple(
                torch.tensor(hidden_state).to(mode) for hidden_state in hidden_states
            )
            next_hidden_states = tuple(
                torch.tensor(hidden_state).to(mode)
                for hidden_state in next_hidden_states
            )

        return RecorderSample(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            importance_factors=importance_factors,
            terminated=terminated,
            truncated=truncated,
            hidden_states=hidden_states,
            next_hidden_states=next_hidden_states,
        )

    def get_sb3_buffer(self, device="auto"):
        """Converts the data to a sb3 usable buffer"""
        try:
            from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
        except ImportError:
            raise Exception(
                "Please install stable_baselines3 to use this feature, e.g. by running 'pip install stable_baselines3"
            )
        if type(self.observation_space) == Dict:
            sb3_buffer = DictReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=device,
                handle_timeout_termination=False,
            )
        else:
            sb3_buffer = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=device,
                handle_timeout_termination=False,
            )
        for i in range(min(self.buffer_size, self.pos - 1)):
            obs = _get_collection_entry(self.observation_space, self.observations, i)
            action = _get_collection_entry(self.action_space, self.actions, i)
            next_obs = _get_collection_entry(
                self.observation_space, self.observations, (i + 1) % self.buffer_size
            )
            terminated = self.terminated[i]
            reward = self.rewards[i]
            sb3_buffer.add(obs, next_obs, action, reward, terminated, {})

        return sb3_buffer

    def get_imitation_trajectories(self):
        """Converts the data to imition usable trajectories"""
        try:
            from imitation.data.types import TrajectoryWithRew
        except ImportError:
            raise Exception(
                "Please install imitation to use this feature, e.g. by running 'pip install imitation"
            )

        if self.pos > self.buffer_size:
            raise Exception(
                "Buffer size was not big enough, observations were overwritten and trajectories cannot be retrieved!"
            )

        obs = []
        actions = []
        rewards = []
        trajectories = []
        for i in range(self.pos + 1):
            obs.append(
                _get_collection_entry(self.observation_space, self.observations, i)
            )
            if not self.terminated[i] and not self.truncated[i] and i != self.pos:
                actions.append(
                    _get_collection_entry(self.action_space, self.actions, i)
                )
                rewards.append(self.rewards[i])
            if self.terminated[i] or self.truncated[i] or i == self.pos:
                obs = np.array(obs)
                actions = np.array(actions)
                rewards = np.array(rewards)
                if (
                    len(actions) > 0
                ):  # length 0 trajectories might happen when env gets reset at last step
                    trajectories.append(
                        TrajectoryWithRew(
                            obs, actions, None, self.terminated[i], rewards
                        )
                    )
                obs = []
                actions = []
                rewards = []
        return trajectories


def _get_collection_entry(space, collection, pos):
    if type(space) != Dict:
        return collection[pos]

    entry = {}
    for key in space:
        entry[key] = _get_collection_entry(space[key], collection[key], pos)
    return entry


def _add_collection_entry(space, collection, entry, pos):
    if type(space) != Dict:
        collection[pos] = entry
        return

    for key in entry:
        _add_collection_entry(space[key], collection[key], entry[key], pos)


def _generate_collection(space, buffer_size):
    if type(space) == Box:
        return np.zeros((buffer_size, *space.shape), dtype=space.dtype)
    if type(space) == Discrete:
        return np.zeros(buffer_size, dtype=np.int32)
    if type(space) == Dict:
        collection = {}
        for key in space:
            collection[key] = _generate_collection(space[key], buffer_size)
        return collection
