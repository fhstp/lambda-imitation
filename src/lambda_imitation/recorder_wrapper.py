import os

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete


class RecorderWrapper(gym.Wrapper):

    def __init__(
        self, env, gamma, buffer_size, hidden_state_dim=0, hidden_state_net=None
    ):
        """
        Gymnasium Wrapper for automatically logging episodes in a circular buffer. Buffer size has to be big enough for single-episode rollouts! (for return calculation)

        :param env: The environment to wrap.
        :param gamma: the discount factor to be used in the environment.
        :param buffer_size: integer, the amount of transitions the buffer can save.
        :param hidden_state_dim: optional, default 0, for saving hidden states.
        :param hidden_state_net: optional, if given, hidden state from the network is saved, should be a callable taking state, action and hidden state and returning a nd-array of dimension `hidden_state_dim`.
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
        self.terminated = np.zeros(self.buffer_size, dtype=np.bool_)
        self.truncated = np.zeros(self.buffer_size, dtype=np.bool_)
        self.setup_hidden_states(hidden_state_dim, hidden_state_net)

        self.pos = 0
        self.last_reset_pos = 0

    def setup_hidden_states(self, hidden_state_dim, hidden_state_net):
        self.hidden_states = np.zeros(
            (self.buffer_size, hidden_state_dim), dtype=np.float32
        )
        self.hidden_state_dim = hidden_state_dim
        self.hidden_state_net = hidden_state_net

    def recalculate_hidden_states(self):
        assert (
            self.pos < self.buffer_size
        ), "hidden state recalculation currently only supported for non-full buffers!"
        assert (
            self.hidden_state_net is not None
        ), "hidden state recalculation only possible for given hidden_state_net!"
        hidden_state = np.zeros((self.hidden_state_dim), dtype=np.float32)
        for i in range(self.pos):
            obs = self.get_observation_at(i)
            action = self.get_action_at(i)
            hidden_state = self.hidden_state_net(obs, action, hidden_state)
            self.hidden_states[i] = hidden_state
            if self.terminated[i] or self.truncated[i]:
                hidden_state = np.zeros((self.hidden_state_dim), dtype=np.float32)

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
        if self.pos > self.last_reset_pos:
            assert (
                self.pos - self.last_reset_pos < self.buffer_size
            ), f"Episode was longer than buffer size, return calculation not possible, {self.pos=}, {self.last_reset_pos=}, {self.pos-self.last_reset_pos}, {self.buffer_size}"
            ret = 0
            for i in reversed(range(self.last_reset_pos, self.pos + 1)):
                ret = self.rewards[i % self.buffer_size] + self.gamma * ret
                self.returns[i % self.buffer_size] = ret
        obs, info = super().reset(**kwargs)
        self.last_reset_pos = self.pos
        self.last_obs = obs
        self.last_hidden_state = np.zeros((self.hidden_state_dim))
        _add_collection_entry(
            self.observation_space, self.observations, obs, self.pos % self.buffer_size
        )
        mod_pos = self.pos % self.buffer_size
        self.hidden_states[mod_pos] = np.zeros((self.hidden_state_dim))
        return obs, info

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
        if self.hidden_state_net is not None:
            self.last_hidden_state = self.hidden_state_net(
                self.last_obs, action, self.last_hidden_state
            )
            self.hidden_states[mod_pos] = self.last_hidden_state

        self.pos += 1
        self.last_obs = obs
        _add_collection_entry(
            self.observation_space, self.observations, obs, self.pos % self.buffer_size
        )

        return obs, reward, terminated, truncated, info

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

    def sample(self, batch_size):
        if self.pos > self.buffer_size:
            upper_bound = self.buffer_size - 1
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            batch_inds = (
                batch_inds - self.pos + 1
            ) % self.buffer_size  # shift so pos never gets sampled
        else:
            upper_bound = self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        observations = self.get_observation_at(batch_inds)
        next_observations = self.get_observation_at((batch_inds + 1) % self.buffer_size)
        actions = self.get_action_at(batch_inds)
        rewards = self.rewards[batch_inds]
        returns = self.returns[batch_inds]
        terminated = self.terminated[batch_inds]
        truncated = self.truncated[batch_inds]
        hidden_states = self.hidden_states[batch_inds]

        return (
            observations,
            next_observations,
            actions,
            rewards,
            returns,
            terminated,
            truncated,
            hidden_states,
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
