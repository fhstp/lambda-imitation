"""Environment registry for the full POPGym B200 benchmark.

Three sources, all exposing the PortEnv interface (reset/transition/step,
horizon, action_dim, obs_shape):

- audited POPJym ports (popgym_jax.PortEnv), unchanged;
- NEW JAX ports written here for families whose POPJym port is broken or
  missing: Autoencode, CountRecall, MineSweeper, VelocityOnlyCartPole. They
  follow NumPy POPGym 1.0.7 step semantics and its Gymnasium encoding
  (FlattenObservation one-hots, row-major raveled MultiDiscrete actions);
  tests/test_popgym_bench_envs.py checks them step-by-step against NumPy POPGym;
- Labyrinth Escape/Explore: no JAX port, host Gymnasium env (HostEnv).

Termination and truncation are combined into one done flag, as in the
previous suite and the POPJym ports.
"""

from typing import NamedTuple

from popgym_jax import PortEnv, port_for

DIFFICULTIES = ('Easy', 'Medium', 'Hard')
POPJYM_FAMILIES = ('Battleship', 'Concentration', 'HigherLower', 'RepeatFirst', 'RepeatPrevious',
                   'PositionOnlyCartPole', 'NoisyPositionOnlyCartPole', 'MultiarmedBandit')
NEW_JAX_FAMILIES = ('Autoencode', 'CountRecall', 'MineSweeper', 'VelocityOnlyCartpole')
HOST_FAMILIES = ('LabyrinthEscape', 'LabyrinthExplore')
FAMILIES = POPJYM_FAMILIES + NEW_JAX_FAMILIES + HOST_FAMILIES


def all_env_ids():
    return [f'popgym-{family}{d}-v0' for family in FAMILIES for d in DIFFICULTIES]


def split_env_id(env_id):
    name = env_id.removeprefix('popgym-').removesuffix('-v0')
    for d in DIFFICULTIES:
        if name.endswith(d):
            return name.removesuffix(d), d
    raise ValueError(f'Not an Easy/Medium/Hard POPGym task: {env_id}')


def backend_for(env_id):
    family, _ = split_env_id(env_id)
    if family in HOST_FAMILIES:
        return 'gymnasium'
    if family in NEW_JAX_FAMILIES:
        return 'jax-new'
    if family in POPJYM_FAMILIES and port_for(env_id)[0] is not None:
        return 'jax-popjym'
    raise ValueError(f'Unsupported task {env_id}')


def make_port(env_id):
    backend = backend_for(env_id)
    if backend == 'jax-popjym':
        return PortEnv(env_id)
    if backend == 'gymnasium':
        return HostEnv(env_id)
    family, difficulty = split_env_id(env_id)
    return NEW_PORTS[family](env_id, difficulty)


def one_hot(index, n):
    import jax
    import jax.numpy as jnp
    return jax.nn.one_hot(index, n, dtype=jnp.float32)


class JaxPort:
    """Shared auto-reset step: reset only when done, like popgym_jax.PortEnv."""

    def step(self, key, state, action):
        import jax
        sk, rk = jax.random.split(key)
        obs, state, reward, done = self.transition(sk, state, action)
        obs, state = jax.lax.cond(done, lambda _: self.reset(rk), lambda _: (obs, state), None)
        return obs, state, reward, done


def card_values(n_decks, kind):
    """Same per-card index arrays as popgym.core.deck.Deck (order irrelevant
    after the uniform shuffle, kept identical anyway)."""
    import numpy as np
    ranks, suits, colors = 13, 4, 2
    n = 52 * n_decks
    if kind == 'suits':
        return np.tile(np.tile(np.arange(suits), ranks), n_decks)
    if kind == 'ranks':
        return np.tile(np.arange(ranks).repeat(suits), n_decks)
    if kind == 'colors':
        return np.tile(np.arange(colors), n // 2)
    raise ValueError(kind)


class AutoencodeState(NamedTuple):
    cards: object  # suit index of each card in deal order
    t: object


class Autoencode(JaxPort):
    """WATCH: one card dealt at reset and one per step (action ignored, reward 0);
    the step dealing the final card already reports PLAY mode. PLAY: recite in
    reverse order, +-1/N per card; constant (PLAY, suit 0) observation."""

    DECKS = dict(Easy=1, Medium=2, Hard=3)

    def __init__(self, env_id, difficulty):
        self.env_id, self.family, self.port_id = env_id, 'Autoencode', f'bench-jax-Autoencode{difficulty}'
        self.n_cards = 52 * self.DECKS[difficulty]
        self.horizon = 2 * self.n_cards - 1
        self.action_dim = 4
        self.obs_shape = (6,)
        self.values = card_values(self.DECKS[difficulty], 'suits')

    def obs(self, mode, suit):
        import jax.numpy as jnp
        return jnp.concatenate([one_hot(mode, 2), one_hot(suit, 4)])

    def reset(self, key):
        import jax
        import jax.numpy as jnp
        cards = jax.random.permutation(key, jnp.asarray(self.values, jnp.int32))
        return self.obs(1, cards[0]), AutoencodeState(cards, jnp.int32(0))

    def transition(self, key, state, action):
        import jax.numpy as jnp
        N, t = self.n_cards, state.t
        watch = t < N - 1
        dealt = jnp.minimum(t + 1, N - 1)
        play = jnp.clip(t - (N - 1), 0, N - 1)
        correct = state.cards[N - 1 - play]
        reward = jnp.where(watch, 0., jnp.where(action == correct, 1. / N, -1. / N))
        mode = jnp.where(watch & (t + 1 < N - 1), 1, 0)
        obs = jnp.where(watch, self.obs(mode, state.cards[dealt]), self.obs(0, 0))
        done = ~watch & (play == N - 1)
        return obs, state._replace(t=t + 1), jnp.float32(reward), done


class CountRecallState(NamedTuple):
    values: object
    queries: object
    counts: object
    t: object


class CountRecall(JaxPort):
    """Independent value/query decks. Reward +-1/(N-1) for naming how often the
    PREVIOUS query card has been dealt (including the card dealt with it)."""

    SETTINGS = dict(Easy=(1, 'colors'), Medium=(2, 'suits'), Hard=(4, 'ranks'))

    def __init__(self, env_id, difficulty):
        decks, kind = self.SETTINGS[difficulty]
        self.env_id, self.family, self.port_id = env_id, 'CountRecall', f'bench-jax-CountRecall{difficulty}'
        self.values = card_values(decks, kind)
        self.n_cards = 52 * decks
        self.distinct = int(len(set(self.values.tolist())))
        self.max_count = self.n_cards // self.distinct
        self.horizon = self.n_cards - 1
        self.action_dim = self.max_count + 1
        self.obs_shape = (2 * self.distinct,)

    def obs(self, value, query):
        import jax.numpy as jnp
        return jnp.concatenate([one_hot(value, self.distinct), one_hot(query, self.distinct)])

    def reset(self, key):
        import jax
        import jax.numpy as jnp
        vk, qk = jax.random.split(key)
        cards = jnp.asarray(self.values, jnp.int32)
        values, queries = jax.random.permutation(vk, cards), jax.random.permutation(qk, cards)
        counts = jnp.zeros(self.distinct, jnp.int32).at[values[0]].add(1)
        return self.obs(values[0], queries[0]), CountRecallState(values, queries, counts, jnp.int32(0))

    def transition(self, key, state, action):
        import jax.numpy as jnp
        t = state.t
        previous_count = state.counts[state.queries[t]]
        nxt = jnp.minimum(t + 1, self.n_cards - 1)
        value, query = state.values[nxt], state.queries[nxt]
        counts = state.counts.at[value].add(1)
        reward = jnp.where(action == previous_count, 1., -1.) / self.horizon
        done = t + 1 >= self.n_cards - 1
        return self.obs(value, query), state._replace(counts=counts, t=t + 1), jnp.float32(reward), done


class MineSweeperState(NamedTuple):
    grid: object      # 0 clear, 1 mine, 2 viewed (flattened row-major)
    neighbours: object  # mines in the 3x3 block INCLUDING the cell itself
    t: object


class MineSweeper(JaxPort):
    SETTINGS = dict(Easy=((4, 4), 2), Medium=((6, 6), 6), Hard=((8, 8), 10))

    def __init__(self, env_id, difficulty):
        (self.rows, self.cols), self.mines = self.SETTINGS[difficulty]
        self.env_id, self.family, self.port_id = env_id, 'MineSweeper', f'bench-jax-MineSweeper{difficulty}'
        self.horizon = self.rows * self.cols - self.mines
        self.action_dim = self.rows * self.cols
        self.n_obs = min(self.mines + 1, 10)
        self.obs_shape = (self.n_obs,)
        self.success = 1. / self.horizon
        self.fail = -0.5 - self.success
        self.bad = -0.5 / (self.horizon - 2)

    def from_mines(self, grid):
        import jax.numpy as jnp
        grid = jnp.asarray(grid, jnp.int32).reshape(-1)
        padded = jnp.pad(grid.reshape(self.rows, self.cols), 1)
        neighbours = sum(padded[1 + i:1 + i + self.rows, 1 + j:1 + j + self.cols]
                         for i in (-1, 0, 1) for j in (-1, 0, 1))
        return MineSweeperState(grid, neighbours.reshape(-1), jnp.int32(0))

    def reset(self, key):
        import jax
        import jax.numpy as jnp
        cells = jax.random.choice(key, self.action_dim, (self.mines,), replace=False)
        return one_hot(0, self.n_obs), self.from_mines(jnp.zeros(self.action_dim, jnp.int32).at[cells].set(1))

    def transition(self, key, state, action):
        import jax.numpy as jnp
        cell = state.grid[action]
        mine, viewed = cell == 1, cell == 2
        reward = jnp.where(mine, self.fail, jnp.where(viewed, self.bad, self.success))
        grid = jnp.where(mine | viewed, state.grid, state.grid.at[action].set(2))
        t = state.t + 1
        done = mine | jnp.all(grid != 0) | (t >= self.horizon)
        return one_hot(state.neighbours[action], self.n_obs), state._replace(grid=grid, t=t), jnp.float32(reward), done


class CartPoleState(NamedTuple):
    physics: object  # x, x_dot, theta, theta_dot
    t: object


class VelocityOnlyCartPole(JaxPort):
    """Gymnasium CartPole-v1 Euler dynamics (float32 here). Reward 1/L every
    step, done on failure or after L steps. Observation is (x_dot, theta_dot),
    EXCEPT the reset observation, which POPGym reports as (x, theta)."""

    LENGTHS = dict(Easy=200, Medium=400, Hard=600)

    def __init__(self, env_id, difficulty):
        self.env_id, self.family, self.port_id = env_id, 'VelocityOnlyCartpole', f'bench-jax-VelocityOnlyCartPole{difficulty}'
        self.horizon = self.LENGTHS[difficulty]
        self.action_dim = 2
        self.obs_shape = (2,)

    def reset(self, key):
        import jax
        import jax.numpy as jnp
        physics = jax.random.uniform(key, (4,), jnp.float32, -0.05, 0.05)
        return physics[jnp.asarray([0, 2])], CartPoleState(physics, jnp.int32(0))

    def transition(self, key, state, action):
        import jax.numpy as jnp
        gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        total_mass, polemass_length = masspole + masscart, masspole * length
        x, x_dot, theta, theta_dot = state.physics
        force = jnp.where(action == 1, force_mag, -force_mag)
        cos, sin = jnp.cos(theta), jnp.sin(theta)
        temp = (force + polemass_length * theta_dot ** 2 * sin) / total_mass
        thetaacc = (gravity * sin - cos * temp) / (length * (4.0 / 3.0 - masspole * cos ** 2 / total_mass))
        xacc = temp - polemass_length * thetaacc * cos / total_mass
        x, x_dot = x + tau * x_dot, x_dot + tau * xacc
        theta, theta_dot = theta + tau * theta_dot, theta_dot + tau * thetaacc
        physics = jnp.stack([x, x_dot, theta, theta_dot])
        t = state.t + 1
        failed = (jnp.abs(x) > 2.4) | (jnp.abs(theta) > 12 * 2 * jnp.pi / 360)
        done = failed | (t >= self.horizon)
        return jnp.stack([x_dot, theta_dot]), CartPoleState(physics, t), jnp.float32(1. / self.horizon), done


NEW_PORTS = dict(Autoencode=Autoencode, CountRecall=CountRecall, MineSweeper=MineSweeper,
                 VelocityOnlyCartpole=VelocityOnlyCartPole)


class HostEnv:
    """Host Gymnasium POPGym task with the previous suite's encoding.

    Same metadata as the JAX ports so the identical networks are built. Only
    the collection/evaluation loops differ (popgym_bench.HostRunner)."""

    is_host = True

    def __init__(self, env_id):
        self.env_id = env_id
        self.family, _ = split_env_id(env_id)
        self.port_id = None
        env = self.make()
        self.action_dim = int(env.action_space.n)
        self.obs_shape = tuple(env.observation_space.shape)
        self.horizon = int(env.unwrapped.max_episode_length)
        env.close()

    def make(self):
        import warnings
        from popgym_suite import make_env
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return make_env(self.env_id)
