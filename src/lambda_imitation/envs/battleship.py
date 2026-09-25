"""Battleship adapted from Allen et al., brownirl/lambda_discrepancy (Apache-2.0).

Vendored from lambda-envs dbd9dfc069430b3c4c1694bf18b8e3b14eda18df.
Local adaptations expose the last-shot outcome, an action mask, configurable
ships/rewards, and a Gymnax interface. Publication cleanup removes unused reset
bookkeeping and fixes rectangular placement bounds (square boards unchanged).
See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from gymnax.environments.environment import Environment, EnvParams
from jax import lax, random


@chex.dataclass
class BattleShipState:
    hits_misses: jax.Array  # 0 unqueried, 1 miss, 2 hit
    board: jax.Array
    last_hit_miss: jax.Array


@partial(jax.jit, static_argnames=["ship_length"])
def place_ship_randomly(key, board, ship_length):
    """Uniform orientation, then uniform nonoverlapping anchor in that orientation."""
    def check_vert(y, x):
        return jnp.all(lax.dynamic_slice(board, (y, x), (ship_length, 1)) == 0)

    def check_horz(y, x):
        return jnp.all(lax.dynamic_slice(board, (y, x), (1, ship_length)) == 0)

    vert = jax.vmap(jax.vmap(check_vert, in_axes=[None, 0]), in_axes=[0, None])(
        jnp.arange(board.shape[0] - ship_length + 1), jnp.arange(board.shape[1]))
    horz = jax.vmap(jax.vmap(check_horz, in_axes=[None, 0]), in_axes=[0, None])(
        jnp.arange(board.shape[0]), jnp.arange(board.shape[1] - ship_length + 1))
    pose_key, choice_key = random.split(key)
    pose = random.bernoulli(pose_key)

    def position(matrix):
        idx = random.categorical(choice_key, jnp.where(matrix.ravel(), 0, -jnp.inf), shape=())
        return jnp.divmod(idx, matrix.shape[1])

    vp, hp = position(vert), position(horz)
    vb = lax.dynamic_update_slice(board, jnp.ones((ship_length, 1), dtype=int), vp)
    hb = lax.dynamic_update_slice(board, jnp.ones((1, ship_length), dtype=int), hp)
    pos, board = lax.cond(pose, lambda: (hp, hb), lambda: (vp, vb))
    return board, pos, pose.astype(int)


class Battleship(Environment):
    """Latest-shot observation, with a legal-action mask in the observation tail.

    Clearing a board in N legal shots gives return ``rows * cols + 1 - N``.
    Ship placement follows the upstream sequential law, not uniform sampling
    over complete boards. The publication configuration is 5×5 with ships (3,2).
    """

    def __init__(self, rows=5, cols=5, ship_lengths=(3, 2)):
        if min(rows, cols) < 1 or not ship_lengths or min(ship_lengths) < 1:
            raise ValueError("board dimensions and ship lengths must be positive")
        if max(ship_lengths) > min(rows, cols):
            raise ValueError("ships must fit in both orientations")
        if sum(ship_lengths) > min(rows, cols):
            raise ValueError("use ships whose total length fits one board dimension")
        self.rows, self.cols = rows, cols
        self.ship_lengths = tuple(ship_lengths)
        self.gamma = 1.0

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=1000)

    @property
    def num_actions(self):
        return self.rows * self.cols

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        return spaces.Box(0, 1, (1 + self.num_actions,))

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params):
        board = jnp.zeros((self.rows, self.cols), dtype=int)
        for length in self.ship_lengths:
            place_key, key = random.split(key)
            board, _, _ = place_ship_randomly(place_key, board, length)
        state = BattleShipState(hits_misses=jnp.zeros_like(board), board=board,
                                 last_hit_miss=jnp.array(False))
        return self.get_obs(state, params), state

    def get_obs(self, state, params=None):
        return jnp.concatenate((jnp.array([state.last_hit_miss], dtype=float),
                                (state.hits_misses == 0).flatten()))

    def is_terminal(self, state, params=None):
        return jnp.all((state.hits_misses == 2) | ~state.board.astype(bool))

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, action, params):
        idx = jnp.unravel_index(action, (self.rows, self.cols))
        hit = state.board[idx]
        state = state.replace(hits_misses=state.hits_misses.at[idx].set(hit + 1),
                              last_hit_miss=hit.astype(bool))
        done = self.is_terminal(state, params)
        reward = (done - 1) + float(self.num_actions) * done
        return lax.stop_gradient(self.get_obs(state, params)), lax.stop_gradient(state), reward, done, {}
