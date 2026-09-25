"""Independent JAX/gymnax version of POPGym's partially observed Minesweeper.

Reference: https://github.com/proroklab/popgym/blob/master/popgym/envs/minesweeper.py
There is no POPGym dependency. Actions are flat row-major cell indices. Each
action queries just ONE square (including when its clue is zero); there is no
flood fill, flag action, legal-action mask, or first-click protection.

For N = rows * cols and S = N - num_mines, a new safe query rewards +1/S,
a repeated safe query rewards -0.5/(S-2), and a mine rewards -0.5-1/S and ends
the episode. Clearing all S safe cells also ends it; otherwise action S times
out. Configurations must have at least three safe cells so the reference's
repeat penalty has a positive, nonzero denominator.

Let K = min(8, num_mines) + 1 be the number of possible safe clues:

The observation is a float32 one-hot vector of size K+1. Indices 0..K-1
encode the last queried clue, and index K denotes reset/no query. Position
is absent: a recurrent agent needs the PREVIOUS ACTION to locate the clue.

Thus 6x6/6 has observation dimension 8, and the small 4x4/2 smoke
configuration has dimension 4. ``get_obs`` reconstructs exactly the observation
belonging to a state, including the reset state returned by gymnax auto-reset.

Interface differences from POPGym: one-hot rather than integer observations,
a distinct reset marker rather than an ambiguous zero clue, flat rather than
two-coordinate actions, and gymnax's combined done + auto-reset convention.
``info['terminated']`` / ``info['truncated']`` preserve the termination reason.
Neighbor counts exclude the center even on mines (POPGym includes the mine
itself there); this only changes low-level ``step_env``'s terminal mine clue,
which public ``step`` replaces with the reset observation. Rewards and episode
boundaries match the reference for the supported configurations.
"""

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class MineSweeperParams(NamedTuple):
    max_steps_in_episode: int


class MineSweeperState(NamedTuple):
    """Immutable pytree; all fields are JAX arrays.

    ``mines`` and ``viewed`` are bool (rows, cols) grids; ``neighbor_counts``
    is an int32 grid. Only safe queries set ``viewed``. ``timestep`` and
    ``last_value`` are int32 scalars, with last_value=-1 at reset. The bool
    scalar ``hit_mine`` records termination for ``is_terminal``/``discount``.
    """

    mines: jax.Array
    neighbor_counts: jax.Array
    viewed: jax.Array
    timestep: jax.Array
    last_value: jax.Array
    hit_mine: jax.Array


def _count_neighbors(grid: jax.Array) -> jax.Array:
    """Sum the ordinary eight-neighborhood, zero padded at board boundaries."""
    values = grid.astype(jnp.int32)
    return jax.lax.reduce_window(
        values, jnp.int32(0), jax.lax.add, (3, 3), (1, 1), "SAME"
    ) - values


class MineSweeper(environment.Environment):
    """Single-query, partially observable Minesweeper.

    ``reset`` and ``step`` are inherited unchanged from gymnax. Use
    ``step_env`` to inspect a terminal board before auto-reset. Params can
    override the timeout; reward scales stay tied to the board's default
    ``episode_length``.
    """

    def __init__(self, rows=6, cols=6, num_mines=6):
        if any(isinstance(x, bool) or not isinstance(x, Integral)
               for x in (rows, cols, num_mines)):
            raise ValueError("rows, cols and num_mines must be integers")
        self.rows, self.cols, self.num_mines = int(rows), int(cols), int(num_mines)
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0 <= self.num_mines < self.num_actions:
            raise ValueError("num_mines must be in [0, rows * cols)")
        self.episode_length = self.num_actions - self.num_mines
        if self.episode_length <= 2:
            raise ValueError("at least three safe cells are required by the repeat penalty")
        self.obs_requires_prev_action = True
        self.num_clues = min(8, self.num_mines) + 1
        self.success_reward_scale = 1.0 / self.episode_length
        self.fail_reward_scale = -0.5 - self.success_reward_scale
        self.bad_action_reward_scale = -0.5 / (self.episode_length - 2)

    @property
    def default_params(self) -> MineSweeperParams:
        return MineSweeperParams(max_steps_in_episode=self.episode_length)

    @property
    def num_actions(self) -> int:
        return self.rows * self.cols

    def reset_env(self, key, params):
        locations = jax.random.permutation(key, self.num_actions)[:self.num_mines]
        mines = jnp.zeros(self.num_actions, dtype=jnp.bool_).at[locations].set(True)
        mines = mines.reshape(self.rows, self.cols)
        state = MineSweeperState(
            mines=mines,
            neighbor_counts=_count_neighbors(mines),
            viewed=jnp.zeros_like(mines),
            timestep=jnp.int32(0),
            last_value=jnp.int32(-1),
            hit_mine=jnp.bool_(False),
        )
        return self.get_obs(state, params), state

    def step_env(self, key, state, action, params):
        action = jnp.asarray(action, dtype=jnp.int32).reshape(())
        hit_mine = state.mines.reshape(-1)[action]
        repeat = state.viewed.reshape(-1)[action]
        reward = jnp.where(
            hit_mine, self.fail_reward_scale,
            jnp.where(repeat, self.bad_action_reward_scale, self.success_reward_scale),
        ).astype(jnp.float32)
        viewed = state.viewed.reshape(-1).at[action].set(~hit_mine)
        next_state = state._replace(
            viewed=viewed.reshape(self.rows, self.cols),
            timestep=state.timestep + jnp.int32(1),
            last_value=state.neighbor_counts.reshape(-1)[action],
            hit_mine=hit_mine,
        )
        revealed = jnp.sum(next_state.viewed, dtype=jnp.int32)
        success = revealed == self.episode_length
        terminated = hit_mine | success
        truncated = (next_state.timestep >= params.max_steps_in_episode) & ~terminated
        done = terminated | truncated

        # Decision diagnostics use the history BEFORE this action. Counts and
        # outcomes use the resulting board and survive gymnax's auto-reset.
        info = self.diagnostics(state, action)
        info.update(
            safe_cells_revealed=revealed.astype(jnp.float32),
            success=success.astype(jnp.float32),
            terminated=terminated,
            truncated=truncated,
            discount=(~done).astype(jnp.float32),
        )
        return self.get_obs(next_state, params), next_state, reward, done, info

    def get_obs(self, state, params=None):
        clue = jnp.where(state.last_value < 0, self.num_clues, state.last_value)
        return jax.nn.one_hot(clue, self.num_clues + 1, dtype=jnp.float32)

    def diagnostics(self, state, action) -> dict[str, jax.Array]:
        """Four float32 scalars describing the current, PRE-ACTION history.

        An opportunity is an unviewed neighbor of a previously viewed zero.
        Neither the mine grid nor unobserved clues determine this set. The
        cumulative count here is before the action; ``step``'s info updates
        that count to include the action, and adds success/termination flags.
        """
        action = jnp.asarray(action, dtype=jnp.int32).reshape(())
        seen_zero = state.viewed & (state.neighbor_counts == 0)
        known_safe = ~state.viewed & (_count_neighbors(seen_zero) > 0)
        opportunity = jnp.any(known_safe)
        return {
            "repeat_action": state.viewed.reshape(-1)[action].astype(jnp.float32),
            "known_safe_opportunity": opportunity.astype(jnp.float32),
            "known_safe_taken": (opportunity & known_safe.reshape(-1)[action]).astype(jnp.float32),
            "safe_cells_revealed": jnp.sum(state.viewed, dtype=jnp.int32).astype(jnp.float32),
        }

    def is_terminal(self, state, params=None):
        if params is None:
            params = self.default_params
        cleared = jnp.sum(state.viewed, dtype=jnp.int32) == self.episode_length
        return state.hit_mine | cleared | (state.timestep >= params.max_steps_in_episode)

    def action_space(self, params=None):
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params=None):
        dim = self.num_clues + 1
        return spaces.Box(0.0, 1.0, (dim,), jnp.float32)

    def state_space(self, params=None):
        if params is None:
            params = self.default_params
        shape = (self.rows, self.cols)
        return spaces.Dict({
            "mines": spaces.Box(0, 1, shape, jnp.bool_),
            "neighbor_counts": spaces.Box(0, self.num_clues - 1, shape, jnp.int32),
            "viewed": spaces.Box(0, 1, shape, jnp.bool_),
            "timestep": spaces.Box(0, params.max_steps_in_episode, (), jnp.int32),
            "last_value": spaces.Box(-1, self.num_clues - 1, (), jnp.int32),
            "hit_mine": spaces.Discrete(2),
        })
