# Portions Copyright 2022 InstaDeep Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0; see licenses/Apache-2.0.txt.
"""PocMan adapted from Allen et al., brownirl/lambda_discrepancy (Apache-2.0).

Vendored from lambda-envs dbd9dfc069430b3c4c1694bf18b8e3b14eda18df, including
its corrected wall-below column index. The Gymnax step adapter derives from
InstaDeep's Apache-2.0 Jumanji PacMan; Jumanji supplies the game dynamics.
Publication changes: optional params in get_obs and removal of the demo main.
See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from gymnax.environments.environment import EnvParams, Environment
from jumanji.environments.routing.pac_man import PacMan
from jumanji.environments.routing.pac_man.generator import AsciiGenerator


# (0,0) must remain unreachable: eaten pellets are moved to this sentinel.
SMALLER_GAME_MAP = [
    'XXXXXXXXXXXXXXXXXXX',
    'X  S           S  X',
    'X XX XXX X XXX XX X',
    'XO               OX',
    'X XX X XXXXX X XX X',
    'X    X  TXT  X    X',
    'XXXX XXX X XXX XXXX',
    'XXXX XT G G TX XXXX',
    'XXXX X X   X X XXXX',
    '     X XG GX X     ',
    'XXXX X XXXXX X XXXX',
    'XXXX X       X XXXX',
    'XXXX X XXXXX X XXXX',
    'X                 X',
    'X XX XXX X XXX XX X',
    'XO X S   P   S X OX',
    'XX X X XXXXX X X XX',
    'X    X   X   X    X',
    'X XXXXXX X XXXXXX X',
    'X                 X',
    'XXXXXXXXXXXXXXXXXXX',
]


def generate_los_map(generator):
    """Precompute the upstream north/west/south/east line-of-sight masks."""
    space = np.array(generator.numpy_maze)
    los = np.zeros((*space.shape, 4, *space.shape))
    for y, x in generator.reachable_spaces:
        mask = np.zeros_like(space)
        mask[x, y] = 1.0
        north, south, east, west = (mask.copy() for _ in range(4))
        a = space[:x, y].copy()
        if a.size:
            idx = np.argmin(np.flip(a))
            if idx > 0:
                a[:-idx] = 0
                north[:x, y] = a
        a = space[x + 1:, y].copy()
        if a.size:
            a[np.argmin(a):] = 0
            south[x + 1:, y] = a
        a = space[x, y + 1:].copy()
        if a.size:
            a[np.argmin(a):] = 0
            east[x, y + 1:] = a
        a = space[x, :y].copy()
        if a.size:
            idx = np.argmin(np.flip(a))
            if idx > 0:
                a[:-idx] = 0
                west[x, :y] = a
        los[x, y] = np.stack([north, west, south, east])
    return jnp.array(los)


class PocMan(PacMan, Environment):
    """Eleven sensor values; position and the pellet map remain hidden.

    Player coordinates are (row, column), whereas pellet and ghost coordinate
    arrays use (column, row), following Jumanji's PacMan representation.
    """

    def __init__(self):
        generator = AsciiGenerator(SMALLER_GAME_MAP)
        super().__init__(generator=generator)
        self.line_sight_map = generate_los_map(generator)
        self.gamma = 0.95

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.time_limit)

    @property
    def num_actions(self):
        return 4

    def observation_space(self, params=None):
        return spaces.Box(0, 1, (11,))

    def action_space(self, params=None):
        return spaces.Discrete(4)

    def get_obs(self, state, params=None):
        row, col = state.player_locations.x, state.player_locations.y
        loc = jnp.array([col, row])
        obs = jnp.zeros(11, dtype=int)
        obs = obs.at[0].set(state.grid[jnp.maximum(row - 1, 0), col])
        obs = obs.at[1].set(state.grid[row, jnp.minimum(col + 1, state.grid.shape[1] - 1)])
        obs = obs.at[2].set(state.grid[jnp.minimum(row + 1, state.grid.shape[0] - 1), col])
        obs = obs.at[3].set(state.grid[row, jnp.maximum(col - 1, 0)])
        pellet_dists = jnp.abs(loc[None] - state.pellet_locations).sum(-1)
        obs = obs.at[4].set(jnp.any(pellet_dists <= 1).astype(obs.dtype))
        ghost_dists = jnp.abs(loc[None] - state.ghost_locations).sum(-1)
        obs = obs.at[5].set(jnp.any(ghost_dists <= 2).astype(obs.dtype))
        ghosts = jnp.zeros_like(state.grid).at[
            state.ghost_locations[:, 1], state.ghost_locations[:, 0]].set(1)
        visible = (self.line_sight_map[row, col] * ghosts[None]).sum(-1).sum(-1)
        obs = obs.at[6:10].set(visible.astype(obs.dtype))
        return obs.at[10].set(state.frightened_state_time > 0)

    def is_terminal(self, state, params=None):
        return (state.dead == 1) | (state.pellets == 0)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        return Environment.reset(self, key, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key, params):
        state, _ = PacMan.reset(self, key)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params=None):
        return Environment.step(self, key, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state, action, params):
        # Adapted from Jumanji's environments/routing/pac_man/env.py.
        updated, reward = self._update_state(state, action)
        state = updated.replace(step_count=state.step_count + 1)
        done = (state.step_count >= self.time_limit) | (state.pellets == 0) | (state.dead == 1)
        return (jax.lax.stop_gradient(self.get_obs(state)), jax.lax.stop_gradient(state),
                jnp.asarray(reward), done, {})
