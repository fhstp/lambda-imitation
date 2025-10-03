from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces
from jax import lax


@struct.dataclass
class EnvState:
    pos: chex.Array  # [x, y] position
    start_pos: chex.Array  # Starting position to remember
    time: int


@struct.dataclass
class EnvParams:
    corridor_length: int = 5
    max_steps_in_episode: int = 100


class PartiallyObservableGridworld(environment.Environment):
    """
    Partially Observable Corridor Environment.

    Layout:
         S1
         |
    ---- C ---- ---- ---- ---- C ----
         |                       |
         S2                      T1
                                 |
                                 T2

    - Agent starts at S1 or S2 (up or down from start of corridor)
    - Must navigate through corridor
    - At end, must go up/down to reach T1 or T2
    - Reward +1 if reaches matching terminal (S1->T1 or S2->T2)
    - Reward -1 if reaches wrong terminal
    - Environment is partially observable (agent doesn't see start position)
    """

    def __init__(self, corridor_length: int = 5):
        super().__init__()
        self.corridor_length = corridor_length

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Step the environment."""
        # Actions: 0=up, 1=right, 2=down, 3=left
        moves = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        move = moves[action]

        new_pos = state.pos + move

        # Bounds checking
        # x: -1 to 1 (three rows: up, corridor, down)
        # y: 0 to corridor_length-1 (corridor positions)
        new_x = jnp.clip(new_pos[0], -1, 1)
        new_y = jnp.clip(new_pos[1], 0, params.corridor_length - 1)

        # Enforce valid positions (can't move into walls)
        valid_pos = self._is_valid_position(new_x, new_y, params.corridor_length)

        # Only update position if valid
        final_x = jnp.where(valid_pos, new_x, state.pos[0])
        final_y = jnp.where(valid_pos, new_y, state.pos[1])
        final_pos = jnp.array([final_x, final_y])

        # Check if terminal position reached
        # Terminal positions are at x=-1 or x=1, y=corridor_length-1 (end of corridor)
        at_end_corridor = final_y == params.corridor_length - 1
        at_terminal = (jnp.abs(final_x) == 1) & at_end_corridor

        # Correct terminal: started at x=-1, ended at x=-1 OR started at x=1, ended at x=1
        correct_terminal = (final_x == state.start_pos[0]) & at_terminal
        wrong_terminal = (final_x != state.start_pos[0]) & at_terminal

        # Calculate reward
        reward = jnp.where(correct_terminal, 1.0, jnp.where(wrong_terminal, -1.0, -0.1))

        # Episode ends when reaching any terminal or max steps
        done = at_terminal | (state.time >= params.max_steps_in_episode - 1)

        new_state = EnvState(
            pos=final_pos, start_pos=state.start_pos, time=state.time + 1
        )

        obs = self.get_obs(new_state)
        info = {"discount": self.discount(new_state, params)}

        return obs, new_state, reward, done, info

    def _is_valid_position(self, x: int, y: int, corridor_length: int) -> bool:
        """Check if position is valid in the environment."""
        # Start positions: x=-1 or x=1, y=0
        at_start = (jnp.abs(x) == 1) & (y == 0)

        # Corridor positions: x=0, 0 <= y < corridor_length
        in_corridor = (x == 0) & (y >= 0) & (y < corridor_length)

        # End positions: x=-1 or x=1, y=corridor_length-1
        at_end = (jnp.abs(x) == 1) & (y == corridor_length - 1)

        return at_start | in_corridor | at_end

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment."""
        # Randomly choose starting position (up=-1 or down=1)
        start_x = jax.random.choice(key, jnp.array([-1, 1]))
        start_pos = jnp.array([start_x, 0])

        state = EnvState(pos=start_pos, start_pos=start_pos, time=0)

        obs = self.get_obs(state)
        return obs, state

    def get_obs(self, state: EnvState) -> chex.Array:
        """
        Get observation (partially observable - only current position).
        Returns one-hot encoding of current position.
        Positions:
        - Start up (x=-1, y=0): 0
        - Start down (x=1, y=0): 1
        - Corridor (x=0, y=0 to corridor_length-1): 2 to corridor_length+1
        - End up (x=-1, y=corridor_length-1): corridor_length+2
        - End down (x=1, y=corridor_length-1): corridor_length+3
        Total: corridor_length + 4 positions
        """
        y = state.pos[1]
        x = state.pos[0]

        # Calculate flat index
        idx = jnp.where(
            (jnp.abs(x) == 1) & (y == 0),
            jnp.where(x == -1, 0, 1),  # Start positions
            jnp.where(
                x == 0,
                2 + y,  # Corridor positions
                jnp.where(
                    x == -1, self.corridor_length + 2, self.corridor_length + 3
                ),  # End positions
            ),
        )

        # One-hot encoding
        obs_size = self.corridor_length + 4
        obs = jax.nn.one_hot(idx, obs_size)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check if state is terminal."""
        at_end_corridor = state.pos[1] == params.corridor_length - 1
        at_terminal = (jnp.abs(state.pos[0]) == 1) & at_end_corridor
        return at_terminal

    @property
    def name(self) -> str:
        return "PartiallyObservableCorridorEnv"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space: up, right, down, left."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space: one-hot encoding of position."""
        obs_size = params.corridor_length + 4
        return spaces.Box(0, 1, (obs_size,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space specification."""
        return spaces.Dict(
            {
                "pos": spaces.Box(-1, params.corridor_length - 1, (2,), jnp.int32),
                "start_pos": spaces.Box(-1, 1, (2,), jnp.int32),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )


# Example usage
if __name__ == "__main__":
    # Create environment
    env = PartiallyObservableCorridorEnv(corridor_length=5)
    params = env.default_params

    # Reset
    rng = jax.random.PRNGKey(0)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset_env(reset_rng, params)

    print(f"Initial observation shape: {obs.shape}")
    print(f"Starting position: {state.pos}")
    print(f"Start was at: {state.start_pos}")
    print(f"Observation: {obs}")

    # Take a few steps
    for i in range(10):
        rng, step_rng = jax.random.split(rng)
        action = jax.random.randint(step_rng, (), 0, 4)
        obs, state, reward, done, info = env.step_env(step_rng, state, action, params)
        print(
            f"\nStep {i+1}: Action={action}, Pos={state.pos}, Reward={reward}, Done={done}"
        )

        if done:
            print("Episode finished!")
            break
