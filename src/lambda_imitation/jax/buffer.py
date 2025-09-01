from typing import Callable, NamedTuple

import gymnax
import jax
import jax.numpy as jnp


class Buffer(NamedTuple):
    observations: jax.Array
    hidden_states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    returns: jax.Array
    behavior_probs: jax.Array
    policy_probs: jax.Array
    importance_factors: jax.Array
    terminated: jax.Array
    sampling_ok: jax.Array
    pos: int
    size: int


class BufferSample(NamedTuple):
    observations: jax.Array
    next_observations: jax.Array
    hidden_states: jax.Array
    next_hidden_states: jax.Array
    actions: jax.Array
    rewards: jax.Array
    returns: jax.Array
    behavior_probs: jax.Array
    policy_probs: jax.Array
    importance_factors: jax.Array
    terminated: jax.Array


class BufferFunctions(NamedTuple):
    add: Callable
    sample: Callable


def create_buffer(
    observation_space_size,
    action_space_size,
    hidden_state_size,
    size: int,
    sampling_size: int,
):
    buffer = Buffer(
        observations=jnp.zeros((size, observation_space_size)),
        hidden_states=jnp.zeros((size, hidden_state_size)),
        actions=jnp.zeros((size, action_space_size)),
        rewards=jnp.zeros((size,)),
        returns=jnp.zeros((size,)),
        behavior_probs=jnp.zeros((size,)),
        policy_probs=jnp.array((size,)),
        importance_factors=jnp.array((size,)),
        terminated=jnp.zeros((size,), dtype=jnp.bool),
        sampling_ok=jnp.zeros((size,), dtype=jnp.bool),
        pos=0,
        size=size,
    )

    def add(
        buffer: Buffer,
        obs: jax.Array,
        hidden_state: jax.Array,
        action: jax.Array,
        reward: float,
        terminated: bool,
    ) -> Buffer:
        pos = (buffer.pos) % buffer.size
        observations = buffer.observations.at[pos].set(obs)
        hidden_states = buffer.hidden_states.at[pos].set(hidden_state)
        actions = buffer.actions.at[pos].set(action)
        rewards = buffer.rewards.at[pos].set(reward)
        terminated_arr = buffer.terminated.at[pos].set(terminated)
        sampling_ok = buffer.sampling_ok.at[
            jnp.array([(pos - 1) % buffer.size, pos])
        ].set((buffer.pos > 0, terminated))
        return Buffer(
            observations=observations,
            hidden_states=hidden_states,
            actions=actions,
            rewards=rewards,
            returns=buffer.returns,
            behavior_probs=buffer.behavior_probs,
            policy_probs=buffer.policy_probs,
            importance_factors=buffer.importance_factors,
            terminated=terminated_arr,
            sampling_ok=sampling_ok,
            pos=buffer.pos + 1,
            size=buffer.size,
        )

    def sample(buffer: Buffer, key):
        probs = buffer.sampling_ok / buffer.sampling_ok.sum()
        indices = jax.random.choice(key, size, (sampling_size,), p=probs)

        return BufferSample(
            observations=jax.lax.stop_gradient(buffer.observations[indices]),
            next_observations=jax.lax.stop_gradient(
                buffer.observations[(indices + 1) % size]
            ),
            hidden_states=jax.lax.stop_gradient(buffer.hidden_states[indices]),
            next_hidden_states=jax.lax.stop_gradient(
                buffer.hidden_states[(indices + 1) % size]
            ),
            actions=jax.lax.stop_gradient(buffer.actions[indices]),
            rewards=jax.lax.stop_gradient(buffer.rewards[indices]),
            returns=jax.lax.stop_gradient(buffer.returns[indices]),
            behavior_probs=jax.lax.stop_gradient(buffer.behavior_probs[indices]),
            policy_probs=jax.lax.stop_gradient(buffer.policy_probs[indices]),
            importance_factors=jax.lax.stop_gradient(
                buffer.importance_factors[indices]
            ),
            terminated=jax.lax.stop_gradient(buffer.terminated[indices]),
        )

    return buffer, BufferFunctions(add=add, sample=sample)


if __name__ == "__main__":
    buffer, functions = create_buffer(3, 2, 0, 10, 4)

    step = 0.0
    buffer = functions.add(
        buffer,
        jnp.array([step, step, step]),
        jnp.zeros((0,)),
        jnp.array([step, step]),
        step,
        False,
    )
    step = 1.0
    buffer = functions.add(
        buffer,
        jnp.array([step, step, step]),
        jnp.zeros((0,)),
        jnp.array([step, step]),
        step,
        False,
    )
    step = 2.0
    buffer = functions.add(
        buffer,
        jnp.array([step, step, step]),
        jnp.zeros((0,)),
        jnp.array([step, step]),
        step,
        True,
    )

    key = jax.random.key(0)
    print(functions.sample(buffer, key))
    print(buffer.sampling_ok)
