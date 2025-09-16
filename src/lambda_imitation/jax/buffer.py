from typing import Callable, NamedTuple

import gymnax
import jax
import jax.numpy as jnp
from numpy import float32


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
    recalculate_hidden_states: Callable


def create_buffer(
    observation_space_size,
    action_space_size,
    hidden_state_size,
    size: int,
    sampling_size: int,
    use_importance_sampling: bool,
):
    buffer = Buffer(
        observations=jnp.zeros((size, observation_space_size)),
        hidden_states=jnp.zeros((size, hidden_state_size)),
        actions=jnp.zeros((size, action_space_size)),
        rewards=jnp.zeros((size,)),
        returns=jnp.zeros((size,)),
        behavior_probs=jnp.ones((size,)),
        policy_probs=jnp.zeros((size,)),
        importance_factors=jnp.zeros((size,)),
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
        action_prob: float,
        reward: float,
        terminated: bool,
        gamma: float,
        calculate_return: bool,
    ) -> Buffer:
        pos = (buffer.pos) % buffer.size
        observations = buffer.observations.at[pos].set(obs)
        hidden_states = buffer.hidden_states.at[pos].set(hidden_state)
        actions = buffer.actions.at[pos].set(action)
        rewards = buffer.rewards.at[pos].set(reward)
        behavior_probs = buffer.behavior_probs.at[pos].set(action_prob)
        policy_probs = buffer.policy_probs.at[pos].set(action_prob)
        terminated_arr = buffer.terminated.at[pos].set(terminated)
        # sampling_ok = buffer.sampling_ok.at[
        #     jnp.array([(pos - 1) % buffer.size, pos])
        # ].set((buffer.pos > 0 and not calculate_return, terminated))
        returns = buffer.returns
        sampling_ok = buffer.sampling_ok.at[pos].set(False)
        importance_factors = buffer.importance_factors
        if calculate_return:
            returns = (
                rewards + (1 - terminated_arr) * jnp.roll(buffer.returns, -1) * gamma
            )
            sampling_ok = sampling_ok | jnp.roll(sampling_ok, -1)
            importance_factors = jax.lax.select(
                terminated_arr,
                jnp.ones_like(terminated_arr, dtype=jnp.float32),
                jnp.roll(importance_factors, -1),
            )
            importance_factors *= policy_probs / behavior_probs
        sampling_ok = sampling_ok.at[
            jnp.array([buffer.size - 1, (pos - 1) % buffer.size, pos])
        ].set(
            (
                jnp.logical_and(buffer.pos >= buffer.size, sampling_ok[-1]),
                jnp.logical_and(
                    buffer.pos > 0, jnp.array(not calculate_return, dtype=jnp.bool)
                ),
                terminated,
            )
        )

        # jax.debug.print("{x}", x=sampling_ok[-1])
        return Buffer(
            observations=observations,
            hidden_states=hidden_states,
            actions=actions,
            rewards=rewards,
            returns=returns,
            behavior_probs=behavior_probs,
            policy_probs=policy_probs,
            importance_factors=importance_factors,
            terminated=terminated_arr,
            sampling_ok=sampling_ok,
            pos=buffer.pos + 1,
            size=buffer.size,
        )

    def sample(buffer: Buffer, key, use_importance_sampling: bool = False):
        probs = jnp.astype(buffer.sampling_ok, jnp.float32)
        probs = jax.lax.cond(
            use_importance_sampling,
            lambda: probs * buffer.importance_factors,
            lambda: probs,
        )
        probs = probs / probs.sum()
        indices = jax.random.choice(key, size, (sampling_size,), p=probs)

        return (
            BufferSample(
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
            ),
            indices,
        )

    def recalculate_hidden_states(buffer: Buffer, feature_extractor):
        def scan_fun(carry, xs):
            pos = xs
            _, hidden_state = feature_extractor(
                buffer.hidden_states[pos], buffer.observations[pos]
            )
            hidden_state *= 1 - buffer.terminated[(pos + 1) % size]
            return None, hidden_state

        indices = (buffer.pos + jnp.arange(1, size, dtype=jnp.int16)) % size
        _, next_hidden_states = jax.lax.scan(scan_fun, None, indices)
        return Buffer(
            observations=buffer.observations,
            hidden_states=buffer.hidden_states.at[(indices + 1) % size].set(
                next_hidden_states
            ),
            actions=buffer.actions,
            rewards=buffer.rewards,
            returns=buffer.returns,
            behavior_probs=buffer.behavior_probs,
            policy_probs=buffer.policy_probs,
            importance_factors=buffer.importance_factors,
            terminated=buffer.terminated,
            sampling_ok=buffer.sampling_ok,
            pos=buffer.pos,
            size=buffer.size,
        )

    return buffer, BufferFunctions(
        add=add, sample=sample, recalculate_hidden_states=recalculate_hidden_states
    )


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
