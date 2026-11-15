"""Real Gymnax collection with untouched boundaries and exact call accounting."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .replay import append_episode, empty_episode


class Rollout(NamedTuple):
    env_state: object
    pending: dict[str, jax.Array]
    age: jax.Array
    calls: jax.Array
    completed: jax.Array
    key: jax.Array


def start_rollout(env, key, obs_shape, max_steps):
    key, reset_key = jax.random.split(key)
    _, env_state = env.reset(reset_key, env.default_params)
    return Rollout(env_state, empty_episode(obs_shape, max_steps), jnp.int32(0),
                   jnp.int32(0), jnp.int32(0), key)


def uniform_probabilities(observation, action_dim, mask_fn):
    legal = (jnp.ones(action_dim, jnp.bool_) if mask_fn is None else mask_fn(observation).astype(bool))
    # Unlike padded network rows, a real pre-action state must have legal actions.
    # A zero denominator remains a visible failure, never an invented policy.
    return legal.astype(jnp.float32) / legal.sum()


def collect_step(replay, rollout, env, policy, *, action_dim, mask_fn,
                 uniform=False, collector_version=0):
    """Collect precisely one real transition; never terminate at a host boundary.

    policy(obs, raw_pending, age) computes the current action probabilities
    from the true raw episode prefix. It must not use latent environment state.
    Uniform collection has a fixed known full-support law on legal actions.
    """
    key, ak, ek = jax.random.split(rollout.key, 3)
    observation = env.get_obs(rollout.env_state, env.default_params)
    probabilities = (uniform_probabilities(observation, action_dim, mask_fn) if uniform
                     else policy(observation, rollout.pending, rollout.age))
    action = jax.random.categorical(ak, jnp.log(probabilities)).astype(jnp.int32)
    _, env_state, reward, done, _ = env.step(ek, rollout.env_state, action, env.default_params)
    transition = dict(observations=observation.astype(jnp.float32), actions=action,
                      rewards=reward.astype(jnp.float32), terminated=done,
                      behaviour_probs=probabilities[action],
                      collector_version=jnp.asarray(-1 if uniform else collector_version, jnp.int32))
    pending = {name: values.at[rollout.age].set(transition[name])
               for name, values in rollout.pending.items()}
    length = rollout.age + 1
    replay = jax.lax.cond(done, lambda r: append_episode(r, pending, length), lambda r: r, replay)
    empty = empty_episode(observation.shape, pending["actions"].shape[0])
    pending = {name: jnp.where(done, empty[name], value) for name, value in pending.items()}
    return replay, Rollout(env_state, pending, jnp.where(done, 0, length),
                           rollout.calls + 1, rollout.completed + done.astype(jnp.int32), key)
