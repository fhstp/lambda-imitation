"""Whole-episode FIFO replay with a transition, not episode-slot, capacity.

In-flight episodes are kept separately. Only a genuine environment boundary
admits an episode. Eviction removes whole oldest episodes until the configured
transition budget fits; no orphaned prefix or invented terminal is retained.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EpisodeReplay(NamedTuple):
    info: dict[str, jax.Array]
    lengths: jax.Array
    write: jax.Array
    oldest: jax.Array
    transitions: jax.Array
    episodes: jax.Array


def empty_episode(obs_shape, max_steps):
    return {
        "observations": jnp.zeros((max_steps, *obs_shape), jnp.float32),
        "actions": jnp.zeros(max_steps, jnp.int32),
        "rewards": jnp.zeros(max_steps, jnp.float32),
        "terminated": jnp.ones(max_steps, jnp.bool_),
        "behaviour_probs": jnp.ones(max_steps, jnp.float32),
        "collector_version": jnp.full(max_steps, -1, jnp.int32),
    }


def create_replay(obs_shape, max_steps, capacity):
    if capacity < max_steps or max_steps < 1:
        raise ValueError("replay capacity must accommodate a whole maximal episode")
    # At most capacity one-step episodes can be simultaneously retained. Padding
    # uses extra storage, but the admitted *active transition* capacity is exact.
    info = {name: jnp.broadcast_to(value, (capacity, *value.shape))
            for name, value in empty_episode(obs_shape, max_steps).items()}
    return EpisodeReplay(info, jnp.zeros(capacity, jnp.int32),
                         *[jnp.int32(0) for _ in range(4)])


def append_episode(replay, episode, length):
    capacity = replay.lengths.shape[0]

    def evict(state):
        n = state.lengths[state.oldest]
        return state._replace(
            lengths=state.lengths.at[state.oldest].set(0),
            oldest=(state.oldest + 1) % capacity,
            transitions=state.transitions - n, episodes=state.episodes - 1)

    replay = jax.lax.while_loop(lambda r: r.transitions + length > capacity, evict, replay)
    return replay._replace(
        info={name: array.at[replay.write].set(episode[name]) for name, array in replay.info.items()},
        lengths=replay.lengths.at[replay.write].set(length),
        write=(replay.write + 1) % capacity,
        transitions=replay.transitions + length, episodes=replay.episodes + 1)


class PackedAnchors(NamedTuple):
    info: dict[str, jax.Array]
    lengths: jax.Array
    episode_indices: jax.Array
    owners: jax.Array
    times: jax.Array
    probabilities: jax.Array
    valid: jax.Array


def packed_anchors(replay, key, *, gamma, measure, episode_batch, score_count):
    """Factor the global anchor law without materializing capacity*max_steps P.

    P(episode) is proportional to its SUM of time weights, not uniform over
    episodes. Conditional time draws complete the same global law as sampling.py.
    Sharing a drawn episode across multiple scored anchors saves reconstruction;
    every anchor has the correct marginal law, but they are NOT independent.
    No extra importance weighting should be applied to these direct draws.

    This function can also operate on adaptive replay. That mode estimates a
    discounted empirical episode mixture, NOT a single stationary-mu posterior.
    """
    if measure not in ("uniform", "discounted"):
        raise ValueError("unknown starting measure")
    max_steps = replay.info["observations"].shape[1]
    ages = jnp.arange(max_steps)
    time_weights = gamma ** ages if measure == "discounted" else jnp.ones(max_steps)
    mass_by_length = jnp.concatenate([jnp.zeros(1), jnp.cumsum(time_weights)])
    masses = mass_by_length[replay.lengths]
    total_mass = masses.sum()
    ek, tk = jax.random.split(key)
    indices = jax.random.choice(ek, replay.lengths.shape[0], shape=(episode_batch,),
                                p=masses / total_mass, replace=True)
    chosen_lengths = replay.lengths[indices]
    owners = jnp.arange(score_count) % episode_batch
    logits = jnp.where(ages[None, :] < chosen_lengths[owners, None],
                       jnp.log(time_weights)[None, :], -jnp.inf)
    times = jax.random.categorical(tk, logits).astype(jnp.int32)
    probabilities = time_weights[times] / total_mass
    return PackedAnchors(
        {name: array[indices] for name, array in replay.info.items()}, chosen_lengths,
        indices, owners, times, probabilities, ages[None, :] < chosen_lengths[:, None])


def select(values, batch, count=None):
    """Select scored anchors from episode-batch-major (E,T,...) arrays."""
    count = batch.times.shape[0] if count is None else count
    return values[batch.owners[:count], batch.times[:count]]
