"""Transition ring with an O(1) completed-episode FIFO index.

Index eviction is insertion-ordered: one overwritten transition can invalidate
at most one complete episode. Sampling costs O(batch * episode_length), never
an O(capacity) scan. All transitions in an episode have the same reuse count.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class EpisodeBatch(NamedTuple):
    observations: jax.Array  # time, episode, hit + legal mask
    actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    behavior: jax.Array
    versions: jax.Array
    episode_ids: jax.Array
    timesteps: jax.Array
    insertion_ids: jax.Array
    valid: jax.Array
    integrity: jax.Array


class EpisodeReplay(NamedTuple):
    data: dict
    inserted: jax.Array
    starts: jax.Array
    lengths: jax.Array
    reuse: jax.Array
    head: jax.Array
    count: jax.Array
    pending_start: jax.Array
    pending_length: jax.Array
    pending_episode: jax.Array
    pending_ok: jax.Array
    evicted: jax.Array
    rejected: jax.Array
    episode_draws: jax.Array
    presentations: jax.Array  # low/high uint32; 12.8B exceeds int32 under x64=False
    unique_transitions: jax.Array


def empty_replay(capacity=200_000, actions=100):
    if capacity < 1 or actions < 1:
        raise ValueError("positive capacity and action count required")
    i = lambda: jnp.asarray(0, jnp.int32)
    data = {
        "observations": jnp.zeros((capacity, actions + 1), jnp.bool_),
        "actions": jnp.zeros(capacity, jnp.int32),
        "rewards": jnp.zeros(capacity, jnp.float32),
        "dones": jnp.zeros(capacity, jnp.bool_),
        "behavior": jnp.ones(capacity, jnp.float32),
        "versions": jnp.zeros(capacity, jnp.int32),
        "episode_ids": jnp.zeros(capacity, jnp.int32),
        "timesteps": jnp.zeros(capacity, jnp.int32),
        "insertion_ids": jnp.full(capacity, -1, jnp.int32),
    }
    return EpisodeReplay(data, i(), jnp.zeros(capacity, jnp.int32),
                         jnp.zeros(capacity, jnp.int32), jnp.zeros(capacity, jnp.uint32),
                         i(), i(), i(), i(), i(), jnp.asarray(True), i(), i(), i(),
                         jnp.zeros(2, jnp.uint32), i())


def add_count(words, value):
    value = jnp.asarray(value, jnp.uint32)
    low = words[0] + value
    return jnp.stack((low, words[1] + (low < words[0]).astype(jnp.uint32)))


def insert(replay, observation, action, reward, done, behavior, version,
           episode_id, timestep, natural_terminal):
    capacity = replay.starts.size
    slot = replay.inserted % capacity
    oldest = jnp.maximum(0, replay.inserted + 1 - capacity)
    evict = (replay.count > 0) & (replay.starts[replay.head] < oldest)
    head = (replay.head + evict.astype(jnp.int32)) % capacity
    count = replay.count - evict.astype(jnp.int32)
    start = jnp.where(timestep == 0, replay.inserted, replay.pending_start)
    length = jnp.where(timestep == 0, 0, replay.pending_length)
    same_episode = (timestep == 0) | (episode_id == replay.pending_episode)
    action_ok = (action >= 0) & (action < observation.size - 1)
    legal = observation[1:][jnp.clip(action, 0, observation.size - 2)].astype(bool)
    ok = (jnp.where(timestep == 0, replay.pending_length == 0, replay.pending_ok)
          & same_episode & (timestep == length) & (replay.inserted == start + length)
          & action_ok & legal & jnp.isfinite(behavior) & (behavior > 0) & (behavior <= 1))
    length += 1
    complete = done & natural_terminal & ok & (length <= observation.size - 1) & (start >= oldest)
    row = dict(observations=observation.astype(jnp.bool_), actions=action,
               rewards=reward.astype(jnp.float32), dones=done, behavior=behavior,
               versions=version, episode_ids=episode_id, timesteps=timestep,
               insertion_ids=replay.inserted)
    data = {name: array.at[slot].set(row[name]) for name, array in replay.data.items()}
    end = (head + count) % capacity
    starts = replay.starts.at[end].set(jnp.where(complete, start, replay.starts[end]))
    lengths = replay.lengths.at[end].set(jnp.where(complete, length, replay.lengths[end]))
    reuse = replay.reuse.at[end].set(jnp.where(complete, jnp.uint32(0), replay.reuse[end]))
    return replay._replace(
        data=data, inserted=replay.inserted + 1, starts=starts, lengths=lengths,
        reuse=reuse, head=head, count=count + complete.astype(jnp.int32),
        pending_start=jnp.where(done, replay.inserted + 1, start),
        pending_length=jnp.where(done, 0, length), pending_episode=episode_id,
        pending_ok=jnp.where(done, True, ok), evicted=replay.evicted + evict,
        rejected=replay.rejected + (done & ~complete),
    )


def sample(replay, key, batch_size=128, length=100):
    draws = jax.random.randint(key, (batch_size,), 0, jnp.maximum(replay.count, 1))
    slots = (replay.head + draws) % replay.starts.size
    starts, lengths = replay.starts[slots], replay.lengths[slots]
    times = jnp.arange(length)[:, None]
    ids = starts[None, :] + times
    valid = times < lengths[None, :]
    rows = {name: value[ids % replay.starts.size] for name, value in replay.data.items()}
    integrity = ((replay.count > 0) & jnp.all((lengths > 0) & (lengths <= length))
                 & jnp.all(jnp.where(valid,
                     (rows["insertion_ids"] == ids) & (rows["timesteps"] == times)
                     & (rows["episode_ids"] == rows["episode_ids"][0:1])
                     & (rows["dones"] == (times == lengths[None, :] - 1)), True)))
    # Finite dummy rows: all actions legal, likelihood 1, terminal, zero rewards.
    clean = {}
    for name, values in rows.items():
        mask = valid[..., None] if values.ndim == 3 else valid
        dummy = True if name in ("observations", "dones", "behavior") else 0
        clean[name] = jnp.where(mask, values, dummy)
    clean["observations"] = clean["observations"].astype(jnp.float32)
    batch = EpisodeBatch(**clean, valid=valid, integrity=integrity)
    # Duplicate draws count repeatedly for loss/reuse, but only once for unique rows.
    ordered = jnp.sort(slots)
    first = jnp.concatenate((jnp.ones(1, bool), ordered[1:] != ordered[:-1]))
    new_unique = jnp.sum(jnp.where(first & (replay.reuse[ordered] == 0), replay.lengths[ordered], 0))
    updated = replay._replace(
        reuse=replay.reuse.at[slots].add(jnp.uint32(1)),
        episode_draws=replay.episode_draws + batch_size,
        presentations=add_count(replay.presentations, valid.sum()),
        unique_transitions=replay.unique_transitions + new_unique,
    )
    ages = replay.inserted - 1 - rows["insertion_ids"]
    mean = lambda x: jnp.sum(jnp.where(valid, x, 0)) / valid.sum()
    metrics = {
        "valid_presentations": valid.sum(), "padded_positions": valid.size - valid.sum(),
        "eligible_episodes": replay.count, "unfinished_transitions": replay.pending_length,
        "evicted_episodes": replay.evicted, "rejected_episodes": replay.rejected,
        "replay_age_mean": mean(ages), "replay_age_max": jnp.max(jnp.where(valid, ages, 0)),
        "reuse_before_mean": jnp.sum(replay.reuse[slots].astype(jnp.float32) * lengths) / valid.sum(),
    }
    return updated, batch, metrics
