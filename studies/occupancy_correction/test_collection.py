"""Validate actual environment collection, whole-episode retention and sampling."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.envs.battleship import Battleship
from lambda_imitation.envs.minesweeper import MineSweeper
from .collection import collect_step, start_rollout, uniform_probabilities
from .replay import append_episode, create_replay, empty_episode, packed_anchors, select
from .sampling import anchor_probabilities, prepare_episodes


def setup(env_name, capacity=1000):
    env = Battleship() if env_name == "battleship" else MineSweeper()
    steps = env.num_actions if env_name == "battleship" else env.episode_length
    mask = (lambda obs: obs[..., 1:].astype(bool)) if env_name == "battleship" else None
    shape = env.observation_space(env.default_params).shape
    return env, steps, mask, shape, create_replay(shape, steps, capacity)


def collect(env_name, n, capacity=1000):
    env, steps, mask, shape, replay = setup(env_name, capacity)
    rollout = start_rollout(env, jax.random.key(410), shape, steps)

    @partial(jax.jit, static_argnums=2)
    def chunk(replay, rollout, n):
        def step(state, _):
            return collect_step(*state, env, None, action_dim=env.num_actions,
                                mask_fn=mask, uniform=True), None
        return jax.lax.scan(step, (replay, rollout), None, length=n)[0]

    return chunk, chunk(replay, rollout, n), (env, steps, mask)


@pytest.mark.parametrize("env_name", ["battleship", "minesweeper"])
def test_real_collection_has_correct_probabilities_and_genuine_boundaries(env_name):
    _, (replay, rollout), (env, steps, mask) = collect(env_name, 500)
    info = jax.tree.map(np.asarray, replay.info)
    lengths = np.asarray(replay.lengths)
    assert int(rollout.calls) == 500
    assert 0 <= int(rollout.age) < steps
    assert int(replay.transitions) + int(rollout.age) == 500
    admitted = prepare_episodes(info, lengths, behaviour_policy_ids=["uniform-fixed-v1"] * len(lengths))
    assert admitted.behaviour_policy_id == "uniform-fixed-v1"
    for e in np.flatnonzero(lengths):
        n = lengths[e]
        probabilities = jax.vmap(lambda o: uniform_probabilities(o, env.num_actions, mask))(
            jnp.asarray(info["observations"][e, :n]))
        expected = np.asarray(probabilities)[np.arange(n), info["actions"][e, :n]]
        np.testing.assert_allclose(info["behaviour_probs"][e, :n], expected, rtol=1e-6)
        np.testing.assert_array_equal(info["collector_version"][e, :n], -1)
        if env_name == "battleship":
            assert len(np.unique(info["actions"][e, :n])) == n
            np.testing.assert_array_equal(info["observations"][e, 0, 1:], 1)
            # Termination is only after all five hits, not the padding limit.
            observed_hits = info["observations"][e, :n, 0].sum()
            assert observed_hits == 4  # fifth hit is the final action's outcome
        else:
            assert info["observations"][e, 0, -1] == 1


@pytest.mark.parametrize("env_name", ["battleship", "minesweeper"])
def test_host_chunk_boundary_does_not_reset_or_invent_terminal(env_name):
    chunk, (replay, rollout), _ = collect(env_name, 1)
    assert int(rollout.calls) == 1
    if int(rollout.age):
        assert int(replay.episodes) == 0
        assert not bool(rollout.pending["terminated"][0])
    together = chunk(replay, rollout, 99)
    # Recreate identical initial stream and collect all 100 in one host chunk.
    _, one_chunk, _ = collect(env_name, 100)
    for actual, expected in zip(jax.tree.leaves(together), jax.tree.leaves(one_chunk)):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_array_equal(actual, expected)


def test_whole_episode_fifo_eviction_and_ring_wrap():
    replay = create_replay((1,), 4, 5)
    for tag, length in enumerate([3, 2, 1, 4, 2, 3, 1, 1]):
        episode = empty_episode((1,), 4)
        valid = jnp.arange(4) < length
        episode = {**episode, "observations": jnp.full((4, 1), tag, jnp.float32),
                   "terminated": ~valid | (jnp.arange(4) == length - 1)}
        replay = jax.jit(append_episode)(replay, episode, jnp.int32(length))
        lengths = np.asarray(replay.lengths)
        assert lengths.sum() == int(replay.transitions) <= 5
        assert np.count_nonzero(lengths) == int(replay.episodes)
        for slot in np.flatnonzero(lengths):
            n = lengths[slot]
            assert not np.asarray(replay.info["terminated"])[slot, :n-1].any()
            assert bool(replay.info["terminated"][slot, n-1])
    retained = sorted(int(replay.info["observations"][e, 0, 0]) for e in np.flatnonzero(replay.lengths))
    assert retained == [5, 6, 7]


@pytest.mark.parametrize("measure", ["uniform", "discounted"])
def test_factored_production_sampler_matches_direct_global_law(measure):
    replay = create_replay((1,), 4, 10)
    for e, length in enumerate([1, 1, 1, 3]):
        episode = empty_episode((1,), 4)
        episode["observations"] = jnp.full((4, 1), e, jnp.float32)
        replay = append_episode(replay, episode, jnp.int32(length))
    gamma, n = 0.5, 50_000
    batch = jax.jit(lambda r, k: packed_anchors(
        r, k, gamma=gamma, measure=measure, episode_batch=n, score_count=n))(replay, jax.random.key(42))
    expected = np.asarray(anchor_probabilities(replay.lengths, 4, gamma=gamma, measure=measure))
    observed = np.zeros_like(expected)
    np.add.at(observed, (np.asarray(batch.episode_indices)[np.asarray(batch.owners)], np.asarray(batch.times)), 1 / n)
    assert np.all(np.abs(observed - expected) <= 6 * np.sqrt(expected * (1-expected) / n) + 2e-5)
    np.testing.assert_allclose(batch.probabilities,
                               expected[np.asarray(batch.episode_indices)[batch.owners], batch.times], atol=1e-7)


def test_packed_reused_episodes_score_only_explicit_anchors():
    _, (replay, _), _ = collect("minesweeper", 500)
    batch = jax.jit(lambda r, k: packed_anchors(
        r, k, gamma=0.99, measure="discounted", episode_batch=16, score_count=127))(
            replay, jax.random.key(8))
    assert batch.owners.shape == batch.times.shape == (127,)
    assert select(batch.info["rewards"], batch, 40).shape == (40,)
    assert np.all(np.asarray(batch.times) < np.asarray(batch.lengths)[batch.owners])
