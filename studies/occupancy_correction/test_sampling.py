"""Independent finite-model and empirical checks for the actual new sampler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.actor_critic import retrace_targets
from studies.occupancy_correction.sampling import (
    anchor_probabilities, at_anchors, create_anchor_sampler, prepare_episodes,
)


def fixture_data(lengths=(1, 1, 1, 3, 0)):
    """Three short episodes and one long: exact 3/4 vs 1/4 initial law.

    Physical states are 0 -> terminal, or 1 -> 2 -> 3 -> terminal. States
    0 and 3 share observation zero, making their posterior depend on time
    weighting. Extra state labels are audit-only, never agent observations.
    """
    lengths = np.array(lengths, dtype=np.int32)
    n, steps = len(lengths), 4
    ages = np.arange(steps)[None, :]
    valid = ages < lengths[:, None]
    states = np.zeros((n, steps), dtype=np.int32)
    for e, length in enumerate(lengths):
        if length > 1:
            states[e, :length] = np.arange(1, length + 1)
    observations = np.where((states == 0) | (states == 3), 0, states)
    info = dict(
        observations=observations[..., None].astype(np.float32),
        actions=np.zeros((n, steps), dtype=np.int32),
        rewards=np.where(valid, states + 1, 0).astype(np.float32),
        terminated=valid & (ages == lengths[:, None] - 1),
        behaviour_probs=np.full((n, steps), 0.5, dtype=np.float32),
        audit_state=states,
    )
    return info, lengths


def dataset(lengths=(1, 1, 1, 3, 0)):
    info, lengths = fixture_data(lengths)
    return prepare_episodes(info, lengths, behaviour_policy_ids=["fixed-uniform-v1"] * len(lengths))


def exact_nonterminal_occupancy(gamma):
    """Resolve the population model independently of episode/sampler code."""
    transition = np.zeros((5, 5), dtype=np.float64)
    transition[0, 4] = transition[1, 2] = transition[2, 3] = 1
    transition[3, 4] = transition[4, 4] = 1
    initial = np.array([0.75, 0.25, 0, 0, 0], dtype=np.float64)
    occupancy = (1 - gamma) * np.linalg.solve(np.eye(5) - gamma * transition.T, initial)
    # The absorbing state's residuals are zero; condition on nonterminal states.
    return occupancy[:4] / occupancy[:4].sum()


@pytest.mark.parametrize("gamma", [0.2, 0.5, 0.9, 0.99])
def test_probabilities_equal_independent_resolvent(gamma):
    episodes = dataset()
    probabilities = np.asarray(anchor_probabilities(
        episodes.lengths, 4, gamma=gamma, measure="discounted"))
    states = np.asarray(episodes.info["audit_state"])
    empirical = np.bincount(states.ravel(), weights=probabilities.ravel(), minlength=4)
    np.testing.assert_allclose(empirical, exact_nonterminal_occupancy(gamma), atol=8e-8)
    # Conditional latent posterior at the aliased observation (states 0,3).
    exact = exact_nonterminal_occupancy(gamma)
    np.testing.assert_allclose(empirical[[0, 3]] / empirical[[0, 3]].sum(),
                               exact[[0, 3]] / exact[[0, 3]].sum(), atol=1e-7)


def test_uniform_and_discounted_laws_really_differ():
    episodes = dataset()
    uniform = np.asarray(anchor_probabilities(episodes.lengths, 4, gamma=0.5, measure="uniform"))
    discounted = np.asarray(anchor_probabilities(episodes.lengths, 4, gamma=0.5, measure="discounted"))
    np.testing.assert_allclose(uniform[uniform > 0], 1 / 6)
    assert not np.allclose(uniform, discounted)
    # Reject the tempting but wrong law: uniform episodes, then normalized times.
    wrong = np.zeros_like(discounted)
    for e, length in enumerate(np.asarray(episodes.lengths)):
        if length:
            weights = 0.5 ** np.arange(length)
            wrong[e, :length] = weights / weights.sum() / 4
    assert not np.allclose(wrong, discounted)


@pytest.mark.parametrize("measure", ["uniform", "discounted"])
def test_actual_draws_match_anchor_and_latent_posterior(measure):
    episodes = dataset()
    count, gamma = 50_000, 0.5
    sample = create_anchor_sampler(batch_size=count, gamma=gamma, measure=measure)
    batch = sample(episodes, jax.random.key(401))
    expected = np.asarray(anchor_probabilities(episodes.lengths, 4, gamma=gamma, measure=measure))
    counts = np.zeros_like(expected)
    np.add.at(counts, (np.asarray(batch.episode_indices), np.asarray(batch.anchor_times)), 1)
    observed = counts / count
    six_sigma = 6 * np.sqrt(expected * (1 - expected) / count)
    assert np.all(np.abs(observed - expected) <= six_sigma + 1 / count)
    assert np.count_nonzero(counts[expected == 0]) == 0
    np.testing.assert_allclose(batch.probabilities, expected[batch.episode_indices, batch.anchor_times])
    sampled_states = np.asarray(at_anchors(batch.info["audit_state"], batch))
    posterior = np.mean(sampled_states[np.isin(sampled_states, [0, 3])] == 0)
    exact = exact_nonterminal_occupancy(gamma)
    expected_posterior = exact[0] / (exact[0] + exact[3]) if measure == "discounted" else 0.75
    assert abs(posterior - expected_posterior) < 0.015


def test_raw_prefix_future_and_logged_behaviour_are_preserved():
    episodes = dataset()
    batch = create_anchor_sampler(batch_size=256, gamma=0.8, measure="discounted")(
        episodes, jax.random.key(14))
    for e in range(256):
        source = int(batch.episode_indices[e])
        anchor, length = int(batch.anchor_times[e]), int(batch.lengths[e])
        np.testing.assert_array_equal(batch.info["observations"][e], episodes.info["observations"][source])
        np.testing.assert_array_equal(np.flatnonzero(batch.prefix_mask[e]), np.arange(anchor + 1))
        np.testing.assert_array_equal(np.flatnonzero(batch.continuation_mask[e]), np.arange(anchor, length))
        np.testing.assert_array_equal(batch.info["behaviour_probs"][e, :length], 0.5)
        assert bool(batch.info["terminated"][e, length - 1])


def test_padding_is_neutral_and_never_becomes_an_anchor():
    info, lengths = fixture_data()
    valid = np.arange(4)[None, :] < lengths[:, None]
    info["observations"][~valid] = np.nan
    info["behaviour_probs"][~valid] = np.nan
    info["rewards"][~valid] = np.nan
    episodes = prepare_episodes(info, lengths, behaviour_policy_ids=["fixed"] * 5)
    for array in episodes.info.values():
        assert np.isfinite(np.asarray(array)).all()
    np.testing.assert_array_equal(np.asarray(episodes.info["terminated"])[~valid], 1)
    np.testing.assert_array_equal(np.asarray(episodes.info["behaviour_probs"])[~valid], 1)


def test_mixed_behaviour_cannot_pass_as_stationary_mu():
    info, lengths = fixture_data()
    with pytest.raises(ValueError, match="fixed behaviour, not a mixture"):
        prepare_episodes(info, lengths, behaviour_policy_ids=["policy-a", "policy-b", "policy-a", "policy-a", "unused"])


@pytest.mark.parametrize("kind", ["incomplete", "early-terminal", "zero-mu", "invalid-mu", "nan-reward", "fractional-action"])
def test_invalid_episodes_are_rejected(kind):
    info, lengths = fixture_data()
    if kind == "incomplete": info["terminated"][3, 2] = False
    if kind == "early-terminal": info["terminated"][3, 0] = True
    if kind == "zero-mu": info["behaviour_probs"][3, 0] = 0
    if kind == "invalid-mu": info["behaviour_probs"][3, 0] = 1.1
    if kind == "nan-reward": info["rewards"][3, 0] = np.nan
    if kind == "fractional-action":
        info["actions"] = info["actions"].astype(float)
        info["actions"][3, 0] = 0.5
    with pytest.raises(ValueError):
        prepare_episodes(info, lengths, behaviour_policy_ids=["fixed"] * 5)


@pytest.mark.parametrize("gamma", [0, 1, -0.1, np.nan, np.inf])
def test_invalid_discount_is_rejected(gamma):
    with pytest.raises(ValueError):
        create_anchor_sampler(batch_size=1, gamma=gamma, measure="discounted")


def test_unknown_measure_empty_data_and_invalid_batch_are_rejected():
    with pytest.raises(ValueError): create_anchor_sampler(batch_size=1, gamma=0.9, measure="window")
    with pytest.raises(ValueError): create_anchor_sampler(batch_size=0, gamma=0.9, measure="discounted")
    info, lengths = fixture_data((0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="no complete episode"):
        prepare_episodes(info, lengths, behaviour_policy_ids=["fixed"] * 5)


@pytest.mark.parametrize("correction", ["uncorrected", "is", "retrace"])
def test_sampled_anchor_targets_against_independent_forward_products(correction):
    episodes = dataset()
    batch = create_anchor_sampler(batch_size=256, gamma=0.9, measure="discounted")(
        episodes, jax.random.key(30))
    rewards = np.asarray(batch.info["rewards"]).T
    dones = np.asarray(batch.info["terminated"]).T
    # Deliberately include ratios above one: ordinary IS must not become Retrace.
    ratios = np.broadcast_to(np.array([2.0, 0.5, 3.0, 1.0])[:, None], rewards.shape)
    q = np.broadcast_to(np.array([0.4, 0.3, 0.2, 0.0])[:, None], rewards.shape)
    v = q * 0.8
    gamma, lam = 0.9, 0.85
    actual = retrace_targets(*map(jnp.asarray, (q, v, rewards, dones, ratios)),
                             gamma, lam, correction=correction)
    v_next = np.concatenate([v[1:], np.zeros_like(v[:1])])
    deltas = rewards + gamma * (1 - dones) * v_next - q
    factors = np.ones_like(ratios) if correction == "uncorrected" else (
        ratios if correction == "is" else np.minimum(1, ratios))
    expected = q.copy()
    for t in range(rewards.shape[0]):
        weight = np.ones(rewards.shape[1])
        for u in range(t, rewards.shape[0]):
            expected[t] += weight * deltas[u]
            if u + 1 < rewards.shape[0]:
                weight *= gamma * (1 - dones[u]) * lam * factors[u + 1]
    np.testing.assert_allclose(at_anchors(actual.T, batch), at_anchors(jnp.asarray(expected.T), batch),
                               rtol=2e-6, atol=2e-6)


def test_same_random_draws_do_not_hide_changed_sampling():
    episodes = dataset()
    key = jax.random.key(42)
    uniform = create_anchor_sampler(batch_size=512, gamma=0.5, measure="uniform")(episodes, key)
    discounted = create_anchor_sampler(batch_size=512, gamma=0.5, measure="discounted")(episodes, key)
    # Matched RNG is allowed; the laws and actual anchors must still change.
    assert not np.array_equal(uniform.anchor_times, discounted.anchor_times)


def test_loss_consumer_must_score_anchors_not_all_returned_positions():
    episodes = dataset()
    batch = create_anchor_sampler(batch_size=50_000, gamma=0.5, measure="discounted")(
        episodes, jax.random.key(71))
    squared_errors = batch.info["rewards"] ** 2
    correct = float(at_anchors(squared_errors, batch).mean())
    full = np.asarray(episodes.info["rewards"]) ** 2
    probabilities = np.asarray(anchor_probabilities(episodes.lengths, 4, gamma=0.5, measure="discounted"))
    expected = float((full * probabilities).sum())
    assert abs(correct - expected) < 0.08
    wrong = float((squared_errors * batch.valid).sum() / batch.valid.sum())
    assert abs(wrong - expected) > 0.2
