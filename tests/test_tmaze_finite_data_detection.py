"""Finite-data sampling, production targets, and frozen-critic training checks."""

import itertools
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_examples = Path(__file__).resolve().parents[1] / "examples/tmaze"
with patch.object(sys, "path", [str(_examples), *sys.path]):
    import tmaze_finite_data_detection as finite
    import tmaze_finite_data_audit as audit


def test_sampling_preserves_prefixes_and_pairs_physical_episodes_across_channels():
    config = finite.Config(hallway_length=1)
    small = finite.sample_episodes(config, 0.5, seed=8, count=16)
    large = finite.sample_episodes(config, 0.5, seed=8, count=64)
    for short, long in zip(small, large):
        np.testing.assert_array_equal(short, long[:, :16])
    perfect = finite.sample_episodes(config, 0, seed=8, count=64)
    aliased = finite.sample_episodes(config, 1, seed=8, count=64)
    for name in ("actions", "rewards", "dones", "ratios"):
        np.testing.assert_array_equal(getattr(perfect, name), getattr(aliased, name))
    np.testing.assert_array_equal(perfect.dones[:-1], 0)
    np.testing.assert_array_equal(perfect.dones[-1], 1)
    np.testing.assert_array_equal(perfect.rewards[:-1], 0)
    assert set(np.unique(aliased.observations[1])) == {1}
    assert set(np.unique(aliased.observations[2])) == {2}


@pytest.mark.parametrize("trace", [0.0, 0.5, 1.0])
def test_markov_reference_is_a_pathwise_fixed_point_of_production_targets(trace):
    config = finite.Config()
    data = finite.sample_episodes(config, 0, seed=3, count=64)
    reference = finite.reference_values(config, 0)
    q = np.nan_to_num(reference["q"][0]) * config.reward_scale
    taken = q[data.observations, data.actions]
    v = np.sum(q * reference["target"], axis=-1)[data.observations]
    targets = finite.retrace_targets(jnp.asarray(taken), jnp.asarray(v),
                                     jnp.asarray(data.rewards * config.reward_scale),
                                     jnp.asarray(data.dones), jnp.asarray(data.ratios),
                                     config.gamma, trace)
    np.testing.assert_allclose(targets, taken, atol=1e-6)


@pytest.mark.parametrize("hallway_length", [1, 5])
def test_discounted_episode_regression_matches_exact_stochastic_observation_operator(hallway_length):
    config = finite.Config(hallway_length=hallway_length)
    model = finite.tm.TMazePOMDP(hallway_length, gamma=config.gamma)
    steps = hallway_length + 2
    channel = finite.alias.observation_channel(model, "both", 0.37)
    reference = finite.reference_values(config, 0.37)
    observations, actions, rewards, probabilities = [], [], [], []
    for goal in (0, 1):
        states = goal * steps + np.arange(steps)
        for obs in itertools.product(*(np.flatnonzero(channel[state]) for state in states)):
            emission_mass = np.prod(channel[states, obs])
            for action in (finite.tm.A_NORTH, finite.tm.A_SOUTH):
                observations.append(obs)
                actions.append((finite.tm.A_EAST,) * (steps - 1) + (action,))
                rewards.append((0,) * (steps - 1) + (model.good_reward if action == goal else model.bad_reward,))
                action_mass = config.behaviour_up if action == finite.tm.A_NORTH else 1 - config.behaviour_up
                probabilities.append(0.5 * emission_mass * action_mass)
    obs, acts, rewards = (np.asarray(array).T for array in (observations, actions, rewards))
    dones = np.zeros_like(rewards)
    dones[-1] = 1
    ratios = np.ones_like(rewards)
    ratios[-1] = np.where(acts[-1] == finite.tm.A_NORTH,
                          config.target_up / config.behaviour_up,
                          (1 - config.target_up) / (1 - config.behaviour_up))
    mass = config.gamma ** np.arange(steps)[:, None] * np.asarray(probabilities)[None, :]
    for index, trace in enumerate(config.lambdas):
        q = np.nan_to_num(reference["q"][index]) * config.reward_scale
        taken = q[obs, acts]
        v = np.sum(q * reference["target"], axis=-1)[obs]
        targets = finite.retrace_targets(jnp.asarray(taken), jnp.asarray(v),
                                         jnp.asarray(rewards * config.reward_scale),
                                         jnp.asarray(dones), jnp.asarray(ratios),
                                         config.gamma, trace)
        residual = np.asarray(targets) - taken
        moment = np.zeros_like(q)
        np.add.at(moment, (obs.ravel(), acts.ravel()), (mass * residual).ravel())
        np.testing.assert_allclose(moment, 0, atol=1e-7)
        assert np.max(np.abs(residual)) < config.huber_delta


def test_metrics_compare_with_off_policy_reference_and_report_missing_coverage():
    config = finite.Config(hallway_length=1)
    reference = finite.reference_values(config, 1)
    on_policy = finite.reference_values(config._replace(behaviour_up=config.target_up), 1)
    assert abs(reference["discrepancy"] - on_policy["discrepancy"]) > 0.1
    q = np.nan_to_num(reference["q"])
    twins = np.repeat(q[:, None], 2, axis=1) * config.reward_scale
    data = finite.sample_episodes(config, 1, seed=1, count=1)
    metrics = finite.evaluate_predictions(twins, reference, data, 1, config.reward_scale)
    assert metrics["estimated_discrepancy"] == pytest.approx(reference["discrepancy"])
    assert metrics["q_rmse"] < 1e-12
    assert metrics["scored_weight_coverage"] < 1
    assert metrics["minimum_scored_visits"] == 0


def test_common_critic_offset_is_visible_even_when_discrepancy_is_unchanged():
    config = finite.Config()
    reference = finite.reference_values(config, 1)
    q = np.nan_to_num(reference["q"]) + 0.25
    twins = np.repeat(q[:, None], 2, axis=1) * config.reward_scale
    data = finite.sample_episodes(config, 1, seed=0, count=64)
    metrics = finite.evaluate_predictions(twins, reference, data, 64, config.reward_scale)
    assert metrics["estimated_discrepancy"] == pytest.approx(reference["discrepancy"])
    assert metrics["q_rmse"] == pytest.approx(0.25)
    np.testing.assert_allclose(metrics["q_bias_by_lambda"], [0.25, 0.25])
    np.testing.assert_allclose(metrics["q_centered_rmse_by_lambda"], 0, atol=1e-12)
    assert metrics["discrepancy_vector_rmse"] < 1e-12


@pytest.mark.parametrize("eta", [0, 0.5, 1])
def test_independent_empirical_roots_satisfy_the_production_update(eta):
    config = finite.Config()
    data = finite.sample_episodes(config, eta, seed=0, count=256)
    reference = finite.reference_values(config, eta)
    systems = [audit.empirical_fixed_point(data, 256, reference["target"], config.gamma, trace)
               for trace in config.lambdas]
    assert audit.verify_production_moments(data, 256, reference, systems, config) < 1e-6
    if eta == 0:
        selected = reference["weights"] > 0
        for i, system in enumerate(systems):
            np.testing.assert_allclose(system["q"][selected], reference["q"][i][selected], atol=1e-8)
    duplicated = finite.Episodes(*(np.concatenate((value, value), axis=1) for value in data))
    for trace, system in zip(config.lambdas, systems):
        repeated = audit.empirical_fixed_point(duplicated, 512, reference["target"], config.gamma, trace)
        np.testing.assert_allclose(system["q"], repeated["q"], atol=1e-12)


def test_missing_empirical_bootstrap_coverage_is_not_regularized_away():
    config = finite.Config(hallway_length=1)
    data = finite.sample_episodes(config, 1, seed=0, count=1)
    reference = finite.reference_values(config, 1)
    with pytest.raises(ValueError, match="unidentified"):
        audit.empirical_fixed_point(data, 1, reference["target"], config.gamma, 1)


def test_training_updates_all_four_branches_and_ema_targets():
    config = finite.Config(hallway_length=1, hidden_dims=(8,), batch_size=8, tau=0.2)
    reference = finite.reference_values(config, 0.5)
    initialize, train, predict = finite.create_trainer(
        config, reference["n_observations"], reference["target"])
    before = initialize(0)
    data = jax.tree.map(jnp.asarray, finite.sample_episodes(config, 0.5, seed=0, count=16))
    after, stats = train(before, data, jnp.asarray(16), steps=2)
    assert all(np.isfinite(value) for value in jax.tree.leaves(stats))
    changes = np.zeros(4)
    for old, new in zip(jax.tree.leaves(before.params), jax.tree.leaves(after.params)):
        changes += np.abs(np.asarray(new - old)).reshape(4, -1).sum(axis=1)
    assert np.all(changes > 0)
    assert any(not np.array_equal(old, new) for old, new in
               zip(jax.tree.leaves(before.targets), jax.tree.leaves(after.targets)))
    assert np.all(np.isfinite(predict(after.params)))
    # The unmodified starting state is reusable for the next independent budget.
    again, _ = train(before, data, jnp.asarray(16), steps=2)
    np.testing.assert_array_equal(predict(after.params), predict(again.params))


def test_settling_phase_changes_the_step_size_at_the_declared_boundary():
    base = finite.Config(hallway_length=1, hidden_dims=(8,), batch_size=8, updates=2)
    scheduled = base._replace(settle_updates=1, settle_learning_rate=1e-5)
    reference = finite.reference_values(base, 0.5)
    data = jax.tree.map(jnp.asarray, finite.sample_episodes(base, 0.5, seed=0, count=16))
    outcomes = []
    for config in (base, scheduled):
        initialize, train, _ = finite.create_trainer(
            config, reference["n_observations"], reference["target"])
        two, _ = train(initialize(0), data, jnp.asarray(16), steps=2)
        three, _ = train(two, data, jnp.asarray(16), steps=1)
        outcomes.append((two, three))
    for a, b in zip(jax.tree.leaves(outcomes[0][0].params), jax.tree.leaves(outcomes[1][0].params)):
        np.testing.assert_allclose(a, b, atol=1e-7)
    sizes = [sum(np.abs(np.asarray(b - a)).sum() for a, b in
                 zip(jax.tree.leaves(two.params), jax.tree.leaves(three.params)))
             for two, three in outcomes]
    assert sizes[1] / sizes[0] == pytest.approx(0.01, rel=0.005)


@pytest.mark.parametrize("arguments", [
    ["--behaviour-up", "0"], ["--lambdas", "1", "1"],
    ["--episodes", "128", "32"], ["--etas", "0", "0"], ["--updates", "0"],
    ["--settle-updates", "-1"], ["--settle-learning-rate", "0"],
])
def test_rejects_invalid_experiment_protocols(arguments):
    with pytest.raises(SystemExit):
        finite.parse_args(arguments)
