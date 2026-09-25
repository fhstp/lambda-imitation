"""Check stochastic-observation Retrace against enumerated physical trajectories."""

import itertools
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest


_examples = Path(__file__).resolve().parents[1] / "examples/tmaze"
with patch.object(sys, "path", [str(_examples), *sys.path]):
    import tmaze_retrace_aliasing as alias

tm = alias.tm


def enumerated_stochastic_retrace(pomdp, channel, b, p, lambda_):
    """Enumerate goals, emitted observations, and final actions; no lifted model.

    Accumulate backward eligibility traces on the *realised* observation. This
    catches replacing that observation with its conditional mean in the solver.
    """
    size = pomdp.n_states * tm.NA
    features = np.eye(size)
    A, rhs, counts = np.zeros((size, size)), np.zeros(size), np.zeros(size)
    for goal in (0, 1):
        states = goal * (pomdp.L + 2) + np.arange(pomdp.L + 2)
        emissions = [np.flatnonzero(channel[s]) for s in states]
        for observations in itertools.product(*emissions):
            obs_prob = 0.5 * np.prod(channel[states, observations])
            for junction_action in (tm.A_NORTH, tm.A_SOUTH):
                probability = obs_prob * (b if junction_action == tm.A_NORTH else 1 - b)
                actions = [tm.A_EAST] * (len(states) - 1) + [junction_action]
                z = np.zeros(size)
                for t, (obs, action) in enumerate(zip(observations, actions)):
                    terminal = t == len(states) - 1
                    x = features[obs * tm.NA + action]
                    ratio = 1.0
                    next_x = np.zeros(size)
                    reward = 0.0
                    if terminal:
                        ratio = min(1, p / b if action == tm.A_NORTH else (1 - p) / (1 - b))
                        reward = pomdp.good_reward if action == goal else pomdp.bad_reward
                    elif t == len(states) - 2:
                        nxt = observations[t + 1] * tm.NA
                        next_x[nxt + tm.A_NORTH] = p
                        next_x[nxt + tm.A_SOUTH] = 1 - p
                    else:
                        next_x[observations[t + 1] * tm.NA + tm.A_EAST] = 1
                    z = pomdp.gamma * lambda_ * ratio * z + pomdp.gamma ** t * x
                    A += probability * np.outer(z, x - pomdp.gamma * next_x)
                    rhs += probability * z * reward
                    counts += probability * x
    supported = counts > 0
    q = np.full(size, np.nan)
    q[supported] = np.linalg.solve(A[np.ix_(supported, supported)], rhs[supported])
    q = q.reshape(pomdp.n_states, tm.NA)
    q[pomdp.terminal] = 0
    return q


@pytest.mark.parametrize("kind", alias.ALIAS_TYPES)
@pytest.mark.parametrize("lambda_", [0.0, 0.6, 1.0])
def test_stochastic_retrace_matches_enumerated_observation_histories(kind, lambda_):
    for length in (1, 5):
        pomdp = tm.TMazePOMDP(length)
        channel = alias.observation_channel(pomdp, kind, 0.37)
        model = alias.observation_augmented_model(pomdp, channel)
        b, p = 1 / 3, 2 / 3
        evaluator = tm.RetraceEvaluator(model, pomdp.phi @ tm.junction_policy(b))
        actual = evaluator.q(pomdp.phi @ tm.junction_policy(p), lambda_)
        expected = enumerated_stochastic_retrace(pomdp, channel, b, p, lambda_)
        np.testing.assert_allclose(actual, expected, atol=4e-12)


@pytest.mark.parametrize("length", [1, 5])
def test_both_aliased_endpoint_matches_existing_behaviour_experiment(length):
    pomdp = tm.TMazePOMDP(length)
    channel = alias.observation_channel(pomdp, "both", 1)
    model = alias.observation_augmented_model(pomdp, channel)
    weights = channel.T @ alias.reference_pair_weights(pomdp)
    assert weights.sum() == pytest.approx(1)
    for behaviour in tm.behaviour_policies():
        old = tm.RetraceEvaluator(pomdp, behaviour.policy)
        new = tm.RetraceEvaluator(model, pomdp.phi @ behaviour.policy)
        for p in (0.0, 0.5, 2 / 3, 1.0):
            target = tm.junction_policy(p)
            q_lo, q_hi = tm.value_pair(new, pomdp.phi @ target, (0.0, 1.0))
            assert alias.weighted_gap(q_lo, q_hi, weights) == pytest.approx(
                tm.discrepancy(old, target, (0.0, 1.0)), abs=3e-12)


def test_perfect_observations_zero_for_every_selected_pair():
    pomdp = tm.TMazePOMDP(5)
    reference = alias.reference_pair_weights(pomdp)
    model = alias.observation_augmented_model(pomdp, np.eye(pomdp.n_states))
    args = alias.parse_args([])
    behaviours = {b.key: b.policy for b in tm.behaviour_policies()}
    for pair in args.pairs:
        evaluator = tm.RetraceEvaluator(model, pomdp.phi @ behaviours[pair.behaviour])
        q = tm.value_pair(evaluator, pomdp.phi @ tm.junction_policy(pair.p_up), (0.05, 0.85))
        assert alias.weighted_gap(*q, reference) < 1e-12


@pytest.mark.parametrize("kind", alias.ALIAS_TYPES)
def test_policy_laws_and_scoring_mass_fixed_during_interpolation(kind):
    pomdp = tm.TMazePOMDP(5)
    reference = alias.reference_pair_weights(pomdp)
    for eta in (0, 0.2, 0.7, 1):
        channel = alias.observation_channel(pomdp, kind, eta)
        np.testing.assert_allclose(channel.sum(axis=1), 1)
        assert np.sum(channel.T @ reference) == pytest.approx(1)
        for behaviour in tm.behaviour_policies():
            pi_obs = pomdp.phi @ behaviour.policy
            np.testing.assert_allclose(channel @ pi_obs, pi_obs, atol=1e-15)


def test_blind_endpoint_can_have_interior_discrepancy():
    pomdp = tm.TMazePOMDP(1)
    pair = alias.PolicyPair("reverse", 2 / 3)
    result = alias.run_sweep(pomdp, [pair], (0.0, 1.0), 3)
    curves = result["pairs"][0]["curves"]
    assert curves["both"][0] < 1e-12
    assert curves["both"][1] > 0.05
    assert curves["both"][-1] < 1e-12
    assert max(curves["junction"]) < 1e-12


def test_stochastic_on_policy_mc_endpoint_with_exploratory_behaviour():
    pomdp = tm.TMazePOMDP(3)
    channel = alias.observation_channel(pomdp, "both", 0.61)
    behaviour = tm.behaviour_policies()[3].policy
    policy = pomdp.phi @ behaviour
    transition = np.einsum("sa,asn->sn", channel @ policy, pomdp.T)
    occupancy = np.linalg.solve(np.eye(pomdp.n_states) - pomdp.gamma * transition.T, pomdp.p0)
    reward_sa = (pomdp.T * pomdp.R).sum(axis=-1).T
    v = np.linalg.solve(np.eye(pomdp.n_states) - pomdp.gamma * transition,
                        np.sum((channel @ policy) * reward_sa, axis=1))
    state_q = reward_sa + pomdp.gamma * (pomdp.T @ v).T
    weights = occupancy[:, None] * channel
    expected = (weights.T @ state_q) / weights.sum(axis=0)[:, None]
    model = alias.observation_augmented_model(pomdp, channel)
    actual = tm.RetraceEvaluator(model, policy).q(policy, 1.0)
    np.testing.assert_allclose(actual, expected, atol=4e-12)
