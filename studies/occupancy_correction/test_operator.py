"""Independent enumerated-trajectory and matrix-operator agreement.

No sampling implementation is used to construct the reference. Observation 1
aliases states at different episode ages, so uniform and discounted posteriors
genuinely differ. Actions affect branch selection and terminal rewards, and the
behaviour/target policies differ. This is not an on-policy or Markov-only test.
"""

from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.actor_critic import retrace_targets


G = np.array([0, 1, 1, 2])
MU = np.array([[0.6, 0.4], [0.35, 0.65], [0.7, 0.3]])
PI = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
REWARDS = np.array([[0., 0.], [1., -1.], [-0.5, 1.5], [0., 0.]])


def trajectories():
    for first, second in product(range(2), repeat=2):
        if first == 0:
            states, actions = [0, 1], [first, second]
            yield states, actions, MU[0, first] * MU[1, second]
        else:
            for third in range(2):
                states, actions = [0, 3, 2], [first, second, third]
                yield states, actions, MU[0, first] * MU[2, second] * MU[1, third]


def continuation(correction, lam):
    ratios = PI / MU
    return lam * (np.ones_like(ratios) if correction == "uncorrected" else
                  ratios if correction == "is" else np.minimum(1, ratios))


def matrix_operator(gamma, lam, correction, measure):
    transition = np.zeros((4, 2, 4))
    transition[0, 0, 1] = transition[0, 1, 3] = 1
    transition[3, :, 2] = 1
    b_mu = (transition * MU[G, :, None]).sum(1)
    initial = np.array([1., 0, 0, 0])
    discount = gamma if measure == "discounted" else 1.0
    occupancy = np.linalg.solve(np.eye(4) - discount * b_mu.T, initial)
    w, j = np.zeros((6, 8)), np.zeros((8, 6))
    for state, action in product(range(4), range(2)):
        row, col = G[state]*2+action, state*2+action
        j[col, row] = 1
        w[row, col] = occupancy[state] / occupancy[G == G[state]].sum()
    p_mu, p_pi = np.zeros((8, 8)), np.zeros((8, 8))
    for s, a, ns, na in product(range(4), range(2), range(4), range(2)):
        p_mu[s*2+a, ns*2+na] = transition[s, a, ns] * MU[G[ns], na]
        p_pi[s*2+a, ns*2+na] = transition[s, a, ns] * PI[G[ns], na]
    c = np.diag(continuation(correction, lam)[G].ravel())
    h = np.linalg.inv(np.eye(8) - gamma * p_mu @ c)
    b = w @ h @ REWARDS.ravel()
    linear = gamma * w @ h @ (p_pi-p_mu@c) @ j
    return b, linear, occupancy


def enumerated_operator(q, gamma, lam, correction, measure, *, occupancy_is=False):
    sums, mass = np.zeros((3, 2)), np.zeros((3, 2))
    coeff = continuation(correction, lam)
    paths = list(trajectories())
    uniform_z = sum(len(s)*p for s, _, p in paths)
    discounted_z = sum(p*sum(gamma**np.arange(len(s))) for s, _, p in paths)
    for states, actions, probability in paths:
        for start in range(len(states)):
            weight = 1.0
            target = q[G[states[start]], actions[start]]
            for future in range(start, len(states)):
                s, a = states[future], actions[future]
                value_next = (PI[G[states[future+1]]] @ q[G[states[future+1]]]
                              if future+1 < len(states) else 0)
                delta = REWARDS[s, a] + gamma*value_next - q[G[s], a]
                target += weight * delta
                if future+1 < len(states):
                    weight *= gamma * coeff[G[states[future+1]], actions[future+1]]
            proposal_mass = probability / uniform_z
            if occupancy_is:
                # Actual uniform proposal with target/proposal loss weighting.
                anchor_mass = proposal_mass * uniform_z*gamma**start/discounted_z
            else:
                anchor_mass = probability * (gamma**start if measure == "discounted" else 1)
            index = (G[states[start]], actions[start])
            sums[index] += anchor_mass*target
            mass[index] += anchor_mass
    return sums/mass, mass/mass.sum()


@pytest.mark.parametrize("gamma", [0.5, 0.99])
@pytest.mark.parametrize("lam", [0.0, 0.85, 1.0])
@pytest.mark.parametrize("correction", ["uncorrected", "is", "retrace"])
@pytest.mark.parametrize("measure", ["uniform", "discounted"])
def test_independent_forward_enumeration_matches_matrix_and_production_targets(gamma, lam, correction, measure):
    q = np.array([[0.4, -0.2], [0.8, -0.5], [0.1, 0.6]])
    b, linear, _ = matrix_operator(gamma, lam, correction, measure)
    actual, _ = enumerated_operator(q, gamma, lam, correction, measure)
    np.testing.assert_allclose(actual.ravel(), b+linear@q.ravel(), atol=1e-13)
    # Compare every trajectory's production targets, not only an averaged result.
    for states, actions, _ in trajectories():
        obs = G[states]
        rewards = REWARDS[states, actions]
        done = np.arange(len(states)) == len(states)-1
        ratios = PI[obs, actions] / MU[obs, actions]
        taken = q[obs, actions]
        values = (q[obs] * PI[obs]).sum(-1)
        targets = np.asarray(retrace_targets(*[jnp.asarray(x[:, None]) for x in
            (taken, values, rewards, done, ratios)], gamma, lam, correction=correction))[:, 0]
        for start in range(len(states)):
            expected, weight = taken[start], 1.0
            for future in range(start, len(states)):
                nv = values[future+1] if future+1 < len(states) else 0
                expected += weight*(rewards[future]+gamma*nv-taken[future])
                if future+1 < len(states):
                    weight *= gamma*continuation(correction, lam)[obs[future+1], actions[future+1]]
            np.testing.assert_allclose(targets[start], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("correction", ["uncorrected", "is", "retrace"])
def test_occupancy_importance_weighting_equals_direct_discounted_measure(correction):
    q = np.array([[0.4, -0.2], [0.8, -0.5], [0.1, 0.6]])
    direct, mass = enumerated_operator(q, 0.9, 0.85, correction, "discounted")
    weighted, weighted_mass = enumerated_operator(q, 0.9, 0.85, correction, "uniform", occupancy_is=True)
    np.testing.assert_allclose(weighted, direct, atol=1e-14)
    np.testing.assert_allclose(weighted_mass, mass, atol=1e-14)


def test_witness_really_distinguishes_discounted_and_uniform_posteriors():
    _, _, discounted = matrix_operator(0.5, 0.85, "retrace", "discounted")
    _, _, uniform = matrix_operator(0.5, 0.85, "retrace", "uniform")
    post_d = discounted[1] / discounted[[1, 2]].sum()
    post_u = uniform[1] / uniform[[1, 2]].sum()
    assert post_d == pytest.approx(0.75)
    assert post_u == pytest.approx(0.6)


@pytest.mark.parametrize("measure", ["uniform", "discounted"])
@pytest.mark.parametrize("correction", ["uncorrected", "is", "retrace"])
def test_independently_solved_fixed_points_are_roots_of_enumerated_operator(measure, correction):
    for lam in (0.05, 0.85):
        b, linear, _ = matrix_operator(0.99, lam, correction, measure)
        root = np.linalg.solve(np.eye(6)-linear, b).reshape(3, 2)
        updated, _ = enumerated_operator(root, 0.99, lam, correction, measure)
        np.testing.assert_allclose(updated, root, atol=1e-12)
