"""Independent checks of the small, exact T-maze experiment (NumPy only).

Forward-policy paths can be enumerated, so the trajectory-level eligibility
trace calculation has no sampling error and tests the matrix operator's indexing,
discounted starting distribution, and placement of the importance ratios.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1] / "examples/tmaze/tmaze_lambda_discrepancy.py"
_spec = importlib.util.spec_from_file_location("tmaze_experiment", _path)
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)


def trajectory_fixed_point(length, gamma, p_behaviour, p_target, lambda_):
    """LSTD from all four possible forward trajectories, independently of T/R."""
    A, rhs = np.zeros((5, 5)), np.zeros(5)
    b = np.array([p_behaviour, 1 - p_behaviour])
    pi = np.array([p_target, 1 - p_target])
    for goal in (0, 1):
        for action in (0, 1):
            probability = 0.5 * b[action]
            # The first two features are the goal cues, third is hallway east,
            # and the last two are junction north/south.
            path = [goal] + [2] * length + [3 + action]
            z = np.zeros(5)
            for t, feature in enumerate(path):
                x = np.eye(5)[feature]
                at_junction = t == len(path) - 1
                ratio = min(1, pi[action] / b[action]) if at_junction else 1
                z = gamma * lambda_ * ratio * z + gamma ** t * x
                next_x = np.zeros(5)
                reward = 0.0
                if at_junction:
                    reward = 4.0 if action == goal else -0.1
                elif t == len(path) - 2:
                    next_x[3:] = pi
                else:
                    next_x[2] = 1
                A += probability * np.outer(z, x - gamma * next_x)
                rhs += probability * z * reward
    return np.linalg.solve(A, rhs)


@pytest.mark.parametrize("length", [1, 5])
@pytest.mark.parametrize("lambda_", [0.0, 0.05, 0.85, 1.0])
def test_matrix_matches_enumerated_retrace_trajectories(length, lambda_):
    pomdp = tm.TMazePOMDP(length)
    for b in (1 / 3, 0.5, 2 / 3):
        evaluator = tm.RetraceEvaluator(pomdp, tm.junction_policy(b))
        for p in (0.0, 1 / 3, 0.5, 2 / 3, 1.0):
            q = evaluator.q(tm.junction_policy(p), lambda_)
            expected = trajectory_fixed_point(length, pomdp.gamma, b, p, lambda_)
            np.testing.assert_allclose([q[o, a] for o, a in tm.SCORE_PAIRS],
                                       expected, atol=2e-12, rtol=2e-12)


def latent_q(pomdp, policy):
    """Ordinary fully observed Bellman evaluation, without Retrace matrices."""
    pi_state = pomdp.phi @ policy
    reward = (pomdp.T * pomdp.R).sum(axis=-1)
    transition = np.einsum("sa,asn->sn", pi_state, pomdp.T)
    v = np.linalg.solve(np.eye(pomdp.n_states) - pomdp.gamma * transition,
                        (pi_state * reward.T).sum(axis=1))
    return reward + pomdp.gamma * (pomdp.T @ v)


@pytest.mark.parametrize("length", [1, 5])
def test_on_policy_lambda_one_is_conditional_monte_carlo(length):
    pomdp = tm.TMazePOMDP(length)
    for behaviour in tm.behaviour_policies():
        evaluator = tm.RetraceEvaluator(pomdp, behaviour.policy)
        state_q = latent_q(pomdp, behaviour.policy)
        weights = evaluator.occupancy[:, None] * pomdp.phi
        posterior = weights / weights.sum(axis=0)
        expected = (state_q @ posterior).T
        actual = evaluator.q(behaviour.policy, 1.0)
        valid = np.isfinite(actual)
        np.testing.assert_allclose(actual[valid], expected[valid], atol=2e-12)


@pytest.mark.parametrize("lambdas", [(0.0, 1.0), (0.05, 0.85)])
def test_exact_zero_control_and_behaviour_specific_blind_spots(lambdas):
    pomdp = tm.TMazePOMDP(1)
    balanced = tm.junction_policy(0.5)
    b_eval = tm.RetraceEvaluator(pomdp, balanced)
    assert tm.discrepancy(b_eval, balanced, lambdas) < 1e-12
    assert tm.discrepancy(b_eval, tm.junction_policy(2 / 3), lambdas) > 0.1
    reverse = tm.RetraceEvaluator(pomdp, tm.junction_policy(1 / 3))
    assert tm.discrepancy(reverse, tm.junction_policy(1 / 3), lambdas) > 0.1
    assert tm.discrepancy(reverse, tm.junction_policy(2 / 3), lambdas) < 1e-12

    random = tm.RetraceEvaluator(pomdp, np.full((tm.NO, tm.NA), 0.25))
    # Fully supported but exceptional behaviour: a whole target interval is blind.
    for p in (0.25, 0.4, 0.5, 0.6, 0.75):
        assert tm.discrepancy(random, tm.junction_policy(p), lambdas) < 1e-12
    assert tm.discrepancy(random, tm.junction_policy(1), lambdas) > 0.01

    # Same balanced policy is NOT exactly zero with corridor-position aliasing.
    long_eval = tm.RetraceEvaluator(tm.TMazePOMDP(5), balanced)
    assert tm.discrepancy(long_eval, balanced, lambdas) > 0.01


@pytest.mark.parametrize("lambda_", [0.0, 0.05, 0.85, 1.0])
def test_markov_observations_remove_all_off_policy_discrepancy(lambda_):
    pomdp = tm.TMazePOMDP(5)
    old_phi = pomdp.phi
    behaviour = old_phi @ tm.behaviour_policies()[3].policy
    target = old_phi @ tm.junction_policy(0.8)
    pomdp.phi = np.eye(pomdp.n_states)
    evaluator = tm.RetraceEvaluator(pomdp, behaviour)
    np.testing.assert_allclose(evaluator.q(target, lambda_), latent_q(pomdp, target).T,
                               atol=5e-12)


def test_missing_behaviour_coverage_is_not_silently_regularised():
    evaluator = tm.RetraceEvaluator(tm.TMazePOMDP(), tm.junction_policy(1.0))
    with pytest.raises(ValueError, match="covered"):
        evaluator.q(tm.junction_policy(0.5), 1.0)
    q = evaluator.q(tm.junction_policy(1.0), 1.0)
    assert np.isnan(q[tm.OBS_JUNC, tm.A_SOUTH])
    with pytest.raises(ValueError, match="unsupported"):
        tm.rms_gap(q, q)


def test_tabular_transitions_match_bundled_environment():
    import jax
    import jax.numpy as jnp
    from lambda_imitation.envs.tmaze import TMaze, TMazeState

    for length in (1, 5):
        model = tm.TMazePOMDP(length)
        env = TMaze(hallway_length=length)
        # Check every nonterminal latent state and action, including no-ops,
        # westward moves, and both rewarding/penalised junction actions.
        states = np.repeat(np.arange(model.terminal), tm.NA)
        actions = np.tile(np.arange(tm.NA), model.terminal)
        env_states = TMazeState(grid_idx=jnp.array(states % (length + 2)),
                               goal_dir=jnp.array(states // (length + 2)))
        keys = jax.random.split(jax.random.key(0), len(states))
        obs = jax.vmap(env.get_obs)(env_states)
        _, nxt, reward, done, _ = jax.jit(jax.vmap(
            lambda k, s, a: env.step_env(k, s, a, env.default_params)
        ))(keys, env_states, jnp.array(actions))
        next_ids = np.where(done, model.terminal,
                            np.asarray(nxt.goal_dir) * (length + 2) + np.asarray(nxt.grid_idx))
        expected_next = np.argmax(model.T[actions, states], axis=-1)
        np.testing.assert_array_equal(next_ids, expected_next)
        np.testing.assert_allclose(reward, model.R[actions, states, expected_next], atol=1e-7)
        np.testing.assert_array_equal(obs, model.phi[states])
