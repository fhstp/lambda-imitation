"""Integration tests: real collection -> chosen anchors -> targets -> gradients."""

from dataclasses import replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from lambda_imitation.utils import RecurrentFeatureExtractor, battleship_projection
from lambda_imitation.envs.battleship import Battleship
from lambda_imitation.envs.minesweeper import MineSweeper
from .learner import create_learner, paper_settings
from .replay import select


def tiny(environment="battleship", **changes):
    return paper_settings(environment, hidden_dim=4, batch_size=2, sequence_length=2,
                          tail=2, replay_capacity=160, **changes)


@lru_cache(maxsize=16)
def prepared(environment="battleship", measure="discounted", correction="retrace", frozen=False):
    settings = tiny(environment, measure=measure, correction=correction, frozen=frozen)
    agent, rollout, fns = create_learner(settings, 12)
    agent, rollout = fns.prefill(agent, rollout, steps=160)
    return settings, agent, rollout, fns


def arrays_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


@pytest.mark.parametrize("environment", ["battleship", "minesweeper"])
def test_reconstruction_equals_independent_full_prefix_unroll(environment):
    settings, agent, _, fns = prepared(environment)
    batch = fns.sample(agent.online_buffer, jax.random.key(8))
    actual = fns.features(agent.feature_extractor, batch.info, batch.lengths)
    action_dim = 25 if environment == "battleship" else 36
    input_shape = (1,) if environment == "battleship" else (8,)
    projection = battleship_projection(4) if environment == "battleship" else 4
    key_fe = jax.random.split(jax.random.key(12))[0]
    prototype = RecurrentFeatureExtractor(input_shape, projection, "gru", 4, action_dim, rngs=nnx.Rngs(key_fe))
    graph, _ = nnx.split(prototype)
    model = nnx.merge(graph, agent.feature_extractor)
    for episode in range(len(batch.lengths)):
        carry, previous = jnp.zeros((1, 4)), jnp.zeros((1, action_dim))
        for age in range(int(batch.lengths[episode])):
            obs = batch.info["observations"][episode, age][None]
            carry, expected = model(carry, obs[:, :1] if environment == "battleship" else obs, previous)
            np.testing.assert_allclose(actual[episode, age], expected[0], rtol=1e-6, atol=1e-6)
            previous = jax.nn.one_hot(batch.info["actions"][episode, age], action_dim)[None]


@pytest.mark.parametrize("environment", ["battleship", "minesweeper"])
def test_actual_adaptive_collector_logs_the_probability_it_used(environment):
    _, agent, rollout, fns = prepared(environment)
    env = Battleship() if environment == "battleship" else MineSweeper()
    obs = env.get_obs(rollout.env_state, env.default_params)
    expected = np.asarray(fns.probabilities(agent, obs, rollout.pending, rollout.age))
    before_age, before_write = int(rollout.age), int(agent.online_buffer.write)
    after, after_rollout, _, _ = fns.train(agent, rollout, jax.random.key(99), steps=1)
    record = after_rollout.pending if int(after_rollout.age) else {
        k: v[before_write] for k, v in after.online_buffer.info.items()}
    action = int(record["actions"][before_age])
    assert float(record["behaviour_probs"][before_age]) == pytest.approx(expected[action], rel=1e-6)
    assert int(record["collector_version"][before_age]) == int(agent.update_step)
    assert int(after_rollout.calls) == int(rollout.calls) + 1


def test_real_loss_scores_anchors_not_padded_episode_positions():
    _, agent, _, fns = prepared()
    zero = lambda state: jax.tree.map(jnp.zeros_like, state)
    replay = agent.online_buffer
    replay = replay._replace(
        lengths=jnp.where(replay.lengths > 0, 1, 0),
        info={**replay.info, "rewards": jnp.full_like(replay.info["rewards"], 2),
              "terminated": jnp.ones_like(replay.info["terminated"])})
    agent = agent._replace(lambda1_critic=zero(agent.lambda1_critic),
        lambda2_critic=zero(agent.lambda2_critic), lambda1_critic_target=zero(agent.lambda1_critic_target),
        lambda2_critic_target=zero(agent.lambda2_critic_target), online_buffer=replay)
    _, metrics = fns.loss(*agent[:5], agent, jax.random.key(7))
    assert float(metrics["lambda1_loss"]) == pytest.approx(4.0)
    assert float(metrics["lambda2_loss"]) == pytest.approx(4.0)
    assert float(metrics["anchor_age_mean"]) == 0.0


def test_gradient_routing_matches_the_paper_objectives():
    _, agent, _, fns = prepared()
    key = jax.random.key(91)
    def metric(fe, l1, l2, name):
        return fns.loss(fe, agent.actor, agent.critic, l1, l2, agent, key)[1][name]
    actor_grad = jax.grad(lambda fe: metric(fe, agent.lambda1_critic, agent.lambda2_critic, "actor_loss"))(
        agent.feature_extractor)
    assert all(np.all(np.asarray(a) == 0) for a in jax.tree.leaves(actor_grad))
    grads = jax.grad(lambda fe, l1, l2: metric(fe, l1, l2, "discrepancy_loss"), argnums=(0, 1, 2))(
        agent.feature_extractor, agent.lambda1_critic, agent.lambda2_critic)
    assert any(np.any(np.asarray(a) != 0) for a in jax.tree.leaves(grads[0]))
    assert all(np.all(np.asarray(a) == 0) for a in jax.tree.leaves(grads[1:]))


@pytest.mark.parametrize("measure", ["uniform", "discounted"])
@pytest.mark.parametrize("correction", ["uncorrected", "is", "retrace"])
def test_all_six_actual_learner_updates_are_finite_and_train_both_auxiliary_heads(measure, correction):
    _, agent, _, fns = prepared(measure=measure, correction=correction)
    after, metrics = fns.update(agent, jax.random.key(551))
    assert int(after.update_step) == 1
    assert all(np.isfinite(a).all() for a in jax.tree.leaves((after[:15], metrics)))
    assert not arrays_equal(agent.feature_extractor, after.feature_extractor)
    for name in ("lambda1_critic", "lambda2_critic"):
        before, updated = getattr(agent, name), getattr(after, name)
        for a, b in zip(before, updated):
            assert not arrays_equal(a, b)


def test_is_arm_uses_unclipped_ratios_in_the_actual_training_loss():
    _, agent, _, is_fns = prepared(correction="is")
    _, _, _, retrace_fns = prepared(correction="retrace")
    replay = agent.online_buffer
    # Pointwise numerical check, not a claim that this modified fixture is a
    # newly collected stationary dataset. Logged probabilities are deliberately
    # made small to exercise ratios greater than one in the real loss path.
    agent = agent._replace(online_buffer=replay._replace(info={**replay.info,
        "behaviour_probs": replay.info["behaviour_probs"] * 0.1}))
    key = jax.random.key(3)
    _, is_metrics = is_fns.loss(*agent[:5], agent, key)
    _, rt_metrics = retrace_fns.loss(*agent[:5], agent, key)
    assert float(is_metrics["importance_ratio_max"]) > 1
    assert not np.isclose(float(is_metrics["lambda2_loss"]), float(rt_metrics["lambda2_loss"]))


def test_frozen_fits_really_freeze_encoder_actor_and_task_critic():
    _, agent, _, fns = prepared(frozen=True)
    after, _, metrics = fns.fit(agent, jax.random.key(42), steps=3)
    for i in (0, 1, 2, 5, 6, 7):
        assert arrays_equal(agent[i], after[i])
    assert not arrays_equal(agent.lambda1_critic, after.lambda1_critic)
    assert not arrays_equal(agent.lambda2_critic, after.lambda2_critic)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))


@pytest.mark.parametrize("environment", ["battleship", "minesweeper"])
def test_evaluation_uses_complete_first_episodes(environment):
    _, agent, _, fns = prepared(environment)
    for greedy in (False, True):
        result = fns.evaluate(agent, jax.random.key(55), episodes=4, greedy=greedy)
        assert float(result["completed"]) == 1.0
        assert all(np.isfinite(x).all() for x in jax.tree.leaves(result))
