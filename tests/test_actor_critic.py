"""Publication objective, gradient routing, masks, and recurrent training."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation import EnvSpec, Hyperparameters, create_actor_critic_from_env


def agent(*, auxiliary=True, masked=False):
    hp = Hyperparameters(batch_size=4, online_buffer_size=32,
                         burn_in_length=1, sequence_length=2, lambda_truncation=2)
    state, fns, debug = create_actor_critic_from_env(
        EnvSpec((3,), 2), hp=hp, projection=4, memory_type="gru", memory_hidden_dim=4,
        use_prev_action=True, actor_dims=(), critic_dims=(),
        lambda1_critic_dims=(), lambda2_critic_dims=(), approximate_lambda=auxiliary,
        mask_fn=(lambda o: o[..., 1:] > 0) if masked else None,
        obs_fn=(lambda o: o[..., :1]) if masked else lambda o: o,
        train_steps=2, seed=2, debug=True)
    info = {**state.online_buffer.info,
            "observations": jax.random.normal(jax.random.key(0), (32, 3)),
            "actions": (jnp.arange(32) % 2).reshape(32, 1).astype(jnp.float32),
            "rewards": jnp.ones(32), "behaviour_weight": jnp.full(32, 0.5),
            "terminated": (jnp.arange(32) % 4 == 0).astype(jnp.float32)}
    state = state._replace(online_buffer=state.online_buffer._replace(
        info=info, sampling_ok=jnp.ones(32, bool), pos=32))
    return state, fns, debug


def test_auxiliary_regression_is_huber_and_trains_both_twins():
    state, _, debug = agent()
    zero = lambda x: jax.tree.map(jnp.zeros_like, x)
    state = state._replace(
        lambda1_critic=zero(state.lambda1_critic), lambda2_critic=zero(state.lambda2_critic),
        lambda1_critic_target=zero(state.lambda1_critic_target),
        lambda2_critic_target=zero(state.lambda2_critic_target),
        online_buffer=state.online_buffer._replace(info={**state.online_buffer.info,
            "rewards": jnp.full(32, 20.0), "terminated": jnp.ones(32)}))
    after, metrics = debug.update_step(state, jax.random.key(3))
    for lam in (0.05, 0.85):
        assert float(metrics[f"lambda{lam}_loss"]) == pytest.approx(19.5)
    for name in ("lambda1_critic", "lambda2_critic"):
        for branch in getattr(after, name):
            assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(branch))


def test_actor_cannot_train_memory_and_discrepancy_cannot_train_heads():
    state, _, debug = agent()
    key = jax.random.key(3)

    def metric(fe, actor, critic, l1, l2, name):
        return debug.loss(fe, actor, critic, l1, l2, state, key)[1][name]

    actor_grad = jax.grad(lambda fe: metric(fe, *state[1:5], "actor_loss"))(state.feature_extractor)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(actor_grad))
    grads = jax.grad(lambda fe, l1, l2: metric(fe, state.actor, state.critic, l1, l2, "ld_loss"),
                     argnums=(0, 1, 2))(state.feature_extractor, state.lambda1_critic, state.lambda2_critic)
    assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads[0]))
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree.leaves(grads[1:]))


@pytest.mark.parametrize("auxiliary", [False, True])
def test_joint_update_changes_memory_and_has_finite_metrics(auxiliary):
    state, _, debug = agent(auxiliary=auxiliary)
    after, metrics = debug.update_step(state, jax.random.key(3))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(metrics))
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(state.feature_extractor), jax.tree.leaves(after.feature_extractor)))
    assert (after.lambda1_critic is not None) == auxiliary
    assert ("ld_loss" in metrics) == auxiliary


def test_action_mask_applies_to_prediction_probabilities_and_gradients():
    state, fns, debug = agent(masked=True)
    obs, memory, pa = jnp.array([0.5, 0.0, 1.0]), jnp.zeros(4), jnp.zeros(2)
    action, _, prob = fns.predict(state, obs, memory, jax.random.key(1),
                                  return_prob=True, prev_action=pa)
    assert action == 1 and prob == 1
    _, probs, _ = debug.predict_qpi(state, obs, memory, pa)
    np.testing.assert_array_equal(probs, [0, 1])
    # Fully masked zero rows must remain numerically well-defined.
    info = {**state.online_buffer.info, "observations": jnp.zeros((32, 3))}
    after, metrics = debug.update_step(state._replace(
        online_buffer=state.online_buffer._replace(info=info)), jax.random.key(1))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((after, metrics)))


def test_continuous_action_space_is_rejected():
    import gymnax
    from lambda_imitation import env_spec_from_gymnax

    env, params = gymnax.make("Pendulum-v1")
    with pytest.raises(ValueError, match="discrete"):
        env_spec_from_gymnax(env, params)
