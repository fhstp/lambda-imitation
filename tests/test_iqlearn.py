"""Tests for IQ-Learn (Inverse Q-Learning) implementation.

Covers: Hyperparameters defaults, create_iqlearn factory, predict, train,
and a small end-to-end integration test with a synthetic buffer.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from lambda_imitation.buffer import create_buffer
from lambda_imitation.iqlearn import (
    ActorHead,
    CriticHead,
    Hyperparameters,
    IQLearnFunctions,
    IQLearnGraphs,
    IQLearnState,
    MLPFeatureExtractor,
    NetworkState,
    create_iqlearn,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2
BUFFER_SIZE = 32
BATCH_SIZE = 8
TRAIN_STEPS = 3
FE_DIMS = (32, 32)  # small FE for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filled_buffer(
    obs_dim=OBS_DIM,
    action_dim=ACTION_DIM,
    size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
):
    """Create and fully fill a buffer with random data.

    The last step is marked terminal so every slot is sampleable.
    """
    buf, fns = create_buffer(
        shapes={"observations": (obs_dim,), "actions": (action_dim,)},
        size=size,
        sampling_size=batch_size,
        this_step_infos=["observations", "actions"],
        next_step_infos=["observations"],
    )
    key = jax.random.key(0)
    for i in range(size):
        key, k_obs, k_act = jax.random.split(key, 3)
        terminated = i == size - 1
        buf = fns.add(
            buf,
            {
                "observations": jax.random.normal(k_obs, (obs_dim,)),
                "actions": jax.random.uniform(k_act, (action_dim,), minval=-1, maxval=1),
            },
            terminated,
        )
    return buf, fns


def make_feature_extractors(obs_dim=OBS_DIM, hidden_dims=FE_DIMS):
    """Create a pair of MLPFeatureExtractor instances for actor and critic."""
    rngs = nnx.Rngs(0)
    actor_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    critic_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    return actor_fe, critic_fe


def make_iqlearn(
    buf=None,
    hp=None,
    train_steps=TRAIN_STEPS,
    autotune_alpha=True,
    action_scale=1.0,
    action_bias=0.0,
    actor_head_dims=(),
    critic_head_dims=(32,),
    obs_dim=OBS_DIM,
):
    """Thin wrapper around create_iqlearn with test-friendly defaults."""
    if buf is None:
        buf, _ = make_filled_buffer(obs_dim=obs_dim)
    if hp is None:
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            autotune_alpha=autotune_alpha,
        )
    actor_fe, critic_fe = make_feature_extractors(obs_dim=obs_dim)
    return create_iqlearn(
        hp,
        buf,
        ACTION_DIM,
        actor_fe,
        critic_fe,
        train_steps=train_steps,
        action_scale=action_scale,
        action_bias=action_bias,
        actor_head_dims=actor_head_dims,
        critic_head_dims=critic_head_dims,
    )


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

class TestHyperparameters:
    def test_defaults(self):
        hp = Hyperparameters()
        assert hp.actor_lr == 1e-3
        assert hp.critic_lr == 1e-3
        assert hp.alpha_lr == 1e-3
        assert hp.alpha == 1.0
        assert hp.autotune_alpha is True
        assert hp.batch_size == 256
        assert hp.gamma == 0.99
        assert hp.regularizer_coef == 1 / 40
        assert hp.target_entropy == -1
        assert hp.tau == 0.005

    def test_override(self):
        hp = Hyperparameters(actor_lr=5e-4, batch_size=64)
        assert hp.actor_lr == 5e-4
        assert hp.batch_size == 64
        assert hp.gamma == 0.99  # unchanged default


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class TestMLPFeatureExtractor:
    def test_output_shape(self):
        fe = MLPFeatureExtractor(OBS_DIM, (64, 64), rngs=nnx.Rngs(0))
        x = jnp.ones((5, OBS_DIM))
        out = fe(x)
        assert out.shape == (5, 64)

    def test_flattens_input(self):
        """2-D obs (e.g. image-like) should be flattened automatically."""
        fe = MLPFeatureExtractor(6, (16,), rngs=nnx.Rngs(0))
        x = jnp.ones((3, 2, 3))  # batch=3, obs shape (2,3) -> flat dim 6
        out = fe(x)
        assert out.shape == (3, 16)

    def test_single_hidden_layer(self):
        fe = MLPFeatureExtractor(OBS_DIM, (8,), rngs=nnx.Rngs(0))
        out = fe(jnp.ones((2, OBS_DIM)))
        assert out.shape == (2, 8)


class TestActorHead:
    def test_default_no_hidden(self):
        head = ActorHead(32, ACTION_DIM, (), rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, 2 * ACTION_DIM)

    def test_with_hidden_layers(self):
        head = ActorHead(32, ACTION_DIM, (16, 16), rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, 2 * ACTION_DIM)


class TestCriticHead:
    def test_default_output(self):
        head = CriticHead(32, ACTION_DIM, (16,), rngs=nnx.Rngs(0))
        features = jnp.ones((4, 32))
        actions = jnp.ones((4, ACTION_DIM))
        out = head(features, actions)
        assert out.shape == (4, 2)  # twin Q

    def test_no_hidden(self):
        head = CriticHead(32, ACTION_DIM, (), rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)), jnp.ones((4, ACTION_DIM)))
        assert out.shape == (4, 2)


# ---------------------------------------------------------------------------
# create_iqlearn
# ---------------------------------------------------------------------------

class TestCreateIQLearn:
    def test_return_types(self):
        state, fns, graphs = make_iqlearn()
        assert isinstance(state, IQLearnState)
        assert isinstance(fns, IQLearnFunctions)
        assert isinstance(graphs, IQLearnGraphs)

    def test_state_fields_are_network_states(self):
        state, _, _ = make_iqlearn()
        for field in (state.actor, state.critic, state.actor_target, state.critic_target):
            assert isinstance(field, NetworkState)
            assert field.fe is not None
            assert field.head is not None

    def test_optimizer_states_populated(self):
        state, _, _ = make_iqlearn()
        assert state.actor_optimizer_state is not None
        assert state.critic_optimizer_state is not None
        assert state.alpha_optimizer_state is not None

    def test_targets_match_initial_params(self):
        state, _, _ = make_iqlearn()
        for online, target in [
            (state.actor, state.actor_target),
            (state.critic, state.critic_target),
        ]:
            for o, t in zip(jax.tree.leaves(online), jax.tree.leaves(target)):
                assert (o == t).all()

    def test_alpha_matches_hp(self):
        state, _, _ = make_iqlearn()
        assert jnp.isclose(state.alpha, 1.0)
        assert jnp.isclose(state.log_alpha, jnp.log(1.0))

    def test_custom_alpha(self):
        hp = Hyperparameters(alpha=0.5, batch_size=BATCH_SIZE)
        buf, _ = make_filled_buffer()
        actor_fe, critic_fe = make_feature_extractors()
        state, _, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_fe, train_steps=TRAIN_STEPS,
        )
        assert jnp.isclose(state.alpha, 0.5, atol=1e-6)

    def test_graphs_have_four_fields(self):
        _, _, graphs = make_iqlearn()
        assert graphs.actor_fe is not None
        assert graphs.actor_head is not None
        assert graphs.critic_fe is not None
        assert graphs.critic_head is not None

    def test_feature_dim_inferred_for_custom_fe(self):
        """A FE with output dim 64 should produce actor head output 2*action_dim."""
        buf, _ = make_filled_buffer()
        actor_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(1))
        critic_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(2))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_fe, train_steps=TRAIN_STEPS,
        )
        # Should be able to predict without shape errors
        action = fns.predict(state, jnp.ones(OBS_DIM), deterministic=True)
        assert action.shape == (ACTION_DIM,)

    def test_separate_fe_architectures(self):
        """Actor and critic can have different FE architectures."""
        buf, _ = make_filled_buffer()
        actor_fe = MLPFeatureExtractor(OBS_DIM, (16,), rngs=nnx.Rngs(0))
        critic_fe = MLPFeatureExtractor(OBS_DIM, (64, 64), rngs=nnx.Rngs(1))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_fe, train_steps=TRAIN_STEPS,
        )
        action = fns.predict(state, jnp.ones(OBS_DIM), deterministic=True)
        assert action.shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture()
    def setup(self):
        state, fns, _ = make_iqlearn()
        obs = jnp.ones(OBS_DIM)
        return state, fns, obs

    def test_output_shape(self, setup):
        state, fns, obs = setup
        action = fns.predict(state, obs, deterministic=True)
        assert action.shape == (ACTION_DIM,)

    def test_deterministic_ignores_key(self, setup):
        state, fns, obs = setup
        a1 = fns.predict(state, obs, key=jax.random.key(0), deterministic=True)
        a2 = fns.predict(state, obs, key=jax.random.key(999), deterministic=True)
        assert jnp.allclose(a1, a2)

    def test_stochastic_varies(self, setup):
        state, fns, obs = setup
        a1 = fns.predict(state, obs, key=jax.random.key(0), deterministic=False)
        a2 = fns.predict(state, obs, key=jax.random.key(1), deterministic=False)
        assert not jnp.allclose(a1, a2)

    def test_actions_bounded(self, setup):
        """With default scale=1 bias=0, tanh output is in (-1, 1)."""
        state, fns, obs = setup
        for seed in range(10):
            action = fns.predict(state, obs, key=jax.random.key(seed), deterministic=False)
            assert (action > -1.0 - 1e-6).all()
            assert (action < 1.0 + 1e-6).all()

    def test_scale_and_bias(self):
        """action_scale and action_bias shift the output range."""
        state, fns, _ = make_iqlearn(action_scale=2.0, action_bias=1.0)
        obs = jnp.ones(OBS_DIM)
        action = fns.predict(state, obs, key=jax.random.key(0), deterministic=True)
        # tanh in (-1,1) * 2 + 1 => (-1, 3)
        assert (action > -1.0 - 1e-6).all()
        assert (action < 3.0 + 1e-6).all()

    def test_multidim_obs_flattened(self):
        """FE should flatten non-flat obs; predict should still work."""
        obs_shape = (2, 3)  # flat dim = 6
        obs_dim = 6
        buf, _ = make_filled_buffer(obs_dim=obs_dim)
        # Reshape obs in buffer to 2-D (simulate image-like)
        import numpy as np
        buf2, fns2 = create_buffer(
            shapes={"observations": obs_shape, "actions": (ACTION_DIM,)},
            size=BUFFER_SIZE,
            sampling_size=BATCH_SIZE,
            this_step_infos=["observations", "actions"],
            next_step_infos=["observations"],
        )
        key = jax.random.key(42)
        for i in range(BUFFER_SIZE):
            key, k_obs, k_act = jax.random.split(key, 3)
            buf2 = fns2.add(
                buf2,
                {
                    "observations": jax.random.normal(k_obs, obs_shape),
                    "actions": jax.random.uniform(k_act, (ACTION_DIM,), minval=-1, maxval=1),
                },
                i == BUFFER_SIZE - 1,
            )
        actor_fe = MLPFeatureExtractor(obs_dim, FE_DIMS, rngs=nnx.Rngs(0))
        critic_fe = MLPFeatureExtractor(obs_dim, FE_DIMS, rngs=nnx.Rngs(1))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf2, ACTION_DIM, actor_fe, critic_fe, train_steps=TRAIN_STEPS,
        )
        obs = jnp.ones(obs_shape)
        action = fns.predict(state, obs, deterministic=True)
        assert action.shape == (ACTION_DIM,)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    @pytest.fixture()
    def trained(self):
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(buf=buf)
        new_state, metrics = fns.train(state, jax.random.key(42))
        return state, new_state, fns, metrics

    def test_return_types(self, trained):
        _, new_state, _, metrics = trained
        assert isinstance(new_state, IQLearnState)
        assert isinstance(metrics, dict)

    def test_expected_metric_keys(self, trained):
        _, _, _, metrics = trained
        expected = {
            "q", "entropy", "v",
            "demonstration_loss", "mixed_loss",
            "regularizer_loss", "critic_loss", "alpha",
        }
        assert expected <= set(metrics.keys())

    def test_metrics_finite(self, trained):
        _, _, _, metrics = trained
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_actor_fe_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.actor.fe)
        new = jax.tree.leaves(new_state.actor.fe)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "actor FE params should change after training"

    def test_actor_head_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.actor.head)
        new = jax.tree.leaves(new_state.actor.head)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "actor head params should change after training"

    def test_critic_fe_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.critic.fe)
        new = jax.tree.leaves(new_state.critic.fe)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "critic FE params should change after training"

    def test_critic_head_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.critic.head)
        new = jax.tree.leaves(new_state.critic.head)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "critic head params should change after training"

    def test_targets_lag_behind_online(self, trained):
        """After training with tau < 1, targets should differ from online params."""
        _, new_state, _, _ = trained
        for online, target in [
            (new_state.actor, new_state.actor_target),
            (new_state.critic, new_state.critic_target),
        ]:
            differs = any(
                not jnp.allclose(o, t)
                for o, t in zip(jax.tree.leaves(online), jax.tree.leaves(target))
            )
            assert differs, "target should lag behind online after training"

    def test_alpha_updates_with_autotune(self, trained):
        old_state, new_state, _, _ = trained
        assert not jnp.allclose(old_state.alpha, new_state.alpha), \
            "alpha should change when autotune_alpha=True"

    def test_alpha_fixed_without_autotune(self):
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(buf=buf, autotune_alpha=False)
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert jnp.allclose(state.alpha, new_state.alpha), \
            "alpha should stay fixed when autotune_alpha=False"
        assert "alpha" not in metrics

    def test_reproducible_with_same_key(self):
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(buf=buf)
        s1, m1 = fns.train(state, jax.random.key(7))
        s2, m2 = fns.train(state, jax.random.key(7))
        for l1, l2 in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
            assert jnp.allclose(l1, l2)
        for k in m1:
            assert jnp.allclose(m1[k], m2[k])

    def test_custom_head_dims(self):
        """Training should work with non-default head dims."""
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(
            buf=buf, actor_head_dims=(16, 16), critic_head_dims=(16,),
        )
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert isinstance(new_state, IQLearnState)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' not finite"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_end_to_end(self):
        """Fill buffer -> create iqlearn -> train 2 rounds -> predict."""
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(buf=buf, train_steps=TRAIN_STEPS)

        # Train two rounds
        key = jax.random.key(0)
        key, k1, k2 = jax.random.split(key, 3)
        state, metrics_1 = fns.train(state, k1)
        state, metrics_2 = fns.train(state, k2)

        # All metrics from both rounds should be finite
        for m in (metrics_1, metrics_2):
            for k, v in m.items():
                assert jnp.isfinite(v), f"metric '{k}' not finite after training"

        # Predict deterministic
        obs = jnp.ones(OBS_DIM)
        action_det = fns.predict(state, obs, deterministic=True)
        assert action_det.shape == (ACTION_DIM,)
        assert jnp.isfinite(action_det).all()
        assert (action_det > -1.0 - 1e-6).all()
        assert (action_det < 1.0 + 1e-6).all()

        # Predict stochastic
        action_sto = fns.predict(state, obs, key=jax.random.key(99), deterministic=False)
        assert action_sto.shape == (ACTION_DIM,)
        assert jnp.isfinite(action_sto).all()
