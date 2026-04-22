"""Tests for IQ-Learn (Inverse Q-Learning) implementation.

Covers: Hyperparameters defaults, create_iqlearn factory, predict, train,
and a small end-to-end integration test with a synthetic buffer.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from typing import NamedTuple

from lambda_imitation.buffer import create_buffer
from lambda_imitation.iqlearn import (
    Head,
    Hyperparameters,
    IQLearnFunctions,
    IQLearnGraphs,
    IQLearnState,
    MLPFeatureExtractor,
    NetworkGraphs,
    NetworkState,
    TwinCriticState,
    create_iqlearn,
    extract_buffer_shapes,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

OBS_DIM = 4
ACTION_DIM = 2
NUM_ACTIONS = 4  # discrete action count
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
    """Create three independent MLPFeatureExtractor instances.

    Returns ``(actor_fe, critic_q1_fe, critic_q2_fe)`` — one per network role.
    """
    rngs = nnx.Rngs(0)
    actor_fe    = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    critic_q1_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    critic_q2_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    return actor_fe, critic_q1_fe, critic_q2_fe


def make_discrete_buffer(
    obs_dim=OBS_DIM,
    num_actions=NUM_ACTIONS,
    size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
):
    """Create and fully fill a buffer for a discrete action space.

    Actions are stored as float32 scalars of shape ``(1,)``.
    """
    buf, fns = create_buffer(
        shapes={"observations": (obs_dim,), "actions": (1,)},
        size=size,
        sampling_size=batch_size,
        this_step_infos=["observations", "actions"],
        next_step_infos=["observations"],
    )
    key = jax.random.key(0)
    for i in range(size):
        key, k_obs, k_act = jax.random.split(key, 3)
        terminated = i == size - 1
        action_idx = jax.random.randint(k_act, (1,), 0, num_actions).astype(jnp.float32)
        buf = fns.add(
            buf,
            {
                "observations": jax.random.normal(k_obs, (obs_dim,)),
                "actions": action_idx,
            },
            terminated,
        )
    return buf, fns


def make_discrete_iqlearn(
    buf=None,
    hp=None,
    train_steps=TRAIN_STEPS,
    num_actions=NUM_ACTIONS,
    obs_dim=OBS_DIM,
    critic_head_dims=(32,),
    autotune_alpha=True,
):
    """Thin wrapper around create_iqlearn with is_discrete=True."""
    import math
    if buf is None:
        buf, _ = make_discrete_buffer(obs_dim=obs_dim, num_actions=num_actions)
    if hp is None:
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            autotune_alpha=autotune_alpha,
            target_entropy=float(0.98 * math.log(num_actions)),
        )
    actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors(obs_dim=obs_dim)
    return create_iqlearn(
        hp,
        buf,
        num_actions,
        actor_fe,
        critic_q1_fe,
        critic_q2_fe,
        train_steps=train_steps,
        critic_head_dims=critic_head_dims,
        is_discrete=True,
    )


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
    actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors(obs_dim=obs_dim)
    return create_iqlearn(
        hp,
        buf,
        ACTION_DIM,
        actor_fe,
        critic_q1_fe,
        critic_q2_fe,
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


class TestHead:
    def test_no_hidden(self):
        """Direct linear projection: no hidden layers."""
        head = Head(32, (), 8, rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, 8)

    def test_with_hidden_layers(self):
        head = Head(32, (16, 16), 8, rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, 8)

    def test_single_output(self):
        """output_dim=1 as used by each continuous critic Q-branch."""
        head = Head(64, (32,), 1, rngs=nnx.Rngs(0))
        out = head(jnp.ones((3, 64)))
        assert out.shape == (3, 1)

    def test_large_output(self):
        """output_dim=2*action_dim as used by the continuous actor head."""
        head = Head(32, (), 2 * ACTION_DIM, rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, 2 * ACTION_DIM)

    def test_discrete_actor_shape(self):
        """output_dim=num_actions as used by the discrete actor head."""
        head = Head(32, (), NUM_ACTIONS, rngs=nnx.Rngs(0))
        out = head(jnp.ones((4, 32)))
        assert out.shape == (4, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# create_iqlearn
# ---------------------------------------------------------------------------

class TestCreateIQLearn:
    def test_return_types(self):
        state, fns, graphs = make_iqlearn()
        assert isinstance(state, IQLearnState)
        assert isinstance(fns, IQLearnFunctions)
        assert isinstance(graphs, IQLearnGraphs)

    def test_actor_state_is_network_state(self):
        state, _, _ = make_iqlearn()
        for field in (state.actor, state.actor_target):
            assert isinstance(field, NetworkState)
            assert field.fe is not None
            assert field.head is not None

    def test_critic_state_is_twin_critic_state(self):
        state, _, _ = make_iqlearn()
        for field in (state.critic, state.critic_target):
            assert isinstance(field, TwinCriticState)
            assert isinstance(field.q1, NetworkState)
            assert isinstance(field.q2, NetworkState)
            assert field.q1.fe is not None
            assert field.q1.head is not None
            assert field.q2.fe is not None
            assert field.q2.head is not None

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
        actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors()
        state, _, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS,
        )
        assert jnp.isclose(state.alpha, 0.5, atol=1e-6)

    def test_graphs_have_correct_fields(self):
        _, _, graphs = make_iqlearn()
        assert isinstance(graphs.actor, NetworkGraphs)
        assert isinstance(graphs.critic_q1, NetworkGraphs)
        assert isinstance(graphs.critic_q2, NetworkGraphs)
        assert graphs.actor.fe is not None
        assert graphs.actor.head is not None
        assert graphs.critic_q1.fe is not None
        assert graphs.critic_q1.head is not None
        assert graphs.critic_q2.fe is not None
        assert graphs.critic_q2.head is not None

    def test_feature_dim_inferred_for_custom_fe(self):
        """A FE with output dim 64 should produce actor head output 2*action_dim."""
        buf, _ = make_filled_buffer()
        actor_fe    = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(1))
        critic_q1_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(2))
        critic_q2_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(3))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS,
        )
        # Should be able to predict without shape errors
        action = fns.predict(state, jnp.ones(OBS_DIM), deterministic=True)
        assert action.shape == (ACTION_DIM,)

    def test_separate_fe_architectures(self):
        """Actor and critic Q-branches can have different FE architectures."""
        buf, _ = make_filled_buffer()
        actor_fe    = MLPFeatureExtractor(OBS_DIM, (16,), rngs=nnx.Rngs(0))
        critic_q1_fe = MLPFeatureExtractor(OBS_DIM, (64, 64), rngs=nnx.Rngs(1))
        critic_q2_fe = MLPFeatureExtractor(OBS_DIM, (64, 64), rngs=nnx.Rngs(2))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS,
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
        actor_fe    = MLPFeatureExtractor(obs_dim, FE_DIMS, rngs=nnx.Rngs(0))
        critic_q1_fe = MLPFeatureExtractor(obs_dim, FE_DIMS, rngs=nnx.Rngs(1))
        critic_q2_fe = MLPFeatureExtractor(obs_dim, FE_DIMS, rngs=nnx.Rngs(2))
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        state, fns, _ = create_iqlearn(
            hp, buf2, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS,
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

    def test_critic_params_change(self, trained):
        """At least one parameter across the twin-critic should change."""
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.critic)
        new = jax.tree.leaves(new_state.critic)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "critic params should change after training"

    def test_twin_q_branches_independent(self):
        """Q1 and Q2 must start with different parameters (true independence)."""
        state, _, _ = make_iqlearn()
        q1_leaves = jax.tree.leaves(state.critic.q1)
        q2_leaves = jax.tree.leaves(state.critic.q2)
        assert any(
            not jnp.allclose(a, b) for a, b in zip(q1_leaves, q2_leaves)
        ), "Q1 and Q2 should have independent (different) initial parameters"

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


# ---------------------------------------------------------------------------
# create_iqlearn (discrete)
# ---------------------------------------------------------------------------

class TestCreateIQLearnDiscrete:
    def test_return_types(self):
        state, fns, graphs = make_discrete_iqlearn()
        assert isinstance(state, IQLearnState)
        assert isinstance(fns, IQLearnFunctions)
        assert isinstance(graphs, IQLearnGraphs)

    def test_actor_state_is_network_state(self):
        state, _, _ = make_discrete_iqlearn()
        for field in (state.actor, state.actor_target):
            assert isinstance(field, NetworkState)
            assert field.fe is not None
            assert field.head is not None

    def test_critic_state_is_twin_critic_state(self):
        state, _, _ = make_discrete_iqlearn()
        for field in (state.critic, state.critic_target):
            assert isinstance(field, TwinCriticState)
            assert isinstance(field.q1, NetworkState)
            assert isinstance(field.q2, NetworkState)

    def test_graphs_have_correct_fields(self):
        _, _, graphs = make_discrete_iqlearn()
        assert isinstance(graphs.actor, NetworkGraphs)
        assert isinstance(graphs.critic_q1, NetworkGraphs)
        assert isinstance(graphs.critic_q2, NetworkGraphs)

    def test_targets_match_initial_params(self):
        state, _, _ = make_discrete_iqlearn()
        for online, target in [
            (state.actor, state.actor_target),
            (state.critic, state.critic_target),
        ]:
            for o, t in zip(jax.tree.leaves(online), jax.tree.leaves(target)):
                assert (o == t).all()

    def test_feature_dim_inferred(self):
        """Feature dim should be inferred correctly; predict must not error."""
        buf, _ = make_discrete_buffer()
        actor_fe    = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(1))
        critic_q1_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(2))
        critic_q2_fe = MLPFeatureExtractor(OBS_DIM, (64,), rngs=nnx.Rngs(3))
        import math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
        )
        state, fns, _ = create_iqlearn(
            hp, buf, NUM_ACTIONS, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS, is_discrete=True,
        )
        action = fns.predict(state, jnp.ones(OBS_DIM), deterministic=True)
        assert action.shape == ()


# ---------------------------------------------------------------------------
# predict (discrete)
# ---------------------------------------------------------------------------

class TestPredictDiscrete:
    @pytest.fixture()
    def setup(self):
        state, fns, _ = make_discrete_iqlearn()
        obs = jnp.ones(OBS_DIM)
        return state, fns, obs

    def test_output_is_float32_scalar(self, setup):
        state, fns, obs = setup
        action = fns.predict(state, obs, deterministic=True)
        assert action.shape == ()
        assert action.dtype == jnp.float32

    def test_deterministic_ignores_key(self, setup):
        state, fns, obs = setup
        a1 = fns.predict(state, obs, key=jax.random.key(0), deterministic=True)
        a2 = fns.predict(state, obs, key=jax.random.key(999), deterministic=True)
        assert jnp.allclose(a1, a2)

    def test_stochastic_varies(self, setup):
        """Different keys should (with high probability) yield different actions."""
        state, fns, obs = setup
        results = {
            int(fns.predict(state, obs, key=jax.random.key(s), deterministic=False))
            for s in range(20)
        }
        assert len(results) > 1, "stochastic predict should not always return same action"

    def test_valid_action_range_deterministic(self, setup):
        state, fns, obs = setup
        action = fns.predict(state, obs, deterministic=True)
        assert float(action) >= 0.0
        assert float(action) < NUM_ACTIONS

    def test_valid_action_range_stochastic(self, setup):
        state, fns, obs = setup
        for seed in range(20):
            action = fns.predict(state, obs, key=jax.random.key(seed), deterministic=False)
            assert float(action) >= 0.0
            assert float(action) < NUM_ACTIONS


# ---------------------------------------------------------------------------
# train (discrete)
# ---------------------------------------------------------------------------

class TestTrainDiscrete:
    @pytest.fixture()
    def trained(self):
        buf, _ = make_discrete_buffer()
        state, fns, _ = make_discrete_iqlearn(buf=buf)
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

    def test_actor_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.actor)
        new = jax.tree.leaves(new_state.actor)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "actor params should change after training"

    def test_critic_q1_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.critic.q1)
        new = jax.tree.leaves(new_state.critic.q1)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "critic Q1 params should change after training"

    def test_critic_q2_params_change(self, trained):
        old_state, new_state, _, _ = trained
        old = jax.tree.leaves(old_state.critic.q2)
        new = jax.tree.leaves(new_state.critic.q2)
        assert any(not (o == n).all() for o, n in zip(old, new)), \
            "critic Q2 params should change after training"

    def test_targets_lag_behind_online(self, trained):
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
        buf, _ = make_discrete_buffer()
        state, fns, _ = make_discrete_iqlearn(buf=buf, autotune_alpha=False)
        new_state, metrics = fns.train(state, jax.random.key(0))
        assert jnp.allclose(state.alpha, new_state.alpha), \
            "alpha should stay fixed when autotune_alpha=False"
        assert "alpha" not in metrics

    def test_reproducible_with_same_key(self):
        buf, _ = make_discrete_buffer()
        state, fns, _ = make_discrete_iqlearn(buf=buf)
        s1, m1 = fns.train(state, jax.random.key(7))
        s2, m2 = fns.train(state, jax.random.key(7))
        for l1, l2 in zip(jax.tree.leaves(s1), jax.tree.leaves(s2)):
            assert jnp.allclose(l1, l2)
        for k in m1:
            assert jnp.allclose(m1[k], m2[k])


# ---------------------------------------------------------------------------
# extract_buffer_shapes
# ---------------------------------------------------------------------------


class TestExtractBufferShapes:
    def test_continuous_buffer(self):
        buf, _ = make_filled_buffer()
        shapes = extract_buffer_shapes(buf)
        assert shapes == {"observations": (OBS_DIM,), "actions": (ACTION_DIM,)}

    def test_discrete_buffer(self):
        buf, _ = make_discrete_buffer()
        shapes = extract_buffer_shapes(buf)
        assert shapes == {"observations": (OBS_DIM,), "actions": (1,)}

    def test_scalar_value(self):
        """Buffers with scalar-valued keys should return empty tuples."""
        from lambda_imitation.buffer import create_buffer
        buf, _ = create_buffer(
            shapes={"obs": (3,), "rew": ()},
            size=8,
            sampling_size=4,
            this_step_infos=["obs", "rew"],
            next_step_infos=["obs"],
        )
        shapes = extract_buffer_shapes(buf)
        assert shapes["obs"] == (3,)
        assert shapes["rew"] == ()

    def test_multidim_obs(self):
        """2-D observation shapes are preserved as-is."""
        from lambda_imitation.buffer import create_buffer
        buf, _ = create_buffer(
            shapes={"observations": (2, 3), "actions": (2,)},
            size=8,
            sampling_size=4,
            this_step_infos=["observations", "actions"],
            next_step_infos=["observations"],
        )
        shapes = extract_buffer_shapes(buf)
        assert shapes["observations"] == (2, 3)


# ---------------------------------------------------------------------------
# IQLearnState.online_buffer
# ---------------------------------------------------------------------------


class TestOnlineBuffer:
    def test_online_buffer_in_state(self):
        state, _, _ = make_iqlearn()
        assert hasattr(state, "online_buffer")
        from lambda_imitation.buffer import Buffer
        assert isinstance(state.online_buffer, Buffer)

    def test_online_buffer_has_reward_and_terminated(self):
        state, _, _ = make_iqlearn()
        assert "rewards" in state.online_buffer.info
        assert "terminated" in state.online_buffer.info

    def test_online_buffer_obs_action_shape(self):
        state, _, _ = make_iqlearn()
        buf = state.online_buffer
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        assert buf.info["observations"].shape == (hp.online_buffer_size, OBS_DIM)
        assert buf.info["actions"].shape == (hp.online_buffer_size, ACTION_DIM)

    def test_online_buffer_reward_shape(self):
        state, _, _ = make_iqlearn()
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        assert state.online_buffer.info["rewards"].shape == (hp.online_buffer_size,)

    def test_online_buffer_terminated_shape(self):
        state, _, _ = make_iqlearn()
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        assert state.online_buffer.info["terminated"].shape == (hp.online_buffer_size,)

    def test_online_buffer_is_prefilled(self):
        """create_iqlearn pre-fills the online buffer to online_batch_size slots."""
        hp = Hyperparameters(batch_size=BATCH_SIZE, online_batch_size=BATCH_SIZE)
        buf, _ = make_filled_buffer()
        actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors()
        state, _, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            train_steps=TRAIN_STEPS,
        )
        assert int(state.online_buffer.sampling_ok.sum()) >= hp.online_batch_size

    def test_iqlearn_train_preserves_online_buffer(self):
        """IQ-Learn train() must thread online_buffer through unchanged."""
        buf, _ = make_filled_buffer()
        state, fns, _ = make_iqlearn(buf=buf)
        new_state, _ = fns.train(state, jax.random.key(0))
        # online buffer is all-empty; its leaves should be identical
        for old, new in zip(
            jax.tree.leaves(state.online_buffer),
            jax.tree.leaves(new_state.online_buffer),
        ):
            assert (old == new).all()

    def test_discrete_online_buffer_action_shape(self):
        state, _, _ = make_discrete_iqlearn()
        hp = Hyperparameters(batch_size=BATCH_SIZE)
        assert state.online_buffer.info["actions"].shape == (hp.online_buffer_size, 1)


# ---------------------------------------------------------------------------
# IQLearnFunctions.train_sac — structural
# ---------------------------------------------------------------------------


class TestIQLearnFunctionsHasTrainSAC:
    def test_train_sac_callable(self):
        _, fns, _ = make_iqlearn()
        assert callable(fns.train_sac)

    def test_train_sac_callable_discrete(self):
        _, fns, _ = make_discrete_iqlearn()
        assert callable(fns.train_sac)


# ---------------------------------------------------------------------------
# Mock gymnax environment (for run_env_step / train_sac tests)
# ---------------------------------------------------------------------------
#
# A minimal gymnax-compatible environment that:
#   - reset(key, params) -> (obs, state)
#   - step(key, state, action, params) -> (obs, state, reward, done, info)
#   - get_obs(state, params) -> obs
#
# Observations are drawn from a fixed Normal distribution; reward is 1.0;
# done is always False (no episodic termination) so auto-reset logic stays
# dormant.  A separate variant sets done=True for every step to test the
# auto-reset path.


class _MockEnvState(NamedTuple):
    obs: jax.Array
    step_count: jax.Array


class _MockEnvParams(NamedTuple):
    pass


class _MockEnv:
    """Gymnax-style environment for testing (deterministic, never terminates)."""

    def reset(self, key, params):
        obs = jnp.ones(OBS_DIM) * 0.5
        state = _MockEnvState(obs=obs, step_count=jnp.int32(0))
        return obs, state

    def step(self, key, state, action, params):
        next_obs = jnp.ones(OBS_DIM) * 0.5
        next_state = _MockEnvState(
            obs=next_obs, step_count=state.step_count + 1
        )
        reward = jnp.float32(1.0)
        done = jnp.bool_(False)
        return next_obs, next_state, reward, done, {}

    def get_obs(self, state, params):
        return state.obs


class _MockEnvAlwaysDone:
    """Gymnax-style environment that terminates on every step (tests auto-reset).

    Mirrors the real gymnax base-class behaviour: ``step()`` performs an
    internal auto-reset so that the returned state is already the fresh reset
    state when ``done=True``.  ``reset()`` returns obs=zeros; the post-step
    (terminal) obs is ones, so the caller can distinguish them.
    """

    def reset(self, key, params):
        obs = jnp.zeros(OBS_DIM)
        state = _MockEnvState(obs=obs, step_count=jnp.int32(0))
        return obs, state

    def step(self, key, state, action, params):
        # Post-step (terminal) state — obs=ones so it differs from reset obs
        next_obs = jnp.ones(OBS_DIM)
        next_state = _MockEnvState(obs=next_obs, step_count=state.step_count + 1)
        reward = jnp.float32(0.0)
        done = jnp.bool_(True)
        # Real gymnax base-class step() always auto-resets on done:
        reset_obs, reset_state = self.reset(key, params)
        final_obs = jax.lax.select(done, reset_obs, next_obs)
        final_state = jax.tree.map(
            lambda r, s: jax.lax.select(done, r, s), reset_state, next_state
        )
        return final_obs, final_state, reward, done, {}

    def get_obs(self, state, params):
        return state.obs


# ---------------------------------------------------------------------------
# train_sac — warm-buffer guard
# ---------------------------------------------------------------------------


class TestTrainSACBufferGuard:
    """train_sac must raise ValueError when the online buffer is not warm."""

    def _make_cold_state(self, discrete=False):
        """Create an IQLearnState whose online_buffer has been replaced with an
        empty one, bypassing the factory pre-fill."""
        from lambda_imitation.buffer import create_buffer

        if discrete:
            state, fns, _ = make_discrete_iqlearn(
                hp=Hyperparameters(
                    batch_size=BATCH_SIZE,
                    online_batch_size=BATCH_SIZE,
                    online_buffer_size=32,
                    target_entropy=float(0.98 * __import__("math").log(NUM_ACTIONS)),
                ),
            )
        else:
            state, fns, _ = make_iqlearn(
                hp=Hyperparameters(
                    batch_size=BATCH_SIZE,
                    online_batch_size=BATCH_SIZE,
                    online_buffer_size=32,
                ),
                train_steps=1,
            )
        # Substitute an empty online buffer to simulate missing pre-fill.
        empty_buf, _ = create_buffer(
            shapes={k: v.shape[1:] for k, v in state.online_buffer.info.items()},
            size=32,
            sampling_size=BATCH_SIZE,
            this_step_infos=list(state.online_buffer.info.keys()),
            next_step_infos=["observations"],
        )
        cold_state = state._replace(online_buffer=empty_buf)
        return cold_state, fns

    def test_raises_value_error_when_buffer_empty_continuous(self):
        cold_state, fns = self._make_cold_state(discrete=False)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        with pytest.raises(ValueError, match="sampleable transitions"):
            fns.train_sac(cold_state, env, env_params, env_state0, jax.random.key(0))

    def test_raises_value_error_when_buffer_empty_discrete(self):
        cold_state, fns = self._make_cold_state(discrete=True)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        with pytest.raises(ValueError, match="sampleable transitions"):
            fns.train_sac(cold_state, env, env_params, env_state0, jax.random.key(0))


# ---------------------------------------------------------------------------
# train_sac — warm online buffer
# ---------------------------------------------------------------------------


def _make_warm_sac(
    discrete=False,
    online_buffer_size=64,
    online_batch_size=8,
    train_steps=20,
):
    """Create an IQLearnState with a warm online buffer and run train_sac."""
    env = _MockEnv()
    env_params = _MockEnvParams()
    _, env_state0 = env.reset(jax.random.key(0), env_params)

    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        online_batch_size=online_batch_size,
        online_buffer_size=online_buffer_size,
    )
    if discrete:
        import math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=online_batch_size,
            online_buffer_size=online_buffer_size,
            target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
        )
        state, fns, _ = make_discrete_iqlearn(hp=hp, train_steps=train_steps)
    else:
        state, fns, _ = make_iqlearn(hp=hp, train_steps=train_steps)

    new_state, new_env_state, metrics = fns.train_sac(
        state, env, env_params, env_state0, jax.random.key(7)
    )
    return state, new_state, new_env_state, metrics


class TestTrainSACWarm:
    def test_return_types_continuous(self):
        state, new_state, new_env_state, metrics = _make_warm_sac(discrete=False)
        assert isinstance(new_state, IQLearnState)
        assert isinstance(new_env_state, _MockEnvState)
        assert isinstance(metrics, dict)

    def test_return_types_discrete(self):
        state, new_state, new_env_state, metrics = _make_warm_sac(discrete=True)
        assert isinstance(new_state, IQLearnState)
        assert isinstance(new_env_state, _MockEnvState)
        assert isinstance(metrics, dict)

    def test_expected_metric_keys_continuous(self):
        _, _, _, metrics = _make_warm_sac(discrete=False)
        expected = {"q", "entropy", "v", "critic_loss", "target_q", "alpha"}
        assert expected <= set(metrics.keys()), (
            f"missing keys: {expected - set(metrics.keys())}"
        )

    def test_expected_metric_keys_discrete(self):
        _, _, _, metrics = _make_warm_sac(discrete=True)
        expected = {"q", "entropy", "v", "critic_loss", "target_q", "alpha"}
        assert expected <= set(metrics.keys())

    def test_metrics_finite_continuous(self):
        _, _, _, metrics = _make_warm_sac(discrete=False)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_metrics_finite_discrete(self):
        _, _, _, metrics = _make_warm_sac(discrete=True)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_actor_params_change_continuous(self):
        old, new, _, _ = _make_warm_sac(discrete=False)
        old_leaves = jax.tree.leaves(old.actor)
        new_leaves = jax.tree.leaves(new.actor)
        assert any(not (o == n).all() for o, n in zip(old_leaves, new_leaves)), (
            "actor params should change after warm SAC training"
        )

    def test_actor_params_change_discrete(self):
        old, new, _, _ = _make_warm_sac(discrete=True)
        old_leaves = jax.tree.leaves(old.actor)
        new_leaves = jax.tree.leaves(new.actor)
        assert any(not (o == n).all() for o, n in zip(old_leaves, new_leaves)), (
            "discrete actor params should change after warm SAC training"
        )

    def test_critic_params_change_continuous(self):
        old, new, _, _ = _make_warm_sac(discrete=False)
        old_leaves = jax.tree.leaves(old.critic)
        new_leaves = jax.tree.leaves(new.critic)
        assert any(not (o == n).all() for o, n in zip(old_leaves, new_leaves)), (
            "critic params should change after warm SAC training"
        )

    def test_online_buffer_fills_up(self):
        """After train_steps env steps the online buffer should have data."""
        _, new_state, _, _ = _make_warm_sac(discrete=False, train_steps=20)
        n_ok = int(new_state.online_buffer.sampling_ok.sum())
        assert n_ok >= 18, f"expected ≥18 sampleable slots, got {n_ok}"

    def test_env_state_advances(self):
        """step_count in env_state should increase monotonically."""
        state, fns, _ = make_iqlearn(train_steps=5)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        _, new_env_state, _ = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(1)
        )
        assert int(new_env_state.step_count) == 5

    def test_auto_reset_on_done(self):
        """When env always terminates the env_state obs should be the reset obs."""
        env = _MockEnvAlwaysDone()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        state, fns, _ = make_iqlearn(train_steps=1)
        _, new_env_state, _ = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(0)
        )
        # After auto-reset, obs should equal the reset obs (zeros)
        assert (new_env_state.obs == jnp.zeros(OBS_DIM)).all()

    def test_reproducible_with_same_key(self):
        state, fns, _ = make_iqlearn(train_steps=5)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        r1 = fns.train_sac(state, env, env_params, env_state0, jax.random.key(3))
        r2 = fns.train_sac(state, env, env_params, env_state0, jax.random.key(3))
        for l1, l2 in zip(jax.tree.leaves(r1[0]), jax.tree.leaves(r2[0])):
            assert jnp.allclose(l1, l2)

    def test_no_autotune_alpha_continuous(self):
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=8,
            online_buffer_size=64,
            autotune_alpha=False,
        )
        state, fns, _ = make_iqlearn(hp=hp, train_steps=20, autotune_alpha=False)
        new_state, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(0)
        )
        assert jnp.allclose(state.alpha, new_state.alpha), (
            "alpha should not change when autotune_alpha=False"
        )
        assert "alpha" not in metrics


# ---------------------------------------------------------------------------
# get_importance_ratios — discrete
# ---------------------------------------------------------------------------

class TestGetImportanceRatiosDiscrete:
    """Tests for fns.get_importance_ratios with a discrete action space."""

    @pytest.fixture()
    def setup(self):
        state, fns, _ = make_discrete_iqlearn()
        obs = jax.random.normal(jax.random.key(0), (BATCH_SIZE, OBS_DIM))
        # Integer action indices stored as float32, shape (BATCH_SIZE, 1)
        actions = jax.random.randint(
            jax.random.key(1), (BATCH_SIZE, 1), 0, NUM_ACTIONS
        ).astype(jnp.float32)
        behaviour_probs = jnp.full((BATCH_SIZE,), 1.0 / NUM_ACTIONS)
        return state, fns, obs, actions, behaviour_probs

    def test_output_shape(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert ratios.shape == (BATCH_SIZE,)

    def test_output_dtype(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert ratios.dtype == jnp.float32

    def test_values_positive(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert jnp.all(ratios > 0)

    def test_ratio_is_one_for_policy_probs(self, setup):
        """When behaviour_probs equal π(a|s), every ratio should be ≈ 1."""
        state, fns, obs, actions, behaviour_probs = setup
        # Compute the true policy probabilities for these (obs, action) pairs
        # by calling get_importance_ratios with a uniform unit denominator.
        policy_probs = fns.get_importance_ratios(
            state.actor, obs, actions, jnp.ones(BATCH_SIZE)
        )
        ratios = fns.get_importance_ratios(state.actor, obs, actions, policy_probs)
        assert jnp.allclose(ratios, jnp.ones(BATCH_SIZE), atol=1e-5)

    def test_ratio_doubles_when_behaviour_halved(self, setup):
        """Halving behaviour_probs should exactly double the ratios."""
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        ratios_halved = fns.get_importance_ratios(
            state.actor, obs, actions, behaviour_probs / 2
        )
        assert jnp.allclose(ratios_halved, 2 * ratios, atol=1e-5)


# ---------------------------------------------------------------------------
# get_importance_ratios — continuous
# ---------------------------------------------------------------------------

class TestGetImportanceRatiosContinuous:
    """Tests for fns.get_importance_ratios with a continuous action space."""

    @pytest.fixture()
    def setup(self):
        state, fns, _ = make_iqlearn()
        obs = jax.random.normal(jax.random.key(0), (BATCH_SIZE, OBS_DIM))
        # Use the policy's own unsquashed mean actions (returned by deterministic
        # predict) so that the evaluation points are at the mode of each
        # per-observation Gaussian.  This guarantees a non-negligible density for
        # any network initialisation, avoiding float32 underflow in tests.
        unsquashed_actions = jnp.stack([
            fns.predict(state, obs[i], deterministic=True, return_unsquashed=True)[1]
            for i in range(BATCH_SIZE)
        ])
        behaviour_probs = jnp.ones(BATCH_SIZE)
        return state, fns, obs, unsquashed_actions, behaviour_probs

    def test_output_shape(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert ratios.shape == (BATCH_SIZE,)

    def test_output_dtype(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert ratios.dtype == jnp.float32

    def test_values_positive(self, setup):
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        assert jnp.all(ratios > 0)

    def test_ratio_is_one_for_policy_probs(self, setup):
        """When behaviour_probs equal π(a|s), every ratio should be ≈ 1."""
        state, fns, obs, actions, _ = setup
        # First call with unit denominator to recover the policy densities
        policy_densities = fns.get_importance_ratios(
            state.actor, obs, actions, jnp.ones(BATCH_SIZE)
        )
        # Second call: ratio π / π should be all-ones
        ratios = fns.get_importance_ratios(state.actor, obs, actions, policy_densities)
        assert jnp.allclose(ratios, jnp.ones(BATCH_SIZE), atol=1e-5)

    def test_ratio_doubles_when_behaviour_halved(self, setup):
        """Halving behaviour_probs should exactly double the ratios."""
        state, fns, obs, actions, behaviour_probs = setup
        ratios = fns.get_importance_ratios(state.actor, obs, actions, behaviour_probs)
        ratios_halved = fns.get_importance_ratios(
            state.actor, obs, actions, behaviour_probs / 2
        )
        assert jnp.allclose(ratios_halved, 2 * ratios, atol=1e-5)
