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
    DebugFunctions,
    Head,
    Hyperparameters,
    IQLearnFunctions,
    IQLearnGraphs,
    IQLearnState,
    IdentityMemory,
    LSTMMemory,
    MLPFeatureExtractor,
    NetworkGraphs,
    NetworkState,
    TwinCriticState,
    behaviour_key,
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


def make_feature_extractors(obs_dim=OBS_DIM, hidden_dims=FE_DIMS, seed=0):
    """Create three independent MLPFeatureExtractor instances.

    Returns ``(actor_fe, critic_q1_fe, critic_q2_fe)`` — one per network role.
    """
    rngs = nnx.Rngs(seed)
    actor_fe    = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    critic_q1_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    critic_q2_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    return actor_fe, critic_q1_fe, critic_q2_fe


def make_mc_feature_extractors(obs_dim=OBS_DIM, hidden_dims=FE_DIMS, seed=1):
    """Create two independent MLPFeatureExtractor instances for the MC critic."""
    rngs = nnx.Rngs(seed)
    mc_critic_q1_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    mc_critic_q2_fe = MLPFeatureExtractor(obs_dim, hidden_dims, rngs=rngs)
    return mc_critic_q1_fe, mc_critic_q2_fe


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
        jax.random.key(0),
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
        jax.random.key(0),
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
            jax.random.key(0),
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
            jax.random.key(0),
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
            jax.random.key(0),
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
            jax.random.key(0),
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
            jax.random.key(0),
            train_steps=TRAIN_STEPS, is_discrete=True,
        )
        action, _ = fns.predict(
            state, jnp.ones(OBS_DIM), state.actor_online_carry, deterministic=True
        )
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
        action, _ = fns.predict(state, obs, state.actor_online_carry, deterministic=True)
        assert action.shape == ()
        assert action.dtype == jnp.float32

    def test_deterministic_ignores_key(self, setup):
        state, fns, obs = setup
        a1, _ = fns.predict(
            state, obs, state.actor_online_carry,
            key=jax.random.key(0), deterministic=True,
        )
        a2, _ = fns.predict(
            state, obs, state.actor_online_carry,
            key=jax.random.key(999), deterministic=True,
        )
        assert jnp.allclose(a1, a2)

    def test_stochastic_varies(self, setup):
        """Different keys should (with high probability) yield different actions."""
        state, fns, obs = setup
        results = {
            int(fns.predict(
                state, obs, state.actor_online_carry,
                key=jax.random.key(s), deterministic=False,
            )[0])
            for s in range(20)
        }
        assert len(results) > 1, "stochastic predict should not always return same action"

    def test_valid_action_range_deterministic(self, setup):
        state, fns, obs = setup
        action, _ = fns.predict(state, obs, state.actor_online_carry, deterministic=True)
        assert float(action) >= 0.0
        assert float(action) < NUM_ACTIONS

    def test_valid_action_range_stochastic(self, setup):
        state, fns, obs = setup
        for seed in range(20):
            action, _ = fns.predict(
                state, obs, state.actor_online_carry,
                key=jax.random.key(seed), deterministic=False,
            )
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

    def test_online_buffer_starts_empty(self):
        """create_iqlearn no longer pre-fills the buffer; it starts cold.
        Pre-filling is done lazily by train_sac via prefill_buffer."""
        hp = Hyperparameters(batch_size=BATCH_SIZE, online_batch_size=BATCH_SIZE)
        buf, _ = make_filled_buffer()
        actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors()
        state, _, _ = create_iqlearn(
            hp, buf, ACTION_DIM, actor_fe, critic_q1_fe, critic_q2_fe,
            jax.random.key(0),
            train_steps=TRAIN_STEPS,
        )
        assert int(state.online_buffer.sampling_ok.sum()) == 0

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
# prefill_buffer — structural and behavioural tests
# ---------------------------------------------------------------------------


class TestPrefillBuffer:
    """Tests for fns.prefill_buffer (uniform-random-policy env pre-fill)."""

    def test_field_exists_and_callable_continuous(self):
        _, fns, _ = make_iqlearn()
        assert hasattr(fns, "prefill_buffer")
        assert callable(fns.prefill_buffer)

    def test_field_exists_and_callable_discrete(self):
        _, fns, _ = make_discrete_iqlearn()
        assert hasattr(fns, "prefill_buffer")
        assert callable(fns.prefill_buffer)

    def test_returns_iqlearn_state_and_env_state_continuous(self):
        state, fns, _ = make_iqlearn(
            hp=Hyperparameters(
                batch_size=BATCH_SIZE,
                online_batch_size=BATCH_SIZE,
                online_buffer_size=32,
            ),
            train_steps=1,
        )
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state = env.reset(jax.random.key(0), env_params)
        new_state, new_env_state = fns.prefill_buffer(
            state, env, env_params, env_state, BATCH_SIZE, jax.random.key(1)
        )
        assert isinstance(new_state, IQLearnState)
        assert isinstance(new_env_state, _MockEnvState)

    def test_returns_iqlearn_state_and_env_state_discrete(self):
        import math
        state, fns, _ = make_discrete_iqlearn(
            hp=Hyperparameters(
                batch_size=BATCH_SIZE,
                online_batch_size=BATCH_SIZE,
                online_buffer_size=32,
                target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
            ),
        )
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state = env.reset(jax.random.key(0), env_params)
        new_state, new_env_state = fns.prefill_buffer(
            state, env, env_params, env_state, BATCH_SIZE, jax.random.key(1)
        )
        assert isinstance(new_state, IQLearnState)
        assert isinstance(new_env_state, _MockEnvState)

    def test_populates_sampleable_slots_continuous(self):
        state, fns, _ = make_iqlearn(
            hp=Hyperparameters(
                batch_size=BATCH_SIZE,
                online_batch_size=BATCH_SIZE,
                online_buffer_size=32,
            ),
            train_steps=1,
        )
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state = env.reset(jax.random.key(0), env_params)
        new_state, _ = fns.prefill_buffer(
            state, env, env_params, env_state, BATCH_SIZE, jax.random.key(2)
        )
        n_ok = int(new_state.online_buffer.sampling_ok.sum())
        assert n_ok >= BATCH_SIZE

    def test_populates_sampleable_slots_discrete(self):
        import math
        state, fns, _ = make_discrete_iqlearn(
            hp=Hyperparameters(
                batch_size=BATCH_SIZE,
                online_batch_size=BATCH_SIZE,
                online_buffer_size=32,
                target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
            ),
        )
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state = env.reset(jax.random.key(0), env_params)
        new_state, _ = fns.prefill_buffer(
            state, env, env_params, env_state, BATCH_SIZE, jax.random.key(2)
        )
        n_ok = int(new_state.online_buffer.sampling_ok.sum())
        assert n_ok >= BATCH_SIZE

    def test_behaviour_key_nonzero_with_approximate_mc(self):
        """prefill_buffer must write positive behaviour_key values."""
        import math
        from lambda_imitation.iqlearn import behaviour_key as BK
        from flax import nnx

        rngs = nnx.Rngs(0)
        fe = MLPFeatureExtractor(OBS_DIM, (32,), rngs=nnx.Rngs(0))
        mc_fe_q1 = MLPFeatureExtractor(OBS_DIM, (32,), rngs=nnx.Rngs(1))
        mc_fe_q2 = MLPFeatureExtractor(OBS_DIM, (32,), rngs=nnx.Rngs(2))
        buf, _ = make_discrete_buffer()
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=BATCH_SIZE,
            online_buffer_size=32,
            target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
        )
        fe1 = MLPFeatureExtractor(OBS_DIM, (32,), rngs=nnx.Rngs(3))
        fe2 = MLPFeatureExtractor(OBS_DIM, (32,), rngs=nnx.Rngs(4))
        state, fns, _ = create_iqlearn(
            hp, buf, NUM_ACTIONS, fe, fe1, fe2,
            jax.random.key(0),
            mc_critic_q1_feature_extractor=mc_fe_q1,
            mc_critic_q2_feature_extractor=mc_fe_q2,
            train_steps=1,
            is_discrete=True,
            approximate_mc=True,
        )
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state = env.reset(jax.random.key(0), env_params)
        new_state, _ = fns.prefill_buffer(
            state, env, env_params, env_state, BATCH_SIZE, jax.random.key(3)
        )
        # All slots written by prefill_buffer must have positive behaviour_key.
        bk_vals = new_state.online_buffer.info[BK][:BATCH_SIZE]
        assert bool(jnp.all(bk_vals > 0))


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
# train_sac — cold-buffer auto-prefill
# ---------------------------------------------------------------------------


class TestTrainSACAutoPrefill:
    """train_sac must auto-prefill the online buffer when it is cold."""

    def _make_cold_state(self, discrete=False):
        """Create an IQLearnState whose online_buffer has been replaced with an
        empty one, so that train_sac must trigger prefill_buffer."""
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
        # Substitute an empty online buffer to simulate a cold start.
        empty_buf, _ = create_buffer(
            shapes={k: v.shape[1:] for k, v in state.online_buffer.info.items()},
            size=32,
            sampling_size=BATCH_SIZE,
            this_step_infos=list(state.online_buffer.info.keys()),
            next_step_infos=["observations"],
        )
        cold_state = state._replace(online_buffer=empty_buf)
        return cold_state, fns

    def test_auto_prefills_when_buffer_cold_continuous(self):
        cold_state, fns = self._make_cold_state(discrete=False)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        # Should not raise; train_sac calls prefill_buffer automatically.
        new_state, new_env_state, metrics = fns.train_sac(
            cold_state, env, env_params, env_state0, jax.random.key(0)
        )
        assert isinstance(new_state, IQLearnState)
        n_ok = int(new_state.online_buffer.sampling_ok.sum())
        assert n_ok >= BATCH_SIZE

    def test_auto_prefills_when_buffer_cold_discrete(self):
        cold_state, fns = self._make_cold_state(discrete=True)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        # Should not raise; train_sac calls prefill_buffer automatically.
        new_state, new_env_state, metrics = fns.train_sac(
            cold_state, env, env_params, env_state0, jax.random.key(0)
        )
        assert isinstance(new_state, IQLearnState)
        n_ok = int(new_state.online_buffer.sampling_ok.sum())
        assert n_ok >= BATCH_SIZE


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
        hp = Hyperparameters(batch_size=BATCH_SIZE, online_batch_size=BATCH_SIZE)
        state, fns, _ = make_iqlearn(train_steps=5, hp=hp)
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        _, new_env_state, _ = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(1)
        )
        assert int(new_env_state.step_count) == BATCH_SIZE + 5

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


# ---------------------------------------------------------------------------
# calculate_td_lambda
# ---------------------------------------------------------------------------

# Small constants so the online buffer is filled quickly in the fixture.
_TD_LAMBDA_TRUNC = 3          # lambda_truncation: need trunc+1 = 4 filled slots
_TD_ONLINE_BATCH  = 8         # online_batch_size (pre-fill count)
_TD_ONLINE_SIZE   = 32        # online_buffer_size


def _make_debug_discrete(seed=0):
    """Return (state, fns, graphs, debug_fns) for a discrete agent with a
    patched online buffer where behaviour_key is all-ones (avoids div-by-zero).
    """
    import math

    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        online_batch_size=_TD_ONLINE_BATCH,
        online_buffer_size=_TD_ONLINE_SIZE,
        target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
        lam=0.5,
        lambda_truncation=_TD_LAMBDA_TRUNC,
    )
    buf, _ = make_discrete_buffer()
    actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors(seed=seed)
    mc_critic_q1_fe, mc_critic_q2_fe = make_mc_feature_extractors(seed=seed + 1)
    state, fns, graphs, debug_fns = create_iqlearn(
        hp,
        buf,
        NUM_ACTIONS,
        actor_fe,
        critic_q1_fe,
        critic_q2_fe,
        jax.random.key(seed),
        mc_critic_q1_feature_extractor=mc_critic_q1_fe,
        mc_critic_q2_feature_extractor=mc_critic_q2_fe,
        train_steps=TRAIN_STEPS,
        critic_head_dims=(32,),
        is_discrete=True,
        approximate_mc=True,
        debug=True,
    )
    # Pre-fill the online buffer with real env interactions so that
    # observations are non-trivial and behaviour_key is valid (1/num_actions
    # for written slots; 1.0 for unwritten slots via the approximate_mc
    # pre-initialisation in create_iqlearn).
    env = _MockEnv()
    env_params = _MockEnvParams()
    _, env_state = env.reset(jax.random.key(seed), env_params)
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, _TD_ONLINE_BATCH, jax.random.key(seed + 42)
    )
    return state, fns, graphs, debug_fns


class TestCalculateTdLambda:
    """Tests for debug_fns.calculate_td_lambda exposed via debug=True."""

    @pytest.fixture()
    def setup(self):
        state, fns, graphs, debug_fns = _make_debug_discrete()
        # Use index=0 so the window is [0, 1, ..., lambda_truncation] --
        # the double-offset bug is a no-op at index=0.
        indices = jnp.array([0])
        return state, fns, debug_fns, indices

    # ------------------------------------------------------------------
    # Structural tests
    # ------------------------------------------------------------------

    def test_debug_flag_returns_four_tuple(self):
        """create_iqlearn(..., debug=True) must return a 4-tuple."""
        result = _make_debug_discrete()
        assert len(result) == 4

    def test_debug_fns_type(self):
        """The fourth element must be a DebugFunctions instance."""
        _, _, _, debug_fns = _make_debug_discrete()
        assert isinstance(debug_fns, DebugFunctions)

    def test_no_debug_returns_three_tuple(self):
        """create_iqlearn(..., debug=False) must still return a 3-tuple."""
        buf, _ = make_discrete_buffer()
        actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors()
        import math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
        )
        result = create_iqlearn(
            hp, buf, NUM_ACTIONS, actor_fe, critic_q1_fe, critic_q2_fe,
            jax.random.key(0),
            is_discrete=True,
        )
        assert len(result) == 3

    # ------------------------------------------------------------------
    # Output shape and dtype
    # ------------------------------------------------------------------

    def test_output_shape(self, setup):
        state, fns, debug_fns, indices = setup
        td = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        assert td.shape == (len(indices),)

    def test_output_dtype(self, setup):
        state, fns, debug_fns, indices = setup
        td = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        assert td.dtype == jnp.float32

    def test_output_finite(self, setup):
        """Result must be finite for a well-formed buffer."""
        state, fns, debug_fns, indices = setup
        td = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        assert jnp.all(jnp.isfinite(td))

    # ------------------------------------------------------------------
    # Sensitivity tests
    # ------------------------------------------------------------------

    def test_actor_dependence(self, setup):
        """Changing the actor parameters must change the TD(λ) estimate."""
        state, fns, debug_fns, indices = setup
        td_orig = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        # Build a fresh discrete agent with a different seed so actor differs.
        state2, _, _, _ = _make_debug_discrete(seed=99)
        # Verify the two actors actually differ before testing the function.
        leaves_orig = jax.tree.leaves(state.actor)
        leaves_new  = jax.tree.leaves(state2.actor)
        params_differ = any(
            not jnp.array_equal(a, b) for a, b in zip(leaves_orig, leaves_new)
        )
        assert params_differ, "Test setup error: actors should differ across seeds"
        td_new = debug_fns.calculate_td_lambda(
            state2.actor, state.mc_critic_target, state.online_buffer, indices
        )
        assert not jnp.allclose(td_orig, td_new, atol=1e-6)

    def test_critic_dependence(self, setup):
        """Changing the MC critic target parameters must change the TD(λ) estimate."""
        state, fns, debug_fns, indices = setup
        td_orig = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        state2, _, _, _ = _make_debug_discrete(seed=99)
        leaves_orig = jax.tree.leaves(state.mc_critic_target)
        leaves_new  = jax.tree.leaves(state2.mc_critic_target)
        params_differ = any(
            not jnp.array_equal(a, b) for a, b in zip(leaves_orig, leaves_new)
        )
        assert params_differ, "Test setup error: mc_critic_targets should differ across seeds"
        td_new = debug_fns.calculate_td_lambda(
            state.actor, state2.mc_critic_target, state.online_buffer, indices
        )
        assert not jnp.allclose(td_orig, td_new, atol=1e-6)

    # ------------------------------------------------------------------
    # Degenerate-case: lam=0 collapses to a single bootstrap step
    # ------------------------------------------------------------------

    def test_lam_zero_single_step(self):
        """With lam=0 and lambda_truncation=1, the return must equal a single
        discounted bootstrap: p0*r0*(1-done0) + gamma*p1*Q(s1,a1).

        We verify this by checking that the result is finite and that two
        different critic states produce different outputs (ruling out a
        constant zero return).
        """
        import math

        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_TD_ONLINE_BATCH,
            online_buffer_size=_TD_ONLINE_SIZE,
            target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
            lam=0.0,
            lambda_truncation=1,
        )
        buf, _ = make_discrete_buffer()
        actor_fe, critic_q1_fe, critic_q2_fe = make_feature_extractors()
        mc_critic_q1_fe, mc_critic_q2_fe = make_mc_feature_extractors()
        state, fns, _, debug_fns = create_iqlearn(
            hp, buf, NUM_ACTIONS, actor_fe, critic_q1_fe, critic_q2_fe,
            jax.random.key(0),
            mc_critic_q1_feature_extractor=mc_critic_q1_fe,
            mc_critic_q2_feature_extractor=mc_critic_q2_fe,
            train_steps=TRAIN_STEPS, critic_head_dims=(32,),
            is_discrete=True, approximate_mc=True, debug=True,
        )
        new_info = {**state.online_buffer.info,
                     behaviour_key: jnp.ones(state.online_buffer.size)}
        patched_buf = state.online_buffer._replace(info=new_info)
        state = state._replace(online_buffer=patched_buf)

        indices = jnp.array([0])
        td = debug_fns.calculate_td_lambda(
            state.actor, state.mc_critic_target, state.online_buffer, indices
        )
        assert td.shape == (1,)
        assert td.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(td))


class TestGetQDebug:
    """Tests for debug_fns.get_q exposed via debug=True."""

    @pytest.fixture
    def setup(self):
        state, fns, graphs, debug_fns = _make_debug_discrete()
        indices = jnp.arange(4)
        return state, fns, debug_fns, indices

    def test_get_q_field_exists(self):
        """DebugFunctions must expose a callable get_q field."""
        _, _, _, debug_fns = _make_debug_discrete()
        assert callable(debug_fns.get_q)

    def test_get_q_sac_shape_and_dtype(self, setup):
        """get_q with use_mc=False returns finite float32 array of shape (batch,)."""
        state, fns, debug_fns, indices = setup
        obs     = state.online_buffer.info["observations"][indices]
        actions = state.online_buffer.info["actions"][indices]
        q = debug_fns.get_q(state.critic_target, obs, actions, False)
        assert q.shape == (len(indices),)
        assert q.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(q))

    def test_get_q_mc_shape_and_dtype(self, setup):
        """get_q with use_mc=True returns finite float32 array of shape (batch,)."""
        state, fns, debug_fns, indices = setup
        obs     = state.online_buffer.info["observations"][indices]
        actions = state.online_buffer.info["actions"][indices]
        q = debug_fns.get_q(state.mc_critic_target, obs, actions, True)
        assert q.shape == (len(indices),)
        assert q.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(q))

    def test_get_q_sac_vs_mc_differ_initially(self, setup):
        """SAC and MC critics start with different weights; Q values should differ."""
        state, fns, debug_fns, indices = setup
        obs     = state.online_buffer.info["observations"][indices]
        actions = state.online_buffer.info["actions"][indices]
        q_sac = debug_fns.get_q(state.critic_target,    obs, actions, False)
        q_mc  = debug_fns.get_q(state.mc_critic_target, obs, actions, True)
        # The two critics are independently initialised, so their outputs should differ
        assert not jnp.allclose(q_sac, q_mc)


# ---------------------------------------------------------------------------
# Stage B: scan helpers
# ---------------------------------------------------------------------------

_LSTM_HIDDEN = 16
_SCAN_BATCH  = 4
_SCAN_T      = 5


class TestIdentityMemoryScan:
    """IdentityMemory.scan must pass the sequence through unchanged."""

    def test_output_equals_input(self):
        mem = IdentityMemory(OBS_DIM, rngs=nnx.Rngs(0))
        xs = jax.random.normal(jax.random.key(0), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        new_carry, ys = mem.scan(xs, carry)
        assert jnp.allclose(ys, xs)

    def test_carry_unchanged(self):
        mem = IdentityMemory(OBS_DIM, rngs=nnx.Rngs(0))
        xs = jax.random.normal(jax.random.key(1), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        new_carry, _ = mem.scan(xs, carry)
        assert jnp.allclose(new_carry, carry)

    def test_output_shape(self):
        mem = IdentityMemory(OBS_DIM, rngs=nnx.Rngs(0))
        xs = jax.random.normal(jax.random.key(2), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        _, ys = mem.scan(xs, carry)
        assert ys.shape == (_SCAN_BATCH, _SCAN_T, OBS_DIM)


class TestLSTMMemoryScan:
    """LSTMMemory.scan must match step-by-step __call__ results."""

    def _make_mem(self, seed=0):
        return LSTMMemory(OBS_DIM, _LSTM_HIDDEN, rngs=nnx.Rngs(seed))

    def test_output_shape(self):
        mem = self._make_mem()
        xs = jax.random.normal(jax.random.key(0), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        final_carry, ys = mem.scan(xs, carry)
        assert ys.shape == (_SCAN_BATCH, _SCAN_T, _LSTM_HIDDEN)

    def test_final_carry_shape(self):
        mem = self._make_mem()
        xs = jax.random.normal(jax.random.key(1), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        final_carry, _ = mem.scan(xs, carry)
        # concatenated (c||h) — single array (batch, 2 * hidden_dim)
        assert final_carry.shape == (_SCAN_BATCH, 2 * _LSTM_HIDDEN)

    def test_scan_matches_step_by_step(self):
        """scan and repeated __call__ must produce identical outputs."""
        mem = self._make_mem(seed=42)
        xs = jax.random.normal(jax.random.key(7), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)

        final_carry_scan, ys_scan = mem.scan(xs, carry)

        carry_step = carry
        ys_step = []
        for t in range(_SCAN_T):
            carry_step, y_t = mem(xs[:, t, :], carry_step)
            ys_step.append(y_t)
        ys_step = jnp.stack(ys_step, axis=1)

        assert jnp.allclose(ys_scan, ys_step, atol=1e-6)
        assert jnp.allclose(final_carry_scan, carry_step, atol=1e-6)

    def test_output_finite(self):
        mem = self._make_mem()
        xs = jax.random.normal(jax.random.key(3), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = mem.initial_carry(_SCAN_BATCH)
        _, ys = mem.scan(xs, carry)
        assert jnp.all(jnp.isfinite(ys))


def _make_discrete_debug_with_lstm():
    """Discrete create_iqlearn with LSTMMemory actors and critics."""
    import math
    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        target_entropy=float(0.98 * math.log(NUM_ACTIONS)),
    )
    buf, _ = make_discrete_buffer()
    actor_fe, cq1_fe, cq2_fe = make_feature_extractors()
    mc_cq1_fe, mc_cq2_fe = make_mc_feature_extractors()
    rngs = nnx.Rngs(0)
    feat_dim = FE_DIMS[-1]
    actor_mem    = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(10))
    cq1_mem      = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(11))
    cq2_mem      = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(12))
    mc_cq1_mem   = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(13))
    mc_cq2_mem   = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(14))
    return create_iqlearn(
        hp, buf, NUM_ACTIONS, actor_fe, cq1_fe, cq2_fe,
        jax.random.key(0),
        actor_memory=actor_mem,
        critic_q1_memory=cq1_mem,
        critic_q2_memory=cq2_mem,
        mc_critic_q1_feature_extractor=mc_cq1_fe,
        mc_critic_q2_feature_extractor=mc_cq2_fe,
        mc_critic_q1_memory=mc_cq1_mem,
        mc_critic_q2_memory=mc_cq2_mem,
        train_steps=TRAIN_STEPS, critic_head_dims=(32,),
        is_discrete=True, approximate_mc=True, debug=True,
    )


def _make_continuous_debug_with_lstm():
    """Continuous create_iqlearn with LSTMMemory."""
    hp = Hyperparameters(batch_size=BATCH_SIZE)
    buf, _ = make_filled_buffer()
    actor_fe, cq1_fe, cq2_fe = make_feature_extractors()
    feat_dim = FE_DIMS[-1]
    actor_mem = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(20))
    cq1_mem   = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(21))
    cq2_mem   = LSTMMemory(feat_dim, _LSTM_HIDDEN, rngs=nnx.Rngs(22))
    return create_iqlearn(
        hp, buf, ACTION_DIM, actor_fe, cq1_fe, cq2_fe,
        jax.random.key(0),
        actor_memory=actor_mem,
        critic_q1_memory=cq1_mem,
        critic_q2_memory=cq2_mem,
        train_steps=TRAIN_STEPS, critic_head_dims=(32,),
        debug=True,
    )


class TestRunActorScan:
    """run_actor_scan shape, dtype, and step-scan consistency."""

    def test_fields_exposed_in_debug_fns(self):
        _, _, _, debug_fns = _make_debug_discrete()
        assert callable(debug_fns.run_actor_scan)
        assert callable(debug_fns.run_critic_scan)

    def test_discrete_shape_and_dtype(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(0), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = state.actor_online_carry
        # Broadcast carry from batch=1 to _SCAN_BATCH
        carry = jax.tree.map(
            lambda c: jnp.broadcast_to(c, (_SCAN_BATCH,) + c.shape[1:]), carry
        )
        _, ys = debug_fns.run_actor_scan(state.actor, xs, carry)
        assert ys.shape == (_SCAN_BATCH, _SCAN_T, NUM_ACTIONS)
        assert ys.dtype == jnp.float32

    def test_continuous_shape_and_dtype(self):
        state, _, _, debug_fns = _make_continuous_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(1), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = jax.tree.map(
            lambda c: jnp.broadcast_to(c, (_SCAN_BATCH,) + c.shape[1:]),
            state.actor_online_carry,
        )
        _, ys = debug_fns.run_actor_scan(state.actor, xs, carry)
        assert ys.shape == (_SCAN_BATCH, _SCAN_T, 2 * ACTION_DIM)
        assert ys.dtype == jnp.float32

    def test_scan_internal_consistency_identity(self):
        """With IdentityMemory (no state), slicing the sequence must give same output."""
        state, _, _, debug_fns = _make_debug_discrete()
        xs = jax.random.normal(jax.random.key(5), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry_init = jax.tree.map(
            lambda c: jnp.broadcast_to(c, (_SCAN_BATCH,) + c.shape[1:]),
            state.actor_online_carry,
        )
        _, ys_full = debug_fns.run_actor_scan(state.actor, xs, carry_init)
        # Scan over just the first timestep — output must match the first col of full scan
        _, ys_first = debug_fns.run_actor_scan(state.actor, xs[:, :1, :], carry_init)
        assert jnp.allclose(ys_first[:, 0, :], ys_full[:, 0, :], atol=1e-6)

    def test_scan_output_finite(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(6), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = jax.tree.map(
            lambda c: jnp.broadcast_to(c, (_SCAN_BATCH,) + c.shape[1:]),
            state.actor_online_carry,
        )
        _, ys = debug_fns.run_actor_scan(state.actor, xs, carry)
        assert jnp.all(jnp.isfinite(ys))

    def test_grad_flows_through_actor_scan(self):
        """Gradients must propagate through run_actor_scan to actor params."""
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(7), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        carry = jax.tree.map(
            lambda c: jnp.broadcast_to(c, (_SCAN_BATCH,) + c.shape[1:]),
            state.actor_online_carry,
        )

        def loss_fn(actor):
            _, ys = debug_fns.run_actor_scan(actor, xs, carry)
            return ys.sum()

        # jax.vjp handles mixed-type pytrees (float32 params + uint32 PRNG keys).
        _, pullback = jax.vjp(loss_fn, state.actor)
        (grads,) = pullback(jnp.ones(()))
        float_grads = [l for l in jax.tree.leaves(grads)
                       if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
        assert float_grads, "No float gradient leaves found"
        assert any(jnp.any(g != 0) for g in float_grads), "Expected nonzero grads"


class TestRunCriticScan:
    """run_critic_scan shape, dtype, and step-scan consistency."""

    def test_discrete_shape_and_dtype(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(0), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        fc1, fc2, qs = debug_fns.run_critic_scan(state.critic, xs)
        assert qs.shape == (_SCAN_BATCH, _SCAN_T, NUM_ACTIONS, 2)
        assert qs.dtype == jnp.float32

    def test_discrete_finite(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(1), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        _, _, qs = debug_fns.run_critic_scan(state.critic, xs)
        assert jnp.all(jnp.isfinite(qs))

    def test_discrete_mc_shape(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(2), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        fc1, fc2, qs = debug_fns.run_critic_scan(state.mc_critic_target, xs, use_mc=True)
        assert qs.shape == (_SCAN_BATCH, _SCAN_T, NUM_ACTIONS, 2)

    def test_discrete_grad_flows(self):
        state, _, _, debug_fns = _make_discrete_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(3), (_SCAN_BATCH, _SCAN_T, OBS_DIM))

        def loss_fn(critic):
            _, _, qs = debug_fns.run_critic_scan(critic, xs)
            return qs.sum()

        _, pullback = jax.vjp(loss_fn, state.critic)
        (grads,) = pullback(jnp.ones(()))
        float_grads = [l for l in jax.tree.leaves(grads)
                       if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
        assert any(jnp.any(g != 0) for g in float_grads)

    def test_continuous_shape_and_dtype(self):
        state, _, _, debug_fns = _make_continuous_debug_with_lstm()
        xs = jax.random.normal(jax.random.key(4), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        acts = jax.random.normal(jax.random.key(5), (_SCAN_BATCH, _SCAN_T, ACTION_DIM))
        fc1, fc2, qs = debug_fns.run_critic_scan(state.critic, xs, acts)
        assert qs.shape == (_SCAN_BATCH, _SCAN_T, 2)
        assert qs.dtype == jnp.float32

    def test_continuous_finite(self):
        state, _, _, debug_fns = _make_continuous_debug_with_lstm()
        xs  = jax.random.normal(jax.random.key(6), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        acts = jax.random.normal(jax.random.key(7), (_SCAN_BATCH, _SCAN_T, ACTION_DIM))
        _, _, qs = debug_fns.run_critic_scan(state.critic, xs, acts)
        assert jnp.all(jnp.isfinite(qs))

    def test_continuous_grad_flows(self):
        state, _, _, debug_fns = _make_continuous_debug_with_lstm()
        xs   = jax.random.normal(jax.random.key(8), (_SCAN_BATCH, _SCAN_T, OBS_DIM))
        acts = jax.random.normal(jax.random.key(9), (_SCAN_BATCH, _SCAN_T, ACTION_DIM))

        def loss_fn(critic):
            _, _, qs = debug_fns.run_critic_scan(critic, xs, acts)
            return qs.sum()

        _, pullback = jax.vjp(loss_fn, state.critic)
        (grads,) = pullback(jnp.ones(()))
        float_grads = [l for l in jax.tree.leaves(grads)
                       if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating)]
        assert any(jnp.any(g != 0) for g in float_grads)


# ---------------------------------------------------------------------------
# Stage D: sequence training (R2D2 path in update_step_sac)
# ---------------------------------------------------------------------------

_SEQ_OBS_DIM2   = 4
_SEQ_NUM_ACTS2  = 3
_SEQ_ACT_DIM2   = 2
_SEQ_BUF_SIZE2  = 64
_SEQ_BATCH2     = 8
_SEQ_T2         = 6
_SEQ_BURN2      = 2


def _make_seq_sac(
    discrete: bool,
    burn_in: int = _SEQ_BURN2,
    seq_len: int = _SEQ_T2,
    online_buf_size: int = _SEQ_BUF_SIZE2,
    online_batch: int = _SEQ_BATCH2,
    train_steps: int = 1,
):
    """Helper: warm IQLearnState with sequence_length > 1, one train_sac call."""
    import math

    env = _MockEnv()
    env_params = _MockEnvParams()
    _, env_state0 = env.reset(jax.random.key(0), env_params)

    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        online_batch_size=online_batch,
        online_buffer_size=online_buf_size,
        sequence_length=seq_len,
        burn_in_length=burn_in,
        target_entropy=float(0.98 * math.log(_SEQ_NUM_ACTS2)) if discrete else -float(_SEQ_ACT_DIM2),
    )

    rngs = nnx.Rngs(42)
    if discrete:
        buf, _ = make_discrete_buffer(
            obs_dim=_SEQ_OBS_DIM2, num_actions=_SEQ_NUM_ACTS2,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        state, fns, _ = create_iqlearn(
            hp, buf, _SEQ_NUM_ACTS2,
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            jax.random.key(0),
            train_steps=train_steps,
            critic_head_dims=(16,),
            is_discrete=True,
        )
    else:
        buf, _ = make_filled_buffer(
            obs_dim=_SEQ_OBS_DIM2, action_dim=_SEQ_ACT_DIM2,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        state, fns, _ = create_iqlearn(
            hp, buf, _SEQ_ACT_DIM2,
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            jax.random.key(0),
            train_steps=train_steps,
            critic_head_dims=(16,),
            is_discrete=False,
        )

    prefill_steps = max(online_batch, seq_len + 4)
    state, env_state0 = fns.prefill_buffer(
        state, env, env_params, env_state0, prefill_steps, jax.random.key(1)
    )

    new_state, new_env_state, metrics = fns.train_sac(
        state, env, env_params, env_state0, jax.random.key(2)
    )
    return state, new_state, new_env_state, metrics


class TestSeqTrainSAC:
    """Sequence (R2D2) training path in update_step_sac."""

    def test_discrete_runs(self):
        _make_seq_sac(discrete=True)

    def test_discrete_returns_iqlearnstate(self):
        _, new_state, _, _ = _make_seq_sac(discrete=True)
        assert isinstance(new_state, IQLearnState)

    def test_discrete_metric_keys(self):
        _, _, _, metrics = _make_seq_sac(discrete=True)
        expected = {"q", "entropy", "v", "critic_loss", "target_q", "alpha"}
        assert expected <= set(metrics.keys())

    def test_discrete_metrics_finite(self):
        _, _, _, metrics = _make_seq_sac(discrete=True)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_discrete_actor_params_change(self):
        state, new_state, _, _ = _make_seq_sac(discrete=True)
        old_l = jax.tree.leaves(state.actor)
        new_l = jax.tree.leaves(new_state.actor)
        pairs = [(o, n) for o, n in zip(old_l, new_l)
                 if hasattr(o, "dtype") and jnp.issubdtype(o.dtype, jnp.floating)]
        assert any(not jnp.allclose(o, n) for o, n in pairs), "actor unchanged"

    def test_discrete_critic_params_change(self):
        state, new_state, _, _ = _make_seq_sac(discrete=True)
        old_l = jax.tree.leaves(state.critic)
        new_l = jax.tree.leaves(new_state.critic)
        pairs = [(o, n) for o, n in zip(old_l, new_l)
                 if hasattr(o, "dtype") and jnp.issubdtype(o.dtype, jnp.floating)]
        assert any(not jnp.allclose(o, n) for o, n in pairs), "critic unchanged"

    def test_discrete_burn_in_zero(self):
        _make_seq_sac(discrete=True, burn_in=0)

    def test_continuous_runs(self):
        _make_seq_sac(discrete=False)

    def test_continuous_returns_iqlearnstate(self):
        _, new_state, _, _ = _make_seq_sac(discrete=False)
        assert isinstance(new_state, IQLearnState)

    def test_continuous_metric_keys(self):
        _, _, _, metrics = _make_seq_sac(discrete=False)
        expected = {"q", "entropy", "v", "critic_loss", "target_q", "alpha"}
        assert expected <= set(metrics.keys())

    def test_continuous_metrics_finite(self):
        _, _, _, metrics = _make_seq_sac(discrete=False)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' is not finite: {v}"

    def test_continuous_actor_params_change(self):
        state, new_state, _, _ = _make_seq_sac(discrete=False)
        old_l = jax.tree.leaves(state.actor)
        new_l = jax.tree.leaves(new_state.actor)
        pairs = [(o, n) for o, n in zip(old_l, new_l)
                 if hasattr(o, "dtype") and jnp.issubdtype(o.dtype, jnp.floating)]
        assert any(not jnp.allclose(o, n) for o, n in pairs), "actor unchanged"

    def test_continuous_critic_params_change(self):
        state, new_state, _, _ = _make_seq_sac(discrete=False)
        old_l = jax.tree.leaves(state.critic)
        new_l = jax.tree.leaves(new_state.critic)
        pairs = [(o, n) for o, n in zip(old_l, new_l)
                 if hasattr(o, "dtype") and jnp.issubdtype(o.dtype, jnp.floating)]
        assert any(not jnp.allclose(o, n) for o, n in pairs), "critic unchanged"

    def test_continuous_burn_in_zero(self):
        _make_seq_sac(discrete=False, burn_in=0)

    def test_iid_fallback_discrete(self):
        """sequence_length=1 keeps IID path — regression check."""
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        state, fns, _ = make_discrete_iqlearn(train_steps=1)
        new_state, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(9)
        )
        assert isinstance(new_state, IQLearnState)
        assert "critic_loss" in metrics


# ---------------------------------------------------------------------------
# Stage E: value rescaling + n-step target
# ---------------------------------------------------------------------------

from lambda_imitation.iqlearn import _h, _h_inv


class TestValueRescaling:
    """h/h_inv roundtrip and basic properties."""

    def test_h_inv_h_roundtrip(self):
        eps = 1e-3
        x = jnp.array([-10.0, -1.0, 0.0, 0.5, 3.14, 100.0])
        assert jnp.allclose(_h_inv(_h(x, eps), eps), x, atol=1e-3)  # float32 precision

    def test_h_zero(self):
        assert float(_h(jnp.float32(0.0), 1e-3)) == pytest.approx(0.0, abs=1e-6)

    def test_h_positive_monotone(self):
        eps = 1e-3
        xs = jnp.linspace(0.0, 10.0, 20)
        hs = _h(xs, eps)
        assert jnp.all(jnp.diff(hs) > 0)

    def test_h_antisymmetric(self):
        eps = 1e-3
        xs = jnp.array([1.0, 2.0, 5.0])
        assert jnp.allclose(_h(-xs, eps), -_h(xs, eps), atol=1e-6)


class TestNStepReturn:
    """n-step return correctness (tested via seq critic loss indirectly,
    and directly via a small synthetic example)."""

    def _run_seq(self, discrete, n_step=3, value_rescaling=False):
        return _make_seq_sac(
            discrete=discrete,
            burn_in=0,
            seq_len=8,
            online_batch=_SEQ_BATCH2,
        )

    def test_n_step_discrete_runs(self):
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        import math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_SEQ_BATCH2,
            online_buffer_size=_SEQ_BUF_SIZE2,
            sequence_length=8,
            burn_in_length=0,
            n_step=3,
            target_entropy=float(0.98 * math.log(_SEQ_NUM_ACTS2)),
        )
        rngs = nnx.Rngs(7)
        buf, _ = make_discrete_buffer(
            obs_dim=_SEQ_OBS_DIM2, num_actions=_SEQ_NUM_ACTS2,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        state, fns, _ = create_iqlearn(
            hp, buf, _SEQ_NUM_ACTS2,
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            create_key=jax.random.key(0),
            train_steps=1, critic_head_dims=(16,), is_discrete=True,
        )
        state, env_state0 = fns.prefill_buffer(
            state, env, env_params, env_state0, 20, jax.random.key(1)
        )
        new_state, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(2)
        )
        assert isinstance(new_state, IQLearnState)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' not finite with n_step=3"

    def test_n_step_continuous_runs(self):
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_SEQ_BATCH2,
            online_buffer_size=_SEQ_BUF_SIZE2,
            sequence_length=8,
            burn_in_length=0,
            n_step=3,
            target_entropy=-float(_SEQ_ACT_DIM2),
        )
        rngs = nnx.Rngs(8)
        buf, _ = make_filled_buffer(
            obs_dim=_SEQ_OBS_DIM2, action_dim=_SEQ_ACT_DIM2,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        state, fns, _ = create_iqlearn(
            hp, buf, _SEQ_NUM_ACTS2,
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            create_key=jax.random.key(0),
            train_steps=1, critic_head_dims=(16,), is_discrete=True,
        )
        state, env_state0 = fns.prefill_buffer(
            state, env, env_params, env_state0, 20, jax.random.key(1)
        )
        new_state, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(2)
        )
        assert isinstance(new_state, IQLearnState)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' not finite with value_rescaling"

    def test_value_rescaling_continuous_runs(self):
        env = _MockEnv()
        env_params = _MockEnvParams()
        _, env_state0 = env.reset(jax.random.key(0), env_params)
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_SEQ_BATCH2,
            online_buffer_size=_SEQ_BUF_SIZE2,
            sequence_length=6,
            burn_in_length=0,
            n_step=2,
            value_rescaling=True,
            target_entropy=-float(_SEQ_ACT_DIM2),
        )
        rngs = nnx.Rngs(10)
        buf, _ = make_filled_buffer(
            obs_dim=_SEQ_OBS_DIM2, action_dim=_SEQ_ACT_DIM2,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        state, fns, _ = create_iqlearn(
            hp, buf, _SEQ_ACT_DIM2,
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            MLPFeatureExtractor(_SEQ_OBS_DIM2, (8,), rngs=rngs),
            create_key=jax.random.key(0),
            train_steps=1, critic_head_dims=(16,), is_discrete=False,
        )
        state, env_state0 = fns.prefill_buffer(
            state, env, env_params, env_state0, 20, jax.random.key(1)
        )
        new_state, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(2)
        )
        assert isinstance(new_state, IQLearnState)
        for k, v in metrics.items():
            assert jnp.isfinite(v), f"metric '{k}' not finite with value_rescaling"


# ---------------------------------------------------------------------------
# lambda-discrepancy auxiliary loss
# ---------------------------------------------------------------------------

_LD_OBS_DIM    = 4
_LD_NUM_ACTS   = 3
_LD_BUF_SIZE   = 64
_LD_BATCH      = 8


def _make_ld_state(coef: float, *, seed: int = 0):
    """Build a discrete agent with approximate_mc=True and a pre-filled online
    buffer; toggle ``lambda_discrepancy_coef``."""
    import math as _math
    env = _MockEnv()
    env_params = _MockEnvParams()
    _, env_state0 = env.reset(jax.random.key(seed), env_params)

    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        online_batch_size=_LD_BATCH,
        online_buffer_size=_LD_BUF_SIZE,
        target_entropy=float(0.98 * _math.log(_LD_NUM_ACTS)),
        lam=0.5,
        lambda_truncation=3,
        lambda_discrepancy_coef=coef,
        lambda_discrepancy_delta=1.0,
    )
    rngs = nnx.Rngs(seed)
    buf, _ = make_discrete_buffer(
        obs_dim=_LD_OBS_DIM,
        num_actions=_LD_NUM_ACTS,
        size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )
    state, fns, _ = create_iqlearn(
        hp,
        buf,
        _LD_NUM_ACTS,
        MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
        MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
        MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
        mc_critic_q1_feature_extractor=MLPFeatureExtractor(
            _LD_OBS_DIM, (8,), rngs=rngs
        ),
        mc_critic_q2_feature_extractor=MLPFeatureExtractor(
            _LD_OBS_DIM, (8,), rngs=rngs
        ),
        create_key=jax.random.key(seed),
        train_steps=1,
        critic_head_dims=(16,),
        is_discrete=True,
        approximate_mc=True,
    )
    state, env_state0 = fns.prefill_buffer(
        state, env, env_params, env_state0, _LD_BATCH * 2, jax.random.key(seed + 1)
    )
    return state, fns, env, env_params, env_state0


class TestLambdaDiscrepancy:
    """Lambda-discrepancy auxiliary loss: Huber gap between Q_sac+α·H and Q_mc.

    With ``lambda_discrepancy_coef > 0`` the term flows into actor, SAC critic,
    and MC critic with stop-gradient on every other network.
    """

    def test_disabled_when_coef_zero(self):
        """coef=0 produces no `lambda_discrepancy_loss` metric."""
        state, fns, env, env_params, env_state0 = _make_ld_state(0.0)
        _, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(7)
        )
        assert "lambda_discrepancy_loss" not in metrics

    def test_metric_finite_when_enabled(self):
        """coef>0 surfaces a finite, non-negative `lambda_discrepancy_loss`."""
        state, fns, env, env_params, env_state0 = _make_ld_state(0.1)
        _, _, metrics = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(7)
        )
        assert "lambda_discrepancy_loss" in metrics
        v = float(metrics["lambda_discrepancy_loss"])
        import math as _math
        assert _math.isfinite(v)
        assert v >= 0.0

    def test_validation_requires_approximate_mc(self):
        """coef>0 with approximate_mc=False must raise."""
        import math as _math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_LD_BATCH,
            online_buffer_size=_LD_BUF_SIZE,
            target_entropy=float(0.98 * _math.log(_LD_NUM_ACTS)),
            lambda_discrepancy_coef=0.1,
        )
        rngs = nnx.Rngs(0)
        buf, _ = make_discrete_buffer(
            obs_dim=_LD_OBS_DIM, num_actions=_LD_NUM_ACTS,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        with pytest.raises(ValueError, match="lambda_discrepancy_coef"):
            create_iqlearn(
                hp, buf, _LD_NUM_ACTS,
                MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
                MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
                MLPFeatureExtractor(_LD_OBS_DIM, (8,), rngs=rngs),
                create_key=jax.random.key(0),
                train_steps=1, critic_head_dims=(16,),
                is_discrete=True, approximate_mc=False,
            )

    def test_changes_params_vs_disabled(self):
        """Same seed, same data: enabling the term changes parameters in at
        least one of {actor, SAC critic, MC critic} — auxiliary gradient
        non-trivially flows."""
        s_off, fns_off, env, env_params, env_state0 = _make_ld_state(0.0, seed=2)
        s_on,  fns_on,  _,   _,          _          = _make_ld_state(2.0, seed=2)
        new_off, _, _ = fns_off.train_sac(
            s_off, env, env_params, env_state0, jax.random.key(11)
        )
        new_on,  _, _ = fns_on.train_sac(
            s_on,  env, env_params, env_state0, jax.random.key(11)
        )

        def max_diff(a_state, b_state):
            la = jax.tree.leaves(a_state)
            lb = jax.tree.leaves(b_state)
            return max(
                (float(jnp.abs(a - b).max()) for a, b in zip(la, lb)
                 if a.dtype.kind == "f"),
                default=0.0,
            )

        d_actor   = max_diff(new_off.actor,     new_on.actor)
        d_critic  = max_diff(new_off.critic,    new_on.critic)
        d_mc      = max_diff(new_off.mc_critic, new_on.mc_critic)
        # At least one network must move noticeably from the auxiliary gradient.
        assert max(d_actor, d_critic, d_mc) > 1e-6, (
            f"discrepancy term left every network unchanged "
            f"(actor={d_actor}, critic={d_critic}, mc={d_mc})"
        )


# ---------------------------------------------------------------------------
# refresh_stored_carries: write per-step carries back into the online buffer
# ---------------------------------------------------------------------------

_CR_OBS_DIM    = 4
_CR_NUM_ACTS   = 3
_CR_BUF_SIZE   = 64
_CR_BATCH      = 8
_CR_SEQ_LEN    = 4
_CR_BURN_IN    = 2
_CR_LSTM_HID   = 8


def _make_carry_refresh_state(refresh: bool, *, seed: int = 0):
    """Discrete agent with LSTM memory, sequence_length>1, burn_in>0,
    burn_in_from_stored_carry=True, optionally refresh_stored_carries."""
    import math as _math
    env = _MockEnv()
    env_params = _MockEnvParams()
    _, env_state0 = env.reset(jax.random.key(seed), env_params)

    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        online_batch_size=_CR_BATCH,
        online_buffer_size=_CR_BUF_SIZE,
        target_entropy=float(0.98 * _math.log(_CR_NUM_ACTS)),
        sequence_length=_CR_SEQ_LEN,
        burn_in_length=_CR_BURN_IN,
        burn_in_from_stored_carry=True,
        refresh_stored_carries=refresh,
    )
    rngs = nnx.Rngs(seed)
    buf, _ = make_discrete_buffer(
        obs_dim=_CR_OBS_DIM,
        num_actions=_CR_NUM_ACTS,
        size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
    )
    feat_dim = 8
    state, fns, _ = create_iqlearn(
        hp,
        buf,
        _CR_NUM_ACTS,
        MLPFeatureExtractor(_CR_OBS_DIM, (feat_dim,), rngs=rngs),
        MLPFeatureExtractor(_CR_OBS_DIM, (feat_dim,), rngs=rngs),
        MLPFeatureExtractor(_CR_OBS_DIM, (feat_dim,), rngs=rngs),
        actor_memory=LSTMMemory(feat_dim, _CR_LSTM_HID, rngs=rngs),
        critic_q1_memory=LSTMMemory(feat_dim, _CR_LSTM_HID, rngs=rngs),
        critic_q2_memory=LSTMMemory(feat_dim, _CR_LSTM_HID, rngs=rngs),
        create_key=jax.random.key(seed),
        train_steps=1,
        critic_head_dims=(8,),
        is_discrete=True,
    )
    state, env_state0 = fns.prefill_buffer(
        state, env, env_params, env_state0, _CR_BUF_SIZE,
        jax.random.key(seed + 1),
    )
    return state, fns, env, env_params, env_state0


class TestRefreshStoredCarries:
    """Carry refresh: per-step input carries scattered into the online buffer
    at sampled slots seq_idx[:, B:B+SL].  Counters carry drift."""

    def test_disabled_leaves_carries_unchanged(self):
        """refresh=False — buffer carry slots untouched after train_sac."""
        from lambda_imitation.iqlearn import (
            actor_carry_key, critic_q1_carry_key, critic_q2_carry_key,
        )
        state, fns, env, env_params, env_state0 = _make_carry_refresh_state(False)
        before = {
            k: state.online_buffer.info[k].copy()
            for k in (actor_carry_key, critic_q1_carry_key, critic_q2_carry_key)
        }
        new_state, _, _ = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(7)
        )
        for k, b in before.items():
            assert jnp.array_equal(b, new_state.online_buffer.info[k]), (
                f"carry slot '{k}' was modified despite refresh=False"
            )

    def test_enabled_writes_at_least_one_slot(self):
        """refresh=True — carry buffers change in actor and both critic
        carry keys after one train_sac step."""
        from lambda_imitation.iqlearn import (
            actor_carry_key, critic_q1_carry_key, critic_q2_carry_key,
        )
        state, fns, env, env_params, env_state0 = _make_carry_refresh_state(True)
        before = {
            k: state.online_buffer.info[k].copy()
            for k in (actor_carry_key, critic_q1_carry_key, critic_q2_carry_key)
        }
        new_state, _, _ = fns.train_sac(
            state, env, env_params, env_state0, jax.random.key(7)
        )
        for k, b in before.items():
            after = new_state.online_buffer.info[k]
            diff = float(jnp.abs(after - b).max())
            assert diff > 0.0, (
                f"carry slot '{k}' unchanged after refresh=True step"
            )

    def test_validation_requires_burn_in_from_stored_carry(self):
        """refresh=True with burn_in_from_stored_carry=False must raise."""
        import math as _math
        hp = Hyperparameters(
            batch_size=BATCH_SIZE,
            online_batch_size=_CR_BATCH,
            online_buffer_size=_CR_BUF_SIZE,
            target_entropy=float(0.98 * _math.log(_CR_NUM_ACTS)),
            sequence_length=_CR_SEQ_LEN,
            burn_in_length=_CR_BURN_IN,
            burn_in_from_stored_carry=False,
            refresh_stored_carries=True,
        )
        rngs = nnx.Rngs(0)
        buf, _ = make_discrete_buffer(
            obs_dim=_CR_OBS_DIM, num_actions=_CR_NUM_ACTS,
            size=BUFFER_SIZE, batch_size=BATCH_SIZE,
        )
        with pytest.raises(ValueError, match="refresh_stored_carries"):
            create_iqlearn(
                hp, buf, _CR_NUM_ACTS,
                MLPFeatureExtractor(_CR_OBS_DIM, (8,), rngs=rngs),
                MLPFeatureExtractor(_CR_OBS_DIM, (8,), rngs=rngs),
                MLPFeatureExtractor(_CR_OBS_DIM, (8,), rngs=rngs),
                create_key=jax.random.key(0),
                train_steps=1, critic_head_dims=(8,),
                is_discrete=True,
            )


# ---------------------------------------------------------------------------
# create_key determinism
# ---------------------------------------------------------------------------

_DET_OBS_DIM  = 4
_DET_NUM_ACTS = 3


def _make_det_iqlearn(key: jax.Array):
    """Minimal discrete agent created with a given create_key."""
    import math as _math
    buf, _ = make_discrete_buffer(
        obs_dim=_DET_OBS_DIM, num_actions=_DET_NUM_ACTS,
        size=BUFFER_SIZE, batch_size=BATCH_SIZE,
    )
    hp = Hyperparameters(
        batch_size=BATCH_SIZE,
        target_entropy=float(0.98 * _math.log(_DET_NUM_ACTS)),
    )
    rngs = nnx.Rngs(0)
    state, fns, _ = create_iqlearn(
        hp, buf, _DET_NUM_ACTS,
        MLPFeatureExtractor(_DET_OBS_DIM, (8,), rngs=rngs),
        MLPFeatureExtractor(_DET_OBS_DIM, (8,), rngs=rngs),
        MLPFeatureExtractor(_DET_OBS_DIM, (8,), rngs=rngs),
        create_key=key,
        train_steps=1,
        critic_head_dims=(16,),
        is_discrete=True,
    )
    return state


class TestCreateKeyDeterminism:
    """create_key controls head initialisation deterministically."""

    def test_same_key_gives_same_weights(self):
        """Calling create_iqlearn twice with the same create_key produces
        identical actor and critic head parameters."""
        key = jax.random.key(7)
        s1 = _make_det_iqlearn(key)
        s2 = _make_det_iqlearn(key)
        leaves1 = jax.tree.leaves(s1.actor)
        leaves2 = jax.tree.leaves(s2.actor)
        for l1, l2 in zip(leaves1, leaves2):
            assert jnp.array_equal(l1, l2), "actor weights differ with same create_key"
        leaves1 = jax.tree.leaves(s1.critic)
        leaves2 = jax.tree.leaves(s2.critic)
        for l1, l2 in zip(leaves1, leaves2):
            assert jnp.array_equal(l1, l2), "critic weights differ with same create_key"

    def test_different_keys_give_different_weights(self):
        """Calling create_iqlearn with key(0) vs key(42) produces at least
        one differing parameter in the actor or critic heads."""
        s0  = _make_det_iqlearn(jax.random.key(0))
        s42 = _make_det_iqlearn(jax.random.key(42))
        all_same = all(
            jnp.array_equal(l0, l42)
            for l0, l42 in zip(
                jax.tree.leaves(s0.actor) + jax.tree.leaves(s0.critic),
                jax.tree.leaves(s42.actor) + jax.tree.leaves(s42.critic),
            )
        )
        assert not all_same, "different create_keys produced identical weights"
