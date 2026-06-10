"""Tests for the prev-action input to the recurrent feature extractor.

The prev-action encoding lives in the trailing ``prev_action_dim`` entries of
the flat carry ("packed carry").  These tests cover:

* equivalence with the original lambda-discrepancy ``ActionConcatWrapper``
  formulation (action one-hot concatenated to the observation before the
  embedding) — the faithfulness proof for the paper,
* carry layout / shape arithmetic per memory type,
* tail pass-through and ``write_prev_action``,
* ``predict`` writing the chosen action into the tail,
* ``calculate_latent`` consuming the shifted, done-masked action sequence,
* construction-time validation in ``create_iqlearn``,
* an end-to-end smoke ``train`` round with ``use_prev_action=True``.
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import (
    RecurrentFeatureExtractor,
    create_iqlearn_from_env,
    env_spec_from_gymnax,
)


OBS_DIM = 5
ACTION_DIM = 3
HIDDEN = 8
MEMORY_TYPES = ("identity", "rnn", "gru", "lstm")


def _make_fe(memory_type, prev_action_dim, projection_dim=16, seed=0):
    return RecurrentFeatureExtractor(
        input_dim=OBS_DIM,
        projection_dim=projection_dim,
        memory_type=memory_type,
        memory_hidden_dim=HIDDEN,
        prev_action_dim=prev_action_dim,
        rngs=nnx.Rngs(jax.random.key(seed)),
    )


def _memory_dim(memory_type):
    if memory_type == "identity":
        return 0
    if memory_type == "lstm":
        return 2 * HIDDEN
    return HIDDEN


# ---------------------------------------------------------------------------
# Carry layout / shapes
# ---------------------------------------------------------------------------


class TestCarryLayout:
    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_carry_dim_includes_tail(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        assert fe.carry_dim == _memory_dim(memory_type) + ACTION_DIM

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_carry_dim_unchanged_when_disabled(self, memory_type):
        fe = _make_fe(memory_type, 0)
        assert fe.carry_dim == _memory_dim(memory_type)

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_initialize_carry_shape(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        carry = fe.initialize_carry(4)
        assert carry.shape == (4, fe.carry_dim)
        assert (carry == 0).all()

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_output_dim_unchanged(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        ref = _make_fe(memory_type, 0)
        assert fe.output_dim == ref.output_dim
        _, y = fe(fe.initialize_carry(2), jnp.ones((2, OBS_DIM)))
        assert y.shape == (2, fe.output_dim)

    def test_negative_prev_action_dim_raises(self):
        with pytest.raises(ValueError, match="prev_action_dim"):
            _make_fe("gru", -1)


# ---------------------------------------------------------------------------
# Tail semantics
# ---------------------------------------------------------------------------


class TestTailSemantics:
    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_tail_passes_through_call(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        carry = fe.initialize_carry(2)
        tail = jnp.tile(jnp.array([1.0, 0.0, 0.5]), (2, 1))
        carry = fe.write_prev_action(carry, tail)
        new_carry, _ = fe(carry, jnp.ones((2, OBS_DIM)))
        assert jnp.array_equal(new_carry[..., -ACTION_DIM:], tail)

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_tail_changes_output(self, memory_type):
        # The action must actually reach the network: two carries differing
        # only in the tail must produce different features.
        fe = _make_fe(memory_type, ACTION_DIM)
        obs = jnp.ones((1, OBS_DIM))
        zero = fe.initialize_carry(1)
        with_action = fe.write_prev_action(zero, jax.nn.one_hot(1, ACTION_DIM))
        _, y0 = fe(zero, obs)
        _, y1 = fe(with_action, obs)
        assert not jnp.allclose(y0, y1)

    def test_write_prev_action_roundtrip(self):
        fe = _make_fe("gru", ACTION_DIM)
        carry = jax.random.normal(jax.random.key(1), (2, fe.carry_dim))
        tail = jax.nn.one_hot(jnp.array([0, 2]), ACTION_DIM)
        written = fe.write_prev_action(carry, tail)
        assert jnp.array_equal(written[..., -ACTION_DIM:], tail)
        assert jnp.array_equal(written[..., :-ACTION_DIM], carry[..., :-ACTION_DIM])

    def test_write_prev_action_noop_when_disabled(self):
        fe = _make_fe("gru", 0)
        carry = jax.random.normal(jax.random.key(1), (2, fe.carry_dim))
        assert fe.write_prev_action(carry, jnp.zeros((2, 0))) is carry

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_disabled_fe_matches_legacy_construction(self, memory_type):
        # prev_action_dim=0 must be bit-identical to an FE built without the
        # argument (same seed => same params => same outputs).
        fe = _make_fe(memory_type, 0, seed=7)
        legacy = RecurrentFeatureExtractor(
            input_dim=OBS_DIM,
            projection_dim=16,
            memory_type=memory_type,
            memory_hidden_dim=HIDDEN,
            rngs=nnx.Rngs(jax.random.key(7)),
        )
        obs = jax.random.normal(jax.random.key(2), (3, OBS_DIM))
        carry = fe.initialize_carry(3)
        new_carry, y = fe(carry, obs)
        legacy_carry, legacy_y = legacy(carry, obs)
        assert jnp.array_equal(y, legacy_y)
        assert jnp.array_equal(new_carry, legacy_carry)


# ---------------------------------------------------------------------------
# Equivalence with the original ActionConcatWrapper formulation
# ---------------------------------------------------------------------------


class TestWrapperEquivalence:
    """Packed carry == action one-hot concatenated to the observation.

    The original lambda-discrepancy code appends the prev-action encoding to
    the observation via an env wrapper; the network then sees
    ``[obs | enc(a_{t-1})]`` (zeros at reset).  Build a reference FE with
    ``input_dim = OBS_DIM + ACTION_DIM`` and ``prev_action_dim=0``, copy the
    packed FE's weights into it, and check both produce bit-identical outputs
    over a random sequence with episode resets.
    """

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_equivalent_to_obs_concat(self, memory_type):
        packed = _make_fe(memory_type, ACTION_DIM, seed=3)
        reference = RecurrentFeatureExtractor(
            input_dim=OBS_DIM + ACTION_DIM,
            projection_dim=16,
            memory_type=memory_type,
            memory_hidden_dim=HIDDEN,
            rngs=nnx.Rngs(jax.random.key(99)),
        )
        # Same projection input width (OBS_DIM + ACTION_DIM), so the states
        # are structurally identical and weights can be copied verbatim.
        graph, packed_state = nnx.split(packed)
        ref_graph, _ = nnx.split(reference)
        reference = nnx.merge(ref_graph, packed_state)

        T, B = 12, 4
        key = jax.random.key(4)
        k_obs, k_act, k_done = jax.random.split(key, 3)
        obs_seq = jax.random.normal(k_obs, (T, B, OBS_DIM))
        act_seq = jax.random.randint(k_act, (T, B), 0, ACTION_DIM)
        done_seq = jax.random.bernoulli(k_done, 0.2, (T, B))

        packed_carry = packed.initialize_carry(B)
        ref_carry = reference.initialize_carry(B)
        prev_enc = jnp.zeros((B, ACTION_DIM))  # episode start: zero vector

        for t in range(T):
            new_packed_carry, y_packed = packed(packed_carry, obs_seq[t])
            ref_obs = jnp.concatenate([obs_seq[t], prev_enc], axis=-1)
            ref_carry, y_ref = reference(ref_carry, ref_obs)
            assert jnp.array_equal(y_packed, y_ref), f"mismatch at t={t}"

            enc = jax.nn.one_hot(act_seq[t], ACTION_DIM)
            packed_carry = packed.write_prev_action(new_packed_carry, enc)
            reset = done_seq[t][:, None]
            packed_carry = jnp.where(reset, 0.0, packed_carry)
            ref_carry = jnp.where(reset, 0.0, ref_carry)
            prev_enc = jnp.where(reset, 0.0, enc)


# ---------------------------------------------------------------------------
# Agent-level tests (discrete, CartPole)
# ---------------------------------------------------------------------------


def _tiny_hp():
    return Hyperparameters(
        target_entropy=0.2,
        batch_size=4,
        online_batch_size=4,
        online_buffer_size=256,
        burn_in_length=2,
        sequence_length=4,
        lambda_truncation=2,
    )


@pytest.fixture(scope="module")
def cartpole_agent():
    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns, debug_fns = create_iqlearn_from_env(
        spec,
        expert_data,
        buffer_size=4,
        hp=_tiny_hp(),
        projection_dim=16,
        memory_type="gru",
        memory_hidden_dim=HIDDEN,
        use_prev_action=True,
        critic_dims=(16,),
        train_steps=4,
        approximate_lambda=True,
        debug=True,
        seed=0,
    )
    return env, env_params, spec, state, fns, debug_fns


class TestPredictTail:
    def test_discrete_tail_is_one_hot_of_action(self, cartpole_agent):
        env, env_params, spec, state, fns, _ = cartpole_agent
        obs = jnp.zeros(spec.obs_shape, dtype=jnp.float32)
        carry = jnp.zeros((HIDDEN + spec.action_dim,), dtype=jnp.float32)
        for deterministic in (True, False):
            action, new_carry = fns.predict(
                state, obs, carry, jax.random.key(1), deterministic=deterministic
            )
            expected = jax.nn.one_hot(jnp.round(action).astype(jnp.int32),
                                      spec.action_dim)
            assert jnp.array_equal(new_carry[-spec.action_dim:], expected)


class TestCalculateLatent:
    def test_latent_matches_manual_unroll(self, cartpole_agent):
        env, env_params, spec, state, fns, debug_fns = cartpole_agent
        B, T, BL = 2, 6, 2
        key = jax.random.key(5)
        k_obs, k_act = jax.random.split(key)
        observations = jax.random.normal(k_obs, (B, T, *spec.obs_shape))
        actions = jax.random.randint(k_act, (B, T, 1), 0, spec.action_dim).astype(
            jnp.float32
        )
        dones = jnp.zeros((B, T))
        dones = dones.at[:, 3].set(1.0)  # episode boundary inside the window

        carry_dim = HIDDEN + spec.action_dim
        init_carries = jnp.zeros((B, carry_dim))
        latent, _ = debug_fns.calculate_latent(
            state.feature_extractor,
            state.feature_extractor_target,
            observations,
            actions,
            dones,
            init_carries,
        )
        assert latent.shape[0] == T - BL

        # Manual reference: prev_a[t] = enc(a[t-1]) * (1 - done[t-1]), zero at
        # t=0; carry zeroed after done — exactly the online-path semantics.
        fe_graph, _ = nnx.split(
            RecurrentFeatureExtractor(
                input_dim=spec.obs_shape[0],
                projection_dim=16,
                memory_type="gru",
                memory_hidden_dim=HIDDEN,
                prev_action_dim=spec.action_dim,
                rngs=nnx.Rngs(jax.random.key(0)),
            )
        )
        fe = nnx.merge(fe_graph, state.feature_extractor)
        carry = jnp.zeros((B, carry_dim))
        manual = []
        for t in range(T):
            new_carry, y = fe(carry, observations[:, t])
            manual.append(y)
            enc = jax.nn.one_hot(
                jnp.round(actions[:, t, 0]).astype(jnp.int32), spec.action_dim
            )
            new_carry = fe.write_prev_action(new_carry, enc)
            carry = jnp.where(dones[:, t][:, None] > 0, 0.0, new_carry)
        manual = jnp.stack(manual[BL:])
        assert jnp.allclose(latent, manual, atol=1e-6)


class TestValidation:
    def test_mismatched_fe_raises(self):
        env, env_params = gymnax.make("CartPole-v1")
        spec = env_spec_from_gymnax(env, env_params)
        expert_data = {
            "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
            "actions": jnp.zeros((4, 1), dtype=jnp.float32),
        }
        from lambda_imitation.iqlearn import create_iqlearn
        from lambda_imitation.buffer import create_buffer

        buffer, buf_fns = create_buffer(
            shapes={"observations": spec.obs_shape, "actions": (1,)},
            size=4,
            sampling_size=2,
            this_step_infos=["observations", "actions"],
            next_step_infos=["observations"],
        )
        fe = RecurrentFeatureExtractor(
            input_dim=spec.obs_shape[0],
            projection_dim=16,
            memory_type="gru",
            memory_hidden_dim=HIDDEN,
            prev_action_dim=0,
            rngs=nnx.Rngs(jax.random.key(0)),
        )
        with pytest.raises(ValueError, match="use_prev_action"):
            create_iqlearn(
                params=_tiny_hp(),
                buffer=buffer,
                action_dim=spec.action_dim,
                feature_extractor=fe,
                key=jax.random.key(1),
                is_discrete=True,
                use_prev_action=True,
            )

    def test_enabled_fe_without_flag_raises(self):
        env, env_params = gymnax.make("CartPole-v1")
        spec = env_spec_from_gymnax(env, env_params)
        from lambda_imitation.iqlearn import create_iqlearn
        from lambda_imitation.buffer import create_buffer

        buffer, _ = create_buffer(
            shapes={"observations": spec.obs_shape, "actions": (1,)},
            size=4,
            sampling_size=2,
            this_step_infos=["observations", "actions"],
            next_step_infos=["observations"],
        )
        fe = RecurrentFeatureExtractor(
            input_dim=spec.obs_shape[0],
            projection_dim=16,
            memory_type="gru",
            memory_hidden_dim=HIDDEN,
            prev_action_dim=spec.action_dim,
            rngs=nnx.Rngs(jax.random.key(0)),
        )
        with pytest.raises(ValueError, match="prev_action_dim"):
            create_iqlearn(
                params=_tiny_hp(),
                buffer=buffer,
                action_dim=spec.action_dim,
                feature_extractor=fe,
                key=jax.random.key(1),
                is_discrete=True,
                use_prev_action=False,
            )


class TestEndToEnd:
    def test_train_round_finite_metrics(self, cartpole_agent):
        env, env_params, spec, state, fns, _ = cartpole_agent
        key = jax.random.key(2)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        new_state, _, metrics = fns.train(state, env, env_params, env_state, key)
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}: {value}"
