"""Tests for the prev-action input to the recurrent feature extractor.

The prev-action encoding is fed to the projection as an **explicit input**
(``feature_extractor(carry, obs, prev_action)``) and threaded *next to* the
memory carry — it is not packed into the carry.  These tests cover:

* equivalence with the original lambda-discrepancy ``ActionConcatWrapper``
  formulation (action one-hot concatenated to the observation before the
  embedding) — the faithfulness proof for the paper,
* carry layout / shape arithmetic per memory type (carry is memory-only),
* the default ``LinearProjection`` and a custom builder projection module,
* ``predict`` returning a memory-only carry + the exposed ``encode_action``,
* ``calculate_latent`` consuming the shifted, done-masked prev-action input,
* construction-time validation in ``create_iqlearn``,
* end-to-end smoke ``train`` rounds with ``use_prev_action=True`` and with a
  custom skip-connection projection.
"""

import gymnax
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from lambda_imitation.iqlearn import Hyperparameters
from lambda_imitation.utils import (
    LinearProjection,
    RecurrentFeatureExtractor,
    create_iqlearn_from_env,
    env_spec_from_gymnax,
)


OBS_DIM = 5
ACTION_DIM = 3
HIDDEN = 8
MEMORY_TYPES = ("identity", "rnn", "gru", "lstm")


def _make_fe(memory_type, prev_action_dim, projection=16, seed=0):
    return RecurrentFeatureExtractor(
        input_shape=OBS_DIM,
        projection=projection,
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
    def test_carry_dim_is_memory_only(self, memory_type):
        # The prev-action is NOT part of the carry: enabling it must not grow
        # the carry.
        fe = _make_fe(memory_type, ACTION_DIM)
        assert fe.carry_dim == _memory_dim(memory_type)

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
    def test_initialize_prev_action_shape(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        pa = fe.initialize_prev_action(4)
        assert pa.shape == (4, ACTION_DIM)
        assert (pa == 0).all()
        # disabled -> width 0
        assert _make_fe(memory_type, 0).initialize_prev_action(4).shape == (4, 0)

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_output_dim(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM)
        carry = fe.initialize_carry(2)
        pa = fe.initialize_prev_action(2)
        _, y = fe(carry, jnp.ones((2, OBS_DIM)), pa)
        assert y.shape == (2, fe.output_dim)

    def test_negative_prev_action_dim_raises(self):
        with pytest.raises(ValueError, match="prev_action_dim"):
            _make_fe("gru", -1)


# ---------------------------------------------------------------------------
# Projection module semantics
# ---------------------------------------------------------------------------


class TestProjection:
    def test_default_linear_matches_manual_concat(self):
        # The default int projection is a Linear over concat([flat_obs, pa]).
        proj = LinearProjection(OBS_DIM, ACTION_DIM, 16, rngs=nnx.Rngs(0))
        obs = jax.random.normal(jax.random.key(1), (4, OBS_DIM))
        pa = jax.nn.one_hot(jnp.array([0, 1, 2, 0]), ACTION_DIM)
        manual = proj.linear(jnp.concatenate([obs, pa], axis=-1))
        assert jnp.array_equal(proj(obs, pa), manual)
        # prev_action=None == zero-width tail == obs only.
        proj0 = LinearProjection(OBS_DIM, 0, 16, rngs=nnx.Rngs(0))
        assert jnp.array_equal(proj0(obs, None), proj0.linear(obs))

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_prev_action_changes_output(self, memory_type):
        # The prev-action must actually reach the network: two calls differing
        # only in prev_action must produce different features.
        fe = _make_fe(memory_type, ACTION_DIM)
        obs = jnp.ones((1, OBS_DIM))
        carry = fe.initialize_carry(1)
        _, y0 = fe(carry, obs, jnp.zeros((1, ACTION_DIM)))
        _, y1 = fe(carry, obs, jax.nn.one_hot(jnp.array([1]), ACTION_DIM))
        assert not jnp.allclose(y0, y1)

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_none_prev_action_normalised(self, memory_type):
        # Passing None must equal passing an explicit zero prev-action.
        fe = _make_fe(memory_type, ACTION_DIM)
        obs = jax.random.normal(jax.random.key(3), (2, OBS_DIM))
        carry = fe.initialize_carry(2)
        c_none, y_none = fe(carry, obs, None)
        c_zero, y_zero = fe(carry, obs, jnp.zeros((2, ACTION_DIM)))
        assert jnp.array_equal(y_none, y_zero)
        assert jnp.array_equal(c_none, c_zero)

    def test_custom_builder_projection(self):
        # A builder callable receives (obs_shape, prev_action_dim, rngs).
        class Skip(nnx.Module):
            def __init__(self, obs_shape, pa_dim, rngs):
                flat = int(jnp.prod(jnp.array(obs_shape)))
                self.w = flat + pa_dim
                self.l1 = nnx.Linear(self.w, 12, rngs=rngs)
                self.l2 = nnx.Linear(12, self.w, rngs=rngs)

            def __call__(self, obs, prev_action):
                x = obs.reshape(obs.shape[0], -1)
                if prev_action is not None and prev_action.shape[-1]:
                    x = jnp.concatenate([x, prev_action], axis=-1)
                return x + self.l2(jax.nn.relu(self.l1(x)))  # residual skip

        fe = RecurrentFeatureExtractor(
            input_shape=(OBS_DIM,),
            projection=lambda s, p, r: Skip(s, p, r),
            memory_type="gru",
            memory_hidden_dim=HIDDEN,
            prev_action_dim=ACTION_DIM,
            rngs=nnx.Rngs(0),
        )
        carry = fe.initialize_carry(2)
        pa = fe.initialize_prev_action(2)
        new_carry, y = fe(carry, jnp.ones((2, OBS_DIM)), pa)
        assert new_carry.shape == (2, fe.carry_dim)
        assert y.shape == (2, fe.output_dim)


# ---------------------------------------------------------------------------
# Equivalence with the original ActionConcatWrapper formulation
# ---------------------------------------------------------------------------


class TestWrapperEquivalence:
    """Explicit prev-action input == action one-hot concatenated to the obs.

    The original lambda-discrepancy code appends the prev-action encoding to
    the observation via an env wrapper; the network then sees
    ``[obs | enc(a_{t-1})]`` (zeros at reset).  Build a reference FE with
    ``input_shape = OBS_DIM + ACTION_DIM`` and ``prev_action_dim=0``, copy the
    prev-action FE's weights into it, and check both produce bit-identical
    outputs over a random sequence with episode resets.
    """

    @pytest.mark.parametrize("memory_type", MEMORY_TYPES)
    def test_equivalent_to_obs_concat(self, memory_type):
        fe = _make_fe(memory_type, ACTION_DIM, seed=3)
        reference = RecurrentFeatureExtractor(
            input_shape=OBS_DIM + ACTION_DIM,
            projection=16,
            memory_type=memory_type,
            memory_hidden_dim=HIDDEN,
            rngs=nnx.Rngs(jax.random.key(99)),
        )
        # Same projection input width (OBS_DIM + ACTION_DIM), so the states are
        # structurally identical and weights can be copied verbatim.
        graph, fe_state = nnx.split(fe)
        ref_graph, _ = nnx.split(reference)
        reference = nnx.merge(ref_graph, fe_state)

        T, B = 12, 4
        key = jax.random.key(4)
        k_obs, k_act, k_done = jax.random.split(key, 3)
        obs_seq = jax.random.normal(k_obs, (T, B, OBS_DIM))
        act_seq = jax.random.randint(k_act, (T, B), 0, ACTION_DIM)
        done_seq = jax.random.bernoulli(k_done, 0.2, (T, B))

        carry = fe.initialize_carry(B)
        ref_carry = reference.initialize_carry(B)
        prev_enc = jnp.zeros((B, ACTION_DIM))  # episode start: zero vector

        for t in range(T):
            new_carry, y = fe(carry, obs_seq[t], prev_enc)
            ref_obs = jnp.concatenate([obs_seq[t], prev_enc], axis=-1)
            ref_carry, y_ref = reference(ref_carry, ref_obs)
            assert jnp.array_equal(y, y_ref), f"mismatch at t={t}"

            enc = jax.nn.one_hot(act_seq[t], ACTION_DIM)
            reset = done_seq[t][:, None]
            carry = jnp.where(reset, 0.0, new_carry)
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
        projection=16,
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


class TestPredict:
    def test_returns_memory_only_carry(self, cartpole_agent):
        env, env_params, spec, state, fns, _ = cartpole_agent
        obs = jnp.zeros(spec.obs_shape, dtype=jnp.float32)
        carry = jnp.zeros((HIDDEN,), dtype=jnp.float32)  # memory only
        prev_action = jnp.zeros((spec.action_dim,), dtype=jnp.float32)
        for deterministic in (True, False):
            action, new_carry = fns.predict(
                state,
                obs,
                carry,
                jax.random.key(1),
                deterministic=deterministic,
                prev_action=prev_action,
            )
            assert new_carry.shape == (HIDDEN,)  # no action packed in

    def test_encode_action_exposed(self, cartpole_agent):
        _, _, spec, _, fns, _ = cartpole_agent
        enc = fns.encode_action(jnp.array([2.0]))
        assert jnp.array_equal(enc, jax.nn.one_hot(2, spec.action_dim))


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

        init_carries = jnp.zeros((B, HIDDEN))
        latent, _ = debug_fns.calculate_latent(
            state.feature_extractor,
            state.feature_extractor_target,
            observations,
            actions,
            dones,
            init_carries,
        )
        assert latent.shape[0] == T - BL

        # Manual reference: prev_a[t] = enc(a[t-1]) zeroed across done[t-1],
        # carry zeroed after done — exactly the online-path semantics.
        fe_graph, _ = nnx.split(
            RecurrentFeatureExtractor(
                input_shape=spec.obs_shape[0],
                projection=16,
                memory_type="gru",
                memory_hidden_dim=HIDDEN,
                prev_action_dim=spec.action_dim,
                rngs=nnx.Rngs(jax.random.key(0)),
            )
        )
        fe = nnx.merge(fe_graph, state.feature_extractor)
        carry = jnp.zeros((B, HIDDEN))
        prev = jnp.zeros((B, spec.action_dim))
        manual = []
        for t in range(T):
            new_carry, y = fe(carry, observations[:, t], prev)
            manual.append(y)
            enc = jax.nn.one_hot(
                jnp.round(actions[:, t, 0]).astype(jnp.int32), spec.action_dim
            )
            reset = dones[:, t][:, None] > 0
            carry = jnp.where(reset, 0.0, new_carry)
            prev = jnp.where(reset, 0.0, enc)
        manual = jnp.stack(manual[BL:])
        # atol is loose because XLA fuses the lax.scan (unroll=8) GRU matmuls
        # differently from this eager reference loop; a logic error (wrong
        # prev-action shift / reset) would diff by O(0.1+), not O(1e-4).
        assert jnp.allclose(latent, manual, atol=1e-3)


class TestValidation:
    def test_mismatched_fe_raises(self):
        env, env_params = gymnax.make("CartPole-v1")
        spec = env_spec_from_gymnax(env, env_params)
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
            input_shape=spec.obs_shape[0],
            projection=16,
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
            input_shape=spec.obs_shape[0],
            projection=16,
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

    def test_custom_projection_train_round(self):
        # A custom skip-connection projection module trains end to end.
        class Skip(nnx.Module):
            def __init__(self, obs_shape, pa_dim, rngs):
                flat = int(jnp.prod(jnp.array(obs_shape)))
                self.l1 = nnx.Linear(flat + pa_dim, 16, rngs=rngs)
                self.l2 = nnx.Linear(16, 16, rngs=rngs)
                self.skip = nnx.Linear(flat + pa_dim, 16, rngs=rngs)

            def __call__(self, obs, prev_action):
                x = obs.reshape(obs.shape[0], -1)
                if prev_action is not None and prev_action.shape[-1]:
                    x = jnp.concatenate([x, prev_action], axis=-1)
                return self.l2(jax.nn.relu(self.l1(x))) + self.skip(x)

        env, env_params = gymnax.make("CartPole-v1")
        spec = env_spec_from_gymnax(env, env_params)
        expert_data = {
            "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
            "actions": jnp.zeros((4, 1), dtype=jnp.float32),
        }
        state, fns, _ = create_iqlearn_from_env(
            spec,
            expert_data,
            buffer_size=4,
            hp=_tiny_hp(),
            projection=lambda s, p, r: Skip(s, p, r),
            memory_type="gru",
            memory_hidden_dim=HIDDEN,
            use_prev_action=True,
            critic_dims=(16,),
            train_steps=4,
            approximate_lambda=True,
            debug=True,
            seed=0,
        )
        key = jax.random.key(2)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        _, _, metrics = fns.train(state, env, env_params, env_state, key)
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}: {value}"
