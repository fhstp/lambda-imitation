"""Tests for the General Value Discrepancy (GVD) branches.

GVD (Koepernik et al. 2025) trains two successor-feature (SF) V-heads via
V-trace λ-returns over observable-feature cumulants and regularises the
shared feature extractor with their squared discrepancy.  These tests cover:

* ``sf_vtrace_targets`` against a hand-rolled numpy recursion (λ ∈ {0, ½, 1},
  done masking, ratio clipping),
* the ``use_gvd=False`` path: ``None`` state fields, no ``gvd*`` metrics,
  and bit-identical actor/critic initialisation (RNG-stream isolation),
* end-to-end train rounds with GVD only, and with GVD + λ-discrepancy,
* the GVD discrepancy vanishing for identical SF heads,
* vmap-over-seeds safety of the GVD pytrees.

NOTE: this file is the executable spec for the user-implemented math core
(``sf_vtrace_targets``, ``loss_vtrace_sf_sequence``, ``loss_gvd`` — see
``~/.claude/plans/cosmic-drifting-dusk.md``).  Until those bodies replace
their ``NotImplementedError`` stubs, every test touching ``use_gvd=True``
fails; ``TestDisabledPath`` must stay green throughout.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.iqlearn import Hyperparameters, sf_vtrace_targets
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


HIDDEN = 8
N_FEATURES = 3

# Fixed projection for the CartPole feature map (obs_dim=4 -> N_FEATURES).
_P = jax.random.normal(jax.random.key(7), (4, N_FEATURES)) / 2.0


def _feature_fn(obs, prev_action):
    # prev_action accepted for the (obs, a_{t-1}) cumulant API; this test's
    # feature map is a plain obs projection and ignores it.
    return obs @ _P


def _tiny_hp(**overrides):
    base = dict(
        target_entropy=0.2,
        batch_size=4,
        online_batch_size=4,
        online_buffer_size=256,
        burn_in_length=2,
        sequence_length=4,
        lambda_truncation=2,
    )
    base.update(overrides)
    return Hyperparameters(**base)


def _make_agent(use_gvd, approximate_lambda=True, seed=0, hp=None):
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
        hp=hp or _tiny_hp(),
        projection_dim=16,
        memory_type="gru",
        memory_hidden_dim=HIDDEN,
        use_prev_action=True,
        critic_dims=(16,),
        train_steps=4,
        approximate_lambda=approximate_lambda,
        use_gvd=use_gvd,
        gvd_feature_fn=_feature_fn if use_gvd else None,
        gvd_sf_dims=(16,),
        debug=True,
        seed=seed,
    )
    return env, env_params, spec, state, fns, debug_fns


@pytest.fixture(scope="module")
def gvd_agent():
    return _make_agent(use_gvd=True, approximate_lambda=False)


# ---------------------------------------------------------------------------
# sf_vtrace_targets: hand-rolled reference recursion
# ---------------------------------------------------------------------------


def _reference_targets(V, f, dones, ratios, gamma, lam, rho_bar, c_bar):
    """Numpy reference of the vector V-trace backward recursion."""
    T, B, n = V.shape
    targets = np.zeros_like(V)
    v_next = np.zeros((B, n))
    V_next = np.zeros((B, n))
    for t in reversed(range(T)):
        rho = np.minimum(rho_bar, ratios[t])[:, None]
        c = lam * np.minimum(c_bar, ratios[t])[:, None]
        nd = (1.0 - dones[t])[:, None]
        delta = f[t] + nd * gamma * V_next - V[t]
        v_t = V[t] + rho * delta + nd * gamma * c * (v_next - V_next)
        targets[t] = v_t
        v_next, V_next = v_t, V[t]
    return targets


class TestSfVtraceTargets:
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
    def test_matches_hand_rolled(self, lam):
        rng = np.random.default_rng(0)
        T, B, n = 3, 1, 2
        V = rng.normal(size=(T, B, n)).astype(np.float32)
        f = rng.normal(size=(T, B, n)).astype(np.float32)
        dones = np.zeros((T, B), dtype=np.float32)
        dones[1] = 1.0  # episode boundary inside the window
        ratios = rng.uniform(0.5, 2.0, size=(T, B)).astype(np.float32)
        gamma, rho_bar, c_bar = 0.9, 1.1, 1.05

        got = sf_vtrace_targets(
            jnp.array(V), jnp.array(f), jnp.array(dones), jnp.array(ratios),
            gamma, lam, rho_bar, c_bar,
        )
        want = _reference_targets(V, f, dones, ratios, gamma, lam, rho_bar, c_bar)
        assert got.shape == (T, B, n)
        assert jnp.allclose(got, want, atol=1e-6)

    def test_lambda_zero_is_one_step_td(self):
        # λ=0 kills the (v_{t+1} − V_{t+1}) correction: the target reduces to
        # V_t + ρ(f_t + γ(1−d)V_{t+1} − V_t) — pure one-step TD.
        rng = np.random.default_rng(1)
        T, B, n = 4, 2, 3
        V = rng.normal(size=(T, B, n)).astype(np.float32)
        f = rng.normal(size=(T, B, n)).astype(np.float32)
        dones = np.zeros((T, B), dtype=np.float32)
        ratios = np.ones((T, B), dtype=np.float32)
        gamma = 0.95

        got = sf_vtrace_targets(
            jnp.array(V), jnp.array(f), jnp.array(dones), jnp.array(ratios),
            gamma, 0.0, 1.0, 1.0,
        )
        V_next = np.concatenate([V[1:], np.zeros((1, B, n))], axis=0)
        want = f + gamma * V_next  # rho=1: V + (f + γV' − V) = f + γV'
        assert jnp.allclose(got, want, atol=1e-6)

    def test_lambda_one_onpolicy_is_discounted_cumulant_sum(self):
        # λ=1, ratios=1, no dones: target_t = Σ_{k≥t} γ^{k−t} f_k (zero
        # terminal bootstrap) — the MC pseudo-return of the truncated window.
        rng = np.random.default_rng(2)
        T, B, n = 5, 1, 2
        V = rng.normal(size=(T, B, n)).astype(np.float32)
        f = rng.normal(size=(T, B, n)).astype(np.float32)
        dones = np.zeros((T, B), dtype=np.float32)
        ratios = np.ones((T, B), dtype=np.float32)
        gamma = 0.9

        got = sf_vtrace_targets(
            jnp.array(V), jnp.array(f), jnp.array(dones), jnp.array(ratios),
            gamma, 1.0, 1.0, 1.0,
        )
        want = np.zeros_like(f)
        acc = np.zeros((B, n))
        for t in reversed(range(T)):
            acc = f[t] + gamma * acc
            want[t] = acc
        assert jnp.allclose(got, want, atol=1e-5)

    def test_done_masks_bootstrap(self):
        # done at step t: target_t = V_t + ρ(f_t − V_t) exactly — nothing
        # may leak from t+1 (neither bootstrap nor λ-correction).
        T, B, n = 3, 1, 2
        V = np.ones((T, B, n), dtype=np.float32) * 5.0
        f = np.ones((T, B, n), dtype=np.float32)
        dones = np.zeros((T, B), dtype=np.float32)
        dones[0] = 1.0
        ratios = np.full((T, B), 0.7, dtype=np.float32)

        got = sf_vtrace_targets(
            jnp.array(V), jnp.array(f), jnp.array(dones), jnp.array(ratios),
            0.99, 1.0, 1.0, 1.0,
        )
        want_0 = V[0] + 0.7 * (f[0] - V[0])
        assert jnp.allclose(got[0], want_0, atol=1e-6)


# ---------------------------------------------------------------------------
# use_gvd=False path: disabled fields, no metric keys, RNG isolation
# ---------------------------------------------------------------------------


class TestDisabledPath:
    def test_state_fields_none_when_disabled(self):
        env, env_params, spec, state, fns, _ = _make_agent(use_gvd=False)
        assert state.gvd_sf1 is None
        assert state.gvd_sf2 is None
        assert state.gvd_sf1_target is None
        assert state.gvd_sf2_target is None
        assert state.gvd_sf1_optimizer_state is None
        assert state.gvd_sf2_optimizer_state is None

        key = jax.random.key(2)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        _, _, metrics = fns.train(state, env, env_params, env_state, key)
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}"
        assert not any(k.startswith("gvd") for k in metrics)

    def test_rng_isolation(self):
        # SF-head keys come from fold_in, so enabling GVD must not shift the
        # actor / critic / λ-critic initialisation stream.
        _, _, _, state_off, _, _ = _make_agent(use_gvd=False, seed=0)
        _, _, _, state_on, _, _ = _make_agent(use_gvd=True, seed=0)
        assert jax.tree.all(
            jax.tree.map(jnp.array_equal, state_off.actor, state_on.actor)
        )
        assert jax.tree.all(
            jax.tree.map(jnp.array_equal, state_off.critic, state_on.critic)
        )
        assert jax.tree.all(
            jax.tree.map(
                jnp.array_equal,
                state_off.feature_extractor,
                state_on.feature_extractor,
            )
        )


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_feature_fn_raises(self):
        env, env_params = gymnax.make("CartPole-v1")
        spec = env_spec_from_gymnax(env, env_params)
        expert_data = {
            "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
            "actions": jnp.zeros((4, 1), dtype=jnp.float32),
        }
        with pytest.raises(ValueError, match="gvd_feature_fn"):
            create_iqlearn_from_env(
                spec, expert_data, buffer_size=4, hp=_tiny_hp(),
                memory_type="gru", memory_hidden_dim=HIDDEN,
                use_prev_action=True, use_gvd=True, seed=0,
            )

    def test_zero_truncation_raises(self):
        with pytest.raises(ValueError, match="lambda_truncation"):
            _make_agent(use_gvd=True, hp=_tiny_hp(lambda_truncation=0))


# ---------------------------------------------------------------------------
# End-to-end train rounds
# ---------------------------------------------------------------------------


class TestEndToEnd:
    GVD_METRIC_KEYS = (
        "gvd_loss",
        "gvd_sf1_loss",
        "gvd_sf2_loss",
        "gvd_sf1_critic:",
        "gvd_sf2_critic:",
        "gvd_sf1_target:",
        "gvd_sf2_target:",
    )

    def test_train_round_finite_metrics_gvd(self, gvd_agent):
        env, env_params, spec, state, fns, _ = gvd_agent
        key = jax.random.key(3)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        _, _, metrics = fns.train(state, env, env_params, env_state, key)
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}"
        for k in self.GVD_METRIC_KEYS:
            assert k in metrics, f"missing GVD metric {k}"

    def test_both_ld_and_gvd_together(self):
        env, env_params, spec, state, fns, _ = _make_agent(
            use_gvd=True, approximate_lambda=True
        )
        key = jax.random.key(4)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        _, _, metrics = fns.train(state, env, env_params, env_state, key)
        assert "ld_loss" in metrics
        assert "gvd_loss" in metrics
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}"

    def test_gvd_loss_zero_for_identical_heads(self):
        # With SF2 ≡ SF1 AND equal λs the two heads receive bitwise-identical
        # targets and gradients every step, so they stay equal across the
        # whole scan and the mean discrepancy is exactly 0.  (With the
        # default λ1=0 ≠ λ2=1 the heads diverge after the first update —
        # different V-trace targets — so exact zero only holds for equal λ.)
        env, env_params, spec, state, fns, _ = _make_agent(
            use_gvd=True,
            approximate_lambda=False,
            hp=_tiny_hp(gvd_lambda1=0.5, gvd_lambda2=0.5),
        )
        state_eq = state._replace(
            gvd_sf2=state.gvd_sf1,
            gvd_sf2_target=state.gvd_sf1_target,
            gvd_sf2_optimizer_state=state.gvd_sf1_optimizer_state,
        )
        key = jax.random.key(5)
        key, reset_key, prefill_key = jax.random.split(key, 3)
        _, env_state = env.reset(reset_key, env_params)
        hp = _tiny_hp()
        prefill = hp.online_batch_size * (
            hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
        )
        state_eq, env_state = fns.prefill_buffer(
            state_eq, env, env_params, env_state, prefill, prefill_key
        )
        carry = jnp.zeros((HIDDEN + spec.action_dim,), dtype=jnp.float32)
        # gvd_agent fixture uses train_steps=4 and metrics are means over the
        # scan.  Identical heads receive identical SF gradients, so they stay
        # equal across all 4 steps and the mean discrepancy must be exactly 0.
        _, _, _, metrics = fns.train_unrolled(
            state_eq, env, env_params, env_state, carry, key
        )
        assert metrics["gvd_loss"] == 0.0

    def test_vmap_two_seeds(self):
        # Stack two GVD agents and run the unrolled trainer under vmap —
        # proves the GVD pytrees (no None leaves when enabled) and the
        # constant feature-projection closure are vmap-safe.
        env, env_params, spec, state_a, fns, _ = _make_agent(use_gvd=True, seed=0)
        _, _, _, state_b, _, _ = _make_agent(use_gvd=True, seed=1)

        key = jax.random.key(6)
        keys = jax.random.split(key, 2)

        def prep(state, k):
            k, reset_key, prefill_key = jax.random.split(k, 3)
            _, env_state = env.reset(reset_key, env_params)
            hp = _tiny_hp()
            prefill = hp.online_batch_size * (
                hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
            )
            state, env_state = fns.prefill_buffer(
                state, env, env_params, env_state, prefill, prefill_key
            )
            return state, env_state, k

        state_a, env_state_a, ka = prep(state_a, keys[0])
        state_b, env_state_b, kb = prep(state_b, keys[1])

        batched = jax.tree.map(lambda *xs: jnp.stack(xs), state_a, state_b)
        env_states = jax.tree.map(lambda *xs: jnp.stack(xs), env_state_a, env_state_b)
        carries = jnp.zeros((2, HIDDEN + spec.action_dim), dtype=jnp.float32)
        run_keys = jnp.stack([ka, kb])

        train_v = jax.jit(
            jax.vmap(
                lambda s, es, ec, k: fns.train_unrolled(
                    s, env, env_params, es, ec, k
                ),
                in_axes=(0, 0, 0, 0),
            )
        )
        _, _, _, metrics = train_v(batched, env_states, carries, run_keys)
        for name, value in metrics.items():
            assert value.shape[0] == 2
            assert jnp.isfinite(value).all(), f"non-finite metric {name}"
