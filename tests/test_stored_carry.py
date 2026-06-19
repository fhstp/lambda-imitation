"""Tests for R2D2 stored-state burn-in (``burn_in_from_stored_carry``).

With the flag enabled, ``run_env_step`` stores the memory carry (and, with
``use_prev_action``, the prev-action input) that the online policy consumed
at each step, and ``loss_combined`` initialises the burn-in of every sampled
training sequence from the stored carry / prev-action of its first step
instead of zeros.

Covered:

* buffer schema: ``carries`` (and ``prev_actions``) key present iff the
  flag is on,
* equivalence: with only pre-filled (zero-carry) data, one update under the
  flag is metric-identical to the zero-init path,
* alignment: stored carries are zero exactly at episode starts and non-zero
  mid-episode after a real rollout,
* end-to-end finite metrics and vmap-over-seeds safety.
"""

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.iqlearn import Hyperparameters, carry_key, prev_action_key
from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax


HIDDEN = 8


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


def _make_agent(stored_carry, seed=0, hp=None, train_steps=4):
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
        projection=16,
        memory_type="gru",
        memory_hidden_dim=HIDDEN,
        use_prev_action=True,
        critic_dims=(16,),
        train_steps=train_steps,
        approximate_lambda=True,
        burn_in_from_stored_carry=stored_carry,
        debug=True,
        seed=seed,
    )
    return env, env_params, spec, state, fns, debug_fns


def _prefill(env, env_params, state, fns, key):
    key, reset_key, prefill_key = jax.random.split(key, 3)
    _, env_state = env.reset(reset_key, env_params)
    hp = _tiny_hp()
    n = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, env_state = fns.prefill_buffer(
        state, env, env_params, env_state, n, prefill_key
    )
    return state, env_state, key


CARRY_DIM = HIDDEN  # GRU(8); the prev-action is threaded separately, not in carry
ACTION_DIM = 2  # CartPole prev-action one-hot width


# ---------------------------------------------------------------------------
# Buffer schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_buffer_has_carry_key_when_enabled(self):
        _, _, spec, state, _, _ = _make_agent(stored_carry=True)
        assert carry_key in state.online_buffer.info
        assert state.online_buffer.info[carry_key].shape[1:] == (CARRY_DIM,)
        # With use_prev_action the prev-action input is stored alongside.
        assert prev_action_key in state.online_buffer.info
        assert state.online_buffer.info[prev_action_key].shape[1:] == (ACTION_DIM,)

    def test_buffer_carry_key_absent_when_disabled(self):
        _, _, _, state, _, _ = _make_agent(stored_carry=False)
        assert carry_key not in state.online_buffer.info
        assert prev_action_key not in state.online_buffer.info


# ---------------------------------------------------------------------------
# Equivalence: stored zeros == zero init
# ---------------------------------------------------------------------------


class TestEquivalence:
    def test_prefill_carries_zero_and_equivalent_update(self):
        # Pre-fill writes no carries, so every stored carry is zero — one
        # update with the flag on must then be metric-identical to the
        # plain zero-init path (same seed, same keys).
        env, env_params, spec, state_off, fns_off, _ = _make_agent(
            stored_carry=False, seed=0
        )
        _, _, _, state_on, fns_on, _ = _make_agent(stored_carry=True, seed=0)

        key = jax.random.key(11)
        state_off, env_state_off, _ = _prefill(env, env_params, state_off, fns_off, key)
        state_on, env_state_on, _ = _prefill(env, env_params, state_on, fns_on, key)

        carries = state_on.online_buffer.info[carry_key]
        assert jnp.all(carries == 0.0)

        carry0 = jnp.zeros((CARRY_DIM,), dtype=jnp.float32)
        run_key = jax.random.key(12)
        _, _, _, m_off = fns_off.train_unrolled(
            state_off, env, env_params, env_state_off, carry0, run_key
        )
        _, _, _, m_on = fns_on.train_unrolled(
            state_on, env, env_params, env_state_on, carry0, run_key
        )
        for k in m_off:
            assert jnp.allclose(m_off[k], m_on[k], atol=0.0), (
                f"metric {k} diverged: {m_off[k]} vs {m_on[k]}"
            )


# ---------------------------------------------------------------------------
# Alignment of stored carries with episode structure
# ---------------------------------------------------------------------------


class TestAlignment:
    @pytest.fixture(scope="class")
    def trained_buffer(self):
        # 60 rollout steps: CartPole under an untrained policy terminates
        # every ~10-25 steps, so the rollout reliably crosses several
        # episode boundaries (needed by the episode-start test below).
        env, env_params, spec, state, fns, _ = _make_agent(
            stored_carry=True, train_steps=60
        )
        key = jax.random.key(21)
        state, env_state, key = _prefill(env, env_params, state, fns, key)
        carry0 = jnp.zeros((CARRY_DIM,), dtype=jnp.float32)
        key, run_key = jax.random.split(key)
        state, _, _, _ = fns.train_unrolled(
            state, env, env_params, env_state, carry0, run_key
        )
        return state.online_buffer

    def test_episode_start_carries_are_zero(self, trained_buffer):
        # Slot t directly after a terminal slot t-1 must hold a zero carry
        # (run_env_step zeroes the threaded carry on done).  Restrict to
        # slots actually written by the rollout (after the prefill region,
        # before the write cursor) so prefill zeros don't mask a bug.
        info = trained_buffer.info
        pos = int(trained_buffer.pos)
        terminated = np.array(info["terminated"])
        carries = np.array(info[carry_key])
        hp = _tiny_hp()
        prefill_n = hp.online_batch_size * (
            hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
        )
        checked = 0
        for t in range(prefill_n + 1, min(pos, len(terminated))):
            if terminated[t - 1] == 1.0:
                assert np.all(carries[t] == 0.0), f"slot {t} not zero"
                checked += 1
        # 60 rollout steps on CartPole cross several episode boundaries —
        # the invariant must have been exercised at least once.
        assert checked >= 1, "rollout crossed no episode boundary"

    def test_stored_carry_nonzero_mid_episode(self, trained_buffer):
        # The rollout threads a real GRU carry, so slots written by
        # run_env_step (after the prefill region) must contain non-zero
        # carries somewhere.
        info = trained_buffer.info
        pos = int(trained_buffer.pos)
        carries = np.array(info[carry_key])
        hp = _tiny_hp()
        prefill_n = hp.online_batch_size * (
            hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
        )
        rollout = carries[prefill_n:pos]
        assert rollout.shape[0] > 0
        assert np.any(rollout != 0.0)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_train_round_finite_metrics(self):
        env, env_params, spec, state, fns, _ = _make_agent(stored_carry=True)
        key = jax.random.key(31)
        key, reset_key = jax.random.split(key)
        _, env_state = env.reset(reset_key, env_params)
        _, _, metrics = fns.train(state, env, env_params, env_state, key)
        for name, value in metrics.items():
            assert jnp.isfinite(value).all(), f"non-finite metric {name}"

    def test_vmap_two_seeds(self):
        env, env_params, spec, state_a, fns, _ = _make_agent(stored_carry=True, seed=0)
        _, _, _, state_b, _, _ = _make_agent(stored_carry=True, seed=1)

        key = jax.random.key(41)
        keys = jax.random.split(key, 2)
        state_a, env_state_a, ka = _prefill(env, env_params, state_a, fns, keys[0])
        state_b, env_state_b, kb = _prefill(env, env_params, state_b, fns, keys[1])

        batched = jax.tree.map(lambda *xs: jnp.stack(xs), state_a, state_b)
        env_states = jax.tree.map(lambda *xs: jnp.stack(xs), env_state_a, env_state_b)
        carries = jnp.zeros((2, CARRY_DIM), dtype=jnp.float32)
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
