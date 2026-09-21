"""Retrace(λ) targets for the λ-critics.

Measured on the offline Battleship runs: the λ-critics' V-trace targets walk
off (E[Q] −18 → −256 while true returns are ≈ +5) because the target policy is
near-deterministic and disagrees with the behaviour policy, so ρ = min(1, π/b)
is 0 on ~97 % of transitions.  V-trace multiplies the *current* step's delta by
ρ, so on those steps the target degenerates to ``v_s = V(s)`` — no reward, no
bootstrap, nothing pinning the value.

Retrace (Munos et al. 2016) targets the taken action's Q and puts the k = t term
outside the trace product, so its coefficient is always 1:

    target_t = Q̄(s_t,a_t) + Σ_{k≥t} γ^(k-t) (Π_{j=t+1..k} c_j) δ_k
    δ_k      = r_k + γ(1-done_k) V̄(s_{k+1}) - Q̄(s_k,a_k)
    c_j      = λ · min(c_bar, ρ_j)

Small ratios then cut the *tail* — the estimator degrades to 1-step TD, which is
Retrace's documented "safe but slow" behaviour — instead of erasing the reward.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.iqlearn import retrace_targets


def _reference(q_taken, v, rewards, dones, ratios, gamma, lam, c_bar):
    """Hand-rolled numpy Retrace recursion (the spec, written independently)."""
    T, B = q_taken.shape
    targets = np.zeros_like(q_taken)
    for b in range(B):
        for t in range(T):
            acc = 0.0
            trace = 1.0
            for k in range(t, T):
                if k > t:
                    trace *= lam * min(c_bar, ratios[k, b])
                v_next = v[k + 1, b] if k + 1 < T else 0.0
                delta = rewards[k, b] + (1 - dones[k, b]) * gamma * v_next - q_taken[k, b]
                acc += (gamma ** (k - t)) * trace * delta
                if dones[k, b]:      # trace cannot cross an episode boundary
                    break
            targets[t, b] = q_taken[t, b] + acc
    return targets


@pytest.fixture
def batch():
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    T, B = 6, 3
    return dict(
        q_taken=jax.random.normal(k1, (T, B)),
        v=jax.random.normal(k2, (T, B)),
        rewards=jax.random.normal(k3, (T, B)),
        dones=(jax.random.uniform(k4, (T, B)) < 0.2).astype(jnp.float32),
        ratios=jnp.abs(jax.random.normal(k1, (T, B))),
    )


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_matches_hand_rolled_recursion(batch, lam):
    got = retrace_targets(gamma=0.99, lam=lam, c_bar=1.0, **batch)
    want = _reference(**{k: np.asarray(v) for k, v in batch.items()},
                      gamma=0.99, lam=lam, c_bar=1.0)
    assert np.allclose(np.asarray(got), want, atol=1e-5), (got, want)


def test_cut_traces_reduce_to_one_step_td(batch):
    """ratio = 0 everywhere: the tail vanishes but the 1-step term survives."""
    z = jnp.zeros_like(batch["ratios"])
    got = retrace_targets(**{**batch, "ratios": z}, gamma=0.99, lam=1.0, c_bar=1.0)
    v_next = jnp.concatenate([batch["v"][1:], jnp.zeros_like(batch["v"][:1])])
    one_step = batch["rewards"] + (1 - batch["dones"]) * 0.99 * v_next
    assert np.allclose(np.asarray(got), np.asarray(one_step), atol=1e-5)


def test_reward_still_reaches_the_target_when_vtrace_would_erase_it(batch):
    """The regression this exists to prevent: with ρ = 0, V-trace gives
    ``v_s = V(s)`` (reward-free); Retrace must still depend on the reward."""
    z = jnp.zeros_like(batch["ratios"])
    base = retrace_targets(**{**batch, "ratios": z}, gamma=0.99, lam=1.0, c_bar=1.0)
    bumped = retrace_targets(**{**batch, "ratios": z,
                                "rewards": batch["rewards"] + 10.0},
                             gamma=0.99, lam=1.0, c_bar=1.0)
    assert np.allclose(np.asarray(bumped - base), 10.0, atol=1e-5)


def test_done_cuts_the_trace(batch):
    """A terminal step must not bootstrap, and must not propagate credit back."""
    dones = jnp.zeros_like(batch["dones"]).at[2].set(1.0)
    got = retrace_targets(**{**batch, "dones": dones}, gamma=0.99, lam=1.0, c_bar=1.0)
    want = _reference(**{k: np.asarray(v) for k, v in {**batch, "dones": dones}.items()},
                      gamma=0.99, lam=1.0, c_bar=1.0)
    assert np.allclose(np.asarray(got), want, atol=1e-5)
    # the terminal step's own target is exactly its reward
    assert np.allclose(np.asarray(got[2]), np.asarray(batch["rewards"][2]), atol=1e-5)


def test_agent_trains_on_retrace_targets_with_finite_metrics():
    """End-to-end: retrace=True runs through update_only and keeps the λ-critic
    targets finite (the V-trace path is unchanged when the flag is off)."""
    import gymnax
    from lambda_imitation.iqlearn import Hyperparameters
    from lambda_imitation.utils import create_iqlearn_from_env, env_spec_from_gymnax

    env, env_params = gymnax.make("CartPole-v1")
    spec = env_spec_from_gymnax(env, env_params)
    hp = Hyperparameters(
        target_entropy=0.2, batch_size=4, online_batch_size=4,
        online_buffer_size=256, burn_in_length=2, sequence_length=4,
        lambda_truncation=2, retrace=True,
    )
    expert_data = {
        "observations": jnp.zeros((4, *spec.obs_shape), dtype=jnp.float32),
        "actions": jnp.zeros((4, 1), dtype=jnp.float32),
    }
    state, fns = create_iqlearn_from_env(
        spec, expert_data, buffer_size=4, hp=hp, projection=16,
        memory_type="gru", memory_hidden_dim=8, use_prev_action=True,
        critic_dims=(16,), train_steps=4, approximate_lambda=True, seed=0,
    )
    key = jax.random.key(0)
    key, reset_key, prefill_key, update_key = jax.random.split(key, 4)
    _obs, env_state = env.reset(reset_key, env_params)
    prefill = hp.online_batch_size * (
        hp.lambda_truncation + hp.sequence_length + hp.burn_in_length
    )
    state, _ = fns.prefill_buffer(
        state, env, env_params, env_state, prefill, prefill_key)
    _state, metrics = fns.update_only(state, 2, update_key)
    assert all(jnp.isfinite(v).all() for v in jax.tree.leaves(metrics)), metrics
