"""Retrace targets against an independent forward-trace NumPy oracle."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lambda_imitation.actor_critic import retrace_targets


def reference(q_taken, v, rewards, dones, ratios, gamma, lam):
    t_count, batch = q_taken.shape
    targets = np.zeros_like(q_taken)
    for b in range(batch):
        for t in range(t_count):
            correction, trace = 0.0, 1.0
            for k in range(t, t_count):
                if k > t:
                    trace *= lam * min(1.0, ratios[k, b])
                next_v = v[k + 1, b] if k + 1 < t_count else 0.0
                delta = rewards[k, b] + (1 - dones[k, b]) * gamma * next_v - q_taken[k, b]
                correction += gamma ** (k - t) * trace * delta
                if dones[k, b]:
                    break
            targets[t, b] = q_taken[t, b] + correction
    return targets


@pytest.fixture
def batch():
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    shape = (6, 3)
    return dict(q_taken=jax.random.normal(k1, shape), v=jax.random.normal(k2, shape),
                rewards=jax.random.normal(k3, shape),
                dones=(jax.random.uniform(k4, shape) < 0.2).astype(jnp.float32),
                ratios=jnp.abs(jax.random.normal(k1, shape)))


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_matches_forward_trace_oracle(batch, lam):
    got = jax.jit(retrace_targets)(gamma=0.99, lam=lam, **batch)
    expected = reference(**{k: np.asarray(v) for k, v in batch.items()}, gamma=0.99, lam=lam)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_cut_traces_keep_one_step_td_and_reward(batch):
    data = {**batch, "ratios": jnp.zeros_like(batch["ratios"])}
    got = retrace_targets(**data, gamma=0.99, lam=1.0)
    next_v = jnp.concatenate([batch["v"][1:], jnp.zeros_like(batch["v"][:1])])
    expected = batch["rewards"] + (1 - batch["dones"]) * 0.99 * next_v
    np.testing.assert_allclose(got, expected, atol=1e-5)
    bumped = retrace_targets(**{**data, "rewards": data["rewards"] + 10}, gamma=0.99, lam=1.0)
    np.testing.assert_allclose(bumped - got, 10, atol=1e-5)


def test_terminal_cuts_bootstrap_and_future_trace(batch):
    data = {**batch, "dones": jnp.zeros_like(batch["dones"]).at[2].set(1)}
    got = retrace_targets(**data, gamma=0.99, lam=1.0)
    expected = reference(**{k: np.asarray(v) for k, v in data.items()}, gamma=0.99, lam=1.0)
    np.testing.assert_allclose(got, expected, atol=1e-5)
    np.testing.assert_allclose(got[2], batch["rewards"][2], atol=1e-5)
