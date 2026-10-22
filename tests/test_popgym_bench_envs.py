"""Step-by-step parity of the new JAX ports against NumPy POPGym 1.0.7.

The NumPy env is reset with a seed, its hidden layout (deal order, mines,
cart-pole physics) is copied into the JAX state, and both are stepped with the
same random actions until done. Requires popgym (skipped otherwise).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
popgym = pytest.importorskip('popgym')
jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

import popgym_bench_envs as envs  # noqa: E402
from popgym_suite import make_env  # noqa: E402


def numpy_env(env_id, seed):
    env = make_env(env_id)
    obs, _ = env.reset(seed=seed)
    return env, np.asarray(obs, np.float32)


def run_parity(env_id, seed, state, port, env, first_obs, initial_obs):
    np.testing.assert_allclose(np.asarray(initial_obs), first_obs, atol=1e-6)
    transition = jax.jit(port.transition)
    rng = np.random.default_rng(seed)
    for step in range(port.horizon + 1):
        action = int(rng.integers(port.action_dim))
        obs, reward, terminated, truncated, _ = env.step(action)
        jobs, state, jreward, jdone = transition(jax.random.PRNGKey(step), state, jnp.int32(action))
        np.testing.assert_allclose(np.asarray(jobs), np.asarray(obs, np.float32), atol=1e-4, err_msg=f'{env_id} step {step} obs')
        np.testing.assert_allclose(float(jreward), reward, rtol=1e-6, atol=1e-7, err_msg=f'{env_id} step {step} reward')
        assert bool(jdone) == (terminated or truncated), f'{env_id} step {step} done'
        if terminated or truncated:
            assert step + 1 <= port.horizon
            return step + 1
    raise AssertionError(f'{env_id}: no termination within horizon')


@pytest.mark.parametrize('difficulty', envs.DIFFICULTIES)
@pytest.mark.parametrize('seed', range(4))
def test_autoencode_parity(difficulty, seed):
    env_id = f'popgym-Autoencode{difficulty}-v0'
    port = envs.make_port(env_id)
    env, first = numpy_env(env_id, seed)
    deck = env.unwrapped.deck
    cards = jnp.asarray(deck.suits_idx[deck.idx[::-1]], jnp.int32)
    length = run_parity(env_id, seed, envs.AutoencodeState(cards, jnp.int32(0)), port, env, first, port.obs(1, cards[0]))
    assert length == port.horizon == env.unwrapped.max_episode_length


@pytest.mark.parametrize('difficulty', envs.DIFFICULTIES)
@pytest.mark.parametrize('seed', range(4))
def test_count_recall_parity(difficulty, seed):
    env_id = f'popgym-CountRecall{difficulty}-v0'
    port = envs.make_port(env_id)
    env, first = numpy_env(env_id, seed)
    u = env.unwrapped
    values = jnp.asarray(u.deck_idx_type[u.value_deck.idx[::-1]], jnp.int32)
    queries = jnp.asarray(u.deck_idx_type[u.query_deck.idx[::-1]], jnp.int32)
    counts = jnp.zeros(port.distinct, jnp.int32).at[values[0]].add(1)
    state = envs.CountRecallState(values, queries, counts, jnp.int32(0))
    assert port.action_dim == env.action_space.n and port.obs_shape == env.observation_space.shape
    length = run_parity(env_id, seed, state, port, env, first, port.obs(values[0], queries[0]))
    assert length == port.horizon


@pytest.mark.parametrize('difficulty', envs.DIFFICULTIES)
@pytest.mark.parametrize('seed', range(6))
def test_minesweeper_parity(difficulty, seed):
    env_id = f'popgym-MineSweeper{difficulty}-v0'
    port = envs.make_port(env_id)
    env, first = numpy_env(env_id, seed)
    u = env.unwrapped
    state = port.from_mines(u.hidden_grid == 1)
    np.testing.assert_array_equal(np.asarray(state.neighbours), u.neighbor_grid.reshape(-1))
    assert port.action_dim == env.action_space.n and port.obs_shape == env.observation_space.shape
    run_parity(env_id, seed, state, port, env, first, envs.one_hot(0, port.n_obs))


@pytest.mark.parametrize('difficulty', envs.DIFFICULTIES)
@pytest.mark.parametrize('seed', range(4))
def test_velocity_cartpole_parity(difficulty, seed):
    env_id = f'popgym-VelocityOnlyCartpole{difficulty}-v0'
    port = envs.make_port(env_id)
    env, first = numpy_env(env_id, seed)
    physics = jnp.asarray(np.asarray(env.unwrapped.state, np.float64), jnp.float32)
    state = envs.CartPoleState(physics, jnp.int32(0))
    assert port.horizon == env.unwrapped.max_episode_length
    run_parity(env_id, seed, state, port, env, first, physics[jnp.asarray([0, 2])])


def test_velocity_cartpole_balanced_episode_reaches_horizon():
    # A pole that never falls must be truncated exactly at L with return 1.
    port = envs.make_port('popgym-VelocityOnlyCartpoleEasy-v0')
    state = envs.CartPoleState(jnp.zeros(4), jnp.int32(0))
    total, done, t = 0., False, 0
    while not done:
        # Bang-bang on angular velocity keeps the pole up for 200 steps.
        action = int(state.physics[3] + 0.5 * state.physics[2] > 0)
        _, state, reward, done = port.transition(None, state, action)
        total += float(reward); t += 1
    assert t == port.horizon and abs(total - 1.) < 1e-4


@pytest.mark.parametrize('env_id', [i for i in envs.all_env_ids() if envs.backend_for(i) == 'jax-new'])
def test_reset_distribution_and_auto_reset(env_id):
    port = envs.make_port(env_id)
    obs, state = jax.jit(port.reset)(jax.random.PRNGKey(3))
    assert obs.shape == port.obs_shape and np.isfinite(np.asarray(obs)).all()
    if isinstance(port, envs.Autoencode):
        np.testing.assert_array_equal(np.sort(np.asarray(state.cards)), np.sort(port.values))
    if isinstance(port, envs.MineSweeper):
        assert int(state.grid.sum()) == port.mines
    step = jax.jit(port.step)
    length, done = 0, False
    while not done:
        obs, state, reward, done = step(jax.random.PRNGKey(length), state, jnp.int32(0))
        length += 1
        assert length <= port.horizon
    assert int(getattr(state, 't')) == 0  # auto-reset after done


def test_registry_covers_all_discrete_popgym_tasks():
    from popgym_suite import discover
    included, excluded = discover()
    assert sorted(r['env_id'] for r in included) == sorted(envs.all_env_ids())
    assert all('continuous' in r['reason'] for r in excluded)
    for row in included:
        port = envs.make_port(row['env_id'])
        assert port.action_dim == row['num_actions']
        if envs.backend_for(row['env_id']) != 'jax-popjym':
            assert list(port.obs_shape) == row['obs_shape']
