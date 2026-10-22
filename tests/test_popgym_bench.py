"""CPU tests for the full POPGym benchmark runner (tiny networks/budgets)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
sys.path.insert(0, str(EXAMPLES))
jax = pytest.importorskip('jax')
import jax.numpy as jnp  # noqa: E402

import popgym_bench as bench  # noqa: E402
from popgym_bench_envs import make_port  # noqa: E402

TINY = dict(n_seeds=2, max_steps=64, eval_every=32, checkpoint_every=64, chunk_steps=32, prefill_steps=128,
            memory_hidden_dim=8, projection_dim=8, actor_width=8, critic_width=8, online_buffer_size=4096,
            batch_transitions=64, min_batch_episodes=2, eval_episodes=2, final_episodes=2)


def tiny_study(tmp_path, **extra):
    (tmp_path / 'study.json').write_text(json.dumps(dict(config={**TINY, **extra})))
    return bench.resolve_config({**TINY, **extra})


def test_noise_duration_law_and_reset():
    c = bench.resolve_config({})
    noise = bench.noise_init(0, 5)
    advance = jax.jit(lambda n: bench.noise_advance(n, c))
    durations, biases, fresh = [], [], 0
    for _ in range(4000):
        noise = advance(noise)
        if int(noise[2]) == int(noise[3]) - 1:
            durations.append(int(noise[3])); fresh += 1
        biases.append(np.asarray(noise[1]))
    durations = np.asarray(durations)
    assert durations.min() >= 1 and durations.max() <= 16
    expected = (np.arange(1, 17) ** -2.) / (np.arange(1, 17) ** -2.).sum()
    assert abs((durations == 1).mean() - expected[0]) < .05
    # Bias is held exactly for `duration` consecutive decisions.
    assert sum(durations) == 4000 or sum(durations) - 4000 < 16
    reset = bench.noise_reset(noise, jnp.bool_(True))
    assert int(reset[2]) == 0 and not np.asarray(reset[1]).any()
    np.testing.assert_array_equal(np.asarray(reset[0]), np.asarray(noise[0]))


@pytest.fixture(scope='module')
def built():
    c = bench.resolve_config(TINY)
    port = make_port('popgym-RepeatPreviousEasy-v0')
    states, runner, seeds = bench.build(c, port, small=True)
    return c, port, states, runner, seeds


def test_zero_strength_behaviour_matches_actor_predict(built):
    c, port, states, runner, seeds = built
    state = jax.tree.map(lambda x: x[0], states)
    logits_fn = bench.policy_logits_fn(runner.fns)
    obs, _ = port.reset(jax.random.PRNGKey(1))
    carry, prev = runner.zero_carry, runner.zero_prev.at[2].set(1.)
    noise = bench.noise_init(0, port.action_dim)
    for k in range(8):
        key = jax.random.PRNGKey(k)
        action, prob, nc, _ = bench.behaviour_step(logits_fn, state, obs, carry, prev, noise, key, {**c, 'noise_strength': 0.}, False, port.action_dim)
        a2, nc2, p2 = runner.fns.predict(state, obs, carry, key, return_prob=True, prev_action=prev)
        assert int(action) == int(a2)
        np.testing.assert_allclose(float(prob), float(p2), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(nc), np.asarray(nc2), rtol=1e-6, atol=1e-7)


def test_correlated_behaviour_probability_is_exact_conditional(built):
    c, port, states, runner, seeds = built
    state = jax.tree.map(lambda x: x[0], states)
    logits_fn = bench.policy_logits_fn(runner.fns)
    obs, _ = port.reset(jax.random.PRNGKey(1))
    noise = bench.noise_init(3, port.action_dim)
    held = None
    for k in range(20):
        action, prob, _, noise = bench.behaviour_step(logits_fn, state, obs, runner.zero_carry, runner.zero_prev, noise,
                                                      jax.random.PRNGKey(k), c, False, port.action_dim)
        logits, _ = logits_fn(state, obs, runner.zero_carry, runner.zero_prev)
        expected = jax.nn.softmax(logits + noise[1])[action]
        np.testing.assert_allclose(float(prob), float(expected), rtol=1e-6)
        if held is not None and int(noise[2]) < held[1]:
            np.testing.assert_array_equal(np.asarray(noise[1]), held[0])  # same bias while held
        held = (np.asarray(noise[1]), int(noise[2]))


def test_buffer_records_behaviour_and_prefill_boundary(built):
    c, port, states, runner, seeds = built
    rollout = runner.initial_rollout(seeds)
    states, rollout, counts = runner.prefill(states, rollout, c['prefill_steps'])
    runner.check_rollout(rollout, at_boundary=True)
    assert (np.asarray(counts) >= c['prefill_steps']).all()
    weights = np.asarray(states.online_buffer.info['behaviour_weight'])[:, :int(counts.min())]
    np.testing.assert_allclose(weights, 1. / port.action_dim)
    states, rollout, metrics = runner.train(states, rollout, 32)
    pos = np.asarray(states.online_buffer.pos)
    new = np.asarray(states.online_buffer.info['behaviour_weight'])[0, pos[0] - 32:pos[0]]
    assert ((new > 0) & (new <= 1)).all() and not np.allclose(new, 1. / port.action_dim)
    assert np.isfinite([np.asarray(v) for v in metrics.values()]).all()


@pytest.mark.parametrize('env_id', ['popgym-BattleshipEasy-v0', 'popgym-AutoencodeEasy-v0'])
def test_validation_gate_cpu(tmp_path, env_id):
    tiny_study(tmp_path)
    proof = bench.validate(tmp_path, [env_id], updates=32, small=True)
    assert proof['passed'] and proof['tasks'][env_id]['passed']


def test_host_runner_checkpoint_resume_is_exact(tmp_path):
    import cloudpickle
    c = bench.resolve_config({**TINY, 'prefill_steps': 64})
    port = make_port('popgym-LabyrinthEscapeEasy-v0')
    states, runner, seeds = bench.build(c, port, small=True)
    rollout = runner.initial_rollout(seeds)
    states, rollout, _ = runner.prefill(states, rollout, c['prefill_steps'])
    runner.check_rollout(rollout, at_boundary=True)
    states, rollout, _ = runner.train(states, rollout, 16)
    blob = cloudpickle.dumps(dict(state=jax.device_get(states), rollout=bench.host_rollout(rollout)))
    a = runner.train(states, rollout, 32)
    saved = cloudpickle.loads(blob)
    b = runner.train(jax.tree.map(jnp.asarray, saved['state']), bench.restore_rollout(saved['rollout']), 32)
    for x, y in zip(jax.tree.leaves(a[0]), jax.tree.leaves(b[0])):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    np.testing.assert_array_equal(a[1]['obs'], b[1]['obs'])


def test_supervisor_end_to_end_runs_primary_before_control(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(TINY))
    out = tmp_path / 'out'
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'PYTHONPATH': os.pathsep.join(sys.path)}
    cmd = [sys.executable, str(EXAMPLES / 'popgym_bench.py'), '--output', str(out), '--config', str(config),
           '--envs', 'popgym-RepeatPreviousEasy-v0', 'popgym-MineSweeperEasy-v0', '--gpus', '0', '--workers-per-gpu', '2',
           '--no-wandb', '--dashboard-port', '0', '--poll-seconds', '1', '--exit-when-done']
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((out / 'validation.json').read_text())['passed']
    finished = {}
    for arm in ('primary', 'control'):
        for env_id in ('popgym-RepeatPreviousEasy-v0', 'popgym-MineSweeperEasy-v0'):
            status = json.loads((out / arm / env_id / 'status.json').read_text())
            assert status['state'] == 'complete'
            finished[arm, env_id] = status['time']
            history = json.loads((out / arm / env_id / 'history.json').read_text())
            assert history[-1]['final'] and set(history[-1]['eval']) == {'stochastic', 'greedy', 'critic', 'random'}
    assert max(v for k, v in finished.items() if k[0] == 'primary') < min(v for k, v in finished.items() if k[0] == 'control')
    control = json.loads((out / 'control' / 'popgym-MineSweeperEasy-v0' / 'config.json').read_text())
    assert control['config']['lambda_coef'] == 0. and control['config']['lambda2'] == .99
    import popgym_bench_dashboard as dashboard
    data = dashboard.build_data(out)
    assert [t['env_id'] for t in data['tasks']] == ['popgym-RepeatPreviousEasy-v0', 'popgym-MineSweeperEasy-v0']
    task = data['tasks'][0]
    assert task['arms']['primary']['final']['random']['n'] == 2 and task['arms']['primary']['best']['n'] == 2
    assert task['refs']['popgym']['mean'] == 1.0 and 'weekend-lambda' in task['refs']
    json.dumps(data, allow_nan=False)
    lines = result.stdout.splitlines()
    starts = [l for l in lines if ' start ' in l]
    assert all('primary' in l for l in starts[:2]) and all('control' in l for l in starts[2:])
