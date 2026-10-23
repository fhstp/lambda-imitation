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


@pytest.mark.parametrize('new_format', [True, False])
def test_wandb_exporter_rows(new_format):
    import popgym_bench_wandb as exporter
    config = dict(seeds=[42, 43], n_seeds=2, seed_base=42, max_steps=40, env_id='popgym-RepeatPreviousEasy-v0', arm='primary', horizon=51)
    rows = [{'steps': 20, 'eval/stochastic/return_mean/mean': .1, 'seed/42/eval/stochastic/return_mean': 0., 'seed/43/eval/stochastic/return_mean': .2,
             'eval/greedy/return_mean/mean': .3},
            {'steps': 40, 'eval/stochastic/return_mean/mean': .0, 'seed/42/eval/stochastic/return_mean': .1, 'seed/43/eval/stochastic/return_mean': -.1},
            {'steps': 40, 'final/stochastic/return_mean/mean': .05, 'final/stochastic/return_mean/std': .01,
             'final/random/return_mean/mean': -.5, 'final/random/return_mean/std': .01}]
    if new_format:
        rows[2].update({'seed/42/final/stochastic/return_mean': .04, 'seed/43/final/stochastic/return_mean': .06})
    summary = dict(steps=40, run_state='complete') if new_format else dict(steps=40)
    arm = exporter.arm_from_rows(rows, config, summary, 'finished', 'https://wandb.ai/x', None)
    assert arm['state'] == 'complete' and [r['steps'] for r in arm['curve']] == [20, 40]
    assert abs(arm['best']['mean'] - .15) < 1e-9 and arm['latest']['mean'] == 0.
    assert arm['final']['random']['mean'] == -.5 and arm['final']['stochastic']['seeds'] == ([.04, .06] if new_format else [])
    resumed = exporter.arm_from_rows([{**rows[0], 'eval/stochastic/return_mean/mean': -9.}] + rows, config, summary, 'finished', None, None)
    assert [r['mean'] for r in resumed['curve']] == [.1, 0.]  # re-logged step: latest row wins
    assert exporter.arm_from_rows(rows[:2], config, dict(steps=40), 'finished', None, None)['state'] == 'stopped'
    assert exporter.arm_from_rows(rows[:2], config, dict(steps=40), 'crashed', None, None)['state'] == 'failed'


class FakeRun:
    def __init__(self, id, arm='primary', env='popgym-RepeatPreviousEasy-v0', state='running', created=0., heartbeat=None,
                 summary=None, job_type=None, config=None):
        import time as _t
        self.id, self.state, self.created_at = id, state, created
        self.heartbeat_at = _t.time() if heartbeat is None else heartbeat
        self.job_type = job_type or arm
        self.config = config if config is not None else dict(arm=arm, env_id=env, seeds=[42, 43], max_steps=40)
        self.summary = summary or {}
        self.url = f'https://wandb.ai/x/{id}'


def test_claim_board_owner_and_availability():
    import time as _t
    import popgym_bench_coord as coord
    key = ('primary', 'popgym-RepeatPreviousEasy-v0')
    runs = [FakeRun('late', created=20.), FakeRun('early', created=10.), FakeRun('dead', created=1., heartbeat=_t.time() - 3600),
            FakeRun('gone', created=0., summary=dict(run_state='yielded')), FakeRun('sup', job_type='supervisor', config={})]
    tasks = coord.board([coord.record(r) for r in runs])
    assert list(tasks) == [key] and coord.owner(tasks[key]) == 'early'
    assert not coord.available(tasks, key) and coord.available(tasks, key, local_ids={'early', 'late'})
    assert coord.available(tasks, ('control', 'popgym-RepeatPreviousEasy-v0'))
    done = coord.board([coord.record(FakeRun('x', state='finished', summary=dict(run_state='complete', steps=40)))])
    assert done[key]['complete'] and not coord.available(done, key, local_ids={'x'})
    paused = coord.board([coord.record(FakeRun('p', state='finished', summary=dict(run_state='paused', steps=20)))])
    assert coord.available(paused, key) and paused[key]['steps'] == 20


def test_merge_attempts_drops_discarded_work_and_keeps_best_state():
    import popgym_bench_coord as coord
    import popgym_bench_wandb as exporter
    def row(steps, value):
        return {'steps': steps, 'eval/stochastic/return_mean/mean': value, 'seed/42/eval/stochastic/return_mean': value,
                'seed/43/eval/stochastic/return_mean': value}
    old = FakeRun('old', state='crashed', created=1., summary=dict(steps=60))
    new = FakeRun('new', created=2., summary=dict(steps=40, run_state='running'))
    old_rows = [row(0, 0.), row(20, .1), row(40, .9), row(60, .9)]  # 40/60: lost after a crash at 60
    new_rows = [row(40, .2)]                                       # resumed from the 20k checkpoint
    arm = exporter.merge_attempts([(old, coord.record(old), old_rows), (new, coord.record(new), new_rows)])
    assert [(r['steps'], r['mean']) for r in arm['curve']] == [(0, 0.), (20, .1), (40, .2)]
    assert abs(arm['best']['mean'] - .2) < 1e-9 and arm['state'] == 'running' and arm['url'].endswith('/new')
    assert arm['attempts'] == 2


def test_find_workers_adopts_live_worker(tmp_path):
    out = tmp_path / 'out'
    fake = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', str(EXAMPLES / 'popgym_bench.py'), 'worker',
                             '--output', str(out), '--arm', 'primary', '--env', 'popgym-RepeatPreviousEasy-v0'],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '5'})
    try:
        found = bench.find_workers(out)
        assert list(found) == [('primary', 'popgym-RepeatPreviousEasy-v0')]
        proc, gpu = found[('primary', 'popgym-RepeatPreviousEasy-v0')]
        assert gpu == '5' and proc.pid == fake.pid and proc.poll() is None
    finally:
        fake.kill(); fake.wait()
    assert proc.poll() is not None


def test_deadline_pauses_then_resumes_to_completion(tmp_path):
    tiny_study(tmp_path)
    env_id = 'popgym-RepeatPreviousEasy-v0'
    import time as _t
    code = bench.train_task(tmp_path, 'primary', env_id, {}, small=True, deadline=_t.time() + 5, stop_margin=60)
    assert code == bench.EXIT_PAUSED
    status = json.loads((tmp_path / 'primary' / env_id / 'status.json').read_text())
    assert status['state'] == 'paused' and status['steps'] == 0
    assert bench.train_task(tmp_path, 'primary', env_id, {}, small=True) == 0
    history = json.loads((tmp_path / 'primary' / env_id / 'history.json').read_text())
    assert [r['steps'] for r in history] == [0, 32, 64, 64] and history[-1]['final']


def test_parse_slurm_time():
    assert bench.parse_slurm_time('1-02:03:04') == 26 + 3 / 60 + 4 / 3600
    assert bench.parse_slurm_time('23:30:00') == 23.5 and abs(bench.parse_slurm_time('05:00') - 5 / 60) < 1e-12
    assert bench.parse_slurm_time('UNLIMITED') is None and bench.parse_slurm_time('') is None


def _trained(config, steps=32, env_id='popgym-RepeatPreviousEasy-v0'):
    c = bench.resolve_config({**TINY, **config})
    port = make_port(env_id)
    states, runner, seeds = bench.build(c, port, small=True)
    rollout = runner.initial_rollout(seeds)
    states, rollout, _ = runner.prefill(states, rollout, c['prefill_steps'])
    return runner.train(states, rollout, steps)


def test_soft_target_with_zero_alpha_matches_original_objective():
    tiny_alpha = dict(alpha_initial=1e-30, alpha_final=1e-30)
    a, b = _trained(tiny_alpha), _trained({**tiny_alpha, 'soft_main_target': True})
    for x, y in zip(jax.tree.leaves(a[0].critic), jax.tree.leaves(b[0].critic)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(float(a[2]['target_q'][0]), float(b[2]['target_q'][0]), rtol=1e-5, atol=1e-6)
    assert float(b[2]['soft_bonus'][0]) < 1e-20


def test_soft_target_adds_entropy_bonus():
    big = dict(alpha_initial=.05, alpha_final=.05)
    hard, soft = _trained(big, 16), _trained({**big, 'soft_main_target': True}, 16)
    bonus = float(soft[2]['soft_bonus'][0])
    assert 0 < bonus <= .05 * np.log(4) + 1e-6  # alpha * H(pi), H <= log A
    assert float(soft[2]['target_q'][0]) > float(hard[2]['target_q'][0])


def test_autotune_alpha_moves_towards_target_entropy():
    out = _trained(dict(autotune_alpha=True, alpha_initial=.5, alpha_lr=1e-2, target_entropy_fraction=.3), 64)
    alpha = np.asarray(out[0].alpha)
    # A near-uniform young policy has entropy above 0.3*log A, so alpha must fall.
    assert (alpha < .5).all() and np.allclose(np.log(alpha), np.asarray(out[0].log_alpha))
    assert np.allclose(np.asarray(out[2]['target_entropy']), .3 * np.log(4))


def test_alpha_scale_and_old_config_compatibility():
    c = bench.resolve_config({'alpha_scale': 'inverse_horizon'})
    assert abs(float(bench.alpha_at(0, c, 100)) - .01) < 1e-7 and abs(float(bench.alpha_at(10 ** 7, c, 100)) - 1e-4) < 1e-9
    old = {k: v for k, v in bench.DEFAULTS.items() if k not in ('alpha_scale', 'soft_main_target', 'autotune_alpha',
                                                                  'target_entropy_fraction', 'alpha_lr')}
    assert bench.resolve_config(old) == bench.resolve_config({})
