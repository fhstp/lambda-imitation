"""Fresh matched online Battleship Q-only bootstrap feature alignment control.

All production files remain immutable. Cloned closures pack original target
policy features and online Q features separately; no actor/ratio retargeting.
"""

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from popgym_alpha_sweep import file_hash, args_from_plan, verify_plan
from popgym_critic_aligned import replace_captures
from popgym_gradient_balance import captures
from popgym_suite import source_fingerprint, write_json


def aligned_functions(fns, feature_dim):
    """Clone this update only. Targets detached; policy uses original target FE."""
    import jax
    import jax.numpy as jnp
    combined = captures(fns.update)['loss_combined']; h = captures(combined)
    if not h['is_discrete'] or h['use_gvd'] or h['burn_in_from_stored_carry']:
        raise ValueError('Control requires the audited discrete, no-GVD configuration')
    calculate = h['calculate_latent']

    def packed_latent(*args, **kwargs):
        x, y = calculate(*args, **kwargs)
        if x.shape[-1] != feature_dim:
            raise ValueError('Feature dimension changed')
        return x, jnp.concatenate((y, jax.lax.stop_gradient(x)), axis=-1)

    def split(z):
        if z.shape[-1] != 2 * feature_dim:
            raise ValueError('Expected separate policy and Q feature inputs')
        return z[..., :feature_dim], z[..., feature_dim:]

    def value(get_v):
        v = captures(get_v); actor = v['run_actor']; critic = v['run_critic']
        def run_actor(state, z):
            return actor(state, split(z)[0])
        def run_critic(state, graph, z):
            return critic(state, graph, split(z)[1])
        return replace_captures(get_v, {'run_actor': run_actor, 'run_critic': run_critic})

    main = h['loss_critic']; main = replace_captures(main, {'get_v': value(captures(main)['get_v'])})
    trace = h['loss_vtrace_lambda_sequence']; low = captures(trace)
    original_q, original_ratio = low['get_q'], low['get_importance_ratios']
    def get_q(state, graph, z, actions):
        # Online regression/diagnostic calls retain the ordinary F-width input.
        return original_q(state, graph, split(z)[1] if z.shape[-1] == 2 * feature_dim else z, actions)
    def ratios(actor, z, actions, behaviour, mask=None):
        return original_ratio(actor, split(z)[0], actions, behaviour, mask)
    trace = replace_captures(trace, {'get_v': value(low['get_v']),
        'get_q': get_q, 'get_importance_ratios': ratios})
    cloned = replace_captures(combined, {'calculate_latent': packed_latent,
        'loss_critic': main, 'loss_vtrace_lambda_sequence': trace})
    return fns._replace(update=replace_captures(fns.update, {'loss_combined': cloned}))


def make_probe(fns, debug, dim, aligned):
    import jax
    import jax.numpy as jnp
    h = captures(captures(fns.update)['loss_combined'])
    if h['mask_fn'] is not None:
        raise ValueError('Battleship diagnostic assumes the audited unmasked port')
    main = captures(h['loss_critic'])
    run_q = captures(main['get_q_both'])['run_critic']
    # Prediction helpers retain the original closures even in the aligned arm.
    actor = captures(debug.get_entropy)['run_actor']
    graphs = [main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph']]
    flat = lambda a: a.reshape(-1, a.shape[-1])
    rms = lambda a: jnp.sqrt(jnp.mean(a ** 2))

    @jax.jit
    def probe(state, key):
        keys = jax.random.split(key, 5)
        sample, indices = h['online_buffer_lambda_sample'](state.online_buffer, keys[0])
        raw = sample.this_info
        x, y = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
            raw['observations'], raw['actions'], raw['terminated'],
            jnp.zeros((h['params'].online_batch_size, dim)))
        target = jnp.concatenate((y, jax.lax.stop_gradient(x)), -1) if aligned else y
        tm = lambda k: jnp.swapaxes(raw[k], 0, 1)[h['params'].burn_in_length:]
        actions, rewards, done, behaviour = [tm(k) for k in ('actions', 'rewards', 'terminated', 'behaviour_weight')]
        losses = [h['loss_critic'](state.actor_target, state.critic, state.critic_target,
            flat(x[:-1]), flat(target[1:]), flat(actions[:-1]), rewards[:-1].reshape(-1),
            done[:-1].reshape(-1), state.alpha, keys[2])[0]]
        for name, lam, graph, key in zip(('lambda1', 'lambda2'), (h['params'].lambda1, h['params'].lambda2), graphs[1:], keys[3:]):
            losses.append(h['loss_vtrace_lambda_sequence'](state.actor_target,
                getattr(state, name + '_critic'), getattr(state, name + '_critic_target'), graph,
                lam, actions, rewards, done, behaviour, x, target, key)[0])
        keep = h['params'].sequence_length; xx, yy = flat(x[:keep]), flat(y[:keep])
        result = {'feature_relative_mismatch': rms(xx - yy) / jnp.maximum(rms(xx), 1e-12),
            'target_policy_feature_probability_l1': jnp.mean(jnp.sum(jnp.abs(
                jax.nn.softmax(actor(state.actor_target, xx)) - jax.nn.softmax(actor(state.actor_target, yy))), -1))}
        q_values = []
        for name, attr, graph, loss in zip(('main', 'lambda1', 'lambda2'),
                ('critic', 'lambda1_critic', 'lambda2_critic'), graphs, losses):
            q = run_q(getattr(state, attr), graph, xx)
            tx = run_q(getattr(state, attr + '_target'), graph, xx)
            ty = run_q(getattr(state, attr + '_target'), graph, yy)
            result[name + '_actual_production_mse'] = loss
            result[name + '_q_rms_all_actions_twins'] = rms(q)
            result[name + '_q_max_abs_all_actions_twins'] = jnp.max(jnp.abs(q))
            result[name + '_target_q_feature_sensitivity_rms'] = rms(tx - ty)
            q_values.append(q.min(-1))
        result['discrepancy_rms_all_actions'] = rms(q_values[1] - q_values[2])
        return result, indices
    return probe


def guard(root):
    from popgym_parallel import runner_fingerprint, current_port_fingerprint
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Plan changed')
    if p['script_sha256'] != file_hash(__file__) or p['sources'] != source_fingerprint():
        raise ValueError('Sources changed')
    if p['runner_sha256'] != runner_fingerprint() or p['port_sha256'] != current_port_fingerprint():
        raise ValueError('Runner/environment changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Packages changed')
    sweep = json.loads(Path(p['sweep_plan']).read_text()); verify_plan(sweep)
    if file_hash(p['sweep_plan']) != p['sweep_plan_sha256']:
        raise ValueError('Sweep plan changed')
    return p, sweep


def prepare(root):
    from popgym_parallel import runner_fingerprint, current_port_fingerprint
    sweep_path = Path('/tmp/opencode/popgym-alpha-followup/plan.json')
    sweep = json.loads(sweep_path.read_text()); verify_plan(sweep)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Use a fresh root')
    p = {'created_at': datetime.now().astimezone().isoformat(), 'script_sha256': file_hash(__file__),
        'sources': source_fingerprint(), 'runner_sha256': runner_fingerprint(), 'port_sha256': current_port_fingerprint(),
        'sweep_plan': str(sweep_path), 'sweep_plan_sha256': file_hash(sweep_path),
        'dependencies': {n: file_hash(Path(__file__).with_name(n)) for n in
            ('popgym_critic_aligned.py', 'popgym_critic_settling.py', 'popgym_gradient_balance.py', 'popgym_alpha_sweep.py')},
        'versions': {n: importlib.metadata.version(n) for n in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym')},
        'jobs': [{'arm': arm, 'seed': seed} for seed in range(42, 47) for arm in ('baseline', 'aligned-q')],
        'steps': 250000, 'alpha': .01, 'diagnostic_minibatches': 2,
        'wandb_project': 'lambda-popgym-online-alignment', 'wandb_entity': sweep['wandb_entity'],
        'protocol': '10 fresh Battleship Easy lambda fits, original production sweep settings and random prefill, five paired seeds42-46, baseline vs Q-only bootstrap feature alignment. Main and both auxiliary TARGET Q heads use online features; target actor logits/ratios stay on ORIGINAL target features. Regression, actor, discrepancy, optimizer/RNG/EMA cadence unchanged. Experiment-local cloned closures only; no production/live worker edits. No oracle-Q claim. Diagnostics/checkpoint snapshots every25k; original evaluation/final holdouts/W&B persistence. Independent serial workers share GPU0 without pausing existing work.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes()); render(root)


def worker(root, index):
    import cloudpickle
    import jax
    import numpy as np
    from unittest.mock import patch
    import popgym_parallel as parallel
    import popgym_jax as backend
    import popgym_critic_settling as base
    from lambda_imitation.buffer import create_buffer
    from popgym_suite import run_config
    p, sweep = guard(root); job = p['jobs'][index]; arm, seed = job['arm'], job['seed']
    (root / 'progress').mkdir(exist_ok=True)
    args = args_from_plan(sweep)
    args.alpha = p['alpha']; args.steps = p['steps']; args.output_dir = root / arm
    args.resume = False; args.wandb_project = p['wandb_project']; args.wandb_entity = p['wandb_entity']
    task = {'env_id': 'popgym-BattleshipEasy-v0', 'method': 'lambda', 'backend': 'jax',
        'seed': seed, 'target_steps': p['steps'], 'expected_source_sha256': p['sources'],
        'bootstrap_alignment_arm': arm, 'bootstrap_alignment_source_sha256': p['script_sha256']}
    cache = {}; original_result = backend.evaluation_result

    def initialize(args, task, port):
        state, fns, debug = base.build_agent(run_config(args, task), port)
        dim = args.memory_hidden_dim
        if arm == 'aligned-q':
            fns = aligned_functions(fns, dim)
        probe = make_probe(fns, debug, dim, arm == 'aligned-q')
        _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
        runner = backend.JaxRunner(args, port, fns, debug, bfns, dim, True)
        evaluate = runner.evaluate
        def wrapped(state, eval_seed, mode, n_episodes):
            outputs = evaluate(state, eval_seed, mode, n_episodes)
            if mode == 'stochastic':
                step = int(state.update_step)
                measurements = []
                folder = root / 'diagnostics' / arm / f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
                for batch in range(p['diagnostic_minibatches']):
                    key = jax.random.fold_in(jax.random.PRNGKey(seed + 9000000), step * 2 + batch)
                    stats, indices = jax.device_get(probe(state, key))
                    row = {k: float(v) for k, v in stats.items()}
                    if not all(np.isfinite(v) for v in row.values()):
                        raise FloatingPointError('Nonfinite read-only probe')
                    measurements.append({'metrics': row, 'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})
                cache['metrics'] = {k: float(np.mean([r['metrics'][k] for r in measurements])) for k in measurements[0]['metrics']}
                cache['steps'] = step
                report = folder / f'step{step}.json'
                content = {'step': step, 'arm': arm, 'seed': seed, 'samples': measurements}
                if report.exists():
                    if json.loads(report.read_text()) != content:
                        raise ValueError('Repeated final probe differs')
                else:
                    write_json(report, content)
                    checkpoint = folder / f'step{step}.pkl'; tmp = checkpoint.with_suffix('.tmp')
                    with tmp.open('wb') as file:
                        cloudpickle.dump({'state': jax.device_get(state), 'steps': step,
                            'config': run_config(args, task), 'plan_sha256': file_hash(root / 'plan.json')}, file)
                    tmp.replace(checkpoint)
                guard(root)
            return outputs
        runner.evaluate = wrapped
        return state, runner.initial_rollout(seed), runner

    def result(outputs, mode, args, approximate_lambda):
        value = original_result(outputs, mode, args, approximate_lambda)
        if mode == 'stochastic':
            value['metrics'].update({'diagnostics/' + k: v for k, v in cache['metrics'].items()})
            write_json(root / 'progress' / f'{arm}-seed{seed}.json', {'arm': arm, 'seed': seed,
                'state': 'running', 'steps': cache['steps'], 'metrics': value['metrics']})
            render(root)
        return value

    with patch.object(parallel, 'initialize', initialize), patch.object(backend, 'evaluation_result', result):
        parallel.train_group(args, [task], p['runner_sha256'])
    progress = root / 'progress' / f'{arm}-seed{seed}.json'
    write_json(progress, {**json.loads(progress.read_text()), 'state': 'complete'})
    guard(root)


def render(root):
    import html
    web = Path('/tmp/opencode/popgym-report-web/online-alignment'); web.mkdir(parents=True, exist_ok=True)
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Online Battleship Q-input alignment</h1><p>Five fresh matched baseline/aligned seeds. Main and auxiliary bootstrap Q inputs align; original target actor/ratios and EMA cadence remain unchanged. No oracle-Q claim.</p><table border="1"><tr><th>Arm/seed</th><th>Status</th><th>Steps</th><th>Stochastic return</th><th>Q-vs-MC RMSE</th></tr>'
    for path in sorted(root.glob('progress/*.json')):
        r = json.loads(path.read_text()); m = r['metrics']
        text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
            (f"{r['arm']}/seed{r['seed']}", r['state'], r['steps'], m.get('return_mean', ''), m.get('q_mc_rmse/all', ''))) + '</tr>'
    tmp = web / 'index.tmp'; tmp.write_text(text + '</table>'); tmp.replace(web / 'index.html')


def supervise(root):
    p, _ = guard(root)
    for index, job in enumerate(p['jobs']):
        guard(root); write_json(root / 'status.json', {'state': 'running', 'completed': index, 'total': len(p['jobs']), 'job': job}); render(root)
        with (root / f"{job['arm']}-seed{job['seed']}.log").open('x') as file:
            subprocess.run([sys.executable, __file__, '--root', str(root), '--worker', str(index)],
                stdout=file, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        render(root)
    guard(root); write_json(root / 'status.json', {'state': 'complete', 'finished_at': datetime.now().astimezone().isoformat()}); render(root)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true'); p.add_argument('--worker', type=int)
    a = p.parse_args(); root = a.root.resolve()
    if a.prepare:
        prepare(root)
    else:
        try:
            worker(root, a.worker) if a.worker is not None else supervise(root)
        except BaseException:
            import traceback
            write_json(root / 'status.json', {'state': 'failed', 'error': traceback.format_exc()}); render(root); raise


if __name__ == '__main__':
    main()
