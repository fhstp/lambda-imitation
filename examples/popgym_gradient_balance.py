"""Read-only production FE gradient audit on immutable alpha-sweep snapshots.

Separates main critic, lambda1/2 regression and discrepancy on identical replay
minibatches. Reports raw and weighted gradients and a same-Adam-state one-step
counterfactual. No parameter/optimizer update is applied and no training changes.
"""

import argparse
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import popgym_critic_settling as base
from popgym_alpha_sweep import file_hash
from popgym_suite import write_json


LABELS = ('main_critic', 'lambda1_regression', 'lambda2_regression', 'discrepancy_unweighted')


def captures(fn):
    fn = inspect.unwrap(fn)
    result = {}
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            result[name] = cell.cell_contents
        except ValueError:
            pass
    return result


def objective_adapter(fns):
    combined = captures(fns.update)['loss_combined']
    h = captures(combined)
    hp = h['params']
    if not hp.stop_actor_fe or not h['approximate_lambda'] or h['use_gvd'] or h['burn_in_from_stored_carry'] or h['mask_fn'] is not None:
        raise ValueError('Audit requires the production lambda sweep configuration')
    return combined, h


def make_probe(fns, debug, carry_dim):
    import jax
    import jax.numpy as jnp
    import optax
    combined, h = objective_adapter(fns)
    hp = h['params']
    flatten = lambda a: a.reshape(-1, *a.shape[2:])
    def components(fe, state, raw, keys):
        latent, target = debug.calculate_latent(fe, state.feature_extractor_target,
            raw[h['obs_key']], raw[h['action_key']], raw['terminated'],
            jnp.zeros((hp.online_batch_size, carry_dim), jnp.float32))
        tm = lambda name: jnp.swapaxes(raw[name], 0, 1)[hp.burn_in_length:]
        actions, rewards, done, behaviour = (tm(k) for k in (h['action_key'], 'rewards', 'terminated', 'behaviour_weight'))
        main = h['loss_critic'](state.actor_target, state.critic, state.critic_target,
            flatten(latent[:-1]), flatten(target[1:]), flatten(actions[:-1]),
            rewards[:-1].reshape(-1), done[:-1].reshape(-1), state.alpha, keys[2])[0]
        trace = h['loss_vtrace_lambda_sequence']
        l1 = trace(state.actor_target, state.lambda1_critic, state.lambda1_critic_target,
            h['lambda1_critic_graph'], hp.lambda1, actions, rewards, done, behaviour, latent, target, keys[3])[0]
        l2 = trace(state.actor_target, state.lambda2_critic, state.lambda2_critic_target,
            h['lambda2_critic_graph'], hp.lambda2, actions, rewards, done, behaviour, latent, target, keys[4])[0]
        ld = h['loss_ld'](state.lambda1_critic, state.lambda2_critic,
            h['lambda1_critic_graph'], h['lambda2_critic_graph'], flatten(latent[:-1]), flatten(actions[:-1]))[0]
        return jnp.stack((main, l1, l2, ld))

    def differential(state, key):
        keys = jax.random.split(key, 5)
        sample, indices = h['online_buffer_lambda_sample'](state.online_buffer, keys[0])
        values, pullback = jax.vjp(lambda fe: components(fe, state, sample.this_info, keys), state.feature_extractor)
        grads = jax.vmap(lambda cotangent: pullback(cotangent)[0])(jnp.eye(4, dtype=values.dtype))
        return values, grads, indices

    @jax.jit
    def probe(state, key):
        values, grads, indices = differential(state, key)
        grams = {'all': jnp.zeros((4, 4))}
        for path, array in jax.tree_util.tree_flatten_with_path(grads)[0]:
            matrix = array.reshape(4, -1)
            gram = matrix @ matrix.T
            group = str(getattr(path[0], 'key', path[0]))
            grams['all'] += gram
            grams[group] = grams.get(group, jnp.zeros((4, 4))) + gram
        weights = jnp.array([1., 1., 1., hp.lambda_coef])
        total_grad = jax.tree.map(lambda a: jnp.einsum('i,i...->...', weights, a), grads)
        other_grad = jax.tree.map(lambda a: a[:3].sum(0), grads)
        optimizer = optax.adam(hp.fe_lr)
        updates, _ = optimizer.update(total_grad, state.fe_optimizer_state)
        without, _ = optimizer.update(other_grad, state.fe_optimizer_state)
        dot = lambda a, b: sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        norm = lambda a: jnp.sqrt(dot(a, a))
        difference = jax.tree.map(lambda a, b: a - b, updates, without)
        u, v = norm(updates), norm(without)
        return {'losses': values, 'grams': grams, 'indices': indices,
                'adam': {'step_norm': u, 'step_without_discrepancy_norm': v,
                         'step_change_norm': norm(difference), 'step_dot': dot(updates, without)}}

    # Test-only oracle: the sum must match the production joint FE gradient.
    @jax.jit
    def parity(state, key):
        _, grads, _ = differential(state, key)
        weights = jnp.array([1., 1., 1., hp.lambda_coef])
        summed = jax.tree.map(lambda a: jnp.einsum('i,i...->...', weights, a), grads)
        def whole(fe):
            return combined(fe, state.actor, state.critic, state.lambda1_critic, state.lambda2_critic,
                state.gvd_sf1, state.gvd_sf2, state.feature_extractor_target, state.actor_target,
                state.critic_target, state.lambda1_critic_target, state.lambda2_critic_target,
                state.gvd_sf1_target, state.gvd_sf2_target, state.alpha, state.online_buffer, key)[0]
        actual = jax.grad(whole)(state.feature_extractor)
        return summed, actual
    return probe, parity


def summarize_sample(result, coefficient):
    import numpy as np
    if not np.isfinite(result['losses']).all() or not all(np.isfinite(a).all() for a in result['grams'].values()):
        raise FloatingPointError('Nonfinite gradient audit')
    ratio = lambda a, b: float(a / b) if b > 0 else None
    groups = {}
    for name, matrix in result['grams'].items():
        g = np.asarray(matrix, dtype=np.float64)
        norms = np.sqrt(np.maximum(0., np.diag(g)))
        other = np.sqrt(max(0., g[:3, :3].sum()))
        weighted = coefficient * norms[3]
        projection = coefficient * g[3, :3].sum()
        groups[name] = {'norms': dict(zip(LABELS, map(float, norms))),
            'weighted_discrepancy_norm': float(weighted), 'other_objectives_sum_norm': float(other),
            'weighted_discrepancy_over_main': ratio(weighted, norms[0]),
            'weighted_discrepancy_over_other_sum': ratio(weighted, other),
            'cosine_discrepancy_other_sum': ratio(projection, weighted * other),
            'discrepancy_projection_over_other_sum': ratio(projection, other ** 2),
            'cosine_discrepancy_main': ratio(g[3, 0], norms[3] * norms[0]),
            'component_cosines': [[ratio(g[i, j], norms[i] * norms[j]) for j in range(4)] for i in range(4)],
            'gram': g.tolist()}
    adam = {k: float(v) for k, v in result['adam'].items()}
    if not all(np.isfinite(v) for v in adam.values()):
        raise FloatingPointError('Nonfinite counterfactual Adam step')
    adam['relative_step_change'] = ratio(adam['step_change_norm'], adam['step_norm'])
    adam['step_cosine_without_discrepancy'] = ratio(adam['step_dot'], adam['step_norm'] * adam['step_without_discrepancy_norm'])
    return {'losses': dict(zip(LABELS, map(float, result['losses']))), 'groups': groups,
            'adam': adam, 'actor_fe_gradient_norm': 0.}


def gradient_parity_certificate(summed, actual):
    """Validate whole directions/norms, not relative error of near-zero entries.

Vmap and joint reverse-mode use different native GPU reduction kernels. CPU
tests establish component wiring; the real-device check limits group L2 error
to 1% and cosine to >=.99995, with a tiny absolute floor for zero gradients.
The unrounded measurements are saved even if the check fails.
"""
    import jax
    import numpy as np
    groups = {}
    maximum = 0.
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(summed)[0], jax.tree.leaves(actual)):
        a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise FloatingPointError('Nonfinite production gradient parity')
        maximum = max(maximum, float(np.max(np.abs(a - b))))
        group = str(getattr(path[0], 'key', path[0]))
        for name in ('all', group):
            totals = groups.setdefault(name, np.zeros(4))
            totals += [np.dot(a - b, a - b), np.dot(a, a), np.dot(b, b), np.dot(a, b)]
    output = {}
    for name, (err2, sum2, ref2, dot) in groups.items():
        error, norm, summed_norm = np.sqrt(err2), np.sqrt(ref2), np.sqrt(sum2)
        relative = float(error / norm) if norm else None
        cosine = float(dot / (norm * summed_norm)) if norm and summed_norm else None
        passed = bool(error <= 1e-7 or (relative is not None and relative <= .01 and cosine is not None and cosine >= .99995))
        output[name] = {'relative_l2_error': relative, 'gradient_cosine': cosine,
                        'absolute_l2_error': float(error), 'reference_norm': float(norm), 'passed': passed}
    return {'groups': output, 'max_absolute_element_error': maximum,
            'passed': all(row['passed'] for row in output.values()),
            'criterion': 'Group relative L2 <=1% and cosine >=.99995, or absolute L2 <=1e-7'}


def prepare(root, source_root):
    from popgym_suite import source_fingerprint
    from popgym_parallel import current_port_fingerprint
    import importlib.metadata
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Use a separate audit root')
    jobs = []
    for env in base.ENVS:
        checkpoints = []
        for branch in ('alpha0p1', 'alpha0p01', 'alpha0p001'):
            for seed in range(42, 47):
                folder = source_root / branch / env / 'lambda' / f'seed{seed}'
                status = json.loads((folder / 'status.json').read_text())
                provenance = json.loads((folder / 'config.json').read_text())
                if status['state'] != 'complete' or provenance['source_sha256'] != source_fingerprint():
                    raise ValueError('Need completed source-matched checkpoints')
                path = folder / 'checkpoint.pkl'
                checkpoints.append({'checkpoint': str(path.resolve()), 'sha256': file_hash(path),
                                    'config': provenance['experiment']})
        jobs.append({'env_id': env, 'checkpoints': checkpoints})
    plan = {'created_at': datetime.now().astimezone().isoformat(), 'jobs': jobs, 'minibatches': 8,
            'source_sha256': source_fingerprint(), 'script_sha256': file_hash(__file__),
            'port_sha256': current_port_fingerprint(),
            'versions': {n: importlib.metadata.version(n) for n in ('jax', 'flax', 'optax', 'popjym')},
            'builder_sha256': file_hash(base.__file__), 'coefficient': .1,
            'protocol': 'Read-only final production snapshots: two tasks, three alphas, five seeds. Eight full production-sized replay minibatches per checkpoint, same keys across alphas. No encoder alignment or refitting. Raw FE gradients separated; projection and GRU separately. Counterfactual beta=0 Adam step retains the saved optimizer state and is never applied. No claim about an entire beta=0 training run.'}
    write_json(root / 'plan.json', plan)


def verify(root):
    from popgym_suite import source_fingerprint
    from popgym_parallel import current_port_fingerprint
    import importlib.metadata
    p = json.loads((root / 'plan.json').read_text())
    if p['script_sha256'] != file_hash(__file__) or p['builder_sha256'] != file_hash(base.__file__) or p['source_sha256'] != source_fingerprint():
        raise ValueError('Audit or production source changed')
    if p['port_sha256'] != current_port_fingerprint() or any(importlib.metadata.version(n) != v for n, v in p['versions'].items()):
        raise ValueError('Environment/package versions changed')
    for job in p['jobs']:
        for item in job['checkpoints']:
            if item['sha256'] != file_hash(item['checkpoint']):
                raise ValueError('Source checkpoint changed')
    return p


def worker(root, index):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_jax import PortEnv
    plan = verify(root)
    job = plan['jobs'][index]
    cfg = job['checkpoints'][0]['config']
    _, fns, debug = base.build_agent(cfg, PortEnv(job['env_id']))
    carry_dim = cfg['memory_hidden_dim'] * (2 if cfg['memory_type'] == 'lstm' else 1)
    probe, parity = make_probe(fns, debug, carry_dim)
    folder = root / job['env_id']
    folder.mkdir(exist_ok=True)
    for item in job['checkpoints']:
        config = item['config']
        if any(config[k] != v for k, v in cfg.items() if k not in ('alpha', 'seed')):
            raise ValueError('Architecture or hyperparameters differ across audit snapshots')
        saved = cloudpickle.loads(Path(item['checkpoint']).read_bytes())
        if saved['config'] != config or config['lambda_coef'] != plan['coefficient']:
            raise ValueError('Checkpoint configuration mismatch')
        state = jax.tree.map(jnp.asarray, saved['state'])
        before = base.array_hash(state)
        samples, indices = [], []
        for batch in range(plan['minibatches']):
            key = jax.random.fold_in(jax.random.PRNGKey(config['seed'] + 8000000), batch)
            out = jax.device_get(probe(state, key))
            samples.append(summarize_sample(out, plan['coefficient']))
            indices.append(out['indices'])
        # Validate the decomposition against the complete production objective
        # at the first snapshot of each environment, on the same exact batch.
        if item is job['checkpoints'][0]:
            a, b = jax.device_get(parity(state, jax.random.fold_in(jax.random.PRNGKey(config['seed'] + 8000000), 0)))
            certificate = gradient_parity_certificate(a, b)
            write_json(folder / 'production-gradient-parity.json', certificate)
            if not certificate['passed']:
                np.savez_compressed(folder / 'parity-failed-gradients.npz',
                    **{f'{name}{i}': array for name, tree in (('summed', a), ('actual', b))
                       for i, array in enumerate(jax.tree.leaves(tree))})
                raise AssertionError(f'Production FE gradient parity failed: {certificate}')
        if base.array_hash(state) != before:
            raise AssertionError('Read-only audit mutated checkpoint state')
        result = {'env_id': job['env_id'], 'alpha': config['alpha'], 'seed': config['seed'],
                  'source': item, 'samples': samples, 'state_unchanged': True,
                  'coefficient': plan['coefficient'], 'actor_fe_gradient_stopped': True}
        name = f"alpha{config['alpha']}-seed{config['seed']}"
        write_json(folder / f'{name}.json', result)
        np.savez_compressed(folder / f'{name}-indices.npz', indices=np.asarray(indices))
        print('AUDITED', job['env_id'], config['alpha'], config['seed'], flush=True)
        del state, saved


def render(root, web=None):
    import html
    import statistics
    web = Path('/tmp/opencode/popgym-report-web/gradient-balance') if web is None else Path(web)
    web.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in root.glob('popgym-*/alpha*-seed*.json'):
        r = json.loads(p.read_text())
        groups.setdefault((r['env_id'], r['alpha']), []).append(r)
    parts = ['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Gradient balance</title>',
        '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse}td,th{padding:.6rem;border:1px solid #ccc}</style>',
        '<h1>Production gradient balance · discrepancy coefficient 0.1</h1><p><a href="/alpha-sweep/">Alpha sweep</a></p>',
        '<p>Read-only final snapshots, eight production-sized minibatches per seed. Actor gradient into memory is zero. Ratios are raw gradients; Adam-step counterfactual uses saved moments and is not applied.</p>',
        '<table><tr><th>Environment</th><th>α</th><th>Seeds</th><th>Weighted discrepancy / main</th><th>Weighted discrepancy / other sum</th><th>Cosine with other sum</th><th>Relative Adam step change without discrepancy</th></tr>']
    summary = []
    def median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None
    fmt = lambda v: f'{v:.3g}' if v is not None else 'undefined (zero denominator)'
    for (env, alpha), rows in sorted(groups.items()):
        fields = ('weighted_discrepancy_over_main', 'weighted_discrepancy_over_other_sum', 'cosine_discrepancy_other_sum')
        metric = lambda r, k: median(x['groups']['all'][k] for x in r['samples'])
        vals = {k: median(metric(r, k) for r in rows) for k in fields}
        adam = median(median(x['adam']['relative_step_change'] for x in r['samples']) for r in rows)
        summary.append({'env_id': env, 'alpha': alpha, 'seeds': len(rows), 'median_of_seed_medians': {**vals, 'relative_adam_step_change': adam}})
        parts.append('<tr>' + ''.join(f'<td>{x}</td>' for x in (html.escape(env), alpha, len(rows), *(fmt(vals[k]) for k in fields), fmt(adam))) + '</tr>')
    parts.append('</table><p>Norm dominance is not evidence of harmful or useful information. Large ratios can reflect cancellation of other gradients. Final-checkpoint measurements do not describe earlier training.</p>')
    tmp = web / 'index.tmp'
    tmp.write_text(''.join(parts)); tmp.replace(web / 'index.html')
    write_json(root / 'summary.json', summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('/tmp/opencode/popgym-alpha-followup'))
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--worker', type=int)
    args = p.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root, args.source_root.resolve())
    elif args.worker is not None:
        worker(root, args.worker)
    else:
        verify(root)
        write_json(root / 'status.json', {'state': 'running', 'sweep_paused': False})
        render(root)
        # Share two GPUs with the sweep; no pause and no new RL training.
        processes = []
        for i, gpu in enumerate(('0', '1')):
            log = (root / f'worker{i}.log').open('a')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'JAX_PLATFORMS': 'cuda',
                   'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONUNBUFFERED': '1'}
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', str(i)],
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
            processes.append((proc, log))
        failures = []
        for proc, log in processes:
            code = proc.wait(); log.close()
            if code:
                failures.append(code)
        render(root)
        verify(root)
        write_json(root / 'status.json', {'state': 'failed' if failures else 'complete', 'failures': failures,
                   'sweep_paused': False, 'finished_at': datetime.now().astimezone().isoformat()})
        if failures:
            raise SystemExit(f'Gradient workers failed: {failures}')


if __name__ == '__main__':
    main()
