"""Read-only Battleship repeat-shot and replay-memory reconstruction audit.

Frozen OFN snapshots only. No training, policy intervention, RNG continuation,
memory update, replay rewrite or production/predecessor source modification.
"""

import argparse
from datetime import datetime
from functools import partial
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import popgym_critic_normalization as normalization
from popgym_alpha_sweep import file_hash
from popgym_critic_aligned import replace_captures
from popgym_gradient_balance import captures
from popgym_suite import write_json


DEFAULT_ROOT = Path('/tmp/opencode/popgym-memory-diagnostics')
PREDECESSOR = Path('/tmp/opencode/popgym-critic-normalization')
STEPS = (100000, 200000, 250000)
MODES = ('stochastic', 'greedy', 'random')


def chronological_metadata(buffer, action_dim):
    """Known episode starts only; oldest ring fragment is not assumed complete."""
    import numpy as np
    size, pos = int(buffer.size), int(buffer.pos)
    count = min(size, pos); order = np.arange(pos - count, pos) % size
    done = np.asarray(buffer.info['terminated'])[order].astype(bool)
    actions = np.asarray(buffer.info['actions'])[order, 0].astype(int)
    known_start = pos <= size
    age = 0; seen = np.zeros(action_dim, bool)
    ages, known, visited = [], [], []
    for action, terminal in zip(actions, done):
        ages.append(age); known.append(known_start); visited.append(seen.copy())
        seen[action] = True; age += 1
        if terminal:
            seen[:] = False; age = 0; known_start = True
    inverse = np.full(size, -1, int); inverse[order] = np.arange(count)
    return {'order': order, 'inverse': inverse, 'age': np.asarray(ages),
        'known': np.asarray(known), 'visited': np.asarray(visited)}


def summarize_shots(data):
    import numpy as np
    active = np.asarray(data['active']).astype(bool)
    lengths = active.sum(-1)
    repeat = np.asarray(data['repeat']).astype(bool)
    actions = np.asarray(data['actions'])
    consecutive = np.concatenate((np.zeros((len(actions), 1), bool), actions[:, 1:] == actions[:, :-1]), axis=1)
    result = {'episodes': len(actions), 'return_mean': float(np.mean(np.sum(data['rewards'], -1))),
        'length_mean': float(lengths.mean()), 'repeat_fraction': float(repeat[active].mean()),
        'consecutive_repeat_fraction': float(consecutive[active].mean()),
        'unique_shots_mean': float(np.mean(np.sum(active & ~repeat, -1))),
        'hits_mean': float(np.mean(np.sum(np.asarray(data['hits']) * active, -1))),
        'success_fraction': float(np.mean(np.sum(np.asarray(data['hits']) * active, -1) == 12)),
        'policy_mass_on_previously_fired_mean': float(np.asarray(data['seen_mass'])[active].mean()),
        'policy_entropy_mean': float(np.asarray(data['entropy'])[active].mean())}
    result['by_episode_age'] = {}
    result['distribution_note'] = 'Seen-mass and entropy describe the actor distribution on each mode\'s histories; repeats/unique shots/hits describe the actually executed stochastic/greedy/uniform actions.'
    for start in range(0, actions.shape[1], 16):
        mask = active[:, start:start + 16]
        result['by_episode_age'][f'{start}-{min(start + 15, actions.shape[1] - 1)}'] = {
            'steps': int(mask.sum()),
            'repeat_fraction': float(repeat[:, start:start + 16][mask].mean()) if mask.any() else None,
            'policy_seen_mass': float(np.asarray(data['seen_mass'])[:, start:start + 16][mask].mean()) if mask.any() else None}
    return result


def make_collect(port, fns, debug, dim):
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=('mode', 'episodes'))
    def collect(state, seed, *, mode, episodes):
        def episode(index):
            key = jax.random.PRNGKey(seed + index); key, rk = jax.random.split(key)
            obs, es = port.reset(rk)
            def body(c, _):
                key, obs, es, carry, prev, active, seen = c
                key, ak, ek = jax.random.split(key, 3)
                values, nc = debug.predict_values(state, obs, carry, prev)
                probs = values['probs']; q = values['q'].min(-1)
                if mode == 'random':
                    action = jax.random.randint(ak, (), 0, port.action_dim)
                elif mode == 'greedy':
                    action = jnp.argmax(probs)
                else:
                    action = jax.random.categorical(ak, jnp.log(probs + 1e-30))
                nobs, nes, reward, done = port.transition(ek, es, action)
                entropy = -(probs * jnp.log(probs + 1e-30)).sum()
                gap = values['lambda1_q'][action].min() - values['lambda2_q'][action].min()
                outputs = {'observations': obs, 'actions': action, 'probs': probs,
                    'rewards': jnp.where(active, reward, 0.), 'active': active,
                    'done': active & done, 'q': jnp.where(active, q[action], 0.),
                    'entropy': jnp.where(active, entropy, 0.), 'ld': jnp.where(active, gap, 0.),
                    'repeat': active & seen[action], 'seen_mass': jnp.where(active, (probs * seen).sum(), 0.),
                    'hits': nobs[0], 'history_label_parity': jnp.all(seen == es.guesses.reshape(-1))}
                seen = seen.at[action].set(True)
                return (key, nobs, nes, nc, fns.encode_action(jnp.atleast_1d(action)), active & ~done, seen), outputs
            return jax.lax.scan(body, (key, obs, es, jnp.zeros(dim), jnp.zeros(port.action_dim),
                jnp.bool_(True), jnp.zeros(port.action_dim, bool)), None, length=port.horizon)[1]
        return jax.vmap(episode)(jnp.arange(episodes, dtype=jnp.uint32))
    return collect


def full_encoder(debug):
    """Original FE/reset/prev-action code, with burn=0 only for reference forwards."""
    calculate = debug.calculate_latent
    hp = captures(calculate)['params']
    return replace_captures(calculate, {'params': hp._replace(burn_in_length=0)})


def make_comparison(fns, debug):
    import jax
    import jax.numpy as jnp
    h = captures(captures(fns.update)['loss_combined'])
    main = captures(h['loss_critic']); run_q = captures(main['get_q_both'])['run_critic']
    actor = captures(debug.get_entropy)['run_actor']
    graphs = (main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph'])

    @jax.jit
    def compare(state, short_x, short_y, full_x, full_y, visited, actions):
        shape = short_x.shape[:2]; flat = lambda z: z.reshape(-1, z.shape[-1])
        metrics = {}
        for label, sx, fx, actor_state in (('online', short_x, full_x, state.actor),
                ('target', short_y, full_y, state.actor_target)):
            delta = sx - fx
            metrics[label + '_feature_l2'] = jnp.linalg.norm(delta, axis=-1)
            metrics[label + '_feature_relative_l2'] = jnp.linalg.norm(delta, axis=-1) / jnp.maximum(jnp.linalg.norm(fx, axis=-1), 1e-8)
            sp = jax.nn.softmax(actor(actor_state, flat(sx))).reshape((*shape, -1))
            fp = jax.nn.softmax(actor(actor_state, flat(fx))).reshape((*shape, -1))
            metrics[label + '_policy_l1'] = jnp.sum(jnp.abs(sp - fp), -1)
            metrics[label + '_greedy_action_disagreement'] = jnp.argmax(sp, -1) != jnp.argmax(fp, -1)
            metrics[label + '_short_seen_mass'] = jnp.sum(sp * visited, -1)
            metrics[label + '_full_seen_mass'] = jnp.sum(fp * visited, -1)
            metrics[label + '_taken_prob_abs_difference'] = jnp.abs(jnp.take_along_axis(sp - fp, actions[..., None], -1)[..., 0])
            for name, attr, graph in zip(('main', 'lambda1', 'lambda2'),
                    ('critic', 'lambda1_critic', 'lambda2_critic'), graphs):
                critic = getattr(state, attr + ('_target' if label == 'target' else ''))
                sq = run_q(critic, graph, flat(sx)).reshape((*shape, -1, 2)).min(-1)
                fq = run_q(critic, graph, flat(fx)).reshape((*shape, -1, 2)).min(-1)
                metrics[label + '_' + name + '_q_rmse_all_actions'] = jnp.sqrt(jnp.mean((sq - fq) ** 2, -1))
        return metrics
    return compare


def replay_audit(state, fns, debug, dim, batches, key):
    import jax
    # Different reference/minibatch shapes select different native GPU matmuls.
    # Isolate history loss from that roundoff; actual policy rollouts stay native.
    with jax.default_matmul_precision('float32'):
        return _replay_audit(state, fns, debug, dim, batches, key)


def _replay_audit(state, fns, debug, dim, batches, key):
    import jax
    import jax.numpy as jnp
    import numpy as np
    hp = captures(captures(fns.update)['loss_combined'])['params']
    sampler = captures(captures(fns.update)['loss_combined'])['online_buffer_lambda_sample']
    action_dim = captures(debug.get_entropy)['run_actor'](state.actor, jnp.zeros((1, dim))).shape[-1]
    metadata = chronological_metadata(state.online_buffer, action_dim)
    order = metadata['order']; raw = {k: v[order] for k, v in state.online_buffer.info.items()}
    encode = jax.jit(full_encoder(debug))
    fx, fy = encode(state.feature_extractor, state.feature_extractor_target,
        raw['observations'][None], raw['actions'][None], raw['terminated'][None], jnp.zeros((1, dim)))
    fx, fy = fx[:, 0], fy[:, 0]
    compare = make_comparison(fns, debug); pieces = []; labels = []; indices_hashes = []
    certificates = []; offset = np.arange(hp.sequence_length)[None, :]
    for batch in range(batches):
        sample, indices = sampler(state.online_buffer, jax.random.fold_in(key, batch))
        r = sample.this_info
        sx, sy = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
            r['observations'], r['actions'], r['terminated'], jnp.zeros((hp.online_batch_size, dim)))
        positions = metadata['inverse'][np.asarray(indices)]
        if np.any(positions < 0) or np.any(np.diff(positions, axis=1) != 1):
            raise AssertionError('Production sample crosses unknown/chronological ring boundary')
        learning = positions[:, hp.burn_in_length:hp.burn_in_length + hp.sequence_length]
        known = metadata['known'][learning]
        seen = metadata['visited'][learning].swapaxes(0, 1)
        actions = np.asarray(r['actions'])[:, hp.burn_in_length:hp.burn_in_length + hp.sequence_length, 0].astype(int).T
        full_x, full_y = fx[learning.T], fy[learning.T]
        result = jax.device_get(compare(state, sx[:hp.sequence_length], sy[:hp.sequence_length], full_x, full_y,
            jnp.asarray(seen), jnp.asarray(actions)))
        # Missing prefix can affect only the current episode until a reset.
        ages = metadata['age'][learning]
        missing = np.maximum(ages - (hp.burn_in_length + offset), 0)
        reset_covered = known & (missing == 0)
        if reset_covered.any():
            for label, a, b in (('online', np.asarray(sx[:hp.sequence_length]).swapaxes(0, 1), np.asarray(full_x).swapaxes(0, 1)),
                    ('target', np.asarray(sy[:hp.sequence_length]).swapaxes(0, 1), np.asarray(full_y).swapaxes(0, 1))):
                delta = a[reset_covered] - b[reset_covered]
                relative = np.linalg.norm(delta) / max(np.linalg.norm(b[reset_covered]), 1e-8)
                if relative > 1e-4:
                    raise AssertionError(f'Known episode-start/reset reference fails feature parity: {label}, relative L2={relative}, positions={reset_covered.sum()}')
                certificates.append({'batch': batch, 'encoder': label, 'reset_covered_feature_relative_l2': float(relative)})
        pieces.append({k: np.asarray(v).T for k, v in result.items()})
        labels.append({'known': known, 'missing': missing, 'age': ages})
        indices_hashes.append(hashlib.sha256(np.asarray(indices).tobytes()).hexdigest())
    data = {k: np.concatenate([r[k] for r in pieces], axis=0) for k in pieces[0]}
    meta = {k: np.concatenate([r[k] for r in labels], axis=0) for k in labels[0]}
    masks = {'all_known': meta['known'], 'missing_prefix': meta['known'] & (meta['missing'] > 0),
        'episode_start_or_reset_covered': meta['known'] & (meta['missing'] == 0)}
    for lo in (0, 16, 32, 48):
        masks[f'age{lo}-{lo + 15}'] = meta['known'] & (meta['age'] >= lo) & (meta['age'] < lo + 16)
    groups = {}
    for name, mask in masks.items():
        groups[name] = {'positions': int(mask.sum()), 'metrics': {k: float(v[mask].mean()) for k, v in data.items()} if mask.any() else {}}
    return {'groups': groups, 'known_position_fraction': float(meta['known'].mean()),
        'missing_prefix_fraction_among_known': float((meta['missing'][meta['known']] > 0).mean()),
        'missing_prefix_steps_mean_when_missing': float(meta['missing'][masks['missing_prefix']].mean()) if masks['missing_prefix'].any() else 0.,
        'indices_sha256': indices_hashes, 'reference_reset_parity': certificates,
        'note': 'Frozen CURRENT checkpoint encoder reconstruction of original stored histories, not stored historical acting carry. Diagnostic-only float32 matmul in both short/full forwards isolates history loss from batch-shape-dependent native CUDA roundoff. Actual frozen policy rollouts use native precision. No neural/oracle value accuracy claim; short/full output differences only.'}


def tree_digest(state):
    import jax
    import numpy as np
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(jax.device_get(state)):
        array = np.asarray(leaf); digest.update(str((array.shape, array.dtype)).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256'] or file_hash(__file__) != p['script_sha256']:
        raise ValueError('Memory diagnostic script/plan changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Diagnostic dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Diagnostic packages changed')
    normalization.guard(Path(p['preceding_root']))
    if file_hash(Path(p['preceding_root']) / 'plan.json') != p['preceding_plan_sha256']:
        raise ValueError('Normalization plan changed')
    return p


def prepare(root):
    old, _ = normalization.guard(PREDECESSOR)
    if json.loads((PREDECESSOR / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Normalization study must be complete')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Fresh diagnostic root required')
    jobs = []
    for step in STEPS:
        for seed in range(42, 47):
            snapshot = PREDECESSOR / 'diagnostics/ofn' / f'seed{seed}' / f'step{step}.pkl'
            jobs.append({'seed': seed, 'step': step, 'snapshot': str(snapshot), 'snapshot_sha256': file_hash(snapshot)})
    p = {'created_at': datetime.now().astimezone().isoformat(), 'script_sha256': file_hash(__file__),
        'preceding_root': str(PREDECESSOR), 'preceding_plan_sha256': file_hash(PREDECESSOR / 'plan.json'),
        'dependencies': {name: file_hash(Path(__file__).with_name(name)) for name in ('popgym_critic_normalization.py',
            'popgym_online_alignment.py', 'popgym_critic_aligned.py', 'popgym_gradient_balance.py', 'popgym_jax.py')},
        'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym')},
        'jobs': jobs, 'episodes': 100, 'replay_batches': 8, 'modes': list(MODES),
        'protocol': 'Read-only OFN snapshots100k/200k/250k x five seeds42-46.100 frozen-policy episodes per stochastic/greedy/uniform mode with independent seed17000000+seed, matched boards across snapshots/modes. Original native forward/RNG/environment construction; unmasked, no intervention. Record repeated/consecutive shots, unique cells,hits,success,actor entropy/mass on fired cells. Ground-truth fired labels are DIAGNOSTICS ONLY, never policy inputs/masks. Compare eight ORIGINAL production replay minibatches against CURRENT frozen online/target encoders reconstructed from known episode starts over chronological replay. Replay comparison alone uses float32 matmul to remove batch-shape native rounding confounds; native original-policy rollouts unchanged. Same observed histories/actions/reset/prev-action rules; no historical-carry or oracle fixed-point claim. Compare short/full features, actor probabilities/greedy actions and critic predictions, stratify missing history/age. Preserve all inputs, original state/replay/RNG/guarded source hashes; per-case input state digest parity. No training, updated states, long-run extension, automatic interventions, optimizer or W&B continuation.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(root / 'status.json', {'state': 'prepared', 'completed': 0, 'total': len(jobs)})
    render(root)


def job_name(job):
    return f"seed{job['seed']}-step{job['step']}"


def worker(root, index):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lambda_imitation.buffer import create_buffer
    from popgym_jax import PortEnv, JaxRunner
    p = guard(root); job = p['jobs'][index]; path = Path(job['snapshot'])
    if file_hash(path) != job['snapshot_sha256']:
        raise ValueError('Diagnostic input snapshot changed')
    folder = root / 'cases' / job_name(job); folder.mkdir(parents=True, exist_ok=False)
    saved = cloudpickle.loads(path.read_bytes()); config = saved['config']
    if saved['steps'] != job['step'] or config['seed'] != job['seed'] or config['critic_normalization_arm'] != 'ofn':
        raise ValueError('Wrong saved snapshot identity')
    state = jax.tree.map(jnp.asarray, saved['state']); before = tree_digest(state)
    port = PortEnv(config['env_id']); _, fns, debug = normalization.build_agent(config, port, 'ofn')
    dim = config['memory_hidden_dim']; collect = make_collect(port, fns, debug, dim)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, config['online_buffer_size'], 1, [], [])
    args = argparse.Namespace(**config); runner = JaxRunner(args, port, fns, debug, bfns, dim, True)
    results = {}; seed = 17000000 + job['seed']; parity = {}
    for mode in p['modes']:
        data = jax.device_get(collect(state, seed, mode=mode, episodes=p['episodes']))
        if not np.asarray(data['history_label_parity']).all():
            raise AssertionError('Action-history fired labels differ from environment guesses')
        original = jax.device_get(runner.evaluate(state, seed, mode, p['episodes']))
        names = ('rewards', 'active', 'done', 'q', 'entropy', 'ld')
        differences = {}
        for name, expected in zip(names, original):
            np.testing.assert_allclose(data[name], expected, rtol=1e-4, atol=1e-5)
            differences[name] = float(np.max(np.abs(np.asarray(data[name], float) - np.asarray(expected, float))))
        parity[mode] = differences
        results[mode] = summarize_shots(data)
        np.savez_compressed(folder / f'{mode}-episodes.npz', **data)
    replay = replay_audit(state, fns, debug, dim, p['replay_batches'], jax.random.PRNGKey(19000000 + job['seed'] + job['step']))
    after = tree_digest(state)
    if before != after or file_hash(path) != job['snapshot_sha256']:
        raise AssertionError('Input snapshot/state changed')
    guard(root)
    write_json(folder / 'result.json', {'job': job, 'shots': results, 'replay': replay,
        'certificates': {'state_digest_before': before, 'state_digest_after': after,
            'production_evaluation_parity': parity, 'input_snapshot_hash_unchanged': True}})
    print(job_name(job), results, replay['groups']['all_known'], flush=True)


def render(root):
    web = Path('/tmp/opencode/popgym-report-web/memory-diagnostics'); web.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Frozen OFN Battleship memory diagnostics</h1><p>Read-only repeat-shot behaviour and short-window vs episode-start memory reconstruction. Same saved histories; no training, masking or oracle-Q claims.</p><pre>' + html.escape(json.dumps(status, indent=2)) + '</pre><table border="1"><tr><th>Seed / steps</th><th>Stochastic return</th><th>Repeated shots</th><th>Unique cells</th><th>Random repeats</th><th>Replay policy L1</th><th>Greedy disagreement</th></tr>'
    for path in sorted(root.glob('cases/*/result.json')):
        r = json.loads(path.read_text()); shot = r['shots']['stochastic']; replay = r['replay']['groups']['all_known']['metrics']
        values = (path.parent.name, shot['return_mean'], shot['repeat_fraction'], shot['unique_shots_mean'],
            r['shots']['random']['repeat_fraction'], replay['online_policy_l1'], replay['online_greedy_action_disagreement'])
        text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    if (root / 'findings.md').exists():
        text += '</table><pre>' + html.escape((root / 'findings.md').read_text()) + '</pre>'
    else:
        text += '</table>'
    tmp = web / 'index.tmp'; tmp.write_text(text); tmp.replace(web / 'index.html')


def supervise(root):
    p = guard(root)
    proof = json.loads((root / 'validation.json').read_text())
    if not proof.get('passed') or proof['script_sha256'] != p['script_sha256']:
        raise ValueError('Missing matching read-only CPU/CUDA diagnostic validation')
    for index, job in enumerate(p['jobs']):
        guard(root); write_json(root / 'status.json', {'state': 'running', 'completed': index, 'total': len(p['jobs']), 'job': job}); render(root)
        with (root / (job_name(job) + '.log')).open('x') as file:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', str(index)],
                stdout=file, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        render(root)
    guard(root); write_json(root / 'status.json', {'state': 'complete', 'completed': len(p['jobs']), 'total': len(p['jobs']),
        'finished_at': datetime.now().astimezone().isoformat()}); render(root)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
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
