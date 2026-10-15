"""Frozen all-categorical GRU probes; no training-policy or replay intervention.

Targets are observable history: shots, unique shots, successful hits, fired cells
and revealed-hit cells. They are NOT the hidden/unfired ship board. Episode-level
splits, validation-selected thresholds, and same-age cross-episode shuffles
separate recall from recent-input and clock/prevalence shortcuts.
"""

import argparse
from datetime import datetime
from functools import partial
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import popgym_flash_prototypes as flash
import popgym_memory_diagnostics as memory
from popgym_alpha_sweep import file_hash
from popgym_suite import write_json

DEFAULT_ROOT = Path('/tmp/opencode/popgym-flash-memory-probe')
MODES = ('stochastic', 'greedy', 'random')
COUNTS = ('shots', 'unique_shots', 'hits')
INPUTS = ('memory', 'last_input', 'clock_last_input')


def history_labels(data, cells=64):
    """Labels at age t precede action t and include outcomes of actions < t."""
    import numpy as np
    actions, active, hits = (np.asarray(data[k]) for k in ('actions', 'active', 'hits'))
    n, horizon = actions.shape
    fired = np.zeros((n, horizon + 1, cells), bool)
    revealed = np.zeros_like(fired)
    last_fire = np.full(fired.shape, -1, np.int16)
    valid = np.zeros((n, horizon + 1), bool)
    counts = np.zeros((n, horizon + 1, 3), np.float32)
    for t in range(horizon):
        fired[:, t + 1] = fired[:, t]; revealed[:, t + 1] = revealed[:, t]
        last_fire[:, t + 1] = last_fire[:, t]; counts[:, t + 1] = counts[:, t]
        rows = np.flatnonzero(active[:, t]); a = actions[rows, t].astype(int)
        if np.any((a < 0) | (a >= cells)):
            raise ValueError('Action outside board')
        if np.any(hits[rows, t].astype(bool) & fired[rows, t, a]):
            raise AssertionError('Repeat shot incorrectly labelled as a new hit')
        fired[rows, t + 1, a] = True
        revealed[rows, t + 1, a] |= hits[rows, t].astype(bool)
        last_fire[rows, t + 1, a] = t + 1
        counts[rows, t + 1, 0] += 1
        counts[rows, t + 1, 1] = fired[rows, t + 1].sum(-1)
        counts[rows, t + 1, 2] = revealed[rows, t + 1].sum(-1)
        valid[rows, t + 1] = True  # Includes the first terminal post-observation.
    if np.any(revealed & ~fired):
        raise AssertionError('Revealed hit without a shot')
    return dict(fired=fired, revealed_hits=revealed, counts=counts, valid=valid,
        age=np.broadcast_to(np.arange(horizon + 1), valid.shape), last_fire=last_fire)


def make_encoder(debug, dim, actions):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def encode(state, observations, executed, outcomes):
        # The final outcome is processed even if terminal, before any reset.
        obs = jnp.concatenate((observations, outcomes[:, -1:, None]), 1)
        pa = jnp.concatenate((jnp.zeros((executed.shape[0], 1, actions)),
            jax.nn.one_hot(executed, actions)), 1)
        def episode(o, p):
            def body(c, inputs):
                values, nc = debug.predict_values(state, inputs[0], c, inputs[1])
                return nc, (nc, values['probs'])
            return jax.lax.scan(body, jnp.zeros(dim), (o, p))[1]
        carry, probs = jax.vmap(episode)(obs, pa)
        return carry, probs, jnp.concatenate((obs, pa), -1)
    return encode


def collect_dataset(state, fns, debug, port, dim, config, folder):
    import jax
    import numpy as np
    collect = memory.make_collect(port, fns, debug, dim)
    encode = make_encoder(debug, dim, port.action_dim)
    pieces = []
    parity = []
    for mode_index, mode in enumerate(MODES):
        for block in range(config['blocks']):
            seed = config['dataset_seed'] + mode_index * 100000 + block * 100
            data = jax.device_get(collect(state, seed, mode=mode, episodes=100))
            if not data['history_label_parity'].all():
                raise AssertionError('Environment fired-label parity failed')
            carry, probs, instant = jax.device_get(encode(state, data['observations'], data['actions'], data['hits']))
            np.testing.assert_allclose(probs[:, :-1], data['probs'], atol=1e-5, rtol=1e-4)
            if not np.isfinite(carry).all():
                raise AssertionError('Nonfinite memory')
            labels = history_labels(data, port.action_dim)
            item = dict(memory=carry, last_input=instant, **labels,
                actions=data['actions'], outcomes=data['hits'],
                mode=np.full(100, mode_index, np.int8),
                episode_id=np.arange(seed, seed + 100, dtype=np.int64),
                split=np.full(100, 0 if block < config['train_blocks'] else
                    1 if block < config['train_blocks'] + config['val_blocks'] else 2, np.int8))
            np.savez_compressed(folder / f'{mode}-block{block}.npz', **item)
            pieces.append(item)
            parity.append(dict(mode=mode, block=block, max_probability_error=float(np.max(np.abs(probs[:, :-1] - data['probs'])))))
            print('COLLECT', mode, block, flush=True)
    merged = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if len(np.unique(merged['episode_id'])) != len(merged['episode_id']):
        raise AssertionError('Episode split leakage')
    return merged, parity


def thresholds(scores, labels):
    """Per-cell validation-only balanced-accuracy threshold; no test tuning."""
    import numpy as np
    grid = np.linspace(.01, .99, 99)
    pos, neg = labels.sum(0), (~labels).sum(0)
    best = np.full(scores.shape[1], .5, np.float32)
    objective = np.full(scores.shape[1], -np.inf)
    supported = (pos > 0) & (neg > 0)
    for value in grid:
        prediction = scores >= value
        balanced = .5 * ((prediction & labels).sum(0) / np.maximum(pos, 1) +
            (~prediction & ~labels).sum(0) / np.maximum(neg, 1))
        improve = supported & (balanced > objective)
        best[improve] = value; objective[improve] = balanced[improve]
    return best


def auc(scores, labels):
    import numpy as np
    y = np.asarray(labels, bool).ravel(); s = np.asarray(scores).ravel()
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return None
    order = np.argsort(s, kind='stable'); sorted_s = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    ranks = np.repeat((starts + 1 + ends) / 2., ends - starts)
    return float((ranks[y[order]].sum() - p * (p + 1) / 2.) / (p * n))


def mask_metrics(scores, labels, threshold, last_fire=None, age=None):
    import numpy as np
    labels = np.asarray(labels, bool); prediction = scores >= threshold
    tp, tn = int((prediction & labels).sum()), int((~prediction & ~labels).sum())
    positive, negative = int(labels.sum()), int((~labels).sum())
    recall = tp / positive if positive else None; specificity = tn / negative if negative else None
    intersection = (prediction & labels).sum(-1); union = (prediction | labels).sum(-1)
    result = dict(positive_cells=positive, negative_cells=negative,
        positive_recall=recall, negative_recall=specificity,
        balanced_accuracy=.5 * (recall + specificity) if recall is not None and specificity is not None else None,
        precision=tp / int(prediction.sum()) if prediction.any() else None,
        micro_auroc=auc(scores, labels), brier=float(np.mean((scores - labels) ** 2)),
        mean_iou=float(np.mean(np.where(union > 0, intersection / np.maximum(union, 1), 1))),
        exact_board_fraction=float(np.all(prediction == labels, -1).mean()))
    if last_fire is not None:
        lag = age[:, None] - last_fire
        result['positive_recall_by_steps_since_last_shot'] = {}
        for low, high in ((0, 0), (1, 3), (4, 7), (8, 15), (16, 31), (32, 64)):
            select = labels & (lag >= low) & (lag <= high) & (last_fire >= 0)
            result['positive_recall_by_steps_since_last_shot'][f'{low}-{high}'] = dict(
                cells=int(select.sum()), recall=float(prediction[select].mean()) if select.any() else None)
    return result


def evaluate(prediction, labels, threshold, last_fire, age):
    import numpy as np
    raw_counts = prediction[:, :3]; truth = labels[:, :3]
    result = {'samples': len(labels), 'counts': {}}
    for i, name in enumerate(COUNTS):
        error = raw_counts[:, i] - truth[:, i]; variance = np.var(truth[:, i])
        result['counts'][name] = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error ** 2))),
            r2=float(1 - np.mean(error ** 2) / variance) if variance > 0 else None,
            rounded_exact_fraction=float((np.rint(raw_counts[:, i]) == truth[:, i]).mean()))
    for offset, name in ((3, 'fired'), (67, 'revealed_hits')):
        result[name] = mask_metrics(prediction[:, offset:offset + 64], labels[:, offset:offset + 64].astype(bool),
            threshold[offset - 3:offset - 3 + 64], last_fire, age)
    return result


def require_disk():
    import shutil
    if shutil.disk_usage('/tmp/opencode').free < 2 * 1024 ** 3:
        raise RuntimeError('Disk below2GiB;stop without deleting artifacts')


def fit_probe(train_x, train_y, val_x, val_y, nonlinear, settings, seed):
    """Independent pure-JAX/Optax probe. No agent parameter enters gradients."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), 1e-4)
    ym, ys = train_y[:, :3].mean(0), np.maximum(train_y[:, :3].std(0), 1.)
    positives = train_y[:, 3:].mean(0)
    # Equal positive/negative contribution when both classes exist. Absent
    # training classes stay unweighted; coverage is reported, never filled.
    wp = np.where((positives > 0) & (positives < 1), .5 / np.maximum(positives, 1e-4), 1.)
    wn = np.where((positives > 0) & (positives < 1), .5 / np.maximum(1 - positives, 1e-4), 1.)
    x = jnp.asarray((train_x - mean) / scale); y = jnp.asarray(train_y)
    vx = jnp.asarray((val_x - mean) / scale); vy = jnp.asarray(val_y)
    key = jax.random.PRNGKey(seed); k1, k2 = jax.random.split(key)
    width = settings['probe_width']; dout = train_y.shape[-1]
    layer = lambda k, a, b: (jax.random.normal(k, (a, b)) * jnp.sqrt(2. / a), jnp.zeros(b))
    params = (layer(k1, x.shape[-1], width), layer(k2, width, dout)) if nonlinear else (layer(k1, x.shape[-1], dout),)
    optimizer = optax.adam(settings['probe_lr']); opt = optimizer.init(params)
    def forward(p, xx):
        for w, b in p[:-1]:
            xx = jax.nn.relu(xx @ w + b)
        return xx @ p[-1][0] + p[-1][1]
    def loss(p, xx, yy):
        out = forward(p, xx)
        count = jnp.mean((out[:, :3] - (yy[:, :3] - ym) / ys) ** 2)
        binary = optax.sigmoid_binary_cross_entropy(out[:, 3:], yy[:, 3:])
        return count + jnp.mean(binary * jnp.where(yy[:, 3:] > 0, wp, wn))
    @jax.jit
    def chunk(p, o, k, xx, yy):
        def step(c, _):
            p, o, k = c; k, sk = jax.random.split(k)
            indices = jax.random.randint(sk, (settings['probe_batch'],), 0, xx.shape[0])
            _, grad = jax.value_and_grad(loss)(p, xx[indices], yy[indices])
            update, o = optimizer.update(grad, o, p)
            return (optax.apply_updates(p, update), o, k), None
        return jax.lax.scan(step, (p, o, k), None, length=settings['probe_chunk'])[0]
    measure = jax.jit(loss); best_loss = float('inf'); best = None; curve = []
    for update in range(0, settings['probe_updates'], settings['probe_chunk']):
        params, opt, key = chunk(params, opt, key, x, y)
        value = float(measure(params, vx, vy))
        if not np.isfinite(value):
            raise AssertionError('Nonfinite probe loss')
        curve.append(dict(updates=update + settings['probe_chunk'], validation_loss=value))
        if value < best_loss:
            best_loss = value; best = jax.device_get(params)
    @jax.jit
    def predict(p, xx):
        out = forward(p, (xx - mean) / scale)
        return jnp.concatenate((out[:, :3] * ys + ym, jax.nn.sigmoid(out[:, 3:])), -1)
    # Fixed bounded shapes prevent unnecessary large inference executables.
    def infer(xx):
        parts = []
        for start in range(0, len(xx), 1024):
            part = xx[start:start + 1024]; padded = np.pad(part, ((0, 1024 - len(part)), (0, 0)))
            parts.append(np.asarray(predict(best, padded))[:len(part)])
        return np.concatenate(parts)
    return infer, dict(params=best, input_mean=mean, input_scale=scale, count_mean=ym,
        count_scale=ys, positive_weights=wp, negative_weights=wn), curve


def run_probes(data, settings, folder):
    import cloudpickle
    import numpy as np
    valid = data['valid']; ep, time = np.nonzero(valid)
    labels = np.concatenate((data['counts'][valid], data['fired'][valid], data['revealed_hits'][valid]), -1).astype(np.float32)
    split = data['split'][ep]; mode = data['mode'][ep]; age = data['age'][valid]
    last_fire = data['last_fire'][valid]
    train, val, test = (split == i for i in range(3))
    if not all(m.any() for m in (train, val, test)):
        raise ValueError('Missing episode split')
    results = {}
    for kind in INPUTS:
        x = data['memory'][valid] if kind == 'memory' else data['last_input'][valid]
        if kind == 'clock_last_input':
            x = np.concatenate((x, age[:, None] / 64.), -1)
        for nonlinear in (False, True):
            require_disk()
            name = f'{kind}-' + ('mlp' if nonlinear else 'linear')
            print('PROBE', name, flush=True)
            infer, saved, curve = fit_probe(x[train], labels[train], x[val], labels[val], nonlinear, settings, 41000 + int(nonlinear))
            threshold = thresholds(infer(x[val])[:, 3:], labels[val, 3:].astype(bool))
            prediction = infer(x[test]); test_y = labels[test]; ta = age[test]; tl = last_fire[test]; tm = mode[test]
            result = dict(validation_curve=curve,
                overall=evaluate(prediction, test_y, threshold, tl, ta), by_mode={}, by_age={})
            for i, mode_name in enumerate(MODES):
                mask = tm == i
                result['by_mode'][mode_name] = evaluate(prediction[mask], test_y[mask], threshold, tl[mask], ta[mask])
            for low, high in ((1, 16), (17, 32), (33, 48), (49, 64)):
                mask = (ta >= low) & (ta <= high)
                result['by_age'][f'{low}-{high}'] = evaluate(prediction[mask], test_y[mask], threshold, tl[mask], ta[mask])
            if kind == 'memory':
                # Shuffle across independent heldout episodes WITHIN mode/age,
                # preserving clock, marginal coverage and policy distribution.
                rng = np.random.default_rng(52000)
                shuffled = x[test].copy()
                for i in range(3):
                    for t in np.unique(ta):
                        rows = np.flatnonzero((tm == i) & (ta == t))
                        shuffled[rows] = shuffled[rng.permutation(rows)]
                result['same_age_mode_shuffled_memory'] = evaluate(infer(shuffled), test_y, threshold, tl, ta)
            saved['threshold'] = threshold
            (folder / f'{name}.pkl').write_bytes(cloudpickle.dumps(saved))
            np.savez_compressed(folder / f'{name}-heldout.npz', prediction=prediction,
                episode_id=data['episode_id'][ep[test]], age=ta, mode=tm, threshold=threshold)
            results[name] = result
    # Honest null: training-only per-mode/per-age cell prevalence and count mean.
    null = np.zeros_like(labels)
    for i in range(3):
        for t in np.unique(age):
            source = train & (mode == i) & (age == t)
            rows = (mode == i) & (age == t)
            if not source.any():
                raise ValueError('No training support for null stratum')
            null[rows] = labels[source].mean(0)
    threshold = thresholds(null[val, 3:], labels[val, 3:].astype(bool))
    results['clock_mode_prevalence'] = dict(overall=evaluate(null[test], labels[test], threshold, last_fire[test], age[test]))
    coverage = dict(episodes_by_split={name: int((data['split'] == i).sum()) for i, name in enumerate(('train', 'validation', 'test'))},
        test_samples=int(test.sum()), train_positive_cells=labels[train, 3:].sum(0).astype(int).tolist(),
        train_negative_cells=(1 - labels[train, 3:]).sum(0).astype(int).tolist(),
        test_positive_cells=labels[test, 3:].sum(0).astype(int).tolist(),
        test_negative_cells=(1 - labels[test, 3:]).sum(0).astype(int).tolist())
    return results, coverage


def prepare(root):
    flash.guard(flash.DEFAULT_ROOT)
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for path in sorted((flash.DEFAULT_ROOT / 'diagnostics/all-categorical').glob('seed*/step*/state.pkl')):
        result = json.loads(path.with_name('result.json').read_text())
        if result['step'] not in flash.RETAIN_STEPS:
            continue
        if file_hash(path) != result['certificates']['snapshot_sha256']:
            raise ValueError('Immutable snapshot certificate mismatch')
        jobs.append(dict(seed=int(path.parent.parent.name.removeprefix('seed')), step=result['step'],
            snapshot=str(path), snapshot_sha256=file_hash(path), result_sha256=file_hash(path.with_name('result.json'))))
    if not jobs:
        raise ValueError('No retained all-categorical snapshots')
    settings = dict(blocks=9, train_blocks=6, val_blocks=1, dataset_seed=31000000,
        probe_updates=2000, probe_chunk=200, probe_batch=512, probe_width=128, probe_lr=1e-3)
    plan = dict(created_at=datetime.now().astimezone().isoformat(), jobs=jobs, settings=settings,
        script_sha256=file_hash(__file__), original_plan_sha256=file_hash(flash.DEFAULT_ROOT / 'plan.json'),
        dependencies={name: file_hash(Path(__file__).with_name(name)) for name in
            ('popgym_flash_prototypes.py', 'popgym_memory_diagnostics.py', 'popgym_jax.py')},
        versions={name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym')},
        scope='Only retained snapshots available at prepare time. No rolling checkpoint race/future automatic extension. '
            'Frozen online GRU post-observation carry alone; labels built from executed action/hit history, never hidden board. '
            'Three native rollout modes;900 episodes each,600 train/100 validation/200 test,entire episodes disjoint. '
            'Linear and one-hidden128 MLP;training-only standardization/class weights;validation-selected update/thresholds. '
            'Baselines:last observation+previous action;clock+last input;clock/mode cell prevalence;same-mode/age shuffled heldout memory. '
            'Counts successful unique hits,shots and unique cells;fired and revealed-hit masks,age and retention-lag slices. '
            'Decodability not causal policy use,no hidden-board inference or original training/state/RNG/W&B/service changes. '
            'Descriptive heldout metrics,not per-transition independent confidence intervals;no automatic probe escalation.')
    write_json(root / 'plan.json', plan); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(root / 'status.json', dict(state='prepared', total=len(jobs), completed=0))


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(__file__) != p['script_sha256'] or file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Probe source/plan changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Pinned probe dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Probe packages changed')
    flash.guard(flash.DEFAULT_ROOT)
    if file_hash(flash.DEFAULT_ROOT / 'plan.json') != p['original_plan_sha256']:
        raise ValueError('Original experiment plan changed')
    return p


def worker(root, index):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    from popgym_jax import PortEnv
    p = guard(root); job = p['jobs'][index]; path = Path(job['snapshot'])
    require_disk()
    if file_hash(path) != job['snapshot_sha256'] or file_hash(path.with_name('result.json')) != job['result_sha256']:
        raise ValueError('Input snapshot/result changed')
    saved = cloudpickle.loads(path.read_bytes()); config = saved['config']
    if saved['steps'] != job['step'] or config['seed'] != job['seed'] or config['flash_arm'] != 'all-categorical':
        raise ValueError('Wrong checkpoint identity')
    state = jax.tree.map(jnp.asarray, saved['state']); before = memory.tree_digest(state)
    port = PortEnv(config['env_id']); _, fns, debug, _, _, context = flash.build(config, port, 'all-categorical')
    dim = context['feature_extractor'].initialize_carry(1).shape[-1]
    folder = root / f"seed{job['seed']}-step{job['step']}"; folder.mkdir(exist_ok=False)
    data, parity = collect_dataset(state, fns, debug, port, dim, p['settings'], folder)
    probes, coverage = run_probes(data, p['settings'], folder)
    after = memory.tree_digest(state)
    if before != after or file_hash(path) != job['snapshot_sha256'] or file_hash(path.with_name('result.json')) != job['result_sha256']:
        raise AssertionError('Read-only state/input certificate failed')
    guard(root)
    write_json(folder / 'result.json', dict(job=job, alpha=float(state.alpha), probes=probes, coverage=coverage,
        certificates=dict(state_digest_before=before, state_digest_after=after,
            snapshot_sha256=file_hash(path), input_result_sha256=file_hash(path.with_name('result.json')),
            episode_split_disjoint=True, full_history_probability_parity=parity)))


def render(root):
    import html
    p = json.loads((root / 'plan.json').read_text()); status = json.loads((root / 'status.json').read_text())
    results = []
    for job in p['jobs']:
        path = root / f"seed{job['seed']}-step{job['step']}" / 'result.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
    web = Path('/tmp/opencode/popgym-report-web/flash-memory/'); web.mkdir(exist_ok=True)
    write_json(web / 'data.json', dict(status=status, plan=p, results=results))
    rows = []
    for r in results:
        for name, probe in r['probes'].items():
            m = probe['overall']; c = m['counts']
            fmt = lambda x: 'unavailable' if x is None else f'{x:.3f}'
            rows.append(f"<tr><td>{r['job']['seed']}</td><td>{r['job']['step']}</td><td>{html.escape(name)}</td>" +
                ''.join(f'<td>{fmt(x)}</td>' for x in (c['shots']['mae'], c['unique_shots']['mae'], c['hits']['mae'],
                    m['fired']['balanced_accuracy'], m['fired']['micro_auroc'], m['revealed_hits']['balanced_accuracy'], m['revealed_hits']['micro_auroc'])) + '</tr>')
    (web / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60">'
        '<title>All-categorical memory probes</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>'
        '<h1>All-categorical frozen memory probes</h1><p>' + html.escape(json.dumps(status)) + '</p>'
        '<p>Episode-disjoint heldout decoding, not causal policy use. Hit mask means previously revealed hits, NOT the hidden ship board. '
        'Linear/MLP inputs: memory alone, last observation/action, or clock+last input. Null uses training-only clock/mode prevalence. '
        'Balanced accuracy and AUROC: .5 chance, 1 perfect. Counts shown as MAE. Raw JSON contains age/mode/retention slices and shuffled-memory controls.</p>'
        '<a href="data.json">Full results and certificates</a><table><tr><th>Seed</th><th>Step</th><th>Probe</th><th>Shots MAE</th><th>Unique MAE</th><th>Hits MAE</th>'
        '<th>Fired BA</th><th>Fired AUC</th><th>Revealed-hit BA</th><th>Revealed-hit AUC</th></tr>' + ''.join(rows) + '</table>')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int); parser.add_argument('--report', action='store_true')
    args = parser.parse_args(); root = args.root.resolve()
    if args.prepare:
        prepare(root); render(root)
    elif args.worker is not None:
        worker(root, args.worker)
    elif args.report:
        render(root)
    else:
        try:
            p = guard(root)
            for index, job in enumerate(p['jobs']):
                write_json(root / 'status.json', dict(state='running', total=len(p['jobs']), completed=index, job=job)); render(root)
                with (root / f'worker{index}.log').open('x') as log:
                    subprocess.run([sys.executable, __file__, '--root', str(root), '--worker', str(index)], stdout=log, stderr=subprocess.STDOUT, check=True)
                guard(root)
            write_json(root / 'status.json', dict(state='complete', completed=len(p['jobs']), total=len(p['jobs']))); render(root)
        except BaseException:
            import traceback
            write_json(root / 'status.json', dict(state='failed', error=traceback.format_exc())); render(root); raise


if __name__ == '__main__':
    main()
