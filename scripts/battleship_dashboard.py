"""Read-only W&B sweep publisher for the existing private report HTTP server.

No JAX imports,checkpoint loads,training-source edits or W&B writes. Downloads
ONLY JSON measurement artifacts into owned temporary directories (no SDK cache),
then keeps compact gzip snapshots. Credentials/signed URLs never reach the web.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_SWEEP = 'fhstp-data-intelligence-research-group/retrace-discrepancy-battleship-popgym/ekvudzlb'
CONFIG_KEYS = ('lambda_coef', 'fe_lr', 'actor_lr', 'critic_lr', 'alpha_initial', 'alpha_final',
               'alpha_decay_steps', 'tau', 'memory_hidden_dim', 'projection_dim', 'n_seeds',
               'seed_base', 'max_steps', 'memory_suite', 'memory_schedule', 'probe_updates')
METRICS = {
    'return': 'eval/stochastic/return_mean',
    'q_rmse': 'eval/stochastic/q_mc_rmse/all',
    'q_bias': 'eval/stochastic/q_mc_bias/all',
    'repeats': 'memory/policy/stochastic/repeat_fraction',
    'unique': 'memory/policy/stochastic/unique_cells',
    'repeat_mass': 'memory/policy/stochastic/actor_repeat_mass',
    'entropy': 'memory/policy/stochastic/entropy',
    'greedy_return': 'memory/policy/greedy/return_mean',
    'uniform_return': 'memory/policy/random/return_mean',
    'alpha': 'train/alpha',
    'main_clip': 'train/main_target_clipped_mass',
    'aux_clip': 'train/lambda2_target_clipped_fraction',
    'critic_loss': 'train/critic_loss',
    'discrepancy': 'train/ld_loss',
    'gradient_ratio': 'memory/grounding/fe_gradient/weighted_discrepancy_to_combined_regression',
    'gradient_cosine': 'memory/grounding/fe_gradient/discrepancy_regression_cosine',
    'target_drift': 'memory/grounding/online_target_feature_relative_l2',
    'actor_q_gap': 'memory/grounding/replay/soft_objective_gap',
    'diagnostic_seconds': 'memory/diagnostic_seconds',
    'parity_max_error': 'memory/parity/max_probability_error',
    'parity_mismatched_fraction': 'memory/parity/mismatched_fraction',
    'parity_warning_blocks': 'memory/parity/warning_blocks',
}
for _task in ('window', 'lag'):
    for _target in ('fired', 'hits'):
        METRICS[f'{_task}_{_target}_16'] = f'memory/horizon/{_task}/16/random/{_target}/memory-linear/auroc'

CURVE_COLUMNS = ['n', 'auroc', 'ci_low', 'ci_high', 'eligible', 'episodes', 'positive', 'supported_cells', 'micro_auroc']


def now():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def number(value):
    return value if finite(value) else None


def identity(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Unsafe run identity')
    return value


def sweep_path(value):
    value = value.removeprefix('https://wandb.ai/').strip('/').replace('/sweeps/', '/')
    parts = value.split('/')
    if len(parts) != 3:
        raise ValueError('Expected ENTITY/PROJECT/SWEEP_ID')
    return '/'.join(identity(p) for p in parts)


def atomic_json(path, value, compressed=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if compressed:
        raw = gzip.compress(raw, compresslevel=6, mtime=0)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(raw)
    temporary.replace(path)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def metric_values(row, seeds):
    result = {}
    for alias, key in METRICS.items():
        values = {str(seed): number(row.get(f'seed/{seed}/{key}')) for seed in seeds}
        values = {k: v for k, v in values.items() if v is not None}
        mean = number(row.get(key + '/mean'))
        if mean is not None or values:
            result[alias] = dict(mean=mean, std=number(row.get(key + '/std')),
                                 n=number(row.get(key + '/n_seeds')), seeds=values)
    return result


def merge_history(previous, rows, seeds):
    """Preserve sparse logging rows;never request an API key intersection."""
    points = {p['step']: p for p in previous.get('points', [])}
    warnings = {p['history_row']: p for p in previous.get('diagnostic_warnings', [])}
    warning_count = previous.get('diagnostic_warning_count', len(warnings))
    cursor = previous.get('cursor', -1)
    for row in rows:
        if finite(row.get('_step')):
            cursor = max(cursor, int(row['_step']))
        if not finite(row.get('steps')):
            continue
        if finite(row.get('diagnostics/warning_count')):
            warning_count = max(warning_count, int(row['diagnostics/warning_count']))
        if isinstance(row.get('diagnostics/warning_json'), str):
            record = json.loads(row['diagnostics/warning_json'])
            if record.get('steps') != row['steps']:
                raise ValueError('Diagnostic warning step mismatch')
            warnings[int(row['_step'])] = dict(history_row=int(row['_step']),
                **{key: record.get(key) for key in ('steps', 'seed', 'stage', 'message', 'details', 'wall_time')})
        metrics = metric_values(row, seeds)
        memory_complete = number(row.get('diagnostics/memory_complete'))
        if not metrics and memory_complete is None:
            continue
        step = int(row['steps'])
        point = points.setdefault(step, dict(step=step, metrics={}))
        if memory_complete is not None:
            point['memory_complete'] = bool(memory_complete)
        for key, value in metrics.items():
            existing = point['metrics'].setdefault(key, dict(mean=None, std=None, n=None, seeds={}))
            existing['seeds'].update(value['seeds'])
            existing.update({k: v for k, v in value.items() if k != 'seeds' and v is not None})
        point['logged_at'] = number(row.get('_timestamp'))
    return dict(points=[points[s] for s in sorted(points)], cursor=cursor,
                diagnostic_warnings=[warnings[s] for s in sorted(warnings)],
                diagnostic_warning_count=max(warning_count, len(warnings)))


def compact_result(result, expected_seed, expected_step):
    if result.get('seed') != expected_seed or result.get('step') != expected_step:
        raise ValueError('Artifact seed/checkpoint identity mismatch')
    cert = result['certificate']
    if not cert.get('state_digest_before') or cert['state_digest_before'] != cert.get('state_digest_after'):
        raise ValueError('Frozen state certificate mismatch')
    if not cert.get('episode_split_disjoint'):
        raise ValueError('Missing episode split certificate')
    curves = {}
    seen = set()
    for task in result['horizons']:
        task_name, n = task['task'], task['n']
        if task_name not in ('window', 'lag') or not isinstance(n, int) or not 1 <= n <= 64 or (task_name, n) in seen:
            raise ValueError('Invalid/duplicate horizon task')
        seen.add((task_name, n))
        for decoder, entry in task['models'].items():
            groups = [(decoder, entry['metrics'])]
            if 'shuffled' in entry['metrics']:
                groups.append((decoder + '-shuffled', entry['metrics']['shuffled']))
            for model, metrics in groups:
                for mode, targets in metrics.items():
                    if mode == 'shuffled':
                        continue
                    for target, value in targets.items():
                        auc = value['episode_auroc']
                        ci = auc.get('ci95') or [None, None]
                        row = [n, number(auc['mean']), *map(number, ci), auc['eligible_episodes'],
                               value['episodes'], value['positive_cells'], value['train_supported_cells'], number(value['micro_auroc'])]
                        curves.setdefault('/'.join((task_name, target, mode, model)), []).append(row)
    for rows in curves.values():
        rows.sort(key=lambda r: r[0])
    ns = range(1, 65) if result['policy']['memory_suite'] == 'full' else (1, 5, 10, 16, 32, 64)
    if seen != {(task, n) for task in ('window', 'lag') for n in ns}:
        raise ValueError('Incomplete declared memory suite')
    # Full-history controls/age/last-shot-lag and count probes are relatively small.
    full = {}
    for model, value in result['full_history'].items():
        full[model] = {k: v for k, v in value.items() if k != 'validation_curve'}
    return dict(seed=expected_seed, curves=curves, full_history=full,
                coverage=result['coverage'], excluded=result['excluded_early_episodes'],
                policy_summary=result['policy_summary'], settings=result['settings'],
                 policy=result['policy'], certificate=dict(state_unchanged=True, episode_split_disjoint=True,
                     policy_memory_parity=cert.get('policy_memory_parity')),
                seconds=result['seconds'], tasks=len(seen))


def compact_boards(rows, expected_seed, expected_step):
    boards = {}
    for row in rows:
        if len(row) != 11 or row[0] != expected_step or row[1] != expected_seed:
            raise ValueError('Reconstruction identity mismatch')
        _, _, episode, mode, age, target, y, x, label, score, binary = row
        if target not in ('fired', 'revealed_hits') or not 0 <= x < 8 or not 0 <= y < 8 or not finite(score):
            raise ValueError('Invalid reconstruction cell')
        key = (episode, mode, age, target)
        board = boards.setdefault(key, dict(episode=episode, mode=mode, age=age, target=target,
                                            labels=[None]*64, scores=[None]*64, binary=[None]*64))
        cell = 8*y+x
        if board['labels'][cell] is not None:
            raise ValueError('Duplicate reconstruction cell')
        board['labels'][cell], board['scores'][cell], board['binary'][cell] = label, score, binary
    if any(any(v is None for v in b['labels']) for b in boards.values()):
        raise ValueError('Incomplete reconstruction board')
    return list(boards.values())


def fetch_snapshot(api, project, run_id, step, seeds, private, output):
    artifact = api.artifact(f'{project}/memory-{run_id}-step{step}:latest')
    metadata = artifact.metadata
    if metadata.get('step') != step or metadata.get('seeds') != seeds:
        raise ValueError('Artifact metadata identity mismatch')
    snapshot = dict(run_id=run_id, step=step, artifact=artifact.name, artifact_digest=artifact.digest,
                    fetched_at=now(), columns=CURVE_COLUMNS, seeds={}, sources=[])
    with tempfile.TemporaryDirectory(prefix='measurement-', dir=private) as temporary:
        for seed in seeds:
            data = {}
            for filename in ('result.json', 'reconstructions.json'):
                entry = artifact.get_entry(f'seed{seed}/{filename}')
                if entry.size is None or entry.size > 32*1024**2:
                    raise ValueError('Unexpected measurement file size')
                path = Path(entry.download(root=temporary, skip_cache=True))
                raw = path.read_bytes()
                snapshot['sources'].append(dict(seed=seed, file=filename, sha256=hashlib.sha256(raw).hexdigest()))
                data[filename] = json.loads(raw)
            result = compact_result(data['result.json'], seed, step)
            result['boards'] = compact_boards(data['reconstructions.json'], seed, step)
            snapshot['seeds'][str(seed)] = result
    atomic_json(output, snapshot, compressed=True)
    return dict(step=step, file=str(output.name), bytes=output.stat().st_size, artifact=artifact.name,
                seeds=seeds, tasks={s: value['tasks'] for s, value in snapshot['seeds'].items()})


def latest_metric(points, key):
    return next((dict(step=p['step'], **p['metrics'][key]) for p in reversed(points) if key in p['metrics']), None)


def public_run(run, cached):
    points = cached['points']
    objective = number(dict(run.summary).get('objective/return_mean'))
    return dict(id=run.id, name=run.name, state=run.state, created_at=run.created_at,
                url=f'https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}',
                config={k: run.config.get(k) for k in CONFIG_KEYS},
                published_step=number(dict(run.summary).get('steps')),
                objective=objective if run.state == 'finished' else None,
                returns=[dict(step=p['step'], **p['metrics']['return']) for p in points if 'return' in p['metrics']],
                memory_curves={target: [dict(step=p['step'], **p['metrics'][f'window_{target}_16'])
                                        for p in points if f'window_{target}_16' in p['metrics']]
                               for target in ('fired', 'hits')},
                latest={key: latest_metric(points, key) for key in ('return', 'repeats', 'window_hits_16', 'q_rmse')},
                checkpoints=cached.get('checkpoints', []), detail=f'runs/{run.id}/data.json',
                diagnostic_warnings=cached.get('diagnostic_warnings', []),
                diagnostic_warning_count=cached.get('diagnostic_warning_count', len(cached.get('diagnostic_warnings', []))),
                 synchronized_at=cached.get('synchronized_at'))


def publish(api, args):
    api.flush()
    project = '/'.join(args.sweep.split('/')[:2])
    runs = list(api.sweep(args.sweep).runs)
    warnings, published, cache = [], [], {}
    previous = read_json(args.output/'data.json', {})
    if previous.get('sweep', args.sweep) != args.sweep:
        raise ValueError('Output directory belongs to a different sweep')
    scope_path = args.state/'scope.json'
    scope = dict(sweep=args.sweep)
    if read_json(scope_path, scope) != scope:
        raise ValueError('Private cache belongs to a different sweep')
    atomic_json(scope_path, scope)
    old_runs = {r['id']: r for r in previous.get('runs', [])}
    for run in runs:
        rid = identity(run.id)
        path = args.state/f'{rid}.json'
        cached = read_json(path, dict(points=[], cursor=-1, checkpoints=[]))
        seeds = list(range(int(run.config.get('seed_base', 42)), int(run.config.get('seed_base', 42))+int(run.config.get('n_seeds', 3))))
        try:
            summary_step = number(dict(run.summary).get('_step'))
            summary_step = -1 if summary_step is None else summary_step
            if summary_step > cached['cursor']:
                cached.update(merge_history(cached, run.scan_history(min_step=cached['cursor']+1, page_size=100), seeds))
            cached['synchronized_at'] = now()
            atomic_json(path, cached)
            cache[rid] = (run, cached, seeds)
            published.append(public_run(run, cached))
            atomic_json(args.output/f'runs/{rid}/data.json', dict(run=published[-1], points=cached['points']))
        except Exception as error:
            warnings.append(f'{rid}: history refresh unavailable ({type(error).__name__}); previous data retained.')
            if rid in old_runs:
                published.append(old_runs[rid])

    def index():
        pending_count = sum(sum('diagnostic_seconds' in p['metrics'] and p['step'] not in {c['step'] for c in cached.get('checkpoints', [])}
                                for p in cached['points']) for _, cached, _ in cache.values())
        atomic_json(args.output/'data.json', dict(schema=1, sweep=args.sweep,
                    sweep_url=f'https://wandb.ai/{project}/sweeps/{args.sweep.split("/")[-1]}',
                    generated_at=now(), refresh_seconds=args.interval, warnings=warnings, runs=published, memory_pending=pending_count,
                    read_only=True, history_sampling='Unsampled scan_history; incremental by W&B history row, not environment step.'))
    index()  # Overview becomes available before potentially slow artifact backfill.
    pending = []
    for rid, (run, cached, seeds) in cache.items():
        done = {p['step'] for p in cached.get('checkpoints', [])}
        for point in cached['points']:
            if 'diagnostic_seconds' in point['metrics'] and point['step'] not in done:
                pending.append((point['step'], rid, run, cached, seeds))
    # Fair round-robin across trials;latest first,then backfill earlier checkpoints.
    pending.sort(key=lambda item: (item[3].get('artifact_attempts', {}).get(str(item[0]), ''), -item[0], item[1]))
    chosen, remaining = [], pending[:]
    while remaining and len(chosen) < args.artifacts_per_cycle:
        taken = set()
        rest = []
        for item in remaining:
            if item[1] not in taken and len(chosen) < args.artifacts_per_cycle:
                chosen.append(item); taken.add(item[1])
            else:
                rest.append(item)
        remaining = rest
    for step, rid, run, cached, seeds in chosen:
        if shutil.disk_usage(args.output).free < args.min_free_gib*1024**3:
            warnings.append('Local disk reserve reached: new memory snapshots deferred; training/W&B untouched.')
            break
        cached.setdefault('artifact_attempts', {})[str(step)] = now()
        atomic_json(args.state/f'{rid}.json', cached)
        try:
            output = args.output/f'runs/{rid}/step{step}.json.gz'
            checkpoint = fetch_snapshot(api, project, rid, step, seeds, args.state, output)
            cached.setdefault('checkpoints', []).append(checkpoint)
            cached['checkpoints'].sort(key=lambda c: c['step'])
            atomic_json(args.state/f'{rid}.json', cached)
            entry = public_run(run, cached)
            published[:] = [entry if r['id'] == rid else r for r in published]
            atomic_json(args.output/f'runs/{rid}/data.json', dict(run=entry, points=cached['points']))
            index()
            print(json.dumps(dict(at=now(), imported_run=rid, step=step, compact_bytes=checkpoint['bytes'])), flush=True)
        except Exception as error:
            warnings.append(f'{rid} at {step}: memory artifact pending/unavailable ({type(error).__name__}); will recheck, no zero substitution.')
    index()
    atomic_json(args.state/'status.json', dict(state='watching', generated_at=now(), runs=len(published),
                memory_pending=read_json(args.output/'data.json')['memory_pending'],
                warnings=warnings))
    print(json.dumps(dict(at=now(), runs=len(published), snapshots_attempted=len(chosen), pending_before=len(pending), warnings=warnings)), flush=True)


def watch(args):
    """Bound SDK/JSON memory lifetime to ONE refresh,not the whole study."""
    import fcntl
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--sweep',args.sweep,
             '--state',str(args.state),'--output',str(args.output),'--interval',str(args.interval),
             '--artifacts-per-cycle',str(args.artifacts_per_cycle),'--min-free-gib',str(args.min_free_gib)]
    # The supervisor and worker use distinct locks. Each worker still owns the
    # normal publisher lock,so independent one-shot invocations cannot overlap.
    with (args.state/'.watch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                subprocess.run(command,check=True,timeout=900)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                status=dict(ok=False,at=now(),error=type(error).__name__)
                if isinstance(error,subprocess.CalledProcessError):status['returncode']=error.returncode
                atomic_json(args.output/'health.json',status)
                print(f'{now()} refresh worker failed: {type(error).__name__};last publication retained',flush=True)
            time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=sweep_path, default=DEFAULT_SWEEP)
    parser.add_argument('--state', type=Path, default=Path('/tmp/opencode/battleship-sweep-dashboard/ekvudzlb'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/opencode/popgym-report-web/b200-sweep'))
    parser.add_argument('--interval', type=int, default=120)
    parser.add_argument('--artifacts-per-cycle', type=int, default=8)
    parser.add_argument('--min-free-gib', type=float, default=2.5)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.interval < 30 or args.artifacts_per_cycle < 1 or args.min_free_gib < 2:
        parser.error('Use interval>=30s, positive artifact budget and at least2GiB disk reserve')
    args.state.mkdir(parents=True, exist_ok=True); args.output.mkdir(parents=True, exist_ok=True)
    if args.watch:
        watch(args)
        return
    import fcntl
    import wandb
    with (args.state/'.publisher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        template = Path(__file__).with_suffix('.html')
        shutil.copyfile(template, args.output/'index.html')
        try:
            # Api.flush() does not clear cached sweep objects. The one-shot
            # worker always starts with a fresh client AND fresh process heap.
            publish(wandb.Api(timeout=45), args)
        except Exception as error:
            # Keep the last good publication. Do not expose exception URLs.
            atomic_json(args.output/'health.json', dict(ok=False, at=now(), error=type(error).__name__))
            print(f'{now()} refresh failed: {type(error).__name__}', flush=True)
            raise
        else:
            atomic_json(args.output/'health.json', dict(ok=True, at=now()))


if __name__ == '__main__':
    main()
