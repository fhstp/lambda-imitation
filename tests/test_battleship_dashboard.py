"""The dashboard reads native JSON only;it must never operate a training run."""
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('dashboard', Path(__file__).parents[1]/'scripts/battleship_dashboard.py')
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def result(seed=42, step=10000):
    metric = dict(episode_auroc=dict(mean=.8, ci95=[.7, .9], eligible_episodes=19),
                  episodes=20, positive_cells=30, train_supported_cells=61, micro_auroc=.79)
    metrics = dict(random=dict(fired=metric, hits={**metric, 'episode_auroc': dict(mean=None, ci95=None, eligible_episodes=0)}))
    return dict(seed=seed, step=step, certificate=dict(state_digest_before='abc', state_digest_after='abc', episode_split_disjoint=True),
                horizons=[dict(task=t, n=n, models={'memory-linear': dict(metrics={**metrics, 'shuffled': metrics})})
                          for t in ('window', 'lag') for n in (1, 5, 10, 16, 32, 64)],
                full_history={}, coverage={}, excluded_early_episodes={}, policy_summary={}, settings={},
                policy=dict(memory_suite='tiered'), seconds=1.)


def boards(seed=42, step=10000):
    return [[step, seed, 100, 2, 64, 'fired', y, x, 0, .2, 0] for y in range(8) for x in range(8)]


def test_scope_is_the_exact_user_sweep_not_the_whole_project():
    assert dashboard.sweep_path('fhstp-data-intelligence-research-group/retrace-discrepancy-battleship-popgym/sweeps/ekvudzlb') == dashboard.DEFAULT_SWEEP
    assert dashboard.sweep_path('https://wandb.ai/'+dashboard.DEFAULT_SWEEP) == dashboard.DEFAULT_SWEEP
    for value in ('../../secret', 'entity/project', 'entity/project/../bad'):
        with pytest.raises(ValueError):
            dashboard.sweep_path(value)


def test_sparse_history_rows_merge_without_losing_the_step_marker_or_zero():
    old = dict(points=[], cursor=-1)
    rows = [dict(_step=0, steps=10000),
            {'_step': 1, 'steps': 10000, 'eval/stochastic/return_mean/mean': 0., 'eval/stochastic/return_mean/std': .1, 'eval/stochastic/return_mean/n_seeds': 3},
            {'_step': 2, 'steps': 10000, 'memory/diagnostic_seconds/mean': 2.},
            {'_step': 3, 'steps': 20000, 'seed/42/eval/stochastic/return_mean': -.4}]
    value = dashboard.merge_history(old, rows, [42, 43, 44])
    assert value['cursor'] == 3 and len(value['points']) == 2
    assert value['points'][0]['metrics']['return']['mean'] == 0
    assert 'diagnostic_seconds' in value['points'][0]['metrics']
    assert value['points'][1]['metrics']['return']['mean'] is None
    assert value['points'][1]['metrics']['return']['seeds'] == {'42': -.4}
    value = dashboard.merge_history(value, [{'_step': 4, 'steps': 10000, 'seed/42/eval/stochastic/return_mean': -.2}], [42])
    assert value['points'][0]['metrics']['return']['mean'] == 0
    assert value['points'][0]['metrics']['return']['std'] == .1
    assert value['points'][0]['metrics']['return']['seeds'] == {'42': -.2}


def test_missing_and_nonfinite_metrics_are_never_filled_with_zero():
    assert dashboard.metric_values({'eval/stochastic/return_mean/mean': float('nan')}, [42]) == {}
    assert not dashboard.finite(True)
    value = dashboard.compact_result(result(), 42, 10000)
    hit = value['curves']['window/hits/random/memory-linear'][0]
    assert hit[1:5] == [None, None, None, 0]
    assert 'window/fired/random/memory-linear-shuffled' in value['curves']


@pytest.mark.parametrize('kind', ['seed', 'step', 'state', 'split', 'duplicate', 'incomplete'])
def test_artifact_identity_integrity_and_complete_suite_are_enforced(kind):
    r = result()
    if kind in ('seed', 'step'):
        r[kind] += 1
    elif kind == 'state':
        r['certificate']['state_digest_after'] = 'changed'
    elif kind == 'split':
        r['certificate']['episode_split_disjoint'] = False
    elif kind == 'duplicate':
        r['horizons'].append(r['horizons'][0])
    else:
        r['horizons'].pop()
    with pytest.raises(ValueError):
        dashboard.compact_result(r, 42, 10000)


def test_reconstructions_require_64_unique_cells_and_correct_identity():
    assert dashboard.compact_boards(boards(), 42, 10000)[0]['scores'] == [.2]*64
    for rows in (boards()[:-1], boards()+[boards()[0]], boards(seed=43)):
        with pytest.raises(ValueError):
            dashboard.compact_boards(rows, 42, 10000)


def test_snapshot_downloads_only_json_without_sdk_cache(tmp_path):
    requested = []
    class Entry:
        size = 10
        def __init__(self, name):
            self.name = name
        def download(self, root, skip_cache):
            assert skip_cache is True
            path = Path(root)/self.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result() if self.name.endswith('result.json') else boards()))
            return path
    class Artifact:
        metadata = dict(step=10000, seeds=[42])
        name = 'memory-test-step10000:v0'
        digest = 'test-digest'
        def get_entry(self, name):
            requested.append(name)
            return Entry(name)
    api = SimpleNamespace(artifact=lambda name: Artifact())
    output = tmp_path/'web/step10000.json.gz'
    checkpoint = dashboard.fetch_snapshot(api, 'entity/project', 'test', 10000, [42], tmp_path, output)
    data = json.loads(gzip.decompress(output.read_bytes()))
    assert data['seeds']['42']['certificate']['state_unchanged']
    assert checkpoint['step'] == 10000
    assert requested == ['seed42/result.json', 'seed42/reconstructions.json']
    assert not list(tmp_path.glob('measurement-*'))


def test_public_metadata_excludes_secrets_and_partial_objectives():
    run = SimpleNamespace(id='test', name='test', state='failed', created_at='today', entity='entity', project='project',
                          config=dict(lambda_coef=10, api_key='SECRET'), summary=dict(steps=10000, **{'objective/return_mean': 1.}))
    public = dashboard.public_run(run, dict(points=[]))
    assert public['objective'] is None and 'SECRET' not in json.dumps(public)
    assert public['state'] == 'failed' and public['returns'] == []


def test_atomic_gzip_serialization_keeps_nulls_and_rejects_nan(tmp_path):
    path = tmp_path/'data.json.gz'
    dashboard.atomic_json(path, dict(metric=None), compressed=True)
    assert json.loads(gzip.decompress(path.read_bytes())) == dict(metric=None)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        dashboard.atomic_json(path, dict(metric=float('nan')), compressed=True)
    assert path.read_bytes() == before


def test_publisher_is_exact_sweep_incremental_and_keeps_last_good_data_on_failure(tmp_path):
    class Run:
        id = 'test'
        name = 'test'
        state = 'running'
        created_at = 'today'
        entity = 'entity'
        project = 'project'
        config = dict(n_seeds=3, seed_base=42)
        summary = {'_step': 1, 'steps': 10000}
        scans = 0
        broken = False
        def scan_history(self, **kwargs):
            assert kwargs['min_step'] == (2 if self.broken else 0)
            assert 'keys' not in kwargs, 'API key intersections drop sparse rows'
            self.scans += 1
            if self.broken:
                raise RuntimeError('Unavailable')
            return [{'_step': 1, 'steps': 10000, 'eval/stochastic/return_mean/mean': -.4}]
    run = Run()
    class API:
        def flush(self):
            pass
        def sweep(self, path):
            assert path == dashboard.DEFAULT_SWEEP
            return SimpleNamespace(runs=[run])
        def artifact(self, *args):
            raise AssertionError('No completed memory measurement to retrieve')
    args = SimpleNamespace(sweep=dashboard.DEFAULT_SWEEP, output=tmp_path/'web', state=tmp_path/'private',
                           interval=120, artifacts_per_cycle=8, min_free_gib=2.5)
    args.output.mkdir(); args.state.mkdir()
    dashboard.publish(API(), args)
    first = dashboard.read_json(args.output/'data.json')
    assert first['runs'][0]['returns'][0]['mean'] == -.4 and run.scans == 1
    dashboard.publish(API(), args)
    assert run.scans == 1
    last_good = dashboard.read_json(args.output/'data.json')
    run.broken = True; run.summary['_step'] = 2
    dashboard.publish(API(), args)
    last = dashboard.read_json(args.output/'data.json')
    assert last['runs'][0] == last_good['runs'][0]
    assert last['warnings'] and run.scans == 2


def test_output_directory_cannot_mix_sweeps(tmp_path):
    args = SimpleNamespace(sweep=dashboard.DEFAULT_SWEEP, output=tmp_path)
    dashboard.atomic_json(tmp_path/'data.json', dict(sweep='different/project/sweep'))
    api = SimpleNamespace(flush=lambda: None, sweep=lambda p: SimpleNamespace(runs=[]))
    with pytest.raises(ValueError, match='different sweep'):
        dashboard.publish(api, args)


def test_private_cache_cannot_mix_sweeps(tmp_path):
    args = SimpleNamespace(sweep=dashboard.DEFAULT_SWEEP, output=tmp_path/'web', state=tmp_path/'private')
    dashboard.atomic_json(args.state/'scope.json', dict(sweep='different/project/sweep'))
    api = SimpleNamespace(flush=lambda: None, sweep=lambda p: SimpleNamespace(runs=[]))
    with pytest.raises(ValueError, match='different sweep'):
        dashboard.publish(api, args)


def test_unavailable_artifacts_do_not_starve_other_trials_or_older_steps(tmp_path):
    attempts = []
    class Run:
        name = 'test'
        state = 'running'
        created_at = 'today'
        entity = 'entity'
        project = 'project'
        config = dict(n_seeds=3, seed_base=42)
        summary = {'_step': 1, 'steps': 20000}
        def __init__(self, rid):
            self.id = rid
        def scan_history(self, **kwargs):
            return [{'_step': i, 'steps': (i+1)*10000, 'memory/diagnostic_seconds/mean': 1.} for i in range(2)]
    class API:
        def flush(self):
            pass
        def sweep(self, path):
            return SimpleNamespace(runs=[Run('a'), Run('b')])
        def artifact(self, name):
            attempts.append(name)
            raise RuntimeError('Upload pending')
    args = SimpleNamespace(sweep=dashboard.DEFAULT_SWEEP, output=tmp_path/'web', state=tmp_path/'private',
                           interval=120, artifacts_per_cycle=1, min_free_gib=2.5)
    args.output.mkdir(); args.state.mkdir()
    for _ in range(4):
        dashboard.publish(API(), args)
    assert len(set(attempts)) == 4
    assert 'memory-a-step20000' in attempts[0]
    data = dashboard.read_json(args.output/'data.json')
    assert data['memory_pending'] == 4
    assert all(r['checkpoints'] == [] for r in data['runs'])
    assert data['warnings']


def test_sparse_warning_rows_survive_and_skipped_checkpoints_are_not_zero_filled():
    record=dict(steps=10000,seed=42,stage='policy_memory_parity',message='Tolerance exceeded',
                details=dict(mismatched_elements=1,elements=409600,max_probability_error=1.2638e-5),wall_time=1.)
    rows=[{'_step':0,'steps':10000,'diagnostics/warning_json':json.dumps(record)},
          {'_step':1,'steps':10000,'diagnostics/memory_complete':0},
          {'_step':2,'steps':10000,'eval/stochastic/return_mean/mean':.3}]
    value=dashboard.merge_history({},rows,[42])
    assert value['diagnostic_warnings'][0]['details']['mismatched_elements']==1
    assert value['points'][0]['memory_complete'] is False
    assert value['points'][0]['metrics']['return']['mean']==.3
    assert 'diagnostic_seconds' not in value['points'][0]['metrics']
    assert dashboard.merge_history(value,rows,[42])==value
    value=dashboard.merge_history(value,[{'_step':3,'steps':10000,'diagnostics/warning_count':2}],[42])
    assert value['diagnostic_warning_count']==2 and len(value['diagnostic_warnings'])==1


def test_finite_parity_mismatch_is_preserved_not_falsely_certified():
    r=result()
    entries=[dict(passed=False,mismatched_elements=1,elements=409600,max_probability_error=1.2638e-5)]
    r['certificate']['policy_memory_parity']=entries
    compact=dashboard.compact_result(r,42,10000)
    assert compact['certificate']['state_unchanged']
    assert compact['certificate']['policy_memory_parity']==entries


def test_watch_refreshes_initially_empty_sweep_and_later_states_and_history(tmp_path,monkeypatch):
    """Match the SDK's separate sweep cache,which Api.flush() does not clear."""
    import copy
    import sys
    import wandb
    remote=[];clients=[];snapshots=[]
    class Run:
        id='new-trial'
        name='new-trial'
        created_at='today'
        entity='entity'
        project='project'
        config=dict(n_seeds=3,seed_base=42)
        state='running'
        summary={'_step':0,'steps':10000}
        rows=[{'_step':0,'steps':10000,'eval/stochastic/return_mean/mean':.1}]
        def scan_history(self,**kwargs):
            return [row for row in self.rows if row['_step']>=kwargs['min_step']]
    class API:
        def __init__(self,**kwargs):
            assert kwargs==dict(timeout=45)
            self.cached_sweep=None
            clients.append(self)
        def flush(self):
            pass # Like the SDK,this does NOT invalidate the cached sweep.
        def sweep(self,path):
            assert path==dashboard.DEFAULT_SWEEP
            if self.cached_sweep is None:self.cached_sweep=SimpleNamespace(runs=copy.deepcopy(remote))
            return self.cached_sweep
    def sleep(interval):
        assert interval==30
        snapshots.append(dashboard.read_json(tmp_path/'web/data.json'))
        if len(snapshots)==1:
            remote.append(Run())
        elif len(snapshots)==2:
            remote[0].state='finished'
            remote[0].summary={'_step':1,'steps':20000}
            remote[0].rows=remote[0].rows+[{'_step':1,'steps':20000,'eval/stochastic/return_mean/mean':.4}]
        else:
            raise KeyboardInterrupt('End three-cycle test')
    monkeypatch.setattr(wandb,'Api',API)
    monkeypatch.setattr(dashboard.time,'sleep',sleep)
    monkeypatch.setattr(sys,'argv',['dashboard','--watch','--interval','30','--output',str(tmp_path/'web'),'--state',str(tmp_path/'private')])
    with pytest.raises(KeyboardInterrupt,match='three-cycle'):
        dashboard.main()
    assert len(clients)==3
    assert snapshots[0]['runs']==[]
    assert snapshots[1]['runs'][0]['state']=='running'
    assert snapshots[1]['runs'][0]['returns'][-1]['mean']==.1
    assert snapshots[2]['runs'][0]['state']=='finished'
    assert [p['mean'] for p in snapshots[2]['runs'][0]['returns']]==[.1,.4]
