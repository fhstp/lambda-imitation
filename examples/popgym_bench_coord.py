"""W&B-only coordination for popgym_bench.py across hosts/clusters.

No shared filesystem is needed. Every training attempt is one W&B run whose
config names its task (arm, env_id). A task is

- complete   if any attempt reported run_state 'complete';
- claimed    if an attempt is alive: state 'running' and a heartbeat within
             LEASE_SECONDS. The OWNER is the earliest-created alive attempt
             (ties: smallest run id). Later attempts yield;
- resumable  from the newest 'checkpoint-<arm>-<task>' artifact (alias latest).

W&B offers no atomic lock, so two attempts started within seconds of each other
may both run until one sees the other. The later one yields (default re-check
after setup and at every evaluation), losing at most that time.
"""
from datetime import datetime, timezone
import time

LEASE_SECONDS = 900
NON_TASK_JOBS = ('supervisor', 'checkpoint-upload')


def ts(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def record(run, now=None):
    """Plain dict for one W&B run (or a test fake with the same attributes)."""
    config, summary = dict(run.config), dict(run.summary)
    heartbeat = ts(getattr(run, 'heartbeat_at', None))
    now = time.time() if now is None else now
    alive = run.state == 'running' and heartbeat is not None and now - heartbeat < LEASE_SECONDS
    return dict(id=run.id, job_type=run.job_type, arm=config.get('arm'), env_id=config.get('env_id'),
                created=ts(getattr(run, 'created_at', None)) or 0., state=run.state, alive=alive,
                run_state=summary.get('run_state'), steps=summary.get('steps') or 0,
                complete=summary.get('run_state') == 'complete' or (run.state == 'finished' and 'final_stochastic_return' in summary
                                                                   and summary.get('run_state') is None),
                host=config.get('host'), url=getattr(run, 'url', None))


def fetch_records(api, entity, project):
    try:
        runs = list(api.runs(f'{entity}/{project}'))
    except ValueError:  # project does not exist yet
        return []
    now = time.time()
    return [record(r, now) for r in runs]


def board(records):
    """{(arm, env_id): dict(complete, alive=[records sorted by claim order], steps)}"""
    tasks = {}
    for r in records:
        if r['job_type'] in NON_TASK_JOBS or not r['arm'] or not r['env_id'] or r['run_state'] == 'yielded':
            continue
        t = tasks.setdefault((r['arm'], r['env_id']), dict(complete=False, alive=[], steps=0))
        t['complete'] |= r['complete']
        t['steps'] = max(t['steps'], r['steps'])
        if r['alive']:
            t['alive'].append(r)
    for t in tasks.values():
        t['alive'].sort(key=lambda r: (r['created'], r['id']))
    return tasks


def owner(task):
    return task['alive'][0]['id'] if task and task['alive'] else None


def available(tasks, key, local_ids=()):
    """True when nobody else holds or finished the task (local runs excepted)."""
    t = tasks.get(key)
    if t is None:
        return True
    if t['complete']:
        return False
    return all(r['id'] in local_ids for r in t['alive'])


def checkpoint_name(arm, env_id):
    return f'checkpoint-{arm}-{env_id.removeprefix("popgym-").removesuffix("-v0")}'


def latest_checkpoint(api, entity, project, arm, env_id):
    """(artifact, steps) of the newest checkpoint, or (None, 0)."""
    try:
        artifact = api.artifact(f'{entity}/{project}/{checkpoint_name(arm, env_id)}:latest', type='checkpoint')
    except Exception:  # noqa: BLE001 -- no checkpoint uploaded yet
        return None, 0
    return artifact, int(artifact.metadata.get('steps', 0))


def upload_checkpoint(run, api, entity, project, path, arm, env_id, steps):
    """Upload, wait until committed, then drop older versions (keep storage flat)."""
    import wandb
    name = checkpoint_name(arm, env_id)
    artifact = wandb.Artifact(name, type='checkpoint', metadata=dict(steps=int(steps), arm=arm, env_id=env_id, time=time.time()))
    artifact.add_file(str(path), name='latest.pkl')
    logged = run.log_artifact(artifact, aliases=['latest', f'step{int(steps)}'])
    logged.wait()
    try:
        for version in api.artifacts('checkpoint', f'{entity}/{project}/{name}'):
            if 'latest' not in version.aliases:
                version.delete(delete_aliases=True)
    except Exception:  # noqa: BLE001 -- cleanup is best effort
        pass
    return int(steps)
