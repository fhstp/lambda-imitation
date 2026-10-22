"""Publish the POPGym bench dashboard from W&B, on any machine.

    python examples/popgym_bench_wandb.py --web-dir /srv/site/popgym-bench --watch 120

Reads only the W&B project written by popgym_bench.py (one run per task x arm
plus a 'supervisor' run) and writes a static index.html + data.json with the
same content as the dashboard served on the training host. No JAX, no
checkpoints, no access to the training host.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import popgym_bench_dashboard as dashboard  # noqa: E402

STUDY = 'popgym-b200-flash-bench'
META_KEYS = ('env_id', 'arm', 'backend', 'port_id', 'horizon', 'action_dim', 'obs_shape', 'batch_episodes')
FINAL_MODES = ('stochastic', 'greedy', 'critic', 'random')


def timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def arm_from_rows(rows, config, summary, state, url, heartbeat):
    seeds = config.get('seeds') or list(range(config.get('seed_base', 42), config.get('seed_base', 42) + config.get('n_seeds', 0)))
    curve, final = [], None
    for r in sorted((r for r in rows if r.get('steps') is not None), key=lambda r: r['steps']):
        if r.get('eval/stochastic/return_mean/mean') is not None:
            values = [r.get(f'seed/{s}/eval/stochastic/return_mean') for s in seeds]
            if any(v is None for v in values):
                continue  # incomplete row: never fill missing seeds
            curve.append(dict(steps=r['steps'], mean=r['eval/stochastic/return_mean/mean'], seeds=values,
                              greedy=r.get('eval/greedy/return_mean/mean')))
        if r.get('final/stochastic/return_mean/mean') is not None:
            final = {}
            for mode in FINAL_MODES:
                mean = r.get(f'final/{mode}/return_mean/mean')
                if mean is None:
                    continue
                values = [r.get(f'seed/{s}/final/{mode}/return_mean') for s in seeds]
                final[mode] = dict(mean=mean, std=r.get(f'final/{mode}/return_mean/std'), n=len(seeds),
                                   seeds=values if all(v is not None for v in values) else [])
    run_state = summary.get('run_state')
    if run_state is None and state == 'finished' and final:
        run_state = 'complete'  # first pushed version did not log run_state
    if run_state == 'complete' and final:
        status = 'complete'
    elif run_state == 'failed' or state in ('crashed', 'failed'):
        status = 'failed'
    elif state == 'running':
        status = 'running'
    else:
        status = 'stopped'  # finished without a final holdout: worker killed/restarting
    return dict(state=status, steps=summary.get('steps', 0) or 0, target=config.get('max_steps'),
                steps_per_second=summary.get('throughput/steps_per_second'), updated=heartbeat,
                error=(summary.get('error') or '')[-600:] or None, curve=curve, best=dashboard.best_of(curve),
                final=final, latest=curve[-1] if curve else None,
                meta={k: config[k] for k in META_KEYS if k in config}, url=url)


class Exporter:
    def __init__(self, entity, project):
        self.entity, self.project = entity, project
        self.cache = {}

    def history(self, run):
        """Full history, refetched only when the run has logged new rows."""
        marker = (run.summary.get('_step'), run.state)
        cached = self.cache.get(run.id)
        if cached and cached[0] == marker:
            return cached[1]
        rows = list(run.scan_history(page_size=1000))
        self.cache[run.id] = (marker, rows)
        return rows

    def collect(self):
        import wandb
        from popgym_bench_envs import all_env_ids
        api = wandb.Api(timeout=60)
        try:
            runs = list(api.runs(f'{self.entity}/{self.project}'))
        except ValueError:  # project not created yet (nothing launched)
            runs = []
        study = dict(wandb_project=self.project, wandb_entity=self.entity, config=None, created=None)
        envs, arms, scheduler, validation = all_env_ids(), {}, None, None
        for run in runs:
            config, summary = dict(run.config), dict(run.summary)
            if run.job_type == 'supervisor':
                study.update(config=config.get('study'), created=timestamp(run.created_at))
                envs = config.get('envs') or envs
                scheduler = json.loads(summary['scheduler']) if summary.get('scheduler') else None
                validation = json.loads(summary['validation']) if summary.get('validation') else None
                continue
            env_id, arm = config.get('env_id'), config.get('arm')
            if not env_id or arm not in dashboard.ARMS:
                continue
            if study['config'] is None:
                import popgym_bench
                study['config'] = {k: config[k] for k in popgym_bench.DEFAULTS if k in config}
            arms.setdefault(env_id, {})[arm] = arm_from_rows(self.history(run), config, summary, run.state, run.url,
                                                             timestamp(getattr(run, 'heartbeat_at', None)))
        if study['config'] is None:
            import popgym_bench
            study['config'] = dict(popgym_bench.DEFAULTS)
        return dashboard.assemble(study, envs, arms, scheduler, validation)


def publish(web_dir, data):
    web_dir = Path(web_dir)
    web_dir.mkdir(parents=True, exist_ok=True)
    for name, write in (('data.json', lambda p: p.write_text(json.dumps(data, allow_nan=False))),
                        ('index.html', lambda p: shutil.copyfile(dashboard.PAGE, p))):
        tmp = web_dir / f'.{name}.tmp'
        write(tmp)
        os.replace(tmp, web_dir / name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--web-dir', type=Path, required=True)
    p.add_argument('--entity', default=os.environ.get('WANDB_ENTITY', 'fhstp-data-intelligence-research-group'))
    p.add_argument('--project', default=os.environ.get('WANDB_PROJECT', STUDY))
    p.add_argument('--watch', type=float, default=0, help='re-export every N seconds (0: once)')
    args = p.parse_args()
    exporter = Exporter(args.entity, args.project)
    while True:
        try:
            data = exporter.collect()
            publish(args.web_dir, data)
            done = sum(t['arms']['primary']['state'] == 'complete' for t in data['tasks'])
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} exported {len(data["tasks"])} tasks, {done} primary complete', flush=True)
        except Exception:  # noqa: BLE001 -- keep serving the last good export
            if not args.watch:
                raise
            print(traceback.format_exc(), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
