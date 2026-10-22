"""Live dashboard for popgym_bench.py: reads only the output directory's JSON.

    python examples/popgym_bench.py dashboard --output /scratch/popgym-bench --port 8765

No JAX/W&B/checkpoint access. Serves index.html and /data.json (rebuilt at
most every 15 s). Scores are compared with literature and our own earlier
runs from popgym_bench_references.json; protocols differ (see the page notes).
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

HERE = Path(__file__).resolve().parent
PAGE = HERE / 'popgym_bench_dashboard.html'
REFERENCES = HERE / 'popgym_bench_references.json'
ARMS = ('primary', 'control')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def split_env_id(env_id):
    name = env_id.removeprefix('popgym-').removesuffix('-v0')
    for d in ('Easy', 'Medium', 'Hard'):
        if name.endswith(d):
            return name.removesuffix(d), d
    return name, ''


def arm_summary(folder):
    status = read_json(folder / 'status.json', {}) or {}
    config = read_json(folder / 'config.json', {}) or {}
    history = read_json(folder / 'history.json', []) or []
    curve, best_by_seed, final = [], None, None
    for row in history:
        sto = row['eval'].get('stochastic', {}).get('return_mean')
        if not sto:
            continue
        if row.get('final'):
            final = {m: {k: v[k] for k in ('mean', 'std', 'n', 'seeds')} for m, d in row['eval'].items()
                     for v in [d['return_mean']]}
            continue
        greedy = row['eval'].get('greedy', {}).get('return_mean', {})
        curve.append(dict(steps=row['steps'], mean=sto['mean'], seeds=sto['seeds'], greedy=greedy.get('mean')))
        seeds = sto['seeds']
        best_by_seed = seeds if best_by_seed is None else [max(a, b) for a, b in zip(best_by_seed, seeds)]
    best = None
    if best_by_seed:
        n = len(best_by_seed)
        mean = sum(best_by_seed) / n
        best = dict(mean=mean, n=n, std=(sum((x - mean) ** 2 for x in best_by_seed) / (n - 1)) ** .5 if n > 1 else 0.)
    meta = config.get('meta', {})
    return dict(state=status.get('state', 'pending'), steps=status.get('steps', 0), target=status.get('target_steps'),
                steps_per_second=status.get('steps_per_second'), updated=status.get('time'),
                error=(status.get('error') or '')[-600:] or None, curve=curve, best=best, final=final,
                latest=curve[-1] if curve else None, meta=meta)


def references_for(env_id, refs):
    out = {}
    for key in ('popgym', 'popjym'):
        source = refs['literature']['sources'][key]
        task = source['tasks'].get(env_id)
        if task:
            best = task['best'][0]
            out[key] = dict(label=source['label'], mean=best['mean'], sd=best['sd'],
                            model=' / '.join(b['model'] for b in task['best']), citation=source['citation'])
    for key, source in refs['internal'].items():
        task = source['tasks'].get(env_id)
        if task:
            out[key] = dict(label=source['label'], **task)
    return out


def build_data(output):
    output = Path(output)
    study = read_json(output / 'study.json', {}) or {}
    refs = read_json(REFERENCES)
    envs = study.get('envs') or sorted({p.name for arm in ARMS for p in (output / arm).glob('popgym-*')})
    tasks = []
    for env_id in envs:
        family, difficulty = split_env_id(env_id)
        arms = {arm: arm_summary(output / arm / env_id) for arm in ARMS}
        meta = arms['primary']['meta'] or arms['control']['meta']
        tasks.append(dict(env_id=env_id, family=family, difficulty=difficulty, meta=meta, arms=arms,
                          refs=references_for(env_id, refs)))
    return dict(now=time.time(), study=dict(config=study.get('config'), created=study.get('created'),
                wandb_project=study.get('wandb_project'), wandb_entity=study.get('wandb_entity')),
                scheduler=read_json(output / 'scheduler.json'), validation=read_json(output / 'validation.json'),
                literature_notes={k: dict(label=v['label'], citation=v['citation'], protocol=v['protocol'], url=v.get('url'))
                                  for k, v in refs['literature']['sources'].items()},
                internal_notes={k: dict(label=v['label'], protocol=v['protocol']) for k, v in refs['internal'].items()},
                tasks=tasks)


def serve(output, host='0.0.0.0', port=8765):
    cache = dict(at=0., body=b'{}')
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                body, kind = PAGE.read_bytes(), 'text/html; charset=utf-8'
            elif path == '/data.json':
                with lock:
                    if time.time() - cache['at'] > 15:
                        cache['body'] = json.dumps(build_data(output), allow_nan=False).encode()
                        cache['at'] = time.time()
                    body = cache['body']
                kind = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer((host, port), Handler).serve_forever()
