"""Single-critic-encoder control, preserving the previous frozen target policy.

Only bootstrap critic inputs change: both online and EMA Q heads use online
frozen features. Policy logits and importance ratios still use the ORIGINAL
frozen target actor/encoder. Thus the behaviour/target policy is not retargeted.
Uses the same production loss code, replay indices, optimizer states and RNG.
"""

import argparse
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

import popgym_critic_settling as base
from popgym_alpha_sweep import file_hash
from popgym_suite import write_json


def captures(fn):
    fn = inspect.unwrap(fn)
    return {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ())}


def replace_captures(fn, overrides):
    """Clone a function's closure locally; never mutate production functions."""
    fn = inspect.unwrap(fn)
    if not set(overrides) <= set(fn.__code__.co_freevars):
        raise ValueError("Production closure changed")
    def cell(value):
        return (lambda: value).__closure__[0]
    closure = tuple(cell(overrides[name]) if name in overrides else original
                    for name, original in zip(fn.__code__.co_freevars, fn.__closure__))
    copied = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, closure)
    copied.__kwdefaults__ = fn.__kwdefaults__
    return copied


def aligned_loss(production_loss):
    """Retain all production target/regression logic, redirect ONLY Q inputs."""
    low = captures(production_loss)
    get_v, get_q = low["get_v"], low["get_q"]
    run_critic = captures(get_v)["run_critic"]
    def loss(actor, q, target, graph, lam, actions, rewards, dones, behaviour,
             latents, policy_latents, key, masks=None):
        flat_online = latents.reshape(-1, latents.shape[-1])
        # get_v's actor still consumes the original policy_latents. Redirect
        # only its Q evaluation to the online representation at the same steps.
        def aligned_run_critic(critic, graph, _policy_features):
            return run_critic(critic, graph, flat_online)
        value = replace_captures(get_v, {"run_critic": aligned_run_critic})
        def taken(critic, graph, _features, actions):
            return get_q(critic, graph, flat_online, actions)
        patched = replace_captures(production_loss, {"get_v": value, "get_q": taken})
        return patched(actor, q, target, graph, lam, actions, rewards, dones, behaviour,
                       latents, policy_latents, key, masks)
    return loss


def verify(root):
    plan = base.verify(root)
    if plan["aligned_script_sha256"] != file_hash(__file__):
        raise ValueError("Aligned control source changed")
    if plan["baseline_plan_sha256"] != file_hash(Path(plan["baseline_root"]) / "plan.json"):
        raise ValueError("Baseline plan changed")
    return plan


def prepare(root, baseline_root):
    original = base.verify(baseline_root)
    if json.loads((baseline_root / 'status.json').read_text())["state"] != "complete":
        raise ValueError("Need a completed baseline")
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError("Use a separate control root")
    plan = {**original, "created_at": datetime.now().astimezone().isoformat(),
            "baseline_root": str(baseline_root.resolve()), "baseline_plan_sha256": file_hash(baseline_root / 'plan.json'),
            "aligned_script_sha256": file_hash(__file__),
            "control": "Both online and EMA critic heads use the same frozen online features. Policy logits/ratios use original frozen target actor and target-encoder features, preserving the baseline target policy exactly.",
            "protocol": "Matched single-critic-encoder control: same checkpoints, 5 seeds per task, learning rates, fixed replay indices, Adam moments, EMA head states, RNG and 12k-update budgets as baseline. Only bootstrap Q feature inputs change. Policy features/probabilities unchanged. No memory/policy/main critic updates or oracle claims."}
    write_json(root / 'plan.json', plan)
    return plan


def worker(root, index):
    import numpy as np
    plan = verify(root)
    original_helpers = base.production_helpers
    def helpers(fns):
        result = original_helpers(fns)
        result["loss_vtrace_lambda_sequence"] = aligned_loss(result["loss_vtrace_lambda_sequence"])
        return result
    base.production_helpers = helpers  # Local to this worker; no file/global process edits.
    base.worker(root, index)
    job = plan['jobs'][index]
    relative = Path('runs') / job['env_id'] / f"seed{job['seed']}"
    with np.load(root / relative / 'window-indices.npz') as current, np.load(Path(plan['baseline_root']) / relative / 'window-indices.npz') as previous:
        for label in ('train', 'holdout'):
            np.testing.assert_array_equal(current[label], previous[label])
    path = root / relative / 'results.json'
    result = json.loads(path.read_text())
    result['control'] = plan['control']
    result['paired_replay_indices_verified'] = True
    write_json(path, result)


def render(root):
    web = Path('/tmp/opencode/popgym-report-web/critic-aligned')
    base.render(root, web)
    path = web / 'index.html'
    text = path.read_text().replace('<h1>Frozen-policy / frozen-memory critic settling</h1>',
        '<h1>Single-critic-encoder control</h1><p>Bootstrap Q values now use the same frozen online features as fitted Q values. Original target-policy logits and importance ratios are unchanged. <a href="/critic-settling/">Previous two-encoder diagnostic</a></p>')
    temp = web / 'aligned.tmp'
    temp.write_text(text); temp.replace(path)


def supervise(root):
    plan = verify(root)
    slots, active, failures = list(plan['gpus']), [], []
    queue = []
    for i, job in enumerate(plan['jobs']):
        p = root / 'runs' / job['env_id'] / f"seed{job['seed']}" / 'status.json'
        if not p.exists() or json.loads(p.read_text()).get('state') != 'complete':
            queue.append(i)
    (root / 'logs').mkdir(exist_ok=True)
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while queue or active:
            while queue and slots:
                i, gpu = queue.pop(0), slots.pop(0)
                log = (root / 'logs' / f'{i}.log').open('a')
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'JAX_PLATFORMS': 'cuda',
                       'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'PYTHONUNBUFFERED': '1'}
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(i), '--root', str(root)],
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
                active.append((p, gpu, i, log))
                print('START', i, gpu, flush=True)
            for item in list(active):
                p, gpu, i, log = item
                code = p.poll()
                if code is None:
                    continue
                log.close(); active.remove(item); slots.append(gpu)
                if code:
                    failures.append(i)
                    job = plan['jobs'][i]
                    folder = root / 'runs' / job['env_id'] / f"seed{job['seed']}"
                    folder.mkdir(parents=True, exist_ok=True)
                    write_json(folder / 'status.json', {'state': 'failed', 'exit_code': code})
                print('DONE' if not code else 'FAILED', i, flush=True)
            render(root)
            write_json(root / 'status.json', {'state': 'running', 'queued': queue, 'active': [i for _, _, i, _ in active], 'failures': failures})
            if active:
                time.sleep(5)
        verify(root)
        if failures:
            raise RuntimeError(f'Control jobs failed: {failures}; sweep remains paused')
        subprocess.run(['systemctl', '--user', 'thaw', base.SWEEP_UNIT], check=True)
        pause = Path('/tmp/opencode/popgym-alpha-followup/priority-pause.json')
        record = json.loads(pause.read_text())
        write_json(pause, {**record, 'state': 'resumed', 'resumed_at': datetime.now().astimezone().isoformat()})
        write_json(root / 'status.json', {'state': 'complete', 'sweep_resumed': True, 'completed_at': datetime.now().astimezone().isoformat()})
    except BaseException as error:
        write_json(root / 'status.json', {'state': 'interrupted', 'error': repr(error), 'sweep_resumed': False})
        raise
    finally:
        for p, _, _, _ in active:
            p.terminate()
        for p, _, i, log in active:
            p.wait(); log.close()
            job = plan['jobs'][i]
            folder = root / 'runs' / job['env_id'] / f"seed{job['seed']}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'status.json'
            if not path.exists() or json.loads(path.read_text()).get('state') != 'complete':
                write_json(path, {'state': 'interrupted', 'exit_code': p.returncode})
        render(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path, default=Path('/tmp/opencode/popgym-critic-settling-v2'))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--supervise', action='store_true')
    mode.add_argument('--worker', type=int)
    args = p.parse_args()
    root = args.root.resolve()
    if args.prepare:
        prepare(root, args.baseline_root.resolve())
    elif args.worker is not None:
        worker(root, args.worker)
    else:
        supervise(root)


if __name__ == '__main__':
    main()
