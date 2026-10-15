"""Queued fresh Battleship critic output-feature normalization (OFN) pairs.

Normalize each main/auxiliary twin head's last hidden activation, not the shared
GRU or actor. Original target encoders, losses, RNG and EMA cadence stay intact.
Factory globals are copied locally; no production or predecessor source edits.
"""

import argparse
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from popgym_alpha_sweep import args_from_plan, file_hash, verify_plan
from popgym_gradient_balance import captures
from popgym_suite import source_fingerprint, write_json
import popgym_online_alignment as preceding


EPSILON = 1e-8
ARMS = ('baseline', 'ofn')
DEFAULT_PREDECESSOR = Path('/tmp/opencode/popgym-online-alignment')
DEFAULT_ROOT = Path('/tmp/opencode/popgym-critic-normalization')


def normalized_features(x):
    """Unit L2 norm per example; a safe floor gives finite gradients at zero."""
    import jax.numpy as jnp
    denominator = jnp.sqrt(jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True), EPSILON ** 2))
    return x / denominator


def normalized_head_class():
    from flax import nnx
    from lambda_imitation.iqlearn import Head

    class OutputNormalizedHead(Head):
        """Same parameters/initialization as Head, OFN before the final Linear."""
        def output_features(self, x):
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                if self.norms:
                    x = self.norms[i](x)
                x = nnx.relu(x)
            return x, normalized_features(x)

        def __call__(self, x):
            return self.layers[-1](self.output_features(x)[1])

    return OutputNormalizedHead


def clone_globals(fn, overrides):
    """A private globals dictionary, never mutating imported package objects."""
    if not set(overrides) <= set(fn.__globals__):
        raise ValueError('Factory globals changed')
    copied = types.FunctionType(fn.__code__, {**fn.__globals__, **overrides},
        fn.__name__, fn.__defaults__, fn.__closure__)
    copied.__kwdefaults__ = fn.__kwdefaults__
    return copied


def build_agent(config, port, arm):
    import popgym_critic_settling as base
    import lambda_imitation.iqlearn as core
    import lambda_imitation.utils as utils
    if arm not in ARMS:
        raise ValueError('Unknown normalization arm')
    if arm == 'baseline':
        return base.build_agent(config, port)
    if config['critic_layer_norm'] or config['memory_type'] != 'gru' or config['use_sac']:
        raise ValueError('OFN pilot requires original GRU, no-LayerNorm, reward-only configuration')
    Head = core.Head; OFN = normalized_head_class(); roles = []

    def head(*args, **kwargs):
        # Audited factory: actor omits layer_norm, all six critics pass it.
        critic = 'layer_norm' in kwargs
        roles.append('critic' if critic else 'actor')
        return (OFN if critic else Head)(*args, **kwargs)

    factory = clone_globals(core.create_iqlearn, {'Head': head})
    wrapper = clone_globals(utils.create_iqlearn_from_env, {'create_iqlearn': factory})
    # base.build_agent imports its factory lazily. Copy its function with a
    # private import hook replacing only that one from-import; no sys.modules
    # or original module/function/class is changed, even within this process.
    import builtins
    original_import = builtins.__import__
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == 'lambda_imitation.utils' and 'create_iqlearn_from_env' in fromlist:
            return types.SimpleNamespace(EnvSpec=utils.EnvSpec, create_iqlearn_from_env=wrapper)
        return module
    namespace = dict(vars(builtins)); namespace['__import__'] = local_import
    builder = clone_globals(base.build_agent, {'__builtins__': namespace})
    result = builder(config, port)
    if roles != ['actor'] + ['critic'] * 6:
        raise ValueError(f'Unexpected head construction order/roles: {roles}')
    return result


def make_probe(fns, debug, dim, arm):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    original = preceding.make_probe(fns, debug, dim, False)
    h = captures(captures(fns.update)['loss_combined'])
    main = captures(h['loss_critic'])
    graphs = (main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph'])
    run_q = captures(main['get_q_both'])['run_critic']

    @jax.jit
    def probe(state, key):
        metrics, indices = original(state, key)
        raw = {k: v[indices] for k, v in state.online_buffer.info.items()}
        x, y = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
            raw['observations'], raw['actions'], raw['terminated'], jnp.zeros((h['params'].online_batch_size, dim)))
        keep = h['params'].sequence_length
        flat = lambda z: z[:keep].reshape(-1, z.shape[-1])
        x, y = flat(x), flat(y)
        for tag, attr, graph in zip(('main', 'lambda1', 'lambda2'),
                ('critic', 'lambda1_critic', 'lambda2_critic'), graphs):
            for label, critic, z in (('online', getattr(state, attr), x),
                    ('target', getattr(state, attr + '_target'), y)):
                for branch in ('q1', 'q2'):
                    head = nnx.merge(getattr(graph, branch), getattr(critic, branch))
                    hidden = z
                    for i, layer in enumerate(head.layers[:-1]):
                        hidden = layer(hidden)
                        if head.norms:
                            hidden = head.norms[i](hidden)
                        hidden = nnx.relu(hidden)
                    used = normalized_features(hidden) if arm == 'ofn' else hidden
                    prefix = f'{tag}_{label}_{branch}_'
                    metrics[prefix + 'raw_hidden_norm_mean'] = jnp.mean(jnp.linalg.norm(hidden, axis=-1))
                    metrics[prefix + 'raw_hidden_norm_min'] = jnp.min(jnp.linalg.norm(hidden, axis=-1))
                    metrics[prefix + 'epsilon_floor_fraction'] = jnp.mean(jnp.sum(hidden * hidden, axis=-1) < EPSILON ** 2)
                    metrics[prefix + 'output_feature_norm_max'] = jnp.max(jnp.linalg.norm(used, axis=-1))
                    metrics[prefix + 'output_weight_action_norm_max'] = jnp.max(jnp.linalg.norm(head.layers[-1].kernel.value, axis=0))
                    metrics[prefix + 'output_bias_abs_max'] = jnp.max(jnp.abs(head.layers[-1].bias.value))
            q = run_q(getattr(state, attr), graph, x)
            metrics[tag + '_fraction_abs_q_over_100'] = jnp.mean(jnp.abs(q) > 100.)
        return metrics, indices
    return probe


def guard(root):
    from popgym_parallel import current_port_fingerprint, runner_fingerprint
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Normalization plan changed')
    if p['script_sha256'] != file_hash(__file__) or p['sources'] != source_fingerprint():
        raise ValueError('Normalization/core sources changed')
    if p['runner_sha256'] != runner_fingerprint() or p['port_sha256'] != current_port_fingerprint():
        raise ValueError('Runner/port changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Normalization dependency changed')
    if any(importlib.metadata.version(name) != version for name, version in p['versions'].items()):
        raise ValueError('Normalization packages changed')
    predecessor, sweep = preceding.guard(Path(p['preceding_root']))
    if file_hash(Path(p['preceding_root']) / 'plan.json') != p['preceding_plan_sha256']:
        raise ValueError('Predecessor plan changed')
    if predecessor['steps'] != p['steps'] or predecessor['alpha'] != p['alpha']:
        raise ValueError('Pilot settings differ from predecessor')
    verify_plan(sweep)
    return p, sweep


def predecessor_evidence(p):
    """No training until all ten original runs and final holdouts are complete."""
    root = Path(p['preceding_root'])
    predecessor, _ = preceding.guard(root)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
    if status.get('state') == 'failed':
        raise ValueError('Alignment predecessor failed; inspect before proceeding')
    if status.get('state') != 'complete':
        return None
    if len(predecessor['jobs']) != 10:
        raise ValueError('Expected all ten alignment runs')
    evidence = []
    for job in predecessor['jobs']:
        arm, seed = job['arm'], job['seed']
        folder = root / arm / 'popgym-BattleshipEasy-v0' / 'lambda' / f'seed{seed}'
        status = json.loads((folder / 'status.json').read_text())
        history = json.loads((folder / 'history.json').read_text())
        if status['state'] != 'complete' or status['steps'] != p['steps']:
            raise ValueError('Predecessor supervisor complete but a run is incomplete')
        if not history[-1]['final'] or history[-1]['steps'] != p['steps'] or 'random' not in history[-1]['eval']:
            raise ValueError('Missing original final holdout/random reference')
        paths = [folder / 'checkpoint.pkl', folder / f"episodes_{p['steps']}_final.json",
            root / 'diagnostics' / arm / f'seed{seed}' / f"step{p['steps']}.pkl"]
        evidence.append({'arm': arm, 'seed': seed,
            'sha256': {str(path.relative_to(root)): file_hash(path) for path in paths}})
    return evidence


def prepare(root, predecessor=DEFAULT_PREDECESSOR):
    from popgym_parallel import current_port_fingerprint, runner_fingerprint
    old, sweep = preceding.guard(predecessor)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Fresh normalization root required')
    dependencies = ('popgym_online_alignment.py', 'popgym_critic_settling.py', 'popgym_gradient_balance.py',
        'popgym_alpha_sweep.py', 'popgym_critic_aligned.py')
    p = {'created_at': datetime.now().astimezone().isoformat(), 'script_sha256': file_hash(__file__),
        'sources': source_fingerprint(), 'runner_sha256': runner_fingerprint(), 'port_sha256': current_port_fingerprint(),
        'preceding_root': str(predecessor.resolve()), 'preceding_plan_sha256': file_hash(predecessor / 'plan.json'),
        'dependencies': {name: file_hash(Path(__file__).with_name(name)) for name in dependencies},
        'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym', 'wandb')},
        'jobs': [{'arm': arm, 'seed': seed} for seed in range(42, 47) for arm in ARMS],
        'steps': old['steps'], 'alpha': old['alpha'], 'diagnostic_minibatches': 2,
        'epsilon': EPSILON, 'wandb_project': 'lambda-popgym-critic-normalization', 'wandb_entity': sweep['wandb_entity'],
        'normalization': 'Per-example L2 unit norm of final post-ReLU hidden activation immediately before output Linear, in all six main/lambda1/lambda2 critic branches, online and target. sqrt(max(sum(x*x), epsilon**2)), epsilon1e-8, no learned scale/bias or added RNG/parameters. Actor and shared GRU untouched. NOT LayerNorm, hard Q bounding, or bootstrap Q-input alignment.',
        'protocol': 'Wait for ALL ten alignment runs/final holdouts. Then ten FRESH Battleship Easy JAX lambda fits, baseline/OFN paired seeds42-46,250k steps plus original random prefill. Original sweep suite_args alpha.01 beta.1 gamma.99 tau.005 GRU512/projection128/head512, FE LR1e-5/head actor LR1e-4, batch128/replay100k/windows10/40/50, prev-action, reward-only targets and stopped actor-to-FE gradients. All original losses, original online/target encoder inputs, target policy/ratios, samplers, optimizer/EMA cadence, native CUDA precision and RNG streams retained. Private cloned factory builds OFN graphs; no package or predecessor edits. Initial parameter arrays paired exactly; initial Q predictions need not match. Fresh baseline, no archived-control splicing. Two read-only probes and immutable states every25k, original validation/final holdouts/W&B. Serial GPU0 workers, host cap8G; no prior-service pause/restart or automatic extra conditions/budget escalation.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(root / 'status.json', {'state': 'queued', 'completed': 0, 'total': len(p['jobs']), 'preceding_root': p['preceding_root']})
    render(root)


def worker(root, index):
    import cloudpickle
    import jax
    import numpy as np
    from unittest.mock import patch
    import popgym_parallel as parallel
    import popgym_jax as backend
    from lambda_imitation.buffer import create_buffer
    from popgym_suite import run_config
    p, sweep = guard(root); job = p['jobs'][index]; arm, seed = job['arm'], job['seed']
    gate = json.loads((root / 'completion-gate.json').read_text())
    if not gate.get('passed') or gate['preceding_plan_sha256'] != p['preceding_plan_sha256'] or len(gate['finals']) != 10:
        raise ValueError('No passing predecessor completion gate')
    (root / 'progress').mkdir(exist_ok=True)
    args = args_from_plan(sweep)
    args.alpha = p['alpha']; args.steps = p['steps']; args.output_dir = root / arm
    args.resume = False; args.wandb_project = p['wandb_project']; args.wandb_entity = p['wandb_entity']
    task = {'env_id': 'popgym-BattleshipEasy-v0', 'method': 'lambda', 'backend': 'jax',
        'seed': seed, 'target_steps': p['steps'], 'expected_source_sha256': p['sources'],
        'critic_normalization_arm': arm, 'critic_normalization_source_sha256': p['script_sha256'],
        'critic_normalization_epsilon': p['epsilon']}
    cache = {}; original_result = backend.evaluation_result

    def initialize(args, task, port):
        state, fns, debug = build_agent(run_config(args, task), port, arm)
        dim = args.memory_hidden_dim; probe = make_probe(fns, debug, dim, arm)
        _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
        runner = backend.JaxRunner(args, port, fns, debug, bfns, dim, True)
        evaluate = runner.evaluate
        def wrapped(state, eval_seed, mode, n_episodes):
            outputs = evaluate(state, eval_seed, mode, n_episodes)
            if mode == 'stochastic':
                step = int(state.update_step); measurements = []
                folder = root / 'diagnostics' / arm / f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
                for batch in range(p['diagnostic_minibatches']):
                    key = jax.random.fold_in(jax.random.PRNGKey(seed + 9000000), step * 2 + batch)
                    stats, indices = jax.device_get(probe(state, key)); row = {k: float(v) for k, v in stats.items()}
                    if not all(np.isfinite(v) for v in row.values()):
                        raise FloatingPointError('Nonfinite normalization probe')
                    if arm == 'ofn' and any(v > 1.00001 for k, v in row.items() if k.endswith('output_feature_norm_max')):
                        raise AssertionError('OFN output features exceed unit norm')
                    measurements.append({'metrics': row, 'indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})
                cache['metrics'] = {k: float(np.mean([r['metrics'][k] for r in measurements])) for k in measurements[0]['metrics']}
                cache['steps'] = step
                report = folder / f'step{step}.json'; content = {'step': step, 'arm': arm, 'seed': seed, 'samples': measurements}
                if report.exists():
                    if json.loads(report.read_text()) != content:
                        raise ValueError('Repeated final normalization probe differs')
                else:
                    checkpoint = folder / f'step{step}.pkl'; tmp = checkpoint.with_suffix('.tmp')
                    with tmp.open('wb') as file:
                        cloudpickle.dump({'state': jax.device_get(state), 'steps': step,
                            'config': run_config(args, task), 'plan_sha256': file_hash(root / 'plan.json')}, file)
                    tmp.replace(checkpoint)
                    write_json(report, {**content})
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
    web = Path('/tmp/opencode/popgym-report-web/critic-normalization'); web.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Battleship critic output-feature normalization</h1><p>Five fresh paired baseline/OFN seeds after all ten alignment runs finish. Unit-normalize final hidden features of all main and auxiliary critic branches; actor, memory and original target construction unchanged. Not hard Q bounding or Q-input alignment. No oracle-Q claim.</p>'
    text += '<pre>' + html.escape(json.dumps(status, indent=2)) + '</pre><table border="1"><tr><th>Arm/seed</th><th>Status</th><th>Steps</th><th>Stochastic return</th><th>Q-vs-MC RMSE</th><th>Main Q RMS</th><th>Auxiliary Q RMS</th></tr>'
    for path in sorted(root.glob('progress/*.json')):
        r = json.loads(path.read_text()); m = r['metrics']
        values = (f"{r['arm']}/seed{r['seed']}", r['state'], r['steps'], m.get('return_mean', ''),
            m.get('q_mc_rmse/all', ''), m.get('diagnostics/main_q_rms_all_actions_twins', ''),
            [m.get('diagnostics/' + k + '_q_rms_all_actions_twins', '') for k in ('lambda1', 'lambda2')])
        text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    tmp = web / 'index.tmp'; tmp.write_text(text + '</table>'); tmp.replace(web / 'index.html')


def supervise(root):
    p, _ = guard(root)
    proof = json.loads((root / 'validation.json').read_text())
    if not proof.get('passed') or proof['script_sha256'] != p['script_sha256']:
        raise ValueError('No matching passing CPU/CUDA validation certificate')
    write_json(root / 'status.json', {'state': 'waiting', 'completed': 0, 'total': len(p['jobs']), 'preceding_root': p['preceding_root']}); render(root)
    # Authorized application-side queue; does not poll/restart/pause old units.
    while (evidence := predecessor_evidence(p)) is None:
        guard(root); time.sleep(60)
    write_json(root / 'completion-gate.json', {'passed': True, 'checked_at': datetime.now().astimezone().isoformat(),
        'preceding_plan_sha256': p['preceding_plan_sha256'], 'finals': evidence})
    for index, job in enumerate(p['jobs']):
        guard(root); write_json(root / 'status.json', {'state': 'running', 'completed': index, 'total': len(p['jobs']), 'job': job}); render(root)
        with (root / f"{job['arm']}-seed{job['seed']}.log").open('x') as file:
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
