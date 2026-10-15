"""Fresh 1M Battleship OFN + complete-episode BPTT + predefined alpha decay.

Experiment-local masked versions of original objectives, unchanged optimizers,
EMA, reward-only targets and networks. No production/predecessor source edits.
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

import popgym_actor_critic_audit as audit
import popgym_critic_normalization as normalization
from popgym_alpha_sweep import args_from_plan, file_hash
from popgym_critic_aligned import replace_captures
from popgym_gradient_balance import captures
from popgym_suite import write_json, run_config


DEFAULT_ROOT = Path('/tmp/opencode/popgym-full-episode')
HORIZON = 64
MC_STEPS = (100000, 200000, 250000, 300000, 400000, 500000,
    600000, 700000, 800000, 900000, 1000000)


def alpha_at(step):
    import jax.numpy as jnp
    return jnp.exp(jnp.log(jnp.float32(.01)) * jnp.clip(jnp.asarray(step) / 500000., 0., 1.))


def episode_candidates(buffer, horizon=HORIZON):
    """Chronological complete episodes only; no invented oldest ring start."""
    import jax
    import jax.numpy as jnp
    N = buffer.info['terminated'].shape[0]; t = jnp.arange(N)
    count = jnp.minimum(buffer.pos, N); oldest = jnp.maximum(buffer.pos - N, 0)
    order = (oldest + t) % N; occupied = t < count
    done = buffer.info['terminated'][order].astype(bool) & occupied
    ends = jax.lax.associative_scan(jnp.minimum, jnp.where(done, t, N), reverse=True)
    starts = jnp.concatenate((jnp.asarray([buffer.pos <= N]), done[:-1])) & occupied
    lengths = ends - t + 1
    valid = starts & (ends < count) & (lengths <= horizon)
    return order, valid, lengths


def make_sampler(batch_size, horizon=HORIZON):
    import jax
    import jax.numpy as jnp
    from lambda_imitation.buffer import BufferSample
    def sample(buffer, key):
        order, candidates, lengths = episode_candidates(buffer, horizon)
        probabilities = candidates.astype(jnp.float32)
        # No fallback to arbitrary windows if no complete episode exists.
        probabilities /= probabilities.sum()
        starts = jax.random.choice(key, len(order), (batch_size,), p=probabilities)
        indices = order[(starts[:, None] + jnp.arange(horizon + 1)) % len(order)]
        valid = jnp.arange(horizon + 1)[None, :] < lengths[starts, None]
        raw = {}
        for k, v in buffer.info.items():
            mask = valid.reshape((*valid.shape, *((1,) * (v.ndim - 1))))
            fill = 1. if k in ('terminated', 'behaviour_weight') else 0.
            raw[k] = jnp.where(mask, v[indices], fill)
        raw['episode_valid'] = valid; raw['episode_length'] = lengths[starts]
        return BufferSample(raw, {}), indices
    return sample


def masked_objective(fns, debug, sampler):
    """Original actor/TD/Retrace/discrepancy, normalized over real transitions.

    Batch samples episodes uniformly; scalar objectives weight real transitions
    equally, excluding the dummy successor and any early-terminal padding.
    """
    import jax
    import jax.numpy as jnp
    from lambda_imitation.iqlearn import retrace_targets
    h = captures(captures(fns.update)['loss_combined']); params = h['params']
    if not params.retrace or not params.stop_actor_fe or params.burn_in_length != 0 or h['use_gvd'] or h['mask_fn'] is not None:
        raise ValueError('Unexpected episode pilot settings')
    low = captures(h['loss_vtrace_lambda_sequence']); main = captures(h['loss_critic'])
    actor = captures(debug.get_entropy)['run_actor']
    run_q = captures(main['get_q_both'])['run_critic']
    graphs = (main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph'])
    flat = lambda x: x.reshape(-1, *x.shape[2:])

    def components(fe, actor_state, critic, l1, l2, fe_target, actor_target,
            critic_target, l1_target, l2_target, alpha, raw, keys):
        valid = raw['episode_valid'].T[:-1].astype(jnp.float32)
        denominator = jnp.maximum(valid.sum(), 1.)
        mean = lambda v: (v.reshape(valid.shape) * valid).sum() / denominator
        x, y = debug.calculate_latent(fe, fe_target, raw['observations'], raw['actions'],
            raw['terminated'], jnp.zeros((params.online_batch_size, h['feature_extractor'].initialize_carry(1).shape[-1])))
        act, reward, done, behaviour = [jnp.swapaxes(raw[k], 0, 1) for k in
            ('actions', 'rewards', 'terminated', 'behaviour_weight')]
        z = flat(jax.lax.stop_gradient(x[:-1])); probs = jax.nn.softmax(actor(actor_state, z))
        logp = jax.nn.log_softmax(actor(actor_state, z))
        stopped_q = run_q(jax.lax.stop_gradient(critic), graphs[0], z).min(-1)
        entropy = -jnp.sum(probs * logp, -1)
        v = main['get_v'](actor_state, jax.lax.stop_gradient(critic), graphs[0], alpha,
            z, keys[1], include_entropy=True)
        actor_loss = -mean(v)
        next_v = main['get_v'](actor_target, critic_target, graphs[0], alpha, flat(y[1:]),
            keys[2], include_entropy=False)
        target = jax.lax.stop_gradient(reward[:-1].reshape(-1) + params.gamma *
            (1. - done[:-1].reshape(-1)) * next_v)
        q1, q2 = main['get_q_both'](critic, graphs[0], flat(x[:-1]), flat(act[:-1]))
        critic_loss = .5 * (mean((q1 - target) ** 2) + mean((q2 - target) ** 2))
        metrics = {'q': mean(jnp.sum(probs * stopped_q, -1)), 'entropy': mean(entropy),
            'v': mean(v), 'critic_loss': critic_loss, 'target_q': mean(target), 'alpha': alpha,
            'episode_mean_length': raw['episode_length'].mean(), 'valid_fraction': valid.mean()}
        losses = [actor_loss, critic_loss]
        for name, lam, online, target_state, graph, key in zip(('lambda1', 'lambda2'),
                (params.lambda1, params.lambda2), (l1, l2), (l1_target, l2_target), graphs[1:], keys[3:]):
            Tp, B = x.shape[:2]
            value = low['get_v'](actor_target, target_state, graph, jnp.float32(0), flat(y), key, False, False).reshape(Tp, B)
            ratios = low['get_importance_ratios'](actor_target, flat(y), flat(act), flat(behaviour)).reshape(Tp, B)
            target_taken = low['get_q'](target_state, graph, flat(y), flat(act)).reshape(Tp, B)
            returns = jax.lax.stop_gradient(retrace_targets(target_taken, value, reward, done, ratios,
                params.gamma, lam, params.c_bar))[:-1]
            a, b = low['get_q_both'](online, graph, flat(x), flat(act))
            a, b = a.reshape(Tp, B)[:-1], b.reshape(Tp, B)[:-1]
            loss = .5 * (mean((a - returns) ** 2) + mean((b - returns) ** 2))
            losses.append(loss)
            metrics[name + '_loss'] = loss; metrics[name + '_target'] = mean(returns)
        # Production discrepancy freezes head parameters,not feature inputs.
        # Keep its separate forwards: reusing regression outputs would train
        # the heads on disagreement and change the algorithm's gradient flow.
        ld1 = low['get_q'](jax.lax.stop_gradient(l1), graphs[1], flat(x[:-1]), flat(act[:-1]))
        ld2 = low['get_q'](jax.lax.stop_gradient(l2), graphs[2], flat(x[:-1]), flat(act[:-1]))
        discrepancy = mean((ld1 - ld2) ** 2)
        losses.append(params.lambda_coef * discrepancy); metrics['ld_loss'] = discrepancy
        return sum(losses), metrics, jnp.stack(losses)

    def combined(fe, actor_state, critic, l1, l2, sf1, sf2, fe_target, actor_target,
            critic_target, l1_target, l2_target, sf1_target, sf2_target, alpha, buffer, key):
        keys = jax.random.split(key, 5); sample, _ = sampler(buffer, keys[0])
        loss, metrics, _ = components(fe, actor_state, critic, l1, l2, fe_target,
            actor_target, critic_target, l1_target, l2_target, alpha, sample.this_info, keys)
        return loss, metrics
    return combined, components, h


def build(config, port):
    import jax
    import jax.numpy as jnp
    state, fns, debug = normalization.build_agent(config, port, 'ofn')
    sampler = make_sampler(config['batch_size'])
    combined, components, context = masked_objective(fns, debug, sampler)
    original_update = replace_captures(fns.update, {'loss_combined': combined})
    @jax.jit
    def update(state, key):
        alpha = alpha_at(state.update_step)
        state, metrics = original_update(state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), key)
        alpha = alpha_at(state.update_step)
        return state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), metrics
    return state, fns._replace(update=update), debug, sampler, components, context


def probe(state, debug, sampler, context, key):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    actor = captures(debug.get_entropy)['run_actor']; main = captures(context['loss_critic'])
    run_q = captures(main['get_q_both'])['run_critic']
    sample, _ = sampler(state.online_buffer, key); raw = sample.this_info
    with jax.default_matmul_precision('float32'):
        x, y = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
            raw['observations'], raw['actions'], raw['terminated'],
            jnp.zeros((context['params'].online_batch_size, context['feature_extractor'].initialize_carry(1).shape[-1])))
        valid = np.asarray(raw['episode_valid'].T[:-1]); flat = lambda a: a[:-1].reshape(-1, a.shape[-1])
        q = run_q(state.critic, main['critic_graph'], flat(x)).reshape((*valid.shape, 64, 2))
        probs = jax.nn.softmax(actor(state.actor, flat(x))).reshape((*valid.shape, 64))
        actions = np.asarray(raw['actions'][..., 0]).astype(int)
        seen = np.zeros((*valid.T.shape, 64), bool)
        for t in range(1, seen.shape[1]):
            seen[:, t] = seen[:, t - 1]; seen[np.arange(len(seen)), t, actions[:, t - 1]] = True
        grounding = jax.device_get(audit.grounding_metrics(probs, q.min(-1), jnp.asarray(seen.transpose(1, 0, 2)), state.alpha))
        keep = valid & seen.any(-1).T
        metrics = {'replay/' + k: v for k, v in audit.summarize_grounding(grounding, keep)['means'].items()}
        metrics['episode_prefix_coverage'] = 1.; metrics['mean_sample_episode_length'] = float(np.asarray(raw['episode_length']).mean())
        conservative = []
        for name, attr, graph in zip(('main', 'lambda1', 'lambda2'), ('critic', 'lambda1_critic', 'lambda2_critic'),
                (main['critic_graph'], context['lambda1_critic_graph'], context['lambda2_critic_graph'])):
            values = np.asarray(run_q(getattr(state, attr), graph, flat(x)).reshape((*valid.shape, 64, 2)))[valid]
            metrics[name + '_q_rms'] = float(np.sqrt(np.mean(values ** 2)))
            metrics[name + '_q_max_abs'] = float(np.max(np.abs(values)))
            conservative.append(values.min(-1))
            for label, z in (('online', x), ('target', y)):
                heads = getattr(state, attr + ('_target' if label == 'target' else ''))
                for branch in ('q1', 'q2'):
                    head = nnx.merge(getattr(graph, branch), getattr(heads, branch))
                    hidden, used = head.output_features(flat(z))
                    hidden, used = np.asarray(hidden).reshape((*valid.shape, -1))[valid], np.asarray(used).reshape((*valid.shape, -1))[valid]
                    tag = f'{name}_{label}_{branch}_'
                    metrics[tag + 'hidden_norm_mean'] = float(np.linalg.norm(hidden, axis=-1).mean())
                    metrics[tag + 'ofn_norm_max'] = float(np.linalg.norm(used, axis=-1).max())
                    metrics[tag + 'weight_norm_max'] = float(np.linalg.norm(np.asarray(head.layers[-1].kernel.value), axis=0).max())
                    metrics[tag + 'bias_abs_max'] = float(np.abs(np.asarray(head.layers[-1].bias.value)).max())
                    if metrics[tag + 'ofn_norm_max'] > 1.00001:
                        raise AssertionError('OFN normalization certificate failed')
        metrics['discrepancy_rms_all_actions'] = float(np.sqrt(np.mean((conservative[1] - conservative[2]) ** 2)))
        metrics['online_target_feature_relative_l2'] = float(np.linalg.norm(np.asarray(x[:-1] - y[:-1])[valid]) /
            max(np.linalg.norm(np.asarray(x[:-1])[valid]), 1e-8))
    return metrics


def measurements(state, fns, debug, port, sampler, context, seed, step, folder, mc):
    """Own diagnostic RNGs only; caller certifies state digest before/after."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    memory = audit.memory; dim = context['feature_extractor'].initialize_carry(1).shape[-1]
    collect = memory.make_collect(port, fns, debug, dim); stats = {}; data = None
    modes = ('stochastic', 'greedy', 'random')
    for mode in modes:
        outputs = jax.device_get(collect(state, 17000000 + seed, mode=mode, episodes=100))
        if not outputs['history_label_parity'].all():
            raise AssertionError('Fired history labels disagree with environment')
        stats[mode] = memory.summarize_shots(outputs)
        if mode == 'stochastic':
            data = outputs
            np.savez_compressed(folder / 'stochastic-episodes.npz', **outputs)
    prefixes = audit.recover_prefixes(state, fns, debug, port, data, 17000000 + seed)
    host = jax.device_get(prefixes)
    np.testing.assert_array_equal(host['predicted_action'], data['actions'])
    for field, ref in (('obs', data['observations']), ('probs', data['probs']), ('q', data['q']), ('reward', data['rewards'])):
        np.testing.assert_allclose(host[field], ref, atol=1e-5, rtol=1e-4)
    valid = host['active'] & host['seen'].any(-1) & (~host['seen']).any(-1)
    grounding = jax.device_get(audit.grounding_metrics(prefixes['probs'], prefixes['q_twins'].min(-1), prefixes['seen'], state.alpha))
    result = {'step': step, 'alpha': float(state.alpha), 'shots': stats,
        'rollout_grounding': audit.summarize_grounding(grounding, valid),
        'replay': probe(state, debug, sampler, context, jax.random.PRNGKey(21000000 + seed + step)),
        'prefix_output_action_parity': True}
    if mc:
        selected = np.sort(np.random.default_rng(23000000 + seed).choice(100, 32, replace=False))
        points = audit.select_points(prefixes, selected); branch = audit.make_branches(fns, port, .99)
        arrays = {}; result['counterfactuals'] = {}
        for policy in audit.POLICIES:
            chunks = []
            for start in range(0, len(points['valid']), 16):
                part = jax.tree.map(lambda x: x[start:start + 16], points)
                values, ended = jax.device_get(branch(state, part, 25000000 + seed, replicas=32, policy=policy))
                if not ended[np.asarray(part['valid'])].all():
                    raise AssertionError('MC branch failed to reach terminal')
                chunks.append(values)
            values = np.concatenate(chunks); arrays[policy + '_returns'] = values
            result['counterfactuals'][policy] = audit.summarize_branches(points, values)
        np.savez_compressed(folder / 'counterfactuals.npz', **arrays,
            **{k: np.asarray(points[k]) for k in ('candidate_actions', 'predicted_q', 'episodes', 'ages', 'valid')})
    return result


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(__file__) != p['script_sha256'] or file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Full-episode source/plan changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Full-episode dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Packages changed')
    audit.guard(audit.DEFAULT_ROOT)
    if file_hash(audit.DEFAULT_ROOT / 'plan.json') != p['preceding_plan_sha256']:
        raise ValueError('Predecessor plan changed')
    return p


def config_for(seed, small=False):
    _, sweep = normalization.guard(audit.memory.PREDECESSOR)
    args = args_from_plan(sweep)
    args.alpha = 1.; args.steps = 1000000; args.burn_in_length = 0; args.sequence_length = 64; args.lambda_truncation = 1
    # Preserve original 12800 random interactions,not the new batch*65 default.
    args.prefill_steps = 12800
    if small:
        args.batch_size = 4; args.memory_hidden_dim = 8; args.projection_dim = 6; args.head_dim = 8; args.online_buffer_size = 256
    config = run_config(args, {'env_id': 'popgym-BattleshipEasy-v0', 'seed': seed, 'method': 'lambda', 'backend': 'jax',
        'critic_normalization_arm': 'ofn', 'episode_sampling': 'complete-uniform-episodes',
        'alpha_schedule': 'geometric1-to-.01-first500k-then-fixed'})
    return args, config


def prepare(root):
    audit.guard(audit.DEFAULT_ROOT)
    if json.loads((audit.DEFAULT_ROOT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Actor/critic audit must be complete')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Fresh output root required')
    _, config = config_for(42)
    p = {'created_at': datetime.now().astimezone().isoformat(), 'script_sha256': file_hash(__file__),
        'preceding_plan_sha256': file_hash(audit.DEFAULT_ROOT / 'plan.json'),
        'dependencies': {name: file_hash(Path(__file__).with_name(name)) for name in ('popgym_actor_critic_audit.py',
            'popgym_memory_diagnostics.py', 'popgym_critic_normalization.py', 'popgym_critic_aligned.py', 'popgym_jax.py')},
        'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym', 'wandb')},
        'config': config, 'jobs': [{'seed': seed} for seed in range(42, 47)], 'steps': 1000000,
        'eval_every': 25000, 'mc_steps': list(MC_STEPS), 'wandb_project': 'lambda-popgym-full-episode',
        'wandb_entity': 'fhstp-data-intelligence-research-group',
        'protocol': 'Five FRESH seeds42-46 serial GPU0,1M online steps plus original12800 random prefill/natural boundary. OFN in all six critic heads,actor/GRU unchanged. Complete known-start terminal episodes uniformly sampled from original100k replay; no burn-in,full BPTT up to64.65th slot is zero dummy successor; early terminal padding contributes NO actor/main/auxiliary/discrepancy loss. Same reward-only TD/Retrace(gamma.99,lambda.05/.85,beta.1),original target FE/policy and EMA/optimizers/LRs. Loss mean weights real transitions equally; no incomplete episode/unknown prefix/newest fragment/ring-edge samples. Alpha geometric1 to.01 over first500k ONLINE updates(one update/interaction),then fixed. Original16 collect/update block,persistent acting carry/prev-action,128 episodes/batch,GRU512/proj128/head512. Native training precision;diagnostic-only replay float32. Every25k:immutable complete state,100 stochastic/greedy/random episodes,shots/coverage/entropy/actor-vs-main-Q optimum,replay same metrics,Q magnitudes and encoder mismatch. Matched current/EMA32episode x three ages x three actions x32 paired MC futures at100k/200k/250k/every100k afterward. All diagnostics separate RNG,state digest before/after,checkpoint hashes. Same-histories expected values not exact representation-conditional oracle Q. Original final100 holdouts/random refs,separate W&B IDs/project and report. Combined exploratory condition,NOT causal single-factor/matched-compute control. No baseline splicing,smoke reuse,production edits,automatic schedule/budget/intervention escalation.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes()); write_json(root / 'status.json', {'state': 'prepared', 'completed': 0, 'total': 5}); render(root)


def worker(root, index):
    import cloudpickle
    import jax
    import numpy as np
    import wandb
    from lambda_imitation.buffer import create_buffer
    from popgym_jax import PortEnv, JaxRunner, evaluation_result
    p = guard(root); seed = p['jobs'][index]['seed']; args, config = config_for(seed)
    folder = root / f'seed{seed}'; folder.mkdir(exist_ok=False)
    write_json(folder / 'config.json', config)
    port = PortEnv(config['env_id']); state, fns, debug, sampler, components, context = build(config, port)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
    runner = JaxRunner(args, port, fns, debug, bfns, args.memory_hidden_dim, True)
    rollout = runner.initial_rollout(seed)
    run = wandb.init(project=p['wandb_project'], entity=p['wandb_entity'], id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16],
        resume='never', config=config, name=f'Battleship-full-episode-seed{seed}', dir=str(folder))
    run.define_metric('steps'); run.define_metric('*', step_metric='steps')
    history = []; step = 0; interactions = 0
    try:
        state, rollout, count = jax.block_until_ready(runner.prefill(state, rollout, 12800)); interactions = int(count)
        if not np.asarray(episode_candidates(state.online_buffer)[1]).any():
            raise ValueError('No complete prefill episodes')
        while True:
            guard(root); checkpoint = folder / 'checkpoint.pkl'; tmp = checkpoint.with_suffix('.tmp')
            payload = {'state': jax.device_get(state), 'rollout': jax.device_get(rollout), 'config': config,
                'steps': step, 'interactions': interactions, 'history': history, 'plan_sha256': file_hash(root / 'plan.json')}
            with tmp.open('wb') as file: cloudpickle.dump(payload, file)
            tmp.replace(checkpoint)
            snapfolder = root / 'diagnostics' / f'seed{seed}' / f'step{step}'; snapfolder.mkdir(parents=True, exist_ok=False)
            snapshot = snapfolder / 'state.pkl'; snapshot.write_bytes(checkpoint.read_bytes())
            before = audit.memory.tree_digest(state); snapshot_hash = file_hash(snapshot)
            result = measurements(state, fns, debug, port, sampler, context, seed, step, snapfolder, step in MC_STEPS)
            after = audit.memory.tree_digest(state)
            if before != after or file_hash(snapshot) != snapshot_hash:
                raise AssertionError('Diagnostics modified training state/snapshot')
            result['certificates'] = {'state_digest_before': before, 'state_digest_after': after, 'snapshot_sha256': snapshot_hash}
            write_json(snapfolder / 'result.json', result)
            final = step == p['steps']; eval_seed = seed + (2000000 if final else 1000000)
            modes = ('greedy', 'stochastic', 'critic', 'random') if final else ('greedy', 'stochastic', 'critic')
            evaluations = {mode: evaluation_result(jax.device_get(runner.evaluate(state, eval_seed, mode,
                args.final_episodes if final else args.eval_episodes)), mode, args, True) for mode in modes}
            write_json(folder / f'episodes_{step}{"_final" if final else ""}.json', evaluations)
            entry = {'steps': step, 'interactions_including_prefill': interactions, 'alpha': float(state.alpha), 'final': final,
                'eval': {mode: e['metrics'] for mode, e in evaluations.items()}, 'diagnostic_return': result['shots']['stochastic']['return_mean'],
                'repeat_fraction': result['shots']['stochastic']['repeat_fraction'], 'unique_cells': result['shots']['stochastic']['unique_shots_mean']}
            history.append(entry); write_json(folder / 'history.json', history)
            log = {'steps': step, 'alpha': float(state.alpha),
                **{f'eval/{m}/{k}': v for m, e in evaluations.items() for k, v in e['metrics'].items()},
                **{f'shots/{m}/{k}': v for m, values in result['shots'].items() for k, v in values.items() if isinstance(v, (int, float))},
                **{'grounding/' + k: v for k, v in result['rollout_grounding']['means'].items()}, **result['replay']}
            if 'counterfactuals' in result:
                for policy, mc in result['counterfactuals'].items():
                    log.update({f'mc/{policy}/{k}': v for k, v in mc.items() if isinstance(v, (int, float))})
                    log.update({f'mc/{policy}/{name}/{k}': v for name, values in mc['action_classes'].items() for k, v in values.items()})
            run.log(log)
            write_json(folder / 'status.json', {'state': 'complete' if final else 'running', **entry})
            # Refresh latest checkpoint with the newly persisted evaluation/history.
            with tmp.open('wb') as file: cloudpickle.dump({**payload, 'history': history}, file)
            tmp.replace(checkpoint); render(root)
            print('EVAL', seed, entry, flush=True)
            if final: break
            boundary = min(p['steps'], step + p['eval_every'])
            while step < boundary:
                count = min(args.jax_train_steps, boundary - step)
                state, rollout, metrics = jax.block_until_ready(runner.train(state, rollout, count))
                if not all(np.isfinite(np.asarray(v)).all() for v in metrics.values()): raise FloatingPointError('Nonfinite training metrics')
                if np.asarray(rollout[-2]): raise RuntimeError('Episode horizon exceeded')
                step += count; interactions += count
            run.log({'steps': step, **{'train/' + k: float(v) for k, v in metrics.items()}})
        guard(root)
    except BaseException:
        import traceback
        write_json(folder / 'status.json', {'state': 'failed', 'steps': step, 'error': traceback.format_exc()}); raise
    finally: run.finish()


def render(root):
    web = Path('/tmp/opencode/popgym-report-web/full-episode'); web.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Battleship 1M: OFN + full episodes + alpha decay</h1><p>Five fresh serial seeds. Complete-episode BPTT,alpha1→.01 by500k then fixed. Combined exploratory condition,not a causal single-factor comparison.</p><pre>' + html.escape(json.dumps(status, indent=2)) + '</pre><table border="1"><tr><th>Seed</th><th>Steps</th><th>Alpha</th><th>Return</th><th>Repeats</th><th>Unique cells</th></tr>'
    for path in sorted(root.glob('seed*/history.json')):
        for e in json.loads(path.read_text()):
            values = (path.parent.name, e['steps'], e['alpha'], e['diagnostic_return'], e['repeat_fraction'], e['unique_cells'])
            text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    tmp = web / 'index.tmp'; tmp.write_text(text + '</table>'); tmp.replace(web / 'index.html')


def supervise(root):
    p = guard(root); proof = json.loads((root / 'validation.json').read_text())
    if not proof.get('passed') or proof['script_sha256'] != p['script_sha256']: raise ValueError('No matching validation')
    for index, job in enumerate(p['jobs']):
        guard(root); write_json(root / 'status.json', {'state': 'running', 'completed': index, 'total': len(p['jobs']), 'job': job}); render(root)
        with (root / f"seed{job['seed']}.log").open('x') as file:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', str(index)],
                stdout=file, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    guard(root); write_json(root / 'status.json', {'state': 'complete', 'completed': 5, 'total': 5,
        'finished_at': datetime.now().astimezone().isoformat()}); render(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int)
    args = parser.parse_args(); root = args.root.resolve()
    if args.prepare: prepare(root)
    else:
        try: worker(root, args.worker) if args.worker is not None else supervise(root)
        except BaseException:
            import traceback
            write_json(root / 'status.json', {'state': 'failed', 'error': traceback.format_exc()}); render(root); raise


if __name__ == '__main__': main()
