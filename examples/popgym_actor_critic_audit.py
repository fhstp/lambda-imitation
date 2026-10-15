"""Read-only actor/main-critic grounding and paired counterfactual returns.

Frozen OFN policies; no training/intervention or hidden-state policy inputs.
Counterfactual MC branches evaluate actual current/EMA policy continuations,
not bootstrapped labels or exact representation-conditional Q oracles.
"""

import argparse
from datetime import datetime
from functools import partial
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import popgym_memory_diagnostics as memory
from popgym_gradient_balance import captures
from popgym_alpha_sweep import file_hash
from popgym_suite import write_json


DEFAULT_ROOT = Path('/tmp/opencode/popgym-actor-critic-audit')
AGES = (16, 32, 48)
ACTION_NAMES = ('actor-repeat', 'critic-fresh', 'actor-greedy')
POLICIES = ('current', 'target')


def grounding_metrics(probs, q, seen, alpha):
    """Actual discrete actor objective's optimum: softmax(min-twin Q / alpha)."""
    import jax.numpy as jnp
    import jax.scipy as jsp
    log_optimal = jax_logsoftmax(q / alpha)
    optimal = jnp.exp(log_optimal)
    entropy = -jnp.sum(jsp.special.xlogy(probs, probs), axis=-1)
    objective = jnp.sum(probs * q, axis=-1) + alpha * entropy
    optimum = alpha * jsp.special.logsumexp(q / alpha, axis=-1)
    actor_action = jnp.argmax(probs, -1); critic_action = jnp.argmax(q, -1)
    taken = jnp.take_along_axis(q, actor_action[..., None], -1)[..., 0]
    repeat_count = seen.sum(-1); fresh_count = (~seen).sum(-1)
    return {'actor_seen_mass': jnp.sum(probs * seen, -1),
        'critic_soft_optimum_seen_mass': jnp.sum(optimal * seen, -1),
        'actor_greedy_repeat': jnp.take_along_axis(seen, actor_action[..., None], -1)[..., 0],
        'critic_greedy_repeat': jnp.take_along_axis(seen, critic_action[..., None], -1)[..., 0],
        'actor_critic_greedy_agreement': actor_action == critic_action,
        'actor_greedy_q_regret': jnp.max(q, -1) - taken,
        'actor_within_0p1alpha_of_max_q': jnp.max(q, -1) - taken <= .1 * alpha,
        'soft_objective_gap': jnp.maximum(optimum - objective, 0.),
        'kl_actor_to_critic_soft_optimum': jnp.sum(jsp.special.xlogy(probs, probs) - probs * log_optimal, -1),
        'actor_critic_soft_optimum_l1': jnp.sum(jnp.abs(probs - optimal), -1),
        'max_fresh_minus_max_repeat_q': jnp.max(jnp.where(~seen, q, -1e9), -1) - jnp.max(jnp.where(seen, q, -1e9), -1),
        'uniform_fresh_minus_uniform_repeat_q': jnp.sum(q * ~seen, -1) / jnp.maximum(fresh_count, 1) - jnp.sum(q * seen, -1) / jnp.maximum(repeat_count, 1),
        'actor_entropy': entropy}


def jax_logsoftmax(x):
    import jax
    return jax.nn.log_softmax(x)


def summarize_grounding(metrics, mask):
    import numpy as np
    return {'positions': int(np.asarray(mask).sum()),
        'means': {k: float(np.asarray(v)[mask].mean()) for k, v in metrics.items()} if np.any(mask) else {}}


def recover_prefixes(state, fns, debug, port, data, seed):
    """Replay existing held-out actions on their original boards; no new choices."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    fe_graph = captures(debug.calculate_latent)['feature_extractor_graph']
    dim = nnx.merge(fe_graph, state.feature_extractor).initialize_carry(1).shape[-1]

    @jax.jit
    def recover(state, actions):
        def episode(index, actions):
            key = jax.random.PRNGKey(seed + index); key, rk = jax.random.split(key)
            obs, es = port.reset(rk)
            def body(c, action):
                key, obs, es, carry, prev, active = c
                key, ak, ek = jax.random.split(key, 3)
                values, nc = debug.predict_values(state, obs, carry, prev)
                # Keep original native evaluator's action/entropy/discrepancy
                # path alive. Target FE runs separately to avoid native GPU
                # fusion of online+target forwards changing online arithmetic.
                probs = values['probs']; q = values['q'].min(-1)
                predicted_action = jax.random.categorical(ak, jnp.log(probs + 1e-30))
                entropy = -(probs * jnp.log(probs + 1e-30)).sum()
                gap = values['lambda1_q'][action].min() - values['lambda2_q'][action].min()
                nobs, nes, reward, done = port.transition(ek, es, action)
                outputs = {'env_state': es, 'obs': obs, 'online_after_obs': nc,
                    'probs': probs, 'q_twins': values['q'], 'seen': es.guesses.reshape(-1).astype(bool),
                    'active': active, 'reward': jnp.where(active, reward, 0.), 'done': active & done,
                    'q': jnp.where(active, q[action], 0.), 'entropy': jnp.where(active, entropy, 0.),
                    'ld': jnp.where(active, gap, 0.), 'predicted_action': predicted_action}
                return (key, nobs, nes, nc, fns.encode_action(jnp.atleast_1d(action)), active & ~done), outputs
            return jax.lax.scan(body, (key, obs, es, jnp.zeros(dim),
                jnp.zeros(port.action_dim), jnp.bool_(True)), actions)[1]
        return jax.vmap(episode)(jnp.arange(actions.shape[0], dtype=jnp.uint32), actions)
    @jax.jit
    def recover_target(state, observations, actions):
        fe = nnx.merge(fe_graph, state.feature_extractor_target)
        def body(c, raw):
            carry, prev = c; obs, action = raw
            carry, _ = fe(carry, obs, prev)
            prev = jax.vmap(fns.encode_action)(action[:, None])
            return (carry, prev), carry
        _, carries = jax.lax.scan(body, (jnp.zeros((actions.shape[0], dim)),
            jnp.zeros((actions.shape[0], port.action_dim))), (observations.swapaxes(0, 1), actions.T))
        return carries.swapaxes(0, 1)
    result = recover(state, jnp.asarray(data['actions']))
    result['target_after_obs'] = recover_target(state, result['obs'], jnp.asarray(data['actions']))
    return result


def select_points(prefixes, episode_indices, ages=AGES):
    import jax
    import jax.numpy as jnp
    import numpy as np
    episodes = np.repeat(np.asarray(episode_indices), len(ages)); times = np.tile(np.asarray(ages), len(episode_indices))
    points = jax.tree.map(lambda x: x[episodes, times], prefixes)
    seen = points['seen']; q = points['q_twins'].min(-1); probs = points['probs']
    actions = jnp.stack((jnp.argmax(jnp.where(seen, probs, -1.), -1),
        jnp.argmax(jnp.where(~seen, q, -1e9), -1), jnp.argmax(probs, -1)), -1)
    points['candidate_actions'] = actions
    points['predicted_q'] = jnp.take_along_axis(q, actions, -1)
    points['episodes'] = jnp.asarray(episodes); points['ages'] = jnp.asarray(times)
    points['valid'] = points['active'] & seen.any(-1) & (~seen).any(-1)
    return points


def make_branches(fns, port, gamma):
    import jax
    import jax.numpy as jnp

    @partial(jax.jit, static_argnames=('replicas', 'policy'))
    def branches(state, points, seed, *, replicas, policy):
        agent = state if policy == 'current' else state._replace(
            feature_extractor=state.feature_extractor_target, actor=state.actor_target)
        carry_name = 'online_after_obs' if policy == 'current' else 'target_after_obs'
        def one_point(point):
            base = jax.random.fold_in(jax.random.PRNGKey(seed), point['episodes'] * 64 + point['ages'])
            keys = jax.random.split(base, replicas)
            def one_action(action):
                def rollout(key):
                    # Same future random numbers across candidate actions and
                    # current/target policies; never touches original RNG state.
                    key, ek = jax.random.split(key)
                    obs, es, reward, done = port.transition(ek, point['env_state'], action)
                    initial = (key, obs, es, point[carry_name], fns.encode_action(jnp.atleast_1d(action)),
                        ~done & point['valid'], reward, jnp.float32(gamma), done)
                    def body(c, _):
                        key, obs, es, carry, prev, active, total, discount, ended = c
                        key, ak, ek = jax.random.split(key, 3)
                        action, nc = fns.predict(agent, obs, carry, ak, prev_action=prev)
                        action = action.astype(jnp.int32)
                        no, ns, r, done = port.transition(ek, es, action)
                        total = total + discount * jnp.where(active, r, 0.)
                        return (key, no, ns, nc, fns.encode_action(jnp.atleast_1d(action)),
                            active & ~done, total, discount * gamma, ended | (active & done)), None
                    end, _ = jax.lax.scan(body, initial, None, length=port.horizon - min(AGES) - 1)
                    return end[6], end[8]
                return jax.vmap(rollout)(keys)
            return jax.vmap(one_action)(point['candidate_actions'])
        return jax.vmap(one_point)(points)
    return branches


def summarize_branches(points, returns, stratify=True):
    import numpy as np
    valid = np.asarray(points['valid']); episodes = np.asarray(points['episodes'])[valid]
    q = np.asarray(points['predicted_q'])[valid]; values = np.asarray(returns)[valid]
    means = values.mean(-1); paired = values[:, 1] - values[:, 0]
    empirical_gap = paired.mean(-1); predicted_gap = q[:, 1] - q[:, 0]
    within_se = paired.std(-1, ddof=1) / np.sqrt(paired.shape[-1])
    confident = np.abs(empirical_gap) > 2 * within_se
    cluster_gaps = np.asarray([empirical_gap[episodes == e].mean() for e in np.unique(episodes)])
    cluster_bias = np.asarray([(predicted_gap - empirical_gap)[episodes == e].mean() for e in np.unique(episodes)])
    result = {'valid_anchors': int(valid.sum()), 'episode_clusters': len(cluster_gaps),
        'mean_predicted_fresh_minus_repeat': float(predicted_gap.mean()),
        'mean_mc_fresh_minus_repeat': float(empirical_gap.mean()),
        'episode_cluster_gap_standard_error': float(cluster_gaps.std(ddof=1) / np.sqrt(len(cluster_gaps))),
        'mean_predicted_minus_mc_gap': float(cluster_bias.mean()),
        'episode_cluster_gap_bias_standard_error': float(cluster_bias.std(ddof=1) / np.sqrt(len(cluster_bias))),
        'mc_fresh_better_fraction': float((empirical_gap > 0).mean()),
        'predicted_fresh_better_fraction': float((predicted_gap > 0).mean()),
        'confident_mc_gap_fraction': float(confident.mean()),
        'wrong_sign_fraction_among_confident_mc_gaps': float((np.sign(predicted_gap[confident]) != np.sign(empirical_gap[confident])).mean()) if confident.any() else None,
        'action_classes': {name: {'mean_predicted_q': float(q[:, i].mean()), 'mean_mc_return': float(means[:, i].mean()),
            'mean_prediction_bias': float((q[:, i] - means[:, i]).mean())} for i, name in enumerate(ACTION_NAMES)},
        'reference_note': 'MC from sampled true hidden states under frozen current/EMA full-history policy continuations. No bootstrap/no policy mask. Per-state squared errors would include hidden-state variance, so not reported as representation-conditional oracle Q RMSE. Candidate fresh action selected by critic, not hidden board. Descriptive two-SE sign filter is not a multiplicity-adjusted significance test.'}
    if stratify:
        result['by_age'] = {}
        for age in np.unique(np.asarray(points['ages'])[valid]):
            mask = valid & (np.asarray(points['ages']) == age)
            result['by_age'][str(age)] = summarize_branches({**points, 'valid': mask}, returns, False)
    return result


def replay_grounding(state, fns, debug, dim, batches, key):
    import jax
    import jax.numpy as jnp
    import numpy as np
    h = captures(captures(fns.update)['loss_combined']); hp = h['params']
    actor = captures(debug.get_entropy)['run_actor']; main = captures(h['loss_critic'])
    run_q = captures(main['get_q_both'])['run_critic']; graph = main['critic_graph']
    meta = memory.chronological_metadata(state.online_buffer, 64)
    results = []; masks = []
    with jax.default_matmul_precision('float32'):
        for batch in range(batches):
            sample, indices = h['online_buffer_lambda_sample'](state.online_buffer, jax.random.fold_in(key, batch))
            raw = sample.this_info
            x, _ = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
                raw['observations'], raw['actions'], raw['terminated'], jnp.zeros((hp.online_batch_size, dim)))
            x = x[:hp.sequence_length]; shape = x.shape[:2]; flat = x.reshape(-1, dim)
            probs = jax.nn.softmax(actor(state.actor, flat)).reshape((*shape, 64))
            q = run_q(state.critic, graph, flat).reshape((*shape, 64, 2)).min(-1)
            learning = meta['inverse'][np.asarray(indices)][:, hp.burn_in_length:hp.burn_in_length + hp.sequence_length].T
            seen = meta['visited'][learning]; valid = meta['known'][learning] & seen.any(-1) & (~seen).any(-1)
            results.append(jax.device_get(grounding_metrics(probs, q, jnp.asarray(seen), state.alpha)))
            masks.append(valid)
    result = {k: np.concatenate([r[k].reshape(-1) for r in results]) for k in results[0]}
    return summarize_grounding(result, np.concatenate([m.reshape(-1) for m in masks]))


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if file_hash(__file__) != p['script_sha256'] or file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Actor/critic audit source/plan changed')
    for name, digest in p['dependencies'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Audit dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Audit packages changed')
    memory.guard(memory.DEFAULT_ROOT)
    if file_hash(memory.DEFAULT_ROOT / 'plan.json') != p['memory_plan_sha256']:
        raise ValueError('Memory audit plan changed')
    return p


def prepare(root):
    old = memory.guard(memory.DEFAULT_ROOT)
    if json.loads((memory.DEFAULT_ROOT / 'status.json').read_text())['state'] != 'complete':
        raise ValueError('Memory audit must be complete')
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists():
        raise FileExistsError('Fresh audit root required')
    jobs = []
    for job in old['jobs']:
        archive = memory.DEFAULT_ROOT / 'cases' / memory.job_name(job) / 'stochastic-episodes.npz'
        jobs.append({**job, 'episodes_archive': str(archive), 'episodes_sha256': file_hash(archive)})
    p = {'created_at': datetime.now().astimezone().isoformat(), 'script_sha256': file_hash(__file__),
        'memory_plan_sha256': file_hash(memory.DEFAULT_ROOT / 'plan.json'),
        'dependencies': {name: file_hash(Path(__file__).with_name(name)) for name in
            ('popgym_memory_diagnostics.py', 'popgym_critic_normalization.py', 'popgym_gradient_balance.py', 'popgym_jax.py')},
        'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym')},
        'jobs': jobs, 'anchor_episodes': 32, 'anchor_ages': list(AGES), 'replicas': 32, 'chunk_anchors': 16, 'replay_batches': 4,
        'protocol': '15 frozen OFN100k/200k/250k snapshots x five seeds. Reconstruct existing100 stochastic diagnostic episodes on original boards/actions, certify observations/rewards/probs/chosen main Q. Compare actor with exact softmax(min-online-twin-main-Q/alpha) optimum and Q rankings, repeated/fresh preference on actual full-history rollout positions and four ORIGINAL short replay minibatches. Main critic alone drives the production actor; auxiliary heads are not actor inputs.32 selected heldout episodes x ages16/32/48,three forced first actions:actor-most-probable repeat,critic-highest-Q fresh,actor-greedy.32 common-random-number full MC futures under BOTH frozen current and original EMA actor/encoder continuations, no bootstrap or masks, gamma.99. Native policy/prefix/branch forwards; replay-only grounding float32 for batch-shape stability. Hidden boards used only to simulate/evaluate counterfactual returns; never actor/FE/mask inputs. Aggregate paired fresh-minus-repeat gaps and calibration bias,MC uncertainty clustered by original episode; no exact history/representation-conditional oracle-Q or neural fixed-point claim. Input snapshot/archive/complete agent digests unchanged. No training,longer run,optimizer/RNG/W&B continuation,source edits or automatic escalation.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    write_json(root / 'status.json', {'state': 'prepared', 'completed': 0, 'total': len(jobs)}); render(root)


def worker(root, index):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    from popgym_jax import PortEnv
    p = guard(root); job = p['jobs'][index]
    snapshot, archive = Path(job['snapshot']), Path(job['episodes_archive'])
    if file_hash(snapshot) != job['snapshot_sha256'] or file_hash(archive) != job['episodes_sha256']:
        raise ValueError('Audit input snapshot/archive changed')
    folder = root / 'cases' / memory.job_name(job); folder.mkdir(parents=True, exist_ok=False)
    saved = cloudpickle.loads(snapshot.read_bytes()); config = saved['config']; state = jax.tree.map(jnp.asarray, saved['state'])
    if saved['steps'] != job['step'] or config['seed'] != job['seed'] or config['critic_normalization_arm'] != 'ofn':
        raise ValueError('Wrong snapshot identity')
    before = memory.tree_digest(state); port = PortEnv(config['env_id'])
    _, fns, debug = memory.normalization.build_agent(config, port, 'ofn')
    with np.load(archive) as loaded:
        data = {k: loaded[k] for k in loaded.files}
    prefixes = recover_prefixes(state, fns, debug, port, data, 17000000 + job['seed'])
    host = jax.device_get(prefixes)
    parity = {}
    for key, expected in (('obs', data['observations']), ('probs', data['probs']), ('reward', data['rewards']),
            ('done', data['done']), ('active', data['active']), ('q', data['q']),
            ('entropy', data['entropy']), ('ld', data['ld'])):
        np.testing.assert_allclose(host[key], expected, atol=1e-5, rtol=1e-4)
        parity[key] = float(np.max(np.abs(np.asarray(host[key], float) - np.asarray(expected, float))))
    np.testing.assert_array_equal(host['predicted_action'], data['actions'])
    parity['original_actions_identical'] = True
    q = host['q_twins'].min(-1)
    taken_q = np.take_along_axis(q, data['actions'][..., None], -1)[..., 0]
    np.testing.assert_allclose(taken_q[data['active']], data['q'][data['active']], atol=1e-5, rtol=1e-4)
    metric = jax.device_get(grounding_metrics(prefixes['probs'], prefixes['q_twins'].min(-1), prefixes['seen'], state.alpha))
    valid = host['active'] & host['seen'].any(-1) & (~host['seen']).any(-1)
    rollout_grounding = summarize_grounding(metric, valid)
    replay = replay_grounding(state, fns, debug, config['memory_hidden_dim'], p['replay_batches'], jax.random.PRNGKey(21000000 + job['seed'] + job['step']))
    rng = np.random.default_rng(23000000 + job['seed'])
    selected = np.sort(rng.choice(data['actions'].shape[0], p['anchor_episodes'], replace=False))
    points = select_points(prefixes, selected, p['anchor_ages']); branch = make_branches(fns, port, config['gamma'])
    results = {}; arrays = {}; seed = 25000000 + job['seed']
    for policy in POLICIES:
        chunks = []
        for start in range(0, len(points['valid']), p['chunk_anchors']):
            part = jax.tree.map(lambda x: x[start:start + p['chunk_anchors']], points)
            values, ended = jax.device_get(branch(state, part, seed, replicas=p['replicas'], policy=policy))
            if not ended[np.asarray(part['valid'])].all():
                raise AssertionError('Counterfactual return truncated before actual terminal')
            chunks.append(values)
        values = np.concatenate(chunks, axis=0); results[policy] = summarize_branches(points, values)
        arrays[policy + '_returns'] = values
    np.savez_compressed(folder / 'counterfactuals.npz', **arrays,
        **{k: np.asarray(points[k]) for k in ('candidate_actions', 'predicted_q', 'episodes', 'ages', 'valid')},
        q_twins=np.asarray(points['q_twins']), probs=np.asarray(points['probs']), seen=np.asarray(points['seen']))
    after = memory.tree_digest(state)
    if before != after or file_hash(snapshot) != job['snapshot_sha256'] or file_hash(archive) != job['episodes_sha256']:
        raise AssertionError('Audit state/input changed')
    guard(root)
    write_json(folder / 'result.json', {'job': job, 'rollout_grounding': rollout_grounding, 'replay_grounding': replay,
        'counterfactuals': results, 'certificates': {'state_digest_before': before, 'state_digest_after': after,
            'prefix_parity': parity, 'input_hashes_unchanged': True}})
    print(memory.job_name(job), 'rollout', rollout_grounding['means'],
        'counterfactuals', {k: {name: value for name, value in r.items() if name not in ('by_age', 'reference_note')}
            for k, r in results.items()}, flush=True)


def render(root):
    web = Path('/tmp/opencode/popgym-report-web/actor-critic-audit'); web.mkdir(parents=True, exist_ok=True)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>Actor / main-critic grounding and counterfactual returns</h1><p>Read-only frozen OFN snapshots. Actor vs main critic and paired repeat/fresh MC continuations. Not an exact representation-conditional Q oracle or a new training intervention.</p><pre>' + html.escape(json.dumps(status, indent=2)) + '</pre><table border="1"><tr><th>Case</th><th>Actor repeat mass</th><th>Critic-optimum repeat mass</th><th>Actor Q regret</th><th>Predicted fresh−repeat Q</th><th>Current-policy MC gap</th><th>Target-policy MC gap</th></tr>'
    for path in sorted(root.glob('cases/*/result.json')):
        r = json.loads(path.read_text()); m = r['rollout_grounding']['means']
        values = (path.parent.name, m['actor_seen_mass'], m['critic_soft_optimum_seen_mass'], m['actor_greedy_q_regret'],
            r['counterfactuals']['target']['mean_predicted_fresh_minus_repeat'],
            r['counterfactuals']['current']['mean_mc_fresh_minus_repeat'], r['counterfactuals']['target']['mean_mc_fresh_minus_repeat'])
        text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    text += '</table>'
    if (root / 'findings.md').exists():
        text += '<pre>' + html.escape((root / 'findings.md').read_text()) + '</pre>'
    tmp = web / 'index.tmp'; tmp.write_text(text); tmp.replace(web / 'index.html')


def supervise(root):
    p = guard(root); proof = json.loads((root / 'validation.json').read_text())
    if not proof.get('passed') or proof['script_sha256'] != p['script_sha256']:
        raise ValueError('No matching actor/critic CPU/CUDA validation')
    for index, job in enumerate(p['jobs']):
        guard(root); write_json(root / 'status.json', {'state': 'running', 'completed': index, 'total': len(p['jobs']), 'job': job}); render(root)
        with (root / (memory.job_name(job) + '.log')).open('x') as file:
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
