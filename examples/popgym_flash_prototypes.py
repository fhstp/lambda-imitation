"""FlashSAC-inspired heads and categorical-mean lambda-head ablation.

Keep the running GPU0/source guards untouched. New GPU1 residual-head arm;
GPU2 matched main-categorical vs all-categorical arms. No distributional LD.
"""

import argparse
import builtins
from datetime import datetime
from functools import partial
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import popgym_full_episode as base
from popgym_gradient_balance import captures
from popgym_critic_aligned import replace_captures
from popgym_suite import write_json


DEFAULT_ROOT = Path('/tmp/opencode/popgym-flash-prototypes')
ARMS = ('residual', 'main-categorical', 'all-categorical')
ATOMS = 51
SUPPORT = (-1.5, 1.5)
RETAIN_STEPS = (0, 250000, 500000, 750000, 1000000)


def support():
    import jax.numpy as jnp
    return jnp.linspace(*SUPPORT, ATOMS)


def project_distribution(probabilities, values):
    """Linear categorical projection,preserving mass and unclipped mean."""
    import jax.numpy as jnp
    z = support(); clipped = jnp.clip(values, z[0], z[-1])
    coordinate = (clipped - z[0]) / (z[-1] - z[0]) * (ATOMS - 1)
    lower = jnp.floor(coordinate).astype(jnp.int32)
    upper = jnp.minimum(lower + 1, ATOMS - 1); fraction = coordinate - lower
    rows = jnp.arange(probabilities.shape[0])[:, None]
    result = jnp.zeros((probabilities.shape[0], ATOMS), probabilities.dtype)
    result = result.at[rows, lower].add(probabilities * (1. - fraction))
    return result.at[rows, upper].add(probabilities * fraction)


def two_hot(values):
    import jax.numpy as jnp
    return project_distribution(jnp.ones((values.size, 1)), values.reshape(-1, 1))


def target_mixture(log_probabilities, policy):
    """Per-action minimum TWIN MEAN,then exact target-policy action mixture."""
    import jax.numpy as jnp
    probabilities = jnp.exp(log_probabilities)
    means = jnp.sum(probabilities * support()[None, None, :, None], -2)
    selected = jnp.where((means[..., 0] <= means[..., 1])[..., None],
        probabilities[..., 0], probabilities[..., 1])
    return jnp.sum(selected * policy[..., None], 1)


def head_classes(config):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from lambda_imitation.iqlearn import Head, _nnx_data
    OFN = base.normalization.normalized_head_class()

    class CategoricalHead(OFN):
        def __init__(self, feature_dim, hidden_dims, output_dim, *, rngs, layer_norm=False):
            self.actions = output_dim
            super().__init__(feature_dim, hidden_dims, output_dim * ATOMS, rngs=rngs, layer_norm=layer_norm)

        def log_probs(self, x):
            features = self.output_features(x)[1]
            return jax.nn.log_softmax(self.layers[-1](features).reshape(-1, self.actions, ATOMS), -1)

        def __call__(self, x):
            return jnp.sum(jnp.exp(self.log_probs(x)) * support(), -1)

    class ResidualHead(nnx.Module):
        def __init__(self, feature_dim, hidden_dims, output_dim, *, rngs, layer_norm=None):
            self.is_critic = layer_norm is not None
            width = config.get('residual_critic_width', 256) if self.is_critic else config.get('residual_actor_width', 128)
            self.embed = nnx.Linear(feature_dim, width, rngs=rngs)
            self.blocks = _nnx_data([(nnx.Linear(width, 4 * width, rngs=rngs),
                nnx.Linear(4 * width, width, rngs=rngs)) for _ in range(2)])
            self.layers = _nnx_data([nnx.Linear(width, output_dim, rngs=rngs)])

        def output_features(self, x):
            def rms(y):
                return y * jax.lax.rsqrt(jnp.mean(y * y, -1, keepdims=True) + 1e-6)
            x = self.embed(x)
            for expand, contract in self.blocks:
                residual = x
                x = nnx.relu(rms(expand(rms(x))))
                x = residual + contract(x)
            x = rms(x)
            # Retain the baseline unit feature norm in critic outputs; actor
            # uses post-RMS features. Fixed gains,no BN,running stats or GRU norm.
            return x, base.normalization.normalized_features(x) if self.is_critic else x

        def __call__(self, x):
            return self.layers[-1](self.output_features(x)[1])

    return Head, OFN, CategoricalHead, ResidualHead


def networks(config, port, arm):
    """Private factory import hook;never replace global Head or sys.modules."""
    import lambda_imitation.iqlearn as core
    import lambda_imitation.utils as utils
    import popgym_critic_settling as builder_module
    Head, OFN, Categorical, Residual = head_classes(config); roles = []
    def head(*args, **kwargs):
        role = len(roles); roles.append('critic' if 'layer_norm' in kwargs else 'actor')
        if arm == 'residual': cls = Residual
        elif role == 0: cls = Head
        elif arm == 'all-categorical' or role <= 2: cls = Categorical
        else: cls = OFN
        return cls(*args, **kwargs)
    factory = base.normalization.clone_globals(core.create_iqlearn, {'Head': head})
    wrapper = base.normalization.clone_globals(utils.create_iqlearn_from_env, {'create_iqlearn': factory})
    original_import = builtins.__import__
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == 'lambda_imitation.utils' and 'create_iqlearn_from_env' in fromlist:
            return types.SimpleNamespace(EnvSpec=utils.EnvSpec, create_iqlearn_from_env=wrapper)
        return module
    namespace = dict(vars(builtins)); namespace['__import__'] = local_import
    builder = base.normalization.clone_globals(builder_module.build_agent, {'__builtins__': namespace})
    result = builder(config, port)
    if roles != ['actor'] + ['critic'] * 6: raise ValueError('Unexpected head roles')
    return result


def categorical_objective(fns, debug, sampler, arm):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from lambda_imitation.iqlearn import retrace_targets
    h = captures(captures(fns.update)['loss_combined']); hp = h['params']
    main = captures(h['loss_critic']); low = captures(h['loss_vtrace_lambda_sequence'])
    actor = captures(debug.get_entropy)['run_actor']
    graphs = (main['critic_graph'], h['lambda1_critic_graph'], h['lambda2_critic_graph'])
    flat = lambda x: x.reshape(-1, *x.shape[2:])
    def distributions(state, graph, x):
        return jnp.stack([nnx.merge(getattr(graph, branch), getattr(state, branch)).log_probs(x)
            for branch in ('q1', 'q2')], -1)
    def taken_log_probs(state, graph, x, actions):
        logp = distributions(state, graph, x)
        indices = jnp.round(actions.reshape(-1)).astype(jnp.int32)
        return logp[jnp.arange(len(indices)), indices]

    def components(fe, actor_state, critic, l1, l2, fe_target, actor_target,
            critic_target, l1_target, l2_target, alpha, raw, keys):
        valid = raw['episode_valid'].T[:-1].astype(jnp.float32)
        denominator = jnp.maximum(valid.sum(), 1.)
        mean = lambda v: (v.reshape(valid.shape) * valid).sum() / denominator
        x, y = debug.calculate_latent(fe, fe_target, raw['observations'], raw['actions'],
            raw['terminated'], jnp.zeros((hp.online_batch_size, h['feature_extractor'].initialize_carry(1).shape[-1])))
        act, reward, done, behaviour = [raw[k].swapaxes(0, 1) for k in ('actions', 'rewards', 'terminated', 'behaviour_weight')]
        actor_x = flat(jax.lax.stop_gradient(x[:-1]))
        probs = jax.nn.softmax(actor(actor_state, actor_x)); logp = jax.nn.log_softmax(actor(actor_state, actor_x))
        entropy = -jnp.sum(probs * logp, -1)
        q = captures(main['get_q_both'])['run_critic'](jax.lax.stop_gradient(critic), graphs[0], actor_x).min(-1)
        v = jnp.sum(probs * q, -1) + alpha * entropy
        actor_loss = -mean(v)
        next_logs = distributions(critic_target, graphs[0], flat(y[1:]))
        policy = jax.nn.softmax(actor(actor_target, flat(y[1:])))
        mixture = target_mixture(next_logs, policy)
        shifted = reward[:-1].reshape(-1, 1) + hp.gamma * (1. - done[:-1].reshape(-1, 1)) * support()[None]
        label = jax.lax.stop_gradient(project_distribution(mixture, shifted))
        predicted_logs = taken_log_probs(critic, graphs[0], flat(x[:-1]), flat(act[:-1]))
        critic_loss = mean(-jnp.sum(predicted_logs * label[..., None], -2).mean(-1))
        metrics = {'q': mean(jnp.sum(probs * q, -1)), 'entropy': mean(entropy), 'v': mean(v),
            'critic_loss': critic_loss, 'target_q': mean(jnp.sum(label * support(), -1)), 'alpha': alpha,
            'episode_mean_length': raw['episode_length'].mean(), 'valid_fraction': valid.mean(),
            'main_target_clipped_mass': mean(jnp.sum(mixture * ((shifted < SUPPORT[0]) | (shifted > SUPPORT[1])), -1)),
            'main_target_edge_mass': mean(label[:, 0] + label[:, -1])}
        losses = [actor_loss, critic_loss]
        for name, lam, online, target, graph, key in zip(('lambda1', 'lambda2'), (hp.lambda1, hp.lambda2),
                (l1, l2), (l1_target, l2_target), graphs[1:], keys[3:]):
            Tp, B = x.shape[:2]
            value = low['get_v'](actor_target, target, graph, jnp.float32(0), flat(y), key, False, False).reshape(Tp, B)
            ratios = low['get_importance_ratios'](actor_target, flat(y), flat(act), flat(behaviour)).reshape(Tp, B)
            taken = low['get_q'](target, graph, flat(y), flat(act)).reshape(Tp, B)
            returns = jax.lax.stop_gradient(retrace_targets(taken, value, reward, done, ratios, hp.gamma, lam, hp.c_bar))[:-1]
            if arm == 'all-categorical':
                label = jax.lax.stop_gradient(two_hot(returns.reshape(-1)))
                predicted = taken_log_probs(online, graph, flat(x[:-1]), flat(act[:-1]))
                loss = mean(-jnp.sum(predicted * label[..., None], -2).mean(-1))
                metrics[name + '_target_clipped_fraction'] = mean((returns < SUPPORT[0]) | (returns > SUPPORT[1]))
                metrics[name + '_target_edge_mass'] = mean(label[:, 0] + label[:, -1])
            else:
                a, b = low['get_q_both'](online, graph, flat(x), flat(act))
                a, b = a.reshape(Tp, B)[:-1], b.reshape(Tp, B)[:-1]
                loss = .5 * (mean((a - returns) ** 2) + mean((b - returns) ** 2))
            losses.append(loss); metrics[name + '_loss'] = loss; metrics[name + '_target'] = mean(returns)
        # Exactly the original scalar LD:minimum of EACH TWIN MEAN first,
        # then squared difference. Head parameters frozen;FE receives gradient.
        ld1 = low['get_q'](jax.lax.stop_gradient(l1), graphs[1], flat(x[:-1]), flat(act[:-1]))
        ld2 = low['get_q'](jax.lax.stop_gradient(l2), graphs[2], flat(x[:-1]), flat(act[:-1]))
        discrepancy = mean((ld1 - ld2) ** 2)
        losses.append(hp.lambda_coef * discrepancy); metrics['ld_loss'] = discrepancy
        return sum(losses), metrics, jnp.stack(losses)

    def combined(fe, actor_state, critic, l1, l2, sf1, sf2, fe_target, actor_target,
            critic_target, l1_target, l2_target, sf1_target, sf2_target, alpha, buffer, key):
        keys = jax.random.split(key, 5); sample, _ = sampler(buffer, keys[0])
        loss, metrics, _ = components(fe, actor_state, critic, l1, l2, fe_target, actor_target,
            critic_target, l1_target, l2_target, alpha, sample.this_info, keys)
        return loss, metrics
    return combined, components, h


def build(config, port, arm):
    import jax
    import jax.numpy as jnp
    state, fns, debug = networks(config, port, arm)
    sampler = base.make_sampler(config['batch_size'])
    combined, components, context = (base.masked_objective(fns, debug, sampler) if arm == 'residual'
        else categorical_objective(fns, debug, sampler, arm))
    original_update = replace_captures(fns.update, {'loss_combined': combined})
    @jax.jit
    def update(state, key):
        alpha = base.alpha_at(state.update_step)
        state, metrics = original_update(state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), key)
        alpha = base.alpha_at(state.update_step)
        return state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), metrics
    return state, fns._replace(update=update), debug, sampler, components, context


def config_for(seed, arm, small=False):
    args, config = base.config_for(seed, small)
    config.update(flash_arm=arm, categorical_atoms=ATOMS, categorical_support=list(SUPPORT),
        residual_actor_width=8 if small else 128, residual_critic_width=8 if small else 256,
        residual_blocks=2, residual_expansion=4, target_steps=1000000,
        snapshot_retention=list(RETAIN_STEPS), architecture='RMS-residual-critic-OFN' if arm == 'residual' else 'original-OFN')
    return args, config


def extra_probe(state, debug, sampler, components, context, arm, seed, step):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    sample, _ = sampler(state.online_buffer, jax.random.PRNGKey(27000000 + seed + step)); raw = sample.this_info
    values = (state.feature_extractor, state.actor, state.critic, state.lambda1_critic, state.lambda2_critic,
        state.feature_extractor_target, state.actor_target, state.critic_target, state.lambda1_critic_target,
        state.lambda2_critic_target, state.alpha, raw, jax.random.split(jax.random.PRNGKey(29000000 + seed + step), 5))
    with jax.default_matmul_precision('float32'):
        _, metrics, _ = jax.jit(components)(*values)
        result = {'objective/' + k: float(v) for k, v in jax.device_get(metrics).items()}
        # Same replay/state/RNG inputs for all contributions;no optimizer change.
        gradient_fn = jax.jit(jax.jacrev(lambda fe: components(fe, *values[1:])[2], argnums=0))
        grads = jax.device_get(gradient_fn(state.feature_extractor))
        vectors = [np.asarray(v, np.float64).reshape(5, -1) for v in jax.tree.leaves(grads)]
        squared = sum(np.sum(v ** 2, axis=1) for v in vectors)
        norms = np.sqrt(squared)
        result.update({f'fe_gradient/{tag}': float(n) for tag, n in zip(('actor', 'main', 'lambda1', 'lambda2', 'weighted_discrepancy'), norms)})
        regression_squared = sum(np.sum(v[1:4].sum(0) ** 2) for v in vectors)
        regression_norm = np.sqrt(regression_squared)
        alignment = sum(np.dot(v[4], v[1:4].sum(0)) for v in vectors)
        result['fe_gradient/combined_regression'] = float(regression_norm)
        result['fe_gradient/weighted_discrepancy_to_combined_regression'] = float(norms[4] / max(regression_norm, 1e-12))
        result['fe_gradient/discrepancy_regression_cosine'] = float(alignment / max(norms[4] * regression_norm, 1e-30))
        if arm != 'residual':
            x, _ = debug.calculate_latent(state.feature_extractor, state.feature_extractor_target,
                raw['observations'], raw['actions'], raw['terminated'], jnp.zeros((raw['actions'].shape[0], context['feature_extractor'].initialize_carry(1).shape[-1])))
            valid = np.asarray(raw['episode_valid'].T[:-1]); flat = x[:-1].reshape(-1, x.shape[-1])
            main = captures(context['loss_critic'])
            for name, attr, graph in zip(('main', 'lambda1', 'lambda2'), ('critic', 'lambda1_critic', 'lambda2_critic'),
                    (main['critic_graph'], context['lambda1_critic_graph'], context['lambda2_critic_graph'])):
                if arm == 'main-categorical' and name != 'main': continue
                for branch in ('q1', 'q2'):
                    head = nnx.merge(getattr(graph, branch), getattr(getattr(state, attr), branch))
                    p = np.exp(np.asarray(head.log_probs(flat)).reshape((*valid.shape, 64, ATOMS)))[valid]
                    means = (p * np.asarray(support())).sum(-1)
                    result[f'{name}_{branch}/edge_mass'] = float((p[..., 0] + p[..., -1]).mean())
                    result[f'{name}_{branch}/return_distribution_std'] = float(np.sqrt(np.maximum((p * np.asarray(support()) ** 2).sum(-1) - means ** 2, 0)).mean())
    if not all(np.isfinite(v) for v in result.values()): raise FloatingPointError('Nonfinite prototype probe')
    return result


def measurements(state, fns, debug, port, sampler, components, context, arm, seed, step, folder, mc):
    result = base.measurements(state, fns, debug, port, sampler, context, seed, step, folder, mc)
    result['prototype'] = extra_probe(state, debug, sampler, components, context, arm, seed, step)
    return result


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if base.file_hash(__file__) != p['script_sha256'] or base.file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Prototype source/plan changed')
    for name, digest in p['dependencies'].items():
        if base.file_hash(Path(__file__).with_name(name)) != digest: raise ValueError('Dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()): raise ValueError('Packages changed')
    base.guard(base.DEFAULT_ROOT)
    if base.file_hash(base.DEFAULT_ROOT / 'plan.json') != p['gpu0_plan_sha256']: raise ValueError('GPU0 plan changed')
    return p


def prepare(root):
    base.guard(base.DEFAULT_ROOT); root.mkdir(parents=True, exist_ok=True)
    if (root / 'plan.json').exists(): raise FileExistsError('Fresh prototype root required')
    jobs = [{'arm': 'residual', 'seed': seed, 'queue': 'architecture', 'gpu': 1} for seed in range(42, 47)]
    jobs += [{'arm': arm, 'seed': seed, 'queue': 'categorical', 'gpu': 2}
        for seed in range(42, 47) for arm in ARMS[1:]]
    p = {'script_sha256': base.file_hash(__file__), 'created_at': datetime.now().astimezone().isoformat(),
        'gpu0_plan_sha256': base.file_hash(base.DEFAULT_ROOT / 'plan.json'),
        'dependencies': {name: base.file_hash(Path(__file__).with_name(name)) for name in ('popgym_full_episode.py',
            'popgym_critic_normalization.py', 'popgym_critic_settling.py', 'popgym_critic_aligned.py')},
        'versions': {name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'numpy', 'cloudpickle', 'popjym', 'wandb')},
        'jobs': jobs, 'steps': 1000000, 'eval_every': 25000, 'mc_steps': list(base.MC_STEPS),
        'retain_steps': list(RETAIN_STEPS), 'atoms': ATOMS, 'support': list(SUPPORT),
        'wandb_entity': 'fhstp-data-intelligence-research-group', 'wandb_project': 'lambda-popgym-flash-prototypes',
        'protocol': 'GPU0 original full-episode OFN queue UNCHANGED. GPU1 five FRESH scalar residual actor/all-six-critic head runs;two expansion4 residual blocks,fixed pre/post RMS,actor width128/critic256,critic final OFN retained. No BatchNorm,learned norm gains,weight projection,GRU normalization or paper reproduction claim. GPU2 ten FRESH matched main-categorical vs all-categorical runs,seeds42-46 interleaved,original OFN hidden heads;51 atoms[-1.5,1.5]. Main target is full projected Bellman distribution:target-policy action mixture of per-action target-twin distribution selected by minimum MEAN,original target FE,NO entropy reward. All-categorical lambda heads fit two-hot EXISTING SCALAR Retrace targets;scalar LD compares min twin means,head parameters detached,not a distributional discrepancy or full-return-distribution Retrace claim. Same gamma.99/lambda.05/.85/beta.1/LRs/EMA/optimizers,1M ONLINE steps+12800 prefill,complete episodes/batch128/no burn/full BPTT/masked padding,scheduledalpha1->.01first500k thenfixed. All25k diagnostics and original holdouts,paired current/EMA MC milestones as GPU0,separate diagnostic RNG+before/after state hashes. Additional target clipping/edge mass,distribution spread(NOT epistemic uncertainty),same-minibatch FE contribution gradient norms at every checkpoint;beta NOT automatically calibrated. User-approved immutable full-state retention0/250k/500k/750k/1M plus rolling latest per run;all intermediate metrics/episodes/MC persist,but intermediate states not recoverable. Same initial FE/actor/main across categorical pairs verified;auxiliary heads differ by intervention. Architecture/capacity/normalization bundle not one-factor control;categorical pair isolates lambda representation/loss change. Native training,float32 diagnostic gradient validation,source/package/plan guards. No production/old-worker changes,extra arms,budget/LR/schedule escalation,smoke reuse or baseline splicing.'}
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': base.file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    for queue in ('architecture', 'categorical'): write_json(root / f'status-{queue}.json', {'state': 'prepared'})
    render(root)


def worker(root, index):
    import cloudpickle
    import jax
    import numpy as np
    import wandb
    from lambda_imitation.buffer import create_buffer
    from popgym_jax import PortEnv, JaxRunner, evaluation_result
    p = guard(root); job = p['jobs'][index]; arm, seed = job['arm'], job['seed']
    args, config = config_for(seed, arm); folder = root / arm / f'seed{seed}'; folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / 'config.json', config)
    port = PortEnv(config['env_id']); state, fns, debug, sampler, components, context = build(config, port, arm)
    _, bfns = create_buffer({k: v.shape[1:] for k, v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
    runner = JaxRunner(args, port, fns, debug, bfns, args.memory_hidden_dim, True); rollout = runner.initial_rollout(seed)
    run = wandb.init(project=p['wandb_project'], entity=p['wandb_entity'], id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16],
        resume='never', config=config, name=f'Battleship-{arm}-seed{seed}', dir=str(folder))
    run.define_metric('steps'); run.define_metric('*', step_metric='steps'); history = []; step = 0
    try:
        state, rollout, count = jax.block_until_ready(runner.prefill(state, rollout, 12800)); interactions = int(count)
        while True:
            guard(root)
            if shutil.disk_usage(root).free < 2 * 1024 ** 3: raise OSError('Less than2GB free;stop for inspection,never delete archives')
            payload = {'state': jax.device_get(state), 'rollout': jax.device_get(rollout), 'config': config,
                'steps': step, 'interactions': interactions, 'history': history, 'plan_sha256': base.file_hash(root / 'plan.json')}
            checkpoint = folder / 'checkpoint.pkl'; temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as file: cloudpickle.dump(payload, file)
            temporary.replace(checkpoint)
            diagnostic = root / 'diagnostics' / arm / f'seed{seed}' / f'step{step}'; diagnostic.mkdir(parents=True, exist_ok=False)
            retained = step in p['retain_steps']; snapshot = diagnostic / 'state.pkl'
            if retained: snapshot.write_bytes(checkpoint.read_bytes())
            before = base.audit.memory.tree_digest(state); checkpoint_hash = base.file_hash(checkpoint)
            result = measurements(state, fns, debug, port, sampler, components, context, arm, seed, step, diagnostic, step in p['mc_steps'])
            after = base.audit.memory.tree_digest(state)
            if before != after or base.file_hash(checkpoint) != checkpoint_hash: raise AssertionError('Diagnostics changed state/checkpoint')
            result['certificates'] = {'state_digest_before': before, 'state_digest_after': after,
                'checkpoint_hash_at_measurement': checkpoint_hash, 'immutable_snapshot_retained': retained,
                'snapshot_sha256': base.file_hash(snapshot) if retained else None}
            write_json(diagnostic / 'result.json', result)
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
                **{'grounding/' + k: v for k, v in result['rollout_grounding']['means'].items()}, **result['replay'], **result['prototype']}
            if 'counterfactuals' in result:
                for policy, mc in result['counterfactuals'].items():
                    log.update({f'mc/{policy}/{k}': v for k, v in mc.items() if isinstance(v, (int, float))})
                    log.update({f'mc/{policy}/{name}/{k}': v for name, values in mc['action_classes'].items() for k, v in values.items()})
            run.log(log); write_json(folder / 'status.json', {'state': 'complete' if final else 'running', **entry})
            with temporary.open('wb') as file: cloudpickle.dump({**payload, 'history': history}, file)
            temporary.replace(checkpoint); render(root); print('EVAL', arm, seed, entry, flush=True)
            if final: break
            boundary = min(p['steps'], step + p['eval_every'])
            while step < boundary:
                count = min(args.jax_train_steps, boundary - step)
                state, rollout, metrics = jax.block_until_ready(runner.train(state, rollout, count))
                if not all(np.isfinite(np.asarray(v)).all() for v in metrics.values()): raise FloatingPointError('Nonfinite training metrics')
                if np.asarray(rollout[-2]): raise RuntimeError('Horizon exceeded')
                step += count; interactions += count
            run.log({'steps': step, **{'train/' + k: float(v) for k, v in metrics.items()}})
        guard(root)
    except BaseException:
        import traceback
        write_json(folder / 'status.json', {'state': 'failed', 'steps': step, 'error': traceback.format_exc()}); raise
    finally: run.finish()


def render(root):
    web = Path('/tmp/opencode/popgym-report-web/flash-prototypes'); web.mkdir(parents=True, exist_ok=True)
    status = {queue: json.loads((root / f'status-{queue}.json').read_text()) for queue in ('architecture', 'categorical')}
    text = '<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>FlashSAC-inspired Battleship prototypes</h1><p>GPU1 normalized residual scalar heads. GPU2 categorical main-only vs categorical main+lambda,scalar mean discrepancy. Five fresh seeds per arm,1M/full episodes/matched alpha schedule.51 atoms[-1.5,1.5],no BatchNorm/reward scaling.</p><pre>' + html.escape(json.dumps(status, indent=2)) + '</pre><table border="1"><tr><th>Arm / seed</th><th>Steps</th><th>Alpha</th><th>Return</th><th>Repeats</th><th>Unique cells</th></tr>'
    for path in sorted(root.glob('*/seed*/history.json')):
        for e in json.loads(path.read_text()):
            values = (str(path.parent.relative_to(root)), e['steps'], e['alpha'], e['diagnostic_return'], e['repeat_fraction'], e['unique_cells'])
            text += '<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in values) + '</tr>'
    temporary = web / f'index-{os.getpid()}.tmp'; temporary.write_text(text + '</table>'); temporary.replace(web / 'index.html')


def supervise(root, queue):
    p = guard(root); proof = json.loads((root / 'validation.json').read_text())
    if not proof.get('passed') or proof['script_sha256'] != p['script_sha256']: raise ValueError('No matching validation')
    jobs = [(i, j) for i, j in enumerate(p['jobs']) if j['queue'] == queue]
    for done, (index, job) in enumerate(jobs):
        guard(root); write_json(root / f'status-{queue}.json', {'state': 'running', 'completed': done, 'total': len(jobs), 'job': job}); render(root)
        with (root / f"{job['arm']}-seed{job['seed']}.log").open('x') as file:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', str(index)],
                stdout=file, stderr=subprocess.STDOUT, check=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
    guard(root); write_json(root / f'status-{queue}.json', {'state': 'complete', 'completed': len(jobs), 'total': len(jobs),
        'finished_at': datetime.now().astimezone().isoformat()}); render(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', type=int)
    parser.add_argument('--queue', choices=('architecture', 'categorical')); args = parser.parse_args(); root = args.root.resolve()
    if args.prepare: prepare(root)
    else:
        try:
            if args.worker is not None: worker(root, args.worker)
            elif args.queue: supervise(root, args.queue)
            else: raise ValueError('Specify queue or worker')
        except BaseException:
            import traceback
            if args.queue:
                write_json(root / f'status-{args.queue}.json', {'state': 'failed', 'error': traceback.format_exc()}); render(root)
            raise


if __name__ == '__main__': main()
