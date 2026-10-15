"""Fresh residual+categorical beta0/1/10 pilot;10M ceiling,explicit review gates.

Original training/objective primitives reused privately. No edits to production
or predecessor sources, no old state reuse, no automatic coefficient/budget
changes. Workers exit at100k/1M/5M;CPU supervisors wait for explicit decisions.
"""

import argparse
from datetime import datetime
from functools import partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

import popgym_flash_prototypes as flash
from popgym_critic_aligned import replace_captures
from popgym_suite import write_json

DEFAULT_ROOT = Path('/tmp/opencode/popgym-flash-combined')
BETAS = (0, 1, 10)
REVIEW_STEPS = (100000, 1000000, 5000000)
RETAIN_STEPS = (0, 100000, 250000, 500000, 750000, 1000000, 5000000, 10000000)


def config_for(beta, small=False):
    if beta not in BETAS:
        raise ValueError('Only user-authorized beta0/1/10')
    args, config = flash.config_for(42, 'all-categorical', small)
    args.fe_lr = 1e-4; args.lambda_coef = float(beta); args.steps = 10000000
    config.update(fe_lr=1e-4, lambda_coef=float(beta), target_steps=10000000,
        architecture='RMS-residual-all-categorical-critic-OFN',
        experiment='flash-combined-beta', snapshot_retention=list(RETAIN_STEPS),
        manual_review_steps=list(REVIEW_STEPS), lambda_td_loss='two_hot_cross_entropy')
    return args, config


def residual_categorical_class(config):
    import jax
    import jax.numpy as jnp
    Residual = flash.head_classes(config)[3]
    class ResidualCategorical(Residual):
        def __init__(self, feature_dim, hidden_dims, output_dim, *, rngs, layer_norm=False):
            self.actions = output_dim
            super().__init__(feature_dim, hidden_dims, output_dim * flash.ATOMS, rngs=rngs, layer_norm=layer_norm)
        def log_probs(self, x):
            features = self.output_features(x)[1]
            return jax.nn.log_softmax(self.layers[-1](features).reshape(-1, self.actions, flash.ATOMS), -1)
        def __call__(self, x):
            return jnp.sum(jnp.exp(self.log_probs(x)) * flash.support(), -1)
    return Residual, ResidualCategorical


def networks(config, port):
    import builtins
    import types
    import lambda_imitation.iqlearn as core
    import lambda_imitation.utils as utils
    import popgym_critic_settling as builder_module
    Actor, Critic = residual_categorical_class(config); roles = []
    def head(*args, **kwargs):
        critic = 'layer_norm' in kwargs; roles.append(critic)
        return (Critic if critic else Actor)(*args, **kwargs)
    clone = flash.base.normalization.clone_globals
    factory = clone(core.create_iqlearn, {'Head': head})
    wrapper = clone(utils.create_iqlearn_from_env, {'create_iqlearn': factory})
    original_import = builtins.__import__
    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = original_import(name, globals, locals, fromlist, level)
        if name == 'lambda_imitation.utils' and 'create_iqlearn_from_env' in fromlist:
            return types.SimpleNamespace(EnvSpec=utils.EnvSpec, create_iqlearn_from_env=wrapper)
        return module
    namespace = dict(vars(builtins)); namespace['__import__'] = local_import
    result = clone(builder_module.build_agent, {'__builtins__': namespace})(config, port)
    if roles != [False] + [True] * 6:
        raise ValueError('Unexpected actor/all-six-critic construction')
    return result


def build(config, port):
    import jax
    import jax.numpy as jnp
    state, fns, debug = networks(config, port)
    sampler = flash.base.make_sampler(config['batch_size'])
    combined, components, context = flash.categorical_objective(fns, debug, sampler, 'all-categorical')
    update_original = replace_captures(fns.update, {'loss_combined': combined})
    @jax.jit
    def update(state, key):
        alpha = flash.base.alpha_at(state.update_step)
        state, metrics = update_original(state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), key)
        alpha = flash.base.alpha_at(state.update_step)
        return state._replace(alpha=alpha, log_alpha=jnp.log(alpha)), metrics
    return state, fns._replace(update=update), debug, sampler, components, context


def guard(root):
    p = json.loads((root / 'plan.json').read_text())
    if flash.base.file_hash(__file__) != p['script_sha256'] or flash.base.file_hash(root / 'plan.json') != json.loads((root / 'plan-integrity.json').read_text())['sha256']:
        raise ValueError('Combined source/plan changed')
    for name, digest in p['dependencies'].items():
        if flash.base.file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Combined dependency changed')
    if any(importlib.metadata.version(k) != v for k, v in p['versions'].items()):
        raise ValueError('Combined packages changed')
    flash.guard(flash.DEFAULT_ROOT)
    for path, digest in p['cancelled_completed_files'].items():
        if flash.base.file_hash(Path(path)) != digest:
            raise ValueError('Archived completed predecessor file changed')
    if flash.base.file_hash(Path(p['cancellation_record'])) != p['cancellation_sha256']:
        raise ValueError('Old queue cancellation certificate changed')
    return p


def prepare(root):
    flash.guard(flash.DEFAULT_ROOT)
    cancellation = Path('/tmp/opencode/popgym-flash-combined-cancel/cancellation.json')
    old = json.loads(cancellation.read_text())
    if not old['user_authorized'] or not old['all_old_units_have_no_process']:
        raise ValueError('Need explicit old-queue cancellation certificate')
    root.mkdir(parents=True, exist_ok=False)
    p = dict(script_sha256=flash.base.file_hash(__file__), created_at=datetime.now().astimezone().isoformat(),
        jobs=[dict(beta=beta, seed=42, gpu=i) for i, beta in enumerate(BETAS)], steps=10000000,
        review_steps=list(REVIEW_STEPS), retain_steps=list(RETAIN_STEPS), eval_every=25000,
        mc_steps=sorted(set(range(100000, 10000001, 100000)) | {200000, 250000}),
        wandb_entity='fhstp-data-intelligence-research-group', wandb_project='lambda-popgym-flash-combined',
        dependencies={name: flash.base.file_hash(Path(__file__).with_name(name)) for name in
            ('popgym_flash_prototypes.py','popgym_full_episode.py','popgym_critic_normalization.py','popgym_critic_settling.py','popgym_critic_aligned.py','popgym_jax.py')},
        versions={name: importlib.metadata.version(name) for name in ('jax','flax','optax','numpy','cloudpickle','popjym','wandb')},
        cancellation_record=str(cancellation), cancellation_sha256=flash.base.file_hash(cancellation),
        cancelled_completed_files={path: digest for files in old['completed_files'].values() for path, digest in files.items()},
        protocol='Exactly3 FRESH matched seed42 fits,beta0/1/10 onGPU0/1/2. Residual actor128/all6 critic256 heads,two expansion4 blocks,fixed pre/post RMS,critic final OFN;all main+lambda categorical51 atoms[-1.5,1.5]. GRU512/projection128 FE LR1e-4,all head LRs1e-4. Original reward-only projected main Bellman law,twin minimum-MEAN law selection,original target FE/policy;two-hot SCALAR Retrace auxiliary targets,scalar mean discrepancy with detached heads. Original actor-to-FE stopped,sampler/EMA/RNG/update cadence unchanged. Full64-step episodes,batch128,replay100k,prefill12800,native precision,gamma.99,tau.005,lambda.05/.85. Alpha1->.01 over500k thenfixed,not automatically tuned.10M ceiling NOT automatic approval past review gates100k/1M/5M. Workers save complete state+rollout+history then exit,releasingGPU;CPU supervisors wait for explicit hash-bound continue/stop decisions. Resume restores exactly saved training state/RNG with SAME independent W&B ID and no repeated evaluation/prefill. Nonfinite state/optimizer/rollout/loss or horizon/integrity/OFN violations fail safely;no arbitrary result-based threshold or coefficient/LR/schedule escalation. All25k native100-episode stochastic/greedy/uniform diagnostics,gradient balance/support pressure,original20 validation/100 final holdouts;paired current/EMA MC100k multiples+250k. Immutable fullstates0/100k/250k/500k/750k/1M/5M/10M pluslatest;ALL measurement/episode/MC artifacts retained. Stop atdisk<2GiB,never delete archives. Same-seed exploratory screen,NOT seed-general/compute-matched/paper reproduction claim;new matched beta0 is separate from user5x5 study. BN,reward scaling,weight projection,noise/action repeat deferred.')
    write_json(root / 'plan.json', p); write_json(root / 'plan-integrity.json', {'sha256': flash.base.file_hash(root / 'plan.json')})
    (root / 'source.py').write_bytes(Path(__file__).read_bytes())
    for beta in BETAS:
        write_json(root / f'status-beta{beta}.json', dict(state='prepared', beta=beta, steps=0))
    return p


def finite_checker():
    import jax
    import jax.numpy as jnp
    @jax.jit
    def check(state, rollout):
        values = [jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((state, rollout))
            if hasattr(x, 'dtype') and getattr(x.dtype, 'kind', '') in ('f','c')]
        return jnp.all(jnp.stack(values))
    return check


def decision_path(root, beta, step):
    return root / 'decisions' / f'beta{beta}-step{step}.json'


def decide(root, beta, step, action, note):
    p = guard(root)
    if step not in p['review_steps'] or beta not in BETAS or action not in ('continue','stop'):
        raise ValueError('Invalid review decision')
    status = json.loads((root / f'status-beta{beta}.json').read_text())
    if status.get('state') != 'awaiting-review' or status['steps'] != step:
        raise ValueError('Run is not waiting at that review gate')
    path = decision_path(root, beta, step)
    if path.exists():
        raise FileExistsError('Never overwrite a recorded review decision')
    checkpoint = root / f'beta{beta}' / 'checkpoint.pkl'
    digest = flash.base.file_hash(checkpoint)
    if digest != status['checkpoint_sha256']:
        raise ValueError('Review checkpoint changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(beta=beta, steps=step, action=action, note=note,
        checkpoint_sha256=digest, plan_sha256=flash.base.file_hash(root / 'plan.json'),
        at=datetime.now().astimezone().isoformat()))


def read_decision(root, beta, status):
    path = decision_path(root, beta, status['steps'])
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d['beta'] != beta or d['steps'] != status['steps'] or d['checkpoint_sha256'] != status['checkpoint_sha256'] or d['plan_sha256'] != flash.base.file_hash(root / 'plan.json') or d['action'] not in ('continue','stop'):
        raise ValueError('Invalid/stale review authorization')
    return d


def save_checkpoint(path, payload):
    import cloudpickle
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        cloudpickle.dump(payload, stream)
    temporary.replace(path)


def publish(root, beta, state, rollout, step, interactions, history, built, run=None, mc=None):
    import jax
    from popgym_jax import evaluation_result
    args, config, port, runner, fns, debug, sampler, components, context = built
    p = guard(root); folder = root / f'beta{beta}'; checkpoint = folder / 'checkpoint.pkl'
    payload = dict(state=jax.device_get(state), rollout=jax.device_get(rollout), config=config,
        steps=step, interactions=interactions, history=history, plan_sha256=flash.base.file_hash(root / 'plan.json'))
    save_checkpoint(checkpoint, payload)
    diagnostic = root / 'diagnostics' / f'beta{beta}' / f'step{step}'; diagnostic.mkdir(parents=True, exist_ok=False)
    retained = step in p['retain_steps']; snapshot = diagnostic / 'state.pkl'
    if retained:
        shutil.copyfile(checkpoint, snapshot)
    before = flash.base.audit.memory.tree_digest(state); checksum = flash.base.file_hash(checkpoint)
    result = flash.measurements(state, fns, debug, port, sampler, components, context, 'all-categorical', 42, step,
        diagnostic, step in p['mc_steps'] if mc is None else mc)
    after = flash.base.audit.memory.tree_digest(state)
    if before != after or flash.base.file_hash(checkpoint) != checksum:
        raise AssertionError('Diagnostics changed state/checkpoint')
    result['certificates'] = dict(state_digest_before=before, state_digest_after=after,
        checkpoint_hash_at_measurement=checksum, immutable_snapshot_retained=retained,
        snapshot_sha256=flash.base.file_hash(snapshot) if retained else None)
    write_json(diagnostic / 'result.json', result)
    final = step == p['steps']; eval_seed = 42 + (2000000 if final else 1000000)
    modes = ('greedy','stochastic','critic','random') if final else ('greedy','stochastic','critic')
    evaluations = {mode: evaluation_result(jax.device_get(runner.evaluate(state, eval_seed, mode,
        args.final_episodes if final else args.eval_episodes)), mode, args, True) for mode in modes}
    write_json(folder / f'episodes_{step}{"_final" if final else ""}.json', evaluations)
    entry = dict(steps=step, beta=beta, interactions_including_prefill=interactions, alpha=float(state.alpha), final=final,
        eval={mode:e['metrics'] for mode,e in evaluations.items()}, diagnostic_return=result['shots']['stochastic']['return_mean'],
        repeat_fraction=result['shots']['stochastic']['repeat_fraction'], unique_cells=result['shots']['stochastic']['unique_shots_mean'])
    history = [*history, entry]; write_json(folder / 'history.json', history)
    save_checkpoint(checkpoint, {**payload, 'history':history})
    phase = 'complete' if final else 'awaiting-review' if step in p['review_steps'] else 'running'
    status = dict(state=phase, **entry, checkpoint_sha256=flash.base.file_hash(checkpoint),
        diagnostic_result_sha256=flash.base.file_hash(diagnostic / 'result.json'))
    write_json(folder / 'status.json', status); write_json(root / f'status-beta{beta}.json', status)
    if run:
        log = dict(steps=step, alpha=float(state.alpha), beta=beta,
            **{f'eval/{m}/{k}':v for m,e in evaluations.items() for k,v in e['metrics'].items()},
            **{f'shots/{m}/{k}':v for m,values in result['shots'].items() for k,v in values.items() if isinstance(v,(int,float))},
            **{'grounding/'+k:v for k,v in result['rollout_grounding']['means'].items()}, **result['replay'], **result['prototype'])
        for policy, values in result.get('counterfactuals',{}).items():
            log.update({f'mc/{policy}/{k}':v for k,v in values.items() if isinstance(v,(int,float))})
            log.update({f'mc/{policy}/{name}/{k}':v for name,group in values['action_classes'].items() for k,v in group.items()})
        run.log(log)
    return history, status


def make_runner(beta, small=False):
    from lambda_imitation.buffer import create_buffer
    from popgym_jax import PortEnv, JaxRunner
    args, config = config_for(beta, small); port = PortEnv(config['env_id'])
    state, fns, debug, sampler, components, context = build(config, port)
    _, bfns = create_buffer({k:v.shape[1:] for k,v in state.online_buffer.info.items()}, args.online_buffer_size, 1, [], [])
    runner = JaxRunner(args, port, fns, debug, bfns, args.memory_hidden_dim, True)
    return state, (args, config, port, runner, fns, debug, sampler, components, context)


def worker(root, beta, resume=False):
    import cloudpickle
    import jax
    import jax.numpy as jnp
    import numpy as np
    import wandb
    p = guard(root); folder = root / f'beta{beta}'
    if not resume:
        folder.mkdir(exist_ok=False)
    with (folder / '.worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state, built = make_runner(beta); args, config, port, runner, *_ = built
        if resume:
            status = json.loads((root / f'status-beta{beta}.json').read_text())
            decision = read_decision(root, beta, status)
            if status['state'] != 'awaiting-review' or not decision or decision['action'] != 'continue':
                raise ValueError('Resume requires explicit current review approval')
            checkpoint = folder / 'checkpoint.pkl'
            if flash.base.file_hash(checkpoint) != status['checkpoint_sha256']:
                raise ValueError('Resume checkpoint changed')
            payload = cloudpickle.loads(checkpoint.read_bytes())
            if payload['plan_sha256'] != flash.base.file_hash(root / 'plan.json') or payload['config'] != config or payload['steps'] != status['steps']:
                raise ValueError('Wrong resume state/config/plan')
            state = jax.tree.map(jnp.asarray,payload['state']); rollout = jax.tree.map(jnp.asarray,payload['rollout'])
            step, interactions, history = payload['steps'], payload['interactions'], payload['history']
            needs_publish = False
        else:
            write_json(folder / 'config.json', config)
            rollout = runner.initial_rollout(42); step=0; interactions=0; history=[]; needs_publish=True
        run = wandb.init(project=p['wandb_project'],entity=p['wandb_entity'],
            id=hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16],resume='must' if resume else 'never',
            config=config,name=f'Battleship-residual-categorical-beta{beta}-seed42',dir=str(folder))
        run.define_metric('steps');run.define_metric('*',step_metric='steps')
        finite = finite_checker()
        try:
            if not resume:
                state,rollout,count = jax.block_until_ready(runner.prefill(state,rollout,12800));interactions=int(count)
            while True:
                guard(root)
                if shutil.disk_usage(root).free < 2*1024**3:
                    raise OSError('Disk below2GiB;stop without deleting archives')
                if not bool(finite(state,rollout)):
                    raise FloatingPointError('Nonfinite model/optimizer/replay/rollout')
                if needs_publish:
                    history,status = publish(root,beta,state,rollout,step,interactions,history,built,run)
                    print('EVAL',beta,step,status['state'],flush=True)
                    if status['state'] in ('complete','awaiting-review'):
                        return
                needs_publish=True
                boundary=min(p['steps'],step+p['eval_every'])
                while step<boundary:
                    count=min(args.jax_train_steps,boundary-step)
                    state,rollout,metrics=jax.block_until_ready(runner.train(state,rollout,count))
                    step+=count;interactions+=count
                    if not all(np.isfinite(np.asarray(v)).all() for v in metrics.values()) or not bool(finite(state,rollout)):
                        raise FloatingPointError('Nonfinite training loss/model/optimizer/replay/rollout')
                    if bool(np.asarray(rollout[-2])) or int(state.update_step)!=step:
                        raise RuntimeError('Horizon/update-counter invariant failed')
                run.log({'steps':step,**{'train/'+k:float(v) for k,v in metrics.items()}})
        except BaseException:
            import traceback
            if shutil.disk_usage(root).free>1024**3:
                save_checkpoint(folder/f'failure-step{step}.pkl',dict(state=jax.device_get(state),rollout=jax.device_get(rollout),steps=step,config=config,history=history))
            status=dict(state='failed',beta=beta,steps=step,error=traceback.format_exc())
            write_json(folder/'status.json',status);write_json(root/f'status-beta{beta}.json',status)
            raise
        finally:
            run.finish()


def supervise(root,beta):
    p=guard(root); validation=json.loads((root/'validation.json').read_text())
    if not validation.get('passed') or validation['script_sha256']!=p['script_sha256']:
        raise ValueError('Full-size combined validation required before production')
    with (root/f'.supervisor-beta{beta}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        resume=False; attempt=0
        while True:
            guard(root)
            command=[sys.executable,__file__,'--root',str(root),'--worker',str(beta)]
            if resume:command+=['--resume']
            with (root/f'beta{beta}-worker{attempt}.log').open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            status=json.loads((root/f'status-beta{beta}.json').read_text())
            if status['state']=='complete':return
            if status['state']!='awaiting-review':raise ValueError('Worker ended outside review gate')
            while True:
                guard(root);d=read_decision(root,beta,status)
                if d:break
                time.sleep(15)
            if d['action']=='stop':
                stopped={**status,'state':'stopped-review','decision':d}
                write_json(root/f'status-beta{beta}.json',stopped);write_json(root/f'beta{beta}'/'status.json',stopped)
                return
            resume=True;attempt+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=BETAS)
    parser.add_argument('--supervise',type=int,choices=BETAS);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--decide',type=int,choices=BETAS);parser.add_argument('--at',type=int,choices=REVIEW_STEPS)
    parser.add_argument('--action',choices=('continue','stop'));parser.add_argument('--note',default='')
    args=parser.parse_args();root=args.root.resolve()
    if args.prepare:prepare(root)
    elif args.decide is not None:
        if args.at is None or args.action is None:parser.error('--decide needs --at and --action')
        decide(root,args.decide,args.at,args.action,args.note)
    elif args.worker is not None:worker(root,args.worker,args.resume)
    elif args.supervise is not None:supervise(root,args.supervise)
    else:parser.error('Choose prepare/worker/supervise/decide')


if __name__=='__main__':main()
