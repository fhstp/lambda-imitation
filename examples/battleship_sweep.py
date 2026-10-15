"""Portable Battleship residual/categorical Bayesian trials with vmapped seeds.

No host-specific service/plan guards are used. The archived experiment modules
are imported ONLY as pure builders/objectives/collectors. This is a NEW study.
"""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time

DEFAULTS=dict(n_seeds=3,seed_base=42,max_steps=5000000,lambda_coef=10.,
    fe_lr=1e-4,actor_lr=1e-4,critic_lr=1e-4,alpha_initial=1.,alpha_final=.01,
    alpha_decay_steps=500000,tau=.005,memory_hidden_dim=512,batch_size=128,
    online_buffer_size=100000,projection_dim=128,gamma=.99,lambda1=.05,lambda2=.85,
    prefill_steps=12800,chunk_steps=256,eval_episodes=100,probe_updates=2000,
    probe_blocks=9,probe_train_blocks=6,probe_val_blocks=1,memory_suite='full',
    memory_schedule='dense',memory_milestones=[100000,500000,1000000,2000000,3000000,4000000,5000000,10000000],
    runtime_policy='',artifact_policy='metrics',min_free_gib=10.,validate_updates=256)


def resolve_config(values):
    unknown=set(values)-set(DEFAULTS)
    if unknown:raise ValueError(f'Unknown configuration keys:{sorted(unknown)}')
    c={**DEFAULTS,**values}
    for k in ('n_seeds','max_steps','seed_base','alpha_decay_steps','memory_hidden_dim','batch_size',
              'online_buffer_size','projection_dim','prefill_steps','chunk_steps','eval_episodes',
              'probe_updates','probe_blocks','probe_train_blocks','probe_val_blocks','validate_updates'):
        if isinstance(c[k],bool) or int(c[k])!=c[k]:raise ValueError(f'{k} must be an integer')
        c[k]=int(c[k])
        if c[k]<1 and k!='seed_base':raise ValueError(f'{k} must be positive')
    if c['seed_base']<0 or c['seed_base']+c['n_seeds']>=2**31:raise ValueError('Invalid seed range')
    if c['max_steps']>10000000 or c['max_steps']%10000:raise ValueError('Budget must be a10k multiple,at most10M')
    if c['probe_updates']%200:raise ValueError('Probe budget must be a multiple of200')
    if c['chunk_steps']%16:raise ValueError('Training chunks must preserve16-step collect/update blocks')
    if c['probe_train_blocks']+c['probe_val_blocks']>=c['probe_blocks']:raise ValueError('Need disjoint train/val/test blocks')
    if c['online_buffer_size']<max(256,c['prefill_steps']):raise ValueError('Replay must hold prefill/complete episodes')
    if c['memory_suite'] not in ('full','tiered'):raise ValueError('Unknown memory suite')
    if c['memory_schedule'] not in ('dense','milestones'):raise ValueError('Unknown memory schedule')
    if c['artifact_policy'] not in ('metrics','all'):raise ValueError('Unknown artifact policy')
    for k in ('fe_lr','actor_lr','critic_lr','alpha_initial','alpha_final','tau','min_free_gib'):
        if not 0<float(c[k])<float('inf'):raise ValueError(f'Invalid {k}')
    if not 0<=float(c['lambda_coef'])<float('inf') or not 0<c['tau']<=1:raise ValueError('Invalid beta/tau')
    if not 0<c['gamma']<1 or not 0<=c['lambda1']<c['lambda2']<=1:raise ValueError('Invalid discount/lambdas')
    if any(isinstance(s,bool) or int(s)!=s or s<=0 or s>10000000 or s%10000 for s in c['memory_milestones']):
        raise ValueError('Memory milestones must be positive10k multiples through10M')
    return c


def memory_steps(max_steps):
    return sorted(s for s in {*range(10000,100001,10000),*range(200000,1000001,100000),
        *range(1500000,5000001,500000),*range(6000000,10000001,1000000)} if s<=max_steps)


def scheduled(step,c,policy):
    # Policy is a separately logged observation policy,NEVER a training override.
    return step==c['max_steps'] or step in (memory_steps(c['max_steps']) if policy['memory_schedule']=='dense' else policy['memory_milestones'])


def read_policy(c):
    allowed=('memory_schedule','memory_suite','memory_milestones')
    policy={k:c[k] for k in allowed}
    if c['runtime_policy']:
        path=Path(c['runtime_policy'])
        if path.exists():
            v=json.loads(path.read_text())
            if set(v)-set(allowed):raise ValueError('Runtime policy may change diagnostics ONLY')
            checked=resolve_config({**c,**v});policy={k:checked[k] for k in allowed}
    return policy


def atomic_json(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_versions():
    return {k:importlib.metadata.version(k) for k in ('jax','jaxlib','flax','optax','numpy','cloudpickle','popjym','gymnax')}


def source_manifest():
    base=Path(__file__).resolve().parents[1]
    paths=sorted((base/'src/lambda_imitation').glob('*.py'))+sorted((base/'examples').glob('*.py'))
    return {str(p.relative_to(base)):file_hash(p) for p in paths}


def disk_check(folder,c):
    if shutil.disk_usage(folder).free<c['min_free_gib']*1024**3:raise OSError('Disk safety reserve reached;no automatic archive deletion')


def deterministic_projection(probabilities,values):
    """Same linear categorical projection,NO colliding scatter-add atomics.

    Dense triangular interpolation reduces over the source atoms in a fixed
    tensor reduction. Algebraically the same law;rounding/trajectories can differ
    from archived scatter kernels. New study,NOT a legacy bitwise-parity claim.
    """
    import jax.numpy as jnp
    import popgym_flash_prototypes as flash
    z=flash.support();coordinate=(jnp.clip(values,z[0],z[-1])-z[0])/(z[-1]-z[0])*(flash.ATOMS-1)
    weights=jnp.maximum(1.-jnp.abs(coordinate[...,None]-jnp.arange(flash.ATOMS)),0.)
    return jnp.sum(probabilities[...,None]*weights,axis=1)


def alpha_at(step,c):
    import jax.numpy as jnp
    t=jnp.clip(jnp.asarray(step)/c['alpha_decay_steps'],0.,1.)
    return jnp.exp((1-t)*jnp.log(jnp.float32(c['alpha_initial']))+t*jnp.log(jnp.float32(c['alpha_final'])))


def builder_config(c,small=False):
    """Standalone defaults;NEVER call old config_for functions with host guards."""
    from popgym_suite import parser,run_config
    args=parser().parse_args([])
    for k in ('fe_lr','actor_lr','critic_lr','tau','memory_hidden_dim','batch_size','online_buffer_size',
              'projection_dim','gamma','lambda1','lambda2','lambda_coef','prefill_steps','eval_episodes'):
        setattr(args,k,c[k])
    args.alpha=c['alpha_initial'];args.steps=c['max_steps'];args.max_episode_steps=64
    args.burn_in_length=0;args.sequence_length=64;args.lambda_truncation=1
    args.memory_type='gru';args.collect_steps=16;args.updates_per_step=1
    args.head_dim=8 if small else 512
    config=run_config(args,dict(env_id='popgym-BattleshipEasy-v0',seed=c['seed_base'],method='lambda',backend='jax',
        critic_normalization_arm='ofn',episode_sampling='complete-uniform-episodes',
        alpha_schedule='configured-geometric-then-fixed',flash_arm='all-categorical',categorical_atoms=51,
        categorical_support=[-1.5,1.5],residual_actor_width=8 if small else 128,residual_critic_width=8 if small else 256,
        residual_blocks=2,residual_expansion=4,target_steps=c['max_steps'],snapshot_retention=[],
        architecture='RMS-residual-all-categorical-critic-OFN',experiment='portable-b200-bayes',
        categorical_projection='dense-triangular-no-scatter'))
    config['lambda_td_loss']='two_hot_cross_entropy'
    return args,config


def build(c,small=False):
    import jax
    import jax.numpy as jnp
    from lambda_imitation.buffer import create_buffer
    import popgym_flash_combined as original
    import popgym_flash_prototypes as flash
    from popgym_critic_aligned import replace_captures
    from popgym_jax import JaxRunner,PortEnv
    clone=flash.base.normalization.clone_globals
    args,config=builder_config(c,small)
    # Clone ONLY function globals for the new objective;no global monkey patch.
    two_hot=lambda values:deterministic_projection(jnp.ones((values.size,1)),values.reshape(-1,1))
    objective=clone(flash.categorical_objective,{'project_distribution':deterministic_projection,'two_hot':two_hot})
    port=PortEnv(config['env_id']);states=[]
    for seed in range(c['seed_base'],c['seed_base']+c['n_seeds']):
        sc={**config,'seed':seed};state,fns,debug=original.networks(sc,port)
        states.append(state)
    sampler=flash.base.make_sampler(config['batch_size'])
    loss,components,context=objective(fns,debug,sampler,'all-categorical')
    update_original=replace_captures(fns.update,{'loss_combined':loss})
    @jax.jit
    def update(state,key):
        a=alpha_at(state.update_step,c)
        state,metrics=update_original(state._replace(alpha=a,log_alpha=jnp.log(a)),key)
        a=alpha_at(state.update_step,c)
        return state._replace(alpha=a,log_alpha=jnp.log(a)),metrics
    fns=fns._replace(update=update)
    _,bfns=create_buffer({k:v.shape[1:] for k,v in states[0].online_buffer.info.items()},c['online_buffer_size'],1,[],[])
    dim=context['feature_extractor'].initialize_carry(1).shape[-1]
    runner=JaxRunner(args,port,fns,debug,bfns,dim,True)
    stacked=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    seeds=jnp.arange(c['seed_base'],c['seed_base']+c['n_seeds'])
    rollouts=jax.vmap(runner.initial_rollout)(seeds)
    return stacked,rollouts,runner,(args,config,port,fns,debug,sampler,components,context)


class SeedRunner:
    """Actual vmap of the FULL update/replay/rollout/RNG state,not a seed loop."""
    def __init__(self,runner):
        import jax
        @partial(jax.jit,static_argnames=('n_steps',))
        def train(states,rollouts,n_steps):return jax.vmap(lambda s,r:runner.train(s,r,n_steps))(states,rollouts)
        @partial(jax.jit,static_argnames=('n_steps',))
        def prefill(states,rollouts,n_steps):return jax.vmap(lambda s,r:runner.prefill(s,r,n_steps))(states,rollouts)
        self.train,self.prefill=train,prefill


def exact(a,b):
    import jax
    import numpy as np
    if jax.tree.structure(a)!=jax.tree.structure(b):raise AssertionError('Tree structure changed')
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        x,y=np.asarray(x),np.asarray(y)
        np.testing.assert_array_equal(x,y)
        if x.dtype!=y.dtype or x.tobytes()!=y.tobytes():raise AssertionError('Native state bytes changed')


def finite(tree):
    import jax
    import numpy as np
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree)):raise FloatingPointError('Nonfinite state/loss')


def save_checkpoint(path,state,rollout,c,step,history,run_id,manifest):
    import cloudpickle
    import jax
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:cloudpickle.dump(dict(state=jax.device_get(state),rollout=jax.device_get(rollout),
        config=c,steps=step,history=history,run_id=run_id,source_manifest=manifest,runtime_versions=runtime_versions()),f)
    tmp.replace(path)


def validate(c,folder,small=False,require_gpu=True):
    """Fail closed on same-kernel nondeterminism,cross-lane coupling or resume."""
    import cloudpickle
    import jax
    import numpy as np
    if require_gpu and jax.default_backend()!='gpu':raise RuntimeError('B200 launch gate requires a real GPU')
    state,rollout,runner,built=build(c,small);batch=SeedRunner(runner)
    state,rollout,count=jax.block_until_ready(batch.prefill(state,rollout,c['prefill_steps']))
    finite((state,rollout))
    if not np.all((np.asarray(count)>=c['prefill_steps'])&(np.asarray(count)<=c['prefill_steps']+64)):
        raise AssertionError('Prefill must end at the first natural episode boundary')
    np.testing.assert_array_equal(rollout[-1],np.zeros(c['n_seeds'],int))
    start=time.monotonic();first=jax.block_until_ready(batch.train(state,rollout,c['validate_updates']))
    exact(first,jax.block_until_ready(batch.train(state,rollout,c['validate_updates'])))
    info=dict(state.online_buffer.info);info['rewards']=info['rewards'].at[0].add(1.)
    changed=state._replace(online_buffer=state.online_buffer._replace(info=info))
    changed_rollout=(rollout[0].at[0].set(jax.random.PRNGKey(987654)),*rollout[1:])
    other=jax.block_until_ready(batch.train(changed,changed_rollout,c['validate_updates']))
    exact(jax.tree.map(lambda x:x[1:],first),jax.tree.map(lambda x:x[1:],other))
    if c['n_seeds']>1:
        if np.array_equal(np.asarray(first[1][0][0]),np.asarray(other[1][0][0])):raise AssertionError('Perturbed lane did not change')
    # Restore DEVICE arrays after host persistence,matching real resume behavior.
    restored=cloudpickle.loads(cloudpickle.dumps(jax.device_get(first[:2])))
    restored=jax.tree.map(jax.numpy.asarray,restored)
    exact(jax.block_until_ready(batch.train(first[0],first[1],c['validate_updates'])),
          jax.block_until_ready(batch.train(*restored,c['validate_updates'])))
    finite(first);np.testing.assert_array_equal(first[0].update_step,np.full(c['n_seeds'],c['validate_updates']))
    if np.asarray(first[1][-2]).any():raise AssertionError('Environment horizon overflow')
    # Verify the deterministic projection against the original mathematical law.
    import popgym_flash_prototypes as flash
    x=jax.numpy.linspace(-2,2,51)[None,:];p=jax.numpy.full_like(x,1/51)
    np.testing.assert_allclose(deterministic_projection(p,x),flash.project_distribution(p,x),atol=2e-7,rtol=2e-6)
    proof=dict(passed=True,config=c,small=small,device=str(jax.devices()[0]),device_kind=jax.devices()[0].device_kind,backend=jax.default_backend(),
        repeatability=True,lane_isolation=True,host_device_resume=True,finite_training=True,
        source_manifest=source_manifest(),versions=runtime_versions(),
        validation_seconds=time.monotonic()-start,projection='explicit dense deterministic interpolation;NOT native legacy equivalence',
        safety='Validation states are discarded. No precision/tolerance relaxation on failure.')
    atomic_json(folder/'validation.json',proof)
    return proof


def flatten_scalars(value,prefix=''):
    import math
    result={}
    if isinstance(value,dict):
        for k,v in value.items():result.update(flatten_scalars(v,prefix+'/'+str(k) if prefix else str(k)))
    elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):result[prefix]=float(value)
    return result


def seed_aggregate(rows,seeds,prefix):
    import numpy as np
    result={}
    for seed,row in zip(seeds,rows):result.update({f'seed/{seed}/{prefix}/{k}':v for k,v in row.items()})
    for k in sorted(set().union(*(r.keys() for r in rows))):
        values=[r[k] for r in rows if k in r]
        result[f'{prefix}/{k}/mean']=float(np.mean(values));result[f'{prefix}/{k}/std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        result[f'{prefix}/{k}/n_seeds']=len(values)
    return result


def run_trial(values,output,resume=None):
    import cloudpickle
    import jax
    import numpy as np
    import wandb
    from battleship_memory import analyze,log_memory,optional_diagnostic,DiagnosticUnavailable
    from popgym_jax import evaluation_result
    saved=None
    if resume:
        with Path(resume).open('rb') as f:saved=cloudpickle.load(f)
        if saved['source_manifest']!=source_manifest():raise ValueError('Resume source manifest differs')
        if saved['runtime_versions']!=runtime_versions():raise ValueError('Resume runtime versions differ')
        values=saved['config']
    run=wandb.init(config=values,id=saved['run_id'] if saved else None,resume='must' if saved else None)
    c=resolve_config(dict(run.config));folder=output/run.id
    folder.mkdir(parents=True,exist_ok=bool(saved));manifest=source_manifest();seeds=list(range(c['seed_base'],c['seed_base']+c['n_seeds']))
    import fcntl
    lock=(folder/'.trial.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.config.update(c,allow_val_change=False)
    run.define_metric('steps');run.define_metric('*',step_metric='steps')
    atomic_json(folder/'config.json',c);atomic_json(folder/'source-manifest.json',manifest)
    # Revalidate each trial's actual shapes and seed count;never fallback to scalar.
    try:
        disk_check(folder,c);proof=validate(c,folder)
        artifact=wandb.Artifact(f'validation-{run.id}',type='validation');artifact.add_file(str(folder/'validation.json'));run.log_artifact(artifact)
        jax.clear_caches()
        states,rollouts,runner,built=build(c);batch=SeedRunner(runner)
        if saved:
            states=jax.tree.map(jax.numpy.asarray,saved['state']);rollouts=jax.tree.map(jax.numpy.asarray,saved['rollout'])
            step=saved['steps'];history=saved['history']
        else:
            states,rollouts,_=jax.block_until_ready(batch.prefill(states,rollouts,c['prefill_steps']));step=0;history=[]
            save_checkpoint(folder/'latest.pkl',states,rollouts,c,step,history,run.id,manifest)
        args,config,port,fns,debug,sampler,components,context=built
        warning_path=folder/'diagnostic-warnings.jsonl'
        warning_count=sum(1 for line in warning_path.read_text().splitlines() if line) if warning_path.exists() else 0
        def diagnostic_warning(seed,stage,message,details=None,error=None):
            nonlocal warning_count
            warning_count+=1
            if error is not None:
                import traceback
                # Preserve the cause locally,without exposing exception URLs/tokens
                # in the public structured warning consumed by the dashboard.
                with (folder/'diagnostic-errors.log').open('a') as f:
                    f.write(f'\nstep={step} seed={seed} stage={stage}\n'+''.join(traceback.format_exception(error)))
                message+=';local traceback: diagnostic-errors.log'
            record=dict(steps=step,seed=seed,stage=stage,message=message,details=details,wall_time=time.time())
            with warning_path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print('DIAGNOSTIC WARNING '+json.dumps(record,allow_nan=False),flush=True)
            # Local evidence survives even if optional W&B warning logging fails.
            try:run.log({'steps':step,'diagnostics/warning_count':warning_count,'diagnostics/warning_json':json.dumps(record,allow_nan=False)})
            except Exception as error:print(f'Warning publication unavailable ({type(error).__name__});retained in {warning_path}',flush=True)
        last_policy=None;next_eval=((step//10000)+1)*10000;started=time.monotonic();start_step=step
        while step<c['max_steps']:
            disk_check(folder,c);n=min(c['chunk_steps'],next_eval-step,c['max_steps']-step)
            states,rollouts,metrics=jax.block_until_ready(batch.train(states,rollouts,n));step+=n
            finite(metrics)
            if np.asarray(rollouts[-2]).any():raise RuntimeError('Environment exceeded64-shot horizon')
            np.testing.assert_array_equal(states.update_step,np.full(c['n_seeds'],step))
            if step!=next_eval and step!=c['max_steps']:continue
            if source_manifest()!=manifest:raise ValueError('Study sources changed during the trial')
            policy=read_policy(c)
            if policy!=last_policy:
                record=dict(steps=step,policy=policy,wall_time=time.time())
                with (folder/'diagnostic-policy.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                run.log({'steps':step,'diagnostic_policy':wandb.Table(columns=['step','policy'],data=[[step,json.dumps(policy)]])})
                last_policy=policy
            # Dense cheap evaluation remains fixed,independent of memory frequency.
            should_eval=step in memory_steps(c['max_steps']) or step==c['max_steps']
            payload={'steps':step}
            if should_eval:
                evaluations=[]
                for lane,seed in enumerate(seeds):
                    state=jax.tree.map(lambda x:x[lane],states)
                    outputs=jax.block_until_ready(runner.evaluate(state,100000000+seed*10000,'stochastic',c['eval_episodes']))
                    evaluations.append(evaluation_result(outputs,'stochastic',args,True)['metrics'])
                payload.update(seed_aggregate(evaluations,seeds,'eval/stochastic'))
                history.append(dict(steps=step,seed_returns=[e['return_mean'] for e in evaluations]))
            if scheduled(step,c,policy):
                finite((states,rollouts))
                from popgym_memory_diagnostics import tree_digest
                before=tree_digest((states,rollouts))
                atomic_json(folder/'status.json',dict(state='observing',steps=step,policy=policy))
                memory_results=[];complete=False
                try:
                    for lane,seed in enumerate(seeds):
                        state=jax.tree.map(lambda x:x[lane],states)
                        try:
                            memory_results.append(analyze(state,built,c,policy,folder,seed,step,
                                on_warning=lambda stage,message,details,seed=seed:diagnostic_warning(seed,stage,message,details)))
                        except DiagnosticUnavailable as error:
                            diagnostic_warning(seed,error.stage,str(error),dict(error_type=error.error_type),error=error)
                            break # Do not publish a partial seed set as a complete memory checkpoint.
                    if len(memory_results)==len(seeds):
                        try:
                            memory_payload=optional_diagnostic('memory_publication',lambda:log_memory(run,memory_results,seeds,step,folder,c))
                            # Media serialization/upload is OPTIONAL;never mix it into
                            # the required return/training scalar publication below.
                            optional_diagnostic('memory_publication',lambda:run.log({'steps':step,**memory_payload}))
                            complete=True
                        except DiagnosticUnavailable as error:
                            diagnostic_warning(None,error.stage,str(error),dict(error_type=error.error_type),error=error)
                finally:
                    # This remains fatal even when the optional measurement failed.
                    if before!=tree_digest((states,rollouts)):raise AssertionError('Diagnostics modified training/rollout/RNG state')
                payload['diagnostics/memory_complete']=int(complete)
                payload['diagnostics/warning_count']=warning_count
            if should_eval or scheduled(step,c,policy):
                payload.update(seed_aggregate([{k:float(np.asarray(v)[i]) for k,v in metrics.items()} for i in range(c['n_seeds'])],seeds,'train'))
                payload['throughput/online_steps_per_second_per_seed']=(step-start_step)/max(time.monotonic()-started,1e-6)
                run.log(payload)
                save_checkpoint(folder/'latest.pkl',states,rollouts,c,step,history,run.id,manifest)
                atomic_json(folder/'status.json',dict(state='running',steps=step,config=c))
            next_eval+=10000
        # Identical final-budget objective for all trials;NOT best checkpoint,
        # not memory or a differently aged partial score. Search/finalist separate.
        last=history[-5:];per_seed=np.mean([r['seed_returns'] for r in last],axis=0)
        checkpoint=wandb.Artifact(f'checkpoint-{run.id}',type='checkpoint',metadata=dict(steps=step,config=c))
        checkpoint.add_file(str(folder/'latest.pkl'));run.log_artifact(checkpoint,aliases=['final'])
        run.log({'steps':step,'objective/return_mean':float(per_seed.mean()),'objective/return_seed_std':float(per_seed.std(ddof=1)) if len(seeds)>1 else 0.,'objective/checkpoints':len(last)})
        atomic_json(folder/'status.json',dict(state='complete',steps=step));run.finish()
    except BaseException:
        import traceback
        atomic_json(folder/'status.json',dict(state='failed',error=traceback.format_exc()))
        run.finish(exit_code=1);raise
    finally:lock.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,default=Path(os.environ.get('BATTLESHIP_OUTPUT','runs/battleship-sweep')))
    parser.add_argument('--resume',type=Path);parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--cpu-smoke',action='store_true',help='Tiny CPU validation only,never a production certificate')
    parser.add_argument('--diagnostics-smoke',action='store_true',help='Also exercise frozen probe/W&B publication with small diagnostic budgets')
    parser.add_argument('--full-memory-smoke',action='store_true',help='Smoke all64 window/lag horizons instead of representative horizons')
    args,overrides=parser.parse_known_args()
    values=json.loads(args.config.read_text()) if args.config else {}
    for item in overrides:
        if not item.startswith('--') or '=' not in item:parser.error('Overrides use --key=value (W&B ${args})')
        key,value=item[2:].split('=',1)
        try:values[key]=json.loads(value)
        except ValueError:values[key]=value
    args.output.mkdir(parents=True,exist_ok=True)
    if args.cpu_smoke:
        values.update(n_seeds=3,max_steps=10000,memory_hidden_dim=8,batch_size=2,online_buffer_size=256,
            projection_dim=8,prefill_steps=128,validate_updates=2,probe_updates=200,min_free_gib=.1)
    if args.validate_only or args.cpu_smoke:
        c=resolve_config(values);folder=args.output/('cpu-smoke' if args.cpu_smoke else 'validation');folder.mkdir(exist_ok=False)
        validate(c,folder,small=args.cpu_smoke,require_gpu=not args.cpu_smoke)
        if args.diagnostics_smoke:
            import jax
            import wandb
            from battleship_memory import analyze,log_memory
            smoke={**c,'probe_blocks':3,'probe_train_blocks':1,'probe_val_blocks':1,'probe_updates':200,
                'memory_suite':'full' if args.full_memory_smoke else 'tiered'}
            states,rollouts,runner,built=build(smoke,small=args.cpu_smoke)
            states,rollouts,_=jax.block_until_ready(SeedRunner(runner).prefill(states,rollouts,smoke['prefill_steps']))
            state=jax.tree.map(lambda x:x[0],states)
            result=analyze(state,built,smoke,read_policy(smoke),folder,smoke['seed_base'],0)
            run=wandb.init(project='battleship-sweep-validation',config=smoke,job_type='diagnostic-smoke',mode='offline' if args.cpu_smoke else None)
            run.log({'steps':0,**log_memory(run,[result],[smoke['seed_base']],0,folder,smoke)})
            run.finish()
            atomic_json(folder/'diagnostic-proof.json',dict(passed=True,config=smoke,seed=smoke['seed_base'],
                certificate=result['certificate'],production_data_reused=False,
                scope=f'One lane,three collection blocks/mode,200 probe updates,{smoke["memory_suite"]} horizons;NOT a full-budget production performance claim'))
    else:run_trial(values,args.output,args.resume)


if __name__=='__main__':main()
