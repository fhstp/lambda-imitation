"""Frozen memory audit + W&B scalar/curve/table/artifact publication.

Original observable-history probes,episode-disjoint splits and horizon targets.
Policy histories change with training;uniform histories are the shared-policy
comparison,NOT fully matched mixed-mode probe-training distributions.
"""
import json
from pathlib import Path
import tempfile


def collect(state,built,c,folder,save_datasets):
    import jax
    import numpy as np
    import popgym_flash_memory_probe as probes
    args,config,port,fns,debug,*_=built
    dim=config['memory_hidden_dim']
    collector=probes.memory.make_collect(port,fns,debug,dim);encode=probes.make_encoder(debug,dim,64)
    pieces=[];parity=[];policy_rows=[]
    for mode_index,mode in enumerate(probes.MODES):
        for block in range(c['probe_blocks']):
            from battleship_sweep import disk_check
            disk_check(folder,c)
            # Fixed episode IDs and environment keys across checkpoints/seeds.
            seed=31000000+mode_index*100000+block*100
            data=jax.device_get(collector(state,seed,mode=mode,episodes=100))
            if not data['history_label_parity'].all():raise AssertionError('Native fired labels disagree')
            carry,probs,instant=jax.device_get(encode(state,data['observations'],data['actions'],data['hits']))
            np.testing.assert_allclose(probs[:,:-1],data['probs'],atol=1e-5,rtol=1e-4)
            if not np.isfinite(carry).all():raise FloatingPointError('Nonfinite frozen memory')
            item=dict(memory=carry,last_input=instant,**probes.history_labels(data,64),actions=data['actions'],outcomes=data['hits'],
                mode=np.full(100,mode_index,np.int8),episode_id=np.arange(seed,seed+100,dtype=np.int64),
                split=np.full(100,0 if block<c['probe_train_blocks'] else 1 if block<c['probe_train_blocks']+c['probe_val_blocks'] else 2,np.int8))
            pieces.append(item)
            active=data['active'].astype(bool);discounted=np.zeros_like(data['rewards']);v=np.zeros(100)
            for t in range(63,-1,-1):
                v=data['rewards'][:,t]+c['gamma']*(1-data['done'][:,t])*v;discounted[:,t]=v
            error=data['q'][active]-discounted[active]
            policy_rows.append(dict(mode=mode,return_mean=float(data['rewards'].sum(-1).mean()),
                repeat_fraction=float(data['repeat'].sum()/active.sum()),unique_cells=float(item['fired'][:,-1].sum(-1).mean()),
                actor_repeat_mass=float(data['seen_mass'].sum()/active.sum()),entropy=float(data['entropy'].sum()/active.sum()),
                q_mc_rmse=float(np.sqrt((error**2).mean())),q_mc_bias=float(error.mean())))
            if save_datasets:np.savez_compressed(folder/f'{mode}-block{block}.npz',**item)
            parity.append(dict(mode=mode,block=block,max_probability_error=float(np.max(np.abs(probs[:,:-1]-data['probs'])))))
    merged={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if len(np.unique(merged['episode_id']))!=len(merged['episode_id']):raise AssertionError('Episode split leakage')
    summaries={mode:{k:float(np.mean([r[k] for r in policy_rows if r['mode']==mode])) for k in policy_rows[0] if k!='mode'} for mode in probes.MODES}
    return merged,parity,summaries


def compact(data):
    import numpy as np
    import popgym_memory_horizon as h
    keep=data['valid'][:,64]&(data['counts'][:,64,0]==64)
    result=dict(memory=data['memory'][keep,64],recent=data['last_input'][keep,64],actions=data['actions'][keep].astype(np.int32),
        hits=data['outcomes'][keep].astype(bool),split=data['split'][keep],mode=data['mode'][keep],episode_id=data['episode_id'][keep])
    full=h.targets(result['actions'],result['hits'],'window',64)
    np.testing.assert_array_equal(full[:,:64],data['fired'][keep,64]);np.testing.assert_array_equal(full[:,64:],data['revealed_hits'][keep,64])
    return result,{mode:int(((data['mode']==i)&~keep).sum()) for i,mode in enumerate(('stochastic','greedy','random'))}


def reconstruction_rows(data,folder,seed,step):
    """Fixed heldout examples,NOT chosen for reconstruction quality."""
    import numpy as np
    rows=[];valid=data['valid'];ep,age=np.nonzero(valid);test=data['split'][ep]==2
    # Fit/prediction order follows the original run_probes flattening convention.
    with np.load(folder/'memory-mlp-heldout.npz',allow_pickle=False) as d:
        pred=d['prediction'];threshold=d['threshold']
        np.testing.assert_array_equal(d['episode_id'],data['episode_id'][ep[test]])
    eligible=data['episode_id'][data['split']==2]
    rng=np.random.default_rng(71000);selected=set(rng.choice(eligible,min(12,len(eligible)),replace=False).tolist())
    for index,(e,t) in enumerate(zip(ep[test],age[test])):
        if int(data['episode_id'][e]) not in selected or t not in (1,16,32,48,64):continue
        for target,start,key in (('fired',3,'fired'),('revealed_hits',67,'revealed_hits')):
            labels=data[key][e,t].astype(int);scores=pred[index,start:start+64]
            binary=scores>=threshold[start-3:start+61]
            for cell in range(64):rows.append([step,seed,int(data['episode_id'][e]),int(data['mode'][e]),int(t),target,
                cell//8,cell%8,int(labels[cell]),float(scores[cell]),int(binary[cell])])
    return rows


def analyze(state,built,c,policy,root,seed,step):
    import jax
    import numpy as np
    import popgym_flash_memory_probe as probes
    import popgym_memory_horizon as h
    import popgym_flash_prototypes as flash
    from battleship_sweep import atomic_json,disk_check
    folder=root/f'diagnostics/step{step}/seed{seed}'
    if folder.exists():
        raise FileExistsError(f'Preserved diagnostic attempt exists:{folder}. Inspect it;move it to a distinct preserved name before an explicit resume. Never overwrite/delete partial measurements.')
    folder.mkdir(parents=True,exist_ok=False)
    before=probes.memory.tree_digest(state);started=__import__('time').monotonic()
    settings=dict(probe_updates=c['probe_updates'],probe_chunk=200,probe_batch=512,probe_width=128,probe_lr=.001,bootstrap_replicates=200)
    # TemporaryDirectory owns ONLY new fit intermediates;no old archives touched.
    # 'all' explicitly retains full raw memories/models/heldout scores on local disk.
    with tempfile.TemporaryDirectory(prefix='probe-',dir=folder) as temporary:
        work=folder/'raw' if c['artifact_policy']=='all' else Path(temporary)
        if c['artifact_policy']=='all':work.mkdir()
        data,parity,policy_summary=collect(state,built,c,work,c['artifact_policy']=='all')
        run_probes=flash.base.normalization.clone_globals(probes.run_probes,{'require_disk':lambda:disk_check(folder,c)})
        full,coverage=run_probes(data,settings,work)
        reconstructions=reconstruction_rows(data,work,seed,step)
        small,excluded=compact(data);train=small['split']==0
        if not all((small['split']==s).any() for s in range(3)):raise ValueError('No horizon-eligible episodes in a split')
        fitters={}
        for model in h.MODELS:
            raw=small['memory'] if model.startswith('memory-') else small['recent']
            mean=raw[train].mean(0);scale=np.maximum(raw[train].std(0),1e-4);x=(raw-mean)/scale
            fit,predict=h.make_fitter(x.shape[1],model!='memory-linear',settings);fitters[model]=(fit,predict,x,mean,scale)
        horizons=[];ns=range(1,65) if policy['memory_suite']=='full' else (1,5,10,16,32,64)
        for task in ('window','lag'):
            for n in ns:
                disk_check(folder,c)
                # Per-task scratch files are kept only in 'all' mode. Metrics
                # remain complete in either policy,including all64 curve points.
                if c['artifact_policy']=='all':result=h.run_one(small,task,n,fitters,settings,work/task/f'n{n:02d}')
                else:
                    with tempfile.TemporaryDirectory(prefix=f'{task}{n}-',dir=temporary) as taskdir:
                        result=h.run_one(small,task,n,fitters,settings,Path(taskdir)/'fit')
                horizons.append(result)
    after=probes.memory.tree_digest(state)
    if before!=after:raise AssertionError('Diagnostics changed frozen training state')
    # Same-batch gradient vectors,clipping/edge mass,OFN/Q norms and replay
    # actor/main-Q grounding. Diagnostic float32 does NOT change training math.
    _,_,port,_,debug,sampler,components,context=built
    extra=flash.extra_probe(state,debug,sampler,components,context,'all-categorical',seed,step)
    extra.update(flash.base.probe(state,debug,sampler,context,jax.random.PRNGKey(28000000+seed)))
    if before!=probes.memory.tree_digest(state):raise AssertionError('Extra diagnostics changed frozen state')
    result=dict(seed=seed,step=step,full_history=full,coverage=coverage,horizons=horizons,excluded_early_episodes=excluded,
        policy_summary=policy_summary,extra_diagnostics=extra,
        certificate=dict(state_digest_before=before,state_digest_after=after,policy_memory_parity=parity,episode_split_disjoint=True),
        settings=settings,policy=policy,seconds=__import__('time').monotonic()-started)
    atomic_json(folder/'result.json',result)
    # Persist selected observable-history score maps regardless of raw retention.
    atomic_json(folder/'reconstructions.json',reconstructions)
    return result


def log_memory(run,results,seeds,step,folder,c):
    import wandb
    import numpy as np
    from battleship_sweep import flatten_scalars,seed_aggregate
    rows=[];scalars=[];reconstructions=[]
    for result in results:
        scalar={}
        for model,value in result['full_history'].items():
            by_mode=value.get('by_mode',{'pooled':value.get('overall',{})})
            for mode,metrics in by_mode.items():
                selected={'counts':{k:{m:v for m,v in values.items() if m in ('mae','rmse','r2')} for k,values in metrics.get('counts',{}).items()}}
                for target in ('fired','revealed_hits'):
                    selected[target]={k:v for k,v in metrics.get(target,{}).items() if k in ('balanced_accuracy','micro_auroc','positive_recall','negative_recall','exact_board_fraction')}
                scalar.update(flatten_scalars(selected,f'full_history/{model}/{mode}'))
        scalar.update(flatten_scalars(result['policy_summary'],'policy'))
        scalar.update(flatten_scalars(result['extra_diagnostics'],'grounding'))
        scalar['diagnostic_seconds']=result['seconds']
        scalar.update(flatten_scalars(result['excluded_early_episodes'],'excluded_early_episodes'))
        for task in result['horizons']:
            for model,entry in task['models'].items():
                groups=[(model,entry['metrics'])]
                if 'shuffled' in entry['metrics']:groups.append((model+'-shuffled',entry['metrics']['shuffled']))
                for name,metrics in groups:
                    for mode,targets in metrics.items():
                        if mode=='shuffled':continue
                        for target,m in targets.items():
                            a=m['episode_auroc'];ci=a['ci95'] or [None,None]
                            rows.append([step,result['seed'],task['task'],task['n'],mode,target,name,a['mean'],*ci,
                                a['eligible_episodes'],m['episodes'],m['positive_cells'],m['train_supported_cells']])
                            if task['n'] in (5,16,64) and mode=='random' and name in ('memory-linear','memory-mlp'):
                                scalar.update(flatten_scalars(dict(auroc=a['mean'],eligible=a['eligible_episodes']),f"horizon/{task['task']}/{task['n']}/{mode}/{target}/{name}"))
        scalars.append(scalar)
        reconstructions.extend(json.loads((folder/f'diagnostics/step{step}/seed{result["seed"]}/reconstructions.json').read_text()))
    columns=['steps','seed','task','n','mode','target','decoder','auroc','ci95_low','ci95_high','eligible_episodes','episodes','positive_cells','train_supported_cells']
    payload=seed_aggregate(scalars,seeds,'memory')
    # Each table stays below W&B's10k history-row limit,including10-seed trials.
    for seed in seeds:
        for task in ('window','lag'):
            payload[f'memory/horizon_table/{seed}/{task}']=wandb.Table(columns=columns,data=[r for r in rows if r[1]==seed and r[2]==task])
    # Whole curves in W&B native plots;mean over TRAINING seeds,not over cells.
    for task in ('window','lag'):
        for target in ('fired','hits'):
            for mode in ('random','stochastic','greedy','pooled'):
                xs=[];ys=[];keys=[]
                for decoder in ('memory-linear','memory-mlp','memory-linear-shuffled','memory-mlp-shuffled','recent-input-mlp','prevalence'):
                    selected=[r for r in rows if r[2]==task and r[4]==mode and r[5]==target and r[6]==decoder and r[7] is not None]
                    ns=sorted({r[3] for r in selected})
                    if ns:
                        xs.append(ns);ys.append([float(np.mean([r[7] for r in selected if r[3]==n])) for n in ns]);keys.append(decoder)
                if xs:payload[f'memory/curves/{task}/{target}/{mode}']=wandb.plot.line_series(xs,ys,keys=keys,title=f'{task} {target} {mode} at{step}',xname='N / lag')
    for seed in seeds:
        payload[f'memory/reconstructions/{seed}']=wandb.Table(columns=['steps','seed','episode_id','mode','age','target','row','column','label','class_weighted_score','validation_binary'],data=[r for r in reconstructions if r[1]==seed])
        # Actual four-map visuals inside W&B,not just a large numeric cell table.
        from PIL import Image
        groups={}
        for row in reconstructions:
            if row[1]==seed:groups.setdefault(tuple(row[2:6]),[]).append(row)
        images=[]
        for key,cells in sorted(groups.items()):
            cells=sorted(cells,key=lambda r:(r[6],r[7]))
            truth=np.array([r[8] for r in cells]).reshape(8,8)
            score=np.array([r[9] for r in cells]).reshape(8,8)
            binary=np.array([r[10] for r in cells]).reshape(8,8)
            maps=[truth,score,binary,np.abs(binary-truth)]
            canvas=np.zeros((8,35,3),np.uint8)
            for index,values in enumerate(maps):
                rgb=np.stack((values,.2+.6*values,1-values),-1)
                canvas[:,index*9:index*9+8]=np.asarray(np.clip(rgb,0,1)*255,np.uint8)
            image=Image.fromarray(canvas).resize((560,128),resample=Image.Resampling.NEAREST)
            caption='Observed history | class-weighted score (NOT calibrated ship probability) | validation binary | error'
            images.append([step,seed,*key,wandb.Image(image,caption=caption)])
        payload[f'memory/maps/{seed}']=wandb.Table(columns=['steps','seed','episode_id','mode','age','target','four_maps'],data=images)
    # Full-history episode-age/last-fire-lag curves are available directly in
    # W&B as well as the complete JSONs. Positive recall is NOT lag AUROC.
    age_rows=[];lag_rows=[]
    for result in results:
        for decoder,v in result['full_history'].items():
            for age,metrics in v.get('by_age',{}).items():
                for target in ('fired','revealed_hits'):
                    m=metrics[target]
                    age_rows.append([step,result['seed'],decoder,age,target,m['balanced_accuracy'],m['micro_auroc'],m['positive_cells']])
            for mode,metrics in v.get('by_mode',{}).items():
                for target in ('fired','revealed_hits'):
                    for lag,m in metrics[target]['positive_recall_by_steps_since_last_shot'].items():
                        lag_rows.append([step,result['seed'],decoder,mode,target,lag,m['recall'],m['cells']])
    payload['memory/episode_age']=wandb.Table(columns=['steps','seed','decoder','episode_age','target','balanced_accuracy','micro_auroc','positive_cells'],data=age_rows)
    payload['memory/last_fire_lag']=wandb.Table(columns=['steps','seed','decoder','mode','target','steps_since_last_fire','positive_recall','positive_cells'],data=lag_rows)
    artifact=wandb.Artifact(f'memory-{run.id}-step{step}',type='memory-diagnostics',metadata=dict(step=step,seeds=seeds,policy=c['artifact_policy']))
    for seed in seeds:
        base=folder/f'diagnostics/step{step}/seed{seed}'
        for name in ('result.json','reconstructions.json'):artifact.add_file(str(base/name),name=f'seed{seed}/{name}')
        if c['artifact_policy']=='all':artifact.add_dir(str(base/'raw'),name=f'seed{seed}/raw')
    run.log_artifact(artifact)
    return payload
