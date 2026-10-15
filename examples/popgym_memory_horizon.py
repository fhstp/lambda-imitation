"""Independent last-N / exact-lag probes using ONLY saved1M NPZ memories.

No checkpoint/model pickle loads,new RL rollouts,agent training or interventions.
Every horizon uses the same post64-shot memories and episode-disjoint splits.
"""
import argparse
from datetime import datetime
import fcntl
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import popgym_flash_memory_probe as original
from popgym_alpha_sweep import file_hash
from popgym_suite import write_json

ROOT=Path('/tmp/opencode/popgym-memory-horizon')
PARENT=Path('/tmp/opencode/popgym-flash-combined-memory')
BETAS=(0,1,10)
TASKS=('window','lag')
MODELS=('memory-linear','memory-mlp','recent-input-mlp')
MODES=('stochastic','greedy','random')
SETTINGS=dict(probe_updates=2000,probe_chunk=200,probe_batch=512,probe_width=128,probe_lr=.001,bootstrap_replicates=200)


def targets(actions,hits,task,n):
    """Window = union of last n actions;lag1 = most recent executed action."""
    actions=np.asarray(actions);hits=np.asarray(hits)
    if task not in TASKS or not 1<=n<=64 or actions.shape!=hits.shape or actions.ndim!=2 or actions.shape[1]!=64:
        raise ValueError('Expected64 executed actions and valid task/n')
    if np.any((actions<0)|(actions>=64)):raise ValueError('Invalid action')
    indices=np.arange(64-n,64) if task=='window' else np.array([64-n])
    fired=np.zeros((len(actions),64),bool);revealed=np.zeros_like(fired)
    for t in indices:
        fired[np.arange(len(actions)),actions[:,t]]=True
        good=np.flatnonzero(hits[:,t].astype(bool));revealed[good,actions[good,t]]=True
    return np.concatenate((fired,revealed),-1)


def load_dataset(folder):
    pieces=[];excluded={}
    for mode_index,mode in enumerate(MODES):
        excluded[mode]=0
        for block in range(9):
            with np.load(folder/f'{mode}-block{block}.npz',allow_pickle=False) as d:
                keep=d['valid'][:,64] & (d['counts'][:,64,0]==64)
                excluded[mode]+=int((~keep).sum())
                if np.any(d['split']!=(0 if block<6 else 1 if block<7 else 2)) or np.any(d['mode']!=mode_index):
                    raise ValueError('Original mode/split mismatch')
                item=dict(memory=d['memory'][keep,64],recent=d['last_input'][keep,64],
                    actions=d['actions'][keep].astype(np.int32),hits=d['outcomes'][keep].astype(bool),
                    split=d['split'][keep],mode=d['mode'][keep],episode_id=d['episode_id'][keep])
                full=targets(item['actions'],item['hits'],'window',64)
                np.testing.assert_array_equal(full[:,:64],d['fired'][keep,64])
                np.testing.assert_array_equal(full[:,64:],d['revealed_hits'][keep,64])
                pieces.append(item)
    result={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    if len(np.unique(result['episode_id']))!=len(result['episode_id']):raise ValueError('Episode split leakage')
    if result['memory'].shape[1]!=512 or not all(np.isfinite(result[k]).all() for k in ('memory','recent')):
        raise ValueError('Expected finite GRU512/recent inputs')
    if not all((result['split']==i).any() for i in range(3)):raise ValueError('Empty split')
    return result,excluded


def episode_auc(scores,labels):
    """Tie-aware per-board AUROC;single-class boards remain unavailable."""
    scores=np.asarray(scores);labels=np.asarray(labels,bool)
    if scores.shape!=labels.shape or scores.ndim!=2:raise ValueError('Expected episode x cell scores')
    if not np.isfinite(scores).all():raise ValueError('Nonfinite scores')
    # Only64 cells:pairwise formulation is transparent and cheap at this scale.
    p=labels.sum(1);n=(~labels).sum(1)
    wins=((scores[:,:,None]>scores[:,None,:])+.5*(scores[:,:,None]==scores[:,None,:]))
    wins=(wins*labels[:,:,None]*(~labels[:,None,:])).sum((1,2))
    return np.divide(wins,p*n,out=np.full(len(labels),np.nan),where=(p*n)>0)


def mean_ci(values,modes,repeats=200,seed=75001):
    values=np.asarray(values,float);modes=np.asarray(modes);valid=np.isfinite(values)
    if not valid.any():return dict(mean=None,ci95=None,eligible_episodes=0)
    rng=np.random.default_rng(seed);draws=[]
    strata=[np.flatnonzero(modes==mode) for mode in np.unique(modes)]
    for _ in range(repeats):
        sampled=np.concatenate([rng.choice(rows,len(rows),replace=True) for rows in strata])
        usable=values[sampled];usable=usable[np.isfinite(usable)]
        if len(usable):draws.append(float(usable.mean()))
    interval=np.quantile(draws,[.025,.975]).tolist() if valid.sum()>=2 and draws else None
    return dict(mean=float(values[valid].mean()),ci95=interval,eligible_episodes=int(valid.sum()))


def metrics(scores,labels,threshold,support,modes,repeats=200,comparison=None):
    result={}
    for name,select in [('pooled',np.ones(len(labels),bool)),*[(name,modes==i) for i,name in enumerate(MODES)]]:
        if not select.any():continue
        entry={}
        for target,sl in [('fired',slice(0,64)),('hits',slice(64,128))]:
            y=labels[select,sl];s=scores[select,sl];supported=support[sl]
            auc=episode_auc(s,y);pred=s>=threshold[sl]
            tp=int((pred&y).sum());fp=int((pred&~y).sum());positive=int(y.sum());negative=int((~y).sum())
            m=dict(episode_auroc=mean_ci(auc,modes[select],repeats),micro_auroc=original.auc(s,y),
                positive_cells=positive,negative_cells=negative,episodes=int(select.sum()),
                train_supported_cells=int(supported.sum()),
                supported_micro_auroc=original.auc(s[:,supported],y[:,supported]) if supported.any() else None,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/positive if positive else None,
                false_positive_rate=fp/negative if negative else None)
            if comparison is not None:
                delta=auc-episode_auc(comparison[select,sl],y)
                m['paired_episode_auroc_gain_over_shuffle']=mean_ci(delta,modes[select],repeats)
            entry[target]=m
        result[name]=entry
    return result


def make_fitter(input_dim,nonlinear,settings):
    """Compile once per model shape;horizons/data/weights are runtime arguments."""
    import jax
    import jax.numpy as jnp
    import optax
    optimizer=optax.adam(settings['probe_lr'])
    def forward(params,x):
        for w,b in params[:-1]:x=jax.nn.relu(x@w+b)
        return x@params[-1][0]+params[-1][1]
    def loss(params,x,y,wp,wn):
        bce=optax.sigmoid_binary_cross_entropy(forward(params,x),y)
        return jnp.mean(bce*jnp.where(y>0,wp,wn))
    @jax.jit
    def chunk(params,opt,key,x,y,wp,wn):
        def body(c,_):
            p,o,k=c;k,sk=jax.random.split(k)
            rows=jax.random.randint(sk,(settings['probe_batch'],),0,x.shape[0])
            _,g=jax.value_and_grad(loss)(p,x[rows],y[rows],wp,wn)
            updates,o=optimizer.update(g,o,p)
            return (optax.apply_updates(p,updates),o,k),None
        return jax.lax.scan(body,(params,opt,key),None,length=settings['probe_chunk'])[0]
    measure=jax.jit(loss);predict=jax.jit(lambda p,x:jax.nn.sigmoid(forward(p,x)))
    def fit(x,y,vx,vy,seed):
        key=jax.random.PRNGKey(seed);k1,k2=jax.random.split(key)
        layer=lambda k,a,b:(jax.random.normal(k,(a,b))*jnp.sqrt(2./a),jnp.zeros(b))
        width=settings['probe_width']
        p=(layer(k1,input_dim,width),layer(k2,width,128)) if nonlinear else (layer(k1,input_dim,128),)
        o=optimizer.init(p);positive=np.asarray(y).mean(0);supported=(positive>0)&(positive<1)
        wp=jnp.asarray(np.where(supported,.5/np.maximum(positive,1e-4),1.))
        wn=jnp.asarray(np.where(supported,.5/np.maximum(1-positive,1e-4),1.))
        best=None;best_loss=float('inf');curve=[]
        x,y,vx,vy=map(jnp.asarray,(x,y,vx,vy))
        for update in range(0,settings['probe_updates'],settings['probe_chunk']):
            p,o,key=chunk(p,o,key,x,y,wp,wn);value=float(measure(p,vx,vy,wp,wn))
            if not np.isfinite(value):raise FloatingPointError('Nonfinite probe validation loss')
            curve.append(dict(updates=update+settings['probe_chunk'],validation_loss=value))
            if value<best_loss:best_loss=value;best=jax.device_get(p)
        if not all(np.isfinite(a).all() for a in jax.tree.leaves(best)):raise FloatingPointError('Nonfinite probe parameters')
        return best,curve,supported
    return fit,lambda p,x:np.asarray(predict(p,x))


def prepare(root):
    parent=json.loads((PARENT/'plan.json').read_text())
    if json.loads((PARENT/'status.json').read_text())['state']!='complete':raise ValueError('Completed parent audit required')
    if file_hash(PARENT/'plan.json')!=json.loads((PARENT/'plan-integrity.json').read_text())['sha256']:raise ValueError('Parent plan changed')
    protected={}
    for name in ('plan.json','plan-integrity.json','status.json','validation.json','final-summary.json'):
        path=PARENT/name;protected[str(path)]=file_hash(path)
    cases=[]
    for beta in BETAS:
        folder=PARENT/f'beta{beta}/seed42-step1000000';r=json.loads((folder/'result.json').read_text());c=r['certificates']
        if r['job']['beta']!=beta or r['job']['seed']!=42 or r['job']['step']!=1000000 or c['state_digest_before']!=c['state_digest_after'] or not c['episode_split_disjoint']:
            raise ValueError('Parent identity/state/split certificate failed')
        paths=[folder/'result.json',*[folder/f'{mode}-block{b}.npz' for mode in MODES for b in range(9)]]
        protected.update({str(path):file_hash(path) for path in paths});cases.append(dict(beta=beta,folder=str(folder)))
    p=dict(created_at=datetime.now().astimezone().isoformat(),cases=cases,ns=list(range(1,65)),tasks=list(TASKS),models=list(MODELS),
        settings=SETTINGS,script_sha256=file_hash(__file__),dependencies={'popgym_flash_memory_probe.py':file_hash(original.__file__)},
        versions={name:importlib.metadata.version(name) for name in ('jax','numpy','optax')},
        protected_inputs=protected,scope='ONLY saved1M beta0/1/10 seed42 NPZ. NO checkpoint/pickle loads/new rollouts/agent changes. Same post64-shot memory for everyN/k,exclude episodes ending earlier,retain exact episode splits. 64 independent lastN-window probes AND64 independent exact-lag probes,each outputs64 fired and64 successful-hit positions;order/multiplicity not reconstructed. Most recent=lag1,last64=full observed history,NOT unseen ships. Hits are successful hit outcomes WITHIN requested window/lag,not older known hits refreshed by repeat misses. Separate linear512→128/MLP512→128→128 memory probes and recent-input MLP;constant per-task capacity,2000 Adam1e-3 updates,batch512,train-only normalization/class weighting,validation-only selection/thresholds. Recent input baseline observes last-action/outcome honestly;age64 constant,no clock advantage. Train-only prevalence and within-mode cross-episode shuffled-memory controls. Native independent supervised fits,not categorical RL vmap;no agent precision/target changes. No warmstarts/budget escalation/test selection. Save all labels/episode IDs/scores/models/curves/thresholds. Per-episode AUROC with episode bootstrap200 95%intervals;microAUROC/precision/recall/support counts too. Single-class boards undefined,never zero;absent training cell support disclosed. No declared hard horizon from a single crossing or multiplicity-adjusted significance claim. Different mixed-policy probe-training distributions remain a limitation;one seed only. Host cap8G,GPU2 shared,no training pause;disk<2GiB stops,never deletes archives.')
    root.mkdir(parents=True,exist_ok=False);write_json(root/'plan.json',p);write_json(root/'plan-integrity.json',{'sha256':file_hash(root/'plan.json')})
    (root/'source.py').write_bytes(Path(__file__).read_bytes());write_json(root/'status.json',dict(state='prepared',completed=0,total=3))
    return p


def guard(root,full=True):
    p=json.loads((root/'plan.json').read_text())
    if file_hash(__file__)!=p['script_sha256'] or file_hash(root/'plan.json')!=json.loads((root/'plan-integrity.json').read_text())['sha256']:raise ValueError('Source/plan changed')
    if [c['beta'] for c in p['cases']]!=list(BETAS) or p['ns']!=list(range(1,65)) or p['tasks']!=list(TASKS) or p['models']!=list(MODELS) or p['settings']!=SETTINGS:raise ValueError('Unauthorized scope/settings')
    if file_hash(original.__file__)!=p['dependencies']['popgym_flash_memory_probe.py']:raise ValueError('Dependency changed')
    if any(importlib.metadata.version(k)!=v for k,v in p['versions'].items()):raise ValueError('Probe packages changed')
    if full:
        for name,digest in p['protected_inputs'].items():
            if file_hash(Path(name))!=digest:raise ValueError('Saved input changed')
    return p


def run_one(data,task,n,fitters,settings,folder):
    folder.mkdir(parents=True,exist_ok=False)
    train,val,test=(data['split']==i for i in range(3));y=targets(data['actions'],data['hits'],task,n)
    modes=data['mode'][test];rng=np.random.default_rng(77201);shuffle=np.arange(test.sum())
    for mode in range(3):
        rows=np.flatnonzero(modes==mode);shuffle[rows]=rng.permutation(rows)
    result=dict(task=task,n=n,models={})
    for model in MODELS:
        fit,predict,x,mean,scale=fitters[model]
        seed=73100+n+(1000 if task=='lag' else 0)+(100 if model=='memory-mlp' else 200 if model=='recent-input-mlp' else 0)
        params,curve,support=fit(x[train],y[train],x[val],y[val],seed)
        threshold=original.thresholds(predict(params,x[val]),y[val])
        score=predict(params,x[test]);other=predict(params,x[test][shuffle]) if model.startswith('memory-') else None
        m=metrics(score,y[test],threshold,support,modes,settings['bootstrap_replicates'],other)
        if other is not None:m['shuffled']=metrics(other,y[test],threshold,support,modes,settings['bootstrap_replicates'])
        result['models'][model]=dict(metrics=m,validation_curve=curve,train_positive=y[train].sum(0).tolist(),
            validation_positive=y[val].sum(0).tolist(),validation_threshold_identified=((y[val].sum(0)>0)&(y[val].sum(0)<val.sum())).tolist())
        saved={f'w{i}':w for i,(w,b) in enumerate(params)};saved.update({f'b{i}':b for i,(w,b) in enumerate(params)})
        np.savez_compressed(folder/f'{model}-parameters.npz',**saved,input_mean=mean,input_scale=scale,threshold=threshold)
        payload=dict(score=score,labels=y[test],episode_id=data['episode_id'][test],mode=modes,threshold=threshold,train_supported=support)
        if other is not None:payload['shuffled_score']=other
        np.savez_compressed(folder/f'{model}-heldout.npz',**payload)
    # Per-mode train-only cell-prevalence baseline,not test-fit.
    prevalence=np.empty((test.sum(),128),np.float32)
    for mode in range(3):
        source=train&(data['mode']==mode)
        if not source.any():raise ValueError('Missing mode training support')
        prevalence[modes==mode]=y[source].mean(0)
    support=(y[train].sum(0)>0)&(y[train].sum(0)<train.sum())
    result['models']['prevalence']=dict(metrics=metrics(prevalence,y[test],np.full(128,.5),support,modes,settings['bootstrap_replicates']))
    write_json(folder/'result.json',result)
    return result


def worker(root,index):
    p=guard(root);case=p['cases'][index];beta=case['beta'];folder=root/f'beta{beta}';folder.mkdir(exist_ok=False)
    data,excluded=load_dataset(Path(case['folder']));np.savez_compressed(folder/'dataset.npz',**data)
    fitters={}
    for model in MODELS:
        raw=data['memory'] if model.startswith('memory-') else data['recent'];train=data['split']==0
        mean=raw[train].mean(0);scale=np.maximum(raw[train].std(0),1e-4);x=(raw-mean)/scale
        fit,predict=make_fitter(x.shape[1],model!='memory-linear',p['settings']);fitters[model]=(fit,predict,x,mean,scale)
    results=[]
    for task in TASKS:
        for n in p['ns']:
            original.require_disk();guard(root,full=False)
            r=run_one(data,task,n,fitters,p['settings'],folder/task/f'n{n:02d}');results.append(r)
            write_json(folder/'summary.json',dict(beta=beta,excluded_early_episodes=excluded,
                episodes_by_split={name:int((data['split']==i).sum()) for i,name in enumerate(('train','validation','test'))},results=results))
            print('PROBE',beta,task,n,flush=True)
    guard(root);write_json(folder/'certificate.json',dict(beta=beta,complete=True,all_original_input_hashes_unchanged=True,
        fixed_observation_age=64,episode_split_disjoint=True,tasks=len(results),fits=len(results)*len(MODELS),
        dataset_sha256=file_hash(folder/'dataset.npz')))


def supervise(root):
    with (root/'.supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=guard(root);v=json.loads((root/'validation.json').read_text())
        if not v.get('passed') or v['script_sha256']!=p['script_sha256']:raise ValueError('Tests/full-shape smoke required')
        try:
            for index,case in enumerate(p['cases']):
                write_json(root/'status.json',dict(state='running',completed=index,total=3,beta=case['beta']))
                with (root/f'worker{index}.log').open('x') as log:subprocess.run([sys.executable,__file__,'--root',str(root),'--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,check=True)
            guard(root);write_json(root/'status.json',dict(state='complete',completed=3,total=3,finished_at=datetime.now().astimezone().isoformat()))
        except BaseException:
            import traceback
            write_json(root/'status.json',dict(state='failed',error=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int,choices=(0,1,2))
    args=parser.parse_args();root=args.root.resolve()
    if args.prepare:prepare(root)
    elif args.worker is not None:worker(root,args.worker)
    else:supervise(root)


if __name__=='__main__':main()
