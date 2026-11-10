"""All-80-seed history ablation, with learned-discrepancy interpretation limits."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

HERE=Path(__file__).resolve().parent
HISTORIES=(0,1,2,5,10,15,20,25)


def interval(x):
    x=np.asarray(x);mean=x.mean(axis=-1)
    half=t.ppf(.975,x.shape[-1]-1)*x.std(axis=-1,ddof=1)/np.sqrt(x.shape[-1])
    return mean,half


def describe(x):
    mean,half=interval(x)
    return {'mean':float(mean),'lower':float(mean-half),'upper':float(mean+half)}


def load(plan):
    data={};hashes={}
    for h in HISTORIES:
        pieces=[];seeds=[];steps=None
        for job in sorted((j for j in plan['jobs'] if j['history']==h),key=lambda j:j['block']):
            receipt=json.loads((HERE/'lifecycle'/f"{job['name']}.json").read_text())
            assert receipt['status']=='complete' and receipt['continuous_no_resume'] and receipt['all_checkpoints_archived']
            config=json.loads((HERE/'records'/f"{job['wandb_id']}.config.json").read_text())
            assert config['resume_from'] is None and config['history_length']==h
            path=HERE/'records'/f"{job['wandb_id']}.metrics.jsonl"
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            assert [r['round'] for r in rows if 'round' in r]==list(range(1,51))
            rows=[r for r in rows if 'round' in r and r['env_interactions']<=500000]
            grid=np.array([r['env_interactions'] for r in rows])
            np.testing.assert_array_equal(grid,np.arange(22800,492801,10000))
            if steps is not None:np.testing.assert_array_equal(grid,steps)
            steps=grid
            metrics={'return':'eval/greedy/return','discrepancy_rms':'eval/sampled/discrepancy_rms',
                'greedy_discrepancy_rms':'eval/greedy/discrepancy_rms',
                'aux_twin_rms':'eval/sampled/aux_twin_rms','replay_discrepancy_rms':'train/ld_loss'}
            values={name:np.array([[r[f'{key}/seed_{i}'] for i in range(5)] for r in rows])
                    for name,key in metrics.items()}
            values['replay_discrepancy_rms']=np.sqrt(values['replay_discrepancy_rms'])
            assert all(np.isfinite(v).all() for v in values.values())
            pieces.append(values);seeds+=job['training_seeds']
        assert len(seeds)==len(set(seeds))==10
        data[h]={'steps':steps,'seeds':seeds,**{name:np.concatenate([p[name] for p in pieces],axis=1) for name in pieces[0]}}
    assert all(d['seeds']==data[0]['seeds'] for d in data.values())
    return data,hashes


def main():
    plan=json.loads((HERE/'protocol.json').read_text());statuses={}
    for job in plan['jobs']:
        path=HERE/'lifecycle'/f"{job['name']}.json"
        statuses[job['name']]=json.loads(path.read_text())['status'] if path.exists() else 'queued'
    output=HERE/'results';output.mkdir(exist_ok=True)
    progress={'jobs':statuses,'complete':sum(v=='complete' for v in statuses.values()),'required':16,
              'no_survivor_only_estimates':True}
    (output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    if progress['complete']!=16:print(json.dumps(progress,indent=2));return
    data,hashes=load(plan)
    result={'all_predeclared_jobs_complete':True,'n_seeds_per_history':10,
        'confidence':'pointwise 95% Student-t, across independent training seeds',
        'historical_results_pooled':False,'primary_endpoint_interactions':492800,
        'history_definition':plan['history_definition'],'mask_caveat':plan['mask_caveat'],
        'primary_discrepancy':plan['primary_discrepancy'],'interpretation_limits':plan['interpretation_limits'],
        'input_sha256':hashes,'histories':{},'exploratory_adjacent_contrasts':{},
        'contrasts_caveat':'Paired nominal seed labels; no multiplicity correction. No equivalence or monotonicity guarantee.'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    table=[]
    metrics=('return','discrepancy_rms','greedy_discrepancy_rms','aux_twin_rms','replay_discrepancy_rms')
    for h,d in data.items():
        entry={'seeds':d['seeds'],'steps':d['steps'].tolist(),'metrics':{}}
        for metric in metrics:
            mean,half=interval(d[metric])
            entry['metrics'][metric]={'mean':mean.tolist(),'ci_halfwidth':half.tolist(),
                'endpoint':describe(d[metric][-1]),'last5_seed_average':describe(d[metric][-5:].mean(0))}
        result['histories'][str(h)]=entry
        for i,seed in enumerate(d['seeds']):
            for k,step in enumerate(d['steps']):table.append([h,seed,int(step),*[float(d[m][k,i]) for m in metrics]])
        for col,metric in enumerate(('return','discrepancy_rms')):
            mean,half=interval(d[metric]);x=d['steps']/1e6
            axes[0,col].plot(x,mean,label=f'H={h}',linewidth=1)
            axes[0,col].fill_between(x,mean-half,mean+half,alpha=.08)
    for col,metric in enumerate(('return','discrepancy_rms')):
        mean=[];half=[]
        for h in HISTORIES:
            m,hw=interval(data[h][metric][-1]);mean.append(m);half.append(hw)
        axes[1,col].errorbar(HISTORIES,mean,yerr=half,marker='o',capsize=3)
        axes[1,col].set_xlabel('Previous network inputs retained (H)')
        axes[0,col].set_xlabel('Interactions including prefill ($10^6$)')
        axes[0,col].legend(frameon=False,ncol=2,fontsize=8)
    for row in range(2):
        axes[row,0].set_ylabel('Mean greedy return')
        axes[row,1].set_ylabel('Sampled-trajectory learned RMS discrepancy')
    for lo,hi in zip(HISTORIES[:-1],HISTORIES[1:]):
        result['exploratory_adjacent_contrasts'][f'{hi}-{lo}']={m:describe(data[hi][m][-1]-data[lo][m][-1])
                                                              for m in ('return','discrepancy_rms')}
    fig.tight_layout();fig.savefig(output/'history-length-return-and-discrepancy.png',dpi=180);plt.close(fig)
    (output/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'seed_metrics.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['history','training_seed','env_interactions',*metrics]);writer.writerows(table)
    print(output)


if __name__=='__main__':main()
