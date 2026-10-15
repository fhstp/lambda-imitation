"""Day5 helper:derive a NEW5/10-seed,10M Bayesian sweep from completed5M trials.

Reads W&B via outbound HTTPS. Does NOT launch anything/change the old sweep.
Inspect generated ranges before registering them. New seed base1000 reduces
reuse of the three training seeds on which hyperparameters were selected.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path


def completed_rows(runs):
    rows=[]
    for run in runs:
        score=run.summary.get('objective/return_mean')
        if isinstance(score,dict):score=score.get('last')
        if run.state=='finished' and run.config.get('max_steps')==5000000 and run.config.get('n_seeds')==3 and isinstance(score,(int,float)):
            import math
            if math.isfinite(score):rows.append((float(score),run.id,dict(run.config)))
    return sorted(rows,key=lambda r:r[0],reverse=True)


def narrow(base,rows,n_seeds):
    if n_seeds not in (5,10) or len(rows)<3:raise ValueError('Need at least3 completed equal-budget trials and5 or10 confirmation seeds')
    import copy
    result=copy.deepcopy(base);result['name']=f'battleship-narrow-{n_seeds}seed-10M';result['run_cap']=128
    params=result['parameters'];params['n_seeds']={'value':n_seeds};params['seed_base']={'value':1000};params['max_steps']={'value':10000000}
    for key,spec in list(params.items()):
        if spec.get('distribution')=='log_uniform_values':
            values=[r[2][key] for r in rows]
            lo=max(spec['min'],min(values)/1.5);hi=min(spec['max'],max(values)*1.5)
            params[key]={'distribution':'log_uniform_values','min':lo,'max':hi} if hi>lo else {'value':lo}
        elif 'values' in spec:
            params[key]={'values':sorted({r[2][key] for r in rows})}
    return result


def main():
    import wandb
    import yaml
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('sweep',help='entity/project/sweep_id')
    parser.add_argument('--top',type=int,default=12);parser.add_argument('--n-seeds',type=int,choices=(5,10),default=5)
    parser.add_argument('--base',type=Path,default=Path('sweeps/battleship_bayes.yaml'))
    parser.add_argument('--output',type=Path,default=Path('sweeps/battleship_narrow.yaml'));args=parser.parse_args()
    if args.top<3:parser.error('--top must be at least3')
    rows=completed_rows(wandb.Api().sweep(args.sweep).runs)[:args.top]
    cfg=narrow(yaml.safe_load(args.base.read_text()),rows,args.n_seeds)
    with args.output.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    selection=dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_sweep=args.sweep,selection=[dict(score=s,run_id=i,config=c) for s,i,c in rows],
        note='Selection-biased candidate set;new seed base1000,10M budget,separate sweep. Inspect ranges;not an automatically authorized old-study extension.')
    with args.output.with_suffix('.selection.json').open('x') as f:json.dump(selection,f,indent=2)
    print('Review before launch:',args.output)


if __name__=='__main__':main()
