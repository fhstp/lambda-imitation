"""Cheap real orchestration/atomic-resume test with a toy vmapped learner.

The native network gate is separately exercised by --cpu-smoke/remote preflight;
this test does NOT claim toy dynamics validate the real learning algorithm.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import battleship_sweep as sweep


class State(NamedTuple):
    update_step:jax.Array


def test_trial_final_equal_budget_score_and_exact_resume(tmp_path,monkeypatch):
    import battleship_memory
    import popgym_jax
    import wandb
    calls=dict(prefill=0,analysis=0,finish=0);logs=[]
    class Config(dict):
        def update(self,*args,**kwargs):return super().update(*args)
    class Artifact:
        def __init__(self,*a,**k):pass
        def add_file(self,*a,**k):pass
    class Run:
        id='toy-trial'
        def __init__(self,values):self.config=Config(values)
        def define_metric(self,*a,**k):pass
        def log(self,value):logs.append(value)
        def log_artifact(self,*a,**k):pass
        def finish(self,*a,**k):calls['finish']+=1
    monkeypatch.setattr(wandb,'init',lambda config,**kw:Run(config))
    monkeypatch.setattr(wandb,'Artifact',Artifact)
    class Runner:
        def train(self,s,r,n_steps):return State(s.update_step+n_steps),r,dict(loss=jnp.float32(1))
        def prefill(self,s,r,n_steps):
            calls['prefill']+=1
            return s,r,jnp.asarray(n_steps)
        def evaluate(self,*a):return 0
    def build(c,*a,**k):
        states=State(jnp.zeros(c['n_seeds'],jnp.int32))
        rollouts=(jnp.zeros((c['n_seeds'],2),jnp.uint32),jnp.zeros(c['n_seeds']),jnp.zeros(c['n_seeds'],bool),jnp.zeros(c['n_seeds'],int))
        return states,rollouts,Runner(),(None,)*8
    def validate(c,folder,**kw):
        sweep.atomic_json(folder/'validation.json',dict(passed=True,test_only=True));return {}
    monkeypatch.setattr(sweep,'build',build);monkeypatch.setattr(sweep,'validate',validate)
    monkeypatch.setattr(popgym_jax,'evaluation_result',lambda *a:dict(metrics=dict(return_mean=.25)))
    def analyze(*a,**k):calls['analysis']+=1;return {}
    monkeypatch.setattr(battleship_memory,'analyze',analyze);monkeypatch.setattr(battleship_memory,'log_memory',lambda *a,**k:{})
    values=dict(max_steps=10000,min_free_gib=.001,memory_schedule='milestones',memory_milestones=[10000])
    sweep.run_trial(values,tmp_path)
    status=json.loads((tmp_path/'toy-trial/status.json').read_text())
    assert status==dict(state='complete',steps=10000)
    assert calls['analysis']==3 and calls['prefill']==1
    scores=[r['objective/return_mean'] for r in logs if 'objective/return_mean' in r]
    assert scores==[.25]
    checkpoint=tmp_path/'toy-trial/latest.pkl';assert checkpoint.exists()
    # Resume a completed test snapshot must NOT replay prefill/diagnostics/steps.
    sweep.run_trial({},tmp_path,resume=checkpoint)
    assert calls['prefill']==1 and calls['analysis']==3 and calls['finish']==2
