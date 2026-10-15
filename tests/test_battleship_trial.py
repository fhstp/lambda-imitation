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
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import battleship_sweep as sweep


class State(NamedTuple):
    update_step:jax.Array


@pytest.mark.parametrize('issue',['none','parity','fit','publication','media_log','integrity','mutation','mutating_fit','return_eval','unexpected'])
def test_trial_final_equal_budget_score_and_exact_resume(tmp_path,monkeypatch,issue):
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
        def log(self,value):
            if issue=='media_log' and 'memory/test_media' in value:raise ValueError('Optional W&B media log failed')
            logs.append(value)
        def log_artifact(self,*a,**k):pass
        def finish(self,*a,**k):calls['finish']+=1
    monkeypatch.setattr(wandb,'init',lambda config,**kw:Run(config))
    monkeypatch.setattr(wandb,'Artifact',Artifact)
    class Runner:
        def train(self,s,r,n_steps):return State(s.update_step+n_steps),r,dict(loss=jnp.float32(1))
        def prefill(self,s,r,n_steps):
            calls['prefill']+=1
            return s,r,jnp.asarray(n_steps)
        def evaluate(self,*a):
            if issue=='return_eval':raise ValueError('Final-return evaluation failed')
            return 0
    def build(c,*a,**k):
        states=State(jnp.zeros(c['n_seeds'],jnp.int32))
        rollouts=(jnp.zeros((c['n_seeds'],2),jnp.uint32),jnp.zeros(c['n_seeds']),jnp.zeros(c['n_seeds'],bool),jnp.zeros(c['n_seeds'],int))
        return states,rollouts,Runner(),(None,)*8
    def validate(c,folder,**kw):
        sweep.atomic_json(folder/'validation.json',dict(passed=True,test_only=True));return {}
    monkeypatch.setattr(sweep,'build',build);monkeypatch.setattr(sweep,'validate',validate)
    monkeypatch.setattr(popgym_jax,'evaluation_result',lambda *a:dict(metrics=dict(return_mean=.25)))
    def analyze(*a,**k):
        calls['analysis']+=1
        if issue=='parity':k['on_warning']('policy_memory_parity','Tolerance exceeded',dict(mismatched_elements=1,elements=409600,max_probability_error=1.2638e-5))
        if issue in ('fit','mutating_fit'):raise battleship_memory.DiagnosticUnavailable('probe_fit',ValueError('Fit failed'))
        if issue=='integrity':raise AssertionError('Corrupt labels')
        if issue=='unexpected':raise ValueError('Unexpected collector bug')
        return {}
    def log_memory(*a,**k):
        if issue=='publication':raise ValueError('Optional media formatting failed')
        return {'memory/test_media':1}
    monkeypatch.setattr(battleship_memory,'analyze',analyze);monkeypatch.setattr(battleship_memory,'log_memory',log_memory)
    if issue in ('mutation','mutating_fit'):
        import popgym_memory_diagnostics
        monkeypatch.setattr(popgym_memory_diagnostics,'tree_digest',lambda state:'before' if calls['analysis']==0 else 'changed')
    values=dict(max_steps=10000,min_free_gib=.001,memory_schedule='milestones',memory_milestones=[10000])
    if issue in ('integrity','mutation','mutating_fit','return_eval','unexpected'):
        with pytest.raises((AssertionError,ValueError)):
            sweep.run_trial(values,tmp_path)
        assert not any('objective/return_mean' in r for r in logs)
        assert json.loads((tmp_path/'toy-trial/status.json').read_text())['state']=='failed'
        return
    sweep.run_trial(values,tmp_path)
    status=json.loads((tmp_path/'toy-trial/status.json').read_text())
    assert status==dict(state='complete',steps=10000)
    expected_analysis=1 if issue=='fit' else 3
    assert calls['analysis']==expected_analysis and calls['prefill']==1
    diagnostic=[r for r in logs if 'diagnostics/memory_complete' in r]
    assert diagnostic[-1]['diagnostics/memory_complete']==int(issue not in ('fit','publication','media_log'))
    warnings=[json.loads(r['diagnostics/warning_json']) for r in logs if 'diagnostics/warning_json' in r]
    assert len(warnings)==(3 if issue=='parity' else 1 if issue in ('fit','publication','media_log') else 0)
    if warnings:
        assert all(r['steps']==10000 for r in warnings)
        assert len((tmp_path/'toy-trial/diagnostic-warnings.jsonl').read_text().splitlines())==len(warnings)
    if issue in ('fit','publication','media_log'):
        assert 'ValueError' in (tmp_path/'toy-trial/diagnostic-errors.log').read_text()
    scores=[r['objective/return_mean'] for r in logs if 'objective/return_mean' in r]
    assert scores==[.25]
    checkpoint=tmp_path/'toy-trial/latest.pkl';assert checkpoint.exists()
    # Resume a completed test snapshot must NOT replay prefill/diagnostics/steps.
    sweep.run_trial({},tmp_path,resume=checkpoint)
    assert calls['prefill']==1 and calls['analysis']==expected_analysis and calls['finish']==2
