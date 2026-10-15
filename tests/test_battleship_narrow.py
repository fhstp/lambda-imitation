import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('narrow',ROOT/'scripts/narrow_battleship_sweep.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_selection_excludes_failed_partial_wrong_budget_and_nan():
    config=dict(max_steps=5000000,n_seeds=3)
    rows=[SimpleNamespace(id='ok',state='finished',config=config,summary={'objective/return_mean':.2}),
        SimpleNamespace(id='failed',state='failed',config=config,summary={'objective/return_mean':1.}),
        SimpleNamespace(id='running',state='running',config=config,summary={'objective/return_mean':1.}),
        SimpleNamespace(id='wrong-budget',state='finished',config={**config,'max_steps':1000000},summary={'objective/return_mean':1.}),
        SimpleNamespace(id='nan',state='finished',config=config,summary={'objective/return_mean':float('nan')})]
    assert [r[1] for r in helper.completed_rows(rows)]==['ok']


def test_narrow_uses_new_seeds_and_separate_budget_without_mutating_base():
    base=yaml.safe_load((ROOT/'sweeps/battleship_bayes.yaml').read_text())
    values=dict(lambda_coef=10.,fe_lr=1e-4,actor_lr=1e-4,critic_lr=1e-4,alpha_final=.01,tau=.005,
        memory_hidden_dim=512,alpha_decay_steps=500000)
    result=helper.narrow(base,[(.2,str(i),values) for i in range(3)],10)
    p=result['parameters'];assert p['n_seeds']['value']==10 and p['seed_base']['value']==1000
    assert p['max_steps']['value']==10000000 and base['parameters']['max_steps']['value']==5000000
    assert p['memory_hidden_dim']['values']==[512]
    assert p['lambda_coef']['min']==10/1.5 and p['lambda_coef']['max']==15
    assert base['parameters']['lambda_coef']['min']==.03


def test_insufficient_evidence_rejected():
    with pytest.raises(ValueError):helper.narrow({},[],5)
