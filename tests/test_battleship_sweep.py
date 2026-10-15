import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
import battleship_sweep as sweep


def test_exact_diagnostic_schedule():
    steps=sweep.memory_steps(10000000)
    assert len(steps)==32 and steps[:10]==list(range(10000,100001,10000))
    assert steps[10:19]==list(range(200000,1000001,100000))
    assert steps[19:27]==list(range(1500000,5000001,500000))
    assert steps[27:]==list(range(6000000,10000001,1000000))
    assert len(sweep.memory_steps(5000000))==27


def test_5m_search_and_configurable_seed_count():
    assert sweep.resolve_config({})['max_steps']==5000000
    for n in (3,5,10):assert sweep.resolve_config({'n_seeds':n})['n_seeds']==n


@pytest.mark.parametrize('values',[
    {'n_seeds':0},{'n_seeds':3.5},{'lambda_coef':-1},{'memory_suite':'disabled'},
    {'max_steps':11000000},{'max_steps':5000001},{'fe_lr':float('nan')},
    {'probe_updates':201},{'probe_blocks':7},{'tau':2},{'lambda2':0},
    {'memory_milestones':[15]},{'seed_base':-1},{'unknown':1},{'chunk_steps':17}])
def test_fail_closed_config(values):
    with pytest.raises(ValueError):sweep.resolve_config(values)


def test_runtime_policy_can_only_change_observation_schedule(tmp_path):
    path=tmp_path/'policy.json';c=sweep.resolve_config({'runtime_policy':str(path)})
    assert sweep.read_policy(c)['memory_schedule']=='dense'
    path.write_text(json.dumps(dict(memory_schedule='milestones',memory_suite='tiered',memory_milestones=[100000,1000000])))
    policy=sweep.read_policy(c)
    assert not sweep.scheduled(10000,c,policy) and sweep.scheduled(100000,c,policy)
    assert sweep.scheduled(5000000,c,policy),'Final diagnostics always run'
    path.write_text('{"fe_lr":0.1}')
    with pytest.raises(ValueError,match='diagnostics ONLY'):sweep.read_policy(c)


def test_seed_mean_not_pooled_episodes_or_best_seed():
    result=sweep.seed_aggregate([{'return':1.},{'return':2.},{'return':3.}],[42,43,44],'eval')
    assert result['eval/return/mean']==2 and result['eval/return/std']==1
    assert result['eval/return/n_seeds']==3 and result['seed/43/eval/return']==2


def test_no_missing_value_zero_fill():
    result=sweep.seed_aggregate([{'auroc':.8},{},{}],[42,43,44],'memory')
    assert result['memory/auroc/mean']==.8 and result['memory/auroc/n_seeds']==1


def test_exact_native_gate_checks_bytes_not_a_numerical_tolerance():
    sweep.exact(np.array([0.],np.float32),np.array([0.],np.float32))
    with pytest.raises(AssertionError):sweep.exact(np.array([0.],np.float32),np.array([-0.],np.float32))
    with pytest.raises(AssertionError):sweep.exact(np.array([1.],np.float32),np.array([1.],np.float64))


def test_projection_mass_mean_and_legacy_reference():
    import jax
    import jax.numpy as jnp
    import popgym_flash_prototypes as flash
    values=jnp.array([[-2.,-.42,.19,1.8],[.15,.15,.15,.15]])
    p=jnp.array([[.1,.2,.3,.4],[.25,.25,.25,.25]])
    actual=sweep.deterministic_projection(p,values)
    np.testing.assert_allclose(actual.sum(-1),1.,atol=2e-6)
    np.testing.assert_allclose((actual*flash.support()).sum(-1),(p*jnp.clip(values,-1.5,1.5)).sum(-1),atol=3e-7)
    np.testing.assert_allclose(actual,flash.project_distribution(p,values),atol=3e-7)
    fn=jax.jit(jax.vmap(lambda q,v:sweep.deterministic_projection(q[None],v[None])[0]))
    sweep.exact(fn(p,values),fn(p,values))
    np.testing.assert_allclose(jax.grad(lambda q:sweep.deterministic_projection(q,values).sum())(p),1.,atol=1e-6)


def test_alpha_schedule_is_explicit_and_clamped():
    c=sweep.resolve_config({'alpha_initial':1.,'alpha_final':.01,'alpha_decay_steps':500000})
    np.testing.assert_allclose([sweep.alpha_at(s,c) for s in (0,250000,500000,1000000)],[1,.1,.01,.01],rtol=1e-6)


def test_actual_seed_vmap_not_python_training_loop():
    import inspect
    source=inspect.getsource(sweep.SeedRunner)
    assert 'jax.vmap' in source and 'runner.train' in source and 'runner.prefill' in source
    assert 'for seed' not in source


def test_portable_builder_never_reads_old_plans(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Host-specific guard/config_for called')
    import popgym_full_episode as old
    monkeypatch.setattr(old.normalization,'guard',forbidden)
    monkeypatch.setattr(old,'config_for',forbidden)
    c=sweep.resolve_config({});args,config=sweep.builder_config(c)
    assert args.collect_steps==config['collect_steps']==16
    assert config['sequence_length']==64 and config['burn_in_length']==0 and config['lambda_truncation']==1
    assert config['lambda_coef']==10 and config['fe_lr']==.0001 and config['residual_critic_width']==256


def test_bayesian_yaml_and_log_value_ranges():
    import yaml
    root=Path(__file__).parents[1];cfg=yaml.safe_load((root/'sweeps/battleship_bayes.yaml').read_text())
    assert cfg['method']=='bayes' and cfg['metric']['name']=='objective/return_mean'
    assert cfg['parameters']['n_seeds']['value']==3 and cfg['parameters']['max_steps']['value']==5000000
    assert 'early_terminate' not in cfg
    for k in ('lambda_coef','fe_lr','actor_lr','critic_lr','alpha_final','tau'):
        p=cfg['parameters'][k];assert p['distribution']=='log_uniform_values' and p['min']>0 and p['max']>p['min']
