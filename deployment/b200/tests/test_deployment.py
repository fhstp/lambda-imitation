"""CPU-only deployment admission. No GPU, tmux, W&B init or learner is launched."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


q=load('b200_deployment_tests_queue',HERE/'worker_queue.py')
local=load('b200_deployment_tests_local',HERE/'local.py')


def test_assignment_is_exact_reserved_second_blocks():
    a=q.assignment()
    assert a['local_dispatch_reserved'] and not a['automatic_expiry']
    assert len(a['jobs'])==8
    assert {j['block'] for j in a['jobs']}=={2}
    assert all(j['training_seeds']==list(range(7451234542,7451234547)) for j in a['jobs'])
    assert len({j['wandb_id'] for j in a['jobs']})==8


def test_persistent_state_cannot_change_assigned_flags_or_ids():
    job={**q.assignment()['jobs'][0],'gpu':cards()[0]}
    q.validate_job(job)
    with pytest.raises(RuntimeError,match='Changed assigned'):q.validate_job({**job,'wandb_id':'replacement'})
    with pytest.raises(RuntimeError,match='Changed assigned'):q.validate_job({**job,'flags':job['flags']+['--rounds','1']})


def test_original_seals_unchanged_and_deployment_seal_valid():
    q.verify(runtime=False)


def test_changed_protocol_rejected_before_any_worker(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'HERE',tmp_path)
    (tmp_path/'assignment.json').write_text(json.dumps({'local_dispatch_reserved':False}))
    with pytest.raises(RuntimeError,match='No issued'):q.assignment()


def cards(name='NVIDIA B200'):
    return [{'index':i,'uuid':f'GPU-{i}','name':name,'memory_total_mib':183000} for i in range(8)]


def test_gpu_selection_order_is_explicit():
    assert [g['index'] for g in q.select_gpus('7,6,5,4,3,2,1,0',cards())]==list(reversed(range(8)))


@pytest.mark.parametrize('indices',('0,1,2','0,1,2,3,4,5,6,6','0,1,2,3,4,5,6,9'))
def test_bad_gpu_allocations_rejected(indices):
    with pytest.raises(RuntimeError):q.select_gpus(indices,cards())


def test_original_cards_cannot_accidentally_launch_b200_assignment():
    with pytest.raises(RuntimeError,match='full B200'):q.select_gpus('0,1,2,3,4,5,6,7',cards('RTX Pro 6000'))


def test_mig_or_small_cards_rejected():
    g=cards();g[-1]['memory_total_mib']=20000
    with pytest.raises(RuntimeError):q.select_gpus('0,1,2,3,4,5,6,7',g)


@pytest.mark.parametrize('index',range(8))
def test_worker_environment_isolated_and_fresh(index,tmp_path,monkeypatch):
    monkeypatch.setenv('WANDB_SWEEP_ID','')
    j={**q.assignment()['jobs'][index],'gpu':cards()[index]}
    env=q.environment(j,{'work':str(tmp_path)})
    assert env['CUDA_VISIBLE_DEVICES']==j['gpu']['uuid']
    assert env['JAX_PLATFORMS']=='cuda' and env['WANDB_RESUME']=='never'
    assert env['PYTHONPATH']==str(q.ROOT/'src') and 'WANDB_SWEEP_ID' not in env
    assert env['WANDB_RUN_ID']==j['wandb_id']
    assert env['WANDB_DIR']==str(tmp_path/'runs')
    assert env['REBUTTAL_SOURCE_MANIFEST']==str(q.RUNS/'source_manifest.json')


def expected_config(job):
    import sys
    sys.path.insert(0,str(q.ROOT/'examples'))
    import battleship
    from _common import parse_args,validate_args,config_dict
    parser=battleship.parser();args=parse_args(parser,job['flags']);validate_args(parser,args)
    return config_dict(args)


@pytest.mark.parametrize('key,value',(('seed',1),('prefill_steps',16128),('batch_size',64),
    ('history_context',0),('eval_episodes',5),('lambda_critic_loss','huber'),('lambda_coef',0),
    ('rounds',1),('resume_from','some-checkpoint.pkl'),('train_steps',5000),('replay_correction','is')))
def test_import_cannot_hide_changed_scientific_parameters(key,value):
    job=q.assignment()['jobs'][0];config=expected_config(job);config[key]=value
    with pytest.raises(RuntimeError,match='scientific configuration'):q.validate_config(config,job)


def test_output_paths_can_change_without_changing_science():
    job=q.assignment()['jobs'][0];config=expected_config(job)
    config.update(output_dir='/new/W&B/files/training',wandb=True,wandb_project='new-host',wandb_run_name='same-job')
    q.validate_config(config,job)


def test_consumed_worker_cannot_restart(tmp_path,monkeypatch):
    job={**q.assignment()['jobs'][0],'gpu':cards()[0]}
    q.write(tmp_path/'settings.json',{'owner_host':'owner','jobs':[job]})
    q.write(tmp_path/'jobs'/job['name']/'status.json',{'status':'failed'})
    monkeypatch.setattr(q,'verify',lambda:None);monkeypatch.setattr(q,'owner',lambda:'owner')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',job['gpu']['uuid'])
    with pytest.raises(RuntimeError,match='Consumed'):q.worker(tmp_path,job['name'])


def test_native_failure_never_starts_production(tmp_path,monkeypatch):
    job={**q.assignment()['jobs'][0],'gpu':cards()[0]}
    q.write(tmp_path/'settings.json',{'owner_host':'owner','jobs':[job],'work':str(tmp_path/'work')})
    monkeypatch.setattr(q,'verify',lambda:None);monkeypatch.setattr(q,'owner',lambda:'owner')
    monkeypatch.setattr(q,'occupied',lambda:set());monkeypatch.setattr(q,'run_logged',lambda *a,**k:1)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',job['gpu']['uuid'])
    with pytest.raises(RuntimeError,match='Native validation failed'):q.worker(tmp_path,job['name'])
    directory=tmp_path/'jobs'/job['name']
    assert not (directory/'production-started.json').exists()
    assert q.read(directory/'status.json')['status']=='failed'


@pytest.mark.parametrize('exit_code',(None,1,-9))
def test_finalize_cannot_resume_or_finish_failed_training(tmp_path,monkeypatch,exit_code):
    monkeypatch.setattr(q,'verify',lambda:None)
    job=q.assignment()['jobs'][0];q.write(tmp_path/'settings.json',{})
    if exit_code is not None:q.write(tmp_path/'jobs'/job['name']/'training-exit.json',{'exit_code':exit_code})
    with pytest.raises(RuntimeError,match='No successful training exit'):q.finalize(tmp_path,job)


def test_successful_finalize_creates_exact_importable_bundle(tmp_path,monkeypatch):
    job={**q.assignment()['jobs'][0],'gpu':cards()[0]}
    state=tmp_path/'state';work=tmp_path/'work'
    config={'work':str(work),'owner_host':'test-owner'};q.write(state/'settings.json',config)
    output=work/'runs/wandb'/f"run-test-{job['wandb_id']}"/'files/training';output.mkdir(parents=True)
    cfg=expected_config(job);q.write(output/'config.json',cfg)
    q.write(output/'summary.json',{'env_interactions':512800})
    (output/'metrics.jsonl').write_text(''.join(json.dumps({'round':r,'env_interactions':12800+r*10000})+'\n'
                                                   for r in range(1,51)))
    for name in ('checkpoint_10.pkl','checkpoint_20.pkl','checkpoint_30.pkl','checkpoint_40.pkl','checkpoint_50.pkl','checkpoint_final.pkl'):
        q.write(output/(name+'.archived.json'),{'remote_committed_verified':True})
    q.write(state/'jobs'/job['name']/'training-exit.json',{'exit_code':0})
    q.write(state/'native'/f"h{job['history']}"/'reviewer_response'/f"native-h{job['history']}.json",{'passed':True})
    monkeypatch.setattr(q,'verify',lambda:None)
    bundle=q.finalize(state,job)
    receipt=local.validate_import(bundle,q.assignment()['jobs'][0],q.assignment())
    assert receipt['continuous_no_resume'] and receipt['all_checkpoints_archived']
    index=q.read(bundle/'index.json')
    assert len(index['files'])==10  # config, metrics, summary, receipt, six checkpoint pointers
    # No learner is invoked and an explicit finalize retry is byte-stable.
    before=q.sha(bundle/'receipt.json');q.finalize(state,job)
    assert q.sha(bundle/'receipt.json')==before


def test_import_rejects_corrupt_bundle_before_copy(tmp_path):
    q.write(tmp_path/'index.json',{'lease_id':q.assignment()['lease_id'],'job':'wrong'})
    with pytest.raises(AssertionError):local.validate_import(tmp_path,q.assignment()['jobs'][0],q.assignment())


def test_status_does_not_restart_lost_worker(tmp_path,monkeypatch):
    job={**q.assignment()['jobs'][0],'gpu':cards()[0]}
    q.write(tmp_path/'settings.json',{'jobs':[job],'work':str(tmp_path/'work')})
    q.write(tmp_path/'jobs'/job['name']/'ticket.json',job)
    monkeypatch.setattr(q,'pane_states',lambda:{})
    result=q.snapshot(tmp_path)
    assert result['jobs'][0]['status']=='worker-lost' and not result['complete']


def test_native_module_is_portable_and_never_overwrites_original_proof(tmp_path):
    program=f'''import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location("portable_native_test",{str(HERE/'worker_queue.py')!r})
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
n=m.load_native_module(Path({str(tmp_path)!r}),25)
assert n.ROOT==Path({str(tmp_path)!r})/"native/h25"
assert n.ROOT!=m.ROOT
'''
    subprocess.run([sys.executable,'-B','-c',program],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})


def test_direct_script_directory_does_not_shadow_stdlib_queue():
    program=f'import sys; sys.path.insert(0,{str(HERE)!r}); from queue import Queue; import wandb; assert Queue'
    subprocess.run([sys.executable,'-B','-c',program],check=True,
                   env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
