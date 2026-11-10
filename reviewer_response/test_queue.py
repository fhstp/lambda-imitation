"""Queue isolation, reservations, no restart, and all-seed analysis gates."""

import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('history_queue_tests',HERE/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
spec=importlib.util.spec_from_file_location('history_report_tests',HERE/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_control_gpus_reserved_until_both_blocks_verified(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'CONTROLS',tmp_path)
    jobs=[{'name':'b1','gpu_uuid':'GPU-0'},{'name':'b2','gpu_uuid':'GPU-0'}]
    (tmp_path/'protocol.json').write_text(json.dumps({'jobs':jobs}));(tmp_path/'lifecycle').mkdir()
    (tmp_path/'lifecycle/b1.json').write_text(json.dumps({'status':'complete','all_checkpoints_committed_verified':True}))
    assert launch.controls_reserved()=={'GPU-0'}
    (tmp_path/'lifecycle/b2.json').write_text(json.dumps({'status':'complete'}))
    assert launch.controls_reserved()=={'GPU-0'}
    (tmp_path/'lifecycle/b2.json').write_text(json.dumps({'status':'complete','all_checkpoints_committed_verified':True}))
    assert not launch.controls_reserved()


@pytest.mark.parametrize('h',launch.HISTORIES)
def test_flags_preserve_paper_budget_and_count(h):
    flags=launch.flags('battleship','retrace',7451234536,5)
    assert flags[flags.index('--rounds')+1]=='50'
    assert flags[flags.index('--train-steps')+1]=='10000'
    assert flags[flags.index('--prefill-steps')+1]=='12800'
    assert flags[flags.index('--burn-in-length')+1]=='10'
    assert flags[flags.index('--sequence-length')+1]=='40'
    assert flags[flags.index('--eval-episodes')+1]=='50'
    assert '--resume-from' not in flags


def test_environment_has_only_assigned_gpu_new_source_and_no_sweep(monkeypatch):
    monkeypatch.setenv('WANDB_SWEEP_ID','inherited-invalid')
    env=launch.environment({'gpu_uuid':'GPU-only-one','wandb_id':'test','history':25})
    assert env['CUDA_VISIBLE_DEVICES']=='GPU-only-one'
    assert env['PYTHONPATH']==str(launch.ROOT/'src')
    assert env['WANDB_RESUME']=='never' and 'WANDB_SWEEP_ID' not in env
    assert env['WANDB_DIR']==str(launch.RAM/'runs')


def test_consumed_attempt_refused_before_any_training(tmp_path,monkeypatch):
    monkeypatch.setattr(launch,'HERE',tmp_path);monkeypatch.setattr(launch,'verify',lambda:None)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-one');(tmp_path/'lifecycle').mkdir()
    (tmp_path/'lifecycle/used.ticket.json').write_text(json.dumps({'gpu_uuid':'GPU-one'}))
    (tmp_path/'lifecycle/used.json').write_text('{}')
    with pytest.raises(AssertionError,match='Never restart'):launch.run_job('used')


def test_no_partial_seed_estimates(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'HERE',tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps({'jobs':[{'name':str(i)} for i in range(16)]}))
    report.main()
    assert not (tmp_path/'results/statistics.json').exists()
    assert json.loads((tmp_path/'results/progress.json').read_text())['complete']==0
