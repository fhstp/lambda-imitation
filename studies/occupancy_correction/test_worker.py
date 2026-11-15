"""Exercise the actual worker's three paths without W&B or production budgets."""

from pathlib import Path

import pytest

from . import run_worker
from .learner import paper_settings
from .records import load_owned_pickle, read, sha, write


@pytest.mark.parametrize("environment", ["battleship", "minesweeper"])
def test_actual_learning_data_and_frozen_fit_worker_paths(tmp_path, monkeypatch, environment):
    monkeypatch.setattr(run_worker,"paper_settings",lambda env,**kw: paper_settings(env,
        hidden_dim=4,batch_size=2,sequence_length=2,tail=2,replay_capacity=300,**kw))
    admission={"source_sha256":{}}
    write(tmp_path/"admission.json",admission)
    budget={"prefill":160,"rounds":2,"steps_per_round":2,"eval_episodes":2,
            "eval_initial":environment=="minesweeper","reference_round":1,"primary_endpoint":162}
    study={"learning":{environment:budget},"frozen":{"data_calls_per_seed":250,"heldout_calls_per_seed":100,
            "updates":4,"eval_every_updates":2},"disk_reserve_bytes":0}
    common={"environment":environment,"seeds":[501,502],"measure":"uniform","correction":"retrace"}
    learning={**common,"kind":"learning","name":"learning","wandb_id":"test-learning"}
    directory=tmp_path/"jobs/learning";directory.mkdir(parents=True)
    summary=run_worker.execute_job(tmp_path,learning,directory,study,admission,None)
    assert summary["env_interactions"]==164
    assert read(directory/"training-exit.json")["exit_code"]==0
    assert read(directory/"status.json")["status"]=="complete"
    record=read(directory/"reference.json")
    reference=load_owned_pickle(directory/record["name"],record["sha256"])
    assert reference["physical_calls"]==162
    with pytest.raises(RuntimeError):load_owned_pickle(directory/record["name"],"wrong-digest")
    data_job={**common,"kind":"data","name":"data","wandb_id":"test-data","reference_job":"learning"}
    data_dir=tmp_path/"jobs/data";data_dir.mkdir()
    run_worker.execute_job(tmp_path,data_job,data_dir,study,admission,None)
    record=read(data_dir/"datasets.json")
    data=load_owned_pickle(data_dir/record["name"],record["sha256"])
    assert data["collector"]=="fixed-uniform"
    assert run_worker.digest_tree(data["fitting"])!=run_worker.digest_tree(data["heldout"])
    fit_job={**common,"kind":"fit","name":"fit","wandb_id":"test-fit","data_job":"data",
             "measure":"discounted","correction":"is"}
    fit_dir=tmp_path/"jobs/fit";fit_dir.mkdir()
    summary=run_worker.execute_job(tmp_path,fit_job,fit_dir,study,admission,None)
    assert summary["updates"]==4
    assert summary["frozen_reference_sha256"]==reference_record_sha(directory)
    assert read(fit_dir/"training-exit.json")["exit_code"]==0
    assert read(fit_dir/"status.json")["status"]=="complete"
    # Existing metrics are a consumed attempt, not an invitation to resume it.
    with pytest.raises(FileExistsError):
        run_worker.execute_job(tmp_path,fit_job,fit_dir,study,admission,None)


def reference_record_sha(directory):return read(directory/"reference.json")["sha256"]
