"""Fail-closed allocation, no duplicate training, and administrative recovery."""

from collections import Counter
from copy import deepcopy
import shlex
import subprocess
import sys

import pytest

from . import queue
from .records import read, write
from .study import build_study


def test_fresh_allocation_has_all_conditions_and_exact_seed_counts():
    study=build_study()
    jobs=study["jobs"]
    assert Counter(j["kind"] for j in jobs)=={"learning":36,"data":6,"fit":36}
    assert len({j["name"] for j in jobs})==len({j["session"] for j in jobs})==len({j["wandb_id"] for j in jobs})==78
    for environment,n in (("battleship",10),("minesweeper",20)):
        reference=None
        for kind in ("learning","fit"):
            for measure in ("uniform","discounted"):
                for correction in ("uncorrected","is","retrace"):
                    selected=[j for j in jobs if (j["environment"],j["kind"],j["measure"],j["correction"])==
                              (environment,kind,measure,correction)]
                    seeds=[s for j in selected for s in j["seeds"]]
                    assert len(seeds)==len(set(seeds))==n
                    if reference is None:reference=seeds
                    assert seeds==reference
                    assert not any(424200<=s<=424500 for s in seeds)
    names={j["name"] for j in jobs}
    assert all(d in names for j in jobs for d in j["dependencies"])


def fake_job(name, dependencies=()):
    return {"name":name,"session":name,"kind":"learning","environment":"battleship",
            "measure":"uniform","correction":"retrace","seeds":[1,2,3,4,5],
            "wandb_id":"new-"+name,"project":"new-project","dependencies":list(dependencies)}


def test_dispatch_skips_consumed_tickets_and_busy_gpus(tmp_path,monkeypatch):
    jobs=[fake_job("first"),fake_job("second")]
    monkeypatch.setattr(queue,"panes",lambda:{})
    monkeypatch.setattr(queue,"busy_gpus",lambda:{"GPU-busy"})
    commands=[]
    monkeypatch.setattr(queue.subprocess,"run",lambda args,**kw:commands.append(args))
    queue.dispatch(tmp_path,{"jobs":jobs},["GPU-busy","GPU-free"])
    assert read(tmp_path/"jobs/first/ticket.json")["gpu_uuid"]=="GPU-free"
    assert not (tmp_path/"jobs/second/ticket.json").exists()
    first_commands=len(commands)
    queue.dispatch(tmp_path,{"jobs":[jobs[0]]},["GPU-free"])
    assert len(commands)==first_commands
    assert any("CUDA_VISIBLE_DEVICES=GPU-free" in str(c) for c in commands)
    assert not any("history" in str(c) or "resume-from" in str(c) for c in commands)


def test_lost_worker_blocks_transitive_dependencies_without_retry(tmp_path,monkeypatch):
    jobs=[fake_job("first"),fake_job("data",["first"]),fake_job("fit",["data"])]
    write(tmp_path/"jobs/first/ticket.json",jobs[0])
    write(tmp_path/"jobs/first/status.json",{"status":"running"})
    monkeypatch.setattr(queue,"panes",lambda:{})
    status=queue.snapshot(tmp_path,{"jobs":jobs})
    assert [r["status"] for r in status["jobs"]]==["worker-lost","blocked-by-failed-dependency","blocked-by-failed-dependency"]
    assert status["training_retry_allowed"] is False


def test_administrative_finalize_requires_zero_exit_and_complete_history(tmp_path):
    job=fake_job("first")
    directory=tmp_path/"jobs/first"
    write(directory/"status.json",{"status":"finalization-failed","error":"upload failed"})
    write(directory/"training-exit.json",{"exit_code":1})
    with pytest.raises(RuntimeError,match="cannot finalize"):
        queue.finalize_administrative(tmp_path,{"jobs":[job]})
    write(directory/"training-exit.json",{"exit_code":0})
    (directory/"metrics.jsonl").write_text('{"round":1}\n')
    study={"jobs":[job],"learning":{"battleship":{"rounds":2}}}
    with pytest.raises(RuntimeError,match="incomplete learner history"):
        queue.finalize_administrative(tmp_path,study)
    (directory/"metrics.jsonl").write_text('{"round":1}\n{"round":2}\n')
    write(directory/"summary.json",{"all_seeds_retained":True})
    queue.finalize_administrative(tmp_path,study)
    assert read(directory/"status.json")["status"]=="complete"
    assert (directory/"administrative-finalization-history.jsonl").exists()


def test_tmux_logging_keeps_output_and_propagates_nonzero_learner_exit(tmp_path):
    path=tmp_path/"console.txt"
    command=shlex.join([sys.executable,"-c","print('validation output',flush=True); raise SystemExit(7)"])
    result=subprocess.run(queue.logged_command(command,path),shell=True,text=True,capture_output=True)
    assert result.returncode==7
    assert "validation output" in result.stdout
    assert path.read_text()==result.stdout
