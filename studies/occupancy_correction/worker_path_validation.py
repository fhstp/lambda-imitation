"""Bounded full-shape GPU execution of the real worker's three job paths."""

import argparse
from pathlib import Path

import jax
import numpy as np

from . import run_worker
from .records import load_owned_pickle, read, sha, utc, write
from .study import DEFAULT_STATE, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment",choices=("battleship","minesweeper"),required=True)
    args=parser.parse_args()
    environment=args.environment
    devices=jax.devices()
    assert len(devices)==1 and devices[0].platform=="gpu" and "B200" in devices[0].device_kind
    state=DEFAULT_STATE/"validation"/("worker-"+environment)
    if (state/"admission.json").exists():raise FileExistsError("preserve this validation attempt")
    critical=[*sorted((ROOT/"src").rglob("*.py")),*[ROOT/"studies/occupancy_correction"/n for n in
        ("learner.py","replay.py","collection.py","sampling.py","run_worker.py","frozen_evaluation.py",
         "records.py","native_validation.py","worker_path_validation.py")]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in critical}
    admission={"source_sha256":hashes}
    write(state/"admission.json",admission,exclusive=True)
    prefill=12800 if environment=="battleship" else 6016
    budget={"prefill":prefill,"rounds":2,"steps_per_round":64,
        "eval_episodes":50 if environment=="battleship" else 128,"eval_initial":environment=="minesweeper",
        "reference_round":1,"primary_endpoint":prefill+64}
    study={"learning":{environment:budget},"frozen":{"data_calls_per_seed":2000,"heldout_calls_per_seed":500,
        "updates":128,"eval_every_updates":64},"disk_reserve_bytes":100*2**30}
    seeds=[425000+(100 if environment=="minesweeper" else 0)+i for i in range(5)]
    common={"environment":environment,"seeds":seeds,"measure":"uniform","correction":"retrace"}
    learning={**common,"kind":"learning","name":"learning","wandb_id":"validation-learning"}
    learning_dir=state/"jobs/learning";learning_dir.mkdir(parents=True)
    run_worker.execute_job(state,learning,learning_dir,study,admission,None)
    data_job={**common,"kind":"data","name":"data","wandb_id":"validation-data","reference_job":"learning"}
    data_dir=state/"jobs/data";data_dir.mkdir()
    run_worker.execute_job(state,data_job,data_dir,study,admission,None)
    record=read(data_dir/"datasets.json")
    data=load_owned_pickle(data_dir/record["name"],record["sha256"])
    assert run_worker.digest_tree(data["fitting"])!=run_worker.digest_tree(data["heldout"])
    fit_job={**common,"kind":"fit","name":"fit","wandb_id":"validation-fit","data_job":"data",
        "measure":"discounted","correction":"is"}
    fit_dir=state/"jobs/fit";fit_dir.mkdir()
    run_worker.execute_job(state,fit_job,fit_dir,study,admission,None)
    for name,digest in hashes.items():assert sha(ROOT/name)==digest
    report={"passed":True,"environment":environment,"kinds":["learning","data","fit"],
            "num_seeds":5,"source_sha256":hashes,"validation_only_no_state_reuse":True,"utc":utc()}
    write(DEFAULT_STATE/"validation"/("worker-paths-"+environment+".json"),report,exclusive=True)
    print("Full-shape worker paths passed:",environment,flush=True)


if __name__=="__main__":main()
