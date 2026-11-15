"""Bounded preflight subprocesses, at most one per selected physical GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import os
from pathlib import Path
import subprocess

from .records import read, utc, write
from .study import CORRECTIONS, DEFAULT_STATE, MEASURES, PYTHON, ROOT, check_native_receipt, prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus",default="0,1,2,3,4,5")
    args=parser.parse_args()
    prepare(DEFAULT_STATE)
    text=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    by_index={int(line.split(",")[0]):line.split(",")[1].strip() for line in text.splitlines()}
    selected=[by_index[int(i)] for i in args.gpus.split(",")]
    if len(set(selected))!=len(selected):raise RuntimeError("duplicate validation GPU")
    occupied=set(subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True).splitlines())
    if set(selected)&occupied:raise RuntimeError("a validation GPU is already in use")
    profiles=list(product(("battleship","minesweeper"),MEASURES,CORRECTIONS))
    groups=[profiles[i::len(selected)] for i in range(len(selected))]
    def worker(pair):
        gpu,profiles=pair
        results=[]
        for environment,measure,correction in profiles:
            path=DEFAULT_STATE/"validation"/f"native-{environment}-{measure}-{correction}.json"
            if path.exists():
                try:
                    check_native_receipt(read(path),environment,measure,correction)
                    results.append({"profile":path.stem,"passed":True,"existing":True})
                    continue
                except RuntimeError:
                    previous=path.parent/"previous"/(path.name+"."+utc().replace(":","-"))
                    previous.parent.mkdir(exist_ok=True)
                    if previous.exists():raise FileExistsError(previous)
                    path.rename(previous)
            log=path.with_suffix(".console.txt")
            if log.exists():
                previous=log.parent/"previous"/(log.name+"."+utc().replace(":","-"))
                previous.parent.mkdir(exist_ok=True);log.rename(previous)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,JAX_PLATFORMS="cuda",XLA_PYTHON_CLIENT_PREALLOCATE="false",
                     OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONPATH=str(ROOT/"src")+":"+str(ROOT))
            with log.open("x") as stream:
                code=subprocess.run([str(PYTHON),"-u","-B","-m","studies.occupancy_correction.native_validation",
                    "--environment",environment,"--measure",measure,"--correction",correction,"--output",str(path)],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
            passed=code==0 and path.exists()
            if passed:check_native_receipt(read(path),environment,measure,correction)
            result={"profile":path.stem,"passed":passed,"exit_code":code,"gpu":gpu,"log":str(log)}
            results.append(result)
            print(result,flush=True)
            if not passed:break  # No hidden validation retry; inspect first.
        return results
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        results=[r for group in pool.map(worker,zip(selected,groups)) for r in group]
    passed=len(results)==12 and all(r["passed"] for r in results)
    write(DEFAULT_STATE/"validation/native-suite.json",{"passed":passed,"results":results,"utc":utc()})
    if not passed:raise RuntimeError("native suite incomplete/failed; production remains blocked")
    print("All 12 full-shape environment/sampler/trace native checks passed",flush=True)


if __name__=="__main__":main()
