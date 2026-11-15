"""Persistent tmux dispatcher/observer. Repairs uploads, never retries training."""

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from .records import read, sha, utc, verify_sources, write
from .study import DEFAULT_STATE, PYTHON, ROOT, admit, prepare, runtime


def panes():
    result = subprocess.run(["tmux","list-panes","-a","-F","#{session_name}\t#{pane_dead}\t#{pane_dead_status}"],
                            text=True,capture_output=True)
    return {name:{"dead":dead=="1","exit":code} for name,dead,code in
            (line.split("\t") for line in result.stdout.splitlines())}


def gpus():
    result=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,name,memory.total","--format=csv,noheader,nounits"],text=True)
    selected=[]
    for line in result.splitlines():
        uuid,name,memory=[s.strip() for s in line.split(",")]
        if "B200" in name and int(memory)>=90000: selected.append(uuid)
    if len(selected)!=8: raise RuntimeError("allocation requires eight full B200 cards")
    return selected


def busy_gpus():
    result=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
    return {line.strip() for line in result.splitlines() if line.strip()}


def logged_command(command, path):
    """Keep output visible in tmux AND persistent, without hiding learner exit."""
    return shlex.join(["bash","-o","pipefail","-c",command+" 2>&1 | tee "+shlex.quote(str(path))])


def snapshot(state, study):
    current=panes()
    records=[]
    statuses={j["name"]:read(state/"jobs"/j["name"]/"status.json") if
              (state/"jobs"/j["name"]/"status.json").exists() else {} for j in study["jobs"]}
    phases={}
    for job in study["jobs"]:
        ticket=(state/"jobs"/job["name"]/"ticket.json").exists()
        phase=statuses[job["name"]].get("status","starting" if ticket else "queued")
        if phase in ("running","starting") and ((job["session"] not in current) or current[job["session"]]["dead"]):
            phase="worker-lost"
        phases[job["name"]]=phase
    for _ in range(len(study["jobs"])):
        changed=False
        for job in study["jobs"]:
            if phases[job["name"]]=="queued" and any(phases[d] in
                ("failed","worker-lost","blocked-by-failed-dependency") for d in job["dependencies"]):
                phases[job["name"]]="blocked-by-failed-dependency"
                changed=True
        if not changed:break
    for job in study["jobs"]:
        directory=state/"jobs"/job["name"]
        record=statuses[job["name"]]
        phase=phases[job["name"]]
        records.append({"name":job["name"],"kind":job["kind"],"status":phase,
                        "round":record.get("round"),"updates":record.get("updates"),
                        "error":record.get("error"),"published":(directory/"result-published.json").exists()})
    return {"utc":utc(),"jobs":records,"training_retry_allowed":False,
            "complete":all(r["status"]=="complete" and r["published"] for r in records)}


def dispatch(state, study, gpu_uuids):
    current=panes()
    occupied=busy_gpus()
    reserved=set()
    for job in study["jobs"]:
        path=state/"jobs"/job["name"]/"ticket.json"
        if path.exists() and job["session"] in current and not current[job["session"]]["dead"]:
            reserved.add(read(path)["gpu_uuid"])
    free=[g for g in gpu_uuids if g not in occupied|reserved]
    if not free: return
    if shutil.disk_usage(state).free<100*2**30: raise RuntimeError("disk reserve prevents dispatch")
    available=int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024
    if available<128*2**30: raise RuntimeError("host memory reserve prevents dispatch")
    for job in study["jobs"]:
        if not free: break
        directory=state/"jobs"/job["name"]
        if (directory/"ticket.json").exists(): continue
        if any(not (state/"jobs"/d/"status.json").exists() or
               read(state/"jobs"/d/"status.json")["status"]!="complete" for d in job["dependencies"]): continue
        if job["session"] in current: raise RuntimeError("refusing to hijack existing tmux session: "+job["session"])
        gpu=free.pop(0)
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError("unfamiliar existing job files without a consumed ticket: "+job["name"])
        directory.mkdir(parents=True,exist_ok=True)
        write(directory/"ticket.json",{**job,"gpu_uuid":gpu},exclusive=True)
        for name in ("runs","cache","data","tmp"):(state/"work"/name).mkdir(parents=True,exist_ok=True)
        env={"CUDA_VISIBLE_DEVICES":gpu,"JAX_PLATFORMS":"cuda","XLA_PYTHON_CLIENT_PREALLOCATE":"false",
             "OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONPATH":str(ROOT/"src")+":"+str(ROOT),
             "WANDB_MODE":"online","WANDB_DIR":str(state/"work/runs"),"WANDB_CACHE_DIR":str(state/"work/cache"),
             "WANDB_DATA_DIR":str(state/"work/data"),"TMPDIR":str(state/"work/tmp")}
        command=shlex.join(["env","-u","WANDB_SWEEP_ID","-u","WANDB_RUN_ID","-u","WANDB_RESUME",
                            *[k+"="+v for k,v in env.items()],str(PYTHON),"-u","-B","-m",
                            "studies.occupancy_correction.run_worker","--state",str(state),"--job",job["name"]])
        command=logged_command(command,directory/"console.txt")
        subprocess.run(["tmux","new-session","-d","-s",job["session"],"-c",str(ROOT)],check=True)
        subprocess.run(["tmux","set-option","-t",job["session"],"remain-on-exit","on"],check=True)
        subprocess.run(["tmux","send-keys","-t",job["session"],"exec "+command,"Enter"],check=True)
        print("STARTED",job["name"],gpu,flush=True)


def publish_uploads(state, study, tracking):
    import wandb
    api=wandb.Api(timeout=120)
    for job in study["jobs"]:
        directory=state/"jobs"/job["name"]
        for record_path in sorted((directory/"uploads").glob("*.json")):
            remote_record=record_path.with_suffix(".committed")
            if remote_record.exists(): continue
            record=read(record_path)
            path=(directory/record["path"]).resolve()
            if not path.is_relative_to(directory.resolve()) or sha(path)!=record["sha256"] or path.stat().st_size!=record["bytes"]:
                raise RuntimeError("changed/unowned upload staging file")
            artifact=wandb.Artifact("oc-"+job["wandb_id"]+"-"+record["sha256"][:16],type=record["kind"],
                metadata={"sha256":record["sha256"],"training_run_id":job["wandb_id"],"job":job["name"]})
            artifact.add_file(str(path),name=path.name,policy="immutable",skip_cache=True)
            uploaded=tracking.log_artifact(artifact)
            uploaded.wait()
            remote=api.artifact(uploaded.qualified_name)
            entry=remote.manifest.entries[path.name]
            with path.open("rb") as stream: md5=base64.b64encode(hashlib.file_digest(stream,"md5").digest()).decode()
            if remote.state!="COMMITTED" or entry.digest!=md5 or entry.size!=record["bytes"] or remote.metadata["sha256"]!=record["sha256"]:
                raise RuntimeError("remote artifact verification failed")
            write(remote_record,{"artifact":remote.qualified_name,"sha256":record["sha256"],"md5":md5,
                                  "bytes":entry.size,"utc":utc(),"remote_committed_verified":True},exclusive=True)
            if record["release_after_remote_verification"]:
                if record["kind"]!="checkpoint" or path.parent!=directory.resolve()/"checkpoints" or "failure" in path.name:
                    raise RuntimeError("refusing staging-file release outside successful study checkpoint")
                path.unlink()
            print("VERIFIED_UPLOAD",job["name"],path.name,flush=True)


def finalize_administrative(state, study):
    """Repair local result assembly only after a recorded zero learner exit."""
    from .records import enqueue_upload
    for job in study["jobs"]:
        directory=state/"jobs"/job["name"]
        if not (directory/"status.json").exists(): continue
        record=read(directory/"status.json")
        if record.get("status")!="finalization-failed": continue
        if not (directory/"training-exit.json").exists() or read(directory/"training-exit.json")["exit_code"]!=0:
            raise RuntimeError("cannot finalize a failed or unrecorded training exit")
        rows=[json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
        if job["kind"]=="learning":
            expected=study["learning"][job["environment"]]["rounds"]
            if [r["round"] for r in rows if r.get("round",0)>0]!=list(range(1,expected+1)):
                raise RuntimeError("incomplete learner history; administrative repair refused")
        elif job["kind"]=="fit":
            if rows[-1].get("updates")!=study["frozen"]["updates"]: raise RuntimeError("incomplete frozen fit")
        if not (directory/"summary.json").exists():
            raise RuntimeError("missing scientific summary requires inspection, not fabrication")
        for name in ("metrics.jsonl","summary.json"):
            pointer=directory/"uploads"/(name+".json")
            if pointer.exists():
                if read(pointer)["sha256"]!=sha(directory/name): raise RuntimeError("conflicting retained results")
            else: enqueue_upload(directory,directory/name,kind="results")
        with (directory/"administrative-finalization-history.jsonl").open("a") as file:
            file.write(json.dumps({"previous":record,"repaired_utc":utc()})+"\n")
        write(directory/"status.json",{"status":"complete","kind":job["kind"],"summary":read(directory/"summary.json"),
                                        "administrative_only_repair":True,"utc":utc()})


def publish_results(state, study, tracking):
    import wandb
    for job in study["jobs"]:
        directory=state/"jobs"/job["name"]
        if not (directory/"status.json").exists() or read(directory/"status.json")["status"]!="complete" or (directory/"result-published.json").exists(): continue
        pointers=sorted((directory/"uploads").glob("*.json"))
        if not pointers or any(not p.with_suffix(".committed").exists() for p in pointers): continue
        files=[directory/"config.json",directory/"metrics.jsonl",directory/"summary.json",directory/"status.json",
               *[p.with_suffix(".committed") for p in pointers]]
        for name in ("training-exit.json","reference.json","datasets.json","prefill-validation.json"):
            if (directory/name).exists(): files.append(directory/name)
        index={"job":job,"study_sha256":sha(state/"study.json"),"admission_sha256":sha(state/"admission.json"),
               "files":{str(p.relative_to(directory)):sha(p) for p in files}}
        write(directory/"result-index.json",index)
        artifact=wandb.Artifact("oc-results-"+job["wandb_id"],type="occupancy-results",metadata={"training_run_id":job["wandb_id"]})
        for path in [*files,directory/"result-index.json"]:
            artifact.add_file(str(path),name=str(path.relative_to(directory)),policy="immutable",skip_cache=True)
        uploaded=tracking.log_artifact(artifact); uploaded.wait()
        if uploaded.state!="COMMITTED": raise RuntimeError("result artifact not committed")
        write(directory/"result-published.json",{"artifact":uploaded.qualified_name,"utc":utc()},exclusive=True)


def start(state):
    state=Path(state)
    study=prepare(state)
    if not (state/"admission.json").exists(): raise RuntimeError("run admission first; training is still blocked")
    admission=read(state/"admission.json")
    if not admission.get("production_admitted") or admission["study_sha256"]!=sha(state/"study.json"):
        raise RuntimeError("invalid/changed production admission")
    verify_sources(ROOT,admission)
    if runtime()!=admission["runtime"]: raise RuntimeError("validated runtime changed")
    with (state/"observer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ("runs","cache","data","tmp"):(state/"work"/name).mkdir(parents=True,exist_ok=True)
        os.environ.update(WANDB_DIR=str(state/"work/runs"),WANDB_CACHE_DIR=str(state/"work/cache"),
                          WANDB_DATA_DIR=str(state/"work/data"),TMPDIR=str(state/"work/tmp"))
        os.environ.pop("WANDB_SWEEP_ID",None)
        import wandb
        tracking=wandb.init(entity=study["entity"],project=study["validation_project"],id=study["coordination_id"],
            resume="allow",name="Occupancy correction coordination 2026-11-13",
            job_type="administrative-observer",config={"production_training":False,"study_sha256":sha(state/"study.json")})
        artifact=wandb.Artifact("oc-source-20261113-v1",type="source")
        for name in admission["source_sha256"]: artifact.add_file(str(ROOT/name),name=name,policy="immutable",skip_cache=True)
        for name in ("study.json","admission.json"):artifact.add_file(str(state/name),name=name,policy="immutable",skip_cache=True)
        tracking.log_artifact(artifact).wait()
        gpu_uuids=gpus()
        try:
            while True:
                verify_sources(ROOT,admission)
                dispatch(state,study,gpu_uuids)
                try:
                    finalize_administrative(state,study)
                    publish_uploads(state,study,tracking)
                    publish_results(state,study,tracking)
                except Exception as e:
                    with (state/"observer-errors.jsonl").open("a") as file:
                        file.write(json.dumps({"utc":utc(),"error":repr(e),"training_retried":False})+"\n")
                    print("ADMINISTRATIVE_ERROR",repr(e),flush=True)
                status=snapshot(state,study)
                write(state/"status.json",status)
                print(json.dumps(status),flush=True)
                try:
                    tracking.log({"jobs/"+phase:sum(r["status"]==phase for r in status["jobs"])
                                  for phase in ("complete","running","starting","queued","failed","worker-lost","blocked-by-failed-dependency")})
                    tracking.summary.update({"published_results":sum(r["published"] for r in status["jobs"]),
                        "complete":status["complete"],"training_retry_allowed":False,
                        "failed_job_names":[r["name"] for r in status["jobs"] if r["status"] in ("failed","worker-lost")]})
                except Exception as e:
                    with (state/"observer-errors.jsonl").open("a") as file:
                        file.write(json.dumps({"utc":utc(),"operation":"heartbeat","error":repr(e)})+"\n")
                if status["complete"]:
                    from .report import generate
                    generate(state)
                    report=wandb.Artifact("oc-report-20261113-v1",type="occupancy-report")
                    for path in sorted((state/"report").iterdir()):
                        if path.name!="incomplete.json":report.add_file(str(path),name=path.name,policy="immutable",skip_cache=True)
                    uploaded=tracking.log_artifact(report);uploaded.wait()
                    if uploaded.state!="COMMITTED":raise RuntimeError("report upload did not commit")
                    write(state/"report/published.json",{"artifact":uploaded.qualified_name,"utc":utc()})
                    break
                terminal={"complete","failed","worker-lost","blocked-by-failed-dependency"}
                if all(r["status"] in terminal for r in status["jobs"]) and any(r["status"]!="complete" for r in status["jobs"]):
                    write(state/"finished-with-failures.json",status,exclusive=True)
                    break
                time.sleep(30)
        finally: tracking.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","admit","start","status"))
    parser.add_argument("--state",type=Path,default=DEFAULT_STATE)
    args=parser.parse_args()
    if args.action=="prepare":print(json.dumps(prepare(args.state),indent=2))
    elif args.action=="admit":print(json.dumps(admit(args.state),indent=2))
    elif args.action=="start":start(args.state)
    else:print(json.dumps(snapshot(args.state,read(args.state/"study.json")),indent=2))


if __name__=="__main__":main()
