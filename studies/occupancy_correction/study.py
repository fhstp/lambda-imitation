"""Explicit fresh allocation and fail-closed admission for the new study."""

from itertools import product
import hashlib
import importlib.metadata
import json
from pathlib import Path
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

from .records import read, sha, utc, verify_sources, write


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = Path("/raid/ustp/experiments/occupancy-correction-2026-11-13")
PYTHON = Path("/raid/ustp/git/lambda-imitation/.venv-b200/bin/python")
MEASURES = ("uniform", "discounted")
CORRECTIONS = ("uncorrected", "is", "retrace")
ENTITY = "fhstp-data-intelligence-research-group"
STUDY_ID = "occupancy-correction-20261113-v1"


def job_name(kind, env, measure, correction, block):
    short = "bs" if env == "battleship" else "ms"
    return f"oc-{kind}-{short}-{measure}-{correction}-b{block:02d}"


def build_study():
    jobs = []
    for environment, count, first_seed in (("battleship", 10, 2026111300), ("minesweeper", 20, 2026111400)):
        project = "retrace-rebuttal-occupancy-"+environment+"-20261113"
        for block in range(1, count//5+1):
            seeds = list(range(first_seed+5*(block-1), first_seed+5*block))
            reference = job_name("learning", environment, "uniform", "retrace", block)
            data_name = job_name("data", environment, "uniform", "retrace", block)
            for kind, measure, correction in product(("learning", "fit"), MEASURES, CORRECTIONS):
                name = job_name(kind, environment, measure, correction, block)
                jobs.append({"name":name, "session":name, "kind":kind, "environment":environment,
                    "measure":measure, "correction":correction, "block":block, "seeds":seeds,
                    "wandb_id":hashlib.sha256((STUDY_ID+"/"+name).encode()).hexdigest()[:12],
                    "project":project, "dependencies":[data_name] if kind=="fit" else [],
                    **({"data_job":data_name} if kind=="fit" else {})})
            jobs.append({"name":data_name, "session":data_name, "kind":"data", "environment":environment,
                "measure":"uniform", "correction":"retrace", "block":block, "seeds":seeds,
                "wandb_id":hashlib.sha256((STUDY_ID+"/"+data_name).encode()).hexdigest()[:12],
                "project":"retrace-rebuttal-validation", "dependencies":[reference], "reference_job":reference})
    # Start common-reference learning first, then other learning arms; data/fits
    # are dependency-gated and cannot affect the source learner's training RNG.
    jobs.sort(key=lambda j: (0 if j["kind"]=="learning" and j["measure"]=="uniform" and j["correction"]=="retrace"
                             else 1 if j["kind"]=="learning" else 2 if j["kind"]=="data" else 3,
                             j["environment"], j["block"], j["measure"], j["correction"]))
    return {
        "study_id":STUDY_ID, "entity":ENTITY, "validation_project":"retrace-rebuttal-validation",
        "coordination_id":"oc-coordination-20261113-v1", "schema":1,
        "scientific_scope":"fresh 2x3 adaptive-learning factorial plus separate fixed-mu frozen-reference critic fits",
        "adaptive_replay_claim":"discounted empirical episode mixture, not one stationary-mu posterior",
        "learning":{
            "battleship":{"seeds_per_condition":10,"prefill":12800,"rounds":50,"steps_per_round":10000,
                          "eval_episodes":50,"eval_initial":False,"reference_round":48,"primary_endpoint":492800},
            "minesweeper":{"seeds_per_condition":20,"prefill":6016,"rounds":200,"steps_per_round":5000,
                           "eval_episodes":128,"eval_initial":True,"reference_round":200,"primary_endpoint":1006016}},
        "frozen":{"data_calls_per_seed":100000,"heldout_calls_per_seed":10000,"updates":12000,
                  "eval_every_updates":1000,"collector":"fixed uniform on actual legal support",
                  "common_reference":"fresh uniform-start Retrace primary-endpoint encoder/online actor",
                  "representation_actor_task_frozen":True,"fresh_auxiliary_heads":True,
                  "shared_fitting_and_heldout_data_across_six_fits":True},
        "no_training_resume":True,"no_automatic_training_retry":True,"all_seeds_reported":True,
        "analysis":"seed-level means, pointwise 95% Student-t CIs; paired exploratory endpoint contrasts",
        "jobs":jobs,
    }


def prepare(state=DEFAULT_STATE):
    state = Path(state).resolve()
    if state != DEFAULT_STATE.resolve():
        raise RuntimeError("use the original persistent state; no duplicate state directories")
    expected = build_study()
    if (state/"study.json").exists():
        if read(state/"study.json") != expected: raise RuntimeError("study allocation changed")
    else:
        if socket.gethostname() != "dgxB200": raise RuntimeError("this allocation is for dgxB200 only")
        write(state/"study.json", expected, exclusive=True)
        write(state/"ownership.json", {"root":str(ROOT),"hostname":socket.gethostname(),"created_utc":utc()}, exclusive=True)
    if read(state/"ownership.json")["root"] != str(ROOT): raise RuntimeError("persistent state belongs to another checkout")
    return expected


def scientific_sources():
    paths = [*sorted((ROOT/"src").rglob("*.py")),
             *sorted((ROOT/"studies/occupancy_correction").glob("*.py")),
             ROOT/"studies/occupancy_correction/PROTOCOL.md",
             ROOT/"studies/occupancy_correction/AGENTS.md",
             ROOT/"studies/occupancy_correction/README.md"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def runtime():
    return {p:importlib.metadata.version(p) for p in ("jax","jaxlib","flax","optax","numpy","scipy","gymnax","wandb")}


def check_native_receipt(receipt, environment, measure, correction):
    if not receipt.get("passed") or receipt.get("production_admitted") or not receipt.get("validation_only_no_state_reuse"):
        raise RuntimeError("invalid native proof")
    if receipt["environment"]!=environment or receipt["settings"]["measure"]!=measure or receipt["settings"]["correction"]!=correction:
        raise RuntimeError("native proof does not cover requested condition")
    if receipt["num_seeds"]!=5 or receipt["native_online_updates"]<128 or receipt["native_frozen_updates"]<64:
        raise RuntimeError("native proof used reduced seed/update dimensions")
    if "B200" not in receipt["device_kind"]: raise RuntimeError("native proof was not run on a B200")
    # Administrative files added after a native run do not change its computation;
    # every executed learner/environment/collector/sampler file must still match.
    critical = [n for n in scientific_sources() if n.startswith("src/") or n.rsplit("/",1)[-1] in
                ("learner.py","replay.py","collection.py","sampling.py","native_validation.py")]
    for name in critical:
        if receipt["source_sha256"].get(name)!=sha(ROOT/name):
            raise RuntimeError("native proof source mismatch: "+name)


def admit(state=DEFAULT_STATE):
    state=Path(state)
    study=prepare(state)
    if any((state/"jobs"/j["name"]/"ticket.json").exists() for j in study["jobs"]):
        raise RuntimeError("workers have consumed tickets; never reseal an active/attempted study")
    if (state/"admission.json").exists():
        admission=read(state/"admission.json")
        verify_sources(ROOT,admission)
        return admission
    if sys.version_info[:2]!=(3,12): raise RuntimeError("use the unchanged validated Python 3.12 runtime")
    xml=state/"validation"/"full-tests.xml"
    suites=list(ET.parse(xml).getroot().iter("testsuite"))
    if sum(int(s.attrib["tests"]) for s in suites)<100 or any(int(s.attrib.get("failures",0))+int(s.attrib.get("errors",0))+int(s.attrib.get("skipped",0)) for s in suites):
        raise RuntimeError("full CPU validation missing/incomplete/failed")
    cpu_receipt=read(state/"validation/full-tests-source.json")
    if cpu_receipt["junit_sha256"]!=sha(xml):raise RuntimeError("CPU test receipt/XML mismatch")
    if cpu_receipt["source_sha256"]!=scientific_sources():
        raise RuntimeError("source changed after full CPU validation")
    proofs={}
    for environment,measure,correction in product(("battleship","minesweeper"),MEASURES,CORRECTIONS):
        path=state/"validation"/f"native-{environment}-{measure}-{correction}.json"
        check_native_receipt(read(path),environment,measure,correction)
        proofs[path.name]=sha(path)
    canary=read(state/"validation"/"cloud-roundtrip.json")
    if not canary.get("passed"): raise RuntimeError("cloud roundtrip is not verified")
    # Each actual worker path, not only learner kernels, must have a GPU proof.
    worker_proof=read(state/"validation"/"worker-paths.json")
    if not worker_proof.get("passed") or set(worker_proof.get("kinds",[]))!={"learning","data","fit"}:
        raise RuntimeError("real worker lifecycle GPU validation incomplete")
    for name,digest in worker_proof["source_sha256"].items():
        if sha(ROOT/name)!=digest: raise RuntimeError("worker lifecycle source changed: "+name)
    admission={"production_admitted":True,"utc":utc(),"study_sha256":sha(state/"study.json"),
        "source_sha256":scientific_sources(),"runtime":runtime(),"native_proofs_sha256":proofs,
        "cpu_tests_sha256":sha(xml),"worker_paths_sha256":sha(state/"validation"/"worker-paths.json"),
        "cpu_source_receipt_sha256":sha(state/"validation/full-tests-source.json"),
        "cloud_roundtrip":canary,"training_resume_allowed":False,"training_retry_allowed":False,
        "base_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    write(state/"admission.json",admission,exclusive=True)
    return admission
