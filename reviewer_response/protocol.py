"""Predeclare the matched 15-job, 90-seed replay-correction experiment."""

import hashlib
import json
from pathlib import Path
import secrets
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ARMS = ("uncorrected", "is", "retrace")
SEED_BLOCKS = {
    "battleship": ((7451234536, 5), (7451234542, 5)),
    "minesweeper": ((1000000, 10), (2342455, 5), (2342465, 5)),
}


def flags(env, arm, seed, count):
    common = ["--method", "ld", "--num-seeds", str(count), "--seed", str(seed),
              "--memory-type", "gru", "--fe-lr", "1e-5", "--actor-lr", "1e-4",
              "--critic-lr", "1e-4", "--gamma", "0.99", "--tau", "0.005",
              "--critic-layer-norm", "--online-buffer-size", "100000",
              "--checkpoint-every", "10", "--eval-every", "1", "--final-return-window", "5",
              "--replay-correction", arm, "--lambda-critic-loss", "mse", "--trace-diagnostics"]
    if env == "battleship":
        return common + ["--rows", "5", "--cols", "5", "--ship-lengths", "3,2",
            "--memory-hidden-dim", "512", "--rounds", "50", "--train-steps", "10000",
            "--batch-size", "128", "--burn-in-length", "10", "--sequence-length", "40",
            "--lambda-truncation", "50", "--lambda1", "0.05", "--lambda2", "0.85",
            "--lambda-coef", "0.01", "--alpha", "0.1", "--prefill-steps", "12800",
            "--eval-episodes", "50"]
    if env == "minesweeper":
        return common + ["--rows", "6", "--cols", "6", "--mines", "6",
            "--memory-hidden-dim", "128", "--projection-dim", "128", "--head-dim", "128",
            "--rounds", "200", "--train-steps", "5000", "--batch-size", "64",
            "--burn-in-length", "30", "--sequence-length", "32", "--lambda-truncation", "32",
            "--lambda1", "0", "--lambda2", "0.95", "--lambda-coef", "0.1", "--alpha", "0.01",
            "--prefill-steps", "6016", "--eval-episodes", "128", "--eval-initial"]
    raise ValueError(env)


def source_hashes():
    paths = [*sorted((ROOT / "src").rglob("*.py")),
             *sorted((ROOT / "examples").glob("*.py")),
             *sorted(HERE.glob("*.py")), *sorted(HERE.glob("*.sh")), HERE / "protocol.json"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.exists()}


def main():
    destination = HERE / "protocol.json"
    if destination.exists():
        raise FileExistsError("Protocol already sealed; never regenerate run IDs")
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True).splitlines()
    gpus = []
    for row in rows:
        index, uuid, name, total, used = [s.strip() for s in row.split(",")]
        if int(used) > 100:
            raise RuntimeError(f"GPU {index} is occupied; do not disturb it")
        gpus.append({"index": int(index), "uuid": uuid, "name": name,
                     "memory_total_mib": int(total)})
    assert len(gpus) == 6
    jobs = []
    for env_index, (env, blocks) in enumerate(SEED_BLOCKS.items()):
        for arm_index, arm in enumerate(ARMS):
            gpu = gpus[env_index * 3 + arm_index]
            for block, (seed, count) in enumerate(blocks, 1):
                name = f"{env}-{arm}-mse-block{block}-{count}seeds"
                jobs.append({"name": name, "env": env, "arm": arm, "base_seed": seed,
                    "num_seeds": count, "training_seeds": list(range(seed, seed + count)),
                    "effective_uint32_seeds": [(seed + i) % 2**32 for i in range(count)],
                    "block": block, "gpu_index": gpu["index"], "gpu_uuid": gpu["uuid"],
                    "session": f"rr-gpu{gpu['index']}-{env}-{arm}-b{block}",
                    "wandb_id": secrets.token_hex(4),
                    "wandb_project": f"retrace-rebuttal-{env}-replay-correction",
                    "flags": flags(env, arm, seed, count)})
    result = {
        "schema_version": 1, "date": "2026-11-07", "purpose": "Reviewer replay-correction comparison",
        "conditions": {"uncorrected": "c_t=lambda", "is": "c_t=lambda*pi(a_t|h_t)/mu(a_t|h_t)",
                       "retrace": "c_t=lambda*min(1,pi/mu)"},
        "current_td_residual_coefficient": 1,
        "sole_between_arm_learning_difference": "future auxiliary trace coefficients",
        "auxiliary_regression": "MSE in ALL new conditions; two independent twins",
        "historical_source_caveat": "Publication default is Huber; historical paper MSE source cannot be certified from Git alone. All three arms are fresh matched runs, not spliced historical controls.",
        "runtime_caveat": "Blackwell-compatible isolated runtime differs from historical runtime; no bitwise historical reproduction claim.",
        "training_seeds_per_condition": {"battleship": 10, "minesweeper": 20},
        "total_jobs": 15, "total_training_seeds": 90,
        "analysis_total_interaction_limit": {"battleship": 500000, "minesweeper": 1006016},
        "battleship_training_note": "50x10000 online steps plus 12800 prefill; only 48 original-grid checkpoints <=500000 enter primary analysis. No use of extra endpoint for primary selection.",
        "tuning": "none; paper-selected parameters fixed across arms",
        "evaluation_rng": "separate fixed fold-in streams, same within matched seed blocks",
        "statistics": "seed-level means and pointwise 95% Student-t CIs; matched-arm contrasts use paired seed trajectories; first-passage uses 10000 whole-seed resamples, strict crossing and right censoring",
        "failure_policy": "No numerical clipping, seed replacement, failed-seed exclusion, automatic retries or budget escalation. Preserve failed block and continue only predeclared independent blocks on numerical failure; runtime/integrity failures stop lane.",
        "launch_gate": "CPU tests plus full-shape native repeat/resume, action/probability/memory checks and source-bound admission.json",
        "gpu_placement": gpus, "jobs": jobs,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
