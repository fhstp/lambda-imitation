#!/usr/bin/env python3
"""Recover the other host's immutable history results; never starts training."""

import base64
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path("/raid/ustp/migration-2026-11-15/history-first-block-downloads")
PROJECT = "fhstp-data-intelligence-research-group/retrace-rebuttal-battleship-history-length-20261109"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def compressed(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(data)


def main():
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    destination = HERE / "history-paper"
    if destination.exists():
        raise FileExistsError("Preserve the first export; do not silently replace it")
    plan = json.loads((ROOT / "reviewer_response/protocol.json").read_text())
    admitted = json.loads((ROOT / "reviewer_response/admission.json").read_text())
    api = wandb.Api(timeout=120)
    data = {}
    provenance = []
    inputs = {}
    metrics = {"return": "eval/greedy/return", "discrepancy_rms": "eval/sampled/discrepancy_rms",
               "greedy_discrepancy_rms": "eval/greedy/discrepancy_rms",
               "aux_twin_rms": "eval/sampled/aux_twin_rms", "replay_discrepancy_rms": "train/ld_loss"}
    for job in sorted(plan["jobs"], key=lambda j: (j["history"], j["block"])):
        if job["block"] == 1:
            run = api.run(PROJECT + "/" + job["wandb_id"])
            assert run.state == "finished", job["name"]
            config = dict(run.config)
            for name, expected in admitted["source_sha256"].items():
                if name.startswith("src/") or name in ("examples/_common.py", "examples/battleship.py"):
                    assert config["source_sha256"][name] == expected, (job["name"], name)
            expected = {"history_length": job["history"], "seed": job["base_seed"], "num_seeds": 5,
                        "resume_from": None, "history_context": 26, "rounds": 50, "train_steps": 10000,
                        "prefill_steps": 12800, "replay_correction": "retrace", "lambda_critic_loss": "mse",
                        "memory_hidden_dim": 512, "batch_size": 128, "burn_in_length": 10,
                        "sequence_length": 40, "lambda_truncation": 50, "lambda1": .05, "lambda2": .85,
                        "lambda_coef": .01, "gamma": .99, "tau": .005, "alpha": .1,
                        "fe_lr": 1e-5, "actor_lr": 1e-4, "critic_lr": 1e-4,
                        "method": "ld", "memory_type": "gru", "critic_layer_norm": True,
                        "rows": 5, "cols": 5, "ship_lengths": "3,2", "online_buffer_size": 100000,
                        "eval_episodes": 50, "eval_every": 1, "trace_diagnostics": True}
            for name, value in expected.items():
                assert config[name] == value, (job["name"], name, config[name], value)
            artifact = api.artifact(PROJECT + "/replay-correction-results-" + job["wandb_id"] + ":v0")
            assert artifact.state == "COMMITTED"
            files = {}
            payloads = {}
            for name in ("config.json", "metrics.jsonl", "summary.json"):
                entry = artifact.manifest.entries[name]
                assert not entry.ref
                path = Path(artifact.get_entry(name).download(root=str(OUT / job["name"]), skip_cache=True))
                assert path.stat().st_size == entry.size
                with path.open("rb") as stream:
                    md5 = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
                assert md5 == entry.digest
                payloads[name] = path.read_bytes()
                target = destination / "first-blocks" / job["name"] / (name + ".gz" if name == "metrics.jsonl" else name)
                if name == "metrics.jsonl":
                    compressed(target, payloads[name])
                else:
                    write(target, json.loads(payloads[name]))
                files[name] = {"sha256": sha(path), "md5": md5, "bytes": entry.size,
                               "saved_path": str(target.relative_to(HERE))}
            local_config = json.loads(payloads["config.json"])
            for name, value in expected.items():
                assert local_config[name] == value
            write(destination / "first-blocks" / job["name"] / "wandb-config.json", config)
            rows = [json.loads(line) for line in payloads["metrics.jsonl"].splitlines()]
            inputs[job["name"]] = files["metrics.jsonl"]["sha256"]
            provenance.append({"job": job, "artifact": artifact.qualified_name, "wandb_state": run.state,
                "source_and_configuration_match_admission": True, "files": files,
                "host_lifecycle_receipt_recovered": False,
                "completion_evidence": "Finished W&B run, committed final result artifact, exact 50-round grid; no-resume config and admitted source hashes verified"})
            print("RECOVERED FIRST HISTORY BLOCK", job["history"], flush=True)
        else:
            directory = HERE / "snapshots/history/jobs" / job["name"] / "result"
            with gzip.open(directory / "metrics.jsonl.gz", "rb") as stream:
                raw = stream.read()
            rows = [json.loads(line) for line in raw.splitlines()]
            inputs[job["name"]] = hashlib.sha256(raw).hexdigest()
            receipt = json.loads((directory / "receipt.json").read_text())
            assert receipt["exit_code"] == 0 and receipt["continuous_no_resume"]
        rounds = [r for r in rows if "round" in r]
        assert [r["round"] for r in rounds] == list(range(1, 51)), job["name"]
        assert [r["env_interactions"] for r in rounds] == list(range(22800, 512801, 10000))
        primary = [r for r in rounds if r["env_interactions"] <= plan["analysis_interaction_limit"]]
        assert len(primary) == 48 and primary[-1]["env_interactions"] == plan["primary_endpoint_interactions"]
        arrays = {name: np.array([[r[f"{key}/seed_{i}"] for i in range(5)] for r in primary])
                  for name, key in metrics.items()}
        arrays["replay_discrepancy_rms"] = np.sqrt(arrays["replay_discrepancy_rms"])
        assert all(np.isfinite(a).all() for a in arrays.values())
        data.setdefault(job["history"], []).append({"seeds": job["training_seeds"], "arrays": arrays})
    # Use the admitted report's unchanged summary function, not a new estimator.
    spec = importlib.util.spec_from_file_location("admitted_history_report", ROOT / "reviewer_response/report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(["history", "training_seed", "env_interactions", *metrics])
    statistics = {"histories": {}, "n_seeds_per_history": 10, "primary_endpoint_interactions": 492800,
                  "confidence": "Pointwise 95% Student-t across independent training seeds",
                  "historical_results_pooled": False, "input_metrics_sha256": inputs,
                  "first_host_lifecycle_receipts_not_recovered": True,
                  "completion_evidence_caveat": "First blocks verified from finished W&B runs, immutable final results, full grids, configs and source hashes; original host lifecycle receipts are not included"}
    for history, blocks in sorted(data.items()):
        seeds = [seed for block in blocks for seed in block["seeds"]]
        assert len(seeds) == len(set(seeds)) == 10
        arrays = {name: np.concatenate([b["arrays"][name] for b in blocks], axis=1) for name in metrics}
        statistics["histories"][str(history)] = {"seeds": seeds, "endpoint": {name: report.describe(a[-1]) for name, a in arrays.items()}}
        for i, seed in enumerate(seeds):
            for k, step in enumerate(range(22800, 492801, 10000)):
                writer.writerow([history, seed, step, *[float(arrays[name][k, i]) for name in metrics]])
    assert len(statistics["histories"]) == 8
    compressed(destination / "seed_metrics.csv.gz", table.getvalue().encode())
    write(destination / "statistics.json", statistics)
    write(destination / "provenance.json", {"utc": datetime.now(timezone.utc).isoformat(),
          "scope": "Supplemental recovery of original-host first blocks; these runs did not execute on dgxB200",
          "result_artifacts": provenance, "all_16_learning_curves_present": True, "seed_instances": 80,
          "primary_grid_rows": 3840, "training_started_or_resumed": False,
          "large_first_host_checkpoints_downloaded": False})
    print("COMPLETE HISTORY PAPER DATA EXPORTED: 80 seed instances, 3840 per-seed checkpoint rows.", flush=True)


if __name__ == "__main__":
    main()
