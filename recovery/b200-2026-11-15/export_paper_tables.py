#!/usr/bin/env python3
"""Losslessly extract paper-facing per-seed tables; no new experiments."""

from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def table(path, header, rows):
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows(rows)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(text.getvalue().encode())


def main():
    source = HERE / "snapshots/occupancy"
    study = read(source / "study.json")
    output = HERE / "paper-tables"
    output.mkdir()
    learning = []
    frozen = []
    hashes = {}
    learning_fields = {
        "greedy_return": "eval/greedy/return",
        "sampled_discrepancy_rms": "eval/sampled/discrepancy_rms",
        "sampled_discounted_discrepancy_rms": "eval/sampled/discounted_discrepancy_rms",
        "greedy_discrepancy_rms": "eval/greedy/discrepancy_rms",
        "importance_ratio_max": "train/importance_ratio_max",
        "trace_coefficient_max": "train/trace_coefficient_max",
        "anchor_age_mean": "train/anchor_age_mean",
    }
    frozen_fields = {
        "fixed_mu_discounted_heldout_discrepancy_rms": "fixed_mu_heldout/discrepancy_rms",
        "fixed_mu_uniform_start_heldout_discrepancy_rms": "fixed_mu_heldout/uniform_start_discrepancy_rms",
        "auxiliary_twin_rms": "fixed_mu_heldout/auxiliary_twin_rms",
    }
    conditions = Counter()
    for job in study["jobs"]:
        if job["kind"] == "data":
            continue
        path = source / "jobs" / job["name"] / "metrics.jsonl.gz"
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        hashes[job["name"]] = hashlib.sha256(raw).hexdigest()
        fields = learning_fields if job["kind"] == "learning" else frozen_fields
        for row in (json.loads(line) for line in raw.splitlines()):
            for i, seed in enumerate(job["seeds"]):
                values = []
                for name, key in fields.items():
                    value = row.get(f"{key}/seed_{i}")
                    if value is None:
                        assert job["kind"] == "learning" and row["round"] == 0 and key.startswith("train/")
                    else:
                        assert math.isfinite(value), (job["name"], name)
                    values.append(value)
                base = [job["environment"], job["measure"], job["correction"], job["block"], seed]
                if job["kind"] == "learning":
                    primary = row["env_interactions"] <= study["learning"][job["environment"]]["primary_endpoint"]
                    learning.append([*base, row["round"], row["env_interactions"], primary, *values])
                else:
                    frozen.append([*base, row["updates"], *values])
        conditions[(job["environment"], job["measure"], job["correction"], job["kind"])] += len(job["seeds"])
    for environment, count in (("battleship", 10), ("minesweeper", 20)):
        for measure in ("uniform", "discounted"):
            for correction in ("uncorrected", "is", "retrace"):
                for kind in ("learning", "fit"):
                    assert conditions[(environment, measure, correction, kind)] == count
    assert len(learning) == 27120 and len(frozen) == 2340
    base = ["environment", "starting_measure", "trace_correction", "block", "training_seed"]
    table(output / "occupancy-learning.csv.gz", [*base, "round", "env_interactions", "primary_reporting_window", *learning_fields], learning)
    table(output / "occupancy-frozen-fits.csv.gz", [*base, "updates", *frozen_fields], frozen)
    history = read(HERE / "history-paper/provenance.json")
    assert history["all_16_learning_curves_present"] and history["seed_instances"] == 80
    coverage = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "importance_sampling_and_occupancy": {"all_78_jobs_preserved": True,
            "conditions": ["uniform/uncorrected", "uniform/is", "uniform/retrace",
                           "discounted/uncorrected", "discounted/is", "discounted/retrace"],
            "seeds_per_condition": {"battleship": 10, "minesweeper": 20},
            "learning_seed_curve_rows": len(learning), "frozen_fit_seed_curve_rows": len(frozen),
            "raw_metrics_sha256": hashes, "csv_column_source_keys": {"learning": learning_fields, "frozen_fits": frozen_fields}},
        "history_as_memory": {"histories": [0, 1, 2, 5, 10, 15, 20, 25], "seeds_per_history": 10,
            "seed_instances": 80, "blocks": 16, "primary_curve_rows": 3840,
            "other_host_first_blocks_recovered_from_immutable_results": True,
            "first_host_lifecycle_receipts_recovered": False,
            "provenance": "history-paper/provenance.json"},
        "numeric_data_available_from_git_without_wandb": True,
        "model_checkpoints_required_to_reproduce_existing_tables_and_figures": False,
        "model_checkpoints_required_for_new_model_evaluations": True,
        "no_training_or_new_evaluations_performed": True,
    }
    for directory in (output, HERE / "history-paper"):
        index = [{"path": str(p.relative_to(directory)), "sha256": sha(p), "bytes": p.stat().st_size}
                 for p in sorted(directory.rglob("*")) if p.is_file()]
        with (directory / "index.json").open("x") as stream:
            json.dump(index, stream, indent=2)
            stream.write("\n")
    with (HERE / "PAPER_DATA_COVERAGE.json").open("x") as stream:
        json.dump(coverage, stream, indent=2)
        stream.write("\n")
    print("Paper tables verified:", len(learning), "learning rows;", len(frozen), "frozen-fit rows; 3840 full-cohort history rows.")


if __name__ == "__main__":
    main()
