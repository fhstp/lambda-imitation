#!/usr/bin/env python3
"""Verify both paper datasets from Git; no packages, W&B, GPU or models needed."""

from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with gzip.open(path, "rt") as stream:
        return list(csv.DictReader(stream))


def main():
    for name in ("history-paper", "paper-tables"):
        directory = HERE / name
        for item in read(directory / "index.json"):
            path = directory / item["path"]
            assert path.stat().st_size == item["bytes"], str(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], str(path)
    history = rows(HERE / "history-paper/seed_metrics.csv.gz")
    assert len(history) == 3840
    grouped = defaultdict(list)
    for row in history:
        grouped[int(row["history"])].append(row)
    assert set(grouped) == {0, 1, 2, 5, 10, 15, 20, 25}
    for history_length, curve in grouped.items():
        seeds = {int(r["training_seed"]) for r in curve}
        assert len(seeds) == 10 and len(curve) == 480
        for seed in seeds:
            steps = sorted(int(r["env_interactions"]) for r in curve if int(r["training_seed"]) == seed)
            assert steps == list(range(22800, 492801, 10000)), (history_length, seed)
    for record in read(HERE / "history-paper/provenance.json")["result_artifacts"]:
        for name, item in record["files"].items():
            path = HERE / item["saved_path"]
            data = gzip.decompress(path.read_bytes()) if name == "metrics.jsonl" else path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == item["sha256"], str(path)
    learning = rows(HERE / "paper-tables/occupancy-learning.csv.gz")
    frozen = rows(HERE / "paper-tables/occupancy-frozen-fits.csv.gz")
    assert len(learning) == 27120 and len(frozen) == 2340
    report = read(HERE / "snapshots/occupancy/report/summary.json")
    for environment, block in report["environments"].items():
        for kind, data, axis, metric in (("learning", learning, "env_interactions", "greedy_return"),
                                         ("fixed_fits", frozen, "updates", "fixed_mu_discounted_heldout_discrepancy_rms")):
            for condition, summary in block[kind]["conditions"].items():
                measure, correction = condition.split("/")
                selected = [r for r in data if r["environment"] == environment and r["starting_measure"] == measure
                            and r["trace_correction"] == correction and int(r[axis]) == summary["endpoint"]]
                actual = {int(r["training_seed"]): float(r[metric]) for r in selected}
                assert actual == dict(zip(summary["seeds"], summary["seed_values"])), (environment, kind, condition)
    print("VERIFIED: full 80-seed history dataset, 3840 per-seed curve rows.")
    print("VERIFIED: all importance-sampling/occupancy conditions, 27120 learning rows and 2340 frozen-fit rows.")
    print("VERIFIED: recovered-file hashes and exact agreement with published occupancy endpoint values.")
    print("No W&B access, model checkpoints, GPU, or third-party Python packages were needed.")


if __name__ == "__main__":
    main()
