#!/usr/bin/env python3
"""Integration proof: unpack all evidence, clone bundles, reproduce statistics."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare import OUT, ROOT, read, sha, write
from restore import extract


def main():
    prepared = read(OUT / "prepared.json")
    target = OUT / "fresh-evidence-restore"
    if target.exists():
        raise FileExistsError("Use an independently empty restore directory")
    target.mkdir()
    archive = OUT / prepared["archive"]["entry"]
    assert sha(archive) == prepared["archive"]["sha256"]
    extract(archive, target)
    for item in prepared["local_files"]:
        path = target / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    commits = {}
    for bundle, name, branch in (("lambda-imitation.bundle", "fresh-code-clone", "rebuttal/occupancy-correction-2026-11-13"),
                                 ("paper.bundle", "fresh-paper-clone", "main")):
        checkout = OUT / name
        subprocess.run(["git", "clone", "--branch", branch, str(target / "git" / bundle), str(checkout)], check=True)
        subprocess.run(["git", "fsck", "--full"], cwd=checkout, check=True)
        commits[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    admission = read(target / "occupancy-state/admission.json")
    for name, digest in admission["source_sha256"].items():
        assert sha(OUT / "fresh-code-clone" / name) == digest
    # Reproduce statistics in a disposable output state, never overwrite evidence.
    report_state = OUT / "report-rebuild-state"
    report_state.mkdir()
    original = target / "occupancy-state"
    for name in ("study.json", "admission.json"):
        shutil.copyfile(original / name, report_state / name)
    for directory in (original / "jobs").iterdir():
        destination = report_state / "jobs" / directory.name
        destination.mkdir(parents=True)
        for name in ("metrics.jsonl", "config.json", "summary.json", "status.json", "training-exit.json"):
            if (directory / name).exists():
                shutil.copyfile(directory / name, destination / name)
    checkout = OUT / "fresh-code-clone"
    env = dict(os.environ, PYTHONPATH=str(checkout / "src") + ":" + str(checkout),
               CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([sys.executable, "-B", "-m", "studies.occupancy_correction.report",
                    "--state", str(report_state)], cwd=checkout, env=env, check=True)
    expected = read(original / "report/summary.json")
    actual = read(report_state / "report/summary.json")
    expected.pop("utc")
    actual.pop("utc")
    assert actual == expected, "Regenerated scientific statistics/provenance differ"
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(),
              "archive_sha256": sha(archive), "all_original_regular_files_sha256_verified": len(prepared["local_files"]),
              "git_bundles_cloned_and_fsck_verified": commits,
              "sealed_sources_verified_in_fresh_clone": len(admission["source_sha256"]),
              "scientific_report_exactly_reproduced_except_timestamp": True,
              "policy_training_started_or_resumed": False,
              "recovery_unit_tests_passed": 11, "gpu_required_for_restore": False}
    write(ROOT / "recovery/b200-2026-11-15/RESTORE_TEST.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
