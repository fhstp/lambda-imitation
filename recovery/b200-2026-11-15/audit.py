#!/usr/bin/env python3
"""Read-only completion/provenance audit; never invokes either queue."""

from collections import Counter
from datetime import datetime, timezone
import json

from prepare import OC, HISTORY, ROOT, read, sha, write


def main():
    import wandb

    study = read(OC / "study.json")
    admission = read(OC / "admission.json")
    for name, digest in admission["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    jobs = []
    expected = {}
    for job in study["jobs"]:
        directory = OC / "jobs" / job["name"]
        assert read(directory / "status.json")["status"] == "complete"
        assert (directory / "result-published.json").is_file()
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        if job["kind"] != "data":
            assert read(directory / "training-exit.json")["exit_code"] == 0
        if job["kind"] == "learning":
            b = study["learning"][job["environment"]]
            assert [r["round"] for r in rows if r["round"] > 0] == list(range(1, b["rounds"] + 1))
            assert rows[-1]["env_interactions"] == b["prefill"] + b["rounds"] * b["steps_per_round"]
        elif job["kind"] == "fit":
            b = study["frozen"]
            assert [r["updates"] for r in rows] == list(range(0, b["updates"] + 1, b["eval_every_updates"]))
        for name, digest in read(directory / "result-index.json")["files"].items():
            assert sha(directory / name) == digest, str(directory / name)
        expected[job["wandb_id"]] = job["project"]
        jobs.append({"name": job["name"], "kind": job["kind"], "project": job["project"],
                     "wandb_id": job["wandb_id"], "status": "complete"})
    project = "retrace-rebuttal-battleship-history-length-20261109"
    for directory in sorted((HISTORY / "jobs").iterdir()):
        receipt = read(directory / "result/receipt.json")
        assert receipt["exit_code"] == 0 and receipt["continuous_no_resume"]
        assert read(directory / "status.json")["status"] == "complete"
        assert (directory / "published.json").is_file()
        for name, digest in read(directory / "result/index.json")["files"].items():
            assert sha(directory / "result" / name) == digest
        expected[receipt["job"]["wandb_id"]] = project
        jobs.append({"name": directory.name, "kind": "history", "project": project,
                     "wandb_id": receipt["job"]["wandb_id"], "status": "complete"})
    api = wandb.Api(timeout=120)
    remote_states = {}
    for project in sorted(set(expected.values())):
        for run in api.runs(study["entity"] + "/" + project, per_page=100):
            if run.id in expected:
                assert expected[run.id] == project
                assert run.state == "finished", (run.id, run.state)
                remote_states[run.id] = run.state
    assert set(remote_states) == set(expected), set(expected) - set(remote_states)
    for job in jobs:
        job["wandb_state"] = remote_states[job["wandb_id"]]
    record = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(),
              "jobs": jobs, "counts": dict(Counter(j["kind"] for j in jobs)),
              "all_remote_runs_finished": True, "all_result_indices_verified": True,
              "occupancy_sealed_sources_verified": len(admission["source_sha256"]),
              "no_training_started_or_resumed": True, "no_local_deletion": True}
    write(ROOT / "recovery/b200-2026-11-15/COMPLETION_AUDIT.json", record)
    print("Verified local provenance and remote finished state for all", len(jobs), "jobs.")


if __name__ == "__main__":
    main()
