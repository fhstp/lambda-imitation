#!/usr/bin/env python3
"""Archive the final committed repository and test a remote-only recovery."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from prepare import OUT, ROOT, read, sha, write


def main():
    for variable, name in (("WANDB_DIR", "wandb-runs"), ("WANDB_CACHE_DIR", "wandb-cache"),
                           ("WANDB_DATA_DIR", "wandb-data"), ("TMPDIR", "tmp")):
        os.environ[variable] = str(OUT / name)
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    directory = ROOT / "recovery/b200-2026-11-15"
    manifest = read(directory / "manifest.json")
    assert manifest["verification"]["all_production_entries_downloaded_and_sha256_verified"]
    assert read(directory / "RESTORE_TEST.json")["passed"]
    assert read(directory / "FINAL_RESTORE_TEST.json")["passed"]
    assert read(directory / "COMPLETION_AUDIT.json")["passed"]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("Commit all recovery material before publishing the final bundle")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    kit = OUT / "final-recovery-kit"
    kit.mkdir()
    subprocess.run(["git", "bundle", "create", str(kit / "repository.bundle"), "--all"], cwd=ROOT, check=True)
    subprocess.run(["git", "bundle", "verify", str(kit / "repository.bundle")], cwd=ROOT, check=True)
    for name in ("README.md", "REMOTE.md", "manifest.json", "restore.py", "PAPER_HANDOFF.md",
                 "RESTORE_TEST.json", "FINAL_RESTORE_TEST.json", "COMPLETION_AUDIT.json"):
        shutil.copyfile(directory / name, kit / name)
    sums = {p.name: sha(p) for p in kit.iterdir()}
    (kit / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items())))
    run = wandb.init(entity="fhstp-data-intelligence-research-group", project="retrace-rebuttal-validation",
        id="migration-b200-20261115-v1", resume="allow", job_type="migration-backup")
    artifact = wandb.Artifact("b200-recovery-kit-20261115-v1", type="recovery-kit",
                             metadata={"git_commit": head, "all_checkpoint_bytes_verified": True})
    for path in sorted(kit.iterdir()):
        artifact.add_file(str(path), name=path.name, policy="immutable", skip_cache=True)
    uploaded = run.log_artifact(artifact)
    uploaded.wait()
    assert uploaded.qualified_name.endswith(":v0"), "Update REMOTE.md if a prior kit version already exists"
    remote = wandb.Api(timeout=180).artifact(uploaded.qualified_name)
    assert remote.state == "COMMITTED"
    fresh = OUT / "remote-kit-roundtrip"
    remote.download(root=str(fresh), skip_cache=True)
    for name, digest in sums.items():
        assert sha(fresh / name) == digest
    assert (fresh / "SHA256SUMS").read_bytes() == (kit / "SHA256SUMS").read_bytes()
    clone = OUT / "remote-kit-clone"
    subprocess.run(["git", "clone", "--branch", "rebuttal/occupancy-correction-2026-11-13",
                    str(fresh / "repository.bundle"), str(clone)], check=True)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip() == head
    subprocess.run(["git", "fsck", "--full"], cwd=clone, check=True)
    subprocess.run([sys.executable, "-B", str(clone / "recovery/b200-2026-11-15/verify_paper_data.py")],
                   cwd=clone, check=True)
    result = {"passed": True, "utc": datetime.now(timezone.utc).isoformat(),
              "artifact": remote.qualified_name, "git_commit": head, "sha256": sums,
              "remote_download_and_fresh_clone_verified": True,
              "paper_data_verifier_passed_in_remote_kit_clone": True, "local_deletion": False}
    write(OUT / "recovery-kit-published.json", result)
    run.summary.update({"recovery_kit": remote.qualified_name, "git_commit": head,
                        "remote_kit_roundtrip_verified": True, "local_deletion": False})
    run.finish()
    print("OFF-SERVER RECOVERY KIT VERIFIED:", remote.qualified_name, "commit", head, flush=True)


if __name__ == "__main__":
    main()
