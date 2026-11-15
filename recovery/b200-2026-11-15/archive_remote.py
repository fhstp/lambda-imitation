#!/usr/bin/env python3
"""Publish local-only evidence and independently download/hash every artifact."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone

from prepare import OUT, ROOT, read, sha, write
from restore import restore_one


ENTITY = "fhstp-data-intelligence-research-group"
PROJECT = "retrace-rebuttal-validation"
LOCAL = threading.local()


def main():
    for variable, name in (("WANDB_DIR", "wandb-runs"), ("WANDB_CACHE_DIR", "wandb-cache"),
                           ("WANDB_DATA_DIR", "wandb-data"), ("TMPDIR", "tmp")):
        directory = OUT / name
        directory.mkdir(exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    prepared = read(OUT / "prepared.json")
    run = wandb.init(entity=ENTITY, project=PROJECT, id="migration-b200-20261115-v1",
                     resume="allow", name="B200 complete recovery archive 2026-11-15",
                     job_type="migration-backup", config={"training_started": False, "local_deletion": False})
    archive = OUT / prepared["archive"]["entry"]
    artifact = wandb.Artifact("b200-local-evidence-20261115-v1", type="recovery-archive",
                             metadata={"sha256": sha(archive), "files": len(prepared["local_files"])})
    artifact.add_file(str(archive), name=archive.name, policy="immutable", skip_cache=True)
    remote = run.log_artifact(artifact)
    remote.wait()
    remote = wandb.Api(timeout=180).artifact(remote.qualified_name)
    entry = remote.manifest.entries[archive.name]
    with archive.open("rb") as stream:
        md5 = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
    assert remote.state == "COMMITTED" and entry.size == archive.stat().st_size and entry.digest == md5
    archive_item = {**prepared["archive"], "artifact": remote.qualified_name, "md5": md5}
    write(OUT / "archive-committed.json", archive_item)
    print("ARCHIVE COMMITTED", remote.qualified_name, flush=True)

    def verify(item):
        result = restore_one(item, OUT / "verified-downloads")
        return {"path": result, "sha256": item["sha256"], "bytes": item["bytes"], "verified": True}

    production = prepared["production"]
    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, receipt in enumerate(pool.map(verify, [archive_item, *production]), 1):
            receipts.append(receipt)
            with (OUT / "byte-verification.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt) + "\n")
            if index % 20 == 0 or index == 1:
                print("DOWNLOADED AND SHA256 VERIFIED", index, "/", len(production) + 1, flush=True)

    publications = []
    api = wandb.Api(timeout=180)
    source = api.artifact(prepared["source_artifact"])
    source_version = prepared["source_artifact"].rsplit(":", 1)[0] + ":" + source.version
    names = [*prepared["publications"], source_version]
    for name in names:
        remote = api.artifact(name)
        assert remote.state == "COMMITTED"
        key = name.rsplit("/", 1)[-1]
        for filename, entry in sorted(remote.manifest.entries.items()):
            target = "publications/" + key + "/" + filename
            path = OUT / "verified-downloads" / target
            path.parent.mkdir(parents=True, exist_ok=True)
            fetched = Path(remote.get_entry(filename).download(root=str(path.parent / ".download"), skip_cache=True))
            assert fetched.stat().st_size == entry.size
            with fetched.open("rb") as stream:
                md5 = base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()
            assert md5 == entry.digest
            item = {"artifact": remote.qualified_name, "entry": filename, "path": target,
                    "sha256": sha(fetched), "md5": md5, "bytes": entry.size,
                    "group": "publication", "kind": "publication"}
            fetched.rename(path)
            publications.append(item)
        print("PUBLICATION VERIFIED", remote.qualified_name, flush=True)
    manifest = {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All experiments performed on dgxB200; history first blocks on the other host are not part of this archive",
        "local_unpacked_bytes": prepared["local_unpacked_bytes"],
        "files": [archive_item, *production, *publications], "local_files": prepared["local_files"],
        "source_artifact": source_version,
        "verification": {"all_production_entries_downloaded_and_sha256_verified": True,
            "local_archive_downloaded_and_sha256_verified": True,
            "all_publication_entries_downloaded_and_md5_verified": True,
            "production_entries": len(production), "publication_entries": len(publications),
            "total_remote_bytes": sum(item["bytes"] for item in [archive_item, *production, *publications])},
        "local_deletion_authorized": False}
    write(ROOT / "recovery/b200-2026-11-15/manifest.json", manifest)
    write(OUT / "verification-complete.json", manifest["verification"])
    run.summary.update(manifest["verification"])
    run.finish()
    print("ALL REMOTE BYTES VERIFIED; manifest written", flush=True)


if __name__ == "__main__":
    main()
