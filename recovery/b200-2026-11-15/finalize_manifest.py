#!/usr/bin/env python3
"""Pin the source alias to its verified immutable version in generated data."""

import json
import os
import re
import shutil

from prepare import OUT, ROOT, read, sha


def main():
    os.environ["WANDB_SILENT"] = "true"
    import wandb

    path = ROOT / "recovery/b200-2026-11-15/manifest.json"
    manifest = read(path)
    alias = manifest["source_artifact"]
    api = wandb.Api(timeout=120)
    artifact = api.artifact(alias)
    version = alias.rsplit(":", 1)[0] + ":" + artifact.version
    pinned = api.artifact(version)
    assert artifact.id == pinned.id and pinned.state == "COMMITTED"
    changed = 0
    for item in manifest["files"]:
        if item["artifact"] == alias:
            entry = pinned.manifest.entries[item["entry"]]
            assert entry.size == item["bytes"] and entry.digest == item["md5"]
            assert sha(OUT / "verified-downloads" / item["path"]) == item["sha256"]
            item["artifact"] = version
            changed += 1
        assert re.search(r":v\d+$", item["artifact"]), item["artifact"]
    assert changed == len(pinned.manifest.entries)
    manifest["source_artifact"] = version
    manifest["verification"]["all_remote_references_pinned_to_immutable_versions"] = True
    previous = OUT / "manifest-before-source-version-pin.json"
    if previous.exists():
        raise FileExistsError(previous)
    shutil.copyfile(path, previous)
    temporary = path.with_suffix(".final.tmp")
    with temporary.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)
    print("Pinned", changed, "source entries to", version)


if __name__ == "__main__":
    main()
