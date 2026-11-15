#!/usr/bin/env python3
"""Generate compact, byte-preserving Git snapshots of results and provenance."""

import gzip
import json
from pathlib import Path
import shutil

from prepare import OC, HISTORY, ROOT, sha, write


def main():
    destination = ROOT / "recovery/b200-2026-11-15/snapshots"
    index = []

    def copy(path, target, compressed=False):
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        if compressed:
            with path.open("rb") as source, output.open("xb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                    shutil.copyfileobj(source, stream)
        else:
            shutil.copyfile(path, output)
        index.append({"snapshot": str(output.relative_to(destination)), "sha256": sha(output),
                      "original_sha256": sha(path), "original_bytes": path.stat().st_size})

    for state, prefix in ((OC, "occupancy"), (HISTORY, "history")):
        for path in sorted(state.glob("*.json")):
            copy(path, prefix + "/" + path.name)
        for path in sorted((state / "jobs").rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in (".json", ".committed"):
                copy(path, prefix + "/" + str(path.relative_to(state)))
            elif path.name == "metrics.jsonl":
                copy(path, prefix + "/" + str(path.relative_to(state)) + ".gz", compressed=True)
        for path in sorted((state / "report").glob("*")):
            if path.is_file():
                copy(path, prefix + "/report/" + path.name)
        # Compact preflight receipts stay readable in Git; large validation
        # checkpoints and complete logs are in the local-evidence archive.
        for subdir in ("validation", "native"):
            for path in sorted((state / subdir).glob("*")):
                if path.is_file() and path.suffix in (".json", ".xml"):
                    copy(path, prefix + "/" + str(path.relative_to(state)))
    write(destination / "index.json", index)
    print("Git snapshots:", len(index), "files,", sum(p.stat().st_size for p in destination.rglob("*") if p.is_file()), "bytes")


if __name__ == "__main__":
    main()
