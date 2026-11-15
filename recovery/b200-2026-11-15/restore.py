#!/usr/bin/env python3
"""Restore archived experiment bytes. Never imports JAX or loads pickle files."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError("Unsafe archive path")
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes destination")
    return path


def restore_one(item, destination):
    import wandb

    path = inside(destination, item["path"])
    if path.exists():
        if path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]:
            return item["path"]
        raise FileExistsError("Refusing to overwrite different bytes: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    # A separate API client per thread avoids sharing mutable SDK state.
    artifact = wandb.Api(timeout=180).artifact(item["artifact"])
    if artifact.state != "COMMITTED":
        raise RuntimeError("Artifact not committed: " + item["artifact"])
    entry = artifact.manifest.entries[item["entry"]]
    if entry.size != item["bytes"] or entry.digest != item["md5"]:
        raise RuntimeError("Remote manifest changed: " + item["artifact"])
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=path.parent) as temporary:
        downloaded = Path(artifact.get_entry(item["entry"]).download(
            root=temporary, skip_cache=True))
        if downloaded.stat().st_size != item["bytes"] or digest(downloaded) != item["sha256"]:
            raise RuntimeError("Downloaded byte hash mismatch: " + item["artifact"])
        downloaded.rename(path)
    return item["path"]


def extract(archive, destination):
    """Extract only ordinary files/directories; never follow archive links."""
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            path = inside(destination, member.name)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if path.exists():
                    if digest(path) != hashlib.file_digest(source, "sha256").hexdigest():
                        raise FileExistsError(path)
                else:
                    with path.open("xb") as output:
                        shutil.copyfileobj(source, output)
                path.chmod(member.mode & 0o777)
            else:
                raise ValueError("Archive links/special files are not supported")


def restore_snapshots(destination):
    import gzip

    snapshots = HERE / "snapshots"
    for item in json.loads((snapshots / "index.json").read_text()):
        source = inside(snapshots, item["snapshot"])
        if digest(source) != item["sha256"]:
            raise RuntimeError("Git snapshot hash mismatch: " + item["snapshot"])
        relative = item["snapshot"].replace("occupancy/", "occupancy-state/", 1) if item["snapshot"].startswith("occupancy/") else item["snapshot"].replace("history/", "history-state/", 1)
        compressed = relative.endswith(".jsonl.gz")
        path = inside(destination, relative[:-3] if compressed else relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with (gzip.open(source, "rb") if compressed else source.open("rb")) as stream, path.open("xb") as output:
                shutil.copyfileobj(stream, output)
        if path.stat().st_size != item["original_bytes"] or digest(path) != item["original_sha256"]:
            raise RuntimeError("Snapshot restoration mismatch (existing files were not overwritten): " + str(path))
    print("Git-only results restored and hash verified; no cloud access was required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checkpoints", action="store_true", help="Also download all production checkpoints")
    parser.add_argument("--all-artifacts", action="store_true", help="Also download supplementary publication artifacts")
    parser.add_argument("--snapshots-only", action="store_true", help="Restore compact results from Git without W&B")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    if args.snapshots_only:
        restore_snapshots(args.destination)
        return
    manifest = json.loads(args.manifest.read_text())
    selected = [item for item in manifest["files"] if
                item["group"] == "archive" or
                (args.checkpoints and item["group"] == "production") or
                (args.all_artifacts and item["group"] == "publication")]
    needed = sum(item["bytes"] for item in selected
                 if not inside(args.destination, item["path"]).exists())
    reserve = manifest["local_unpacked_bytes"] + 2 * 2**30
    if shutil.disk_usage(args.destination).free < needed + reserve:
        raise RuntimeError("Insufficient free disk for downloads, extraction and 2 GiB headroom")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for path in pool.map(lambda item: restore_one(item, args.destination), selected):
            print("VERIFIED", path, flush=True)
    for item in selected:
        if item["group"] == "archive":
            extract(inside(args.destination, item["path"]), args.destination)
    for item in manifest["local_files"]:
        path = inside(args.destination, item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise RuntimeError("Restored local evidence mismatch: " + item["path"])
    print("Restoration verified. No training was started; no pickle was loaded.")


if __name__ == "__main__":
    main()
