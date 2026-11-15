"""Study-owned atomic records and checkpoints; no training-resume entry point."""

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle


def utc(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, data, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def save_pickle(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return {"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size}


def load_owned_pickle(path, expected_sha):
    """Only load this study's trusted, hash-verified artifacts, never user inputs."""
    path = Path(path)
    if sha(path) != expected_sha: raise RuntimeError("checkpoint/reference hash mismatch")
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def enqueue_upload(directory, path, *, kind, release=False):
    directory, path = Path(directory), Path(path)
    relative = path.relative_to(directory)
    record = {"path": str(relative), "sha256": sha(path), "bytes": path.stat().st_size,
              "kind": kind, "release_after_remote_verification": release, "utc": utc()}
    write(directory/"uploads"/(path.name+".json"), record, exclusive=True)
    return record


def verify_sources(root, admission):
    for name, expected in admission["source_sha256"].items():
        if sha(Path(root)/name) != expected:
            raise RuntimeError("sealed new-study source changed: "+name)
