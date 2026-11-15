#!/usr/bin/env python3
"""One-off migration packaging. Does not edit sealed scientific files."""

import base64
import hashlib
import importlib.metadata
import json
import netrc
import os
from pathlib import Path
import platform
import subprocess
import tarfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/raid/ustp/migration-2026-11-15")
OC = Path("/raid/ustp/experiments/occupancy-correction-2026-11-13")
HISTORY = Path("/home/ustp/.local/state/lambda-history-b200/32df839bad43f7e9")
ORIGINAL = Path("/raid/ustp/git/lambda-imitation")
PAPER = Path("/raid/ustp/git/6aa299d71f7b5eb457c918de")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "prepared.json").exists():
        raise FileExistsError("Preserve the completed migration preparation")
    # Inspect secrets only to reject accidental inclusion; never print/store them.
    secrets = [auth[2].encode() for auth in netrc.netrc().hosts.values()
               if auth[2] and len(auth[2]) >= 16]
    secrets += [v.encode() for k, v in os.environ.items()
                if any(t in k.upper() for t in ("TOKEN", "API_KEY", "PASSWORD")) and len(v) >= 16]
    files = []
    exclusions = []
    links = []

    def collect(source, prefix, omit=()):
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if any(part in omit for part in rel.parts):
                if path.is_dir() and len(rel.parts) == 1:
                    exclusions.append({"path": str(path), "reason": "environment/cache/git internals; separate bundles and runtime pins"})
                continue
            target = str(Path(prefix) / rel)
            if path.is_symlink():
                links.append({"path": target, "target": os.readlink(path), "reason": "nonessential runtime alias; not followed"})
            elif path.is_file():
                if path.name in (".netrc", ".env"):
                    raise RuntimeError("Credential-like file in project scope: " + str(path))
                with path.open("rb") as stream:
                    tail = b""
                    while block := stream.read(4 * 2**20):
                        combined = tail + block
                        if any(secret in combined for secret in secrets):
                            raise RuntimeError("Credential value found; inspect privately: " + str(path))
                        tail = combined[-4096:]
                files.append((path, target))

    collect(OC, "occupancy-state", ("__pycache__", ".pytest_cache"))
    collect(HISTORY, "history-state", ("__pycache__", ".pytest_cache"))
    collect(ORIGINAL, "history-checkout", (".git", ".venv-b200", ".pytest_cache", "__pycache__"))
    collect(ROOT, "occupancy-checkout", (".git", ".pytest_cache", "__pycache__", "recovery"))
    collect(PAPER, "paper-checkout", (".git", "__pycache__"))
    temporary = Path("/tmp/opencode")
    if temporary.exists():
        collect(temporary, "preflight-temporary-evidence", ("__pycache__",))
    # Git bundles are portable and contain no remote URL/configuration/credentials.
    for checkout, name in ((ROOT, "lambda-imitation.bundle"), (PAPER, "paper.bundle")):
        bundle = OUT / name
        subprocess.run(["git", "bundle", "create", str(bundle), "--all"], cwd=checkout, check=True)
        subprocess.run(["git", "bundle", "verify", str(bundle)], cwd=checkout, check=True)
        files.append((bundle, "git/" + name))
    runtime = OUT / "runtime.json"
    write(runtime, {"python": platform.python_version(), "platform": platform.platform(),
                    "packages": sorted(set(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))})
    files.append((runtime, "runtime/runtime.json"))
    inventory = [{"path": target, "bytes": path.stat().st_size, "sha256": sha(path)} for path, target in files]
    write(OUT / "local-inventory.json", inventory)
    write(OUT / "exclusions.json", {"excluded": exclusions, "symlinks": links,
          "global_exclusions": ["credentials, shell history, machine-global config/safety instructions, OpenCode databases",
                                "installed venv/cache/driver binaries (recorded package versions instead)"]})
    files.extend([(OUT / "local-inventory.json", "local-inventory.json"),
                  (OUT / "exclusions.json", "exclusions.json")])
    archive = OUT / "local-evidence.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=1) as stream:
        for path, target in files:
            stream.add(path, arcname=target, recursive=False)
    # The remote index supplies exact restore locations and independent byte hashes.
    production = []
    for job in read(OC / "study.json")["jobs"]:
        directory = OC / "jobs" / job["name"]
        for pointer in sorted((directory / "uploads").glob("*.json")):
            p, c = read(pointer), read(pointer.with_suffix(".committed"))
            assert p["sha256"] == c["sha256"] and p["bytes"] == c["bytes"] and c["remote_committed_verified"]
            production.append({"artifact": c["artifact"], "entry": Path(p["path"]).name,
                "path": "occupancy-state/jobs/" + job["name"] + "/" + p["path"],
                "sha256": c["sha256"], "md5": c["md5"], "bytes": c["bytes"],
                "kind": p["kind"], "group": "production"})
    for directory in sorted((HISTORY / "jobs").iterdir()):
        for pointer in sorted((directory / "result").glob("*.archived.json")):
            p = read(pointer)
            production.append({"artifact": p["artifact"], "entry": p["name"],
                "path": "history-state/jobs/" + directory.name + "/checkpoints/" + p["name"],
                "sha256": p["sha256"], "md5": p["md5"], "bytes": p["bytes"],
                "kind": "checkpoint", "group": "production"})
    publications = []
    for pointer in sorted(OC.glob("jobs/*/result-published.json")):
        publications.append(read(pointer)["artifact"])
    publications.append(read(OC / "report/published.json")["artifact"])
    for pointer in sorted(HISTORY.glob("jobs/*/published.json")):
        publications.append(read(pointer)["artifact"])
    write(OUT / "prepared.json", {"utc": datetime.now(timezone.utc).isoformat(),
        "production": production, "publications": publications,
        "source_artifact": "fhstp-data-intelligence-research-group/retrace-rebuttal-validation/oc-source-20261113-v1:latest",
        "archive": {"entry": archive.name, "sha256": sha(archive), "bytes": archive.stat().st_size,
                    "path": "archives/" + archive.name, "group": "archive", "kind": "local-evidence"},
        "local_files": inventory, "local_unpacked_bytes": sum(item["bytes"] for item in inventory)})
    print("Prepared", len(inventory), "files;", len(production), "production artifact entries.", flush=True)


if __name__ == "__main__":
    main()
