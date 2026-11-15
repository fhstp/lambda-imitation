"""Recovery safety tests; no cloud connection, GPU, or scientific execution."""

import hashlib
import importlib.util
import io
from pathlib import Path
import tarfile
import types

import pytest


SPEC = importlib.util.spec_from_file_location("recovery_restore", Path(__file__).with_name("restore.py"))
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def archive(path, name="evidence/metric.json", data=b"{}", kind=None):
    with tarfile.open(path, "w:gz") as stream:
        entry = tarfile.TarInfo(name)
        if kind is not None:
            entry.type = kind
            entry.linkname = "outside"
            stream.addfile(entry)
        else:
            entry.size = len(data)
            stream.addfile(entry, io.BytesIO(data))


def test_safe_extract_is_idempotent_and_preserves_existing_bytes(tmp_path):
    source = tmp_path / "evidence.tar.gz"
    archive(source)
    target = tmp_path / "restored"
    restore.extract(source, target)
    restore.extract(source, target)
    path = target / "evidence/metric.json"
    assert path.read_bytes() == b"{}"
    path.write_bytes(b"user work")
    with pytest.raises(FileExistsError):
        restore.extract(source, target)
    assert path.read_bytes() == b"user work"


@pytest.mark.parametrize("name", ["../outside", "/absolute", "evidence/../../outside"])
def test_archive_path_traversal_rejected(tmp_path, name):
    source = tmp_path / "bad.tar.gz"
    archive(source, name=name)
    with pytest.raises(ValueError):
        restore.extract(source, tmp_path / "restored")


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_archive_links_and_special_files_rejected(tmp_path, kind):
    source = tmp_path / "bad.tar.gz"
    archive(source, kind=kind)
    with pytest.raises(ValueError):
        restore.extract(source, tmp_path / "restored")


def test_existing_external_symlink_is_not_followed(tmp_path):
    target = tmp_path / "restored"
    target.mkdir()
    (target / "evidence").symlink_to(tmp_path / "outside", target_is_directory=True)
    source = tmp_path / "bad.tar.gz"
    archive(source)
    with pytest.raises(ValueError):
        restore.extract(source, target)


def test_download_verifies_bytes_and_refuses_overwrite(tmp_path, monkeypatch):
    content = b"trusted bytes"
    item = {"path": "job/checkpoint.pkl", "entry": "checkpoint.pkl", "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(), "md5": "expected", "artifact": "entity/project/a:v0"}

    def download(root, skip_cache):
        path = Path(root) / item["entry"]
        path.write_bytes(content)
        return str(path)

    remote = types.SimpleNamespace(state="COMMITTED",
        manifest=types.SimpleNamespace(entries={item["entry"]: types.SimpleNamespace(size=len(content), digest="expected")}),
        get_entry=lambda entry: types.SimpleNamespace(download=download))
    monkeypatch.setitem(__import__("sys").modules, "wandb",
                        types.SimpleNamespace(Api=lambda **kwargs: types.SimpleNamespace(artifact=lambda name: remote)))
    assert restore.restore_one(item, tmp_path) == item["path"]
    assert restore.restore_one(item, tmp_path) == item["path"]
    path = tmp_path / item["path"]
    assert path.read_bytes() == content
    path.write_bytes(b"user work")
    with pytest.raises(FileExistsError):
        restore.restore_one(item, tmp_path)
    assert path.read_bytes() == b"user work"


def test_bad_download_is_never_installed(tmp_path, monkeypatch):
    content = b"bad"
    item = {"path": "job/checkpoint.pkl", "entry": "checkpoint.pkl", "bytes": len(content),
            "sha256": "0" * 64, "md5": "expected", "artifact": "entity/project/a:v0"}

    def download(root, skip_cache):
        path = Path(root) / item["entry"]
        path.write_bytes(content)
        return str(path)

    remote = types.SimpleNamespace(state="COMMITTED",
        manifest=types.SimpleNamespace(entries={item["entry"]: types.SimpleNamespace(size=len(content), digest="expected")}),
        get_entry=lambda entry: types.SimpleNamespace(download=download))
    monkeypatch.setitem(__import__("sys").modules, "wandb",
                        types.SimpleNamespace(Api=lambda **kwargs: types.SimpleNamespace(artifact=lambda name: remote)))
    with pytest.raises(RuntimeError, match="byte hash mismatch"):
        restore.restore_one(item, tmp_path)
    assert not (tmp_path / item["path"]).exists()
