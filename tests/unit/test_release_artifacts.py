"""Exercise the release gate with actual source files and package archives."""

import hashlib
import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/verify_release_artifacts.py"
spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def release(tmp_path):
    source = tmp_path / "attestdb/__init__.py"
    source.parent.mkdir()
    source.write_bytes(b'__version__ = "0.2.1"\n')
    manifest = {
        "schema_version": 1,
        "version": "0.2.1",
        "files": {
            "attestdb/__init__.py": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "release-manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def wheel(root, extra=(), metadata=b"Name: attestdb\nVersion: 0.2.1\n"):
    path = root / "attestdb-0.2.1-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("attestdb/__init__.py", (root / "attestdb/__init__.py").read_bytes())
        archive.writestr("attestdb-0.2.1.dist-info/METADATA", metadata)
        for name, content in extra:
            archive.writestr(name, content)
    return path


def test_reviewed_source_and_wheel_pass(release):
    root, manifest = release
    assert gate.checked_manifest(root, root / "release-manifest.json") == manifest
    result = gate.check_artifact(wheel(root), root, manifest)
    assert result["files"] == 2
    assert len(result["sha256"]) == 64


def test_changed_source_fails(release):
    root, manifest = release
    (root / "attestdb/__init__.py").write_text("changed")
    with pytest.raises(gate.ArtifactError, match="Source changed"):
        gate.checked_manifest(root, root / "release-manifest.json")
    with pytest.raises(gate.ArtifactError, match="changed payload"):
        gate.check_artifact(wheel(root), root, manifest)


@pytest.mark.parametrize(
    "name",
    [
        "../secret.py",
        "/secret.py",
        "attestdb/intelligence/private.py",
        "enterprise/connectors.py",
        "models/private.safetensors",
        ".env.production",
        "tests/fixtures/reader_blind_labels.json",
        "unknown.py",
    ],
)
def test_unapproved_payload_fails(release, name):
    root, manifest = release
    with pytest.raises(gate.ArtifactError):
        gate.check_artifact(wheel(root, [(name, b"secret")]), root, manifest)


@pytest.mark.parametrize(
    "metadata",
    [
        b"Name: other\nVersion: 0.2.1\n",
        b"Name: attestdb\nVersion: 0.2.0\n",
        b"Name: attestdb\nVersion: 0.2.1\n\ngithub_pat_example",
    ],
)
def test_bad_metadata_fails(release, metadata):
    root, manifest = release
    with pytest.raises(gate.ArtifactError):
        gate.check_artifact(wheel(root, metadata=metadata), root, manifest)


def test_linked_archive_fails(release):
    root, manifest = release
    path = root / "attestdb-0.2.1.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("attestdb-0.2.1/attestdb/__init__.py")
        info.type = tarfile.SYMTYPE
        info.linkname = "/private/secret"
        archive.addfile(info)
    with pytest.raises(gate.ArtifactError, match="Unsafe source archive"):
        gate.check_artifact(path, root, manifest)


def test_reviewed_source_archive_passes(release):
    root, manifest = release
    path = root / "attestdb-0.2.1.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [
            ("attestdb/__init__.py", (root / "attestdb/__init__.py").read_bytes()),
            ("PKG-INFO", b"Name: attestdb\nVersion: 0.2.1\n"),
        ]:
            info = tarfile.TarInfo("attestdb-0.2.1/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    assert gate.check_artifact(path, root, manifest)["files"] == 2


def test_linked_source_parent_fails(release):
    root, manifest = release
    source = root / "attestdb"
    source.rename(root / "real_source")
    source.symlink_to(root / "real_source", target_is_directory=True)
    with pytest.raises(gate.ArtifactError, match="linked approved source"):
        gate.checked_manifest(root, root / "release-manifest.json")
