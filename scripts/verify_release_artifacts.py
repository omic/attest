#!/usr/bin/env python3
"""Check built distributions against the reviewed public source manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import stat
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {
    ".git",
    ".env",
    "enterprise",
    "plugins",
    "models",
    "weights",
    "training",
    "evals",
    "corpora",
}
PRIVATE_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".onnx",
    ".pkl",
    ".joblib",
    ".db",
    ".lmdb",
}
PUBLIC_HEURISTICS = {
    "attestdb/intelligence/__init__.py",
    "attestdb/intelligence/heuristic_extractor.py",
    "attestdb/intelligence/extraction_types.py",
}
NATIVE_INIT = (
    b"from .attest_rust import *\n\n__doc__ = attest_rust.__doc__\n"
    b'if hasattr(attest_rust, "__all__"):\n    __all__ = attest_rust.__all__'
)


class ArtifactError(ValueError):
    """A distribution contains unreviewed or unsafe content."""


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() != name
        or "\\" in name
        or ".." in path.parts
    ):
        raise ArtifactError(f"Unsafe archive path: {name!r}")
    parts = {p.casefold() for p in path.parts}
    if (
        parts & PRIVATE_PARTS
        or path.suffix.casefold() in PRIVATE_SUFFIXES
        or any(p.startswith(".env") or "reader_blind_labels" in p for p in parts)
        or path.name in {"curator_eval_dataset.py", "calibration_questions.py"}
    ):
        raise ArtifactError(f"Private asset path: {name}")
    if "intelligence" in parts and name not in PUBLIC_HEURISTICS:
        raise ArtifactError(f"Unapproved intelligence implementation: {name}")
    return name


def checked_manifest(root: Path, path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != 1
        or not manifest.get("version")
        or not manifest.get("files")
    ):
        raise ArtifactError("Invalid or empty release manifest")
    for name, expected in manifest["files"].items():
        safe_name(name)
        source = root / name
        if (
            source.is_symlink()
            or any(p.is_symlink() for p in source.parents if p != root and root in p.parents)
            or not source.is_file()
        ):
            raise ArtifactError(f"Missing or linked approved source: {name}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ArtifactError(f"Source changed since review: {name}")
    return manifest


def check_metadata(name: str, data: bytes, package: str, version: str) -> None:
    secret_markers = (
        b"PRIVATE KEY-----",
        b"sk_live_",
        b"github_pat_",
        b"ghp_",
        b"glpat-",
    )
    if any(marker in data for marker in secret_markers):
        raise ArtifactError(f"Sensitive metadata in {name}")
    if name.endswith(("METADATA", "PKG-INFO")):
        message = BytesParser().parsebytes(data)
        if (
            message.get("Name", "").replace("-", "_") != package
            or message.get("Version") != version
        ):
            raise ArtifactError(f"Distribution identity mismatch: {name}")
    if name.endswith(".cyclonedx.json") and json.loads(data).get("bomFormat") != "CycloneDX":
        raise ArtifactError(f"Invalid generated SBOM: {name}")


def approved_payload(root: Path, manifest: dict, name: str, data: bytes, native: bool) -> None:
    source = name
    if native:
        source = "rust/" + name
        if name in {"pyproject.toml", "LICENSE"}:
            source = "rust/attest-py/" + name
    expected = manifest["files"].get(source)
    if expected and hashlib.sha256(data).hexdigest() == expected:
        return
    # Maturin rewrites only these two manifests when making its source archive.
    if native and name in {"Cargo.toml", "pyproject.toml"} and expected:
        original = tomllib.loads((root / source).read_text())
        normalised = copy.deepcopy(original)
        if name == "Cargo.toml":
            normalised["workspace"]["members"] = [
                m for m in original["workspace"]["members"] if m != "attest-migrate"
            ]
        else:
            normalised["tool"]["maturin"]["manifest-path"] = "attest-py/Cargo.toml"
        if tomllib.loads(data.decode()) == normalised:
            return
    raise ArtifactError(f"Unreviewed file or changed payload: {name}")


def check_artifact(path: Path, root: Path, manifest: dict) -> dict:
    version = manifest["version"]
    package = next(
        (
            p
            for p in ("attestdb", "attest_py")
            if path.name.startswith(p + "-" + version + "-")
            or path.name == p + "-" + version + ".tar.gz"
        ),
        None,
    )
    if package is None:
        raise ArtifactError(f"Unexpected artifact identity: {path.name}")
    wheel = path.name.endswith(".whl")
    native = package == "attest_py"
    seen: set[str] = set()
    binaries = 0
    metadata_found = False

    def check(name: str, data: bytes) -> None:
        nonlocal binaries, metadata_found
        safe_name(name)
        if name in seen:
            raise ArtifactError(f"Duplicate archive member: {name}")
        seen.add(name)
        if name == "PKG-INFO":
            check_metadata(name, data, package, version)
            metadata_found = True
            return
        dist_info = package + "-" + version + ".dist-info/"
        if wheel and name.startswith(dist_info):
            member = name.removeprefix(dist_info)
            if member not in {
                "METADATA",
                "WHEEL",
                "RECORD",
                "entry_points.txt",
                "LICENSE",
                "licenses/LICENSE",
                "sboms/attest-py.cyclonedx.json",
            }:
                raise ArtifactError(f"Unexpected wheel metadata file: {name}")
            check_metadata(name, data, package, version)
            metadata_found |= member == "METADATA"
            if member.endswith("LICENSE") and data != (root / "LICENSE").read_bytes():
                raise ArtifactError(f"License payload mismatch: {name}")
            return
        if native and wheel:
            if name == "attest_rust/__init__.py" and data == NATIVE_INIT:
                return
            member = PurePosixPath(name)
            if (
                member.parent.as_posix() == "attest_rust"
                and member.name.startswith("attest_rust.")
                and member.suffix in {".so", ".pyd"}
            ):
                binaries += 1
                return
            raise ArtifactError(f"Unexpected native wheel payload: {name}")
        approved_payload(root, manifest, name, data, native)

    if wheel:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ArtifactError(f"Linked archive member: {member.filename}")
                if member.is_dir():
                    continue
                if member.file_size > 64 * 1024 * 1024:
                    raise ArtifactError(f"Oversized archive member: {member.filename}")
                safe_name(member.filename)
                check(member.filename, archive.read(member))
    elif path.name.endswith(".tar.gz"):
        prefix = package + "-" + version + "/"
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile() or not member.name.startswith(prefix):
                    raise ArtifactError(f"Unsafe source archive member: {member.name}")
                if member.size > 64 * 1024 * 1024:
                    raise ArtifactError(f"Oversized archive member: {member.name}")
                name = member.name.removeprefix(prefix)
                safe_name(name)
                check(name, archive.extractfile(member).read())
    else:
        raise ArtifactError(f"Unknown archive format: {path.name}")
    if not metadata_found or (native and wheel and binaries != 1):
        raise ArtifactError(f"Incomplete artifact: {path.name}")
    return {
        "artifact": path.name,
        "files": len(seen),
        "native_binaries": binaries,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        manifest = checked_manifest(ROOT, ROOT / "release-manifest.json")
        for path in args.artifacts:
            print(json.dumps(check_artifact(path, ROOT, manifest), sort_keys=True))
    except (ArtifactError, OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Artifact check failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
