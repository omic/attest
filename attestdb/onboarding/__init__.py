"""Versioned AutoCompany onboarding resources shared by OSS and Enterprise."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

SKILL_NAME = "autocompany-setup"
KIT_ROOT = Path(__file__).parent / "skill"


def kit_files() -> dict[str, bytes]:
    """Read only the public files listed in the kit's integrity manifest."""
    manifest_bytes = (KIT_ROOT / "kit-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != 1 or not re.fullmatch(
        r"\d+\.\d+\.\d+", manifest.get("version", "")
    ):
        raise ValueError("Invalid onboarding kit manifest")
    listed = manifest.get("files")
    if not isinstance(listed, dict) or not {
        "SKILL.md",
        "references/fabric-catalog.md",
        "scripts/prepare_workspace.py",
    }.issubset(listed):
        raise ValueError("Incomplete onboarding kit manifest")
    result = {"kit-manifest.json": manifest_bytes}
    for name, expected in listed.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or "\\" in name
            or name in result
        ):
            raise ValueError("Unsafe onboarding kit path")
        source = KIT_ROOT
        for part in path.parts:
            source /= part
            if source.is_symlink():
                raise ValueError("Linked onboarding kit resource")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Onboarding kit resource changed: {name}")
        result[name] = content
    return result


def kit_version() -> str:
    """Return the version only after verifying the kit's resources."""
    return json.loads(kit_files()["kit-manifest.json"])["version"]


def skill_archive(*, enterprise_reference: bytes | None = None) -> bytes:
    """Build a reproducible skill ZIP; Enterprise may add one separate reference."""
    contents = kit_files()
    extensions = {}
    if enterprise_reference is not None:
        name = "references/enterprise.md"
        contents[name] = enterprise_reference
        extensions[name] = hashlib.sha256(enterprise_reference).hexdigest()
    contents["distribution.json"] = (
        json.dumps(
            {
                "schema_version": 1,
                "kit_version": json.loads(contents["kit-manifest.json"])["version"],
                "kit_manifest_sha256": hashlib.sha256(contents["kit-manifest.json"]).hexdigest(),
                "edition": "enterprise" if enterprise_reference is not None else "oss",
                "extensions": extensions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(contents.items()):
            info = zipfile.ZipInfo(f"{SKILL_NAME}/{name}", date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return output.getvalue()


def export_skill(destination: Path) -> None:
    """Export a complete portable skill folder without overwriting existing files."""
    contents = kit_files()
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
