"""The installed kit exports real, portable resources without a domain service."""

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from attestdb.onboarding import export_skill, kit_files, kit_version, skill_archive


def test_zip_is_reproducible_complete_and_records_integrity():
    content = skill_archive()
    assert content == skill_archive()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        files = {
            name.removeprefix("autocompany-setup/"): archive.read(name)
            for name in archive.namelist()
        }
    distribution = json.loads(files.pop("distribution.json"))
    assert files == kit_files()
    assert distribution["kit_version"] == kit_version()
    assert distribution["edition"] == "oss"
    assert distribution["extensions"] == {}
    assert (
        distribution["kit_manifest_sha256"]
        == hashlib.sha256(files["kit-manifest.json"]).hexdigest()
    )


def test_exported_skill_creates_a_versioned_private_workspace(tmp_path):
    skill = tmp_path / "skill"
    export_skill(skill)
    workspace = tmp_path / "company"
    result = subprocess.run(
        [
            sys.executable,
            str(skill / "scripts/prepare_workspace.py"),
            "--root",
            str(workspace),
            "--company",
            "Example company",
            "--fabrics",
            "memory",
            "engineering",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads(result.stdout)
    assert manifest["onboarding_kit_version"] == kit_version()
    assert manifest["automation"]["state"] == "paused"
    assert {item["connection"] for item in manifest["fabrics"].values()} == {"not_checked"}
    assert workspace.joinpath("company.json").stat().st_mode & 0o077 == 0
    original = skill.joinpath("SKILL.md").read_bytes()
    with pytest.raises(FileExistsError):
        export_skill(skill)
    assert skill.joinpath("SKILL.md").read_bytes() == original


def test_cli_exports_a_zip_and_refuses_overwrite(tmp_path):
    target = tmp_path / "kit.zip"
    command = [sys.executable, "-m", "attestdb.onboarding", "--zip", str(target)]
    subprocess.run(command, check=True, capture_output=True)
    original = target.read_bytes()
    assert original == skill_archive()
    refused = subprocess.run(command, capture_output=True)
    assert refused.returncode != 0
    assert target.read_bytes() == original


def test_changed_resource_cannot_be_distributed(tmp_path):
    # Load an actual isolated copy so the maintained kit remains untouched.
    source = Path(__file__).resolve().parents[2] / "attestdb/onboarding"
    copy = tmp_path / "onboarding"
    shutil.copytree(source, copy, ignore=shutil.ignore_patterns("__pycache__"))
    copy.joinpath("skill/SKILL.md").write_text("Changed without a versioned manifest\n")
    spec = importlib.util.spec_from_file_location("changed_kit", copy / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="resource changed"):
        module.skill_archive()
