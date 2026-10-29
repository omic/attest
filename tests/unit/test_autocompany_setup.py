import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "attestdb/onboarding/skill/scripts/prepare_workspace.py"
)
spec = importlib.util.spec_from_file_location("company_setup", SOURCE)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def test_workspace_is_private_paused_and_never_claims_connectivity(tmp_path):
    result = setup.prepare(
        tmp_path, "Example company", ["engineering", "memory"], "First milestone"
    )
    assert result["setup_state"] == "workspace_created"
    assert result["automation"]["state"] == "paused"
    assert all(fabric["connection"] == "not_checked" for fabric in result["fabrics"].values())
    assert tmp_path.joinpath("company.json").stat().st_mode & 0o077 == 0
    assert result == json.loads(tmp_path.joinpath("company.json").read_text())


def test_repeat_setup_is_idempotent_and_preserves_existing_workspace(tmp_path):
    setup.prepare(tmp_path, "Company", ["engineering"])
    original = tmp_path.joinpath("company.json").read_bytes()
    setup.prepare(tmp_path, "Company", ["engineering"])
    assert tmp_path.joinpath("company.json").read_bytes() == original
    with pytest.raises(ValueError, match="Existing workspace differs"):
        setup.prepare(tmp_path, "Changed company", ["engineering"])
    assert tmp_path.joinpath("company.json").read_bytes() == original


def test_unknown_fabric_does_not_create_a_workspace(tmp_path):
    destination = tmp_path / "new-company"
    with pytest.raises(ValueError):
        setup.prepare(destination, "Company", ["nonexistent-service"])
    assert not destination.exists()


def test_manifest_symlink_is_not_followed(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"private":"original"}')
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "company.json").symlink_to(source)
    with pytest.raises(ValueError, match="symbolic link"):
        setup.prepare(root, "Company", ["memory"])
    assert json.loads(source.read_text()) == {"private": "original"}
