"""Real subprocess acceptance checks and LMDB; no live providers or mocked DB."""

import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from attestdb import AttestDB
from attestdb.mcp_tools_learning import recall

PATH = Path(__file__).resolve().parents[2] / "examples/swarm_reference.py"
SPEC = importlib.util.spec_from_file_location("swarm_reference", PATH)
swarm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(swarm)


def test_complete_loop_replay_and_brain_reopen(tmp_path):
    db_path = str(tmp_path / "memory.attest")
    run_dir = tmp_path / "first"
    result = swarm.run_demo(db_path, run_dir)
    assert result["outcome"] == "passed"
    assert result["builder_attempts"] == 1
    assert not result["evidence"]["before"]["passed"]
    assert result["evidence"]["after"]["passed"]
    assert result["evidence"]["artifact_unchanged"]
    events = swarm.replay(run_dir)
    assert [event["event"] for event in events] == [
        "reserved",
        "gap_detected",
        "builder_started",
        "verification_completed",
        "completed",
    ]
    db = AttestDB(db_path, embedding_dim=None)
    try:
        records = recall(db, swarm.TASK)
        assert records[0]["claim_id"] == result["memory_claim_id"]
        assert records[0]["verification"] == "agent_reported"
        assert records[0]["evidence"] == result["evidence"]
    finally:
        db.close()

    def builder(workspace, prior):
        assert prior[0]["claim_id"] == result["memory_claim_id"]
        swarm.synthetic_builder(workspace, prior)

    second = swarm.run_demo(db_path, tmp_path / "second", builder=builder)
    assert second["prior_records"] == 1
    # Replay reads evidence even if code artifacts have subsequently disappeared.
    (run_dir / "workspace/solution.py").unlink()
    assert swarm.replay(run_dir) == events


@pytest.mark.parametrize("broken", [True, False])
def test_failed_acceptance_or_explicit_review_never_retries(tmp_path, broken):
    calls = []

    def builder(workspace, prior):
        calls.append(workspace)
        if not broken:
            swarm.synthetic_builder(workspace, prior)

    result = swarm.run_demo(
        str(tmp_path / "memory.attest"),
        tmp_path / "run",
        builder=builder,
        require_review=not broken,
    )
    assert result["outcome"] == "review_required"
    assert result["evidence"]["review_reason"] == (
        "acceptance_failed" if broken else "explicit_policy"
    )
    assert len(calls) == 1


def test_crashed_builder_consumes_attempt_and_surfaces_error(tmp_path):
    run_dir = tmp_path / "run"
    calls = []

    def crash(workspace, prior):
        calls.append(workspace)
        raise RuntimeError("fixture crash")

    with pytest.raises(RuntimeError, match="fixture crash"):
        swarm.run_demo(str(tmp_path / "memory.attest"), run_dir, builder=crash)
    with pytest.raises(FileExistsError):
        swarm.run_demo(str(tmp_path / "memory.attest"), run_dir, builder=crash)
    assert len(calls) == 1
    assert swarm.replay(run_dir)[-1] == {"event": "failed", "error_type": "RuntimeError"}


def test_existing_reservation_prevents_builder_launch(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(FileExistsError):
        swarm.run_demo(
            str(tmp_path / "memory.attest"),
            run_dir,
            builder=lambda *_: pytest.fail("must not launch"),
        )


def test_verifier_timeout_is_not_success(tmp_path):
    (tmp_path / "solution.py").write_text("import time\ntime.sleep(2)\n")
    assert swarm.verify(tmp_path, timeout=0.05) == {
        "passed": False,
        "reason": "verification_timeout",
    }


def test_simultaneous_launches_share_one_reservation(tmp_path):
    calls = []

    def builder(workspace, prior):
        calls.append(workspace)
        swarm.synthetic_builder(workspace, prior)

    def launch():
        try:
            return swarm.run_demo(
                str(tmp_path / "memory.attest"), tmp_path / "run", builder=builder
            )["outcome"]
        except FileExistsError:
            return "already_reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: launch(), range(2))) == ["already_reserved", "passed"]
    assert len(calls) == 1


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_limits_rejected_before_reservation(tmp_path, timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        swarm.run_demo(
            str(tmp_path / "memory.attest"), tmp_path / "run", verification_timeout=timeout
        )
    assert not (tmp_path / "run").exists()


def test_baseline_timeout_stops_before_builder(tmp_path, monkeypatch):
    monkeypatch.setattr(swarm, "ACCEPTANCE", "import time; time.sleep(2)")
    with pytest.raises(RuntimeError, match="Baseline verification timed out"):
        swarm.run_demo(
            str(tmp_path / "memory.attest"),
            tmp_path / "run",
            verification_timeout=0.05,
            builder=lambda *_: pytest.fail("must not launch"),
        )
    assert "builder_started" not in {event["event"] for event in swarm.replay(tmp_path / "run")}


def test_cli_review_exit_and_read_only_replay(tmp_path):
    command = [sys.executable, "-I", str(PATH), "--run-dir", str(tmp_path / "run")]
    result = subprocess.run(
        command + ["--db", str(tmp_path / "memory.attest"), "--require-review"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 2, result.stderr
    assert json.loads(result.stdout)["outcome"] == "review_required"
    replayed = subprocess.run(
        command + ["--replay"], capture_output=True, text=True, check=False, timeout=15
    )
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout)[-1]["outcome"] == "review_required"
    assert not (Path.cwd() / "swarm-memory.attest").exists()
