"""Offline Swarm reference: one synthetic builder, real tests, Brain memory, replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from attestdb import AttestDB
from attestdb.mcp_tools_learning import recall, remember

TASK = "synthetic:trim-display-name"
BROKEN = "def display_name(value):\n    return value\n"
FIXED = "def display_name(value):\n    return value.strip() or 'Anonymous'\n"
ACCEPTANCE = """
from pathlib import Path
scope = {}
exec(compile(Path('solution.py').read_text(), 'solution.py', 'exec'), scope)
f = scope['display_name']
assert f(' Ada ') == 'Ada'
assert f('') == 'Anonymous'
assert f(' \\t ') == 'Anonymous'
assert f('Grace Hopper') == 'Grace Hopper'
assert f(' 李 ') == '李'
"""


def append_event(run_dir: Path, event: str, **data: object) -> None:
    """Stop on journal errors; never start work after an unsuccessful reservation."""
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **data}, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def replay(run_dir: Path) -> list[dict]:
    """Read recorded evidence only; replay never executes a builder or test."""
    with (run_dir / "events.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def verify(workspace: Path, timeout: float) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", ACCEPTANCE],
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return {"passed": result.returncode == 0, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"passed": False, "reason": "verification_timeout"}


def synthetic_builder(workspace: Path, prior: list[dict]) -> None:
    """Deterministic fixture, not a model or an autonomous coding agent."""
    (workspace / "solution.py").write_text(FIXED, encoding="utf-8")


def run_demo(
    db_path: str,
    run_dir: Path,
    *,
    builder: Callable[[Path, list[dict]], None] = synthetic_builder,
    require_review: bool = False,
    verification_timeout: float = 5,
) -> dict:
    """One attempt per unique directory; failed/interrupted attempts are never resumed."""
    if not math.isfinite(verification_timeout) or verification_timeout <= 0:
        raise ValueError("verification_timeout must be finite and positive")
    # Exclusive directory creation also rejects simultaneous or repeated use.
    run_dir.mkdir(parents=True, exist_ok=False)
    append_event(
        run_dir, "reserved", task=TASK, max_builder_attempts=1, billing_mode="offline_fixture"
    )
    try:
        workspace = run_dir / "workspace"
        workspace.mkdir()
        (workspace / "solution.py").write_text(BROKEN, encoding="utf-8")
        db = AttestDB(db_path, embedding_dim=None)
        try:
            prior = recall(db, TASK)
        finally:
            db.close()
        before = verify(workspace, verification_timeout)
        append_event(run_dir, "gap_detected", acceptance=before, prior_records=len(prior))
        if before.get("reason") == "verification_timeout":
            raise RuntimeError("Baseline verification timed out; investigate before building")
        if before["passed"]:
            raise RuntimeError("Synthetic baseline unexpectedly passed; investigate fixture drift")
        append_event(run_dir, "builder_started", attempt=1)
        builder(workspace, prior)
        evidence_hash = hashlib.sha256((workspace / "solution.py").read_bytes()).hexdigest()
        after = verify(workspace, verification_timeout)
        unchanged = (
            evidence_hash == hashlib.sha256((workspace / "solution.py").read_bytes()).hexdigest()
        )
        passed = after["passed"] and unchanged
        outcome = "passed" if passed and not require_review else "review_required"
        evidence = {
            "before": before,
            "after": after,
            "artifact_sha256": evidence_hash,
            "acceptance_sha256": hashlib.sha256(ACCEPTANCE.encode()).hexdigest(),
            "artifact_unchanged": unchanged,
            "outcome": outcome,
            "review_reason": (
                "explicit_policy" if require_review else "acceptance_failed" if not passed else None
            ),
        }
        append_event(run_dir, "verification_completed", **evidence)
        db = AttestDB(db_path, embedding_dim=None)
        try:
            memory = remember(
                db,
                TASK,
                f"Synthetic display-name task: {outcome}",
                "fix" if passed else "negative",
                evidence=evidence,
            )
        finally:
            db.close()
        append_event(run_dir, "completed", outcome=outcome, memory_claim_id=memory["claim_id"])
        return {
            "outcome": outcome,
            "builder_attempts": 1,
            "prior_records": len(prior),
            "memory_claim_id": memory["claim_id"],
            "evidence": evidence,
        }
    except Exception as error:
        append_event(run_dir, "failed", error_type=type(error).__name__)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--db", default="swarm-memory.attest")
    parser.add_argument("--require-review", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay(args.run_dir), indent=2, ensure_ascii=False))
        return 0
    result = run_demo(args.db, args.run_dir, require_review=args.require_review)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["outcome"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
