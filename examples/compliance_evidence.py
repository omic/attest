"""Synthetic evidence-coverage example, not a regulatory compliance assessment."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from attestdb import AttestDB
from attestdb.core.types import ClaimStatus

CONTROLS = ("demo:access-review", "demo:backup-restore")


def uncovered(db: AttestDB) -> list[str]:
    """An obligation stays uncovered unless an active evidence link exists."""
    return [control for control in CONTROLS if not any(
        claim.predicate.id == "supported_by" and claim.status == ClaimStatus.ACTIVE
        for claim in db.claims_for(control, limit=100)
    )]


def run_demo(db_path: str) -> dict:
    db = AttestDB(db_path, embedding_dim=None)
    try:
        for control in CONTROLS:
            db.ingest(
                subject=(control, "control"), predicate=("requires", "relates_to"),
                object=("documented evidence", "requirement"), confidence=0.5,
                provenance={"source_id": "synthetic-demo-policy", "source_type": "manual"},
            )
        before = uncovered(db)
        db.ingest(
            subject=(CONTROLS[0], "control"), predicate=("supported_by", "relates_to"),
            object=("synthetic-review-record", "evidence"), confidence=0.5,
            provenance={"source_id": "synthetic-demo-evidence", "source_type": "manual"},
            payload={"schema_ref": "demo/evidence", "data": {"synthetic": True}},
        )
        after = uncovered(db)
        db.retract("synthetic-demo-evidence", reason="Demo evidence withdrawn")
        withdrawn = uncovered(db)
        return {"uncovered_before": before, "uncovered_after_evidence": after,
                "uncovered_after_withdrawal": withdrawn,
                "scope": "Evidence-link coverage only; not certification or evidence acceptance"}
    finally:
        db.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="attest-compliance-demo-") as directory:
        print(json.dumps(run_demo(str(Path(directory) / "demo.attest")), indent=2))
