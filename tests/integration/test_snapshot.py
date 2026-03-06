"""Tests for AttestDB.snapshot() and AttestDB.restore()."""

import os
import tempfile

import pytest

from attestdb import ClaimInput, AttestDB


def _make_claim(subj: str, pred: str, obj: str, source_id: str = "test") -> ClaimInput:
    return ClaimInput(
        subject=(subj, "gene"),
        predicate=(pred, pred),
        object=(obj, "disease"),
        provenance={"source_type": "database_import", "source_id": source_id},
        confidence=0.9,
    )


def test_snapshot_creates_files():
    """Verify that snapshot writes expected files to dest directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "db")
        snap_dir = os.path.join(tmpdir, "snap")

        db = AttestDB(db_path, embedding_dim=64)
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("associated_with", "associated_with"),
            object=("Breast Cancer", "disease"),
            provenance={"source_type": "database_import", "source_id": "test"},
            confidence=0.9,
        )
        result = db.snapshot(snap_dir)

        assert result == snap_dir
        assert os.path.isdir(snap_dir)
        entries = os.listdir(snap_dir)
        assert any(e.endswith(".attest") for e in entries), f"No .attest in snapshot: {entries}"
        db.close()


def test_restore_recovers_data():
    """Ingest claims, snapshot, restore to new path, verify data is intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "original")
        snap_dir = os.path.join(tmpdir, "snap")
        restore_path = os.path.join(tmpdir, "restored")

        # Create and populate
        db = AttestDB(db_path, embedding_dim=64)
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("associated_with", "associated_with"),
            object=("Breast Cancer", "disease"),
            provenance={"source_type": "database_import", "source_id": "pmid:1234"},
            confidence=0.9,
        )
        db.ingest(
            subject=("TP53", "protein"),
            predicate=("inhibits", "inhibits"),
            object=("Cell Growth", "biological_process"),
            provenance={"source_type": "experimental", "source_id": "exp:001"},
            confidence=0.85,
        )
        original_stats = db.stats()
        db.snapshot(snap_dir)
        db.close()

        # Restore
        db2 = AttestDB.restore(snap_dir, restore_path, embedding_dim=64)
        restored_stats = db2.stats()

        assert restored_stats["total_claims"] == original_stats["total_claims"]
        assert restored_stats["entity_count"] == original_stats["entity_count"]

        # Verify query works
        claims = db2.claims_for("brca1")
        assert len(claims) >= 1
        assert any(c.object.display_name == "Breast Cancer" for c in claims)
        db2.close()


def test_snapshot_does_not_close_db():
    """DB should remain usable after taking a snapshot."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "db")
        snap_dir = os.path.join(tmpdir, "snap")

        db = AttestDB(db_path, embedding_dim=64)
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("associated_with", "associated_with"),
            object=("Breast Cancer", "disease"),
            provenance={"source_type": "database_import", "source_id": "test"},
            confidence=0.9,
        )
        db.snapshot(snap_dir)

        # DB should still be usable — ingest another claim
        db.ingest(
            subject=("TP53", "protein"),
            predicate=("inhibits", "inhibits"),
            object=("Cell Growth", "biological_process"),
            provenance={"source_type": "experimental", "source_id": "exp:001"},
            confidence=0.85,
        )
        stats = db.stats()
        assert stats["total_claims"] >= 2
        db.close()
