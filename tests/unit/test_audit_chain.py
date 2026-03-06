"""Tests for the tamper-evident Merkle hash chain audit log."""

import json

import pytest

from attestdb.core.hashing import compute_chain_hash
from attestdb.infrastructure.attest_db import AttestDB


@pytest.fixture()
def db(tmp_path):
    db = AttestDB(str(tmp_path / "test"), embedding_dim=None)
    yield db
    db.close()


def _make_claim(subject="A", predicate="relates_to", obj="B", source_id="s1"):
    return dict(
        subject=(subject, "entity"),
        predicate=(predicate, "relation"),
        object=(obj, "entity"),
        provenance={"source_type": "test", "source_id": source_id},
    )


class TestChainHash:
    def test_chain_hash_changes_with_prev(self):
        h1 = compute_chain_hash("hash_a", "claim_1")
        h2 = compute_chain_hash("hash_b", "claim_1")
        assert h1 != h2


class TestAuditChain:
    def test_ingest_builds_chain(self, db):
        cid1 = db.ingest(**_make_claim(source_id="s1"))
        cid2 = db.ingest(**_make_claim(subject="C", source_id="s2"))
        assert len(db._chain_log) == 2
        assert db._chain_log[0][0] == cid1
        assert db._chain_log[1][0] == cid2

    def test_chain_is_valid(self, db):
        db.ingest(**_make_claim(source_id="s1"))
        db.ingest(**_make_claim(subject="C", source_id="s2"))
        db.ingest(**_make_claim(subject="D", source_id="s3"))
        result = db.verify_integrity()
        assert result["valid"] is True
        assert result["length"] == 3
        assert result["error"] is None

    def test_tampered_chain_detected(self, db, tmp_path):
        db.ingest(**_make_claim(source_id="s1"))
        db.ingest(**_make_claim(subject="C", source_id="s2"))
        db.ingest(**_make_claim(subject="D", source_id="s3"))

        # Flush chain to disk so we can tamper with the file
        db._flush_chain_log()

        # Tamper with the chain log file
        chain_path = db._chain_log_path()
        with open(chain_path) as f:
            data = json.load(f)
        data[1]["chain_hash"] = "tampered_hash_value"
        with open(chain_path, "w") as f:
            json.dump(data, f)

        # Reload chain from disk
        db._load_chain_log()
        result = db.verify_integrity()
        assert result["valid"] is False
        assert "position 1" in result["error"]

    def test_chain_persists_across_sessions(self, tmp_path):
        path = str(tmp_path / "persist_test")
        db1 = AttestDB(path, embedding_dim=None)
        cid1 = db1.ingest(**_make_claim(source_id="s1"))
        cid2 = db1.ingest(**_make_claim(subject="C", source_id="s2"))
        db1.close()

        db2 = AttestDB(path, embedding_dim=None)
        assert len(db2._chain_log) == 2
        assert db2._chain_log[0][0] == cid1
        assert db2._chain_log[1][0] == cid2
        result = db2.verify_integrity()
        assert result["valid"] is True

        # New ingest extends the chain
        cid3 = db2.ingest(**_make_claim(subject="E", source_id="s3"))
        assert len(db2._chain_log) == 3
        result = db2.verify_integrity()
        assert result["valid"] is True
        db2.close()

    def test_snapshot_includes_chain_log(self, db, tmp_path):
        db.ingest(**_make_claim(source_id="s1"))
        db.ingest(**_make_claim(subject="C", source_id="s2"))
        snap_dir = str(tmp_path / "snap")
        db.snapshot(snap_dir)

        # Restore and verify
        restore_path = str(tmp_path / "restored")
        db2 = AttestDB.restore(snap_dir, restore_path, embedding_dim=None)
        assert len(db2._chain_log) == 2
        result = db2.verify_integrity()
        assert result["valid"] is True
        db2.close()

    def test_ingest_batch_appends_to_chain(self, db):
        from attestdb.core.types import ClaimInput

        claims = [
            ClaimInput(
                subject=("X", "entity"), predicate=("rel", "rel"),
                object=("Y", "entity"),
                provenance={"source_type": "test", "source_id": f"batch_{i}"},
            )
            for i in range(5)
        ]
        result = db.ingest_batch(claims)
        assert result.ingested == 5
        assert len(db._chain_log) == 5
        integrity = db.verify_integrity()
        assert integrity["valid"] is True

