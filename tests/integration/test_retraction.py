"""Tests for claim retraction (Feature 1)."""

import time

import pytest

from attestdb.core.types import ClaimInput, ClaimStatus


@pytest.fixture
def retract_db(make_db):
    """DB with 3 claims from same source + 1 from different source."""
    db = make_db(embedding_dim=None)
    now = time.time_ns()
    for i in range(3):
        db.ingest(
            subject=(f"drug_{i}", "compound"),
            predicate=("inhibits", "relates_to"),
            object=("target_x", "protein"),
            provenance={"source_type": "document_extraction", "source_id": "paper_123"},
            confidence=0.8,
            timestamp=now + i,
        )
    # One claim from a different source
    db.ingest(
        subject=("drug_other", "compound"),
        predicate=("activates", "relates_to"),
        object=("target_x", "protein"),
        provenance={"source_type": "observation", "source_id": "experiment_456"},
        confidence=0.9,
        timestamp=now + 10,
    )
    return db


def test_retract_marks_claims_tombstoned(retract_db):
    result = retract_db.retract("paper_123", "Paper retracted by journal")
    assert result.retracted_count == 3
    assert result.source_id == "paper_123"
    assert result.reason == "Paper retracted by journal"
    assert len(result.claim_ids) == 3

    # Verify original claims are tombstoned (exclude the retraction meta-claim)
    claims = retract_db.claims_by_source_id("paper_123")
    original_claims = [c for c in claims if c.predicate.id != "retracted"]
    assert len(original_claims) == 3
    for c in original_claims:
        assert c.status == ClaimStatus.TOMBSTONED


def test_retracted_claims_excluded_from_query(retract_db):
    # Before retraction, target_x should have relationships from paper_123
    frame_before = retract_db.query("target_x", depth=1)
    count_before = frame_before.claim_count

    retract_db.retract("paper_123", "Retracted")

    # After retraction, paper_123 claims should be excluded
    frame_after = retract_db.query("target_x", depth=1)
    assert frame_after.claim_count < count_before


def test_retract_nonexistent_source(retract_db):
    result = retract_db.retract("nonexistent_source", "test")
    assert result.retracted_count == 0
    assert result.claim_ids == []


def test_retract_audit_trail(retract_db):
    retract_db.retract("paper_123", "Fabricated data")

    # There should be a retraction meta-claim
    claims = retract_db.claims_by_source_id("paper_123")
    retraction_claims = [c for c in claims if c.predicate.id == "retracted"]
    assert len(retraction_claims) >= 1
