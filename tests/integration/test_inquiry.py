"""Tests for inquiry tracking (Feature 9)."""

import time

import pytest

from attestdb.core.types import ClaimInput


@pytest.fixture
def inquiry_db(make_db):
    db = make_db(embedding_dim=None)
    now = time.time_ns()
    # Seed a basic graph so entities exist
    db.ingest(
        subject=("drug_x", "compound"),
        predicate=("inhibits", "relates_to"),
        object=("target_y", "protein"),
        provenance={"source_type": "observation", "source_id": "s1"},
        confidence=0.8, timestamp=now,
    )
    return db


def test_ingest_inquiry(inquiry_db):
    claim_id = inquiry_db.ingest_inquiry(
        question="Does drug X inhibit target Y?",
        subject=("drug_x", "compound"),
        object=("target_y", "protein"),
        predicate_hint="inhibits",
    )
    assert claim_id is not None
    assert len(claim_id) > 0


def test_open_inquiries(inquiry_db):
    inquiry_db.ingest_inquiry(
        question="Does drug X inhibit target Y?",
        subject=("drug_x", "compound"),
        object=("target_y", "protein"),
    )
    inquiry_db.ingest_inquiry(
        question="Does drug X bind target Z?",
        subject=("drug_x", "compound"),
        object=("target_z", "protein"),
    )

    inquiries = inquiry_db.open_inquiries()
    assert len(inquiries) == 2
    questions = [c.payload.data.get("question", "") for c in inquiries if c.payload]
    assert "Does drug X inhibit target Y?" in questions
    assert "Does drug X bind target Z?" in questions


def test_inquiry_in_context_frame(inquiry_db):
    inquiry_db.ingest_inquiry(
        question="Does drug X inhibit target Y?",
        subject=("drug_x", "compound"),
        object=("target_y", "protein"),
    )

    frame = inquiry_db.query("drug_x", depth=1)
    assert len(frame.open_inquiries) > 0


def test_check_inquiry_matches_pair(inquiry_db):
    """Matching subject+object returns inquiry claim_id."""
    inquiry_db.ingest_inquiry(
        question="Does drug X inhibit target Y?",
        subject=("drug_x", "compound"),
        object=("target_y", "protein"),
    )
    matches = inquiry_db.check_inquiry_matches(
        subject_id="drug_x", object_id="target_y",
    )
    assert len(matches) == 1

    # Reverse direction should also match
    matches_rev = inquiry_db.check_inquiry_matches(
        subject_id="target_y", object_id="drug_x",
    )
    assert len(matches_rev) == 1


def test_check_inquiry_matches_predicate_hint(inquiry_db):
    """Predicate hint match works."""
    inquiry_db.ingest_inquiry(
        question="Does drug X inhibit target Y?",
        subject=("drug_x", "compound"),
        object=("target_y", "protein"),
        predicate_hint="inhibits",
    )
    matches = inquiry_db.check_inquiry_matches(predicate_id="inhibits")
    assert len(matches) == 1


