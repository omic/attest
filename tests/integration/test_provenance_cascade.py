"""Tests for provenance cascade and downstream tracing (Phase C)."""

import time

import pytest

from attestdb.core.types import ClaimInput, ClaimStatus


@pytest.fixture
def cascade_db(make_db):
    """DB with a provenance chain: A -> B -> C, plus an unrelated claim."""
    db = make_db(embedding_dim=None)
    now = time.time_ns()

    # Claim A from source "paper_1"
    claim_a_id = db.ingest(
        subject=("drug_a", "compound"),
        predicate=("inhibits", "relates_to"),
        object=("target_x", "protein"),
        provenance={"source_type": "experimental", "source_id": "paper_1"},
        confidence=0.9, timestamp=now,
    )

    # Claim B depends on A (via provenance chain)
    claim_b_id = db.ingest(
        subject=("drug_a", "compound"),
        predicate=("treats", "relates_to"),
        object=("disease_y", "disease"),
        provenance={
            "source_type": "llm_inference", "source_id": "model_1",
            "chain": [claim_a_id],
        },
        confidence=0.7, timestamp=now + 1,
    )

    # Claim C depends on B (transitive chain)
    claim_c_id = db.ingest(
        subject=("disease_y", "disease"),
        predicate=("associated_with", "relates_to"),
        object=("gene_z", "gene"),
        provenance={
            "source_type": "llm_inference", "source_id": "model_2",
            "chain": [claim_b_id],
        },
        confidence=0.6, timestamp=now + 2,
    )

    # Unrelated claim from different source
    db.ingest(
        subject=("drug_other", "compound"),
        predicate=("activates", "relates_to"),
        object=("target_w", "protein"),
        provenance={"source_type": "observation", "source_id": "experiment_99"},
        confidence=0.8, timestamp=now + 3,
    )

    return db, claim_a_id, claim_b_id, claim_c_id


def test_cascade_marks_downstream_degraded(cascade_db):
    db, claim_a_id, claim_b_id, claim_c_id = cascade_db
    result = db.retract_cascade("paper_1", "Paper retracted")

    assert result.source_retract.retracted_count == 1  # only claim A from paper_1
    assert result.degraded_count == 2  # B and C
    assert claim_b_id in result.degraded_claim_ids
    assert claim_c_id in result.degraded_claim_ids


def test_cascade_no_downstream(make_db):
    """Retract with no dependents -> degraded_count=0."""
    db = make_db(embedding_dim=None)
    now = time.time_ns()
    db.ingest(
        subject=("drug_a", "compound"),
        predicate=("inhibits", "relates_to"),
        object=("target_x", "protein"),
        provenance={"source_type": "experimental", "source_id": "paper_solo"},
        confidence=0.9, timestamp=now,
    )
    result = db.retract_cascade("paper_solo", "No dependents")
    assert result.source_retract.retracted_count == 1
    assert result.degraded_count == 0
    assert result.degraded_claim_ids == []


def test_cascade_preserves_unrelated(cascade_db):
    """Other sources stay ACTIVE after cascade retraction."""
    db, *_ = cascade_db
    db.retract_cascade("paper_1", "Retracted")

    # The unrelated claim from experiment_99 should still be active
    claims = db.claims_by_source_id("experiment_99")
    assert len(claims) > 0
    for c in claims:
        assert c.status == ClaimStatus.ACTIVE


def test_trace_downstream_tree(cascade_db):
    """Verify tree structure: A -> B -> C."""
    db, claim_a_id, claim_b_id, claim_c_id = cascade_db
    tree = db.trace_downstream(claim_a_id)

    assert tree.claim_id == claim_a_id
    assert len(tree.dependents) == 1
    assert tree.dependents[0].claim_id == claim_b_id
    assert len(tree.dependents[0].dependents) == 1
    assert tree.dependents[0].dependents[0].claim_id == claim_c_id


def test_trace_downstream_no_dependents(cascade_db):
    """Leaf claim returns empty tree."""
    db, _, _, claim_c_id = cascade_db
    tree = db.trace_downstream(claim_c_id)

    assert tree.claim_id == claim_c_id
    assert tree.dependents == []
