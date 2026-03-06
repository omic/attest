"""Integration tests for each of the 13 validation rules."""

import logging

import numpy as np
import pytest

from attestdb.core.errors import (
    CircularProvenanceError,
    DimensionalityError,
    DuplicateClaimError,
    PredicateConstraintError,
    ProvenanceError,
    SchemaValidationError,
    VocabularyError,
)
from attestdb.core.types import ClaimInput, claim_from_dict, entity_summary_from_dict
from attestdb.infrastructure.embedding_index import EmbeddingIndex
from attestdb.infrastructure.ingestion import IngestionPipeline

logging.disable(logging.WARNING)


@pytest.fixture
def pipeline(make_store):
    store = make_store()
    emb_idx = EmbeddingIndex(ndim=4)
    pipe = IngestionPipeline(store, emb_idx, embedding_dim=4)
    yield pipe, store


@pytest.fixture
def strict_pipeline(make_store):
    store = make_store()
    emb_idx = EmbeddingIndex(ndim=4)
    pipe = IngestionPipeline(store, emb_idx, embedding_dim=4, strict=True)
    yield pipe, store


def _basic_claim(**overrides) -> ClaimInput:
    defaults = {
        "subject": ("A", "entity"),
        "predicate": ("relates_to", "relates_to"),
        "object": ("B", "entity"),
        "provenance": {"source_type": "observation", "source_id": "test"},
        "timestamp": 1000,
    }
    defaults.update(overrides)
    return ClaimInput(**defaults)


def test_rule1_entity_normalization(pipeline):
    """Rule 1: Entity IDs are normalized."""
    pipe, store = pipeline
    pipe.ingest(_basic_claim(subject=("  FOO  BAR  ", "entity"), timestamp=1001))
    raw = store.get_entity("foo bar")
    assert raw is not None
    entity = entity_summary_from_dict(raw)
    assert entity.id == "foo bar"


def test_rule2_claim_id_computed(pipeline):
    """Rule 2: claim_id is computed from canonical inputs."""
    pipe, store = pipeline
    cid = pipe.ingest(_basic_claim())
    assert len(cid) == 64  # SHA-256 hex


def test_rule3_content_id_computed(pipeline):
    """Rule 3: content_id is computed for corroboration grouping."""
    pipe, store = pipeline
    pipe.ingest(_basic_claim(
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    ))
    pipe.ingest(_basic_claim(
        provenance={"source_type": "computation", "source_id": "src2"},
        timestamp=2000,
    ))
    # Both should share the same content_id
    claims = [claim_from_dict(d) for d in store.claims_for("a")]
    content_ids = {c.content_id for c in claims}
    assert len(content_ids) == 1  # Same content_id


def test_rule4_duplicate_claim_error(pipeline):
    """Rule 4: Duplicate claim_id raises DuplicateClaimError."""
    pipe, _ = pipeline
    pipe.ingest(_basic_claim())
    with pytest.raises(DuplicateClaimError):
        pipe.ingest(_basic_claim())  # Same exact inputs = same claim_id


def test_rule5_corroboration_tracking(pipeline):
    """Rule 5: Corroborating claims are logged, not rejected."""
    pipe, store = pipeline
    pipe.ingest(_basic_claim(
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    ))
    # Same content, different source
    pipe.ingest(_basic_claim(
        provenance={"source_type": "computation", "source_id": "src2"},
        timestamp=2000,
    ))
    claims = store.claims_for("a")
    assert len(claims) == 2


def test_rule6_source_type_validation_strict(strict_pipeline):
    """Rule 6: Unknown source_type rejected in strict mode."""
    pipe, _ = strict_pipeline
    with pytest.raises(VocabularyError, match="source_type"):
        pipe.ingest(_basic_claim(
            provenance={"source_type": "made_up_type", "source_id": "test"},
        ))


def test_rule6_source_type_allowed_nonstrict(pipeline):
    """Rule 6: Unknown source_type allowed in non-strict mode."""
    pipe, _ = pipeline
    # Should not raise
    pipe.ingest(_basic_claim(
        provenance={"source_type": "made_up_type", "source_id": "test"},
    ))


def test_rule7_entity_type_validation_strict(strict_pipeline):
    """Rule 7: Unknown entity types rejected in strict mode."""
    pipe, _ = strict_pipeline
    with pytest.raises(VocabularyError, match="entity_type"):
        pipe.ingest(_basic_claim(subject=("X", "unknown_type")))


def test_rule8_predicate_type_validation_strict(strict_pipeline):
    """Rule 8: Unknown predicate type rejected in strict mode."""
    pipe, _ = strict_pipeline
    with pytest.raises(VocabularyError, match="predicate_type"):
        pipe.ingest(_basic_claim(predicate=("foo", "unknown_pred_type")))


def test_rule9_predicate_constraints(pipeline):
    """Rule 9: Predicate constraints enforce subject/object types."""
    pipe, store = pipeline
    store.register_predicate("binds", {
        "subject_types": ["protein"],
        "object_types": ["protein", "compound"],
    })
    with pytest.raises(PredicateConstraintError, match="subject type"):
        pipe.ingest(_basic_claim(
            subject=("X", "entity"),
            predicate=("binds", "relates_to"),
            object=("Y", "protein"),
        ))
    with pytest.raises(PredicateConstraintError, match="object type"):
        pipe.ingest(_basic_claim(
            subject=("X", "protein"),
            predicate=("binds", "relates_to"),
            object=("Y", "entity"),
            timestamp=2000,
        ))


def test_rule10_payload_schema_validation(pipeline):
    """Rule 10: Payload data validated against registered schema."""
    pipe, store = pipeline
    store.register_payload_schema("binding_affinity", {
        "type": "object",
        "properties": {
            "metric": {"enum": ["Kd", "Ki", "IC50"]},
            "value": {"type": "number"},
            "unit": {"enum": ["nM", "uM"]},
        },
        "required": ["metric", "value", "unit"],
    })
    # Valid payload
    pipe.ingest(_basic_claim(
        payload={"schema": "binding_affinity", "data": {"metric": "Kd", "value": 5.0, "unit": "nM"}},
    ))
    # Invalid payload (missing required field)
    with pytest.raises(SchemaValidationError):
        pipe.ingest(_basic_claim(
            payload={"schema": "binding_affinity", "data": {"metric": "Kd"}},
            timestamp=2000,
        ))


def test_rule11_provenance_chain_existence(pipeline):
    """Rule 11: Provenance chain references must exist."""
    pipe, _ = pipeline
    with pytest.raises(ProvenanceError, match="non-existent"):
        pipe.ingest(_basic_claim(
            provenance={
                "source_type": "observation",
                "source_id": "test",
                "chain": ["nonexistent_claim_id"],
            },
        ))


def test_rule11_provenance_chain_valid(pipeline):
    """Rule 11: Valid provenance chains are accepted."""
    pipe, _ = pipeline
    cid = pipe.ingest(_basic_claim(timestamp=1000))
    # This claim references the first one
    pipe.ingest(_basic_claim(
        subject=("C", "entity"),
        object=("D", "entity"),
        provenance={
            "source_type": "computation",
            "source_id": "derived",
            "chain": [cid],
        },
        timestamp=2000,
    ))


def test_rule12_provenance_dag_acyclicity(pipeline):
    """Rule 12: Circular provenance is rejected."""
    pipe, _ = pipeline
    # Create claim A (no chain — valid)
    cid_a = pipe.ingest(_basic_claim(timestamp=1000))
    # Create claim B that references A (linear chain — valid)
    cid_b = pipe.ingest(_basic_claim(
        subject=("C", "entity"), object=("D", "entity"),
        provenance={"source_type": "computation", "source_id": "test", "chain": [cid_a]},
        timestamp=2000,
    ))
    # Linear chain A → B → C is fine
    pipe.ingest(_basic_claim(
        subject=("E", "entity"), object=("F", "entity"),
        provenance={"source_type": "computation", "source_id": "test", "chain": [cid_b]},
        timestamp=3000,
    ))

    # Test cycle detection directly via _check_provenance_acyclicity.
    # At the ingestion API level, cycles are hard to construct because
    # Rule 11 requires chain entries to exist (so you can't reference
    # a claim that hasn't been ingested yet). But the safety check
    # catches the scenario where a claim's upstream ancestry loops
    # back to itself. We verify this by calling the check directly.

    # Linear chain: new_id → cid_b → cid_a — no cycle
    pipe._check_provenance_acyclicity("new_id", [cid_a])
    pipe._check_provenance_acyclicity("new_id", [cid_b])

    # Cycle: pretend we're inserting cid_a again with chain=[cid_b].
    # Walk: cid_b's chain = [cid_a] → matches new_claim_id=cid_a → cycle!
    with pytest.raises(CircularProvenanceError):
        pipe._check_provenance_acyclicity(cid_a, [cid_b])


def test_rule13_embedding_dimensionality(pipeline):
    """Rule 13: Embedding dimensionality must match."""
    pipe, _ = pipeline
    with pytest.raises(DimensionalityError):
        pipe.ingest(_basic_claim(
            embedding=[0.1, 0.2, 0.3],  # 3-dim, expected 4
        ))
    # Valid dimensionality
    pipe.ingest(_basic_claim(
        embedding=[0.1, 0.2, 0.3, 0.4],  # 4-dim, matches
    ))


def test_provenance_required(pipeline):
    """Provenance source_type and source_id are required."""
    pipe, _ = pipeline
    with pytest.raises(ProvenanceError, match="required"):
        pipe.ingest(_basic_claim(provenance={"source_type": "", "source_id": "test"}))
    with pytest.raises(ProvenanceError, match="required"):
        pipe.ingest(_basic_claim(provenance={"source_type": "observation", "source_id": ""}))


def test_batch_ingestion(pipeline):
    """Batch ingestion returns correct counts."""
    pipe, _ = pipeline
    claims = [
        _basic_claim(timestamp=(i + 1) * 1000)
        for i in range(10)
    ]
    # First batch should all succeed
    result = pipe.ingest_batch(claims)
    assert result.ingested == 10
    assert result.duplicates == 0
    assert result.errors == []

    # Re-ingesting should produce duplicates
    result2 = pipe.ingest_batch(claims)
    assert result2.duplicates == 10
    assert result2.ingested == 0
