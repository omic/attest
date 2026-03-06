"""Tests for symmetric predicate traversal (Feature 2)."""

import time

import pytest

from attestdb.infrastructure.embedding_index import EmbeddingIndex
from attestdb.infrastructure.ingestion import IngestionPipeline
from attestdb.infrastructure.query_engine import QueryEngine
from attestdb.core.types import ClaimInput


@pytest.fixture
def sym_db(make_store):
    store = make_store()
    pipeline = IngestionPipeline(store, None, embedding_dim=None)
    qe = QueryEngine(store)
    now = time.time_ns()

    # Symmetric: interacts
    pipeline.ingest(ClaimInput(
        subject=("protein_a", "protein"), predicate=("interacts", "relates_to"),
        object=("protein_b", "protein"),
        provenance={"source_type": "observation", "source_id": "s1"},
        confidence=0.9, timestamp=now,
    ))
    # Asymmetric: inhibits
    pipeline.ingest(ClaimInput(
        subject=("drug_x", "compound"), predicate=("inhibits", "relates_to"),
        object=("protein_a", "protein"),
        provenance={"source_type": "observation", "source_id": "s2"},
        confidence=0.85, timestamp=now,
    ))

    yield store, qe
    store.close()


def test_symmetric_relationship_marked(sym_db):
    _, qe = sym_db
    frame = qe.query("protein_a", depth=1)
    interacts_rels = [r for r in frame.direct_relationships if r.predicate == "interacts"]
    assert len(interacts_rels) > 0
    for r in interacts_rels:
        assert r.is_symmetric is True


def test_asymmetric_relationship_not_marked(sym_db):
    _, qe = sym_db
    frame = qe.query("protein_a", depth=1)
    inhibits_rels = [r for r in frame.direct_relationships if r.predicate == "inhibits"]
    assert len(inhibits_rels) > 0
    for r in inhibits_rels:
        assert r.is_symmetric is False


def test_symmetric_query_from_both_sides(sym_db):
    _, qe = sym_db
    # Query from A
    frame_a = qe.query("protein_a", depth=1)
    targets_a = {r.target.id for r in frame_a.direct_relationships if r.predicate == "interacts"}
    assert "protein_b" in targets_a

    # Query from B — bidirectional BFS should find the relationship
    frame_b = qe.query("protein_b", depth=1)
    targets_b = {r.target.id for r in frame_b.direct_relationships if r.predicate == "interacts"}
    assert "protein_a" in targets_b
