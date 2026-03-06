"""Integration tests for ContextFrame query engine."""

import logging
import time

import pytest

from attestdb.core.types import ClaimInput
from attestdb.infrastructure.embedding_index import EmbeddingIndex
from attestdb.infrastructure.ingestion import IngestionPipeline
from attestdb.infrastructure.query_engine import QueryEngine

logging.disable(logging.WARNING)


@pytest.fixture
def query_db(make_store):
    """Set up a small graph for query tests."""
    store = make_store()
    emb_idx = EmbeddingIndex(ndim=4)
    pipeline = IngestionPipeline(store, emb_idx, embedding_dim=4)
    query_engine = QueryEngine(store)

    # Use recent timestamps so tier2 recency factor is ~1.0
    now_ns = time.time_ns()

    # Build a small graph: A -> B -> C -> D, with branches
    claims = [
        ClaimInput(subject=("A", "entity"), predicate=("activates", "relates_to"), object=("B", "entity"),
                   provenance={"source_type": "observation", "source_id": "s1"}, confidence=0.9, timestamp=now_ns - 5000),
        ClaimInput(subject=("B", "entity"), predicate=("activates", "relates_to"), object=("C", "entity"),
                   provenance={"source_type": "observation", "source_id": "s1"}, confidence=0.8, timestamp=now_ns - 4000),
        ClaimInput(subject=("C", "entity"), predicate=("activates", "relates_to"), object=("D", "entity"),
                   provenance={"source_type": "computation", "source_id": "s2"}, confidence=0.6, timestamp=now_ns - 3000),
        ClaimInput(subject=("A", "entity"), predicate=("inhibits", "relates_to"), object=("E", "entity"),
                   provenance={"source_type": "human_annotation", "source_id": "s3"}, confidence=0.95, timestamp=now_ns - 2000),
        ClaimInput(subject=("B", "entity"), predicate=("relates_to", "relates_to"), object=("F", "entity"),
                   provenance={"source_type": "llm_inference", "source_id": "s4"}, confidence=0.3, timestamp=now_ns - 1000),
    ]
    for ci in claims:
        pipeline.ingest(ci)

    yield store, query_engine
    store.close()


def test_query_depth_1(query_db):
    """Depth 1 should only return direct neighbors."""
    _, qe = query_db
    frame = qe.query("A", depth=1)
    assert frame.focal_entity.id == "a"
    target_ids = {r.target.id for r in frame.direct_relationships}
    assert "b" in target_ids
    assert "e" in target_ids
    # C and D should NOT be in depth-1
    assert "c" not in target_ids
    assert "d" not in target_ids


def test_query_depth_2(query_db):
    """Depth 2 should include 2-hop neighbors."""
    _, qe = query_db
    frame = qe.query("A", depth=2)
    target_ids = {r.target.id for r in frame.direct_relationships}
    assert "b" in target_ids
    assert "c" in target_ids or "f" in target_ids  # 2-hop from A via B


def test_query_depth_3(query_db):
    """Depth 3 should reach D."""
    _, qe = query_db
    frame = qe.query("A", depth=3)
    target_ids = {r.target.id for r in frame.direct_relationships}
    assert "d" in target_ids


def test_min_confidence_filter(query_db):
    """min_confidence should filter out low-confidence claims."""
    _, qe = query_db
    frame = qe.query("A", depth=2, min_confidence=0.5)
    # The llm_inference claim (0.3 confidence) should be excluded
    for r in frame.direct_relationships:
        assert r.confidence >= 0.5


def test_exclude_source_types(query_db):
    """exclude_source_types should filter out claims from those sources."""
    _, qe = query_db
    frame = qe.query("A", depth=2, exclude_source_types=["llm_inference"])
    source_types = set()
    for r in frame.direct_relationships:
        source_types.update(r.source_types)
    assert "llm_inference" not in source_types


def test_context_frame_structure(query_db):
    """ContextFrame should have all expected fields."""
    _, qe = query_db
    frame = qe.query("A", depth=2)
    assert frame.focal_entity is not None
    assert isinstance(frame.direct_relationships, list)
    assert isinstance(frame.quantitative_data, list)
    assert isinstance(frame.contradictions, list)
    assert isinstance(frame.knowledge_gaps, list)
    assert isinstance(frame.provenance_summary, dict)
    assert frame.claim_count > 0
    assert len(frame.confidence_range) == 2
    assert frame.confidence_range[0] <= frame.confidence_range[1]


def test_relationships_sorted_by_confidence(query_db):
    """Direct relationships should be sorted by confidence descending."""
    _, qe = query_db
    frame = qe.query("A", depth=2)
    confidences = [r.confidence for r in frame.direct_relationships]
    assert confidences == sorted(confidences, reverse=True)


