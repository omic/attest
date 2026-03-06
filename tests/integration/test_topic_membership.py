"""Tests for ContextFrame topic_membership enrichment (Feature 5)."""

import time

import pytest

from attestdb.core.types import ClaimInput


@pytest.fixture
def topo_db(make_db):
    """Build a small graph and compute topology."""
    db = make_db(embedding_dim=None)
    now = time.time_ns()

    # Build a connected component with enough nodes for community detection
    entities = [f"node_{i}" for i in range(10)]
    for i in range(len(entities)):
        for j in range(i + 1, min(i + 3, len(entities))):
            db.ingest(
                subject=(entities[i], "entity"),
                predicate=("relates_to", "relates_to"),
                object=(entities[j], "entity"),
                provenance={"source_type": "observation", "source_id": "s1"},
                confidence=0.8,
                timestamp=now + i * 10 + j,
            )

    try:
        db.compute_topology(resolutions=[0.001], min_community_size=2)
    except ImportError:
        pytest.skip("leidenalg/igraph not available")

    return db


def test_topic_membership_populated(topo_db):
    frame = topo_db.query("node_0", depth=1)
    assert len(frame.topic_membership) > 0


def test_topic_membership_empty_without_topology(make_db):
    db = make_db(embedding_dim=None)
    now = time.time_ns()
    db.ingest(
        subject=("a", "entity"), predicate=("relates_to", "relates_to"),
        object=("b", "entity"),
        provenance={"source_type": "observation", "source_id": "s1"},
        confidence=0.8, timestamp=now,
    )
    frame = db.query("a", depth=1)
    assert frame.topic_membership == []


def test_query_topic_returns_frames(topo_db):
    topics = topo_db.topics()
    assert len(topics) > 0
    topic_id = topics[0].id
    frames = topo_db.query_topic(topic_id, depth=1)
    assert isinstance(frames, list)
    assert len(frames) > 0
    # Each frame should be a ContextFrame
    for f in frames:
        assert hasattr(f, "focal_entity")


def test_query_topic_requires_topology(make_db):
    db = make_db(embedding_dim=None)
    now = time.time_ns()
    db.ingest(
        subject=("a", "entity"), predicate=("relates_to", "relates_to"),
        object=("b", "entity"),
        provenance={"source_type": "observation", "source_id": "s1"},
        confidence=0.8, timestamp=now,
    )
    with pytest.raises(RuntimeError, match="compute_topology"):
        db.query_topic("some_topic")


