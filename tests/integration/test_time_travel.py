"""Tests for time-travel queries via db.at(timestamp) (Feature 3)."""

import pytest

from attestdb.core.types import ClaimInput


@pytest.fixture
def temporal_db(make_db):
    db = make_db(embedding_dim=None)
    # Ingest claims at distinct timestamps
    db.ingest(
        subject=("entity_a", "entity"), predicate=("activates", "relates_to"),
        object=("entity_b", "entity"),
        provenance={"source_type": "observation", "source_id": "s1"},
        confidence=0.9, timestamp=1000,
    )
    db.ingest(
        subject=("entity_b", "entity"), predicate=("inhibits", "relates_to"),
        object=("entity_c", "entity"),
        provenance={"source_type": "observation", "source_id": "s2"},
        confidence=0.8, timestamp=2000,
    )
    db.ingest(
        subject=("entity_a", "entity"), predicate=("relates_to", "relates_to"),
        object=("entity_d", "entity"),
        provenance={"source_type": "computation", "source_id": "s3"},
        confidence=0.7, timestamp=3000,
    )
    return db


def test_at_returns_snapshot(temporal_db):
    snapshot = temporal_db.at(1500)
    assert snapshot is not None
    # Should be able to query through snapshot
    frame = snapshot.query("entity_a", depth=1)
    assert frame.focal_entity.id == "entity_a"


def test_snapshot_query_filters_future(temporal_db):
    # At t=1500, only the t=1000 claim should be visible
    snapshot = temporal_db.at(1500)
    frame = snapshot.query("entity_a", depth=2)
    assert frame.claim_count == 1
    target_ids = {r.target.id for r in frame.direct_relationships}
    assert "entity_b" in target_ids
    assert "entity_c" not in target_ids
    assert "entity_d" not in target_ids


def test_snapshot_includes_all_at_time(temporal_db):
    # At t=2500, claims at t=1000 and t=2000 should be visible
    snapshot = temporal_db.at(2500)
    frame = snapshot.query("entity_a", depth=2)
    assert frame.claim_count == 2
    target_ids = {r.target.id for r in frame.direct_relationships}
    assert "entity_b" in target_ids
    assert "entity_c" in target_ids
    assert "entity_d" not in target_ids


def test_snapshot_stats(temporal_db):
    snapshot = temporal_db.at(1500)
    stats = snapshot.stats()
    assert stats["timestamp"] == 1500
