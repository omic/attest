"""Tests for AttestDB.knowledge_health() metrics."""

import pytest

from attestdb.core.types import KnowledgeHealth
from attestdb.infrastructure.attest_db import AttestDB


@pytest.fixture()
def db(tmp_path):
    db = AttestDB(str(tmp_path / "test"), embedding_dim=None)
    yield db
    db.close()


def _ingest(db, subject="A", predicate="relates_to", obj="B", source_id="s1", source_type="test"):
    return db.ingest(
        subject=(subject, "entity"),
        predicate=(predicate, "relation"),
        object=(obj, "entity"),
        provenance={"source_type": source_type, "source_id": source_id},
    )


class TestKnowledgeHealth:
    def test_empty_db(self, db):
        h = db.knowledge_health()
        assert isinstance(h, KnowledgeHealth)
        assert h.total_claims == 0
        assert h.total_entities == 0
        assert h.health_score == 0.0

    def test_single_claim(self, db):
        _ingest(db)
        h = db.knowledge_health()
        assert h.total_claims >= 1
        assert h.total_entities >= 1
        assert h.avg_confidence > 0
        assert h.source_diversity >= 1
        assert h.knowledge_density > 0

    def test_multi_source_ratio(self, db):
        # Entity A has claims from two sources
        _ingest(db, subject="A", obj="B", source_id="s1")
        _ingest(db, subject="A", obj="C", source_id="s2")
        h = db.knowledge_health()
        # A has 2 sources, B has 1, C has 1 → ratio = 1/3
        assert h.multi_source_ratio > 0

    def test_corroboration_ratio(self, db):
        # Same claim from two sources → corroboration
        _ingest(db, subject="A", predicate="rel", obj="B", source_id="s1")
        _ingest(db, subject="A", predicate="rel", obj="B", source_id="s2")
        h = db.knowledge_health()
        assert h.corroboration_ratio > 0

    def test_source_diversity(self, db):
        _ingest(db, source_type="paper")
        _ingest(db, subject="C", obj="D", source_type="experiment")
        h = db.knowledge_health()
        assert h.source_diversity >= 2

    def test_health_score_bounded(self, db):
        # Ingest several diverse claims
        for i in range(5):
            _ingest(db, subject=f"E{i}", obj=f"F{i}", source_id=f"s{i}", source_type=f"type{i}")
        h = db.knowledge_health()
        assert 0 <= h.health_score <= 100

    def test_freshness_score_recent_claims(self, db):
        """Claims ingested just now should have freshness near 1.0."""
        _ingest(db)
        h = db.knowledge_health()
        # Recently ingested claims have near-perfect freshness
        assert h.freshness_score > 0.9

    def test_knowledge_density(self, db):
        _ingest(db, subject="A", obj="B")
        _ingest(db, subject="A", obj="C")
        h = db.knowledge_health()
        # 2 claims, 3 entities → density ≈ 0.67
        assert 0.5 < h.knowledge_density < 1.0

    def test_confidence_trend(self, db):
        """With identical confidences, trend should be near zero."""
        for i in range(4):
            _ingest(db, subject=f"E{i}", obj=f"F{i}", source_id=f"s{i}")
        h = db.knowledge_health()
        # All claims have default confidence, so trend ≈ 0
        assert abs(h.confidence_trend) < 0.01

