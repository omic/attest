"""End-to-end user journey test — exercises AttestDB features through
the public API exactly as a real user would.

Only tests that are NOT covered by dedicated test files live here.
Dedicated coverage lives in: test_ingestion.py, test_query.py,
test_retraction.py, test_time_travel.py, test_snapshot.py,
test_insight_engine.py, test_event_hooks.py, test_inquiry.py,
test_narrative.py, test_quality_report.py, test_knowledge_health.py,
test_verify_integrity.py.
"""

from __future__ import annotations

import json
import time

import pytest

from attestdb.infrastructure.attest_db import AttestDB


@pytest.fixture()
def db(tmp_path):
    """Fresh database — no embeddings (fast)."""
    _db = AttestDB(str(tmp_path / "journey"), embedding_dim=None)
    yield _db
    _db.close()


@pytest.fixture()
def db_with_embeddings(tmp_path):
    """Fresh database with embedding index enabled (dim=8)."""
    _db = AttestDB(str(tmp_path / "emb_journey"), embedding_dim=8)
    yield _db
    _db.close()


# ── Core ingest + query ──────────────────────────────────────────────


class TestIngestAndQuery:
    """Core CRUD: ingest claims, query them back, verify structure."""

    def test_ingest_multiple_and_query(self, db):
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("interacts_with", "relates_to"),
            object=("TP53", "protein"),
            provenance={"source_type": "observation", "source_id": "paper1"},
            confidence=0.9,
        )
        db.ingest(
            subject=("TP53", "protein"),
            predicate=("inhibits", "relates_to"),
            object=("MDM2", "protein"),
            provenance={"source_type": "observation", "source_id": "paper2"},
            confidence=0.85,
        )
        db.ingest(
            subject=("MDM2", "protein"),
            predicate=("binds", "relates_to"),
            object=("P21", "protein"),
            provenance={"source_type": "experiment", "source_id": "lab1"},
            confidence=0.75,
        )

        frame = db.query("BRCA1", depth=2)
        assert frame.focal_entity.id == "brca1"
        assert frame.claim_count > 0
        assert len(frame.direct_relationships) >= 1
        neighbor_ids = {r.target.id for r in frame.direct_relationships}
        assert "tp53" in neighbor_ids

    def test_list_entities_by_type(self, db):
        db.ingest(
            subject=("Gene1", "gene"),
            predicate=("relates_to", "relates_to"),
            object=("Disease1", "disease"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        genes = db.list_entities(entity_type="gene")
        assert all(e.entity_type == "gene" for e in genes)
        assert any(e.id == "gene1" for e in genes)


# ── Schema descriptor ─────────────────────────────────────────────────


class TestSchema:
    def test_schema_returns_descriptor(self, db):
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("interacts_with", "relates_to"),
            object=("TP53", "protein"),
            provenance={"source_type": "observation", "source_id": "paper1"},
        )
        schema = db.schema()
        assert schema.total_entities >= 2
        assert "gene" in schema.entity_types
        assert "protein" in schema.entity_types
        assert len(schema.relationship_patterns) >= 1


# ── Graph paths ────────────────────────────────────────────────────────


class TestGraphPaths:
    def test_path_exists(self, db):
        db.ingest(
            subject=("A", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("B", "entity"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        db.ingest(
            subject=("B", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("C", "entity"),
            provenance={"source_type": "observation", "source_id": "s2"},
        )
        assert db.path_exists("A", "C", max_depth=2) is True
        assert db.path_exists("A", "Z", max_depth=5) is False

    def test_find_paths(self, db):
        db.ingest(
            subject=("A", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("B", "entity"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        db.ingest(
            subject=("B", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("C", "entity"),
            provenance={"source_type": "observation", "source_id": "s2"},
        )
        paths = db.find_paths("A", "C", max_depth=3)
        assert len(paths) >= 1
        assert any("b" in [s.entity_id for s in p.steps] for p in paths)


# ── Entity merging ─────────────────────────────────────────────────────


class TestEntityMerging:
    def test_merge_entities(self, db):
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("relates_to", "relates_to"),
            object=("TP53", "protein"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        claim_id = db.merge_entities("BRCA1", "BRCA1_HUMAN", reason="UniProt mapping")
        assert claim_id
        resolved = db.resolve("BRCA1_HUMAN")
        assert resolved == "brca1" or resolved == "brca1_human"


# ── Embedding search ──────────────────────────────────────────────────


class TestEmbeddingSearch:
    def test_search_with_embeddings(self, db_with_embeddings):
        import numpy as np

        db = db_with_embeddings
        emb = np.random.randn(8).tolist()
        cid = db.ingest(
            subject=("A", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("B", "entity"),
            provenance={"source_type": "observation", "source_id": "s1"},
            embedding=emb,
        )
        results = db.search(np.random.randn(8).tolist(), top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1

        stored = db.get_embedding(cid)
        assert stored is not None
        assert len(stored) == 8


# ── Vocabulary registration ──────────────────────────────────────────


class TestVocabularyRegistration:
    def test_register_vocabulary(self, db):
        db.register_vocabulary("custom", {
            "entity_types": {"widget": True, "gadget": True},
            "predicate_types": {"powers": True},
            "source_types": {"manual_entry": True},
        })
        cid = db.ingest(
            subject=("Widget1", "widget"),
            predicate=("powers", "powers"),
            object=("Gadget1", "gadget"),
            provenance={"source_type": "manual_entry", "source_id": "test"},
        )
        assert cid
        claims = db.claims_for("widget1")
        assert len(claims) >= 1


# ── Claims filtering ──────────────────────────────────────────────────


class TestClaimsFiltering:
    def test_claims_for_with_filters(self, db):
        db.ingest(
            subject=("X", "gene"),
            predicate=("inhibits", "relates_to"),
            object=("Y", "protein"),
            provenance={"source_type": "observation", "source_id": "lab1"},
            confidence=0.9,
        )
        db.ingest(
            subject=("X", "gene"),
            predicate=("activates", "activates"),
            object=("Z", "protein"),
            provenance={"source_type": "observation", "source_id": "paper1"},
            confidence=0.6,
        )

        relates_claims = db.claims_for("x", predicate_type="relates_to")
        assert len(relates_claims) >= 1
        assert all(c.predicate.predicate_type == "relates_to" for c in relates_claims)

        high_conf = db.claims_for("x", min_confidence=0.8)
        assert all(c.confidence >= 0.8 for c in high_conf)
        assert len(high_conf) >= 1

    def test_claims_by_content_id(self, db):
        from attestdb.core.hashing import compute_content_id
        from attestdb.core.normalization import normalize_entity_id

        db.ingest(
            subject=("A", "entity"),
            predicate=("relates_to", "relates_to"),
            object=("B", "entity"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        content_id = compute_content_id(
            normalize_entity_id("A"), "relates_to", normalize_entity_id("B"),
        )
        claims = db.claims_by_content_id(content_id)
        assert len(claims) >= 1


# ── Explain (query + profiling) ───────────────────────────────────────


class TestExplain:
    def test_explain_returns_profile(self, db):
        db.ingest(
            subject=("BRCA1", "gene"),
            predicate=("interacts_with", "relates_to"),
            object=("TP53", "protein"),
            provenance={"source_type": "observation", "source_id": "paper1"},
        )
        frame, profile = db.explain("BRCA1", depth=1)
        assert frame.focal_entity.id == "brca1"
        assert hasattr(profile, "elapsed_ms")
        assert profile.elapsed_ms >= 0


# ── Context Manager ───────────────────────────────────────────────────


class TestContextManager:
    def test_with_statement(self, tmp_path):
        """AttestDB can be used as a context manager."""
        path = str(tmp_path / "ctx")
        with AttestDB(path, embedding_dim=None) as ctx_db:
            ctx_db.ingest(
                subject=("A", "entity"),
                predicate=("activates", "relates_to"),
                object=("B", "entity"),
                provenance={"source_type": "observation", "source_id": "s1"},
            )
            assert len(ctx_db.list_entities()) >= 2
        # After exiting, the DB should be closed (re-open to verify data persisted)
        db2 = AttestDB(path, embedding_dim=None)
        assert len(db2.list_entities()) >= 2
        db2.close()


# ── Ingest Return Value ──────────────────────────────────────────────


class TestIngestReturnValue:
    def test_ingest_returns_claim_id(self, db):
        """ingest() returns a string claim_id."""
        cid = db.ingest(
            subject=("A", "entity"),
            predicate=("activates", "relates_to"),
            object=("B", "entity"),
            provenance={"source_type": "observation", "source_id": "s1"},
        )
        assert isinstance(cid, str)
        assert len(cid) > 0
