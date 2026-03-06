"""Integration tests for Phase C methods: temporal queries and text search."""

import pytest

from attestdb.core.types import ClaimInput, claim_from_dict, entity_summary_from_dict
from attestdb.infrastructure.embedding_index import EmbeddingIndex
from attestdb.infrastructure.ingestion import IngestionPipeline


@pytest.fixture
def loaded_store(make_store):
    """Store with 10 claims at timestamps 1000..10000."""
    store = make_store()
    emb_idx = EmbeddingIndex(ndim=4)
    pipeline = IngestionPipeline(store, emb_idx, embedding_dim=4)

    for i in range(10):
        ci = ClaimInput(
            subject=(f"Gene_{i}", "gene"),
            predicate=("relates_to", "relates_to"),
            object=(f"Disease_{i}", "disease"),
            provenance={"source_type": "experimental", "source_id": f"src_{i}"},
            confidence=0.8,
            timestamp=(i + 1) * 1000,
        )
        pipeline.ingest(ci)

    yield store
    store.close()


class TestClaimsInRange:
    def test_full_range(self, loaded_store):
        claims = [claim_from_dict(d) for d in loaded_store.claims_in_range(1000, 10000)]
        assert len(claims) == 10

    def test_partial_range(self, loaded_store):
        claims = [claim_from_dict(d) for d in loaded_store.claims_in_range(2500, 5500)]
        timestamps = [c.timestamp for c in claims]
        assert all(2500 <= t <= 5500 for t in timestamps)
        assert len(claims) == 3  # 3000, 4000, 5000

    def test_single_timestamp(self, loaded_store):
        claims = [claim_from_dict(d) for d in loaded_store.claims_in_range(5000, 5000)]
        assert len(claims) == 1
        assert claims[0].timestamp == 5000


class TestMostRecentClaims:
    def test_returns_most_recent(self, loaded_store):
        claims = [claim_from_dict(d) for d in loaded_store.most_recent_claims(3)]
        assert len(claims) == 3
        assert claims[0].timestamp >= claims[1].timestamp >= claims[2].timestamp
        assert claims[0].timestamp == 10000

    def test_returns_one(self, loaded_store):
        claims = [claim_from_dict(d) for d in loaded_store.most_recent_claims(1)]
        assert len(claims) == 1
        assert claims[0].timestamp == 10000


class TestSearchEntities:
    def test_search_by_name(self, loaded_store):
        results = [entity_summary_from_dict(d) for d in loaded_store.search_entities("gene", 10)]
        assert len(results) == 10
        assert all(r.entity_type == "gene" for r in results)

    def test_search_by_specific_id(self, loaded_store):
        results = [entity_summary_from_dict(d) for d in loaded_store.search_entities("disease_3", 10)]
        assert len(results) >= 1
        assert any(r.id == "disease_3" for r in results)

    def test_search_top_k_limit(self, loaded_store):
        results = [entity_summary_from_dict(d) for d in loaded_store.search_entities("gene", 3)]
        assert len(results) == 3
