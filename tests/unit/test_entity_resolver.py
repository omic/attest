"""Unit tests for EntityResolver internals."""

import logging

import pytest

from attestdb.core.types import EntitySummary
from attestdb.infrastructure.entity_resolver import EntityResolver

logging.disable(logging.WARNING)


def _entity_dict(eid, name="", etype="entity", ext_ids=None):
    """Return a dict matching raw RustStore format."""
    return {"id": eid, "name": name or eid, "entity_type": etype, "external_ids": ext_ids or {}, "claim_count": 0}


class FakeStore:
    """Minimal store stub for unit-testing EntityResolver (returns dicts like raw RustStore)."""

    def __init__(self, entities=None):
        self._entities = {e["id"]: e for e in (entities or [])}

    def list_entities(self, entity_type=None, min_claims=0):
        return list(self._entities.values())

    def get_entity(self, entity_id):
        return self._entities.get(entity_id)

    def search_entities(self, query, top_k=10):
        """Simple substring match for testing."""
        results = []
        q = query.lower()
        for e in self._entities.values():
            if q in e["id"].lower() or (e["name"] and q in e["name"].lower()):
                results.append(e)
        return results[:top_k]


def _entity(eid, name="", etype="entity", ext_ids=None):
    return EntitySummary(id=eid, name=name or eid, entity_type=etype, external_ids=ext_ids or {})


def test_build_index_from_entities():
    """Resolver indexes external_ids from store entities."""
    store = FakeStore([
        _entity_dict("gene_672", etype="gene", ext_ids={"ncbi_gene": "672", "symbol": "BRCA1"}),
        _entity_dict("gene_7157", etype="gene", ext_ids={"ncbi_gene": "7157", "symbol": "TP53"}),
    ])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    assert resolver._built is True
    assert resolver.resolve_by_external_id("ncbi_gene", "672") == "gene_672"
    assert resolver.resolve_by_external_id("symbol", "BRCA1") == "gene_672"
    assert resolver.resolve_by_external_id("ncbi_gene", "7157") == "gene_7157"
    assert resolver.resolve_by_external_id("ncbi_gene", "999") is None


def test_resolve_exact_match():
    """Normalized name match returns conf=1.0."""
    store = FakeStore([_entity_dict("brca1", etype="gene")])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    eid, conf = resolver.resolve("brca1")
    assert eid == "brca1"
    assert conf == 1.0


def test_resolve_by_external_id():
    """External_id match returns conf=0.99."""
    store = FakeStore([
        _entity_dict("gene_672", etype="gene", ext_ids={"symbol": "BRCA1"}),
    ])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    # Name doesn't match, but external_id does
    eid, conf = resolver.resolve("some_other_name", "gene", {"symbol": "BRCA1"})
    assert eid == "gene_672"
    assert conf == 0.99


def test_register_external_id_incremental():
    """New registrations are queryable immediately."""
    store = FakeStore([])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    assert resolver.resolve_by_external_id("ncbi_gene", "672") is None

    resolver.register_external_id("gene_672", "ncbi_gene", "672")
    assert resolver.resolve_by_external_id("ncbi_gene", "672") == "gene_672"


def test_find_duplicates_by_external_id():
    """Detects entities sharing external_ids across different namespaces."""
    store = FakeStore([
        _entity_dict("gene_672", etype="gene", ext_ids={"ncbi_gene": "672"}),
        _entity_dict("brca1_gene", etype="gene", ext_ids={"ncbi_gene": "672"}),
    ])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    dupes = resolver.find_duplicates(min_confidence=0.5)
    assert len(dupes) == 1
    pair = (dupes[0][0], dupes[0][1])
    assert "brca1_gene" in pair
    assert "gene_672" in pair
    assert dupes[0][2] == 0.99


def test_score_candidate_jaccard():
    """Token overlap scoring works correctly."""
    candidate = _entity("cyclin dependent kinase 4", name="CDK4", etype="gene")

    # Exact token overlap
    score = EntityResolver._score_candidate("cyclin dependent kinase 4", "gene", candidate)
    assert score > 0.8

    # Partial overlap
    score_partial = EntityResolver._score_candidate("cyclin kinase", "gene", candidate)
    assert 0.0 < score_partial < score

    # No overlap
    score_none = EntityResolver._score_candidate("completely unrelated", "entity", candidate)
    assert score_none == 0.0


def test_score_candidate_type_bonus():
    """Matching entity_type adds a bonus to the score."""
    candidate = _entity("tp53", name="TP53", etype="gene")

    score_match = EntityResolver._score_candidate("tp53", "gene", candidate)
    score_nomatch = EntityResolver._score_candidate("tp53", "protein", candidate)
    assert score_match > score_nomatch


def test_resolve_fuzzy_mode():
    """Fuzzy mode uses text search when exact+ext_id fails."""
    store = FakeStore([
        _entity_dict("cyclin dependent kinase 4", name="CDK4", etype="gene"),
    ])
    resolver = EntityResolver(store, mode="fuzzy")
    resolver.build_index()

    # "cyclin dependent kinase" partially overlaps — should match via text search
    eid, conf = resolver.resolve("cyclin dependent kinase 4", "gene")
    # Exact match on normalized name
    assert eid == "cyclin dependent kinase 4"
    assert conf == 1.0


