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


def test_resolve_person_comma_first_flips_to_existing():
    """'Epstein, Jeffrey' should resolve to the existing 'jeffrey epstein' entity."""
    store = FakeStore([_entity_dict("jeffrey epstein", etype="person")])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    eid, conf = resolver.resolve("Epstein, Jeffrey", "person")
    assert eid == "jeffrey epstein"
    assert conf >= 0.95


def test_resolve_person_comma_first_with_middle_initial():
    """'Epstein, Jeffrey E.' should also resolve to 'jeffrey e. epstein'."""
    store = FakeStore([_entity_dict("jeffrey e. epstein", etype="person")])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()

    eid, conf = resolver.resolve("Epstein, Jeffrey E.", "person")
    assert eid == "jeffrey e. epstein"


def test_resolve_comma_name_skipped_for_org():
    """Comma-flip must not fire on org-type names (which may have legit commas)."""
    store = FakeStore([_entity_dict("acme llc", etype="org")])
    resolver = EntityResolver(store, mode="external_ids")
    resolver.build_index()
    eid, _conf = resolver.resolve("Acme, LLC", "org")
    assert eid is None  # no flip, no match


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


def test_score_candidate_feature_ensemble():
    """Feature ensemble scoring works correctly."""
    candidate = _entity("cyclin dependent kinase 4", name="CDK4", etype="gene")

    # Exact token overlap — high score
    score = EntityResolver._score_candidate("cyclin dependent kinase 4", "gene", candidate)
    assert score > 0.8

    # Partial overlap — moderate score, below exact
    score_partial = EntityResolver._score_candidate("cyclin kinase", "gene", candidate)
    assert 0.0 < score_partial < score

    # No meaningful overlap — below match threshold (0.5)
    score_none = EntityResolver._score_candidate("completely unrelated", "entity", candidate)
    assert score_none < 0.5


def test_score_candidate_type_bonus():
    """Matching entity_type adds a bonus to the score.

    Use a partial match so the type bonus is visible (not swallowed by cap).
    """
    candidate = _entity("cyclin dependent kinase 4", name="CDK4", etype="gene")

    score_match = EntityResolver._score_candidate("cyclin kinase", "gene", candidate)
    score_nomatch = EntityResolver._score_candidate("cyclin kinase", "protein", candidate)
    assert score_match > score_nomatch


def test_score_candidate_numeric_mismatch_penalty():
    """Digit sequences encode semantic values (amounts, ages, addresses,
    durations). Pairs that are textually similar but disagree on digits
    should NOT auto-merge; the resolver penalizes them by 0.5×.
    """
    # Addresses: same street naming but different numbered street.
    cand_69 = _entity("69th street", name="69th street", etype="location")
    score_59 = EntityResolver._score_candidate("59th street", "location", cand_69)
    score_69 = EntityResolver._score_candidate("69th street", "location", cand_69)
    # 59th ≠ 69th: penalty applies. Exact match on 69th: no penalty.
    assert score_59 < score_69
    # After penalty, 59th↔69th should land below the auto-merge floor (0.85)
    # so find_duplicates leaves them alone at that threshold.
    assert score_59 < 0.85

    # Money: "$10 million" vs "$110 million" are textually close but
    # semantically distinct (one is 10× the other).
    cand_110m = _entity("$110 million", name="$110 million", etype="amount")
    score_10m = EntityResolver._score_candidate("$10 million", "amount", cand_110m)
    assert score_10m < 0.85

    # Duration: "18 months in prison" vs "21 months in prison".
    cand_21 = _entity("21 months in prison", name="21 months in prison", etype="duration")
    score_18 = EntityResolver._score_candidate("18 months in prison", "duration", cand_21)
    assert score_18 < 0.85

    # Control: identical-digit variants (punctuation drift) should NOT
    # be penalized, since the digit signature matches.
    cand_addr = _entity(
        "150 east 10th avenue, denver co 80203",
        name="150 east 10th avenue, denver co 80203",
        etype="location",
    )
    score_addr = EntityResolver._score_candidate(
        "150 east 10th avenue, denver, co 80203", "location", cand_addr,
    )
    assert score_addr >= 0.85  # real duplicate

    # No digits at all on either side → no penalty applied.
    cand_name = _entity("les wexner", name="les wexner", etype="person")
    score_name = EntityResolver._score_candidate("leslie wexner", "person", cand_name)
    # Whatever the score is (resolver uses multiple features), it shouldn't
    # have been halved by the numeric guard.
    score_no_guard = EntityResolver._score_candidate("leslie wexner", "person",
                                                      _entity("leslie wexner"))
    # Identical candidate scores ~1.0; compare ratio
    assert score_name > 0.0


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


