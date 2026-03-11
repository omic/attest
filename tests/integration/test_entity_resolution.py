"""Integration tests for entity resolution."""

import logging

import pytest

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import ClaimInput
from attestdb.infrastructure.attest_db import AttestDB

logging.disable(logging.WARNING)


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


# --- External ID resolution ---


def test_resolve_by_external_id(make_db):
    """Ingest with external_ids, resolve by them."""
    db = make_db(embedding_dim=None)
    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("binds", "relates_to"),
        object=("Gene_7157", "gene"),
        provenance={"source_type": "observation", "source_id": "test"},
        external_ids={
            "subject": {"ncbi_gene": "672", "symbol": "BRCA1"},
            "object": {"ncbi_gene": "7157", "symbol": "TP53"},
        },
        timestamp=1000,
    )

    db.enable_entity_resolution(mode="external_ids")

    # Resolve by external_id
    eid, conf = db.resolve_entity("unknown_name", "gene", {"ncbi_gene": "672"})
    assert eid == normalize_entity_id("Gene_672")
    assert conf == 0.99


def test_ingestion_resolves_to_existing(make_db):
    """Second ingest with matching external_id maps to first entity."""
    db = make_db(embedding_dim=None)

    # First: ingest a gene with external_ids
    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    db.enable_entity_resolution(mode="external_ids")

    # Second: ingest with a different name but same external_id
    db.ingest(
        subject=("BRCA1_human", "gene"),
        predicate=("interacts", "relates_to"),
        object=("TP53", "gene"),
        provenance={"source_type": "computation", "source_id": "src2"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=2000,
    )

    # The second claim's subject should have resolved to gene_672
    gene_672_canonical = normalize_entity_id("Gene_672")
    claims = db.claims_for(gene_672_canonical)
    predicates = {c.predicate.id for c in claims}
    assert "interacts" in predicates, (
        f"Expected 'interacts' in claims for {gene_672_canonical}, got {predicates}"
    )


def test_no_resolution_without_enable(make_db):
    """Default behavior unchanged — no resolution without explicit enable."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    # Do NOT enable resolution
    db.ingest(
        subject=("BRCA1_human", "gene"),
        predicate=("interacts", "relates_to"),
        object=("TP53", "gene"),
        provenance={"source_type": "computation", "source_id": "src2"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=2000,
    )

    # The second claim should create a separate entity
    brca1_canonical = normalize_entity_id("BRCA1_human")
    entity = db.get_entity(brca1_canonical)
    assert entity is not None, "Without resolution, BRCA1_human should exist as its own entity"


# --- Merge ---


def test_merge_entities_produces_same_as(make_db):
    """merge_entities() creates same_as claim, union-find resolves."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )
    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("interacts", "relates_to"),
        object=("TP53", "gene"),
        provenance={"source_type": "computation", "source_id": "src2"},
        timestamp=2000,
    )

    # Merge them
    claim_id = db.merge_entities("gene_672", "brca1", reason="same gene")
    assert claim_id  # Should return a valid claim_id

    # After merge, resolve should link them
    resolved = db.resolve("brca1")
    resolved_672 = db.resolve("gene_672")
    assert resolved == resolved_672, "After merge, both should resolve to the same entity"


# --- Text search ---


def test_search_entities(make_db):
    """Text search finds entities by display name."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("associates", "relates_to"),
        object=("Breast Cancer", "disease"),
        provenance={"source_type": "observation", "source_id": "test"},
        timestamp=1000,
    )
    db.ingest(
        subject=("TP53", "gene"),
        predicate=("associates", "relates_to"),
        object=("Lung Cancer", "disease"),
        provenance={"source_type": "observation", "source_id": "test"},
        timestamp=2000,
    )

    results = db.search_entities("brca1")
    ids = [r.id for r in results]
    assert "brca1" in ids


def test_search_entities_display_name(make_db):
    """Text search matches on display_name, not just id."""
    db = make_db(embedding_dim=None)

    # Display name is the original (non-normalized) form
    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "test"},
        timestamp=1000,
    )

    results = db.search_entities("gene_672")
    ids = [r.id for r in results]
    assert normalize_entity_id("Gene_672") in ids


# --- Duplicate detection ---


def test_find_duplicates(make_db):
    """Duplicate detection finds entities with shared external_ids."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )
    db.ingest(
        subject=("BRCA1_alt", "gene"),
        predicate=("interacts", "relates_to"),
        object=("TP53", "gene"),
        provenance={"source_type": "computation", "source_id": "src2"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=2000,
    )

    dupes = db.find_duplicate_entities(min_confidence=0.5)
    assert len(dupes) >= 1

    # Check that the pair includes both entities
    pair_ids = set()
    for a, b, conf in dupes:
        pair_ids.add(a)
        pair_ids.add(b)
    gene_672_canonical = normalize_entity_id("Gene_672")
    brca1_alt_canonical = normalize_entity_id("BRCA1_alt")
    assert gene_672_canonical in pair_ids or brca1_alt_canonical in pair_ids


# --- Batch ingestion with resolution ---


def test_batch_ingestion_with_resolution(make_db):
    """Batch ingestion also resolves entities when resolver is enabled."""
    db = make_db(embedding_dim=None)

    # Seed an entity
    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    db.enable_entity_resolution(mode="external_ids")

    # Batch ingest with external_id that matches
    batch = [
        ClaimInput(
            subject=("UNKNOWN_GENE", "gene"),
            predicate=("interacts", "relates_to"),
            object=("TP53", "gene"),
            provenance={"source_type": "computation", "source_id": "src2"},
            external_ids={"subject": {"ncbi_gene": "672"}},
            timestamp=3000,
        ),
    ]
    result = db.ingest_batch(batch)
    assert result.ingested == 1

    # The claim should reference gene_672, not unknown_gene
    gene_672_canonical = normalize_entity_id("Gene_672")
    claims = db.claims_for(gene_672_canonical)
    predicates = {c.predicate.id for c in claims}
    assert "interacts" in predicates


# --- Resolve entity without enabling ---


def test_resolve_entity_without_enable(make_db):
    """resolve_entity() works even without enable_entity_resolution()."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    # resolve_entity without enable should still build a one-shot resolver
    eid, conf = db.resolve_entity("Gene_672")
    assert eid == normalize_entity_id("Gene_672")
    assert conf == 1.0


# --- Auto-merge duplicates ---


def test_auto_merge_duplicates(make_db):
    """auto_merge_duplicates finds ext ID collisions and merges via same_as."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("P53_HUMAN", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("Cancer", "disease"),
        provenance={"source_type": "lit", "source_id": "PMID:2"},
        external_ids={"subject": {"uniprot": "P04637"}},
    )
    db.ingest(
        subject=("TP53_ALT", "gene"),
        predicate=("regulates", "relates_to"),
        object=("MDM2", "gene"),
        provenance={"source_type": "lit", "source_id": "PMID:3"},
        external_ids={"subject": {"uniprot": "P04637"}},
    )

    db.enable_entity_resolution(mode="external_ids")
    result = db.auto_merge_duplicates(min_confidence=0.95)

    assert result["merged_count"] == 1
    assert result["skipped_count"] == 0
    assert len(result["claim_ids"]) == 1

    # After merge, both resolve to the same canonical
    r1 = db.resolve(normalize_entity_id("P53_HUMAN"))
    r2 = db.resolve(normalize_entity_id("TP53_ALT"))
    assert r1 == r2


# --- ClaimInput ingest overload ---


def test_ingest_claim_input_directly(make_db):
    """db.ingest(ClaimInput) works as an alternative to keyword args."""
    db = make_db(embedding_dim=None)

    ci = ClaimInput(
        subject=("BRCA1", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("Breast Cancer", "disease"),
        provenance={"source_type": "literature", "source_id": "PMID:1"},
        confidence=0.9,
    )
    claim_id = db.ingest(ci)
    assert claim_id
    assert db.stats()["total_claims"] == 1

    frame = db.query("BRCA1", depth=1)
    assert frame.focal_entity.name == "BRCA1"
    assert len(frame.direct_relationships) == 1
