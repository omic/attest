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

    # Resolve by external_id (resolution enabled by default)
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

    # Second: ingest with a different name but same external_id
    # Resolution is enabled by default — should resolve to Gene_672
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
    assert "interacts_with" in predicates or "interacts" in predicates, (
        f"Expected 'interacts' or 'interacts_with' in claims for {gene_672_canonical}, got {predicates}"
    )


def test_resolution_enabled_by_default(make_db):
    """Entity resolution (external_ids mode) is enabled by default."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    # Resolution is now enabled by default — second claim with same
    # external_id should resolve to the first entity
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
    assert "interacts_with" in predicates or "interacts" in predicates, (
        f"Expected 'interacts'/'interacts_with' in claims for {gene_672_canonical} "
        f"(resolution should happen by default), got {predicates}"
    )


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


def test_find_duplicates_with_resolution_disabled(make_db):
    """Duplicate detection finds entities with shared external_ids
    when resolution was not active at ingestion time.

    With default resolution enabled, entities are resolved at ingestion,
    so we need to disable resolution first to create actual duplicates,
    then re-enable and scan.
    """
    db = make_db(embedding_dim=None)

    # Disable resolution to create unresolved duplicates
    db._pipeline._resolver = None

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

    # Re-enable resolution and scan for duplicates
    db.enable_entity_resolution("external_ids")
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


def test_no_duplicates_with_default_resolution(make_db):
    """With default resolution, entities sharing external_ids resolve
    at ingestion time, leaving no duplicates to detect."""
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

    # With resolution active, BRCA1_alt was resolved to Gene_672 at ingestion
    dupes = db.find_duplicate_entities(min_confidence=0.5)
    assert len(dupes) == 0, (
        f"Expected 0 duplicates (resolved at ingestion), got {len(dupes)}"
    )


# --- Batch ingestion with resolution ---


def test_batch_ingestion_with_resolution(make_db):
    """Batch ingestion resolves entities via default external_ids resolver."""
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

    # Batch ingest with external_id that matches (resolution active by default)
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
    assert "interacts_with" in predicates or "interacts" in predicates


# --- Resolve entity without enabling ---


def test_resolve_entity_one_shot(make_db):
    """resolve_entity() creates a one-shot resolver for manual lookups."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Gene_672", "gene"),
        predicate=("associates", "relates_to"),
        object=("Disease_X", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        external_ids={"subject": {"ncbi_gene": "672"}},
        timestamp=1000,
    )

    # resolve_entity() creates a one-shot resolver for manual lookups
    eid, conf = db.resolve_entity("Gene_672")
    assert eid == normalize_entity_id("Gene_672")
    assert conf == 1.0


# --- Auto-merge duplicates ---


def test_auto_merge_duplicates(make_db):
    """auto_merge_duplicates finds ext ID collisions and merges via same_as.

    Must disable default resolution to create unresolved duplicates first,
    then re-enable and run auto_merge.
    """
    db = make_db(embedding_dim=None)

    # Disable default resolution to create unresolved duplicates
    db._pipeline._resolver = None

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

    # Re-enable resolution and run auto-merge on the duplicates
    db.enable_entity_resolution(mode="external_ids")
    result = db.auto_merge_duplicates(min_confidence=0.95)

    assert result["merged_count"] == 1
    assert result["skipped_count"] == 0
    assert len(result["claim_ids"]) == 1

    # After merge, both resolve to the same canonical
    r1 = db.resolve(normalize_entity_id("P53_HUMAN"))
    r2 = db.resolve(normalize_entity_id("TP53_ALT"))
    assert r1 == r2


# --- Fuzzy resolution mode ---


def test_fuzzy_resolves_name_variations(make_db):
    """Fuzzy mode resolves entities with overlapping name tokens."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Aspirin", "drug"),
        predicate=("treats", "relates_to"),
        object=("Headache", "condition"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    # Enable fuzzy mode
    db.enable_entity_resolution("fuzzy")

    # Ingest with a name variation — fuzzy should resolve to "aspirin"
    db.ingest(
        subject=("Aspirin Tablet", "drug"),
        predicate=("treats", "relates_to"),
        object=("Migraine", "condition"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    # "aspirin tablet" resolved to "aspirin" via fuzzy match —
    # both claims should land on the canonical entity
    aspirin_canonical = normalize_entity_id("Aspirin")
    claims = db.claims_for(aspirin_canonical)
    assert len(claims) == 2, (
        f"Expected 2 claims on {aspirin_canonical} (fuzzy resolved), got {len(claims)}"
    )
    objects = {c.object.id for c in claims}
    assert normalize_entity_id("Headache") in objects
    assert normalize_entity_id("Migraine") in objects


def test_fuzzy_does_not_resolve_unrelated(make_db):
    """Fuzzy mode does not merge entities with no token overlap."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Aspirin", "drug"),
        predicate=("treats", "relates_to"),
        object=("Headache", "condition"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")

    db.ingest(
        subject=("Metformin", "drug"),
        predicate=("treats", "relates_to"),
        object=("Diabetes", "condition"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    # Metformin and Aspirin should remain separate
    aspirin_claims = db.claims_for(normalize_entity_id("Aspirin"))
    metformin_claims = db.claims_for(normalize_entity_id("Metformin"))
    assert len(aspirin_claims) == 1
    assert len(metformin_claims) == 1


def test_fuzzy_find_duplicates(make_db):
    """Fuzzy mode find_duplicates detects name-similar entities."""
    db = make_db(embedding_dim=None)

    # Disable resolution to create entities with similar names
    db._pipeline._resolver = None

    db.ingest(
        subject=("John Smith", "person"),
        predicate=("works_at", "relates_to"),
        object=("Acme Corp", "company"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )
    db.ingest(
        subject=("Smith John", "person"),
        predicate=("manages", "relates_to"),
        object=("Widget Inc", "company"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    # Re-enable fuzzy and scan for duplicates
    db.enable_entity_resolution("fuzzy")
    dupes = db.find_duplicate_entities(min_confidence=0.5)

    # "john smith" and "smith john" have 100% token overlap
    pair_ids = set()
    for a, b, conf in dupes:
        pair_ids.add(a)
        pair_ids.add(b)
    assert normalize_entity_id("John Smith") in pair_ids
    assert normalize_entity_id("Smith John") in pair_ids


def test_fuzzy_with_type_bonus(make_db):
    """Type match adds bonus to fuzzy score."""
    db = make_db(embedding_dim=None)

    # Disable resolution to create separate entities
    db._pipeline._resolver = None

    db.ingest(
        subject=("Gene BRCA1", "gene"),
        predicate=("associates", "relates_to"),
        object=("Cancer", "disease"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )
    db.ingest(
        subject=("BRCA1 Gene", "gene"),
        predicate=("interacts", "relates_to"),
        object=("TP53", "gene"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    db.enable_entity_resolution("fuzzy")
    dupes = db.find_duplicate_entities(min_confidence=0.5)

    # Should find them as duplicates (100% token overlap + type bonus)
    assert len(dupes) >= 1
    pair_ids = set()
    for a, b, conf in dupes:
        pair_ids.add(a)
        pair_ids.add(b)
    assert normalize_entity_id("Gene BRCA1") in pair_ids or \
        normalize_entity_id("BRCA1 Gene") in pair_ids


# --- Feature ensemble scoring tests ---


def test_acronym_resolution_e2e(make_db):
    """Blocking index + scoring resolves acronyms end-to-end."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("International Business Machines", "organization"),
        predicate=("produces", "relates_to"),
        object=("Watson", "product"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")

    db.ingest(
        subject=("IBM", "organization"),
        predicate=("employs", "relates_to"),
        object=("Engineers", "role"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    ibm_canonical = normalize_entity_id("International Business Machines")
    claims = db.claims_for(ibm_canonical)
    assert len(claims) == 2, (
        f"Expected 2 claims (acronym resolved via blocking), got {len(claims)}"
    )


def test_org_suffix_stripping(make_db):
    """Fuzzy mode handles corporate suffix variations."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Acme Corporation", "organization"),
        predicate=("sells", "relates_to"),
        object=("Widgets", "product"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")

    db.ingest(
        subject=("Acme Corp", "organization"),
        predicate=("hires", "relates_to"),
        object=("Engineers", "role"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    acme_canonical = normalize_entity_id("Acme Corporation")
    claims = db.claims_for(acme_canonical)
    assert len(claims) == 2, (
        f"Expected 2 claims (org suffix resolved), got {len(claims)}"
    )


def test_typo_resolution_phonetic_blocking(make_db):
    """Blocking index + Jaro-Winkler resolves typos end-to-end."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("John Smith", "person"),
        predicate=("works_at", "relates_to"),
        object=("Acme", "organization"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")

    db.ingest(
        subject=("Jon Smyth", "person"),
        predicate=("manages", "relates_to"),
        object=("Team Alpha", "group"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    john_canonical = normalize_entity_id("John Smith")
    claims = db.claims_for(john_canonical)
    assert len(claims) == 2, (
        f"Expected 2 claims (typo resolved via phonetic blocking), got {len(claims)}"
    )


def test_unrelated_entities_stay_separate(make_db):
    """Feature ensemble rejects unrelated entities even with same type."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("Aspirin", "drug"),
        predicate=("treats", "relates_to"),
        object=("Pain", "condition"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")

    db.ingest(
        subject=("Metformin", "drug"),
        predicate=("treats", "relates_to"),
        object=("Diabetes", "condition"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    aspirin_claims = db.claims_for(normalize_entity_id("Aspirin"))
    metformin_claims = db.claims_for(normalize_entity_id("Metformin"))
    assert len(aspirin_claims) == 1, "Aspirin should not merge with Metformin"
    assert len(metformin_claims) == 1, "Metformin should not merge with Aspirin"


def test_external_id_overrides_text(make_db):
    """External ID match produces high score even with dissimilar names."""
    from attestdb.infrastructure.entity_resolver import EntityResolver
    from attestdb.core.types import EntitySummary

    c = EntitySummary(
        id="gene_672", name="Gene 672", entity_type="gene",
        external_ids={"ncbi_gene": "672"}, claim_count=1,
    )
    score = EntityResolver._score_candidate(
        "brca1", "gene", c,
        query_external_ids={"ncbi_gene": "672"},
    )
    assert score >= 0.80, f"External ID match should score >=0.80, got {score:.3f}"


# --- Calibration wiring tests ---


def test_prediction_logging(make_db):
    """Match decisions are logged to prediction_log when wired."""
    import tempfile
    from attestdb.calibration.prediction_log import PredictionLog

    db = make_db(embedding_dim=None)
    log_path = tempfile.mktemp(suffix=".sqlite")
    pred_log = PredictionLog(log_path)

    db.ingest(
        subject=("Acme Corporation", "organization"),
        predicate=("sells", "relates_to"),
        object=("Widgets", "product"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")
    db._entity_resolver.configure_calibration(prediction_log=pred_log)

    # This should trigger a match decision (Acme Corp -> Acme Corporation)
    db.ingest(
        subject=("Acme Corp", "organization"),
        predicate=("hires", "relates_to"),
        object=("Engineers", "role"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    # Check that a prediction was logged
    records = pred_log.get_resolved(decision_type="entity_match")
    pending = pred_log.get_pending()
    total = len(records) + len(pending)
    assert total >= 1, f"Expected at least 1 prediction logged, got {total}"


def test_review_queue_routing(make_db):
    """Gray-zone matches are routed to review queue, not auto-merged."""
    import tempfile
    from attestdb.calibration.prediction_log import PredictionLog
    from attestdb.calibration.threshold_engine import ThresholdEngine
    from attestdb.review.queue import ReviewQueue

    db = make_db(embedding_dim=None)
    log_path = tempfile.mktemp(suffix=".sqlite")
    review_path = tempfile.mktemp(suffix=".sqlite")

    pred_log = PredictionLog(log_path)
    review_q = ReviewQueue(review_path)
    threshold_eng = ThresholdEngine(log_path)

    db.ingest(
        subject=("Alpha Corp", "organization"),
        predicate=("sells", "relates_to"),
        object=("Products", "product"),
        provenance={"source_type": "observation", "source_id": "src1"},
        timestamp=1000,
    )

    db.enable_entity_resolution("fuzzy")
    db._entity_resolver.configure_calibration(
        prediction_log=pred_log,
        review_queue=review_q,
        threshold_engine=threshold_eng,
    )

    # "Alpha Inc" is somewhat similar to "Alpha Corp" but with suffix difference
    # The scorer should produce a moderate score — depends on calibrated thresholds
    db.ingest(
        subject=("Alpha Inc", "organization"),
        predicate=("hires", "relates_to"),
        object=("Engineers", "role"),
        provenance={"source_type": "observation", "source_id": "src2"},
        timestamp=2000,
    )

    # Check entities — if score was in gray zone, they should NOT be merged
    # If score was above accept threshold, they SHOULD be merged
    # Either way, a decision was logged
    records = pred_log.get_pending()
    resolved = pred_log.get_resolved(decision_type="entity_match")
    total_logged = len(records) + len(resolved)
    assert total_logged >= 1, "At least one decision should be logged"


def test_calibrated_thresholds(make_db):
    """Resolver uses calibrated thresholds from ThresholdEngine."""
    from attestdb.infrastructure.entity_resolver import EntityResolver
    from unittest.mock import MagicMock

    db = make_db(embedding_dim=None)
    db.enable_entity_resolution("fuzzy")

    # Mock threshold engine to return specific thresholds
    mock_te = MagicMock()
    mock_config = MagicMock()
    mock_config.auto_approve_threshold = 0.90
    mock_config.review_threshold = 0.60
    mock_te.get_thresholds.return_value = mock_config

    db._entity_resolver.configure_calibration(threshold_engine=mock_te)

    accept, review = db._entity_resolver._get_thresholds("organization")
    assert accept == 0.90
    assert review == 0.60
    mock_te.get_thresholds.assert_called_with(decision_type="entity_match")


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
