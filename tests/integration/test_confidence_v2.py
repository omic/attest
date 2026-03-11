"""Integration tests for Tier 2 confidence in ContextFrame assembly."""

from attestdb.core.confidence import count_independent_sources


def test_corroborated_claim_higher_confidence(make_db):
    """Corroborated assertions should report multiple independent sources."""
    db = make_db(embedding_dim=None)

    # Same assertion from two independent sources
    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("breast_cancer", "disease"),
        provenance={"source_type": "experimental", "source_id": "PMID:100"},
    )
    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("breast_cancer", "disease"),
        provenance={"source_type": "database_import", "source_id": "hetionet"},
    )

    frame = db.query("BRCA1", depth=1)
    breast_cancer_rel = [
        r for r in frame.direct_relationships
        if r.target.id == "breast_cancer"
    ]
    assert len(breast_cancer_rel) == 1
    assert breast_cancer_rel[0].n_independent_sources == 2
    assert len(breast_cancer_rel[0].source_types) == 2


def test_content_id_groups_corroboration(make_db):
    """claims_by_content_id returns corroborating claims for counting."""
    db = make_db(embedding_dim=None)

    db.ingest(
        subject=("TP53", "gene"),
        predicate=("associates", "relates_to"),
        object=("cancer", "disease"),
        provenance={"source_type": "experimental", "source_id": "lab1"},
    )
    db.ingest(
        subject=("TP53", "gene"),
        predicate=("associates", "relates_to"),
        object=("cancer", "disease"),
        provenance={"source_type": "database_import", "source_id": "diseases_db"},
    )
    db.ingest(
        subject=("TP53", "gene"),
        predicate=("associates", "relates_to"),
        object=("cancer", "disease"),
        provenance={"source_type": "literature", "source_id": "pubmed"},
    )

    claims = db.claims_for("tp53")
    cancer_claims = [c for c in claims if c.object.id == "cancer"]
    assert len(cancer_claims) == 3

    content_id = cancer_claims[0].content_id
    corroborating = db.claims_by_content_id(content_id)
    assert len(corroborating) == 3
    assert count_independent_sources(corroborating) == 3


def test_dependent_sources_grouped(make_db):
    """Claims with shared provenance chain count as 1 independent source."""
    db = make_db(embedding_dim=None)

    # First, ingest a parent claim that both downstream claims will reference
    db.ingest(
        subject=("CDK4", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("cell_cycle", "pathway"),
        provenance={
            "source_type": "experimental",
            "source_id": "raw_data_batch_1",
        },
    )

    # Get the parent claim's ID to use in provenance chains
    parent_claims = db.claims_for("cdk4")
    parent_claim_id = parent_claims[0].claim_id

    # Two claims from different pipeline versions but sharing upstream provenance
    db.ingest(
        subject=("CDK4", "gene"),
        predicate=("interacts", "relates_to"),
        object=("CCND1", "protein"),
        provenance={
            "source_type": "computation",
            "source_id": "pipeline_v1",
            "chain": [parent_claim_id],
        },
    )
    db.ingest(
        subject=("CDK4", "gene"),
        predicate=("interacts", "relates_to"),
        object=("CCND1", "protein"),
        provenance={
            "source_type": "computation",
            "source_id": "pipeline_v2",
            "chain": [parent_claim_id],
        },
    )

    claims = db.claims_for("cdk4")
    ccnd1_claims = [c for c in claims if c.object.id == "ccnd1"]
    assert len(ccnd1_claims) == 2

    # They share upstream parent_claim_id, so count as 1 independent source
    assert count_independent_sources(ccnd1_claims) == 1


def test_corroboration_report(make_db):
    """corroboration_report() identifies corroborated and single-source claims."""
    db = make_db(embedding_dim=None)

    # Corroborated: 2 independent sources for same fact
    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("breast_cancer", "disease"),
        provenance={"source_type": "experimental", "source_id": "lab1"},
    )
    db.ingest(
        subject=("BRCA1", "gene"),
        predicate=("associated_with", "relates_to"),
        object=("breast_cancer", "disease"),
        provenance={"source_type": "database_import", "source_id": "hetionet"},
    )

    # Single-source
    db.ingest(
        subject=("TP53", "gene"),
        predicate=("inhibits", "relates_to"),
        object=("MDM2", "gene"),
        provenance={"source_type": "experimental", "source_id": "lab2"},
    )

    report = db.corroboration_report(min_sources=2)
    assert report["corroborated_count"] >= 1
    assert report["single_source_count"] >= 1
    assert report["corroboration_ratio"] > 0

    # The BRCA1-breast_cancer fact should be corroborated
    corr = report["corroborated"]
    brca_corr = [c for c in corr if c["subject"] == "brca1" and c["object"] == "breast_cancer"]
    assert len(brca_corr) == 1
    assert brca_corr[0]["n_independent_sources"] == 2
    assert brca_corr[0]["confidence_boost"] > 1.0
