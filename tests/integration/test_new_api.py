"""Tests for the 9 new API methods (Stage 7): impact, blindspots, consensus,
fragile, stale, audit, drift, source_reliability, hypothetical."""

import time

import pytest

from attestdb.core.types import (
    AuditTrail,
    ClaimInput,
    ConsensusReport,
    HypotheticalReport,
    ImpactReport,
)


@pytest.fixture
def rich_db(make_db):
    """DB with multiple sources and overlapping claims for testing new API."""
    db = make_db(embedding_dim=None)

    # Source s1: two claims
    db.ingest(
        subject=("brca1", "gene"), predicate=("binds", "interaction"),
        object=("rad51", "gene"),
        provenance={"source_type": "literature", "source_id": "s1"},
        confidence=0.9, timestamp=1000,
    )
    db.ingest(
        subject=("brca1", "gene"), predicate=("inhibits", "interaction"),
        object=("parp1", "gene"),
        provenance={"source_type": "literature", "source_id": "s1"},
        confidence=0.8, timestamp=2000,
    )

    # Source s2: corroborates the brca1-rad51 binding
    db.ingest(
        subject=("brca1", "gene"), predicate=("binds", "interaction"),
        object=("rad51", "gene"),
        provenance={"source_type": "experiment", "source_id": "s2"},
        confidence=0.95, timestamp=3000,
    )

    # Source s3: independent claim
    db.ingest(
        subject=("tp53", "gene"), predicate=("regulates", "interaction"),
        object=("mdm2", "gene"),
        provenance={"source_type": "computation", "source_id": "s3"},
        confidence=0.7, timestamp=4000,
    )

    return db


# --- impact ---

class TestImpact:
    def test_impact_returns_report(self, rich_db):
        report = rich_db.impact("s1")
        assert isinstance(report, ImpactReport)
        assert report.source_id == "s1"
        assert report.direct_claims == 2

    def test_impact_affected_entities(self, rich_db):
        report = rich_db.impact("s1")
        assert "brca1" in report.affected_entities
        assert "rad51" in report.affected_entities
        assert "parp1" in report.affected_entities


# --- blindspots ---

class TestBlindspots:
    def test_blindspots_single_source(self, rich_db):
        # tp53-mdm2 is only from s3
        report = rich_db.blindspots(min_claims=1)
        # tp53 has 1 claim from 1 source
        assert "tp53" in report.single_source_entities or "mdm2" in report.single_source_entities


# --- consensus ---

class TestConsensus:
    def test_consensus_returns_report(self, rich_db):
        report = rich_db.consensus("brca1")
        assert isinstance(report, ConsensusReport)
        assert report.topic == "brca1"
        assert report.total_claims >= 2

    def test_consensus_multiple_sources(self, rich_db):
        report = rich_db.consensus("brca1")
        assert report.unique_sources >= 2
        assert report.avg_confidence > 0


# --- fragile ---

class TestFragile:
    def test_fragile_finds_single_source_claims(self, rich_db):
        claims = rich_db.fragile(max_sources=1)
        # brca1-parp1 and tp53-mdm2 are single-source
        claim_subjects = {c.subject.id for c in claims}
        claim_objects = {c.object.id for c in claims}
        entities = claim_subjects | claim_objects
        assert "parp1" in entities or "mdm2" in entities

    def test_fragile_excludes_corroborated(self, rich_db):
        claims = rich_db.fragile(max_sources=1)
        # brca1-binds-rad51 has 2 sources, so it shouldn't appear as fragile
        for c in claims:
            if c.subject.id == "brca1" and c.object.id == "rad51" and c.predicate.id == "binds":
                pytest.fail("Corroborated claim should not be fragile")


# --- stale ---

class TestStale:
    def test_stale_with_recent_threshold(self, make_db):
        db = make_db(embedding_dim=None)
        now_ns = int(time.time() * 1_000_000_000)
        # Ingest one recent claim
        db.ingest(
            subject=("a", "entity"), predicate=("links", "rel"),
            object=("b", "entity"),
            provenance={"source_type": "test", "source_id": "t1"},
            confidence=0.9, timestamp=now_ns,
        )
        # This claim is fresh, so stale(days=1) should not find it
        claims = db.stale(days=1)
        assert len(claims) == 0


# --- audit ---

class TestAudit:
    def test_audit_returns_trail(self, rich_db):
        # Get a claim_id first
        claims = rich_db.claims_for("brca1")
        assert len(claims) > 0
        claim = claims[0]

        trail = rich_db.audit(claim.claim_id)
        assert isinstance(trail, AuditTrail)
        assert trail.claim_id == claim.claim_id
        assert trail.content_id != ""
        assert trail.source_type != ""
        assert trail.confidence > 0

    def test_audit_corroborating_claims(self, rich_db):
        # Find the brca1-binds-rad51 claim from s1
        claims = rich_db.claims_for("brca1")
        binds_claims = [c for c in claims if c.predicate.id == "binds" and c.provenance.source_id == "s1"]
        if binds_claims:
            trail = rich_db.audit(binds_claims[0].claim_id)
            # There should be a corroborating claim from s2
            assert len(trail.corroborating_claims) >= 1


# --- drift ---

class TestDrift:
    def test_drift_claim_counts(self, rich_db):
        report = rich_db.drift(days=30)
        # All claims have old timestamps, so claim_count_after should include all
        assert report.claim_count_after >= 3


# --- source_reliability ---

class TestSourceReliability:
    def test_single_source(self, rich_db):
        result = rich_db.source_reliability(source_id="s1")
        assert "total_claims" in result
        assert result["total_claims"] == 2
        assert "corroboration_rate" in result
        assert "retraction_rate" in result

    def test_source_with_corroboration(self, rich_db):
        result = rich_db.source_reliability(source_id="s1")
        # s1's brca1-binds-rad51 is corroborated by s2
        assert result["corroboration_rate"] > 0


# --- hypothetical ---

class TestHypothetical:
    def test_hypothetical_corroboration(self, rich_db):
        """Hypothetical that matches existing content should flag corroboration."""
        claim = ClaimInput(
            subject=("brca1", "gene"),
            predicate=("binds", "interaction"),
            object=("rad51", "gene"),
            provenance={"source_type": "review", "source_id": "s_hyp"},
        )
        report = rich_db.hypothetical(claim)
        assert isinstance(report, HypotheticalReport)
        assert report.would_corroborate is True
        assert report.existing_corroborations >= 2

    def test_hypothetical_new_claim(self, rich_db):
        """Hypothetical with novel content should not corroborate."""
        claim = ClaimInput(
            subject=("egfr", "gene"),
            predicate=("activates", "interaction"),
            object=("kras", "gene"),
            provenance={"source_type": "review", "source_id": "s_hyp"},
        )
        report = rich_db.hypothetical(claim)
        assert report.would_corroborate is False
        assert report.existing_corroborations == 0

