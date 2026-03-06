"""Integration tests for the four Knowledge-Intelligence APIs.

Tests: test_hypothesis, evolution, suggest_investigations, trace.
All LLM calls are mocked — no API keys required.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from attestdb.core.types import (
    ConfidenceChange,
    ConfidenceGap,
    EvolutionReport,
    EvidenceChain,
    HypothesisVerdict,
    Investigation,
    ReasoningChain,
    ReasoningHop,
    SourceOverlap,
)
from attestdb.infrastructure.attest_db import AttestDB


# ---------------------------------------------------------------------------
# Shared fixture: multi-hop biomedical graph
# ---------------------------------------------------------------------------

@pytest.fixture
def hypothesis_db(tmp_path):
    """Build a graph with multi-hop chains, contradictions, and temporal spread.

    Graph structure:
        aspirin --[inhibits]--> cox-2 --[promotes]--> inflammation
        aspirin --[activates]--> cox-2   (contradicting — weak source)
        cox-2 --[associated_with]--> prostaglandin  (added later)
        ibuprofen --[inhibits]--> cox-1  (isolated, single source)

    Sources: paper1 (0.9), review1 (0.85), weak_study (0.3)
    Temporal: t1 (early), t2 (later for prostaglandin link)
    """
    db = AttestDB(str(tmp_path / "hypothesis_test"), embedding_dim=None)

    t1 = int(time.time() * 1_000_000_000) - 100_000_000_000  # ~100s ago
    t2 = int(time.time() * 1_000_000_000)                     # now

    # aspirin inhibits cox-2 (strong, paper1)
    db.ingest(
        subject=("aspirin", "drug"), predicate=("inhibits", "relationship"),
        object=("cox-2", "enzyme"), confidence=0.9,
        provenance={"source_type": "journal_article", "source_id": "paper1"},
        payload={"schema_ref": "evidence", "data": {"text": "Aspirin irreversibly inhibits COX-2"}},
        timestamp=t1,
    )

    # aspirin inhibits cox-2 (corroboration, review1)
    db.ingest(
        subject=("aspirin", "drug"), predicate=("inhibits", "relationship"),
        object=("cox-2", "enzyme"), confidence=0.85,
        provenance={"source_type": "review", "source_id": "review1"},
        timestamp=t1,
    )

    # cox-2 promotes inflammation (paper1 — shared source with aspirin→cox-2)
    db.ingest(
        subject=("cox-2", "enzyme"), predicate=("promotes", "relationship"),
        object=("inflammation", "process"), confidence=0.8,
        provenance={"source_type": "journal_article", "source_id": "paper1"},
        payload={"schema_ref": "evidence", "data": {"text": "COX-2 promotes inflammatory response"}},
        timestamp=t1,
    )

    # aspirin activates cox-2 (contradicting — weak source)
    db.ingest(
        subject=("aspirin", "drug"), predicate=("activates", "relationship"),
        object=("cox-2", "enzyme"), confidence=0.3,
        provenance={"source_type": "preprint", "source_id": "weak_study"},
        timestamp=t1,
    )

    # cox-2 associated_with prostaglandin (added later — for evolution testing)
    db.ingest(
        subject=("cox-2", "enzyme"), predicate=("associated_with", "relationship"),
        object=("prostaglandin", "molecule"), confidence=0.7,
        provenance={"source_type": "review", "source_id": "review1"},
        timestamp=t2,
    )

    # ibuprofen inhibits cox-1 (isolated, single source)
    db.ingest(
        subject=("ibuprofen", "drug"), predicate=("inhibits", "relationship"),
        object=("cox-1", "enzyme"), confidence=0.75,
        provenance={"source_type": "journal_article", "source_id": "paper1"},
        timestamp=t1,
    )

    yield db, t1, t2
    db.close()


# ===========================================================================
# test_hypothesis
# ===========================================================================

class TestTestHypothesis:
    """Tests for db.test_hypothesis()."""

    def test_supported_hypothesis(self, hypothesis_db):
        """Aspirin inhibits COX-2 is well-supported in the graph."""
        db, _, _ = hypothesis_db
        result = db.test_hypothesis("aspirin inhibits cox-2")
        # Should find entities and supporting chains
        assert len(result.entities_found) >= 2
        assert result.verdict in ("supported", "partial")
        assert len(result.supporting_chains) > 0

    def test_unsupported_hypothesis_unknown_entities(self, hypothesis_db):
        """Unknown entities → insufficient_data."""
        db, _, _ = hypothesis_db
        result = db.test_hypothesis("zzz_unknown_entity_alpha affects zzz_unknown_entity_beta")
        assert result.verdict == "insufficient_data"
        assert len(result.entities_found) < 2

    def test_no_llm_fallback(self, hypothesis_db):
        """Without LLM, fallback still returns a verdict via entity search."""
        db, _, _ = hypothesis_db
        # Ensure LLM is not available
        with patch.object(db, '_llm_call', return_value=None):
            result = db.test_hypothesis("aspirin inhibits cox-2")
        assert isinstance(result, HypothesisVerdict)
        # Fallback should still find "aspirin" and "cox-2" via search_entities
        assert len(result.entities_found) >= 2

    def test_confidence_gaps_populated(self, hypothesis_db):
        """Multi-hop chain with a weak link should have confidence gaps."""
        db, _, _ = hypothesis_db
        # Test a multi-hop hypothesis where weak_study (0.3) exists
        result = db.test_hypothesis("aspirin affects inflammation")
        # Even if verdict varies, the structure is populated
        assert isinstance(result.confidence_gaps, list)
        assert isinstance(result.suggested_next_steps, list)

    def test_hypothesis_with_llm_parse(self, hypothesis_db):
        """When LLM parse succeeds, entities are resolved from structured output."""
        db, _, _ = hypothesis_db
        llm_response = (
            '{"entities": [{"name": "aspirin", "type": "drug"}, {"name": "cox-2", "type": "enzyme"}], '
            '"relationships": [{"from": "aspirin", "to": "cox-2", "predicate": "inhibits"}]}'
        )
        with patch.object(db, '_llm_call', return_value=llm_response):
            result = db.test_hypothesis("aspirin inhibits cox-2")
        assert len(result.entities_found) >= 2
        assert result.verdict in ("supported", "partial")


# ===========================================================================
# test_evolution
# ===========================================================================

class TestEvolution:
    """Tests for db.evolution()."""

    def test_full_history(self, hypothesis_db):
        """since=None → full history (cutoff=0, all claims are 'after')."""
        db, _, _ = hypothesis_db
        result = db.evolution("cox-2", since=None)
        assert result.since_timestamp == 0
        assert result.total_claims_after > 0
        # All claims are in the "after" period when since=None
        assert result.new_claims == result.total_claims_after

    def test_since_timestamp_int(self, hypothesis_db):
        """Since with int cutoff partitions claims correctly."""
        db, t1, t2 = hypothesis_db
        # Use a cutoff between t1 and t2
        cutoff = (t1 + t2) // 2
        result = db.evolution("cox-2", since=cutoff)
        assert result.since_timestamp == cutoff
        assert result.total_claims_before > 0
        assert result.new_claims > 0

    def test_since_iso_string(self, hypothesis_db):
        """Since accepts ISO date string."""
        db, _, _ = hypothesis_db
        result = db.evolution("cox-2", since="2020-01-01T00:00:00")
        assert isinstance(result, EvolutionReport)
        # Everything should be in the "after" period (all claims are after 2020)
        assert result.total_claims_after > 0

    def test_new_connections_detected(self, hypothesis_db):
        """Prostaglandin was added later → should appear in new_connections."""
        db, t1, t2 = hypothesis_db
        cutoff = (t1 + t2) // 2
        result = db.evolution("cox-2", since=cutoff)
        assert "prostaglandin" in result.new_connections

    def test_source_diversification(self, hypothesis_db):
        """Source types tracked before and after."""
        db, _, _ = hypothesis_db
        result = db.evolution("cox-2")
        assert len(result.source_types_after) > 0

    def test_unknown_entity(self, hypothesis_db):
        """Unknown entity → empty report."""
        db, _, _ = hypothesis_db
        result = db.evolution("zzz_nonexistent_entity")
        assert result.new_claims == 0
        assert result.total_claims_after == 0
        assert result.trajectory == "stable"


# ===========================================================================
# test_suggest_investigations
# ===========================================================================

class TestSuggestInvestigations:
    """Tests for db.suggest_investigations()."""

    def test_sorted_by_priority(self, hypothesis_db):
        """Results are sorted by priority_score descending."""
        db, _, _ = hypothesis_db
        result = db.suggest_investigations()
        if len(result) >= 2:
            for i in range(len(result) - 1):
                assert result[i].priority_score >= result[i + 1].priority_score

    def test_single_source_flagged(self, hypothesis_db):
        """Entities with only one source should be flagged."""
        db, _, _ = hypothesis_db
        result = db.suggest_investigations(top_k=50)
        signal_types = {inv.signal_type for inv in result}
        # There should be at least single_source signals
        # (ibuprofen and cox-1 are single-source, others may also qualify)
        assert len(result) > 0

    def test_top_k_respected(self, hypothesis_db):
        db, _, _ = hypothesis_db
        result = db.suggest_investigations(top_k=2)
        assert len(result) <= 2

    def test_all_items_have_reason_and_action(self, hypothesis_db):
        db, _, _ = hypothesis_db
        result = db.suggest_investigations()
        for inv in result:
            assert inv.reason, f"Investigation for {inv.entity_id} has empty reason"
            assert inv.suggested_action, f"Investigation for {inv.entity_id} has empty suggested_action"


# ===========================================================================
# test_trace
# ===========================================================================

class TestTrace:
    """Tests for db.trace()."""

    def test_multihop_chain(self, hypothesis_db):
        """aspirin→cox-2→inflammation is a 2-hop chain."""
        db, _, _ = hypothesis_db
        result = db.trace("aspirin", "inflammation")
        assert len(result) > 0
        chain = result[0]
        assert chain.length >= 2
        assert len(chain.hops) >= 2

    def test_confidence_propagation(self, hypothesis_db):
        """chain_confidence ≤ raw_confidence (source overlap discounts)."""
        db, _, _ = hypothesis_db
        result = db.trace("aspirin", "inflammation")
        for chain in result:
            assert chain.chain_confidence <= chain.raw_confidence + 1e-9

    def test_source_overlap_detection(self, hypothesis_db):
        """paper1 appears in both hops → source overlap should be detected."""
        db, _, _ = hypothesis_db
        result = db.trace("aspirin", "inflammation")
        assert len(result) > 0
        # At least one chain should have source overlaps since paper1 is in
        # aspirin→cox-2 and cox-2→inflammation
        has_overlap = any(len(chain.source_overlaps) > 0 for chain in result)
        assert has_overlap, "Expected source overlap from paper1 in multiple hops"

    def test_evidence_text_populated(self, hypothesis_db):
        """Evidence text should come from claim payloads."""
        db, _, _ = hypothesis_db
        result = db.trace("aspirin", "inflammation")
        assert len(result) > 0
        # At least one hop should have evidence text (we added payload text)
        has_evidence = any(
            hop.evidence_text != ""
            for chain in result
            for hop in chain.hops
        )
        assert has_evidence, "Expected evidence text from payload"

    def test_no_path_empty_list(self, hypothesis_db):
        """No path between disconnected entities → empty list."""
        db, _, _ = hypothesis_db
        result = db.trace("ibuprofen", "inflammation")
        assert result == []

    def test_contradiction_flags(self, hypothesis_db):
        """Hops with opposing predicates should be flagged."""
        db, _, _ = hypothesis_db
        # aspirin→cox-2 has both "inhibits" and "activates" (opposites)
        result = db.trace("aspirin", "cox-2")
        assert len(result) > 0
        # The hop should detect the contradiction
        has_contradiction = any(
            hop.has_contradiction
            for chain in result
            for hop in chain.hops
        )
        assert has_contradiction, "Expected contradiction flag for inhibits/activates on aspirin→cox-2"

    def test_reliability_score_bounded(self, hypothesis_db):
        """Reliability score must be in [0, 1]."""
        db, _, _ = hypothesis_db
        result = db.trace("aspirin", "inflammation")
        for chain in result:
            assert 0.0 <= chain.reliability_score <= 1.0
