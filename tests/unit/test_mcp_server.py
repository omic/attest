"""Tests for the Attest MCP server."""

import json

import pytest

from attestdb.infrastructure.attest_db import AttestDB
from attestdb.mcp_server import (
    configure,
    mcp,
    ingest_claim,
    query_entity,
    search_entities,
    get_entity,
    claims_for,
    find_paths,
    retract_source,
    quality_report,
    knowledge_health,
    ingest_batch,
    ingest_text,
    find_bridges,
    find_gaps,
    list_all_entities,
    get_schema,
)


@pytest.fixture()
def db(tmp_path):
    db = AttestDB(str(tmp_path / "test"), embedding_dim=None)
    configure(db)
    yield db
    db.close()


class TestToolRegistration:
    def test_all_tools_registered(self):
        """All 31 tools should be registered on the FastMCP instance."""
        tools = mcp._tool_manager.list_tools()
        names = {t.name for t in tools}
        expected = {
            # Core (16)
            "ingest_claim", "ingest_text", "ingest_batch",
            "query_entity", "search_entities", "get_entity", "claims_for",
            "find_paths", "retract_source", "quality_report", "knowledge_health",
            "find_bridges", "find_gaps", "schema", "stats",
            "attest_ask",
            # Analysis (9)
            "attest_impact", "attest_blindspots", "attest_consensus",
            "attest_fragile", "attest_stale", "attest_audit",
            "attest_drift", "attest_source_reliability", "attest_hypothetical",
            # Research (2)
            "attest_investigate", "attest_research",
            # Learning Layer (4)
            "attest_observe_session", "attest_record_outcome",
            "attest_get_prior_approaches", "attest_confidence_trail",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"
        assert len(names) == 31, f"Expected 31 tools, got {len(names)}"


class TestToolExecution:
    def test_ingest_claim(self, db):
        result = ingest_claim(
            subject_id="TP53", subject_type="gene",
            predicate_id="interacts_with", predicate_type="interaction",
            object_id="MDM2", object_type="gene",
            source_type="paper", source_id="pmid:12345",
        )
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_query_entity(self, db):
        ingest_claim(
            subject_id="TP53", subject_type="gene",
            predicate_id="interacts_with", predicate_type="interaction",
            object_id="MDM2", object_type="gene",
            source_type="paper", source_id="pmid:12345",
        )
        result = json.loads(query_entity("TP53"))
        assert result["entity"] == "tp53"
        assert result["claim_count"] >= 1
        assert isinstance(result["relationships"], list)

    def test_search_entities(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="s1",
        )
        result = json.loads(search_entities())
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_get_entity(self, db):
        ingest_claim(
            subject_id="X", subject_type="thing",
            predicate_id="has", predicate_type="relation",
            object_id="Y", object_type="thing",
            source_type="test", source_id="s1",
        )
        result = json.loads(get_entity("X"))
        assert result["id"] == "x"
        assert result["type"] == "thing"
        assert "claim_count" in result

    def test_claims_for(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="s1",
        )
        result = json.loads(claims_for("A"))
        assert isinstance(result, list)
        assert len(result) >= 1
        assert "claim_id" in result[0]

    def test_retract_source(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="src_to_retract",
        )
        result = json.loads(retract_source("src_to_retract", "test retraction"))
        assert result["source_id"] == "src_to_retract"
        assert result["retracted_count"] >= 1

    def test_quality_report(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="s1",
        )
        result = json.loads(quality_report())
        assert "total_claims" in result

    def test_knowledge_health(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="s1",
        )
        result = json.loads(knowledge_health())
        assert "health_score" in result
        assert 0 <= result["health_score"] <= 100

    def test_ingest_batch(self, db):
        claims = [
            {
                "subject": ["A", "entity"],
                "predicate": ["rel", "relation"],
                "object": ["B", "entity"],
                "provenance": {"source_type": "test", "source_id": "s1"},
            },
            {
                "subject": ["C", "entity"],
                "predicate": ["rel", "relation"],
                "object": ["D", "entity"],
                "provenance": {"source_type": "test", "source_id": "s2"},
            },
        ]
        result = json.loads(ingest_batch(claims))
        assert result["ingested"] == 2

    def test_find_paths(self, db):
        ingest_claim(
            subject_id="A", subject_type="entity",
            predicate_id="rel", predicate_type="relation",
            object_id="B", object_type="entity",
            source_type="test", source_id="s1",
        )
        result = json.loads(find_paths("A", "B"))
        assert isinstance(result, list)


class TestResources:
    def test_list_all_entities(self, db):
        ingest_claim(
            subject_id="P", subject_type="protein",
            predicate_id="binds", predicate_type="interaction",
            object_id="Q", object_type="protein",
            source_type="experiment", source_id="exp1",
        )
        result = json.loads(list_all_entities())
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_get_schema_resource(self, db):
        result = json.loads(get_schema())
        assert "entity_types" in result
