"""Integration tests for the autodidact self-learning daemon."""

import time

import pytest

from attestdb import AttestDB
from attestdb.core.types import AutodidactStatus, ClaimInput, CycleReport


@pytest.fixture
def db():
    """Create an in-memory AttestDB for testing."""
    _db = AttestDB(":memory:", embedding_dim=None)
    yield _db
    _db.close()


def _seed_single_source_entity(db: AttestDB, entity_id: str = "test_entity") -> None:
    """Seed an entity with claims from only one source so it shows up as a blindspot."""
    claims = []
    for i in range(5):
        claims.append(
            ClaimInput(
                subject=(entity_id, "protein"),
                predicate=(f"interacts_with_{i}", "biological"),
                object=(f"target_{i}", "protein"),
                provenance={"source_id": "single_source", "source_type": "database"},
                confidence=0.8,
            )
        )
    db.ingest_batch(claims)


class TestEnableDisable:
    def test_enable_disable(self, db):
        """Start autodidact, verify enabled, stop, verify stopped."""
        status = db.enable_autodidact(interval=999999, sources="none")
        assert status.enabled is True
        assert status.running is True

        db.disable_autodidact()
        status = db.autodidact_status()
        assert status.enabled is False
        assert status.running is False

    def test_enable_replaces_existing(self, db):
        """Enabling autodidact twice stops the first daemon."""
        db.enable_autodidact(interval=999999, sources="none")
        db.enable_autodidact(interval=999999, sources="none")
        # Should not raise, old daemon should be stopped
        db.disable_autodidact()

    def test_status_when_not_enabled(self, db):
        """Status returns disabled state when not enabled."""
        status = db.autodidact_status()
        assert status.enabled is False
        assert status.cycle_count == 0


class TestManualRunNow:
    def test_manual_run_now(self, db):
        """run_now() triggers an immediate cycle (gap detection only)."""
        _seed_single_source_entity(db)

        db.enable_autodidact(interval=999999, sources="none")
        db.autodidact_run_now()

        # Wait for cycle to complete
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.cycle_count >= 1
        assert status.last_cycle is not None
        assert status.last_cycle.tasks_generated > 0

        db.disable_autodidact()


class TestCycleWithMockSearch:
    def test_cycle_with_mock_search(self, db):
        """Register search_fn returning synthetic text, verify claims ingested."""
        _seed_single_source_entity(db)

        def mock_search(query: str) -> str:
            return "test_entity interacts with novel_target via phosphorylation"

        db.enable_autodidact(interval=999999, search_fn=mock_search)
        db.autodidact_run_now()
        time.sleep(1.5)

        status = db.autodidact_status()
        assert status.cycle_count >= 1
        # Should have ingested at least the fallback raw claim
        assert status.total_claims_ingested >= 1

        db.disable_autodidact()


class TestNegativeResult:
    def test_negative_result_on_empty_search(self, db):
        """search_fn returns empty string -> negative result recorded."""
        _seed_single_source_entity(db)

        def empty_search(query: str) -> str:
            return ""

        db.enable_autodidact(interval=999999, search_fn=empty_search)
        db.autodidact_run_now()
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.cycle_count >= 1
        assert status.last_cycle is not None
        assert status.last_cycle.negative_results > 0

        db.disable_autodidact()


class TestEventTriggers:
    def test_event_trigger_retraction(self, db):
        """Retract a source -> daemon wakes up."""
        _seed_single_source_entity(db)
        # Add a second source so retraction is meaningful
        db.ingest_batch([
            ClaimInput(
                subject=("test_entity", "protein"),
                predicate=("found_in", "biological"),
                object=("cell_membrane", "location"),
                provenance={"source_id": "retractable_src", "source_type": "paper"},
                confidence=0.9,
            )
        ])

        triggered = {"count": 0}

        def on_cycle(**kwargs):
            triggered["count"] += 1

        db.on("autodidact_cycle_completed", on_cycle)
        db.enable_autodidact(interval=999999, sources="none")

        # Wait — no initial cycle (sleep-first loop)
        time.sleep(1.5)
        initial = triggered["count"]

        # Retract a source to trigger event
        db.retract("retractable_src", reason="test")
        time.sleep(1.5)

        assert triggered["count"] > initial
        db.disable_autodidact()

    def test_event_trigger_inquiry(self, db):
        """Ingest an inquiry -> daemon wakes up."""
        _seed_single_source_entity(db)

        triggered = {"count": 0}

        def on_cycle(**kwargs):
            triggered["count"] += 1

        db.on("autodidact_cycle_completed", on_cycle)
        db.enable_autodidact(interval=999999, sources="none")

        # Wait — no initial cycle (sleep-first loop)
        time.sleep(1.5)
        initial = triggered["count"]

        # Create an inquiry to trigger event
        db.ingest_inquiry(
            "What does test_entity do?",
            subject=("test_entity", "protein"),
            object=("function", "concept"),
        )
        time.sleep(1.5)

        assert triggered["count"] > initial
        db.disable_autodidact()

    def test_event_debounce(self, db):
        """Two rapid events within cooldown only trigger one cycle."""
        _seed_single_source_entity(db)
        db.ingest_batch([
            ClaimInput(
                subject=("test_entity", "protein"),
                predicate=("found_in", "biological"),
                object=("cell_membrane", "location"),
                provenance={"source_id": "src_a", "source_type": "paper"},
                confidence=0.9,
            ),
            ClaimInput(
                subject=("test_entity", "protein"),
                predicate=("located_in", "biological"),
                object=("nucleus", "location"),
                provenance={"source_id": "src_b", "source_type": "paper"},
                confidence=0.9,
            ),
        ])

        triggered = {"count": 0}

        def on_cycle(**kwargs):
            triggered["count"] += 1

        db.on("autodidact_cycle_completed", on_cycle)
        # Set a long cooldown so the second event is definitely debounced
        db.enable_autodidact(interval=999999, sources="none", trigger_cooldown=300)

        time.sleep(1.0)
        initial = triggered["count"]

        # Fire two retractions rapidly
        db.retract("src_a", reason="test")
        time.sleep(0.1)
        db.retract("src_b", reason="test")
        time.sleep(2.0)

        # Should have triggered exactly one cycle, not two
        assert triggered["count"] == initial + 1
        db.disable_autodidact()


class TestJournal:
    def test_journal_persistence(self, db):
        """Run cycle, verify autodidact_history() returns reports."""
        _seed_single_source_entity(db)

        db.enable_autodidact(interval=999999, sources="none")
        db.autodidact_run_now()
        time.sleep(1.0)

        history = db.autodidact_history(limit=5)
        assert len(history) >= 1
        assert isinstance(history[0], CycleReport)
        assert history[0].cycle_number >= 1

        db.disable_autodidact()

    def test_history_survives_restart(self, db):
        """Disable then re-enable autodidact, verify history recovered from journal claims."""
        _seed_single_source_entity(db)

        db.enable_autodidact(interval=999999, sources="none")
        db.autodidact_run_now()
        time.sleep(1.0)

        status_before = db.autodidact_status()
        assert status_before.cycle_count >= 1

        # Stop and restart
        db.disable_autodidact()
        db.enable_autodidact(interval=999999, sources="none")

        # History should be recovered from journal claims
        history = db.autodidact_history(limit=5)
        assert len(history) >= 1
        status_after = db.autodidact_status()
        assert status_after.cycle_count >= status_before.cycle_count

        db.disable_autodidact()


class TestBudget:
    def test_budget_exhaustion(self, db):
        """Set max_llm_calls_per_day=1, verify second cycle skips research."""
        _seed_single_source_entity(db)

        call_count = {"n": 0}

        def counting_search(query: str) -> str:
            call_count["n"] += 1
            return "some evidence"

        db.enable_autodidact(
            interval=999999,
            max_llm_calls_per_day=1,
            search_fn=counting_search,
        )

        # First cycle
        db.autodidact_run_now()
        time.sleep(1.0)

        # Second cycle — should be budget-exhausted
        db.autodidact_run_now()
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.budget_exhausted is True

        db.disable_autodidact()

    def test_cost_budget_exhaustion(self, db):
        """Set max_cost_per_day very low, verify budget exhausted by cost cap."""
        _seed_single_source_entity(db)

        def cheap_search(query: str) -> str:
            return "some evidence"

        db.enable_autodidact(
            interval=999999,
            max_llm_calls_per_day=9999,
            max_cost_per_day=0.001,  # tiny cost cap
            sources="none",  # don't auto-register, we register manually with cost
        )
        # Register source with non-zero cost so cost tracking works even without LLM
        db._autodidact.register_source("paid", cheap_search, cost_per_call=0.002)

        db.autodidact_run_now()
        time.sleep(1.0)

        # Second cycle should be budget-exhausted by cost
        db.autodidact_run_now()
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.budget_exhausted is True
        assert status.estimated_cost_today > 0

        db.disable_autodidact()


class TestGracefulDegradation:
    def test_without_enterprise(self, db):
        """No Researcher, custom search_fn -> works fine."""
        _seed_single_source_entity(db)

        def my_search(query: str) -> str:
            return "evidence text here"

        db.enable_autodidact(interval=999999, search_fn=my_search)
        db.autodidact_run_now()
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.cycle_count >= 1
        assert status.total_claims_ingested >= 1

        db.disable_autodidact()

    def test_without_anything(self, db):
        """No Researcher, no search_fn -> gap detection still runs."""
        _seed_single_source_entity(db)

        db.enable_autodidact(interval=999999, sources="none")
        db.autodidact_run_now()
        time.sleep(1.0)

        status = db.autodidact_status()
        assert status.cycle_count >= 1
        # Tasks should be generated even without sources
        assert status.last_cycle is not None
        assert status.last_cycle.tasks_generated > 0
        # But no claims ingested (no sources)
        assert status.total_claims_ingested == 0

        db.disable_autodidact()


class TestAutoSources:
    def test_auto_registers_free_sources(self, db):
        """sources='auto' registers PubMed and Semantic Scholar."""
        db.enable_autodidact(interval=999999, sources="auto")
        assert db._autodidact is not None
        source_names = [s.name for s in db._autodidact._sources]
        assert "pubmed" in source_names
        assert "semantic_scholar" in source_names
        db.disable_autodidact()

    def test_search_fn_overrides_auto(self, db):
        """Explicit search_fn prevents auto-registration."""

        def my_fn(q: str) -> str:
            return ""

        db.enable_autodidact(interval=999999, search_fn=my_fn)
        assert db._autodidact is not None
        source_names = [s.name for s in db._autodidact._sources]
        assert "default" in source_names
        assert "pubmed" not in source_names
        db.disable_autodidact()


class TestCostEstimate:
    def test_cost_estimate_returns_breakdown(self, db):
        """cost_estimate() returns cost breakdown without running cycles."""
        _seed_single_source_entity(db)

        def my_search(q: str) -> str:
            return ""

        db.enable_autodidact(interval=999999, search_fn=my_search)
        estimate = db.autodidact_cost_estimate(cycles=24)

        assert estimate["tasks_available_now"] > 0
        assert estimate["primary_source"] == "default"
        assert estimate["cost_per_task"] > 0
        assert estimate["cost_per_cycle"] > 0
        assert estimate["cost_per_day_capped"] <= estimate["max_cost_per_day"]
        assert estimate["cost_per_month_capped"] <= estimate["max_cost_per_day"] * 30
        assert "max_llm_calls_per_day" in estimate

        db.disable_autodidact()

    def test_cost_estimate_not_enabled(self, db):
        """cost_estimate() returns empty dict when autodidact is not enabled."""
        assert db.autodidact_cost_estimate() == {}

    def test_cost_estimate_free_sources(self, db):
        """Free sources (no search_fn) report zero search cost."""
        _seed_single_source_entity(db)

        db.enable_autodidact(interval=999999, sources="none")
        estimate = db.autodidact_cost_estimate(cycles=1)

        assert estimate["primary_source"] == "none"
        assert estimate["cost_per_task"] >= 0
        # No paid sources, cost comes only from extraction
        db.disable_autodidact()


class TestCloseStopsDaemon:
    def test_close_stops_autodidact(self):
        """db.close() stops the autodidact daemon cleanly."""
        db = AttestDB(":memory:", embedding_dim=None)
        db.enable_autodidact(interval=999999, sources="none")
        assert db.autodidact_status().enabled is True
        db.close()
        # After close, the daemon reference is cleared
        assert db._autodidact is None
