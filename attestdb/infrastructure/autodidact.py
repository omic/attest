"""Autodidact: autonomous self-learning daemon for AttestDB.

Runs in a background thread, continuously detecting knowledge gaps,
researching them via registered evidence sources or the enterprise
Researcher, and ingesting findings. Ties together the existing gap
detection (agents.py), researcher, text extraction, and curator.

Usage::

    db = AttestDB(":memory:", embedding_dim=None)
    db.enable_autodidact(interval=1800, search_fn=my_search)
    db.autodidact_status()
    db.disable_autodidact()

Cost model (per task researched):
  - 1 search call  (Perplexity ~$0.001, PubMed/S2 free)
  - 1 ingest_text  (extraction LLM call ~$0.001-0.01 depending on provider)
  - N curator calls (~$0.001 each, N = claims extracted)
  Estimated: ~$0.005-0.02 per task, tracked via estimated_cost_today.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.types import AutodidactConfig, AutodidactStatus, ClaimInput, CycleReport

logger = logging.getLogger(__name__)

# Default cost for free sources (PubMed/Semantic Scholar)
_COST_SEARCH_FREE = 0.0


@dataclass
class EvidenceSource:
    """A registered evidence source for the autodidact daemon."""

    name: str
    search_fn: Callable[[str], str]
    cost_per_call: float = 0.0
    priority: int = 0  # lower = tried first


class AutodidactDaemon:
    """Background daemon that continuously detects and closes knowledge gaps.

    Lifecycle: create -> start() -> [cycles run in background] -> stop().
    The daemon is interruptible via events (retraction, inquiry) and run_now().

    Budget enforcement:
      - max_llm_calls_per_day: hard cap on API calls (search + extraction)
      - max_cost_per_day: estimated dollar cap (default $1.00)
      Both must be within limits for a cycle to proceed.
    """

    def __init__(self, db: "AttestDB", config: AutodidactConfig) -> None:
        self._db = db
        self._config = config

        # Threading
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._lock = threading.Lock()  # guards mutable state below

        # State (all access must hold self._lock)
        self._running = False
        self._cycle_count = 0
        self._total_claims_ingested = 0
        self._llm_calls_today = 0
        self._estimated_cost_today = 0.0
        self._budget_date: str = ""  # YYYY-MM-DD for daily reset
        self._budget_exhausted = False
        self._last_cycle: CycleReport | None = None
        self._next_cycle_at: float = 0.0
        self._history: list[CycleReport] = []
        self._pending_trigger: str | None = None
        self._last_event_trigger_time: float = 0.0

        # Evidence sources
        self._sources: list[EvidenceSource] = []

        # Enterprise researcher (lazy)
        self._researcher = None

    def register_source(
        self,
        name: str,
        search_fn: Callable[[str], str],
        cost_per_call: float = 0.0,
        priority: int = 0,
    ) -> None:
        """Register an evidence source."""
        # Replace existing source with same name
        self._sources = [s for s in self._sources if s.name != name]
        self._sources.append(
            EvidenceSource(
                name=name,
                search_fn=search_fn,
                cost_per_call=cost_per_call,
                priority=priority,
            )
        )
        self._sources.sort(key=lambda s: s.priority)

    def start(self) -> None:
        """Start the daemon background thread."""
        with self._lock:
            if self._running:
                return
            self._stop_event.clear()
            self._wake_event.clear()
            self._running = True

        # Recover history from persisted journal claims
        self._recover_history()

        # Register event hooks
        if "retraction" in self._config.enabled_triggers:
            self._db.on("source_retracted", self._on_retraction)
        if "inquiry" in self._config.enabled_triggers:
            self._db.on("inquiry_created", self._on_inquiry)

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="attest-autodidact",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the daemon and wait for thread to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        # Unregister event hooks BEFORE signaling stop to prevent
        # race where an event fires mid-shutdown
        self._db.off("source_retracted", self._on_retraction)
        self._db.off("inquiry_created", self._on_inquiry)
        self._stop_event.set()
        self._wake_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._thread = None

    def run_now(self) -> None:
        """Trigger an immediate cycle."""
        with self._lock:
            self._pending_trigger = "manual"
        self._wake_event.set()

    def cost_estimate(self, cycles: int = 24) -> dict:
        """Estimate cost for N cycles without running them.

        Args:
            cycles: Number of cycles to estimate (default 24 = 1 day at 1hr interval).

        Returns:
            Dict with per_cycle, daily, monthly estimates and breakdown.
        """
        from attestdb.infrastructure.agents import generate_tasks

        tasks = generate_tasks(
            self._db,
            max_tasks=self._config.max_questions_per_cycle,
            gap_types=self._config.gap_types,
            entity_types=self._config.entity_types,
        )
        tasks_per_cycle = len(tasks)

        # Determine which source would be used (first by priority)
        search_cost = _COST_SEARCH_FREE
        source_name = "none"
        for src in self._sources:
            source_name = src.name
            search_cost = src.cost_per_call
            break

        # Per-task cost: search + extraction + curator
        avg_claims_per_task = 8  # typical extraction yields 5-15 claims per task
        cost_per_task = search_cost + self._config.cost_ingest_text
        if self._config.use_curator:
            cost_per_task += self._config.cost_curator_per_claim * avg_claims_per_task

        cost_per_cycle = cost_per_task * tasks_per_cycle
        daily = cost_per_cycle * cycles
        # Apply caps
        daily_capped = min(daily, self._config.max_cost_per_day)

        return {
            "tasks_available_now": tasks_per_cycle,
            "tasks_per_cycle": min(tasks_per_cycle, self._config.max_questions_per_cycle),
            "primary_source": source_name,
            "cost_per_task": round(cost_per_task, 4),
            "cost_per_cycle": round(cost_per_cycle, 4),
            "cost_per_day_uncapped": round(daily, 4),
            "cost_per_day_capped": round(daily_capped, 4),
            "cost_per_month_capped": round(daily_capped * 30, 2),
            "max_cost_per_day": self._config.max_cost_per_day,
            "max_llm_calls_per_day": self._config.max_llm_calls_per_day,
        }

    @property
    def status(self) -> AutodidactStatus:
        """Return current daemon status."""
        with self._lock:
            return AutodidactStatus(
                enabled=self._running,
                running=self._running,
                cycle_count=self._cycle_count,
                total_claims_ingested=self._total_claims_ingested,
                total_llm_calls_today=self._llm_calls_today,
                estimated_cost_today=round(self._estimated_cost_today, 4),
                max_cost_per_day=self._config.max_cost_per_day,
                budget_exhausted=self._budget_exhausted,
                last_cycle=self._last_cycle,
                next_cycle_at=self._next_cycle_at,
            )

    @property
    def history(self) -> list[CycleReport]:
        """Return cycle history."""
        with self._lock:
            return list(self._history)

    # --- Event handlers ---

    def _on_retraction(self, **kwargs) -> None:
        """Handle source_retracted event."""
        with self._lock:
            now = time.time()
            if now - self._last_event_trigger_time < self._config.trigger_cooldown:
                return  # debounce
            self._last_event_trigger_time = now
            self._pending_trigger = "retraction"
        self._wake_event.set()

    def _on_inquiry(self, **kwargs) -> None:
        """Handle inquiry_created event."""
        with self._lock:
            now = time.time()
            if now - self._last_event_trigger_time < self._config.trigger_cooldown:
                return
            self._last_event_trigger_time = now
            self._pending_trigger = "inquiry"
        self._wake_event.set()

    # --- Budget ---

    def _budget_ok(self) -> bool:
        """Check if both call count and cost budgets allow more work."""
        with self._lock:
            if self._llm_calls_today >= self._config.max_llm_calls_per_day:
                return False
            if self._estimated_cost_today >= self._config.max_cost_per_day:
                return False
            return True

    def _record_cost(self, amount: float, calls: int = 1) -> None:
        """Record API cost and call count."""
        with self._lock:
            self._llm_calls_today += calls
            self._estimated_cost_today += amount

    def _reset_daily_budget(self) -> None:
        """Reset daily budgets when the date changes."""
        today = time.strftime("%Y-%m-%d")
        with self._lock:
            if today != self._budget_date:
                self._budget_date = today
                self._llm_calls_today = 0
                self._estimated_cost_today = 0.0
                self._budget_exhausted = False

    # --- Main loop ---

    def _loop(self) -> None:
        """Background loop: sleep -> cycle -> repeat.

        Waits for the configured interval (or a wake event) before the
        first cycle, so ``run_now()`` and event triggers work predictably.
        """
        while not self._stop_event.is_set():
            # Sleep first — interruptible by run_now() or event triggers
            jitter_factor = 1.0 + random.uniform(
                -self._config.jitter, self._config.jitter
            )
            wait_time = self._config.interval_seconds * jitter_factor
            self._next_cycle_at = time.time() + wait_time
            self._wake_event.wait(timeout=wait_time)

            if self._stop_event.is_set():
                break

            with self._lock:
                trigger = self._pending_trigger or "timer"
                self._pending_trigger = None
            self._wake_event.clear()

            try:
                report = self._run_cycle(trigger)
                with self._lock:
                    self._last_cycle = report
                    self._cycle_count += 1
                    self._total_claims_ingested += report.claims_ingested
                    self._history.append(report)
                    if len(self._history) > self._config.max_history:
                        self._history = self._history[-self._config.max_history:]
            except Exception as exc:
                logger.warning("Autodidact cycle failed: %s", exc)

    # --- Cycle logic ---

    def _run_cycle(self, trigger: str) -> CycleReport:
        """Execute one autodidact cycle."""
        from attestdb.infrastructure.agents import (
            claim_task,
            generate_tasks,
            get_task_context,
            submit_negative_result,
        )

        started = time.time()
        report = CycleReport(
            cycle_number=self._cycle_count + 1,
            started_at=started,
            finished_at=0.0,
            trigger=trigger,
        )

        # 1. Budget check
        self._reset_daily_budget()
        if not self._budget_ok():
            with self._lock:
                self._budget_exhausted = True
            self._db._fire("autodidact_budget_exhausted")
            # Still measure blindspots for reporting
            try:
                bs = self._db.blindspots(min_claims=1)
                report.blindspot_before = len(bs.single_source_entities)
                report.blindspot_after = report.blindspot_before
            except Exception:
                pass
            report.finished_at = time.time()
            return report
        with self._lock:
            self._budget_exhausted = False

        # 2. Detect gaps
        tasks = generate_tasks(
            self._db,
            max_tasks=self._config.max_questions_per_cycle,
            gap_types=self._config.gap_types,
            entity_types=self._config.entity_types,
        )
        report.tasks_generated = len(tasks)

        if not tasks:
            report.finished_at = time.time()
            self._journal(report)
            return report

        # 3. Measure blindspots before
        try:
            bs_before = self._db.blindspots(min_claims=1)
            report.blindspot_before = len(bs_before.single_source_entities)
        except Exception:
            pass

        # 4. Research each task
        has_sources = bool(self._sources) or self._get_researcher() is not None

        for task in tasks:
            if self._stop_event.is_set():
                break
            if not self._budget_ok():
                with self._lock:
                    self._budget_exhausted = True
                self._db._fire("autodidact_budget_exhausted")
                break

            # Check negative result limit
            try:
                context = get_task_context(self._db, task.entity_id)
                neg_count = len(context.get("negative_results", []))
                if neg_count >= self._config.negative_result_limit:
                    continue
            except Exception:
                pass

            # Claim the task
            try:
                claim_task(self._db, "autodidact", task.entity_id)
            except Exception:
                pass

            if not has_sources:
                # No sources available — gap detection only mode
                continue

            # Try evidence sources
            found = False
            for source in self._sources:
                if self._stop_event.is_set():
                    break
                try:
                    result_text = source.search_fn(task.description)
                    search_cost = source.cost_per_call
                    self._record_cost(search_cost)
                    report.llm_calls += 1
                    report.estimated_cost += search_cost

                    if result_text and result_text.strip() and not self._is_refusal(result_text):
                        # Ingest the evidence (costs additional LLM calls)
                        ingested, rejected, ingest_cost = self._ingest_evidence(
                            result_text, task.entity_id, source.name
                        )
                        report.claims_ingested += ingested
                        report.claims_rejected += rejected
                        report.estimated_cost += ingest_cost
                        report.tasks_researched += 1
                        found = True
                        break  # stop after first successful source
                except Exception as exc:
                    report.errors.append(f"{source.name}: {exc}")

            # Try enterprise Researcher as fallback
            if not found:
                researcher = self._get_researcher()
                if researcher is not None:
                    try:
                        res = researcher.research(task.description)
                        self._record_cost(self._config.cost_search_paid)
                        report.llm_calls += 1
                        report.estimated_cost += self._config.cost_search_paid

                        if res and getattr(res, "text", None):
                            ingested, rejected, ingest_cost = self._ingest_evidence(
                                res.text, task.entity_id, "researcher"
                            )
                            report.claims_ingested += ingested
                            report.claims_rejected += rejected
                            report.estimated_cost += ingest_cost
                            report.tasks_researched += 1
                            found = True
                    except Exception as exc:
                        report.errors.append(f"researcher: {exc}")

            if not found and has_sources:
                # Record negative result
                try:
                    submit_negative_result(
                        self._db,
                        "autodidact",
                        subject=(task.entity_id, task.entity_type),
                        hypothesis=(task.description, "research_question"),
                        search_strategy="autodidact_cycle",
                    )
                    report.negative_results += 1
                except Exception:
                    pass

        # 5. Measure blindspots after
        try:
            bs_after = self._db.blindspots(min_claims=1)
            report.blindspot_after = len(bs_after.single_source_entities)
        except Exception:
            pass

        report.finished_at = time.time()
        report.estimated_cost = round(report.estimated_cost, 4)

        # 6. Journal
        self._journal(report)

        return report

    def _ingest_evidence(
        self, text: str, entity_id: str, source_name: str
    ) -> tuple[int, int, float]:
        """Ingest evidence text into the DB.

        Returns (ingested, rejected, estimated_cost_usd).
        """
        try:
            result = self._db.ingest_text(
                text,
                source_id=f"autodidact:{source_name}:{entity_id}",
                use_curator=self._config.use_curator,
            )
            # Record cost only after successful extraction
            extraction_cost = self._config.cost_ingest_text
            curator_cost = (
                self._config.cost_curator_per_claim * result.raw_count
                if self._config.use_curator
                else 0.0
            )
            cost = extraction_cost + curator_cost
            self._record_cost(cost, calls=1 + (result.raw_count if self._config.use_curator else 0))

            if result.n_valid > 0:
                return result.n_valid, result.raw_count - result.n_valid, cost
            # LLM returned nothing — fall through to raw claim
        except Exception:
            cost = 0.0  # No cost if extraction didn't complete

        # Fallback: store as a raw evidence claim (no LLM cost incurred).
        # Skip refusal/non-informative responses.
        if self._is_refusal(text):
            return 0, 0, 0.0

        claim = ClaimInput(
            subject=(entity_id, "entity"),
            predicate=("has_evidence", "research"),
            object=(text[:200], "evidence"),
            provenance={
                "source_id": f"autodidact:{source_name}",
                "source_type": "autodidact",
            },
            confidence=0.5,
            payload={
                "schema_ref": "autodidact_evidence",
                "data": {"full_text": text, "source": source_name},
            },
        )
        result = self._db.ingest_batch([claim])
        return result.ingested, 0, 0.0

    def _journal(self, report: CycleReport) -> None:
        """Store cycle report as a claim and fire event."""
        try:
            report_data = {
                "cycle_number": report.cycle_number,
                "trigger": report.trigger,
                "tasks_generated": report.tasks_generated,
                "tasks_researched": report.tasks_researched,
                "claims_ingested": report.claims_ingested,
                "claims_rejected": report.claims_rejected,
                "negative_results": report.negative_results,
                "llm_calls": report.llm_calls,
                "estimated_cost": report.estimated_cost,
                "blindspot_before": report.blindspot_before,
                "blindspot_after": report.blindspot_after,
                "duration_seconds": round(report.finished_at - report.started_at, 2),
                "errors": report.errors[:10],  # cap
            }
            claim = ClaimInput(
                subject=("autodidact", "system"),
                predicate=("completed_cycle", "autodidact"),
                object=(f"cycle_{report.cycle_number}", "cycle"),
                provenance={
                    "source_id": "autodidact",
                    "source_type": "autodidact",
                },
                confidence=1.0,
                payload={
                    "schema_ref": "autodidact_cycle_report",
                    "data": report_data,
                },
            )
            self._db.ingest_batch([claim])
        except Exception as exc:
            logger.debug("Autodidact journal write failed: %s", exc)

        self._db._fire(
            "autodidact_cycle_completed",
            cycle_number=report.cycle_number,
            claims_ingested=report.claims_ingested,
            tasks_generated=report.tasks_generated,
            estimated_cost=report.estimated_cost,
        )

    def _recover_history(self) -> None:
        """Recover cycle history from persisted journal claims."""
        try:
            claims = self._db.claims_for("autodidact")
            for claim in claims:
                if claim.predicate.id != "completed_cycle":
                    continue
                data = {}
                payload = claim.payload
                if payload and hasattr(payload, "data"):
                    data = payload.data or {}
                elif payload and isinstance(payload, dict):
                    data = payload.get("data", {})
                if not data or not isinstance(data, dict):
                    continue
                report = CycleReport(
                    cycle_number=data.get("cycle_number", 0),
                    started_at=0.0,
                    finished_at=0.0,
                    tasks_generated=data.get("tasks_generated", 0),
                    tasks_researched=data.get("tasks_researched", 0),
                    claims_ingested=data.get("claims_ingested", 0),
                    claims_rejected=data.get("claims_rejected", 0),
                    negative_results=data.get("negative_results", 0),
                    llm_calls=data.get("llm_calls", 0),
                    estimated_cost=data.get("estimated_cost", 0.0),
                    blindspot_before=data.get("blindspot_before", 0),
                    blindspot_after=data.get("blindspot_after", 0),
                    trigger=data.get("trigger", "timer"),
                    errors=data.get("errors", []),
                )
                self._history.append(report)
            # Sort by cycle number and cap
            self._history.sort(key=lambda r: r.cycle_number)
            if len(self._history) > self._config.max_history:
                self._history = self._history[-self._config.max_history:]
            if self._history:
                self._cycle_count = self._history[-1].cycle_number
                self._total_claims_ingested = sum(r.claims_ingested for r in self._history)
                self._last_cycle = self._history[-1]
                logger.info("Autodidact recovered %d cycle reports from journal", len(self._history))
        except Exception as exc:
            logger.debug("Autodidact history recovery failed: %s", exc)

    @staticmethod
    def _is_refusal(text: str) -> bool:
        """Detect non-informative search responses (LLM refusals, empty results)."""
        lower = text[:500].lower()
        return any(
            phrase in lower
            for phrase in [
                "i cannot provide",
                "i don't have enough",
                "no relevant results",
                "no results found",
                "i'm unable to",
                "i could not find",
                "unable to find relevant",
            ]
        )

    def _get_researcher(self):
        """Lazily get the enterprise Researcher, or None if unavailable."""
        if self._researcher is not None:
            return self._researcher
        try:
            from attestdb_enterprise.researcher import Researcher

            self._researcher = Researcher(self._db)
            return self._researcher
        except ImportError:
            logger.debug("Enterprise Researcher not available (attestdb-enterprise not installed)")
            return None
