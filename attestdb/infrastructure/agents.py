"""Multi-agent collaborative research: sessions, task queue, and federation.

Agents are sources with source_type="agent" and source_id="agent:<agent_id>".
Agent metadata is stored as claims about the agent entity.
Research sessions are batches of claims with consistent provenance.
Tasks are ephemeral views derived from blindspots and gaps.
Federation syncs claims between instances via NDJSON over HTTP.
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import (
    Claim,
    ClaimInput,
    ClaimStatus,
    EntityRef,
    Payload,
    PredicateRef,
    Provenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — Agent Research Sessions
# ---------------------------------------------------------------------------


@dataclass
class AgentInfo:
    agent_id: str
    total_claims: int = 0
    active: int = 0
    retracted: int = 0
    corroboration_rate: float = 0.0
    retraction_rate: float = 0.0


@dataclass
class ResearchResult:
    """Result of submitting a research batch."""

    agent_id: str
    claims_ingested: int
    claims_duplicate: int
    gaps_recorded: int
    narrative_claim_id: str | None = None


def register_agent(
    db: "AttestDB",
    agent_id: str,
    capabilities: list[str] | None = None,
    model: str | None = None,
) -> str:
    """Register an agent by ingesting meta-claims about it.

    Returns the normalized agent entity ID.
    """
    entity_id = f"agent:{agent_id}"
    source_id = f"agent:{agent_id}"

    claims = []
    if capabilities:
        for cap in capabilities:
            claims.append(
                ClaimInput(
                    subject=(entity_id, "agent"),
                    predicate=("has_capability", "agent_meta"),
                    object=(cap, "capability"),
                    provenance={"source_id": source_id, "source_type": "agent"},
                    confidence=1.0,
                )
            )
    if model:
        claims.append(
            ClaimInput(
                subject=(entity_id, "agent"),
                predicate=("has_model", "agent_meta"),
                object=(model, "model"),
                provenance={"source_id": source_id, "source_type": "agent"},
                confidence=1.0,
            )
        )

    if claims:
        db.ingest_batch(claims)

    return normalize_entity_id(entity_id)


def submit_research(
    db: "AttestDB",
    agent_id: str,
    claims: list[ClaimInput],
    narrative: str | None = None,
    topic: str | None = None,
    gaps_discovered: list[str] | None = None,
) -> ResearchResult:
    """Submit a research batch from an agent.

    All claims get source_type="agent" and source_id="agent:<agent_id>" applied.
    Optionally records a narrative finding and discovered gaps.
    """
    source_id = f"agent:{agent_id}"
    entity_id = f"agent:{agent_id}"

    # Override provenance on all claims
    normalized = []
    for c in claims:
        normalized.append(
            ClaimInput(
                subject=c.subject,
                predicate=c.predicate,
                object=c.object,
                provenance={"source_id": source_id, "source_type": "agent"},
                confidence=c.confidence,
                payload=c.payload if hasattr(c, "payload") else None,
            )
        )

    result = db.ingest_batch(normalized)

    narrative_claim_id = None
    gaps_recorded = 0

    # Record narrative as a finding claim
    if narrative and topic:
        topic_id = normalize_entity_id(topic)
        narrative_claims = [
            ClaimInput(
                subject=(entity_id, "agent"),
                predicate=("has_finding", "research"),
                object=(topic_id, "topic"),
                provenance={"source_id": source_id, "source_type": "agent"},
                confidence=0.8,
                payload={
                    "schema_ref": "research_narrative",
                    "data": {"narrative": narrative},
                },
            )
        ]
        nr = db.ingest_batch(narrative_claims)
        if nr.ingested > 0:
            # Get the claim_id of the narrative claim
            agent_claims = db.claims_by_source_id(source_id)
            for c in reversed(agent_claims):
                if c.predicate.id == normalize_entity_id("has_finding"):
                    narrative_claim_id = c.claim_id
                    break

    # Record discovered gaps
    if gaps_discovered:
        gap_claims = []
        for gap_desc in gaps_discovered:
            gap_claims.append(
                ClaimInput(
                    subject=(topic or "unknown", "topic"),
                    predicate=("has_gap", "research"),
                    object=(gap_desc, "gap"),
                    provenance={"source_id": source_id, "source_type": "agent"},
                    confidence=0.6,
                )
            )
        gr = db.ingest_batch(gap_claims)
        gaps_recorded = gr.ingested

    return ResearchResult(
        agent_id=agent_id,
        claims_ingested=result.ingested,
        claims_duplicate=result.duplicates,
        gaps_recorded=gaps_recorded,
        narrative_claim_id=narrative_claim_id,
    )


@dataclass
class NegativeResult:
    """A recorded negative result — something an agent looked for and didn't find."""

    agent_id: str
    subject: str
    hypothesis: str
    search_strategy: str | None = None
    claim_id: str | None = None


def submit_negative_result(
    db: "AttestDB",
    agent_id: str,
    subject: tuple[str, str],
    hypothesis: tuple[str, str],
    confidence: float = 0.7,
    search_strategy: str | None = None,
    hours_spent: float | None = None,
) -> NegativeResult:
    """Record that an agent investigated a hypothesis and found no evidence.

    Confidence here means "how thoroughly they looked" — higher confidence
    means a more exhaustive search, making the negative result stronger.
    """
    source_id = f"agent:{agent_id}"
    payload_data: dict = {}
    if search_strategy:
        payload_data["search_strategy"] = search_strategy
    if hours_spent is not None:
        payload_data["hours_spent"] = hours_spent

    claims = [
        ClaimInput(
            subject=subject,
            predicate=("no_evidence_for", "research"),
            object=hypothesis,
            provenance={"source_id": source_id, "source_type": "agent"},
            confidence=confidence,
            payload={"schema_ref": "negative_result", "data": payload_data}
            if payload_data
            else None,
        )
    ]
    result = db.ingest_batch(claims)

    claim_id = None
    if result.ingested > 0:
        agent_claims = db.claims_by_source_id(source_id)
        for c in reversed(agent_claims):
            if c.predicate.id == normalize_entity_id("no_evidence_for"):
                claim_id = c.claim_id
                break

    return NegativeResult(
        agent_id=agent_id,
        subject=subject[0],
        hypothesis=hypothesis[0],
        search_strategy=search_strategy,
        claim_id=claim_id,
    )


def get_negative_results(
    db: "AttestDB",
    entity_id: str | None = None,
) -> list[dict]:
    """Get negative results, optionally filtered by entity.

    Returns dicts with: subject, hypothesis, agent_id, confidence, search_strategy,
    corroboration_count (how many agents independently found no evidence).
    """
    try:
        neg_claims = db.claims_for_predicate("no_evidence_for")
    except Exception:
        return []

    # Group by content_id to count corroboration
    by_content: dict[str, list[Claim]] = {}
    for c in neg_claims:
        if c.status != ClaimStatus.ACTIVE:
            continue
        if entity_id:
            norm_id = normalize_entity_id(entity_id)
            if c.subject.id != norm_id and c.object.id != norm_id:
                continue
        by_content.setdefault(c.content_id, []).append(c)

    results = []
    for content_id, claims in by_content.items():
        representative = claims[0]
        search_strategies = []
        for c in claims:
            if c.payload and c.payload.data:
                strat = c.payload.data.get("search_strategy")
                if strat:
                    search_strategies.append(strat)
        results.append({
            "subject": representative.subject.id,
            "subject_type": representative.subject.entity_type,
            "hypothesis": representative.object.id,
            "hypothesis_type": representative.object.entity_type,
            "agents": [c.provenance.source_id for c in claims],
            "confidence": max(c.confidence for c in claims),
            "corroboration_count": len(claims),
            "search_strategies": search_strategies,
            "content_id": content_id,
        })

    results.sort(key=lambda r: (-r["corroboration_count"], -r["confidence"]))
    return results


def agent_leaderboard(db: "AttestDB", min_claims: int = 1) -> list[AgentInfo]:
    """Return agent reliability rankings sorted by corroboration rate."""
    reliability = db.source_reliability()

    agents = []
    for source_id, metrics in reliability.items():
        if not source_id.startswith("agent:"):
            continue
        if metrics["total_claims"] < min_claims:
            continue
        agents.append(
            AgentInfo(
                agent_id=source_id.removeprefix("agent:"),
                total_claims=metrics["total_claims"],
                active=metrics["active"],
                retracted=metrics["retracted"],
                corroboration_rate=metrics["corroboration_rate"],
                retraction_rate=metrics["retraction_rate"],
            )
        )

    agents.sort(key=lambda a: (-a.corroboration_rate, -a.total_claims))
    return agents


# ---------------------------------------------------------------------------
# Phase 2 — Gap-Driven Task Queue
# ---------------------------------------------------------------------------


@dataclass
class ResearchTask:
    """An open research task derived from a knowledge gap."""

    entity_id: str
    entity_type: str
    gap_type: str  # "single_source", "missing_predicate", "low_confidence", "has_gap"
    description: str
    priority: int  # 0=highest
    claimed_by: str | None = None


# Stale investigation threshold (24 hours in nanoseconds)
_STALE_INVESTIGATION_NS = 24 * 3600 * 10**9


def generate_tasks(
    db: "AttestDB",
    max_tasks: int = 50,
    gap_types: list[str] | None = None,
    entity_types: list[str] | None = None,
) -> list[ResearchTask]:
    """Generate research tasks from current knowledge gaps.

    Tasks are ephemeral — derived from current DB state, not stored.
    Filters out entities already being investigated (non-stale).
    """
    tasks: dict[str, ResearchTask] = {}  # entity_id -> best task
    now_ns = time.time_ns()

    # Find currently investigated entities (exclude stale investigations)
    investigating = set()
    try:
        inv_claims = db.claims_for_predicate("investigating")
        for c in inv_claims:
            if (
                c.status == ClaimStatus.ACTIVE
                and (now_ns - c.timestamp) < _STALE_INVESTIGATION_NS
            ):
                investigating.add(c.object.id)
    except Exception:
        pass  # claims_for_predicate may not exist for all predicates

    # Source 1: Blindspots (single-source entities)
    entity_types_set = set(entity_types) if entity_types else None
    try:
        if entity_types_set:
            # Optimized path for large DBs: scan only entities of the requested types
            # instead of calling blindspots() which scans ALL entities.
            candidates = []
            for et in entity_types_set:
                candidates.extend(db.list_entities(entity_type=et, min_claims=1))
        else:
            # Full blindspots scan (fine for small/medium DBs)
            blindspots = db.blindspots(min_claims=1)
            candidates = []
            for entity in blindspots.single_source_entities:
                eid = (
                    entity
                    if isinstance(entity, str)
                    else getattr(entity, "id", str(entity))
                )
                e = db.get_entity(eid)
                if e:
                    candidates.append(e)

        for entity in candidates:
            eid = entity.id if hasattr(entity, "id") else str(entity)
            etype = entity.entity_type if hasattr(entity, "entity_type") else "unknown"
            if eid in investigating or eid in tasks:
                continue
            # For the optimized path, check single-source inline
            if entity_types_set:
                src_counts = db._store.entity_source_counts(eid)
                if len(src_counts) != 1:
                    continue  # not single-source
            # Skip entities whose only sources are autodidact — don't chase own tail
            claims = db.claims_for(eid)
            if claims and all(
                (c.provenance.source_id or "").startswith("autodidact:")
                or (c.provenance.source_id or "").startswith("agent:autodidact")
                or c.provenance.source_type == "autodidact"
                for c in claims
            ):
                continue
            tasks[eid] = ResearchTask(
                entity_id=eid,
                entity_type=etype,
                gap_type="single_source",
                description=f"Entity '{eid}' has claims from only one source. Needs independent corroboration.",
                priority=1,
            )
    except Exception as exc:
        logger.debug("Blindspot scan skipped: %s", exc)

    # Source 2: Explicit gaps (from agent-reported has_gap claims)
    try:
        gap_claims = db.claims_for_predicate("has_gap")
        for c in gap_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            eid = c.subject.id
            if eid in investigating:
                continue
            if eid not in tasks or tasks[eid].priority > 0:
                tasks[eid] = ResearchTask(
                    entity_id=eid,
                    entity_type=c.subject.entity_type,
                    gap_type="has_gap",
                    description=f"Reported gap: {c.object.id}",
                    priority=0,
                )
    except Exception as exc:
        logger.debug("Gap claim scan skipped: %s", exc)

    # Source 3: Low confidence entities
    try:
        entities = db.list_entities()
        for entity in entities:
            eid = entity.id
            if eid in investigating or eid in tasks:
                continue
            if entity_types_set and entity.entity_type not in entity_types_set:
                continue
            claims = db.claims_for(eid)
            if not claims:
                continue
            avg_conf = sum(c.confidence for c in claims) / len(claims)
            if avg_conf < 0.3 and len(claims) >= 2:
                tasks[eid] = ResearchTask(
                    entity_id=eid,
                    entity_type=entity.entity_type,
                    gap_type="low_confidence",
                    description=f"Entity '{eid}' has low average confidence ({avg_conf:.2f}). Needs stronger evidence.",
                    priority=2,
                )
    except Exception as exc:
        logger.debug("Low confidence scan skipped: %s", exc)

    # Deprioritize entities with multiple negative results
    neg_counts: dict[str, int] = {}
    try:
        neg_claims = db.claims_for_predicate("no_evidence_for")
        for c in neg_claims:
            if c.status == ClaimStatus.ACTIVE:
                eid = c.subject.id
                neg_counts[eid] = neg_counts.get(eid, 0) + 1
    except Exception:
        pass

    for eid, task in tasks.items():
        n = neg_counts.get(eid, 0)
        if n >= 3:
            task.priority += 3  # heavily deprioritize — 3+ agents found nothing
            task.description += f" (⚠ {n} negative results recorded)"
        elif n >= 1:
            task.priority += 1  # slightly deprioritize
            task.description += f" ({n} negative result{'s' if n > 1 else ''})"

    # Apply filters
    result = list(tasks.values())
    if gap_types:
        result = [t for t in result if t.gap_type in gap_types]
    if entity_types:
        result = [t for t in result if t.entity_type in entity_types]

    # Sort by priority, limit
    result.sort(key=lambda t: (t.priority, t.entity_id))
    return result[:max_tasks]


def claim_task(db: "AttestDB", agent_id: str, entity_id: str) -> str:
    """Agent claims a research task. Returns the claim_id of the investigation claim."""
    source_id = f"agent:{agent_id}"
    claims = [
        ClaimInput(
            subject=(f"agent:{agent_id}", "agent"),
            predicate=("investigating", "research"),
            object=(entity_id, "entity"),
            provenance={"source_id": source_id, "source_type": "agent"},
            confidence=1.0,
        )
    ]
    result = db.ingest_batch(claims)
    if result.ingested > 0:
        # Find the claim we just created
        agent_claims = db.claims_by_source_id(source_id)
        for c in reversed(agent_claims):
            if c.predicate.id == normalize_entity_id(
                "investigating"
            ) and c.object.id == normalize_entity_id(entity_id):
                return c.claim_id
    return ""


def get_task_context(db: "AttestDB", entity_id: str) -> dict:
    """Get research context for an entity: negative results, strategy claims, and findings.

    This is what a curator/head-of-research agent would synthesize,
    and what any agent should review before starting work on a task.
    """
    context: dict = {
        "entity_id": entity_id,
        "negative_results": [],
        "strategies": [],
        "findings": [],
        "prior_investigators": [],
    }

    norm_id = normalize_entity_id(entity_id)

    # Negative results about this entity
    context["negative_results"] = get_negative_results(db, entity_id=entity_id)

    # Strategy claims (from curator agents)
    try:
        strategy_claims = db.claims_for_predicate("has_strategy")
        for c in strategy_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            if c.subject.id == norm_id or c.object.id == norm_id:
                context["strategies"].append({
                    "strategy": c.object.id if c.subject.id == norm_id else c.subject.id,
                    "confidence": c.confidence,
                    "agent": c.provenance.source_id,
                    "payload": c.payload.data if c.payload else None,
                })
    except Exception:
        pass

    # Prior findings
    try:
        finding_claims = db.claims_for_predicate("has_finding")
        for c in finding_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            if c.object.id == norm_id or c.subject.id == norm_id:
                narrative_text = None
                if c.payload and c.payload.data:
                    narrative_text = c.payload.data.get("narrative")
                context["findings"].append({
                    "agent": c.provenance.source_id,
                    "narrative": narrative_text,
                    "confidence": c.confidence,
                })
    except Exception:
        pass

    # Who has investigated this before
    try:
        inv_claims = db.claims_for_predicate("investigating")
        for c in inv_claims:
            if c.object.id == norm_id:
                context["prior_investigators"].append({
                    "agent": c.provenance.source_id,
                    "timestamp": c.timestamp,
                    "status": c.status.value,
                })
    except Exception:
        pass

    return context


def complete_task(
    db: "AttestDB",
    agent_id: str,
    entity_id: str,
    claims: list[ClaimInput],
    narrative: str | None = None,
) -> ResearchResult:
    """Complete a research task by submitting findings."""
    return submit_research(
        db,
        agent_id,
        claims,
        narrative=narrative,
        topic=entity_id,
    )


# ---------------------------------------------------------------------------
# Phase 2.5 — AttestAgent SDK
# ---------------------------------------------------------------------------


class AttestAgent:
    """High-level SDK for building research agents on AttestDB.

    Wraps the full agent lifecycle: register, get tasks, research,
    report findings + negative results, and record session outcomes.
    """

    def __init__(
        self,
        db: "AttestDB",
        agent_id: str,
        model: str | None = None,
        capabilities: list[str] | None = None,
    ):
        self.db = db
        self.agent_id = agent_id
        self.model = model
        self.capabilities = capabilities
        self._entity_id = register_agent(
            db, agent_id, capabilities=capabilities, model=model
        )

    def next_task(
        self,
        gap_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> tuple[ResearchTask, dict] | None:
        """Claim the highest-priority task from the queue.

        Returns (task, context) where context is the research context for the
        task's entity, or None if no tasks are available.
        """
        tasks = generate_tasks(
            self.db, max_tasks=1, gap_types=gap_types, entity_types=entity_types
        )
        if not tasks:
            return None
        task = tasks[0]
        claim_task(self.db, self.agent_id, task.entity_id)
        task.claimed_by = self.agent_id
        context = get_task_context(self.db, task.entity_id)
        return task, context

    def report(
        self,
        claims: list[ClaimInput],
        narrative: str | None = None,
        topic: str | None = None,
        gaps_discovered: list[str] | None = None,
    ) -> ResearchResult:
        """Submit research findings."""
        return submit_research(
            self.db,
            self.agent_id,
            claims,
            narrative=narrative,
            topic=topic,
            gaps_discovered=gaps_discovered,
        )

    def report_negative(
        self,
        subject: tuple[str, str],
        hypothesis: tuple[str, str],
        search_strategy: str | None = None,
        confidence: float = 0.7,
    ) -> NegativeResult:
        """Record a negative result -- something looked for and not found."""
        return submit_negative_result(
            self.db,
            self.agent_id,
            subject=subject,
            hypothesis=hypothesis,
            confidence=confidence,
            search_strategy=search_strategy,
        )

    def publish_strategy(
        self,
        entity: str,
        strategy: str,
        rationale: str | None = None,
        confidence: float = 0.8,
    ) -> ResearchResult:
        """Publish a strategy claim (for curator agents).

        Records a has_strategy claim linking the entity to the strategy,
        with an optional rationale in the payload.
        """
        source_id = f"agent:{self.agent_id}"
        payload = None
        if rationale:
            payload = {
                "schema_ref": "strategy",
                "data": {"rationale": rationale},
            }
        claims = [
            ClaimInput(
                subject=(entity, "entity"),
                predicate=("has_strategy", "research"),
                object=(strategy, "strategy"),
                provenance={"source_id": source_id, "source_type": "agent"},
                confidence=confidence,
                payload=payload,
            )
        ]
        return submit_research(self.db, self.agent_id, claims)

    def done(
        self,
        outcome: str = "success",
        summary: str = "",
        next_steps: str = "",
    ) -> ResearchResult:
        """Record session end with outcome metadata."""
        source_id = f"agent:{self.agent_id}"
        payload_data: dict = {"outcome": outcome}
        if summary:
            payload_data["summary"] = summary
        if next_steps:
            payload_data["next_steps"] = next_steps

        claims = [
            ClaimInput(
                subject=(f"agent:{self.agent_id}", "agent"),
                predicate=("has_session_outcome", "agent_meta"),
                object=(outcome, "outcome"),
                provenance={"source_id": source_id, "source_type": "agent"},
                confidence=1.0,
                payload={
                    "schema_ref": "session_outcome",
                    "data": payload_data,
                },
            )
        ]
        return submit_research(self.db, self.agent_id, claims)

    @property
    def stats(self) -> AgentInfo | None:
        """Get this agent's leaderboard stats."""
        leaderboard = agent_leaderboard(self.db, min_claims=0)
        for info in leaderboard:
            if info.agent_id == self.agent_id:
                return info
        return None


# ---------------------------------------------------------------------------
# Phase 3 — Federation
# ---------------------------------------------------------------------------


@dataclass
class SyncStats:
    """Stats from a federation sync operation."""

    exported: int = 0
    imported: int = 0
    skipped_duplicate: int = 0
    errors: int = 0


def _claim_to_record(claim: Claim) -> dict:
    """Convert a Claim to an NDJSON record (same format as migration.py)."""
    record = {
        "type": "claim",
        "claim_id": claim.claim_id,
        "content_id": claim.content_id,
        "subject_id": claim.subject.id,
        "subject_type": claim.subject.entity_type,
        "subject_display_name": claim.subject.display_name,
        "subject_external_ids": claim.subject.external_ids,
        "object_id": claim.object.id,
        "object_type": claim.object.entity_type,
        "object_display_name": claim.object.display_name,
        "object_external_ids": claim.object.external_ids,
        "predicate_id": claim.predicate.id,
        "predicate_type": claim.predicate.predicate_type,
        "confidence": claim.confidence,
        "source_type": claim.provenance.source_type,
        "source_id": claim.provenance.source_id,
        "method": claim.provenance.method,
        "chain": claim.provenance.chain,
        "model_version": claim.provenance.model_version,
        "organization": claim.provenance.organization,
        "timestamp": claim.timestamp,
        "status": claim.status.value,
    }
    if claim.payload:
        record["payload_schema"] = claim.payload.schema_ref
        record["payload_data"] = claim.payload.data
    return record


def _record_to_claim(record: dict) -> Claim:
    """Convert an NDJSON record back to a Claim (same format as migration.py)."""
    payload = None
    if "payload_schema" in record:
        payload = Payload(
            schema_ref=record["payload_schema"],
            data=record.get("payload_data", {}),
        )
    return Claim(
        claim_id=record["claim_id"],
        content_id=record["content_id"],
        subject=EntityRef(
            id=record["subject_id"],
            entity_type=record["subject_type"],
            display_name=record.get("subject_display_name", ""),
            external_ids=record.get("subject_external_ids", {}),
        ),
        predicate=PredicateRef(
            id=record["predicate_id"],
            predicate_type=record["predicate_type"],
        ),
        object=EntityRef(
            id=record["object_id"],
            entity_type=record["object_type"],
            display_name=record.get("object_display_name", ""),
            external_ids=record.get("object_external_ids", {}),
        ),
        confidence=record["confidence"],
        provenance=Provenance(
            source_type=record["source_type"],
            source_id=record["source_id"],
            method=record.get("method"),
            chain=record.get("chain", []),
            model_version=record.get("model_version"),
            organization=record.get("organization"),
        ),
        payload=payload,
        timestamp=record["timestamp"],
        status=ClaimStatus(record.get("status", "active")),
    )


def export_claims_since(db: "AttestDB", since_ns: int, stream: IO[str]) -> int:
    """Export claims newer than since_ns (nanosecond timestamp) as NDJSON.

    Returns the count of claims exported.
    """
    count = 0
    all_claims = db._all_claims()
    for claim in all_claims:
        if claim.timestamp >= since_ns:
            record = _claim_to_record(claim)
            stream.write(
                json.dumps(record, separators=(",", ":"), default=str) + "\n"
            )
            count += 1
    return count


def import_claims_from_stream(db: "AttestDB", stream: IO[str]) -> SyncStats:
    """Import claims from an NDJSON stream, skipping duplicates.

    Entities are upserted before their claims are inserted.
    Returns sync stats.
    """
    stats = SyncStats()

    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            stats.errors += 1
            continue

        if record.get("type") != "claim":
            continue

        # Check if claim already exists
        claim_id = record.get("claim_id", "")
        if db._store.claim_exists(claim_id):
            stats.skipped_duplicate += 1
            continue

        try:
            # Upsert entities
            db._store.upsert_entity(
                entity_id=record["subject_id"],
                entity_type=record["subject_type"],
                display_name=record.get("subject_display_name", ""),
                external_ids=record.get("subject_external_ids") or None,
                timestamp=record.get("timestamp", 0),
            )
            db._store.upsert_entity(
                entity_id=record["object_id"],
                entity_type=record["object_type"],
                display_name=record.get("object_display_name", ""),
                external_ids=record.get("object_external_ids") or None,
                timestamp=record.get("timestamp", 0),
            )

            # Insert claim
            claim = _record_to_claim(record)
            db._store.insert_claim(claim)
            stats.imported += 1
        except Exception as exc:
            logger.warning("Failed to import claim %s: %s", claim_id, exc)
            stats.errors += 1

    return stats


def sync_with_remote(
    db: "AttestDB",
    remote_url: str,
    api_key: str,
    since_ns: int = 0,
) -> dict:
    """Bidirectional sync with a remote AttestDB instance.

    1. Pull: GET remote/api/v1/federation/export?since=<since_ns>
    2. Import pulled claims
    3. Push: POST local claims since since_ns to remote/api/v1/federation/import

    Returns {"pull": SyncStats, "push": {"exported": int, "status_code": int}}.
    """
    import urllib.request

    headers = {"Authorization": f"Bearer {api_key}"}

    # Pull from remote
    pull_stats = SyncStats()
    try:
        pull_url = f"{remote_url.rstrip('/')}/api/v1/federation/export?since={since_ns}"
        req = urllib.request.Request(pull_url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
            pull_stats = import_claims_from_stream(db, io.StringIO(text))
    except Exception as exc:
        logger.error("Pull from %s failed: %s", remote_url, exc)
        pull_stats.errors += 1

    # Push to remote
    push_exported = 0
    push_status = 0
    try:
        buf = io.StringIO()
        push_exported = export_claims_since(db, since_ns, buf)
        if push_exported > 0:
            push_url = f"{remote_url.rstrip('/')}/api/v1/federation/import"
            data = buf.getvalue().encode("utf-8")
            req = urllib.request.Request(
                push_url,
                data=data,
                headers={**headers, "Content-Type": "application/x-ndjson"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                push_status = resp.status
    except Exception as exc:
        logger.error("Push to %s failed: %s", remote_url, exc)

    return {
        "pull": {
            "imported": pull_stats.imported,
            "skipped_duplicate": pull_stats.skipped_duplicate,
            "errors": pull_stats.errors,
        },
        "push": {
            "exported": push_exported,
            "status_code": push_status,
        },
    }
