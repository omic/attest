"""Analytics engine — read-only analytical methods extracted from AttestDB."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.core.hashing import compute_content_id
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import (
    Analogy,
    BeliefChange,
    BriefSection,
    Citation,
    Claim,
    ClaimInput,
    ClaimStatus,
    ConfidenceChange,
    ConfidenceGap,
    ConfidenceShift,
    ConnectionLoss,
    ContradictionAnalysis,
    ContradictionReport,
    ContradictionSide,
    Discovery,
    EntitySummary,
    EvidenceChain,
    EvolutionReport,
    Explanation,
    ExplanationStep,
    Forecast,
    ForecastConnection,
    HypothesisVerdict,
    Investigation,
    KnowledgeBrief,
    KnowledgeDiff,
    KnowledgeHealth,
    MergeConflict,
    MergeReport,
    PathResult,
    QualityReport,
    ReasoningChain,
    ReasoningHop,
    RelationshipPattern,
    SimulationReport,
    SourceOverlap,
    claim_from_dict,
    entity_summary_from_dict,
)

logger = logging.getLogger(__name__)

FRESHNESS_HALF_LIFE_DAYS = 30


class AnalyticsEngine:
    """Read-only analytical methods over the knowledge graph.

    All methods access the database through self.db (an AttestDB instance).
    """

    def __init__(self, db: "AttestDB") -> None:
        self.db = db
        self._discovered_causal_preds: set[str] | None = None

    def _resolve_causal_predicates(
        self,
        extra: set[str] | None = None,
        directional_only: bool = False,
    ) -> set[str]:
        """Build the effective set of causal predicate IDs.

        Combines:
        1. CAUSAL_PREDICATES (hardcoded, always included for backward compat)
        2. Predicate IDs discovered from the store where predicate_type is causal
        3. User-provided extra causal predicates
        """
        from attestdb.core.vocabulary import CAUSAL_PREDICATES, CAUSAL_PREDICATE_TYPES

        base = set(CAUSAL_PREDICATES)
        if directional_only:
            base.discard("regulates")

        # Discover additional causal predicates from store metadata (once)
        if self._discovered_causal_preds is None:
            discovered: set[str] = set()
            try:
                if hasattr(self.db._store, "predicate_id_to_type_map"):
                    type_map = self.db._store.predicate_id_to_type_map()
                    for pred_id, pred_type in type_map.items():
                        if pred_type in CAUSAL_PREDICATE_TYPES:
                            discovered.add(pred_id)
                else:
                    # Fallback: sample claims from a few entities
                    try:
                        entities = self.db._store.list_entities(
                            None, 0, 0, 20
                        )
                        for e in entities:
                            eid = e["id"] if isinstance(e, dict) else e.id
                            claims = self.db._store.claims_for(eid, None, None, 0.0)
                            for d in claims[:50]:
                                c = claim_from_dict(d)
                                if (c.predicate.predicate_type in CAUSAL_PREDICATE_TYPES
                                        and c.predicate.id not in base):
                                    discovered.add(c.predicate.id)
                    except Exception:
                        pass
            except Exception:
                pass
            self._discovered_causal_preds = discovered

        result = base | self._discovered_causal_preds
        if extra:
            result |= extra
        return result

    def corroboration_report(self, min_sources: int = 2) -> dict:
        """Report on corroboration status across the knowledge graph.

        Returns corroborated claims (grouped by content_id) and single-source
        claims that need independent confirmation.
        """
        cache_key = f"corroboration_report:{min_sources}"
        cached = self.db._cache.get(cache_key)
        if cached is not None and (time.time() - self.db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        from attestdb.core.confidence import count_independent_sources, corroboration_boost

        content_id_claims: dict[str, list] = {}

        for claim in self.db.iter_claims():
            cid = claim.content_id
            if cid not in content_id_claims:
                content_id_claims[cid] = []
            content_id_claims[cid].append(claim)

        corroborated = []
        single_source = []

        for cid, claims in content_id_claims.items():
            n_indep = count_independent_sources(claims)
            if n_indep >= min_sources:
                boost = corroboration_boost(n_indep)
                corroborated.append({
                    "content_id": cid,
                    "subject": claims[0].subject.id,
                    "predicate": claims[0].predicate.id,
                    "object": claims[0].object.id,
                    "n_claims": len(claims),
                    "n_independent_sources": n_indep,
                    "confidence_boost": round(boost, 3),
                    "source_types": list({c.provenance.source_type for c in claims}),
                })
            elif len(claims) == 1:
                single_source.append({
                    "content_id": cid,
                    "subject": claims[0].subject.id,
                    "predicate": claims[0].predicate.id,
                    "object": claims[0].object.id,
                    "source_type": claims[0].provenance.source_type,
                    "confidence": claims[0].confidence,
                })

        corroborated.sort(key=lambda x: x["n_independent_sources"], reverse=True)
        single_source.sort(key=lambda x: x["confidence"])

        total = len(content_id_claims)
        result = {
            "total_content_ids": total,
            "corroborated_count": len(corroborated),
            "single_source_count": len(single_source),
            "corroboration_ratio": len(corroborated) / total if total else 0,
            "corroborated": corroborated,
            "needs_corroboration": single_source[:50],
        }
        self.db._cache[cache_key] = result
        self.db._cache_ts[cache_key] = time.time()
        return result

    def diagnose_corroboration(self, content_id: str) -> dict:
        """Show how corroboration is counted for a specific content_id.

        Returns a breakdown of external ID clustering vs provenance overlap
        grouping, making corroboration inflation visible and debuggable.
        """
        from attestdb.core.confidence import (
            _extract_external_id,
            _count_by_provenance_overlap,
            count_independent_sources,
        )

        raw = self.db._store.claims_by_content_id(content_id)
        claims = [claim_from_dict(d) for d in raw]

        # Reproduce the two-pass logic with full detail
        external_id_groups: dict[str, list] = {}
        unclustered: list = []
        claim_details: list[dict] = []

        for claim in claims:
            ext_id = _extract_external_id(claim)
            detail = {
                "claim_id": claim.claim_id,
                "source_id": claim.provenance.source_id,
                "source_type": claim.provenance.source_type,
                "external_id": ext_id,
                "provenance_chain": claim.provenance.chain,
            }
            claim_details.append(detail)
            if ext_id:
                external_id_groups.setdefault(ext_id, []).append(claim)
            else:
                unclustered.append(claim)

        provenance_overlap_groups = _count_by_provenance_overlap(unclustered) if unclustered else 0

        return {
            "content_id": content_id,
            "total_claims": len(claims),
            "external_id_groups": {k: len(v) for k, v in external_id_groups.items()},
            "unclustered_claims": len(unclustered),
            "provenance_overlap_groups": provenance_overlap_groups,
            "final_independent_count": count_independent_sources(claims),
            "claims": claim_details,
        }

    # --- Quality report ---

    def quality_report(
        self,
        stale_threshold: int = 0,
        expected_patterns: dict[str, set[str]] | None = None,
    ) -> QualityReport:
        """Generate a quality report over the entire knowledge graph.

        Uses Rust-native analytics indexes where available:
        - stats() for entity type counts and totals
        - predicate_counts() for predicate distribution
        - entity_source_counts() for single-source detection
        - Single-pass iter_claims for source_type distribution and staleness
        """
        cache_key = f"quality_report:{stale_threshold}:{id(expected_patterns)}"
        cached = self.db._cache.get(cache_key)
        if cached is not None and (time.time() - self.db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        report = QualityReport()
        rust_stats = self.db._store.stats()
        report.total_entities = rust_stats.get("entity_count", 0)
        report.total_claims = rust_stats.get("total_claims", 0)

        # Entity type counts from Rust (O(1) — pre-indexed)
        report.entity_type_counts = {
            k: v for k, v in rust_stats.get("entity_types", {}).items()
        }

        # Predicate distribution from Rust (O(1) — counter table)
        pred_counts = self.db._store.predicate_counts()
        report.predicate_distribution = {k: int(v) for k, v in pred_counts.items()}

        # Source type distribution + stale entity detection via single claim pass.
        # Single-source entity detection via Rust entity_source_counts().
        source_type_dist: dict[str, int] = {}
        # Use source_types from stats for distribution (count per source_type)
        source_types_from_stats = rust_stats.get("source_types", {})
        if source_types_from_stats:
            source_type_dist = {k: v for k, v in source_types_from_stats.items()}

        # Single-source detection: use entity_source_counts per entity
        # which returns (source_id, count, avg_conf) without materializing claims
        single_source_count = 0
        stale_count = 0
        entities = self.db.list_entities()

        for entity in entities:
            src_counts = self.db._store.entity_source_counts(entity.id)
            if len(src_counts) == 1:
                single_source_count += 1

        report.single_source_entity_count = single_source_count
        report.source_type_distribution = source_type_dist

        # Stale detection requires claim timestamps — only scan if requested
        if stale_threshold > 0:
            entity_latest: dict[str, int] = {}
            for claim in self.db.iter_claims():
                for eid in (claim.subject.id, claim.object.id):
                    if claim.timestamp > entity_latest.get(eid, 0):
                        entity_latest[eid] = claim.timestamp
            stale_count = sum(
                1 for ts in entity_latest.values() if ts < stale_threshold
            )
        report.stale_entity_count = stale_count

        if report.total_entities > 0:
            report.avg_claims_per_entity = report.total_claims / report.total_entities

        # Gap and confidence alert counts via insight engine
        if expected_patterns:
            gaps = self.db.find_gaps(expected_patterns, min_claims=1)
            report.gap_count = len(gaps)

        alerts = self.db.find_confidence_alerts(min_claims=2)
        report.confidence_alert_count = len(alerts)

        self.db._cache[cache_key] = report
        self.db._cache_ts[cache_key] = time.time()
        return report

    # --- Confidence calibration ---

    def calibration_report(
        self,
        labeled_claims: list[tuple[dict, bool]],
        n_bins: int = 5,
    ) -> dict:
        """Compare predicted confidence scores against ground truth.

        See :meth:`EvalHarness.calibration_report` for full documentation.
        """
        from attestdb.intelligence.eval_harness import EvalHarness
        harness = EvalHarness(self)
        return harness.calibration_report(labeled_claims, n_bins=n_bins)

    # --- Knowledge health ---

    def knowledge_health(self) -> KnowledgeHealth:
        """Compute quantified health metrics for the knowledge graph.

        Uses two efficient passes:
        1. Rust entity_source_counts() for multi-source detection (no claim materialization)
        2. Single iter_claims pass for confidence/timestamp/corroboration stats
        """
        cached = self.db._cache.get("knowledge_health")
        if cached is not None and (time.time() - self.db._cache_ts.get("knowledge_health", 0)) < 120:
            return cached
        h = KnowledgeHealth()
        rust_stats = self.db._store.stats()
        h.total_entities = rust_stats.get("entity_count", 0)
        if h.total_entities == 0:
            return h

        # Multi-source detection via Rust index (no claim materialization)
        multi_source_count = 0
        entities = self.db.list_entities()
        for entity in entities:
            src_counts = self.db._store.entity_source_counts(entity.id)
            if len(src_counts) > 1:
                multi_source_count += 1

        # Single pass over claims for confidence/timestamp/content_id stats
        content_id_counts: dict[str, int] = {}
        source_types: set[str] = set()
        all_confidences: list[float] = []
        all_timestamps: list[int] = []
        chain_lengths: list[int] = []

        # Session metadata predicates that don't represent real knowledge
        _session_predicates = {
            "awaiting_outcome", "had_outcome", "used_tool",
            "session_metadata", "produced_by", "confidence_updated",
        }

        for claim in self.db.iter_claims():
            # Skip session bookkeeping claims from health calculations
            if (
                claim.provenance.source_type in ("session_end", "outcome_report")
                or claim.predicate.id in _session_predicates
            ):
                continue

            all_confidences.append(claim.confidence)
            all_timestamps.append(claim.timestamp)
            source_types.add(claim.provenance.source_type)
            chain_lengths.append(len(claim.provenance.chain))
            content_id_counts[claim.content_id] = (
                content_id_counts.get(claim.content_id, 0) + 1
            )

        h.total_claims = len(all_confidences)
        if h.total_claims == 0:
            return h

        # Basic metrics
        h.avg_confidence = sum(all_confidences) / len(all_confidences)
        h.multi_source_ratio = multi_source_count / h.total_entities
        h.source_diversity = len(source_types)
        h.knowledge_density = h.total_claims / h.total_entities
        h.avg_provenance_depth = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0.0

        # Corroboration ratio: fraction of content_ids with >1 claim
        if content_id_counts:
            corroborated = sum(1 for c in content_id_counts.values() if c > 1)
            h.corroboration_ratio = corroborated / len(content_id_counts)

        # Freshness: exponential decay based on age of claims (half-life = 30 days)
        # Timestamps are in nanoseconds (see ingestion.py)
        now_ns = int(time.time() * 1_000_000_000)
        if all_timestamps and max(all_timestamps) > 0:
            half_life_ns = FRESHNESS_HALF_LIFE_DAYS * 86400 * 1_000_000_000
            freshness_sum = 0.0
            for ts in all_timestamps:
                age_ns = max(0, now_ns - ts)
                freshness_sum += 2.0 ** (-age_ns / half_life_ns)
            h.freshness_score = freshness_sum / len(all_timestamps)

        # Confidence trend: compare avg confidence of recent half vs older half
        if len(all_confidences) >= 2:
            paired = sorted(zip(all_timestamps, all_confidences))
            mid = len(paired) // 2
            old_avg = sum(c for _, c in paired[:mid]) / mid
            new_avg = sum(c for _, c in paired[mid:]) / (len(paired) - mid)
            h.confidence_trend = new_avg - old_avg

        # Composite health score (0-100)
        h.health_score = min(100.0, max(0.0, (
            h.multi_source_ratio * 30.0
            + h.corroboration_ratio * 25.0
            + h.freshness_score * 20.0
            + min(1.0, h.source_diversity / 5.0) * 15.0
            + max(0.0, min(1.0, 0.5 + h.confidence_trend)) * 10.0
        )))

        self.db._cache["knowledge_health"] = h
        self.db._cache_ts["knowledge_health"] = time.time()
        return h

    # --- New API methods (Stage 7) ---

    def impact(self, source_id: str) -> "ImpactReport":
        """Analyze the impact of a source: how many claims depend on it."""
        from attestdb.core.types import ImpactReport

        claims = self.db.claims_by_source_id(source_id)
        claim_ids = [c.claim_id for c in claims]

        # Find affected entities
        affected = set()
        for c in claims:
            affected.add(c.subject.id)
            affected.add(c.object.id)

        # Count downstream dependents
        downstream_count = 0
        for cid in claim_ids:
            tree = self.db.trace_downstream(cid)
            downstream_count += len(tree.dependents)

        return ImpactReport(
            source_id=source_id,
            direct_claims=len(claims),
            downstream_claims=downstream_count,
            affected_entities=sorted(affected),
            claim_ids=claim_ids,
        )

    def blindspots(self, min_claims: int = 5) -> "BlindspotMap":
        """Find knowledge blindspots: single-source entities and gaps.

        Uses Rust entity_source_counts() for single-source detection (no claim
        materialization), and a single iter_claims pass for unresolved warnings.
        """
        cache_key = f"blindspots:{min_claims}"
        cached = self.db._cache.get(cache_key)
        if cached is not None and (time.time() - self.db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        from attestdb.core.types import BlindspotMap
        from attestdb.core.vocabulary import knowledge_sort_key

        # Single-source detection via bulk Rust scan (one call, no per-entity roundtrip).
        from attestdb.infrastructure.agents import is_autodidact_source

        raw_single = self.db._store.find_single_source_entities(min_claims)
        single_source = []
        for entity_id in raw_single:
            # Filter out entities whose sole source is autodidact
            src_counts = self.db._store.entity_source_counts(entity_id)
            sole_source_id = src_counts[0][0] if src_counts else ""
            if is_autodidact_source(sole_source_id):
                continue
            single_source.append(entity_id)

        # Single pass over claims for subject→predicate map (unresolved warnings)
        subject_predicates: dict[str, set[str]] = {}
        subject_claims: dict[str, list] = {}

        for c in self.db.iter_claims():
            subj = c.subject.id
            subject_predicates.setdefault(subj, set()).add(c.predicate.id)
            subject_claims.setdefault(subj, []).append(c)

        # Compute knowledge gaps from predicate constraints
        knowledge_gaps: list[dict] = []
        try:
            constraints = self.db._store.get_predicate_constraints()
            if constraints:
                from attestdb.intelligence.insight_engine import build_expected_patterns
                patterns = build_expected_patterns(constraints)
                gap_results = self.db.find_gaps(patterns, min_claims=min_claims)
                for gap in gap_results:
                    knowledge_gaps.append({
                        "entity": gap.entity_id,
                        "entity_type": gap.entity_type,
                        "missing": gap.missing_predicate_types,
                    })
        except Exception:
            logger.debug("Could not compute knowledge gaps", exc_info=True)

        # Get confidence alerts as low-confidence areas
        low_conf: list[dict] = []
        try:
            alerts = self.db.find_confidence_alerts(min_claims=2)
            for alert in alerts:
                low_conf.append({
                    "entity": alert.entity_id,
                    "entity_type": alert.entity_type,
                    "alert_type": alert.alert_type.value,
                    "explanation": alert.explanation,
                    "source_id": alert.source_id,
                })
        except Exception:
            logger.debug("Could not compute confidence alerts", exc_info=True)

        # Find entities with bugs/issues but no corresponding fix.
        # Uses subject→predicate map built in the single pass above.
        unresolved: list[dict] = []
        problem_predicates = {"had_bug", "has_issue", "has_vulnerability"}
        resolution_predicates = {"has_fix", "resolved", "has_status"}
        try:
            for subj, preds in subject_predicates.items():
                problems = preds & problem_predicates
                resolutions = preds & resolution_predicates
                if problems and not resolutions:
                    problem_list = [
                        c for c in subject_claims[subj]
                        if c.predicate.id in problems
                    ]
                    problem_list.sort(
                        key=lambda c: knowledge_sort_key(c.predicate.id, c.confidence),
                    )
                    top = problem_list[0]
                    unresolved.append({
                        "entity": subj,
                        "entity_type": top.subject.entity_type,
                        "predicate": top.predicate.id,
                        "description": top.object.id,
                        "confidence": top.confidence,
                    })
        except Exception:
            logger.debug("Could not compute unresolved warnings", exc_info=True)

        result = BlindspotMap(
            single_source_entities=single_source,
            knowledge_gaps=knowledge_gaps,
            low_confidence_areas=low_conf,
            unresolved_warnings=unresolved,
        )
        self.db._cache[cache_key] = result
        self.db._cache_ts[cache_key] = time.time()
        return result

    def consensus(self, topic: str) -> "ConsensusReport":
        """Analyze consensus around a topic (entity)."""
        from attestdb.core.types import ConsensusReport

        claims = self.db.claims_for(topic)[:500]
        if not claims:
            return ConsensusReport(topic=topic)

        sources = {}
        content_ids = {}
        total_conf = 0.0

        for c in claims:
            sid = c.provenance.source_id
            sources[sid] = sources.get(sid, 0) + 1
            content_ids[c.content_id] = content_ids.get(c.content_id, 0) + 1
            total_conf += c.confidence

        corroborated = [cid for cid, count in content_ids.items() if count > 1]

        return ConsensusReport(
            topic=topic,
            total_claims=len(claims),
            unique_sources=len(sources),
            avg_confidence=total_conf / len(claims),
            agreement_ratio=len(corroborated) / len(content_ids) if content_ids else 0.0,
            claims_by_source=sources,
            corroborated_content_ids=corroborated,
        )

    def fragile(self, max_sources: int = 1, min_age_days: int = 0) -> list[Claim]:
        """Find fragile claims: backed by few sources, optionally filtered by age."""
        cache_key = f"fragile:{max_sources}:{min_age_days}"
        cached = self.db._cache.get(cache_key)
        if cached is not None and (time.time() - self.db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        now_ns = int(time.time() * 1_000_000_000)
        min_age_ns = min_age_days * 86400 * 1_000_000_000

        # Single-pass: group source_ids by content_id, then filter
        content_sources: dict[str, set[str]] = {}
        content_claims: dict[str, list] = {}
        for claim in self.db.iter_claims():
            if claim.status != ClaimStatus.ACTIVE:
                continue
            if min_age_ns > 0 and (now_ns - claim.timestamp) < min_age_ns:
                continue
            cid = claim.content_id
            content_sources.setdefault(cid, set()).add(claim.provenance.source_id)
            content_claims.setdefault(cid, []).append(claim)

        fragile_claims = []
        for cid, sources in content_sources.items():
            if len(sources) <= max_sources:
                fragile_claims.extend(content_claims[cid])

        self.db._cache[cache_key] = fragile_claims
        self.db._cache_ts[cache_key] = time.time()
        return fragile_claims

    def stale(self, days: int = 90, limit: int = 0) -> list[Claim]:
        """Find stale claims: not updated within the given number of days.

        Args:
            days: Age threshold.
            limit: Max results (0 = unlimited).
        """
        cache_key = f"stale:{days}:{limit}"
        cached = self.db._cache.get(cache_key)
        if cached is not None and (time.time() - self.db._cache_ts.get(cache_key, 0)) < 120:
            return cached
        cutoff_ns = int(time.time() * 1_000_000_000) - (days * 86400 * 1_000_000_000)
        stale_claims = []
        for claim in self.db.iter_claims():
            if claim.status != ClaimStatus.ACTIVE:
                continue
            if claim.timestamp < cutoff_ns:
                stale_claims.append(claim)
                if limit and len(stale_claims) >= limit:
                    break
        self.db._cache[cache_key] = stale_claims
        self.db._cache_ts[cache_key] = time.time()
        return stale_claims

    def audit(self, claim_id: str) -> "AuditTrail":
        """Build a full audit trail for a claim."""
        from attestdb.core.types import AuditTrail

        found = self.db.get_claim(claim_id)
        if found is None:
            return AuditTrail(claim_id=claim_id)

        # Get corroborating claims
        corroborating = self.db.claims_by_content_id(found.content_id)
        corr_ids = [c.claim_id for c in corroborating if c.claim_id != claim_id]

        # Get downstream dependents
        tree = self.db.trace_downstream(claim_id)
        downstream_count = len(tree.dependents)

        return AuditTrail(
            claim_id=claim_id,
            content_id=found.content_id,
            corroborating_claims=corr_ids,
            provenance_chain=found.provenance.chain,
            downstream_dependents=downstream_count,
            source_type=found.provenance.source_type,
            source_id=found.provenance.source_id,
            confidence=found.confidence,
            timestamp=found.timestamp,
        )

    def drift(self, days: int = 30) -> "DriftReport":
        """Measure knowledge drift over the given time period."""
        from attestdb.core.types import DriftReport

        now_ns = int(time.time() * 1_000_000_000)
        cutoff_ns = now_ns - (days * 86400 * 1_000_000_000)

        # Include retracted claims so we can count tombstones
        self.db._set_include_retracted(True)
        try:
            # Single-pass aggregation via iter_claims()
            n_before = 0
            n_after_new = 0
            n_total = 0
            retracted = 0
            conf_sum_before = 0.0
            conf_sum_total = 0.0
            entities_before: set[str] = set()
            entities_all: set[str] = set()
            old_source_types: set[str] = set()
            new_source_types: set[str] = set()

            after_source_types: set[str] = set()
            for c in self.db.iter_claims():
                n_total += 1
                conf_sum_total += c.confidence
                entities_all.add(c.subject.id)
                entities_all.add(c.object.id)
                if c.timestamp <= cutoff_ns:
                    n_before += 1
                    conf_sum_before += c.confidence
                    entities_before.add(c.subject.id)
                    entities_before.add(c.object.id)
                    old_source_types.add(c.provenance.source_type)
                else:
                    n_after_new += 1
                    after_source_types.add(c.provenance.source_type)
                    if c.status == ClaimStatus.TOMBSTONED:
                        retracted += 1

            new_source_types = after_source_types - old_source_types
        finally:
            self.db._set_include_retracted(False)

        conf_before = conf_sum_before / n_before if n_before else 0.0
        conf_after = conf_sum_total / n_total if n_total else 0.0

        return DriftReport(
            period_days=days,
            new_claims=n_after_new,
            new_entities=len(entities_all - entities_before),
            retracted_claims=retracted,
            confidence_delta=conf_after - conf_before,
            new_source_types=sorted(new_source_types),
            entity_count_before=len(entities_before),
            entity_count_after=len(entities_all),
            claim_count_before=n_before,
            claim_count_after=n_total,
        )

    def source_reliability(self, source_id: str | None = None) -> dict:
        """Compute per-source reliability metrics (corroboration/retraction rates).

        If source_id is given, returns metrics for that source only.
        Otherwise, returns a dict of {source_id: metrics} for all sources.

        Uses a single pass to group claims by source AND by content_id,
        avoiding N+1 claims_by_content_id lookups.
        """
        # Include retracted claims so we can compute retraction rates
        self.db._set_include_retracted(True)
        try:
            by_source: dict[str, list[Claim]] = {}
            content_id_sources: dict[str, set[str]] = {}
            if source_id is not None:
                # Fast path: only scan claims for the requested source
                for c in self.db.claims_by_source_id(source_id):
                    by_source.setdefault(c.provenance.source_id, []).append(c)
                # Check corroboration: for each unique content_id, look up all sources
                unique_cids = {c.content_id for claims in by_source.values() for c in claims}
                for cid in unique_cids:
                    for c in self.db.claims_by_content_id(cid):
                        content_id_sources.setdefault(cid, set()).add(c.provenance.source_id)
            else:
                # Full scan: group by source AND build content_id → source_ids map
                for c in self.db.iter_claims():
                    by_source.setdefault(c.provenance.source_id, []).append(c)
                    content_id_sources.setdefault(c.content_id, set()).add(c.provenance.source_id)
        finally:
            self.db._set_include_retracted(False)

        def _metrics(claims: list[Claim]) -> dict:
            total = len(claims)
            active = sum(1 for c in claims if c.status == ClaimStatus.ACTIVE)
            retracted = sum(1 for c in claims if c.status == ClaimStatus.TOMBSTONED)
            degraded = sum(1 for c in claims if c.status == ClaimStatus.PROVENANCE_DEGRADED)

            # Corroboration: check if content_id has claims from other sources
            corroborated = 0
            for c in claims:
                sources_for_cid = content_id_sources.get(c.content_id, set())
                if len(sources_for_cid) > 1:
                    corroborated += 1

            return {
                "total_claims": total,
                "active": active,
                "retracted": retracted,
                "degraded": degraded,
                "corroboration_rate": corroborated / total if total else 0.0,
                "retraction_rate": retracted / total if total else 0.0,
            }

        if source_id is not None:
            claims = by_source.get(source_id, [])
            return _metrics(claims)

        return {sid: _metrics(claims) for sid, claims in by_source.items()}

    def hypothetical(self, claim: ClaimInput) -> "HypotheticalReport":
        """What-if analysis: what would happen if this claim were ingested."""
        from attestdb.core.types import HypotheticalReport

        content_id = compute_content_id(
            normalize_entity_id(claim.subject[0]),
            claim.predicate[0],
            normalize_entity_id(claim.object[0]),
        )

        existing = self.db.claims_by_content_id(content_id)

        # Check if it fills a gap by looking at entity connectivity
        subj_exists = self.db._store.get_entity(normalize_entity_id(claim.subject[0])) is not None
        obj_exists = self.db._store.get_entity(normalize_entity_id(claim.object[0])) is not None
        fills_gap = subj_exists and obj_exists and not self.db.path_exists(
            claim.subject[0], claim.object[0], max_depth=2
        )

        related = set()
        if subj_exists:
            for c in self.db.claims_for(claim.subject[0])[:200]:
                related.add(c.object.id)
        if obj_exists:
            for c in self.db.claims_for(claim.object[0])[:200]:
                related.add(c.subject.id)

        return HypotheticalReport(
            would_corroborate=len(existing) > 0,
            existing_corroborations=len(existing),
            fills_gap=fills_gap,
            content_id=content_id,
            related_entities=sorted(related)[:20],
        )

    def what_if(self, subject: tuple, predicate: tuple, object: tuple,
                confidence: float = 0.6,
                extra_causal_predicates: set[str] | None = None):
        """One-call hypothetical reasoning — queries the knowledge graph directly.

        No sandbox creation. Directly analyzes whether a hypothetical claim
        is supported, contradicted, or plausible based on:
        - Direct contradictions (opposing predicates on same entity pair)
        - Direct corroborations (existing claims with same content_id)
        - Multi-hop causal composition (2-hop BFS, causal edges first)
        - Gap detection (does this bridge disconnected entities?)
        - Follow-up suggestions (nearby unconnected entity pairs)

        The causal-priority BFS follows causal edges first (recognized by
        predicate name or predicate_type="causes"), then weak edges only
        if causal paths are sparse.

        Works with read-only databases and scales to millions of claims.

        Usage:
            verdict = db.what_if(("BRCA1", "gene"), ("inhibits", "relation"), ("PD-L1", "protein"))
        """
        from attestdb.core.types import IndirectEvidence, SandboxVerdict
        from attestdb.core.vocabulary import (
            CAUSAL_PREDICATE_TYPES,
            CAUSAL_PREDICATES,
            OPPOSITE_PREDICATES,
            PREDICATE_EQUIVALENCE,
            compose_predicates,
            predicates_agree,
        )

        _causal_set = self._resolve_causal_predicates(extra_causal_predicates)

        subj_id = subject[0]
        obj_id = object[0]
        hyp_pred = predicate[0]
        desc = f"{subj_id} {hyp_pred} {obj_id}"
        n_subj_id = normalize_entity_id(subj_id)
        n_obj_id = normalize_entity_id(obj_id)
        opposite = OPPOSITE_PREDICATES.get(hyp_pred)

        # Single pass over subject's claims — extract direct evidence + neighbors
        subj_claims = self.db._store.claims_for(n_subj_id, None, None, 0.0)
        direct_contradictions: list[tuple[str, str]] = []
        direct_supporting: int = 0
        causal_neighbors: dict[str, tuple[str, float]] = {}
        weak_neighbors: dict[str, tuple[str, float]] = {}

        for d in subj_claims:
            c = claim_from_dict(d)
            obj = c.object.id
            pred = c.predicate.id
            conf = c.confidence

            # Direct evidence: count supporting + contradicting claims to target
            _is_causal = (pred in _causal_set
                          or c.predicate.predicate_type in CAUSAL_PREDICATE_TYPES)
            if obj == n_obj_id and _is_causal:
                if predicates_agree(pred, hyp_pred):
                    direct_supporting += 1
                else:
                    # Count as contradiction if: (a) it's the opposite predicate,
                    # or (b) it's in the opposite equivalence class
                    pred_class = PREDICATE_EQUIVALENCE.get(pred)
                    hyp_class = PREDICATE_EQUIVALENCE.get(hyp_pred)
                    if (opposite and pred == opposite) or \
                       (pred_class and hyp_class and pred_class != hyp_class):
                        direct_contradictions.append((hyp_pred, pred))

            # Collect neighbors for BFS
            neighbor = obj if c.subject.id == n_subj_id else c.subject.id
            if neighbor != n_subj_id and neighbor != n_obj_id:
                if _is_causal:
                    if neighbor not in causal_neighbors or conf > causal_neighbors[neighbor][1]:
                        causal_neighbors[neighbor] = (pred, conf)
                else:
                    if neighbor not in weak_neighbors or conf > weak_neighbors[neighbor][1]:
                        weak_neighbors[neighbor] = (pred, conf)

        # Direct corroborations — same content_id already exists
        content_id = compute_content_id(n_subj_id, hyp_pred, n_obj_id)
        corroborations = len(self.db.claims_by_content_id(content_id))

        # 2-hop BFS with causal priority
        all_indirect: list[IndirectEvidence] = []
        checked = 0
        max_intermediaries = 50

        # Phase 1: follow causal neighbors — these give meaningful composition
        for mid, (pred1, conf1) in sorted(
            causal_neighbors.items(), key=lambda x: -x[1][1]
        ):
            if checked >= max_intermediaries:
                break
            mid_claims = self.db._store.claims_for(mid, None, None, 0.0)[:200]
            checked += 1
            for d in mid_claims:
                mc = claim_from_dict(d)
                if mc.object.id == n_obj_id:
                    pred2 = mc.predicate.id
                    composed = compose_predicates(pred1, pred2)
                    if composed == hyp_pred or predicates_agree(composed, hyp_pred):
                        direction = "supporting"
                    elif (OPPOSITE_PREDICATES.get(composed) == hyp_pred
                          or OPPOSITE_PREDICATES.get(hyp_pred) == composed
                          or (composed != hyp_pred
                              and not predicates_agree(composed, hyp_pred)
                              and predicates_agree(composed, OPPOSITE_PREDICATES.get(hyp_pred, "")))):
                        direction = "contradicting"
                    else:
                        direction = "neutral"
                    all_indirect.append(IndirectEvidence(
                        path=[n_subj_id, mid, n_obj_id],
                        predicates=[pred1, pred2],
                        predicted_predicate=composed,
                        confidence=conf1 * mc.confidence,
                        direction=direction,
                    ))
            if len(all_indirect) >= 20:
                break

        # Phase 2: weak neighbors (only if causal paths sparse)
        if len(all_indirect) < 5:
            for mid, (pred1, conf1) in sorted(
                weak_neighbors.items(), key=lambda x: -x[1][1]
            ):
                if mid in causal_neighbors:
                    continue
                if checked >= max_intermediaries + 10:
                    break
                mid_claims = self.db._store.claims_for(mid, None, None, 0.0)[:100]
                checked += 1
                for d in mid_claims:
                    mc = claim_from_dict(d)
                    if mc.object.id == n_obj_id:
                        composed = compose_predicates(pred1, mc.predicate.id)
                        all_indirect.append(IndirectEvidence(
                            path=[n_subj_id, mid, n_obj_id],
                            predicates=[pred1, mc.predicate.id],
                            predicted_predicate=composed,
                            confidence=conf1 * mc.confidence,
                            direction="neutral",
                        ))
                if len(all_indirect) >= 10:
                    break

        # Gap detection — does this bridge disconnected entities?
        gaps_closed = 0
        gap_descs: list[str] = []
        subj_exists = self.db._store.get_entity(n_subj_id) is not None
        obj_exists = self.db._store.get_entity(n_obj_id) is not None
        if subj_exists and obj_exists:
            if not self.db.path_exists(subj_id, obj_id, max_depth=1):
                gaps_closed = 1
                gap_descs.append(
                    f"{subj_id} and {obj_id} exist but are not directly connected"
                )

        # Follow-up suggestions — nearby unconnected pairs
        follow_ups: list[str] = []
        subj_neighbor_set = set(causal_neighbors.keys()) | set(
            list(weak_neighbors.keys())[:50]
        )
        obj_neighbors: set[str] = set()
        for d in self.db._store.claims_for(n_obj_id, None, None, 0.0)[:200]:
            c = claim_from_dict(d)
            neighbor = c.object.id if c.subject.id == n_obj_id else c.subject.id
            obj_neighbors.add(neighbor)
        for sn in list(subj_neighbor_set)[:30]:
            if sn == n_obj_id or sn == n_subj_id:
                continue
            for on in list(obj_neighbors)[:30]:
                if on == n_subj_id or on == n_obj_id or on == sn:
                    continue
                if not self.db.path_exists(sn, on, max_depth=1):
                    follow_ups.append(f"{sn} may relate to {on}")
                    if len(follow_ups) >= 5:
                        break
            if len(follow_ups) >= 5:
                break

        # Score and verdict — use directional_confidence for evidence-weighted scoring
        from attestdb.core.vocabulary import directional_confidence

        supporting = [ie for ie in all_indirect if ie.direction == "supporting"]
        contradicting = [ie for ie in all_indirect if ie.direction == "contradicting"]

        # Combine direct + indirect evidence
        total_supporting = direct_supporting + len(supporting)
        total_contradicting = len(direct_contradictions) + len(contradicting)
        dir_conf, dir_verdict = directional_confidence(total_supporting, total_contradicting)

        score = 0.5
        score += min(corroborations * 0.15, 0.3)
        score += min(dir_conf * 0.4, 0.4)
        score -= min(total_contradicting * 0.05, 0.3)
        score += min(gaps_closed * 0.05, 0.1)
        score = max(0.0, min(1.0, score))

        # Verdict logic: clear precedence, no ambiguity
        has_support = corroborations > 0 or total_supporting > 0
        has_contradiction = len(direct_contradictions) > 0 or total_contradicting > 0

        if not has_support and not has_contradiction:
            verdict = "insufficient_data" if not gaps_closed else "plausible"
        elif has_support and not has_contradiction:
            verdict = "supported" if dir_verdict in ("strong", "moderate") or corroborations > 0 else "plausible"
        elif has_contradiction and not has_support:
            verdict = "contradicted"
        else:
            # Both supporting and contradicting evidence exist
            if total_supporting > total_contradicting * 2:
                verdict = "supported" if dir_verdict == "strong" else "plausible"
            elif total_contradicting > total_supporting * 2:
                verdict = "contradicted"
            else:
                verdict = "contested"

        # Best predicted predicate
        if supporting:
            predicted = supporting[0].predicted_predicate
        elif all_indirect:
            predicted = all_indirect[0].predicted_predicate
        else:
            predicted = ""

        # Build explanation
        parts = []
        if direct_supporting:
            parts.append(f"{direct_supporting} direct supporting claim(s)")
        if direct_contradictions:
            parts.append(f"{len(direct_contradictions)} direct contradiction(s)")
        if corroborations:
            parts.append(f"{corroborations} existing corroboration(s)")
        if supporting:
            parts.append(f"{len(supporting)} causal path(s) supporting")
        if contradicting:
            parts.append(f"{len(contradicting)} causal path(s) contradicting")
        neutral = [ie for ie in all_indirect if ie.direction == "neutral"]
        if neutral and not supporting and not contradicting:
            parts.append(f"{len(neutral)} indirect path(s) (neutral)")
        if gaps_closed:
            parts.append(f"closes {gaps_closed} gap(s)")
        explanation = "; ".join(parts) if parts else "No evidence found in graph."

        return SandboxVerdict(
            hypothesis=desc,
            verdict=verdict,
            direct_contradictions=direct_contradictions,
            direct_corroborations=corroborations,
            indirect_evidence=all_indirect,
            predicted_predicate=predicted,
            gaps_closed=gaps_closed,
            gap_descriptions=gap_descs,
            follow_up_hypotheses=follow_ups,
            confidence_score=score,
            explanation=explanation,
        )

    def build_entity_aliases(self, entity_type: str = "gene", batch_size: int = 50000) -> dict[str, str]:
        """Build an alias map by matching display names across entity ID formats.

        For each entity of the given type, groups entities by display name and
        maps non-NCBI IDs (gene_tp53, gene_pharmgkb:pa36679) to NCBI Gene IDs
        (gene_7157). Also maps protein_NCBI to gene_NCBI.

        Returns a dict mapping non-canonical IDs to canonical NCBI IDs.
        Pass the result to predict(entity_aliases=...) for cross-database
        entity resolution without the same_as performance penalty.
        """
        aliases: dict[str, str] = {}
        from collections import defaultdict

        # Paginate through all entities of this type
        ncbi_by_name: dict[str, tuple[str, int]] = {}
        non_ncbi: list[tuple[str, str, int]] = []
        offset = 0

        while True:
            entities = self.db._store.list_entities(entity_type, 0, offset, batch_size)
            if not entities:
                break
            for e in entities:
                eid = e["id"]
                name = e.get("name", "?")
                cc = e.get("claim_count", 0)
                if not eid.startswith(f"{entity_type}_") or name == "?" or len(name) < 2:
                    continue
                clean_name = name.split("|")[0].strip().lower()
                rest = eid[len(entity_type) + 1:]
                if rest.isdigit():
                    if clean_name not in ncbi_by_name or cc > ncbi_by_name[clean_name][1]:
                        ncbi_by_name[clean_name] = (eid, cc)
                else:
                    non_ncbi.append((eid, clean_name, cc))
            offset += batch_size
            if len(entities) < batch_size:
                break

        for eid, name, cc in non_ncbi:
            if name in ncbi_by_name:
                canonical = ncbi_by_name[name][0]
                if eid != canonical:
                    aliases[eid] = canonical

        # Also map protein_NCBI → gene_NCBI
        if entity_type == "gene":
            offset = 0
            while True:
                entities = self.db._store.list_entities("protein", 0, offset, batch_size)
                if not entities:
                    break
                for e in entities:
                    eid = e["id"]
                    if eid.startswith("protein_"):
                        rest = eid[8:]
                        if rest.isdigit():
                            gene_id = f"gene_{rest}"
                            if self.db._store.get_entity(gene_id):
                                aliases[eid] = gene_id
                offset += batch_size
                if len(entities) < batch_size:
                    break

        return aliases

    def predict(
        self,
        entity_id: str,
        max_intermediaries: int = 100,
        min_paths: int = 3,
        min_consensus: float = 0.65,
        directional_only: bool = False,
        entity_aliases: dict[str, str] | None = None,
        extra_causal_predicates: set[str] | None = None,
    ) -> list["Prediction"]:
        """Discover novel regulatory predictions via causal composition.

        Follows causal edges from the entity through intermediaries, then
        composes predicates at each 2-hop path. Filters for convergent
        predictions where multiple independent intermediaries agree on direction.

        Returns predictions ranked by number of supporting intermediaries,
        with genuine gaps (no existing connection) sorted first.

        Works with read-only databases and scales to millions of claims.

        Usage:
            predictions = db.predict("gene_7157")  # TP53
            for p in predictions[:10]:
                print(f"{entity_id} --[{p.predicted_predicate}]--> {p.target}")
                print(f"  {p.supporting_paths} supporting, {p.opposing_paths} opposing")
        """
        from attestdb.core.types import IndirectEvidence, Prediction
        from attestdb.core.vocabulary import (
            CAUSAL_PREDICATE_TYPES,
            CAUSAL_PREDICATES,
            OPPOSITE_PREDICATES,
            PREDICATE_COMPOSITION,
            PREDICATE_EQUIVALENCE,
            compose_predicates,
            predicates_agree,
        )

        n_eid = normalize_entity_id(entity_id)

        # Entity alias resolution: map non-canonical IDs to canonical
        # (e.g., gene_tp53 → gene_7157). Applied at Python level, not
        # store level, to avoid the same_as union-find performance issue.
        if entity_aliases:
            _alias_set = set(entity_aliases.keys())
            _resolve = lambda eid: entity_aliases[eid] if eid in _alias_set else eid
        else:
            _resolve = lambda eid: eid

        # Try fast path: Rust-native outgoing_causal_edges (no full claim deserialization)
        _has_fast_path = hasattr(self.db._store, "outgoing_causal_edges")

        # Collect direct neighbors and causal neighbors
        direct_neighbors: dict[str, set[str]] = {}
        causal_neighbors: dict[str, tuple[str, float]] = {}
        neighbor_causal_preds: dict[str, set[str]] = {}
        outgoing_causal: dict[str, tuple[str, float]] = {}
        incoming_causal: dict[str, tuple[str, float]] = {}

        # Build effective causal predicate set — includes hardcoded bio predicates,
        # discovered domain predicates (via predicate_type), and user-provided extras
        _causal_set = self._resolve_causal_predicates(extra_causal_predicates, directional_only)

        _used_fast_path = False
        if _has_fast_path:
            # Fast path: use Rust-native lightweight edge query for causal edges
            causal_pred_list = list(_causal_set)
            edges = self.db._store.outgoing_causal_edges(n_eid, causal_pred_list)
            _used_fast_path = len(edges) > 0
            for raw_target, pred, conf in edges:
                target = _resolve(raw_target)
                if target == n_eid:
                    continue
                direct_neighbors.setdefault(target, set()).add(pred)
                neighbor_causal_preds.setdefault(target, set()).add(pred)
                # Prefer directional predicates over "regulates"
                if target not in outgoing_causal:
                    outgoing_causal[target] = (pred, conf)
                else:
                    existing_pred = outgoing_causal[target][0]
                    # Directional beats non-directional, even at lower confidence
                    if existing_pred == "regulates" and pred != "regulates":
                        outgoing_causal[target] = (pred, conf)
                    elif pred != "regulates" and conf > outgoing_causal[target][1]:
                        outgoing_causal[target] = (pred, conf)
                    elif existing_pred != "regulates" and pred == "regulates":
                        pass  # keep existing directional
                    elif conf > outgoing_causal[target][1]:
                        outgoing_causal[target] = (pred, conf)
        if not _has_fast_path or not _used_fast_path:
            # Slow path: iterate all claims (also used when causal_adj index is empty)
            claims = self.db._store.claims_for(n_eid, None, None, 0.0)
            for d in claims:
                c = claim_from_dict(d)
                _is_causal = (c.predicate.id in _causal_set
                              or c.predicate.predicate_type in CAUSAL_PREDICATE_TYPES)
                if not _is_causal:
                    raw_n = c.object.id if c.subject.id == n_eid else c.subject.id
                    neighbor = _resolve(raw_n)
                    if neighbor != n_eid:
                        direct_neighbors.setdefault(neighbor, set()).add(c.predicate.id)
                    continue
                if c.subject.id == n_eid:
                    mid = _resolve(c.object.id)
                    if mid == n_eid:
                        continue
                    direct_neighbors.setdefault(mid, set()).add(c.predicate.id)
                    neighbor_causal_preds.setdefault(mid, set()).add(c.predicate.id)
                    if mid not in outgoing_causal:
                        outgoing_causal[mid] = (c.predicate.id, c.confidence)
                    else:
                        ep = outgoing_causal[mid][0]
                        if ep == "regulates" and c.predicate.id != "regulates":
                            outgoing_causal[mid] = (c.predicate.id, c.confidence)
                        elif c.predicate.id != "regulates" and c.confidence > outgoing_causal[mid][1]:
                            outgoing_causal[mid] = (c.predicate.id, c.confidence)
                        elif ep != "regulates" and c.predicate.id == "regulates":
                            pass
                        elif c.confidence > outgoing_causal[mid][1]:
                            outgoing_causal[mid] = (c.predicate.id, c.confidence)
                else:
                    mid = _resolve(c.subject.id)
                    if mid == n_eid:
                        continue
                    direct_neighbors.setdefault(mid, set()).add(c.predicate.id)
                    neighbor_causal_preds.setdefault(mid, set()).add(c.predicate.id)
                    if mid not in incoming_causal or c.confidence > incoming_causal[mid][1]:
                        incoming_causal[mid] = (c.predicate.id, c.confidence)

        # Merge: outgoing neighbors preferred, incoming as fallback
        causal_neighbors = {**incoming_causal, **outgoing_causal}
        outgoing_set = set(outgoing_causal.keys())

        # Score intermediaries (soft weighting, not hard filtering)
        # Ablation showed hard filters reduce recall by up to 47%
        scored_neighbors: list[tuple[str, str, float, float]] = []  # (mid, pred, conf, weight)
        for mid, (pred1, conf1) in causal_neighbors.items():
            weight = conf1

            # Contradictory legs: discount 0.5× instead of skip
            preds = neighbor_causal_preds.get(mid, set())
            has_contradiction = any(
                OPPOSITE_PREDICATES.get(p, "") in preds for p in preds
            )
            if has_contradiction:
                weight *= 0.5

            # Hub discount: scale by 1/log(claims) instead of skip
            mid_entity = self.db._store.get_entity(mid)
            if mid_entity:
                cc = mid_entity.get("claim_count", 0)
                if cc > 20_000:
                    import math
                    weight *= 1.0 / math.log(cc / 1000 + 1)

            scored_neighbors.append((mid, pred1, conf1, weight))

        # Diversified sampling: mix high-confidence and random intermediaries
        # (ablation showed random selection +47% recall over confidence-sorted)
        import random as _random
        _random.seed(hash(n_eid) & 0xFFFFFFFF)

        # Take top half by weight, bottom half random from remainder
        scored_neighbors.sort(key=lambda x: -x[3])
        half = max_intermediaries // 2
        top_half = scored_neighbors[:half]
        rest = scored_neighbors[half:]
        if rest:
            _random.shuffle(rest)
            random_half = rest[:max_intermediaries - half]
        else:
            random_half = []
        selected = top_half + random_half

        # 2-hop BFS through selected intermediaries
        # target -> {composed_pred -> [IndirectEvidence]}
        target_preds: dict[str, dict[str, list[IndirectEvidence]]] = {}
        checked = 0

        for mid, pred1, conf1, weight in selected:
            if checked >= max_intermediaries:
                break
            checked += 1

            # Second hop: get outgoing causal edges from intermediary
            hop2_iter = []
            if _has_fast_path:
                mid_edges = self.db._store.outgoing_causal_edges(
                    mid, list(_causal_set)
                )
                hop2_iter = [
                    (_resolve(t), pred2, conf2)
                    for t, pred2, conf2 in mid_edges
                    if _resolve(t) != n_eid and t != mid
                ]
            if not hop2_iter:
                # Slow path fallback
                mid_claims = self.db._store.claims_for(mid, None, None, 0.0)[:300]
                for d in mid_claims:
                    mc = claim_from_dict(d)
                    if (mc.subject.id == mid
                            and mc.predicate.id in _causal_set):
                        resolved = _resolve(mc.object.id)
                        if resolved != n_eid and resolved != mid:
                            hop2_iter.append((resolved, mc.predicate.id, mc.confidence))

            for target, pred2, conf2 in hop2_iter:
                composed = compose_predicates(pred1, pred2)
                if composed == "associated_with":
                    continue
                # Skip if this exact predicate already exists directly
                if composed in direct_neighbors.get(target, set()):
                    continue
                # Path confidence incorporates intermediary quality weight
                is_causal = mid in outgoing_set
                path_conf = weight * conf2  # weight already includes conf1 + discounts
                if not is_causal:
                    path_conf *= 0.5  # co-regulation discount

                target_preds.setdefault(target, {}).setdefault(composed, []).append(
                    IndirectEvidence(
                        path=[n_eid, mid, target],
                        predicates=[pred1, pred2],
                        predicted_predicate=composed,
                        confidence=path_conf,
                        direction="supporting" if is_causal else "co-regulation",
                    )
                )

        # Score and filter
        results: list[Prediction] = []
        for target, pred_map in target_preds.items():
            for pred, paths in pred_map.items():
                unique_mids = set(ie.path[1] for ie in paths)
                if len(unique_mids) < min_paths:
                    continue

                # Count opposing paths (any predicate in the opposite equivalence class)
                pred_class = PREDICATE_EQUIVALENCE.get(pred)
                opp_count = 0
                for other_pred, other_paths in pred_map.items():
                    if other_pred == pred:
                        continue
                    other_class = PREDICATE_EQUIVALENCE.get(other_pred)
                    if pred_class and other_class and pred_class != other_class:
                        opp_count += len(other_paths)
                    elif OPPOSITE_PREDICATES.get(pred) == other_pred:
                        opp_count += len(other_paths)
                total = len(paths) + opp_count
                consensus = len(paths) / total if total > 0 else 0

                if consensus < min_consensus:
                    continue

                # Use adjacency index for gap detection (fast, no claim loading)
                if _has_fast_path:
                    is_gap = not self.db._store.path_exists(n_eid, target, 1)
                else:
                    is_gap = target not in direct_neighbors
                existing = sorted(direct_neighbors.get(target, set()))

                # Keep top evidence paths by confidence
                top_evidence = sorted(paths, key=lambda ie: -ie.confidence)[:10]

                results.append(Prediction(
                    target=target,
                    predicted_predicate=pred,
                    supporting_paths=len(paths),
                    opposing_paths=opp_count,
                    consensus=consensus,
                    intermediaries=len(unique_mids),
                    is_gap=is_gap,
                    existing_predicates=existing,
                    evidence=top_evidence,
                ))

        # Sort: gaps first, then by intermediary count
        results.sort(key=lambda p: (not p.is_gap, -p.intermediaries, p.opposing_paths))
        return results

    # --- Knowledge-Intelligence APIs ---

    def evolution(self, entity_id: str, since: str | int | None = None) -> EvolutionReport:
        """Entity-specific knowledge evolution over time.

        Shows how understanding of an entity has changed: new connections,
        confidence shifts, retractions, source diversification.

        Args:
            entity_id: The entity to analyze.
            since: Cutoff — ISO string, nanosecond int, or None (full history).

        Returns:
            EvolutionReport with trajectory and change details.
        """
        eid = normalize_entity_id(entity_id)

        # Timestamp resolution
        if since is None:
            cutoff = 0
        elif isinstance(since, int):
            cutoff = since
        else:
            from datetime import datetime
            dt = datetime.fromisoformat(str(since))
            cutoff = int(dt.timestamp() * 1_000_000_000)

        # Include retracted claims to track retractions in evolution
        self.db._set_include_retracted(True)
        try:
            all_claims = self.db.claims_for(eid)[:500]
        finally:
            self.db._set_include_retracted(False)
        if not all_claims:
            return EvolutionReport(entity_id=eid, since_timestamp=cutoff)

        before = [c for c in all_claims if c.timestamp <= cutoff]
        after = [c for c in all_claims if c.timestamp > cutoff]

        # Neighbors before and after
        def _neighbors(claims: list[Claim]) -> set[str]:
            n = set()
            for c in claims:
                if c.subject.id == eid:
                    n.add(c.object.id)
                else:
                    n.add(c.subject.id)
            return n

        neighbors_before = _neighbors(before)
        neighbors_after = _neighbors(after)
        new_connections = sorted(neighbors_after - neighbors_before)

        # Retracted claims in the after period
        retracted = sum(1 for c in after if c.status == ClaimStatus.TOMBSTONED)

        # Source diversification
        src_before = sorted({c.provenance.source_type for c in before})
        src_after = sorted({c.provenance.source_type for c in all_claims})
        new_src = sorted(set(src_after) - set(src_before))

        # Confidence changes: compare corroboration counts by content_id
        content_before: dict[str, list[Claim]] = {}
        for c in before:
            content_before.setdefault(c.content_id, []).append(c)
        content_after: dict[str, list[Claim]] = {}
        for c in all_claims:
            content_after.setdefault(c.content_id, []).append(c)

        confidence_changes: list[ConfidenceChange] = []
        new_corroborations = 0
        for cid, after_claims in content_after.items():
            before_claims = content_before.get(cid, [])
            if before_claims and len(after_claims) != len(before_claims):
                conf_b = max(c.confidence for c in before_claims)
                conf_a = max(c.confidence for c in after_claims)
                delta = conf_a - conf_b
                # Determine target entity from the first claim
                fc = after_claims[0]
                target = fc.object.id if fc.subject.id == eid else fc.subject.id
                reason = (
                    "new_corroboration"
                    if len(after_claims) > len(before_claims)
                    else "source_retracted"
                )
                confidence_changes.append(ConfidenceChange(
                    content_id=cid,
                    predicate=fc.predicate.id,
                    target_entity=target,
                    confidence_before=conf_b,
                    confidence_after=conf_a,
                    delta=delta,
                    reason=reason,
                ))
            if len(before_claims) == 1 and len(after_claims) > 1:
                new_corroborations += 1

        # Trajectory
        new_claim_count = len(after)
        if new_claim_count > retracted and (new_connections or new_corroborations > 0):
            trajectory = "growing"
        elif retracted > new_claim_count:
            trajectory = "declining"
        else:
            trajectory = "stable"

        return EvolutionReport(
            entity_id=eid,
            since_timestamp=cutoff,
            new_connections=new_connections,
            new_claims=new_claim_count,
            retracted_claims=retracted,
            confidence_changes=confidence_changes,
            new_corroborations=new_corroborations,
            source_types_before=src_before,
            source_types_after=src_after,
            new_source_types=new_src,
            trajectory=trajectory,
            total_claims_before=len(before),
            total_claims_after=len(all_claims),
        )

    def trace(
        self,
        entity_a: str,
        entity_b: str,
        max_depth: int = 3,
        top_k: int = 5,
    ) -> list[ReasoningChain]:
        """Reasoning chains with source-overlap-discounted confidence.

        Unlike find_paths(), accounts for source overlap (shared sources
        reduce effective confidence), includes evidence quotes from payloads,
        flags contradictions per hop, computes a reliability score.

        Args:
            entity_a: Start entity.
            entity_b: End entity.
            max_depth: Maximum path length.
            top_k: Maximum chains to return.

        Returns:
            List of ReasoningChain sorted by chain_confidence desc.
        """
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        paths = self.db.find_paths(entity_a, entity_b, max_depth=max_depth, top_k=top_k)
        if not paths:
            return []

        b_norm = normalize_entity_id(entity_b)
        chains: list[ReasoningChain] = []

        for path in paths:
            hops: list[ReasoningHop] = []
            # PathStep.entity_id is the SOURCE of each hop.
            # Destination = next step's entity_id, or entity_b for last step.
            for i, step in enumerate(path.steps):
                from_eid = step.entity_id
                if i + 1 < len(path.steps):
                    to_eid = path.steps[i + 1].entity_id
                else:
                    to_eid = b_norm

                # Get all claims between from_eid and to_eid
                from_claims = self.db.claims_for(from_eid)[:300]
                hop_claims = [
                    c for c in from_claims
                    if (c.subject.id == from_eid and c.object.id == to_eid)
                    or (c.object.id == from_eid and c.subject.id == to_eid)
                ]

                source_types = list({c.provenance.source_type for c in hop_claims})
                source_ids = list({c.provenance.source_id for c in hop_claims})

                # Evidence text from payload
                evidence = ""
                for c in hop_claims:
                    if c.payload and c.payload.data:
                        text = c.payload.data.get("text", c.payload.data.get("evidence", ""))
                        if text:
                            evidence = str(text)
                            break

                # Contradiction check
                has_contradiction = False
                contradiction_pred = ""
                predicates = {c.predicate.id for c in hop_claims}
                for pred in predicates:
                    opp = OPPOSITE_PREDICATES.get(pred)
                    if opp and opp in predicates:
                        has_contradiction = True
                        contradiction_pred = opp
                        break

                best_conf = max((c.confidence for c in hop_claims), default=step.confidence)

                hops.append(ReasoningHop(
                    from_entity=from_eid,
                    to_entity=to_eid,
                    predicate=step.predicate,
                    confidence=best_conf,
                    source_types=source_types,
                    source_ids=source_ids,
                    evidence_text=evidence,
                    has_contradiction=has_contradiction,
                    contradiction_predicate=contradiction_pred,
                ))

            # Source overlap detection
            raw_conf = 1.0
            for h in hops:
                raw_conf *= h.confidence

            overlaps: list[SourceOverlap] = []
            for i in range(len(hops)):
                for j in range(i + 1, len(hops)):
                    shared = sorted(set(hops[i].source_ids) & set(hops[j].source_ids))
                    if shared:
                        discount = 1.0 / (1.0 + 0.3 * len(shared))
                        overlaps.append(SourceOverlap(
                            hop_a_index=i,
                            hop_b_index=j,
                            shared_sources=shared,
                            discount_factor=discount,
                        ))

            discount_product = 1.0
            for o in overlaps:
                discount_product *= o.discount_factor
            chain_conf = raw_conf * discount_product

            # Reliability score
            total_sources = len({sid for h in hops for sid in h.source_ids})
            source_diversity = total_sources / (len(hops) * 2) if hops else 0.0
            contradiction_ratio = (
                (sum(1 for h in hops if h.has_contradiction)
                 / len(hops))
                if hops else 0.0
            )
            reliability = 0.6 * chain_conf + 0.3 * source_diversity - 0.1 * contradiction_ratio
            reliability = max(0.0, min(1.0, reliability))

            chains.append(ReasoningChain(
                hops=hops,
                raw_confidence=raw_conf,
                source_overlaps=overlaps,
                chain_confidence=chain_conf,
                length=len(hops),
                reliability_score=reliability,
            ))

        chains.sort(key=lambda c: c.chain_confidence, reverse=True)
        return chains[:top_k]

    def suggest_investigations(self, top_k: int = 10) -> list[Investigation]:
        """Unified prioritized investigation recommendations.

        Synthesizes all existing insight signals (confidence alerts,
        blindspots, gaps, bridges) into a single ranked list.

        Args:
            top_k: Maximum recommendations to return.

        Returns:
            List of Investigation sorted by priority_score desc.
        """
        investigations: list[Investigation] = []
        seen_entities: set[str] = set()

        # 1. Confidence alerts → single_source / stale
        try:
            alerts = self.db.find_confidence_alerts(min_claims=2)
            for alert in alerts:
                if alert.entity_id in seen_entities:
                    continue
                seen_entities.add(alert.entity_id)
                # "single_source", "stale_evidence", "mixed_source_quality"
                signal = alert.alert_type.value
                if signal == "stale_evidence":
                    signal = "stale"
                elif signal == "mixed_source_quality":
                    signal = "confidence_gap"
                base = {"single_source": 0.6, "stale": 0.2, "confidence_gap": 0.5}.get(signal, 0.3)
                investigations.append(Investigation(
                    entity_id=alert.entity_id,
                    entity_type=alert.entity_type,
                    reason=alert.explanation,
                    signal_type=signal,
                    priority_score=base,
                    suggested_action=(
                        f"Seek additional sources for {alert.entity_id}"
                        if signal == "single_source"
                        else f"Update evidence for {alert.entity_id}"
                        if signal == "stale"
                        else f"Reconcile confidence spread for {alert.entity_id}"
                    ),
                ))
        except Exception:
            logger.debug("Could not compute confidence alerts for investigations", exc_info=True)

        # 2. Blindspots → single_source (deduplicate with alerts)
        try:
            bs = self.db.blindspots(min_claims=1)
            for eid in bs.single_source_entities:
                if eid in seen_entities:
                    continue
                seen_entities.add(eid)
                entity = self.db._store.get_entity(eid)
                etype = entity_summary_from_dict(entity).entity_type if entity else ""
                investigations.append(Investigation(
                    entity_id=eid,
                    entity_type=etype,
                    reason=f"Entity {eid} has only one source",
                    signal_type="single_source",
                    priority_score=0.6,
                    suggested_action=f"Seek additional sources for {eid}",
                ))
        except Exception:
            logger.debug("Could not compute blindspots for investigations", exc_info=True)

        # 3. Knowledge gaps (if vocabulary registered)
        try:
            constraints = self.db._store.get_predicate_constraints()
            if constraints:
                from attestdb.intelligence.insight_engine import build_expected_patterns
                patterns = build_expected_patterns(constraints)
                gaps = self.db.find_gaps(patterns, min_claims=1)
                for gap in gaps:
                    if gap.entity_id in seen_entities:
                        continue
                    seen_entities.add(gap.entity_id)
                    missing = ", ".join(gap.missing_predicate_types[:3])
                    investigations.append(Investigation(
                        entity_id=gap.entity_id,
                        entity_type=gap.entity_type,
                        reason=f"Missing predicates: {missing}",
                        signal_type="missing_predicate",
                        priority_score=0.4,
                        suggested_action=f"Add {missing} relationships for {gap.entity_id}",
                    ))
        except Exception:
            logger.debug("Could not compute gaps for investigations", exc_info=True)

        # 4. Predicted links (bridges)
        try:
            bridges = self.db.find_bridges(top_k=top_k * 2)
            for bridge in bridges:
                eid = bridge.entity_a
                if eid in seen_entities:
                    eid = bridge.entity_b
                if eid in seen_entities:
                    continue
                seen_entities.add(eid)
                entity = self.db._store.get_entity(eid)
                etype = entity_summary_from_dict(entity).entity_type if entity else ""
                score = 0.3 + bridge.similarity * 0.3
                investigations.append(Investigation(
                    entity_id=eid,
                    entity_type=etype,
                    reason=(
                        f"Predicted link: {bridge.entity_a} "
                        f"↔ {bridge.entity_b} "
                        f"(similarity={bridge.similarity:.2f})"
                    ),
                    signal_type="predicted_link",
                    priority_score=score,
                    suggested_action=(
                        f"Investigate link between "
                        f"{bridge.entity_a} and {bridge.entity_b}"
                    ),
                ))
        except Exception:
            logger.debug("Could not compute bridges for investigations", exc_info=True)

        # Centrality boost + affected downstream
        adj = self.db.get_adjacency_list()
        for inv in investigations:
            degree = len(adj.get(inv.entity_id, set()))
            boost = min(degree / 20.0, 1.0)
            inv.priority_score *= (1.0 + 0.5 * boost)
            # Use adjacency degree as proxy — avoids materializing all claims
            inv.affected_downstream = degree

        investigations.sort(key=lambda x: x.priority_score, reverse=True)
        return investigations[:top_k]

    def close_gaps(self, hypothesis=None, top_k=5, search_fn=None, use_curator=True):
        """Close knowledge gaps by researching and ingesting new evidence.

        Two modes:
        - **Hypothesis mode** (hypothesis is not None): test_hypothesis → extract
          gaps → research each → re-test → return before/after.
        - **Blindspot mode** (hypothesis is None): suggest_investigations → research
          each → measure blindspot improvement.

        Args:
            hypothesis: Natural-language hypothesis to close gaps for.
            top_k: Max gaps/investigations to research.
            search_fn: Optional callback(question_text) -> text.
            use_curator: Whether to triage discovered claims through curator.

        Returns:
            CloseGapsReport with before/after metrics.
        """
        return self.db._get_researcher().close_gaps(
            hypothesis=hypothesis, top_k=top_k,
            search_fn=search_fn, use_curator=use_curator,
        )

    def test_hypothesis(self, hypothesis: str) -> HypothesisVerdict:
        """Evaluate a natural-language hypothesis against the knowledge graph.

        Takes a hypothesis like "aspirin reduces inflammation via COX-2",
        uses 1 LLM call to parse entity pairs, then evaluates multi-hop
        chains against the graph.

        Args:
            hypothesis: Natural language hypothesis to test.

        Returns:
            HypothesisVerdict with supporting/contradicting evidence chains.
        """
        import math

        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        # Step 1: Parse hypothesis — LLM or fallback
        parsed_entities: list[dict] = []
        parsed_rels: list[dict] = []
        fallback_missing: list[str] = []

        llm_response = self.db._llm_call(
            f"Extract entities and relationships from this hypothesis. "
            f"Return ONLY valid JSON, no markdown:\n"
            f'{{"entities": [{{"name": "...", "type": "..."}}], '
            f'"relationships": [{{"from": "...", "to": "...", "predicate": "..."}}]}}\n\n'
            f"Hypothesis: {hypothesis}",
            max_tokens=512,
            temperature=0.0,
        )

        if llm_response:
            try:
                # Strip markdown fences if present
                text = llm_response.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                parsed = json.loads(text.strip())
                parsed_entities = parsed.get("entities", [])
                parsed_rels = parsed.get("relationships", [])
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: extract words and search
        if not parsed_entities:
            words = [w.strip(".,;:!?()\"'") for w in hypothesis.split() if len(w) >= 2]
            stop = {"is", "an", "at", "or", "in", "on", "to", "of", "by", "as", "if", "so",
                    "up", "do", "no", "my", "we", "he", "it", "am", "be",
                    "the", "and", "for", "are", "but", "not", "can", "had", "has", "was",
                    "that", "this", "with", "from", "into", "than", "also", "have", "been",
                    "does", "will", "would", "could", "should", "through", "between", "which",
                    "their", "there", "about", "after", "before", "over", "under", "very",
                    "much", "more", "some", "such", "most", "each", "only", "both", "then",
                    "just", "like", "when", "what", "where", "them", "they", "your",
                    "may", "via", "how", "its", "out", "any", "all", "our", "who", "new",
                    "affects", "causes", "reduces", "increases", "leads", "involves",
                    "through", "because", "since", "using", "during"}
            words = [w for w in words if w.lower() not in stop]
            seen_ids: set[str] = set()
            fallback_missing: list[str] = []
            for word in words:
                results = self.db.search_entities(word, top_k=3)
                if results:
                    best = max(results, key=lambda e: e.claim_count)
                    if best.id not in seen_ids:
                        seen_ids.add(best.id)
                        parsed_entities.append({"name": best.id, "type": best.entity_type})
                else:
                    fallback_missing.append(word)

        # Step 2: Entity resolution
        entities_found: list[str] = []
        entities_missing: list[str] = list(fallback_missing)
        resolved: dict[str, str] = {}  # parsed name → resolved entity_id

        for ent in parsed_entities:
            name = ent.get("name", "")
            if not name:
                continue
            # Search with both original and lowercase — text index is case-sensitive
            results = self.db.search_entities(name, top_k=5)
            if name != name.lower():
                results += self.db.search_entities(name.lower(), top_k=5)
            # Deduplicate
            seen_res: dict[str, object] = {}
            for r in results:
                if r.id not in seen_res or r.claim_count > seen_res[r.id].claim_count:
                    seen_res[r.id] = r
            results = list(seen_res.values())
            if results:
                best = max(results, key=lambda e: e.claim_count)
                resolved[name.lower()] = best.id
                entities_found.append(best.id)
            else:
                entities_missing.append(name)

        # Insufficient data check
        if len(entities_found) < 2:
            return HypothesisVerdict(
                hypothesis=hypothesis,
                verdict="insufficient_data",
                entities_found=sorted(set(entities_found)),
                entities_missing=entities_missing,
                suggested_next_steps=(
                    [f"Add data about {e}" for e in entities_missing]
                    or ["Add more entities to the knowledge graph"]
                ),
            )

        # Build relationships to evaluate — from LLM or from entity pairs
        relationships: list[tuple[str, str, str]] = []  # (from_id, to_id, predicate)
        if parsed_rels:
            for rel in parsed_rels:
                from_name = rel.get("from", "").lower()
                to_name = rel.get("to", "").lower()
                pred = rel.get("predicate", "relates_to")
                from_id = resolved.get(from_name)
                to_id = resolved.get(to_name)
                if from_id and to_id:
                    relationships.append((from_id, to_id, pred))

        # Fallback: pair consecutive resolved entities
        if not relationships:
            found_ids = sorted(set(entities_found))
            for i in range(len(found_ids) - 1):
                relationships.append((found_ids[i], found_ids[i + 1], "relates_to"))

        # Step 3: Evidence gathering per relationship
        supporting_chains: list[EvidenceChain] = []
        contradicting_chains: list[EvidenceChain] = []
        confidence_gaps: list[ConfidenceGap] = []
        best_confidences: list[float] = []

        for from_id, to_id, expected_pred in relationships:
            paths = self.db.find_paths(from_id, to_id, max_depth=3, top_k=5)

            if not paths:
                confidence_gaps.append(ConfidenceGap(
                    from_entity=from_id,
                    to_entity=to_id,
                    issue="missing",
                    explanation=f"No path found between {from_id} and {to_id}",
                ))
                continue

            rel_best = 0.0
            for path in paths:
                direction = "supporting"
                # Check if path predicates oppose the expected predicate
                for step in path.steps:
                    opp = OPPOSITE_PREDICATES.get(expected_pred)
                    if opp and step.predicate == opp:
                        direction = "contradicting"
                        break
                    opp2 = OPPOSITE_PREDICATES.get(step.predicate)
                    if opp2 and opp2 == expected_pred:
                        direction = "contradicting"
                        break

                chain = EvidenceChain(
                    steps=list(path.steps),
                    propagated_confidence=path.total_confidence,
                    direction=direction,
                )
                if direction == "supporting":
                    supporting_chains.append(chain)
                    rel_best = max(rel_best, path.total_confidence)
                else:
                    contradicting_chains.append(chain)

                # Weak hop detection
                for step in path.steps:
                    if step.confidence < 0.3:
                        confidence_gaps.append(ConfidenceGap(
                            from_entity=step.entity_id,
                            to_entity=to_id,
                            issue="weak",
                            current_confidence=step.confidence,
                            explanation=(
                                f"Weak evidence for {step.predicate} "
                                f"(confidence={step.confidence:.2f})"
                            ),
                        ))

            if rel_best > 0:
                best_confidences.append(rel_best)

        # Step 5: Verdict
        has_supporting = len(supporting_chains) > 0
        has_contradicting = len(contradicting_chains) > 0
        all_rels_have_support = all(
            any(
                any(s.entity_id == to_id for s in chain.steps)
                for chain in supporting_chains
                if chain.propagated_confidence >= 0.3
            )
            for _, to_id, _ in relationships
        ) if relationships else False

        if all_rels_have_support and not has_contradicting:
            verdict = "supported"
        elif has_contradicting and not has_supporting:
            verdict = "contradicted"
        elif has_supporting and has_contradicting:
            verdict = "partial"
        elif has_supporting:
            verdict = "partial"
        else:
            verdict = "unsupported"

        # Verdict confidence = geometric mean of best chain confidences
        if best_confidences:
            log_sum = sum(math.log(c) for c in best_confidences)
            verdict_confidence = math.exp(
                log_sum / len(best_confidences)
            )
        else:
            verdict_confidence = 0.0

        # Step 6: Suggested next steps
        next_steps: list[str] = []
        for gap in confidence_gaps:
            if gap.issue == "missing":
                next_steps.append(f"Investigate link between {gap.from_entity} and {gap.to_entity}")
            elif gap.issue == "weak":
                next_steps.append(f"Seek corroboration for {gap.from_entity} ({gap.explanation})")
        for e in entities_missing:
            next_steps.append(f"Add data about {e}")

        return HypothesisVerdict(
            hypothesis=hypothesis,
            verdict=verdict,
            verdict_confidence=verdict_confidence,
            supporting_chains=supporting_chains,
            contradicting_chains=contradicting_chains,
            confidence_gaps=confidence_gaps,
            entities_found=sorted(set(entities_found)),
            entities_missing=entities_missing,
            suggested_next_steps=next_steps[:10],
        )


    # --- Discovery engine ---

    def discover(self, top_k: int = 10) -> list[Discovery]:
        """Proactive hypothesis generation from graph structure.

        Three signals (all pure computation, no LLM):
        1. Bridge predictions — ensemble-scored pairs with composed predicates
        2. Cross-domain insights — topology bridge entities with differing predicates
        3. Chain completion — 2-hop pairs missing a direct connection

        Returns list of Discovery sorted by confidence * novelty, truncated to top_k.
        """
        from attestdb.core.vocabulary import predict_predicate_from_paths

        entities = self.db.list_entities()
        if len(entities) < 3:
            return []

        discoveries: list[Discovery] = []
        entity_type_map = {e.id: e.entity_type for e in entities}
        entity_name_map = {e.id: e.name or e.id for e in entities}

        # Degree map for novelty scoring
        adj = self.db.get_adjacency_list()
        degree = {eid: len(neighbors) for eid, neighbors in adj.items()}
        max_degree = max(degree.values()) if degree else 1

        # --- Signal 1: Bridge predictions via EnsembleScorer ---
        try:
            from attestdb_enterprise.ensemble_scorer import EnsembleScorer

            from attestdb.intelligence.graph_embeddings import (
                compute_graph_embeddings,
                compute_weighted_graph_embeddings,
            )

            weighted_adj = self.db.get_weighted_adjacency()
            embeddings = compute_graph_embeddings(adj, dim=min(64, len(adj)))
            w_embeddings = compute_weighted_graph_embeddings(weighted_adj, dim=min(64, len(adj)))
            # Merge embeddings (weighted wins if available)
            for k, v in w_embeddings.items():
                embeddings[k] = v

            # Build community map
            communities: dict[str, set[str]] = {}
            if hasattr(self, "_topology") and self.db._topology is not None:
                for _res, comms in self.db._topology.communities.items():
                    for comm in comms:
                        communities[comm.id] = set(comm.members)

            scorer = EnsembleScorer(
                weighted_adj=weighted_adj,
                adj=adj,
                embeddings=embeddings,
                communities=communities,
                entity_names=entity_name_map,
                entity_types=entity_type_map,
            )
            predictions = scorer.score_candidates(top_k=top_k * 3, seed=42)

            # Entity to community mapping for inter-community detection
            entity_to_comm: dict[str, str] = {}
            for cid, members in communities.items():
                for m in members:
                    entity_to_comm[m] = cid

            for pred in predictions:
                if pred.already_connected if hasattr(pred, "already_connected") else False:
                    continue

                # Compose predicates from evidence paths
                path_dicts = []
                bridge_names = []
                for ep in pred.evidence_paths:
                    path_dicts.append({
                        "ac_predicates": ep.ac_predicates,
                        "cb_predicates": ep.cb_predicates,
                        "path_weight": ep.path_weight,
                    })
                    bridge_names.append(ep.bridge_name or ep.bridge_entity)

                predicted_pred, vote_frac = predict_predicate_from_paths(path_dicts)
                confidence = pred.composite_score

                # Novelty: 1 - shared_neighbors/max_degree, boosted for inter-community
                shared = len(
                    adj.get(pred.entity_a, set())
                    & adj.get(pred.entity_b, set())
                )
                novelty = 1.0 - (shared / max_degree) if max_degree > 0 else 1.0

                is_inter = pred.bridge_type == "inter-community"
                if is_inter:
                    novelty = min(1.0, novelty * 1.2)

                # Penalize hub entities (high-degree = less surprising)
                hub_penalty = 1.0
                for eid in [pred.entity_a, pred.entity_b]:
                    if degree.get(eid, 0) > max_degree * 0.5:
                        hub_penalty *= 0.8

                novelty *= hub_penalty

                name_a = entity_name_map.get(pred.entity_a, pred.entity_a)
                name_b = entity_name_map.get(pred.entity_b, pred.entity_b)
                bridge_str = ", ".join(bridge_names[:3]) if bridge_names else "shared neighbors"

                discoveries.append(Discovery(
                    hypothesis=f"{name_a} may {predicted_pred} {name_b} via {bridge_str}",
                    evidence_summary=(
                        f"{len(pred.evidence_paths)} bridging paths, "
                        f"vote={vote_frac:.0%} for {predicted_pred}"
                    ),
                    discovery_type="bridge_prediction",
                    confidence=confidence,
                    novelty_score=novelty,
                    entities=[pred.entity_a, pred.entity_b],
                    supporting_paths=path_dicts,
                    suggested_action=f"Search for evidence of {name_a} {predicted_pred} {name_b}",
                    predicted_predicate=predicted_pred,
                    entity_types={
                        pred.entity_a: entity_type_map.get(pred.entity_a, "entity"),
                        pred.entity_b: entity_type_map.get(pred.entity_b, "entity"),
                    },
                ))
        except Exception as exc:
            logger.debug("Bridge prediction signal skipped: %s", exc)

        # --- Signal 2: Cross-domain insights ---
        try:
            if hasattr(self, "_topology") and self.db._topology is not None:
                bridges = self.db.cross_domain_bridges(
                    top_k=top_k * 2,
                )
                for bridge in bridges:
                    if len(bridge.communities) < 2:
                        continue
                    # Compare predicates used in each community
                    bridge_claims = self.db.claims_for(bridge.entity_id)[:300]
                    comm_predicates: dict[str, set[str]] = {}
                    for claim in bridge_claims:
                        for cid in bridge.communities:
                            members = communities.get(cid, set())
                            if (
                                claim.object.id in members
                                or claim.subject.id in members
                            ):
                                comm_predicates.setdefault(cid, set()).add(claim.predicate.id)

                    comm_ids = list(comm_predicates.keys())
                    for i in range(len(comm_ids)):
                        for j in range(i + 1, len(comm_ids)):
                            preds_a = comm_predicates.get(comm_ids[i], set())
                            preds_b = comm_predicates.get(comm_ids[j], set())
                            diff_preds = preds_a - preds_b
                            if diff_preds:
                                name = entity_name_map.get(bridge.entity_id, bridge.entity_id)
                                for dp in diff_preds:
                                    discoveries.append(Discovery(
                                        hypothesis=(
                                            f"{name}'s {dp} pattern in "
                                            f"{comm_ids[i]} may extend "
                                            f"to {comm_ids[j]}"
                                        ),
                                        evidence_summary=(
                                            f"Bridge entity spans "
                                            f"{len(bridge.communities)} "
                                            f"communities, bridge_score="
                                            f"{bridge.bridge_score:.2f}"
                                        ),
                                        discovery_type="cross_domain",
                                        confidence=bridge.bridge_score * 0.5,
                                        novelty_score=0.8,
                                        entities=[bridge.entity_id],
                                        suggested_action=(
                                            f"Investigate {dp} "
                                            f"relationships for {name} "
                                            f"in community {comm_ids[j]}"
                                        ),
                                        predicted_predicate=dp,
                                        entity_types={bridge.entity_id: bridge.entity_type},
                                    ))
        except Exception as exc:
            logger.debug("Cross-domain signal skipped: %s", exc)

        # --- Signal 3: Chain completion ---
        try:
            from attestdb.core.vocabulary import compose_predicates

            # Find 2-hop pairs not directly connected
            entity_subset = [e for e in entities if degree.get(e.id, 0) >= 2]
            entity_subset.sort(key=lambda e: degree.get(e.id, 0))
            entity_subset = entity_subset[:50]  # Cap for performance

            direct_pairs: set[tuple[str, str]] = set()
            for (a, b) in self.db.get_weighted_adjacency():
                direct_pairs.add((a, b))
                direct_pairs.add((b, a))

            for entity in entity_subset:
                neighbors = adj.get(entity.id, set())
                for n1 in neighbors:
                    n1_neighbors = adj.get(n1, set())
                    for n2 in n1_neighbors:
                        if n2 == entity.id or n2 == n1:
                            continue
                        pair = (min(entity.id, n2), max(entity.id, n2))
                        if pair in direct_pairs:
                            continue
                        # Get predicates along the path
                        claims_a_n1 = [
                            c for c in self.db.claims_for(entity.id)[:300]
                            if c.object.id == n1 or c.subject.id == n1
                        ]
                        claims_n1_n2 = [
                            c for c in self.db.claims_for(n1)[:300]
                            if c.object.id == n2 or c.subject.id == n2
                        ]
                        if not claims_a_n1 or not claims_n1_n2:
                            continue

                        ac_preds = {c.predicate.id for c in claims_a_n1}
                        cb_preds = {c.predicate.id for c in claims_n1_n2}
                        avg_conf = (
                            sum(c.confidence for c in claims_a_n1) / len(claims_a_n1)
                            + sum(c.confidence for c in claims_n1_n2) / len(claims_n1_n2)
                        ) / 2

                        # Pick the best composed predicate
                        best_composed = "associated_with"
                        for ap in ac_preds:
                            for cp in cb_preds:
                                result = compose_predicates(ap, cp)
                                if result != "associated_with":
                                    best_composed = result
                                    break
                            if best_composed != "associated_with":
                                break

                        name_a = entity_name_map.get(entity.id, entity.id)
                        name_n1 = entity_name_map.get(n1, n1)
                        name_b = entity_name_map.get(n2, n2)

                        novelty = (
                            1.0 - (len(neighbors & n1_neighbors) / max_degree)
                            if max_degree > 0 else 1.0
                        )

                        discoveries.append(Discovery(
                            hypothesis=f"{name_a} may {best_composed} {name_b} via {name_n1}",
                            evidence_summary=(
                                f"2-hop path: {name_a} "
                                f"→[{','.join(ac_preds)}]→ {name_n1} "
                                f"→[{','.join(cb_preds)}]→ {name_b}"
                            ),
                            discovery_type="chain_completion",
                            confidence=avg_conf * 0.7,
                            novelty_score=novelty,
                            entities=[entity.id, n1, n2],
                            supporting_paths=[{
                                "ac_predicates": ac_preds,
                                "cb_predicates": cb_preds,
                                "path_weight": avg_conf,
                            }],
                            suggested_action=(
                                f"Search for direct evidence of "
                                f"{name_a} {best_composed} {name_b}"
                            ),
                            predicted_predicate=best_composed,
                            entity_types={
                                entity.id: entity_type_map.get(entity.id, "entity"),
                                n1: entity_type_map.get(n1, "entity"),
                                n2: entity_type_map.get(n2, "entity"),
                            },
                        ))
                        direct_pairs.add(pair)  # Don't re-discover same pair
        except Exception as exc:
            logger.debug("Chain completion signal skipped: %s", exc)

        # Deduplicate by entity pair
        seen_pairs: set[tuple[str, ...]] = set()
        deduped: list[Discovery] = []
        for d in discoveries:
            key = tuple(sorted(d.entities[:2]))
            if key not in seen_pairs:
                seen_pairs.add(key)
                deduped.append(d)

        # Final ranking: confidence * (0.6 + 0.4 * novelty)
        deduped.sort(key=lambda d: d.confidence * (0.6 + 0.4 * d.novelty_score), reverse=True)
        return deduped[:top_k]

    def analogies(
        self,
        entity_a: str,
        entity_b: str,
        top_k: int = 5,
    ) -> list[Analogy]:
        """Find structural analogies: A:B :: C:D.

        Uses structural embeddings to find entities similar to A and B,
        then predicts the C:D pair and relationship.

        Returns empty list gracefully if no structural embeddings exist.
        """
        from attestdb.core.normalization import normalize_entity_id as _norm

        entity_a = _norm(entity_a)
        entity_b = _norm(entity_b)

        # Get A:B relationship predicates
        ab_claims = [
            c for c in self.db.claims_for(entity_a)[:500]
            if c.object.id == entity_b or c.subject.id == entity_b
        ]
        if not ab_claims:
            return []

        source_predicates = list({c.predicate.id for c in ab_claims})

        # Get structural embeddings
        emb_a = self.db.get_embedding(f"_struct_{entity_a}")
        emb_b = self.db.get_embedding(f"_struct_{entity_b}")
        if emb_a is None or emb_b is None:
            return []

        import numpy as np

        emb_a = np.array(emb_a, dtype=np.float32)
        emb_b = np.array(emb_b, dtype=np.float32)

        # Get all entities with structural embeddings
        entities = self.db.list_entities()
        entity_type_map = {e.id: e.entity_type for e in entities}
        entity_name_map = {e.id: e.name or e.id for e in entities}

        entity_embeddings: dict[str, np.ndarray] = {}
        for e in entities:
            emb = self.db.get_embedding(f"_struct_{e.id}")
            if emb is not None:
                entity_embeddings[e.id] = np.array(emb, dtype=np.float32)

        if len(entity_embeddings) < 4:
            return []

        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            dot = float(np.dot(a, b))
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return dot / (na * nb)

        # Find top 20 entities similar to A, top 20 similar to B
        sims_a = []
        sims_b = []
        for eid, emb in entity_embeddings.items():
            if eid in (entity_a, entity_b):
                continue
            sims_a.append((eid, _cosine(emb_a, emb)))
            sims_b.append((eid, _cosine(emb_b, emb)))

        sims_a.sort(key=lambda x: -x[1])
        sims_b.sort(key=lambda x: -x[1])
        top_c = sims_a[:20]
        top_d = sims_b[:20]

        # Check which pairs are already connected
        adj = self.db.get_adjacency_list()

        # Score (C, D) pairs
        candidates: list[Analogy] = []
        for c_id, sim_ac in top_c:
            for d_id, sim_bd in top_d:
                if c_id == d_id:
                    continue
                # Skip if C and D are already connected
                if d_id in adj.get(c_id, set()):
                    continue

                score = sim_ac * sim_bd

                # Type pattern boost: if type(C)==type(A) and type(D)==type(B)
                type_a = entity_type_map.get(entity_a, "")
                type_b = entity_type_map.get(entity_b, "")
                type_c = entity_type_map.get(c_id, "")
                type_d = entity_type_map.get(d_id, "")
                if (
                    type_a and type_c and type_a == type_c
                    and type_b and type_d and type_b == type_d
                ):
                    score *= 1.5

                name_a = entity_name_map.get(entity_a, entity_a)
                name_b = entity_name_map.get(entity_b, entity_b)
                name_c = entity_name_map.get(c_id, c_id)
                name_d = entity_name_map.get(d_id, d_id)

                candidates.append(Analogy(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    entity_c=c_id,
                    entity_d=d_id,
                    predicted_predicate=(
                        source_predicates[0]
                        if source_predicates
                        else "associated_with"
                    ),
                    score=score,
                    explanation=(
                        f"{name_a}:{name_b} :: {name_c}:{name_d}"
                        f" — sim(A,C)={sim_ac:.2f},"
                        f" sim(B,D)={sim_bd:.2f}"
                    ),
                    source_predicates=source_predicates,
                    entity_types={
                        entity_a: type_a or "entity",
                        entity_b: type_b or "entity",
                        c_id: type_c or "entity",
                        d_id: type_d or "entity",
                    },
                ))

        candidates.sort(key=lambda a: -a.score)
        return candidates[:top_k]

    # --- Crown jewels ---

    def _parse_timestamp(self, ts: str | int | float | None) -> int:
        """Convert ISO string, int, or float to nanosecond timestamp."""
        if ts is None:
            return 0
        if isinstance(ts, (int, float)):
            # int: already nanoseconds. float: assume seconds (time.time() style).
            if isinstance(ts, float):
                return int(ts * 1_000_000_000)
            return ts
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts))
        # Treat naive datetimes as UTC to match Rust store convention
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)

    def diff(self, since: str | int, until: str | int | None = None) -> KnowledgeDiff:
        """Knowledge diff — what changed between two time periods.

        Like git diff for knowledge: shows new beliefs, strengthened or
        weakened claims, and new contradictions.

        Args:
            since: Start of period — ISO string or nanosecond int.
            until: End of period — ISO string, nanosecond int, or None (now).

        Returns:
            KnowledgeDiff with categorized belief changes.
        """
        from attestdb.core.confidence import tier1_confidence, tier2_confidence
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        since_ns = self.db._parse_timestamp(since)
        until_ns = self.db._parse_timestamp(until) if until is not None else time.time_ns()

        # Include retracted claims so we can count tombstones and detect weakened beliefs
        self.db._set_include_retracted(True)
        try:
            all_claims = self.db._all_claims()
        finally:
            self.db._set_include_retracted(False)
        before_claims = [c for c in all_claims if c.timestamp < since_ns]
        period_claims = [c for c in all_claims if since_ns <= c.timestamp < until_ns]

        # Group by content_id
        before_cids: dict[str, list[Claim]] = {}
        for c in before_claims:
            before_cids.setdefault(c.content_id, []).append(c)

        period_cids: dict[str, list[Claim]] = {}
        for c in period_claims:
            period_cids.setdefault(c.content_id, []).append(c)

        new_beliefs: list[BeliefChange] = []
        strengthened: list[BeliefChange] = []
        weakened: list[BeliefChange] = []
        new_contradictions: list[BeliefChange] = []

        # Collect entities and sources
        before_entities = (
            {c.subject.id for c in before_claims}
            | {c.object.id for c in before_claims}
        )
        period_entities = (
            {c.subject.id for c in period_claims}
            | {c.object.id for c in period_claims}
        )
        before_sources = {c.provenance.source_id for c in before_claims}
        period_sources = {c.provenance.source_id for c in period_claims}

        for cid, p_claims in period_cids.items():
            rep = p_claims[0]  # representative claim
            new_src = sorted({c.provenance.source_id for c in p_claims} - before_sources)

            if cid not in before_cids:
                # New belief
                conf_after = tier2_confidence(rep, p_claims)
                new_beliefs.append(BeliefChange(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    change_type="new",
                    claims_before=0,
                    claims_after=len(p_claims),
                    confidence_before=0.0,
                    confidence_after=conf_after,
                    new_sources=new_src,
                ))
            else:
                # Existing — check strengthened (only active claims for confidence)
                b_claims = before_cids[cid]
                active_b = [c for c in b_claims if c.status == ClaimStatus.ACTIVE]
                all_cid_claims = active_b + p_claims
                conf_before = tier2_confidence(b_claims[0], active_b) if active_b else 0.0
                conf_after = tier2_confidence(rep, all_cid_claims)
                if len(all_cid_claims) > len(b_claims):
                    strengthened.append(BeliefChange(
                        content_id=cid,
                        subject=rep.subject.id,
                        predicate=rep.predicate.id,
                        object=rep.object.id,
                        change_type="strengthened",
                        claims_before=len(b_claims),
                        claims_after=len(all_cid_claims),
                        confidence_before=conf_before,
                        confidence_after=conf_after,
                        new_sources=new_src,
                    ))

        # Check for weakened — claims that existed before but have fewer active
        # claims when considering the state at `until` (not current live state).
        # Combine before + period claims for the time-bounded view.
        until_cids: dict[str, list[Claim]] = {}
        for c in before_claims + period_claims:
            until_cids.setdefault(c.content_id, []).append(c)
        for cid, b_claims in before_cids.items():
            active_before = [c for c in b_claims if c.status == ClaimStatus.ACTIVE]
            all_cid_at_until = until_cids.get(cid, [])
            active_now = [c for c in all_cid_at_until if c.status == ClaimStatus.ACTIVE]
            if len(active_now) < len(active_before):
                rep = b_claims[0]
                conf_before = tier2_confidence(rep, active_before) if active_before else 0.0
                conf_after = tier2_confidence(rep, active_now) if active_now else 0.0
                weakened.append(BeliefChange(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    change_type="weakened",
                    claims_before=len(active_before),
                    claims_after=len(active_now),
                    confidence_before=conf_before,
                    confidence_after=conf_after,
                    new_sources=[],
                ))

        # Check for new contradictions: period claims whose (subj, obj) pair
        # has an opposite predicate — but only if the contradiction didn't
        # already exist in the before period.
        before_preds: dict[tuple[str, str], set[str]] = {}
        for c in before_claims:
            key = (c.subject.id, c.object.id)
            before_preds.setdefault(key, set()).add(c.predicate.id)

        all_preds: dict[tuple[str, str], set[str]] = {}
        for c in before_claims + period_claims:
            key = (c.subject.id, c.object.id)
            all_preds.setdefault(key, set()).add(c.predicate.id)

        seen_contradictions: set[str] = set()
        for c in period_claims:
            pred = c.predicate.id
            opp = OPPOSITE_PREDICATES.get(pred)
            if not opp:
                continue
            key = (c.subject.id, c.object.id)
            if opp not in all_preds.get(key, set()):
                continue
            # Skip if this contradiction already existed before the period
            bp = before_preds.get(key, set())
            if pred in bp and opp in bp:
                continue
            contra_key = f"{c.subject.id}|{c.object.id}|{min(pred,opp)}|{max(pred,opp)}"
            if contra_key not in seen_contradictions:
                seen_contradictions.add(contra_key)
                new_contradictions.append(BeliefChange(
                    content_id=c.content_id,
                    subject=c.subject.id,
                    predicate=pred,
                    object=c.object.id,
                    change_type="contradicted",
                    claims_before=0,
                    claims_after=1,
                    confidence_before=0.0,
                    confidence_after=tier1_confidence(c.provenance.source_type),
                    new_sources=[c.provenance.source_id],
                ))

        total_retracted = sum(1 for c in period_claims if c.status == ClaimStatus.TOMBSTONED)
        new_entity_list = sorted(period_entities - before_entities)
        new_source_list = sorted(period_sources - before_sources)

        # Net confidence: average confidence of period claims
        net_conf = 0.0
        if period_claims:
            net_conf = sum(c.confidence for c in period_claims) / len(period_claims)

        # Summary
        parts = []
        if new_beliefs:
            parts.append(f"{len(new_beliefs)} new beliefs")
        if strengthened:
            parts.append(f"{len(strengthened)} strengthened")
        if weakened:
            parts.append(f"{len(weakened)} weakened")
        if new_contradictions:
            parts.append(f"{len(new_contradictions)} new contradictions")
        if new_entity_list:
            parts.append(f"{len(new_entity_list)} new entities")
        if new_source_list:
            parts.append(f"{len(new_source_list)} new sources")
        summary = "; ".join(parts) if parts else "No changes"

        return KnowledgeDiff(
            since=since_ns,
            until=until_ns,
            new_beliefs=new_beliefs,
            strengthened=strengthened,
            weakened=weakened,
            new_contradictions=new_contradictions,
            new_entities=new_entity_list,
            new_sources=new_source_list,
            total_new_claims=len(period_claims),
            total_retracted=total_retracted,
            net_confidence=net_conf,
            summary=summary,
        )

    def resolve_contradictions(
        self, top_k: int = 10, auto_resolve: bool = False, use_llm: bool = False,
        for_entity: str | None = None,
    ) -> ContradictionReport:
        """Find and score contradictions in the knowledge graph.

        Uses OPPOSITE_PREDICATES to detect contradicting claims on the
        same (subject, object) pair, scores evidence quality on each side,
        and optionally resolves clear winners.

        Args:
            top_k: Maximum contradictions to return.
            auto_resolve: If True, ingest meta-claims for resolved contradictions.
            use_llm: If True and ambiguous, use LLM for adjudication.
            for_entity: If set, only find contradictions involving this entity
                (much faster than full-scan for large databases).

        Returns:
            ContradictionReport with scored analyses.
        """
        from attestdb.core.confidence import (
            count_independent_sources,
            recency_factor,
            tier1_confidence,
        )
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        if for_entity:
            all_claims = self.db.claims_for(for_entity)[:500]
        else:
            all_claims = self.db._all_claims()

        # Group by (subject.id, object.id) → {predicate: [claims]}
        pair_preds: dict[tuple[str, str], dict[str, list[Claim]]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            key = (c.subject.id, c.object.id)
            pair_preds.setdefault(key, {}).setdefault(c.predicate.id, []).append(c)

        analyses: list[ContradictionAnalysis] = []

        for (subj, obj), pred_map in pair_preds.items():
            predicates = list(pred_map.keys())
            checked: set[tuple[str, str]] = set()
            for pred_a in predicates:
                opp = OPPOSITE_PREDICATES.get(pred_a)
                if opp and opp in pred_map:
                    pair_key = tuple(sorted([pred_a, opp]))
                    if pair_key in checked:
                        continue
                    checked.add(pair_key)

                    claims_a = pred_map[pred_a]
                    claims_b = pred_map[opp]

                    def _build_side(claims: list[Claim], pred: str) -> ContradictionSide:
                        sources = {c.provenance.source_id for c in claims}
                        source_types = sorted({c.provenance.source_type for c in claims})
                        corr = count_independent_sources(claims)
                        # Tier-weighted confidence: experimental > database > llm
                        tier_scores = [tier1_confidence(c.provenance.source_type) for c in claims]
                        avg_conf = sum(tier_scores) / len(tier_scores)
                        # Bonus for having high-tier sources (experimental, curated)
                        max_tier = max(tier_scores) if tier_scores else 0.0
                        newest = max(c.timestamp for c in claims)
                        rec = recency_factor(newest)
                        weight = (
                            0.30 * min(corr / 5, 1.0)        # corroboration
                            + 0.25 * avg_conf                 # avg source tier
                            + 0.15 * max_tier                 # best source quality
                            + 0.15 * min(len(sources) / 3, 1.0)  # source diversity
                            + 0.15 * rec                      # recency
                        )
                        return ContradictionSide(
                            predicate=pred,
                            claim_count=len(claims),
                            source_count=len(sources),
                            source_types=source_types,
                            avg_confidence=avg_conf,
                            corroboration_count=corr,
                            newest_timestamp=newest,
                            evidence_weight=weight,
                        )

                    side_a = _build_side(claims_a, pred_a)
                    side_b = _build_side(claims_b, opp)

                    margin = abs(side_a.evidence_weight - side_b.evidence_weight)
                    if margin >= 0.15:
                        winner = (
                            "side_a"
                            if side_a.evidence_weight > side_b.evidence_weight
                            else "side_b"
                        )
                        resolution = "a_preferred" if winner == "side_a" else "b_preferred"
                    else:
                        winner = "ambiguous"
                        resolution = "unresolved"

                    explanation = (
                        f"{pred_a} ({side_a.claim_count} claims,"
                        f" weight={side_a.evidence_weight:.2f})"
                        f" vs {opp} ({side_b.claim_count} claims,"
                        f" weight={side_b.evidence_weight:.2f})"
                    )

                    # LLM adjudication for ambiguous cases
                    if winner == "ambiguous" and use_llm:
                        prompt = (
                            f"Two contradicting claims about"
                            f" {subj} and {obj}:\n"
                            f"Side A: '{pred_a}' — "
                            f"{side_a.claim_count} claims from "
                            f"{side_a.source_count} sources\n"
                            f"Side B: '{opp}' — "
                            f"{side_b.claim_count} claims from "
                            f"{side_b.source_count} sources\n"
                            f"Which is more likely correct?"
                            f" Reply with just 'A' or 'B'"
                            f" or 'ambiguous'."
                        )
                        llm_resp = self.db._llm_call(prompt, max_tokens=32)
                        if llm_resp:
                            resp = llm_resp.strip().lower()
                            # Match explicit "a"/"b"/"side a"/"side b" tokens
                            import re
                            a_match = bool(re.search(r'\bside\s*a\b|\ba\b', resp))
                            b_match = bool(re.search(r'\bside\s*b\b|\bb\b', resp))
                            if a_match and not b_match:
                                winner = "side_a"
                                resolution = "a_preferred"
                                explanation += f" — LLM chose {pred_a}"
                            elif b_match and not a_match:
                                winner = "side_b"
                                resolution = "b_preferred"
                                explanation += f" — LLM chose {opp}"

                    analyses.append(ContradictionAnalysis(
                        subject=subj,
                        object=obj,
                        side_a=side_a,
                        side_b=side_b,
                        winner=winner,
                        margin=margin,
                        resolution=resolution,
                        explanation=explanation,
                    ))

        # Sort by margin descending (clearest resolutions first)
        analyses.sort(key=lambda a: -a.margin)
        analyses = analyses[:top_k]

        resolved = sum(1 for a in analyses if a.resolution != "unresolved")
        ambiguous = sum(1 for a in analyses if a.resolution == "unresolved")

        # Auto-resolve: ingest meta-claims for resolved contradictions
        claims_added = 0
        if auto_resolve:
            for a in analyses:
                if a.resolution == "unresolved":
                    continue
                winning_pred = (
                    a.side_a.predicate
                    if a.resolution == "a_preferred"
                    else a.side_b.predicate
                )
                self.db.ingest(
                    subject=(a.subject, "entity"),
                    predicate=("contradiction_resolved", "contradiction_resolved"),
                    object=(a.object, "entity"),
                    provenance={
                        "source_type": "computation",
                        "source_id": "attest:resolve_contradictions",
                    },
                    payload={
                        "schema": "contradiction_resolution",
                        "data": {
                            "winner": winning_pred,
                            "margin": a.margin,
                            "explanation": a.explanation,
                        },
                    },
                )
                claims_added += 1

        return ContradictionReport(
            total_found=len(analyses),
            resolved=resolved,
            ambiguous=ambiguous,
            analyses=analyses,
            claims_added=claims_added,
        )

    def simulate(
        self,
        retract_source: str | None = None,
        add_claim: ClaimInput | None = None,
        remove_entity: str | None = None,
    ) -> SimulationReport:
        """Counterfactual simulation — what-if analysis without modifying the DB.

        Computes cascading effects of hypothetical changes: source retractions,
        new claims, or entity removal.

        Args:
            retract_source: Source ID to simulate retracting.
            add_claim: A ClaimInput to simulate adding.
            remove_entity: Entity ID to simulate removing.

        Returns:
            SimulationReport with impact analysis.
        """
        if retract_source:
            return self.db._simulate_retract_source(retract_source)
        elif add_claim:
            return self.db._simulate_add_claim(add_claim)
        elif remove_entity:
            return self.db._simulate_remove_entity(remove_entity)
        else:
            return SimulationReport(scenario="no-op", summary="No scenario specified")

    def _simulate_retract_source(self, source_id: str) -> SimulationReport:
        """Simulate retracting a source — read-only analysis."""
        from attestdb.core.confidence import tier2_confidence

        # Direct claims from this source
        direct_claims = self.db.claims_by_source_id(source_id)
        direct_ids = {c.claim_id for c in direct_claims}

        if not direct_claims:
            return SimulationReport(
                scenario=f"retract_source:{source_id}",
                summary=f"Source '{source_id}' has no claims — no impact.",
            )

        # Downstream cascade (read-only BFS)
        reverse_index = self.db._build_reverse_provenance_index()
        degraded_ids: set[str] = set()
        queue = deque(direct_ids)
        while queue:
            cid = queue.popleft()
            for dep_id in reverse_index.get(cid, []):
                if dep_id not in direct_ids and dep_id not in degraded_ids:
                    degraded_ids.add(dep_id)
                    queue.append(dep_id)

        # All affected claim IDs
        all_affected_ids = direct_ids | degraded_ids
        all_claims = self.db._all_claims()

        # Entities affected
        entities_affected: set[str] = set()
        for c in direct_claims:
            entities_affected.add(c.subject.id)
            entities_affected.add(c.object.id)

        # Build entity → sources mapping (before and after)
        entity_sources_before: dict[str, set[str]] = {}
        entity_sources_after: dict[str, set[str]] = {}
        entity_claim_count_before: dict[str, int] = {}
        entity_claim_count_after: dict[str, int] = {}

        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            for eid in (c.subject.id, c.object.id):
                entity_sources_before.setdefault(eid, set()).add(c.provenance.source_id)
                entity_claim_count_before[eid] = entity_claim_count_before.get(eid, 0) + 1
                if c.claim_id not in all_affected_ids:
                    entity_sources_after.setdefault(eid, set()).add(c.provenance.source_id)
                    entity_claim_count_after[eid] = entity_claim_count_after.get(eid, 0) + 1

        # Orphaned entities: those that lose ALL claims
        orphaned = sum(
            1 for eid in entities_affected
            if entity_claim_count_before.get(eid, 0) > 0
            and entity_claim_count_after.get(eid, 0) == 0
        )

        # Entities going from multi-source to single-source
        single_source = sum(
            1 for eid in entities_affected
            if len(entity_sources_before.get(eid, set())) > 1
            and len(entity_sources_after.get(eid, set())) <= 1
        )

        # Confidence shifts for affected content_ids
        confidence_shifts: list[ConfidenceShift] = []
        affected_content_ids: dict[str, list[Claim]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            if c.claim_id in all_affected_ids:
                affected_content_ids.setdefault(c.content_id, [])

        for cid in affected_content_ids:
            all_cid_claims = self.db.claims_by_content_id(cid)
            active_claims = [c for c in all_cid_claims if c.status == ClaimStatus.ACTIVE]
            remaining = [c for c in active_claims if c.claim_id not in all_affected_ids]
            if active_claims:
                conf_before = tier2_confidence(active_claims[0], active_claims)
                conf_after = tier2_confidence(remaining[0], remaining) if remaining else 0.0
                rep = active_claims[0]
                confidence_shifts.append(ConfidenceShift(
                    content_id=cid,
                    subject=rep.subject.id,
                    predicate=rep.predicate.id,
                    object=rep.object.id,
                    confidence_before=conf_before,
                    confidence_after=conf_after,
                ))

        # Connection losses: entity pairs losing all edges
        connection_losses: list[ConnectionLoss] = []
        edge_before: dict[tuple[str, str], set[str]] = {}
        edge_after: dict[tuple[str, str], set[str]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pair = (c.subject.id, c.object.id)
            edge_before.setdefault(pair, set()).add(c.predicate.id)
            if c.claim_id not in all_affected_ids:
                edge_after.setdefault(pair, set()).add(c.predicate.id)

        for pair, preds in edge_before.items():
            remaining_preds = edge_after.get(pair, set())
            lost = preds - remaining_preds
            if lost:
                connection_losses.append(ConnectionLoss(
                    entity_a=pair[0],
                    entity_b=pair[1],
                    lost_predicates=sorted(lost),
                ))

        # Risk score
        total_active = sum(1 for c in all_claims if c.status == ClaimStatus.ACTIVE)
        claims_fraction = len(direct_ids) / max(total_active, 1)
        total_pairs = len(edge_before)
        conn_loss_fraction = len(connection_losses) / max(total_pairs, 1)
        degraded_fraction = len(degraded_ids) / max(total_active, 1)
        avg_conf_loss = 0.0
        if confidence_shifts:
            avg_conf_loss = sum(
                cs.confidence_before - cs.confidence_after for cs in confidence_shifts
            ) / len(confidence_shifts)
        risk_score = (
            0.3 * claims_fraction
            + 0.3 * conn_loss_fraction
            + 0.2 * degraded_fraction
            + 0.2 * min(avg_conf_loss, 1.0)
        )
        risk_score = min(risk_score, 1.0)
        if risk_score > 0.5:
            risk_level = "critical"
        elif risk_score > 0.3:
            risk_level = "high"
        elif risk_score > 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        summary = (
            f"Retracting source '{source_id}': {len(direct_ids)} claims removed, "
            f"{len(degraded_ids)} degraded, {orphaned} entities orphaned, "
            f"{len(connection_losses)} connections lost. Risk: {risk_level}."
        )

        return SimulationReport(
            scenario=f"retract_source:{source_id}",
            claims_affected=len(all_affected_ids),
            claims_removed=len(direct_ids),
            claims_degraded=len(degraded_ids),
            entities_affected=sorted(entities_affected),
            entities_now_orphaned=orphaned,
            entities_now_single_source=single_source,
            connection_losses=connection_losses,
            confidence_shifts=confidence_shifts,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
        )

    def _simulate_add_claim(self, claim_input: ClaimInput) -> SimulationReport:
        """Simulate adding a claim — check corroborations, contradictions, gaps."""
        from attestdb.core.hashing import compute_content_id
        from attestdb.core.normalization import normalize_entity_id
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        subj_id = normalize_entity_id(claim_input.subject[0])
        obj_id = normalize_entity_id(claim_input.object[0])
        pred_id = claim_input.predicate[0]

        content_id = compute_content_id(subj_id, pred_id, obj_id)

        # Check existing corroborations
        existing = self.db.claims_by_content_id(content_id)
        new_corroborations = len(existing)

        # Check contradictions
        new_contradictions: list[str] = []
        opp = OPPOSITE_PREDICATES.get(pred_id)
        if opp:
            # Look for opposite predicate on same (subj, obj)
            all_subj_claims = self.db.claims_for(subj_id)[:300]
            for c in all_subj_claims:
                if c.predicate.id == opp and c.object.id == obj_id:
                    new_contradictions.append(
                        f"{subj_id} {pred_id} {obj_id} contradicts"
                        f" existing {subj_id} {opp} {obj_id}"
                    )
                    break

        # Check if this closes a gap (entities exist but unconnected)
        gaps_closed = 0
        subj_entity = self.db.get_entity(subj_id)
        obj_entity = self.db.get_entity(obj_id)
        if subj_entity and obj_entity:
            if not self.db.path_exists(subj_id, obj_id, max_depth=1):
                gaps_closed = 1

        summary = f"Adding {subj_id} {pred_id} {obj_id}: "
        parts = []
        if new_corroborations:
            parts.append(f"corroborates {new_corroborations} existing claims")
        if new_contradictions:
            parts.append(f"{len(new_contradictions)} contradictions")
        if gaps_closed:
            parts.append("closes a knowledge gap")
        summary += "; ".join(parts) if parts else "new isolated claim"

        return SimulationReport(
            scenario=f"add_claim:{subj_id}_{pred_id}_{obj_id}",
            claims_affected=0,
            new_contradictions=new_contradictions,
            new_corroborations=new_corroborations,
            gaps_closed=gaps_closed,
            risk_score=0.0,
            risk_level="low",
            summary=summary,
        )

    def _simulate_remove_entity(self, entity_id: str) -> SimulationReport:
        """Simulate removing an entity — analyze downstream cascade."""
        from attestdb.core.normalization import normalize_entity_id

        eid = normalize_entity_id(entity_id)
        entity_claims = self.db.claims_for(eid)[:500]

        if not entity_claims:
            return SimulationReport(
                scenario=f"remove_entity:{eid}",
                summary=f"Entity '{eid}' has no claims — no impact.",
            )

        # Treat all entity claims as removed
        removed_ids = {c.claim_id for c in entity_claims}

        # Downstream cascade
        reverse_index = self.db._build_reverse_provenance_index()
        degraded_ids: set[str] = set()
        queue = deque(removed_ids)
        while queue:
            cid = queue.popleft()
            for dep_id in reverse_index.get(cid, []):
                if dep_id not in removed_ids and dep_id not in degraded_ids:
                    degraded_ids.add(dep_id)
                    queue.append(dep_id)

        # Entities affected
        entities_affected: set[str] = set()
        for c in entity_claims:
            entities_affected.add(c.subject.id)
            entities_affected.add(c.object.id)

        # Connection losses
        connection_losses: list[ConnectionLoss] = []
        all_affected_ids = removed_ids | degraded_ids
        all_claims = self.db._all_claims()

        edge_before: dict[tuple[str, str], set[str]] = {}
        edge_after: dict[tuple[str, str], set[str]] = {}
        for c in all_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pair = (c.subject.id, c.object.id)
            edge_before.setdefault(pair, set()).add(c.predicate.id)
            if c.claim_id not in all_affected_ids:
                edge_after.setdefault(pair, set()).add(c.predicate.id)

        for pair, preds in edge_before.items():
            remaining_preds = edge_after.get(pair, set())
            lost = preds - remaining_preds
            if lost:
                connection_losses.append(ConnectionLoss(
                    entity_a=pair[0],
                    entity_b=pair[1],
                    lost_predicates=sorted(lost),
                ))

        # Risk score based on entity degree and downstream
        total_active = sum(1 for c in all_claims if c.status == ClaimStatus.ACTIVE)
        claims_fraction = len(removed_ids) / max(total_active, 1)
        total_pairs = len(edge_before)
        conn_loss_fraction = len(connection_losses) / max(total_pairs, 1)
        risk_score = min(
            0.4 * claims_fraction
            + 0.4 * conn_loss_fraction
            + 0.2 * (len(degraded_ids) / max(total_active, 1)),
            1.0,
        )

        if risk_score > 0.5:
            risk_level = "critical"
        elif risk_score > 0.3:
            risk_level = "high"
        elif risk_score > 0.1:
            risk_level = "medium"
        else:
            risk_level = "low"

        summary = (
            f"Removing entity '{eid}': {len(removed_ids)} claims removed, "
            f"{len(degraded_ids)} degraded, {len(connection_losses)} connections lost. "
            f"Risk: {risk_level}."
        )

        return SimulationReport(
            scenario=f"remove_entity:{eid}",
            claims_affected=len(all_affected_ids),
            claims_removed=len(removed_ids),
            claims_degraded=len(degraded_ids),
            entities_affected=sorted(entities_affected),
            connection_losses=connection_losses,
            risk_score=risk_score,
            risk_level=risk_level,
            summary=summary,
        )

    def compile(self, topic: str, max_entities: int = 50, use_llm: bool = False) -> KnowledgeBrief:
        """Compile a structured research brief on a topic.

        Searches for relevant entities, clusters them by graph proximity,
        and generates sections with citations, contradictions, and gaps.

        Args:
            topic: The topic to compile a brief for.
            max_entities: Maximum entities to include.
            use_llm: If True, use LLM for narrative generation.

        Returns:
            KnowledgeBrief with structured sections.
        """
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        # Search for entities matching the topic
        entities: list[EntitySummary] = []
        seen_ids: set[str] = set()

        # Search full topic
        for e in self.db.search_entities(topic, top_k=max_entities):
            if e.id not in seen_ids:
                seen_ids.add(e.id)
                entities.append(e)

        # Also search individual words for broader coverage
        words = topic.split()
        if len(words) > 1:
            for word in words:
                if len(word) < 3:
                    continue
                for e in self.db.search_entities(word, top_k=max_entities):
                    if e.id not in seen_ids and len(entities) < max_entities:
                        seen_ids.add(e.id)
                        entities.append(e)

        if not entities:
            return KnowledgeBrief(
                topic=topic,
                executive_summary=f"No entities found for topic '{topic}'.",
            )

        entity_ids = [e.id for e in entities]
        entity_map = {e.id: e for e in entities}

        # Cluster entities by graph proximity
        clusters = self.db._cluster_entities(entity_ids)

        sections: list[BriefSection] = []
        all_citations: list[Citation] = []
        all_sources: set[str] = set()
        all_contradictions: list[str] = []
        all_gaps: list[str] = []
        total_claims_cited = 0

        for cluster in clusters:
            # Label the cluster
            title = self.db._label_cluster(cluster, entity_map)

            # Gather all claims for entities in this cluster
            cluster_claims: list[Claim] = []
            claim_ids_seen: set[str] = set()
            for eid in cluster:
                for c in self.db.claims_for(eid)[:200]:
                    if c.claim_id not in claim_ids_seen and c.status == ClaimStatus.ACTIVE:
                        claim_ids_seen.add(c.claim_id)
                        cluster_claims.append(c)

            # Build citations
            citations: list[Citation] = []
            content_id_counts: dict[str, int] = {}
            for c in cluster_claims:
                content_id_counts[c.content_id] = content_id_counts.get(c.content_id, 0) + 1

            # Dedupe citations by content_id, keep highest confidence
            content_id_best: dict[str, Claim] = {}
            for c in cluster_claims:
                if (
                    c.content_id not in content_id_best
                    or c.confidence > content_id_best[c.content_id].confidence
                ):
                    content_id_best[c.content_id] = c

            for cid, c in content_id_best.items():
                all_sources.add(c.provenance.source_id)
                citations.append(Citation(
                    claim_id=c.claim_id,
                    subject=c.subject.id,
                    predicate=c.predicate.id,
                    object=c.object.id,
                    confidence=c.confidence,
                    source_id=c.provenance.source_id,
                    source_type=c.provenance.source_type,
                    corroboration_count=content_id_counts.get(cid, 0),
                ))

            citations.sort(key=lambda ct: -ct.confidence)
            total_claims_cited += len(citations)
            all_citations.extend(citations)

            # Key findings: top 5 by confidence
            key_findings: list[str] = []
            for ct in citations[:5]:
                corr_str = (
                    f", {ct.corroboration_count} sources"
                    if ct.corroboration_count > 1 else ""
                )
                key_findings.append(
                    f"{ct.subject} {ct.predicate} {ct.object} "
                    f"(confidence: {ct.confidence:.2f}{corr_str})"
                )

            # Detect contradictions within section
            section_contradictions: list[str] = []
            seen_pairs: set[tuple[str, str, str]] = set()
            subj_obj_preds: dict[tuple[str, str], set[str]] = {}
            for c in cluster_claims:
                key = (c.subject.id, c.object.id)
                subj_obj_preds.setdefault(key, set()).add(c.predicate.id)
            for (subj, obj), preds in subj_obj_preds.items():
                for p in preds:
                    opp = OPPOSITE_PREDICATES.get(p)
                    if opp and opp in preds:
                        dedup = tuple(sorted([p, opp]))
                        pair_key = (subj, dedup[0], dedup[1])
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            section_contradictions.append(f"{subj}: {p} vs {opp} {obj}")
            all_contradictions.extend(section_contradictions)

            # Detect gaps: single-source entities in cluster
            section_gaps: list[str] = []
            for eid in cluster:
                eid_claims = [
                    c for c in cluster_claims
                    if c.subject.id == eid or c.object.id == eid
                ]
                sources = {c.provenance.source_id for c in eid_claims}
                if len(sources) == 1 and len(eid_claims) > 0:
                    section_gaps.append(
                        f"{eid}: single source"
                        f" ({next(iter(sources))})"
                    )
            all_gaps.extend(section_gaps)

            # Section confidence
            avg_conf = (
                (sum(c.confidence for c in cluster_claims)
                 / len(cluster_claims))
                if cluster_claims else 0.0
            )

            sections.append(BriefSection(
                title=title,
                entities=cluster,
                key_findings=key_findings,
                citations=citations,
                contradictions=section_contradictions,
                gaps=section_gaps,
                avg_confidence=avg_conf,
            ))

        # Sort sections by citation count descending
        sections.sort(key=lambda s: -len(s.citations))

        # Overall metrics
        avg_confidence = (
            (sum(s.avg_confidence for s in sections) / len(sections))
            if sections else 0.0
        )

        # Strongest findings: top 3 by confidence * corroboration
        all_citations.sort(key=lambda ct: -(ct.confidence * ct.corroboration_count))
        strongest = [
            f"{ct.subject} {ct.predicate} {ct.object}"
            f" (conf={ct.confidence:.2f},"
            f" {ct.corroboration_count} sources)"
            for ct in all_citations[:3]
        ]

        # Weakest areas: sections with lowest avg_confidence or most gaps
        weak_sections = sorted(sections, key=lambda s: s.avg_confidence)
        weakest = [
            f"{s.title} (avg confidence:"
            f" {s.avg_confidence:.2f}, {len(s.gaps)} gaps)"
            for s in weak_sections[:3]
        ]

        # Executive summary
        if use_llm:
            findings_text = "\n".join(strongest[:5])
            prompt = (
                f"Write a 2-3 sentence executive summary about '{topic}' based on:\n"
                f"- {len(entities)} entities analyzed across {len(sections)} topic clusters\n"
                f"- Key findings:\n{findings_text}\n"
                f"- {len(all_contradictions)} contradictions, {len(all_gaps)} knowledge gaps\n"
                f"Be concise and factual."
            )
            exec_summary = self.db._llm_call(prompt, max_tokens=256)
            if not exec_summary:
                exec_summary = self.db._template_summary(
                    topic, entities, sections,
                    all_contradictions, all_gaps, strongest,
                )
        else:
            exec_summary = self.db._template_summary(
                topic, entities, sections,
                all_contradictions, all_gaps, strongest,
            )

        return KnowledgeBrief(
            topic=topic,
            sections=sections,
            executive_summary=exec_summary,
            total_entities=len(entities),
            total_claims_cited=total_claims_cited,
            total_sources=len(all_sources),
            avg_confidence=avg_confidence,
            total_contradictions=len(all_contradictions),
            total_gaps=len(all_gaps),
            strongest_findings=strongest,
            weakest_areas=weakest,
        )

    def _template_summary(
        self, topic: str, entities: list, sections: list,
        contradictions: list, gaps: list, strongest: list,
    ) -> str:
        """Generate a template-based executive summary."""
        parts = [f"Brief on '{topic}': {len(entities)} entities across {len(sections)} clusters."]
        if strongest:
            parts.append(f"Top finding: {strongest[0]}.")
        if contradictions:
            parts.append(f"{len(contradictions)} contradictions detected.")
        if gaps:
            parts.append(f"{len(gaps)} knowledge gaps identified.")
        return " ".join(parts)

    def explain_why(
        self, entity_a: str, entity_b: str, max_depth: int = 4, use_llm: bool = False,
    ) -> Explanation:
        """Explain how two entities are connected with full provenance.

        Traces the best reasoning chain between two entities and produces
        a human-readable narrative with source citations at every hop.

        Args:
            entity_a: Start entity.
            entity_b: End entity.
            max_depth: Maximum path length to search.
            use_llm: If True, use LLM to generate narrative.

        Returns:
            Explanation with steps, confidence, and narrative.
        """
        a_norm = normalize_entity_id(entity_a)
        b_norm = normalize_entity_id(entity_b)

        chains = self.db.trace(a_norm, b_norm, max_depth=max_depth, top_k=5)

        if not chains:
            return Explanation(
                entity_a=a_norm,
                entity_b=b_norm,
                connected=False,
                narrative=(
                    f"No connection found between '{a_norm}'"
                    f" and '{b_norm}' within {max_depth} hops."
                ),
            )

        best = chains[0]
        steps: list[ExplanationStep] = []
        all_sources: set[str] = set()

        for hop in best.hops:
            source_ids = hop.source_ids or []
            all_sources.update(source_ids)

            source_summary = ""
            if source_ids:
                source_summary = f"{len(source_ids)} source(s): {', '.join(source_ids[:3])}"
                if len(source_ids) > 3:
                    source_summary += f" (+{len(source_ids) - 3} more)"

            steps.append(ExplanationStep(
                from_entity=hop.from_entity,
                to_entity=hop.to_entity,
                predicate=hop.predicate,
                confidence=hop.confidence,
                source_summary=source_summary,
                evidence_text=hop.evidence_text,
                has_contradiction=hop.has_contradiction,
            ))

        # Build template narrative
        narrative_parts = [
            f"Connection: {a_norm} → {b_norm}"
            f" ({len(steps)} hops,"
            f" confidence={best.chain_confidence:.2f})"
        ]
        for i, step in enumerate(steps, 1):
            line = (
                f"  {i}. {step.from_entity} —[{step.predicate}]→"
                f" {step.to_entity} (conf={step.confidence:.2f})"
            )
            if step.source_summary:
                line += f" [{step.source_summary}]"
            if step.has_contradiction:
                line += " ⚠ contradicted"
            if step.evidence_text:
                line += f'\n     Evidence: "{step.evidence_text[:100]}"'
            narrative_parts.append(line)

        template_narrative = "\n".join(narrative_parts)

        if use_llm:
            prompt = (
                f"Explain in 2-3 sentences how {a_norm} connects to {b_norm}:\n"
                f"{template_narrative}\n"
                f"Be concise and cite sources."
            )
            llm_narrative = self.db._llm_call(prompt, max_tokens=256)
            narrative = llm_narrative if llm_narrative else template_narrative
        else:
            narrative = template_narrative

        return Explanation(
            entity_a=a_norm,
            entity_b=b_norm,
            connected=True,
            steps=steps,
            chain_confidence=best.chain_confidence,
            narrative=narrative,
            alternative_paths=len(chains) - 1,
            source_count=len(all_sources),
        )

    def forecast(self, entity_id: str, top_k: int = 10) -> Forecast:
        """Predict next connections for an entity.

        Uses graph structure (2-hop neighbors not yet directly connected)
        and historical growth patterns to predict likely future connections.

        Args:
            entity_id: Entity to forecast for.
            top_k: Maximum predictions.

        Returns:
            Forecast with predicted connections and growth trajectory.
        """
        eid = normalize_entity_id(entity_id)
        entity_claims = self.db.claims_for(eid)[:500]

        if not entity_claims:
            return Forecast(entity_id=eid)

        adj = self.db.get_adjacency_list()
        direct_neighbors = adj.get(eid, set())

        # Entity type and name maps
        entity_map: dict[str, EntitySummary] = {}
        for e in self.db.list_entities():
            entity_map[e.id] = e

        # 2-hop candidates: neighbors of neighbors not directly connected
        candidates: dict[str, dict] = {}
        for neighbor in direct_neighbors:
            for hop2 in adj.get(neighbor, set()):
                if hop2 != eid and hop2 not in direct_neighbors and hop2 not in candidates:
                    candidates[hop2] = {"bridges": [], "predicates": []}
                if hop2 != eid and hop2 not in direct_neighbors:
                    candidates.setdefault(hop2, {"bridges": [], "predicates": []})
                    candidates[hop2]["bridges"].append(neighbor)
                    # Collect predicates from bridge
                    for c in self.db.claims_for(neighbor)[:200]:
                        if c.object.id == hop2 or c.subject.id == hop2:
                            candidates[hop2]["predicates"].append(c.predicate.id)

        # Score candidates
        predictions: list[ForecastConnection] = []
        for cand_id, info in candidates.items():
            bridges = info["bridges"]
            predicates = info["predicates"]

            # Score: more bridges = more likely, weighted by bridge degree
            bridge_count = len(set(bridges))
            bridge_score = min(bridge_count / 3, 1.0)

            # Degree similarity (similar degree entities are more likely to connect)
            cand_degree = len(adj.get(cand_id, set()))
            my_degree = len(direct_neighbors)
            degree_sim = 1.0 - abs(cand_degree - my_degree) / max(cand_degree + my_degree, 1)

            score = 0.6 * bridge_score + 0.4 * degree_sim

            # Predicted predicate: most common among bridge predicates
            pred = "associated_with"
            if predicates:
                from collections import Counter
                pred_counts = Counter(predicates)
                pred = pred_counts.most_common(1)[0][0]

            cand_entity = entity_map.get(cand_id)
            cand_type = cand_entity.entity_type if cand_entity else "entity"

            reason = f"{bridge_count} bridging entities ({', '.join(sorted(set(bridges))[:3])})"

            predictions.append(ForecastConnection(
                target_entity=cand_id,
                target_type=cand_type,
                predicted_predicate=pred,
                score=score,
                reason=reason,
                evidence_entities=sorted(set(bridges))[:5],
            ))

        predictions.sort(key=lambda p: -p.score)
        predictions = predictions[:top_k]

        # Growth trajectory from timestamps
        timestamps = sorted(c.timestamp for c in entity_claims if c.timestamp > 0)
        growth_rate = 0.0
        trajectory = "stable"
        if len(timestamps) >= 2:
            span_days = (timestamps[-1] - timestamps[0]) / 1_000_000_000 / 86400
            if span_days > 0:
                growth_rate = len(timestamps) / max(span_days / 30, 0.1)  # per month
                # Compare first half vs second half
                mid = len(timestamps) // 2
                first_half_span = (timestamps[mid] - timestamps[0]) / 1_000_000_000 / 86400
                second_half_span = (timestamps[-1] - timestamps[mid]) / 1_000_000_000 / 86400
                first_rate = mid / max(first_half_span / 30, 0.1)
                second_rate = (len(timestamps) - mid) / max(second_half_span / 30, 0.1)
                if second_rate > first_rate * 1.3:
                    trajectory = "growing"
                elif second_rate < first_rate * 0.7:
                    trajectory = "declining"

        return Forecast(
            entity_id=eid,
            predictions=predictions,
            growth_rate=growth_rate,
            trajectory=trajectory,
            total_current_connections=len(direct_neighbors),
        )

    def merge_report(self, other: "AttestDB") -> MergeReport:
        """Compare two knowledge bases — what each knows that the other doesn't.

        Args:
            other: Another AttestDB instance to compare against.

        Returns:
            MergeReport with unique/shared beliefs, entity diffs, and conflicts.
        """
        from attestdb.core.confidence import tier2_confidence
        from attestdb.core.vocabulary import OPPOSITE_PREDICATES

        self_claims = self.db._all_claims()
        other_claims = other._all_claims()

        # Group by content_id
        self_by_cid: dict[str, list[Claim]] = {}
        for c in self_claims:
            if c.status == ClaimStatus.ACTIVE:
                self_by_cid.setdefault(c.content_id, []).append(c)

        other_by_cid: dict[str, list[Claim]] = {}
        for c in other_claims:
            if c.status == ClaimStatus.ACTIVE:
                other_by_cid.setdefault(c.content_id, []).append(c)

        self_cids = set(self_by_cid.keys())
        other_cids = set(other_by_cid.keys())

        shared = self_cids & other_cids
        self_unique = self_cids - other_cids
        other_unique = other_cids - self_cids

        # Entity comparison
        self_entities = {c.subject.id for c in self_claims} | {c.object.id for c in self_claims}
        other_entities = {c.subject.id for c in other_claims} | {c.object.id for c in other_claims}

        # Find conflicts: opposing predicates on same (subj, obj) across DBs
        conflicts: list[MergeConflict] = []

        # Build self's (subj, obj, pred) index
        self_triples: dict[tuple[str, str], dict[str, list[Claim]]] = {}
        for c in self_claims:
            if c.status == ClaimStatus.ACTIVE:
                key = (c.subject.id, c.object.id)
                self_triples.setdefault(key, {}).setdefault(c.predicate.id, []).append(c)

        for c in other_claims:
            if c.status != ClaimStatus.ACTIVE:
                continue
            pred = c.predicate.id
            opp = OPPOSITE_PREDICATES.get(pred)
            if not opp:
                continue
            key = (c.subject.id, c.object.id)
            if key in self_triples and opp in self_triples[key]:
                # Conflict: other says pred, self says opposite
                self_opp_claims = self_triples[key][opp]
                other_pred_claims = [
                    oc for oc in other_claims
                    if oc.status == ClaimStatus.ACTIVE
                    and oc.subject.id == c.subject.id
                    and oc.object.id == c.object.id
                    and oc.predicate.id == pred
                ]
                if other_pred_claims:
                    # Avoid duplicate conflicts
                    if not any(
                        mc.subject == c.subject.id and mc.object == c.object.id
                        for mc in conflicts
                    ):
                        self_conf = tier2_confidence(self_opp_claims[0], self_opp_claims)
                        other_conf = tier2_confidence(other_pred_claims[0], other_pred_claims)
                        conflicts.append(MergeConflict(
                            content_id=c.content_id,
                            subject=c.subject.id,
                            predicate=f"{opp} (self) vs {pred} (other)",
                            object=c.object.id,
                            self_confidence=self_conf,
                            other_confidence=other_conf,
                            self_sources=len({sc.provenance.source_id for sc in self_opp_claims}),
                            other_sources=len(
                                {oc.provenance.source_id for oc in other_pred_claims}
                            ),
                        ))

        # Summary
        parts = []
        parts.append(f"Self: {len(self_unique)} unique beliefs, {len(self_entities)} entities")
        parts.append(f"Other: {len(other_unique)} unique beliefs, {len(other_entities)} entities")
        shared_ents = len(self_entities & other_entities)
        parts.append(
            f"Shared: {len(shared)} beliefs,"
            f" {shared_ents} entities"
        )
        if conflicts:
            parts.append(f"{len(conflicts)} conflicts")
        summary = "; ".join(parts)

        return MergeReport(
            self_unique_beliefs=len(self_unique),
            other_unique_beliefs=len(other_unique),
            shared_beliefs=len(shared),
            conflicts=conflicts,
            self_unique_entities=sorted(self_entities - other_entities),
            other_unique_entities=sorted(other_entities - self_entities),
            shared_entities=sorted(self_entities & other_entities),
            self_total_claims=len(self_claims),
            other_total_claims=len(other_claims),
            summary=summary,
        )


    def _gather_consensus_evidence(self, question: str, max_claims: int = 50) -> str:
        """Pull relevant claims from the DB to ground consensus in real evidence.

        Uses text_search to find claims related to the question, then formats
        them with source, confidence, and corroboration info so LLM providers
        can evaluate actual evidence instead of answering from training data.

        Returns empty string if no relevant claims found.
        """
        try:
            claims = self.db.text_search(question, top_k=max_claims)
        except Exception:
            return ""

        if not claims:
            return ""

        # Group by content_id for corroboration counts
        content_groups: dict[str, list] = {}
        for c in claims:
            content_groups.setdefault(c.content_id, []).append(c)

        lines = []
        seen_content = set()
        for claim in claims:
            if claim.content_id in seen_content:
                continue
            seen_content.add(claim.content_id)

            corroboration = len(content_groups.get(claim.content_id, []))
            sources = sorted({
                c.provenance.source_id if c.provenance else "unknown"
                for c in content_groups.get(claim.content_id, [])
            })

            subj = claim.subject.display_name or claim.subject.id
            pred = claim.predicate.id
            obj = claim.object.display_name or claim.object.id
            line = (
                f"- {subj} {pred} {obj} "
                f"[confidence: {claim.confidence:.2f}, "
                f"sources: {', '.join(sources)}"
            )
            if corroboration > 1:
                line += f", corroborated by {corroboration} independent sources"
            line += "]"
            lines.append(line)

            if len(lines) >= 30:  # cap evidence to keep prompts reasonable
                break

        if not lines:
            return ""

        return (
            f"Found {len(lines)} relevant claims"
            f" ({len(claims)} total including corroborations):\n"
            + "\n".join(lines)
        )

    def agent_consensus(
        self,
        question: str,
        context: str = "",
        max_rounds: int = 3,
        convergence_threshold: float = 0.8,
        providers: list[str] | None = None,
        model_overrides: dict[str, str] | None = None,
        ingest: bool = True,
    ) -> "AgentConsensusResult":
        """Query multiple LLM providers, cross-pollinate, converge, and optionally
        ingest the consensus as claims.

        Each provider's response is stored as a claim with source_id=provider_name.
        The final consensus is stored with source_type="agent_consensus".

        Args:
            question: The question to get consensus on.
            context: Optional document/chat context.
            max_rounds: Maximum cross-pollination rounds.
            convergence_threshold: Agreement score (0-1) to declare convergence.
            providers: Filter to specific providers.
            model_overrides: Override default models, e.g. {"openai": "gpt-4o"}.
            ingest: If True, store consensus and individual responses as claims.

        Returns:
            AgentConsensusResult with synthesized answer and provenance.
        """
        from attestdb.core.consensus import ConsensusEngine

        # Pull evidence from the DB to ground the consensus in real data
        if not context:
            context = self.db._gather_consensus_evidence(question)

        engine = ConsensusEngine(
            providers=providers,
            model_overrides=model_overrides,
            env_path=os.path.join(os.path.dirname(self.db._db_path or ""), ".env")
            if self.db._db_path and self.db._db_path != ":memory:" else None,
        )
        result = engine.run(
            question=question,
            context=context,
            max_rounds=max_rounds,
            convergence_threshold=convergence_threshold,
            providers=providers,
        )

        from attestdb.core.types import (
            AgentConsensusResult,
            JudgeVote as JVType,
            ProviderResponse as PRType,
        )

        # Convert to shared types
        typed_responses = [
            PRType(
                provider=r.provider,
                model=r.model,
                response=r.response,
                tokens_in=r.tokens_in,
                tokens_out=r.tokens_out,
                latency_ms=r.latency_ms,
                error=r.error,
                round_number=r.round_number,
            )
            for r in result.responses
        ]

        typed_votes = [
            JVType(
                provider=v.provider,
                converged=v.converged,
                best_provider=v.best_provider,
                rating=v.rating,
                critique=v.critique,
            )
            for v in result.votes
        ]

        consensus_result = AgentConsensusResult(
            question=result.question,
            consensus=result.consensus,
            confidence=result.confidence,
            rounds=result.rounds,
            providers_used=result.providers_used,
            responses=typed_responses,
            dissents=result.dissents,
            votes=typed_votes,
            total_tokens=result.total_tokens,
            total_cost=result.total_cost,
            converged=result.converged,
        )

        if ingest and result.consensus:
            # Ingest individual provider responses
            for r in result.responses:
                if r.error or not r.response:
                    continue
                try:
                    self.db.ingest(ClaimInput(
                        subject=("question:" + question[:100], "question"),
                        predicate=("has_response", "agent_response"),
                        object=(r.provider + ":" + r.model, "llm_provider"),
                        provenance={
                            "source_id": r.provider,
                            "source_type": "llm_api",
                            "method": "agent_consensus",
                            "model_version": r.model,
                        },
                        confidence=result.confidence,
                        payload={
                            "schema_ref": "agent_consensus/response/v1",
                            "data": {
                                "response": r.response[:4000],
                                "round": r.round_number,
                                "tokens_in": r.tokens_in,
                                "tokens_out": r.tokens_out,
                                "latency_ms": r.latency_ms,
                            },
                        },
                    ))
                except Exception as exc:
                    logger.warning("Failed to ingest response from %s: %s", r.provider, exc)

            # Ingest consensus
            try:
                self.db.ingest(ClaimInput(
                    subject=("question:" + question[:100], "question"),
                    predicate=("has_consensus", "agent_consensus"),
                    object=("consensus:" + question[:80], "consensus_answer"),
                    provenance={
                        "source_id": "agent_consensus",
                        "source_type": "agent_consensus",
                        "method": f"multi_llm_{len(result.providers_used)}",
                    },
                    confidence=result.confidence,
                    payload={
                        "schema_ref": "agent_consensus/result/v1",
                        "data": {
                            "consensus": result.consensus[:4000],
                            "rounds": result.rounds,
                            "converged": result.converged,
                            "providers": result.providers_used,
                            "total_tokens": result.total_tokens,
                        },
                    },
                ))
            except Exception as exc:
                logger.warning("Failed to ingest consensus: %s", exc)

        return consensus_result

