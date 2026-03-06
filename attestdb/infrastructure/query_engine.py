"""Query engine implementing the 8-step ContextFrame assembly algorithm."""

from __future__ import annotations

import time

from attestdb.core.confidence import count_independent_sources, tier2_confidence
from attestdb.core.errors import EntityNotFoundError
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import (
    Claim,
    ClaimStatus,
    ContextFrame,
    Contradiction,
    EntitySummary,
    PathResult,
    PathStep,
    QueryProfile,
    QuantitativeClaim,
    Relationship,
    claim_from_dict,
    entity_summary_from_dict,
)
from attestdb.core.vocabulary import CONTRADICTION_PREDICATES, OPPOSITE_PREDICATES, QUANTITATIVE_SCHEMAS, SYMMETRIC_PREDICATES

# Recency bonus: 7-day half-life
RECENCY_HALF_LIFE_SECONDS = 7 * 24 * 3600
# Token estimate: ~4 chars per token
CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _recency_bonus(claim_timestamp: int, now_ns: int) -> float:
    """Exponential decay with 7-day half-life."""
    age_seconds = (now_ns - claim_timestamp) / 1_000_000_000
    if age_seconds <= 0:
        return 1.0
    return 0.5 ** (age_seconds / RECENCY_HALF_LIFE_SECONDS)


def _source_diversity_bonus(claim: Claim, seen_source_types: set[str]) -> float:
    """1.1x bonus if this claim adds a new source_type to the result set."""
    if claim.provenance.source_type not in seen_source_types:
        return 1.1
    return 1.0


class QueryEngine:
    """Assembles ContextFrames from the store using the 8-step algorithm."""

    def __init__(self, store, claim_converter=None):
        self._store = store
        self._convert_claim = claim_converter or claim_from_dict

    def query(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = 500,
        max_tokens: int = 4000,
        confidence_threshold: float = 0.0,
        predicate_types: list[str] | None = None,
    ) -> ContextFrame:
        return self._execute_query(
            focal_entity, depth=depth, min_confidence=min_confidence,
            exclude_source_types=exclude_source_types,
            max_claims=max_claims, max_tokens=max_tokens,
            confidence_threshold=confidence_threshold,
            predicate_types=predicate_types,
        )

    def _execute_query(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = 500,
        max_tokens: int = 4000,
        confidence_threshold: float = 0.0,
        predicate_types: list[str] | None = None,
        claim_filter: "Callable[[Claim], bool] | None" = None,
        include_quantitative: bool = True,
        include_contradictions: bool = True,
        include_narrative: bool = True,
    ) -> ContextFrame:
        exclude = set(exclude_source_types or [])
        # Effective confidence floor: max of min_confidence and confidence_threshold
        effective_min_conf = max(min_confidence, confidence_threshold)
        allowed_predicates = set(predicate_types) if predicate_types else None
        now_ns = time.time_ns()

        # STEP 1: Resolve focal entity
        canonical = self._store.resolve(normalize_entity_id(focal_entity))
        raw_entity = self._store.get_entity(canonical)
        if raw_entity is None:
            raise EntityNotFoundError(f"Entity not found: {focal_entity} (canonical: {canonical})")
        entity_summary = entity_summary_from_dict(raw_entity)

        # STEP 2: Collect candidate claims via BFS
        # Each candidate tracks: (claim, hop, from_entity_id)
        candidates: list[tuple[Claim, int, str]] = []
        visited_entities: set[str] = {canonical}
        # Include aliases
        aliases = self._store.get_alias_group(canonical)
        visited_entities.update(aliases)
        frontier: set[str] = set(aliases)
        seen_claim_ids: set[str] = set()

        for hop in range(1, depth + 1):
            if not frontier:
                break
            next_frontier: set[str] = set()
            # Sort frontier by max-confidence edge when threshold is active
            frontier_list = list(frontier)
            for eid in frontier_list:
                for claim in [self._convert_claim(d) for d in self._store.claims_for(eid)]:
                    if claim.claim_id in seen_claim_ids:
                        continue
                    if claim_filter is not None and not claim_filter(claim):
                        continue
                    if claim.confidence < effective_min_conf:
                        continue
                    if claim.provenance.source_type in exclude:
                        continue
                    if claim.status != ClaimStatus.ACTIVE:
                        continue
                    if allowed_predicates and claim.predicate.id not in allowed_predicates:
                        continue
                    seen_claim_ids.add(claim.claim_id)
                    candidates.append((claim, hop, eid))
                    # Determine "other" entity — the one that isn't in visited
                    if claim.subject.id == eid or claim.subject.id in visited_entities:
                        other = claim.object.id
                    else:
                        other = claim.subject.id
                    if other not in visited_entities:
                        next_frontier.add(other)
                        visited_entities.add(other)
            frontier = next_frontier

        # STEP 3: Score and rank (using Tier 2 confidence with corroboration)
        seen_source_types: set[str] = set()
        scored: list[tuple[Claim, int, float, str]] = []  # (claim, hop, score, from_eid)
        # Cache content_id -> corroborating claims to avoid repeated lookups
        _content_cache: dict[str, list[Claim]] = {}
        for claim, hop, from_eid in candidates:
            # Tier 2: corroboration-boosted confidence
            if claim.content_id not in _content_cache:
                _content_cache[claim.content_id] = [
                    self._convert_claim(d) for d in self._store.claims_by_content_id(claim.content_id)
                ]
            corroborating = _content_cache[claim.content_id]
            conf = tier2_confidence(claim, corroborating)

            score = (
                conf
                * (1.0 / hop)
                * _source_diversity_bonus(claim, seen_source_types)
                * _recency_bonus(claim.timestamp, now_ns)
            )
            seen_source_types.add(claim.provenance.source_type)
            scored.append((claim, hop, score, from_eid))

        scored.sort(key=lambda x: x[2], reverse=True)
        scored = scored[:max_claims]

        # STEP 4: Group by relationship
        relationships: dict[tuple[str, str, str], Relationship] = {}
        for claim, hop, score, from_eid in scored:
            key = (claim.subject.id, claim.predicate.id, claim.object.id)
            # Determine which entity is the "other" (not the one we traversed from)
            if claim.subject.id == from_eid or claim.subject.id in aliases:
                other_summary = EntitySummary(
                    id=claim.object.id,
                    name=claim.object.display_name,
                    entity_type=claim.object.entity_type,
                    external_ids=claim.object.external_ids,
                )
            else:
                other_summary = EntitySummary(
                    id=claim.subject.id,
                    name=claim.subject.display_name,
                    entity_type=claim.subject.entity_type,
                    external_ids=claim.subject.external_ids,
                )

            if key not in relationships:
                # Count independent sources via content_id grouping
                corroborating = _content_cache.get(claim.content_id, [claim])
                n_indep = count_independent_sources(corroborating)
                relationships[key] = Relationship(
                    predicate=claim.predicate.id,
                    target=other_summary,
                    confidence=tier2_confidence(claim, corroborating),
                    n_independent_sources=n_indep,
                    source_types=[claim.provenance.source_type],
                    latest_claim_timestamp=claim.timestamp,
                    payload=claim.payload.data if claim.payload else None,
                    is_symmetric=claim.predicate.id in SYMMETRIC_PREDICATES,
                )
            else:
                rel = relationships[key]
                corroborating = _content_cache.get(claim.content_id, [claim])
                new_conf = tier2_confidence(claim, corroborating)
                rel.confidence = max(rel.confidence, new_conf)
                if claim.provenance.source_type not in rel.source_types:
                    rel.source_types.append(claim.provenance.source_type)
                rel.n_independent_sources = max(
                    rel.n_independent_sources,
                    count_independent_sources(corroborating),
                )
                rel.latest_claim_timestamp = max(
                    rel.latest_claim_timestamp, claim.timestamp
                )
                if claim.payload and not rel.payload:
                    rel.payload = claim.payload.data

        # STEP 5: Extract quantitative data
        quantitative: list[QuantitativeClaim] = []
        if include_quantitative:
            for claim, hop, score, from_eid in scored:
                if claim.payload and claim.payload.schema_ref in QUANTITATIVE_SCHEMAS:
                    data = claim.payload.data
                    if "value" in data and "unit" in data:
                        other_id = (
                            claim.object.id
                            if claim.subject.id == from_eid or claim.subject.id in aliases
                            else claim.subject.id
                        )
                        quantitative.append(
                            QuantitativeClaim(
                                predicate=claim.predicate.id,
                                target=other_id,
                                value=data["value"],
                                unit=data["unit"],
                                metric=data.get("metric", ""),
                                source_type=claim.provenance.source_type,
                                confidence=claim.confidence,
                            )
                        )

        # STEP 6: Find contradictions (implicit via opposing predicates)
        contradictions: list[Contradiction] = []
        if include_contradictions:
            # Detect implicit contradictions (same subject-object with opposing predicates)
            subj_obj_predicates: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for claim, *_ in scored:
                so_key = (claim.subject.id, claim.object.id)
                if so_key not in subj_obj_predicates:
                    subj_obj_predicates[so_key] = []
                subj_obj_predicates[so_key].append((claim.predicate.id, claim.claim_id))

            for (subj, obj), preds in subj_obj_predicates.items():
                pred_set = {p for p, _ in preds}
                for p1, p2 in OPPOSITE_PREDICATES.items():
                    if p1 in pred_set and p2 in pred_set:
                        cid_a = next(cid for p, cid in preds if p == p1)
                        cid_b = next(cid for p, cid in preds if p == p2)
                        contradictions.append(
                            Contradiction(
                                claim_a=cid_a,
                                claim_b=cid_b,
                                description=f"{subj} has both '{p1}' and '{p2}' relationship with {obj}",
                                status="unresolved",
                            )
                        )

        # STEP 7: Token budget enforcement
        rel_list = sorted(
            relationships.values(), key=lambda r: r.confidence, reverse=True
        )

        while rel_list and _estimate_tokens(
            " ".join(
                f"{r.predicate}:{r.target.id}:{r.confidence}"
                for r in rel_list
            )
        ) > max_tokens:
            rel_list.pop()  # Remove lowest-scored (last)

        # STEP 8: Assemble
        all_confidences = [c.confidence for c, *_ in scored] if scored else [0.0]
        provenance_counts: dict[str, int] = {}
        for claim, *_ in scored:
            st = claim.provenance.source_type
            provenance_counts[st] = provenance_counts.get(st, 0) + 1
        total_claims = len(scored)
        provenance_summary = {
            st: count / total_claims for st, count in provenance_counts.items()
        } if total_claims > 0 else {}

        # Check for open inquiries involving the focal entity
        inquiry_ids = []
        try:
            for d in self._store.claims_for(canonical):
                ic = self._convert_claim(d)
                if ic.predicate.id == "inquiry" and ic.status == ClaimStatus.ACTIVE:
                    inquiry_ids.append(ic.claim_id)
        except Exception:
            pass

        frame = ContextFrame(
            focal_entity=entity_summary,
            direct_relationships=rel_list,
            quantitative_data=quantitative,
            contradictions=contradictions,
            knowledge_gaps=[],
            narrative="",
            provenance_summary=provenance_summary,
            claim_count=total_claims,
            confidence_range=(min(all_confidences), max(all_confidences)),
            open_inquiries=inquiry_ids,
        )

        if include_narrative:
            try:
                from attestdb.intelligence.narrative import generate_narrative
                frame.narrative = generate_narrative(frame)
            except ImportError:
                pass  # narrative generation requires attestdb-intelligence

        return frame

    def explain(
        self,
        focal_entity: str,
        depth: int = 2,
        min_confidence: float = 0.0,
        exclude_source_types: list[str] | None = None,
        max_claims: int = 500,
        max_tokens: int = 4000,
        confidence_threshold: float = 0.0,
        predicate_types: list[str] | None = None,
    ) -> tuple[ContextFrame, QueryProfile]:
        """Like query() but also returns a QueryProfile with timing and counts."""
        t0 = time.time()
        exclude = set(exclude_source_types or [])
        effective_min_conf = max(min_confidence, confidence_threshold)
        allowed_predicates = set(predicate_types) if predicate_types else None

        # STEP 1: Resolve
        canonical = self._store.resolve(normalize_entity_id(focal_entity))
        raw_entity = self._store.get_entity(canonical)
        if raw_entity is None:
            raise EntityNotFoundError(f"Entity not found: {focal_entity}")

        # STEP 2: BFS
        candidates: list[tuple[Claim, int, str]] = []
        visited_entities: set[str] = {canonical}
        aliases = self._store.get_alias_group(canonical)
        visited_entities.update(aliases)
        frontier: set[str] = set(aliases)
        seen_claim_ids: set[str] = set()

        for hop in range(1, depth + 1):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for eid in list(frontier):
                for claim in [self._convert_claim(d) for d in self._store.claims_for(eid)]:
                    if claim.claim_id in seen_claim_ids:
                        continue
                    if claim.confidence < effective_min_conf:
                        continue
                    if claim.provenance.source_type in exclude:
                        continue
                    if claim.status != ClaimStatus.ACTIVE:
                        continue
                    if allowed_predicates and claim.predicate.id not in allowed_predicates:
                        continue
                    seen_claim_ids.add(claim.claim_id)
                    candidates.append((claim, hop, eid))
                    if claim.subject.id == eid or claim.subject.id in visited_entities:
                        other = claim.object.id
                    else:
                        other = claim.subject.id
                    if other not in visited_entities:
                        next_frontier.add(other)
                        visited_entities.add(other)
            frontier = next_frontier

        total_candidates = len(candidates)

        # Run normal query for the frame
        frame = self.query(
            focal_entity, depth, min_confidence, exclude_source_types,
            max_claims, max_tokens,
            confidence_threshold=confidence_threshold,
            predicate_types=predicate_types,
        )

        elapsed_ms = (time.time() - t0) * 1000

        filters_applied = {}
        if min_confidence > 0:
            filters_applied["min_confidence"] = min_confidence
        if exclude_source_types:
            filters_applied["exclude_source_types"] = exclude_source_types
        if confidence_threshold > 0:
            filters_applied["confidence_threshold"] = confidence_threshold
        if predicate_types:
            filters_applied["predicate_types"] = predicate_types

        profile = QueryProfile(
            focal_entity=canonical,
            total_candidates=total_candidates,
            after_scoring=frame.claim_count,
            after_budget=len(frame.direct_relationships),
            elapsed_ms=elapsed_ms,
            depth=depth,
            filters_applied=filters_applied,
        )

        return frame, profile

    def find_paths(
        self, entity_a: str, entity_b: str, max_depth: int = 3, top_k: int = 5,
    ) -> list[PathResult]:
        """Find top-k paths between two entities using Python-side BFS."""
        ra = self._store.resolve(entity_a)
        rb = self._store.resolve(entity_b)
        if ra == rb:
            return []

        # BFS with parent tracking
        parent: dict[str, list[tuple[str | None, Claim | None]]] = {ra: [(None, None)]}
        frontier: set[str] = {ra}
        found_paths: list[list[tuple[str, Claim]]] = []

        for _hop in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for eid in list(frontier):
                claims = [self._convert_claim(d) for d in self._store.claims_for(eid, None, None, 0.0)]
                for claim in claims:
                    other = claim.object.id if claim.subject.id == eid else claim.subject.id
                    if other == rb:
                        path = self._reconstruct_path(parent, eid, claim, ra)
                        if path:
                            found_paths.append(path)
                    elif other not in parent:
                        parent.setdefault(other, []).append((eid, claim))
                        next_frontier.add(other)
            frontier = next_frontier

        # Convert to PathResult
        results: list[PathResult] = []
        for path in found_paths:
            steps = []
            total_conf = 1.0
            for step_eid, step_claim in path:
                raw_entity = self._store.get_entity(step_eid)
                etype = entity_summary_from_dict(raw_entity).entity_type if raw_entity else ""
                steps.append(PathStep(
                    entity_id=step_eid,
                    entity_type=etype,
                    predicate=step_claim.predicate.id,
                    confidence=step_claim.confidence,
                    source_types=[step_claim.provenance.source_type],
                ))
                total_conf *= step_claim.confidence
            results.append(PathResult(steps=steps, total_confidence=total_conf, length=len(steps)))

        results.sort(key=lambda p: p.total_confidence, reverse=True)
        return results[:top_k]

    @staticmethod
    def _reconstruct_path(
        parent: dict[str, list[tuple[str | None, Claim | None]]],
        last_eid: str,
        final_claim: Claim,
        start_eid: str,
    ) -> list[tuple[str, Claim]]:
        """Reconstruct a single path from parent tracking back to start."""
        path = [(last_eid, final_claim)]
        current = last_eid
        while current != start_eid:
            entries = parent.get(current, [])
            if not entries:
                return []
            prev_eid, prev_claim = entries[0]
            if prev_eid is None:
                break
            if prev_claim is not None:
                path.append((prev_eid, prev_claim))
            current = prev_eid
        path.reverse()
        return path
