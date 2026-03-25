"""Topic thread subsystem — investigative threads over the knowledge graph."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import (
    Claim,
    ClaimStatus,
    EntitySummary,
    claim_from_dict,
    entity_summary_from_dict,
)

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)


class TopicThreads:
    """Manages topic threads — live investigative queries over the knowledge graph.

    Threads are claims in the graph (not sidecars), re-execute traversal on
    resume to auto-discover new claims.
    """

    def __init__(self, db: "AttestDB") -> None:
        self.db = db

    # --- Static helpers ---

    @staticmethod
    def _format_claim_oneliner(c: "Claim") -> str:
        """Format a claim as a single-line string."""
        subj = c.subject.display_name or c.subject.id
        obj = c.object.display_name or c.object.id
        return f"{subj} {c.predicate.id} {obj}"

    @staticmethod
    def _select_key_findings(
        claims: list,
        target: int = 3,
        max_count: int = 10,
        initial_threshold: float = 0.8,
        floor: float = 0.4,
    ) -> tuple[list, float]:
        """Select key findings with adaptive confidence threshold.

        Starts at initial_threshold, lowers by 0.1 until at least `target`
        findings are found or the floor is reached.

        Returns (findings, threshold_used).
        """
        threshold = initial_threshold
        while threshold >= floor:
            findings = [c for c in claims if c.confidence >= threshold]
            if len(findings) >= target:
                findings.sort(key=lambda c: c.confidence, reverse=True)
                return findings[:max_count], threshold
            threshold = round(threshold - 0.1, 2)
        # Below floor: return whatever we have
        findings = sorted(claims, key=lambda c: c.confidence, reverse=True)[:max_count]
        effective = findings[-1].confidence if findings else 0.0
        return findings, effective

    # --- Contradiction details ---

    def _build_contradiction_details(
        self,
        contradictions: list,
        claims_dict: dict,
    ) -> list[dict]:
        """Build structured contradiction details with evidence counts and confidence."""
        details = []
        for ctr in contradictions:
            claim_a = claims_dict.get(ctr.claim_a) or self.db.get_claim(ctr.claim_a)
            claim_b = claims_dict.get(ctr.claim_b) or self.db.get_claim(ctr.claim_b)
            if not claim_a or not claim_b:
                continue

            evidence_a = ctr.evidence_a
            evidence_b = ctr.evidence_b
            conf_a = claim_a.confidence
            conf_b = claim_b.confidence

            # If evidence counts weren't populated, estimate from source counts
            if evidence_a == 0:
                evidence_a = max(1, getattr(claim_a, 'n_independent_sources', 1) or 1)
            if evidence_b == 0:
                evidence_b = max(1, getattr(claim_b, 'n_independent_sources', 1) or 1)

            # Determine resolution status
            if ctr.status != "unresolved":
                status = ctr.status
            elif conf_a > conf_b and evidence_a > evidence_b:
                status = "a_preferred"
            elif conf_b > conf_a and evidence_b > evidence_a:
                status = "b_preferred"
            elif conf_a > conf_b + 0.2:
                status = "a_preferred"
            elif conf_b > conf_a + 0.2:
                status = "b_preferred"
            else:
                status = "unresolved"

            summary_a = self._format_claim_oneliner(claim_a)
            summary_b = self._format_claim_oneliner(claim_b)

            details.append({
                "description": ctr.description,
                "claim_a_summary": summary_a,
                "claim_b_summary": summary_b,
                "confidence_a": round(conf_a, 2),
                "confidence_b": round(conf_b, 2),
                "evidence_a": evidence_a,
                "evidence_b": evidence_b,
                "status": status,
            })
        return details

    # --- Thread synthesis ---

    def _generate_thread_synthesis(
        self,
        key_findings: list,
        contradiction_details: list[dict],
        seed_entities: list[str],
        claim_count: int,
        entity_count: int,
    ) -> str:
        """Generate a narrative synthesis of a thread via LLM.

        Falls back to structural summary when no LLM provider is configured
        or on any error.
        """
        if self.db._intel._curator_model == "heuristic":
            return self._structural_thread_summary(
                key_findings, contradiction_details, seed_entities,
                claim_count, entity_count,
            )

        try:
            extractor = self.db._get_text_extractor()
            client = extractor._client
            model = extractor._llm_model
            if not client or not model:
                raise ValueError("No LLM client available")

            findings_text = "\n".join(
                f"- {self._format_claim_oneliner(f)} (conf={f.confidence:.2f})"
                for f in key_findings[:10]
            ) or "(none)"

            contradictions_text = "\n".join(
                f"- {c['claim_a_summary']} vs {c['claim_b_summary']} ({c['status']})"
                for c in contradiction_details[:5]
            ) or "(none)"

            prompt = (
                "Summarize the following research thread in 2-3 sentences. "
                "Focus on: what is established, what is contested, and what is unknown.\n\n"
                f"Topic: {', '.join(seed_entities[:5])}\n\n"
                f"Key findings:\n{findings_text}\n\n"
                f"Contradictions:\n{contradictions_text}\n\n"
                "Write a concise scientific summary. No preamble."
            )

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                timeout=60,
            )
            text = resp.choices[0].message.content
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        return self._structural_thread_summary(
            key_findings, contradiction_details, seed_entities,
            claim_count, entity_count,
        )

    @staticmethod
    def _structural_thread_summary(
        key_findings: list,
        contradiction_details: list[dict],
        seed_entities: list[str],
        claim_count: int,
        entity_count: int,
    ) -> str:
        """Structural metadata summary (no LLM)."""
        parts = [f"Thread covering {entity_count} entities with {claim_count} claims."]
        if seed_entities:
            parts.append(f"Seeded from: {', '.join(seed_entities[:5])}")
        if key_findings:
            parts.append(f"{len(key_findings)} key findings.")
        if contradiction_details:
            parts.append(f"{len(contradiction_details)} contradictions.")
        return " ".join(parts)

    # --- Public API ---

    def create_thread(
        self,
        seed: "str | list[str]",
        depth: int = 2,
        min_confidence: float = 0.5,
        source_filter: list[str] | None = None,
        exclude_source_types: list[str] | None = None,
    ) -> "ThreadContext":
        """Create a new topic thread from a seed.

        seed: entity ID, list of entity IDs, or NL query (prefix with '?').
        Returns ThreadContext with claims, summary, open questions, key findings, frontier.
        """
        from attestdb.core.types import ThreadContext, ThreadState

        # Resolve seed -> entity IDs
        seed_entities = self._resolve_thread_seed(seed)
        if not seed_entities:
            return ThreadContext(thread_id="", summary="No matching entities found")

        seed_query = seed if isinstance(seed, str) and seed.startswith("?") else None

        # Traverse from each seed, merge results
        all_claims: dict[str, Claim] = {}
        all_frontier: set[str] = set()
        all_contradictions: list = []
        all_gaps: list[str] = []
        entities_seen: set[str] = set()

        for eid in seed_entities:
            try:
                frame = self.db.query(
                    focal_entity=eid, depth=depth, min_confidence=min_confidence,
                    exclude_source_types=exclude_source_types,
                )
            except Exception:
                continue
            # Collect claims from relationships
            for rel in frame.direct_relationships:
                target_id = rel.target.id
                entities_seen.add(target_id)
                # Get claims for this entity-relationship pair
                for c in self.db.claims_for(eid):
                    all_claims[c.claim_id] = c
            # Also get claims directly for the seed entity
            for c in self.db.claims_for(eid):
                all_claims[c.claim_id] = c

            all_contradictions.extend(frame.contradictions)
            all_gaps.extend(frame.knowledge_gaps)

            # Frontier = entities at max depth (targets with unexplored edges)
            for rel in frame.direct_relationships:
                tid = rel.target.id
                entities_seen.add(tid)
                # Check if this target has more unexplored edges
                try:
                    further = self.db._store.claims_for(tid, None, None, 0.0)
                    targets_of_target = {d.get("subject", {}).get("id") for d in further}
                    targets_of_target |= {d.get("object", {}).get("id") for d in further}
                    unexplored = targets_of_target - entities_seen - {eid}
                    if unexplored:
                        all_frontier.add(tid)
                except Exception:
                    pass

        claims_list = list(all_claims.values())
        claim_ids = [c.claim_id for c in claims_list]

        # Extract narrative metadata -- adaptive key findings
        finding_claims, findings_threshold = self._select_key_findings(claims_list)
        key_findings = []
        for c in finding_claims:
            status_tag = f" [{c.status.value}]" if c.status != ClaimStatus.ACTIVE else ""
            key_findings.append(
                f"{c.subject.display_name or c.subject.id} "
                f"{c.predicate.id} "
                f"{c.object.display_name or c.object.id} "
                f"(conf={c.confidence:.2f}{status_tag})"
            )

        # Build structured contradiction details
        contradiction_details = self._build_contradiction_details(
            all_contradictions, all_claims,
        )

        open_questions = []
        for ctr in all_contradictions[:5]:
            if ctr.description:
                open_questions.append(f"Contradiction: {ctr.description}")
        for gap in all_gaps[:5]:
            open_questions.append(f"Gap: {gap}")

        frontier_list = list(all_frontier)[:20]

        # Generate summary (LLM synthesis if configured, else structural)
        summary = self._generate_thread_synthesis(
            key_findings=finding_claims,
            contradiction_details=contradiction_details,
            seed_entities=seed_entities,
            claim_count=len(claims_list),
            entity_count=len(entities_seen),
        )

        # Generate thread_id and persist as claims in the graph
        now = int(time.time() * 1e9)
        thread_id = f"thread_{hashlib.sha256(f'{seed}{now}'.encode()).hexdigest()[:12]}"

        state = ThreadState(
            thread_id=thread_id,
            seed_entities=seed_entities,
            seed_query=seed_query,
            traversal_depth=depth,
            min_confidence=min_confidence,
            source_filter=source_filter,
            exclude_source_types=exclude_source_types,
            claim_ids=claim_ids,
            frontier_entities=frontier_list,
            created_at=now,
            last_accessed=now,
            summary=summary,
            open_questions=open_questions,
            key_findings=key_findings,
            contradiction_details=contradiction_details,
            key_findings_threshold=findings_threshold,
        )

        self._persist_thread_state(state, seed_entities)

        # Update entity-thread index
        for eid in seed_entities:
            self.db._entity_thread_index.setdefault(eid, set()).add(thread_id)
        for eid in entities_seen:
            self.db._entity_thread_index.setdefault(eid, set()).add(thread_id)

        # Build frontier EntitySummary list
        frontier_summaries = []
        for fid in frontier_list:
            try:
                es = entity_summary_from_dict(self.db._store.get_entity(fid))
                frontier_summaries.append(es)
            except Exception:
                frontier_summaries.append(EntitySummary(id=fid, name=fid, entity_type="entity"))

        return ThreadContext(
            thread_id=thread_id,
            claims=claims_list,
            summary=summary,
            open_questions=open_questions,
            key_findings=key_findings,
            frontier=frontier_summaries,
            claim_count=len(claims_list),
            entity_count=len(entities_seen),
            seed_entities=seed_entities,
        )

    def resume_thread(self, thread_id: str) -> "ThreadContext":
        """Resume an existing thread. Re-executes traversal, diffs against previous state."""
        from attestdb.core.types import ThreadContext

        state = self._load_thread_state(thread_id)
        if state is None:
            return ThreadContext(thread_id=thread_id, summary="Thread not found")

        previous_claim_ids = set(state.claim_ids)

        # Re-execute traversal directly (not via create_thread to avoid creating new thread claims)
        all_claims: dict[str, Claim] = {}
        all_frontier: set[str] = set()
        all_contradictions: list = []
        all_gaps: list[str] = []
        entities_seen: set[str] = set()

        for eid in state.seed_entities:
            try:
                frame = self.db.query(
                    focal_entity=eid, depth=state.traversal_depth,
                    min_confidence=state.min_confidence,
                    exclude_source_types=state.exclude_source_types,
                )
            except Exception:
                continue
            for c in self.db.claims_for(eid):
                all_claims[c.claim_id] = c
            for rel in frame.direct_relationships:
                entities_seen.add(rel.target.id)
            all_contradictions.extend(frame.contradictions)
            all_gaps.extend(frame.knowledge_gaps)
            for rel in frame.direct_relationships:
                all_frontier.add(rel.target.id)

        claims_list = list(all_claims.values())
        current_claim_ids = {c.claim_id for c in claims_list}
        new_claims = [c for c in claims_list if c.claim_id not in previous_claim_ids]
        removed_ids = list(previous_claim_ids - current_claim_ids)

        # Regenerate narrative metadata -- adaptive key findings
        finding_claims, findings_threshold = self._select_key_findings(claims_list)
        key_findings = []
        for c in finding_claims:
            status_tag = f" [{c.status.value}]" if c.status != ClaimStatus.ACTIVE else ""
            key_findings.append(
                f"{c.subject.display_name or c.subject.id} "
                f"{c.predicate.id} {c.object.display_name or c.object.id} "
                f"(conf={c.confidence:.2f}{status_tag})"
            )

        # Build structured contradiction details
        contradiction_details = self._build_contradiction_details(
            all_contradictions, all_claims,
        )

        open_questions = []
        for ctr in all_contradictions[:5]:
            if ctr.description:
                open_questions.append(f"Contradiction: {ctr.description}")
        for gap in all_gaps[:5]:
            open_questions.append(f"Gap: {gap}")

        # Generate summary (LLM synthesis if configured, else structural)
        summary = self._generate_thread_synthesis(
            key_findings=finding_claims,
            contradiction_details=contradiction_details,
            seed_entities=state.seed_entities,
            claim_count=len(claims_list),
            entity_count=len(entities_seen),
        )
        if new_claims:
            summary += f" {len(new_claims)} new claims since last access."
        if removed_ids:
            summary += f" {len(removed_ids)} claims removed."

        # Update persisted state
        state.claim_ids = [c.claim_id for c in claims_list]
        state.frontier_entities = list(all_frontier)[:20]
        state.last_accessed = int(time.time() * 1e9)
        state.summary = summary
        state.open_questions = open_questions
        state.key_findings = key_findings
        state.contradiction_details = contradiction_details
        state.key_findings_threshold = findings_threshold
        state.stale = False

        self._persist_thread_state(state, state.seed_entities, overwrite_id=thread_id)

        # Build frontier summaries
        frontier_summaries = []
        for fid in state.frontier_entities[:20]:
            try:
                es = entity_summary_from_dict(self.db._store.get_entity(fid))
                frontier_summaries.append(es)
            except Exception:
                frontier_summaries.append(EntitySummary(id=fid, name=fid, entity_type="entity"))

        return ThreadContext(
            thread_id=thread_id,
            claims=claims_list,
            summary=summary,
            open_questions=open_questions,
            key_findings=key_findings,
            frontier=frontier_summaries,
            claim_count=len(claims_list),
            entity_count=len(entities_seen),
            seed_entities=state.seed_entities,
            new_since_last=new_claims,
            removed_since_last=removed_ids,
        )

    def extend_thread(
        self,
        thread_id: str,
        direction: "str | list[str] | None" = None,
        additional_depth: int = 1,
    ) -> "ThreadContext":
        """Extend a thread deeper into the graph from frontier entities."""
        from attestdb.core.types import ThreadContext

        state = self._load_thread_state(thread_id)
        if state is None:
            return ThreadContext(thread_id=thread_id, summary="Thread not found")

        # Resolve extension targets
        if direction is None:
            targets = state.frontier_entities
        elif isinstance(direction, list):
            targets = direction
        elif isinstance(direction, str) and direction.startswith("?"):
            targets = self._resolve_thread_seed(direction)
        else:
            targets = [direction]

        if not targets:
            targets = state.frontier_entities[:10]

        # Update state with extended seeds and depth
        state.seed_entities = list(set(state.seed_entities + targets))
        state.traversal_depth = state.traversal_depth + additional_depth

        # Persist extension tracking claim
        persist_id = f"thread_engine_{int(time.time() * 1e9)}"
        try:
            self.db.ingest(
                subject=(thread_id, "thread"),
                predicate=("thread_extended_by", "threads"),
                object=(targets[0] if targets else "unknown", "entity"),
                provenance={"source_type": "system", "source_id": persist_id},
                confidence=1.0,
            )
        except Exception:
            pass

        # Resume with updated parameters to get fresh traversal
        state.stale = True
        self._persist_thread_state(state, state.seed_entities, overwrite_id=thread_id)
        return self.resume_thread(thread_id)

    def branch_thread(
        self,
        parent_thread_id: str,
        new_seed: "str | list[str]",
        keep_parent_claims: bool = True,
    ) -> "ThreadContext":
        """Branch a thread to explore a tangent while preserving the parent."""
        parent_state = self._load_thread_state(parent_thread_id)
        if parent_state is None:
            from attestdb.core.types import ThreadContext
            return ThreadContext(thread_id="", summary="Parent thread not found")

        new_seeds = self._resolve_thread_seed(new_seed)
        if keep_parent_claims:
            combined_seeds = list(set(parent_state.seed_entities + new_seeds))
        else:
            combined_seeds = new_seeds

        ctx = self.create_thread(
            seed=combined_seeds,
            depth=parent_state.traversal_depth,
            min_confidence=parent_state.min_confidence,
            source_filter=parent_state.source_filter,
            exclude_source_types=parent_state.exclude_source_types,
        )

        # Record branch provenance
        self.db.ingest(
            subject=(ctx.thread_id, "thread"),
            predicate=("thread_branched_from", "threads"),
            object=(parent_thread_id, "thread"),
            provenance={"source_type": "system", "source_id": "thread_engine"},
            confidence=1.0,
        )

        return ctx

    def thread_context(
        self,
        thread_id: str,
        max_tokens: int = 4000,
        focus: str | None = None,
    ) -> str:
        """Serialize a thread into a context string for an agent's context window."""
        state = self._load_thread_state(thread_id)
        if state is None:
            return f"Thread {thread_id} not found."

        lines = []

        # Header
        seed_desc = ", ".join(state.seed_entities[:5])
        lines.append(f"## Thread: {seed_desc}")
        lines.append("")

        # Summary
        if state.summary:
            lines.append("### Summary")
            lines.append(state.summary)
            lines.append("")

        # Key findings -- label shows threshold when it was lowered
        if state.key_findings:
            threshold = getattr(state, 'key_findings_threshold', 0.8)
            if threshold < 0.8:
                lines.append(f"### Key Findings (confidence \u2265 {threshold:.1f})")
            else:
                lines.append("### Key Findings (verified)")
            for kf in state.key_findings[:10]:
                lines.append(f"- {kf}")
            lines.append("")

        # Contradictions -- structured resolution summaries
        contradiction_details = getattr(state, 'contradiction_details', None)
        if contradiction_details:
            lines.append("### Contradictions")
            for c in contradiction_details[:10]:
                status_label = {
                    "a_preferred": f"\u2192 Evidence favors: {c['claim_a_summary']}",
                    "b_preferred": f"\u2192 Evidence favors: {c['claim_b_summary']}",
                    "unresolved": "\u2192 Unresolved \u2014 evidence is balanced",
                }.get(c["status"], "\u2192 Status unknown")
                lines.append(
                    f"- {c['claim_a_summary']} (conf={c['confidence_a']:.2f}, "
                    f"{c['evidence_a']} sources) vs "
                    f"{c['claim_b_summary']} (conf={c['confidence_b']:.2f}, "
                    f"{c['evidence_b']} sources)"
                )
                lines.append(f"  {status_label}")
            lines.append("")
        elif state.open_questions:
            # Backward compat: old threads without contradiction_details
            lines.append("### Open Questions")
            for oq in state.open_questions[:10]:
                lines.append(f"- {oq}")
            lines.append("")

        # Non-contradiction open questions (gaps)
        gap_questions = [oq for oq in state.open_questions if not oq.startswith("Contradiction:")]
        if gap_questions:
            lines.append("### Open Questions")
            for oq in gap_questions[:10]:
                lines.append(f"- {oq}")
            lines.append("")

        # Claims (budget-aware)
        claims = []
        for cid in state.claim_ids:
            c = self.db.get_claim(cid)
            if c:
                claims.append(c)

        # Focus prioritization
        if focus:
            focus_lower = focus.lower()
            claims.sort(key=lambda c: (
                0 if focus_lower in c.subject.id.lower() or focus_lower in c.object.id.lower() else 1,
                -c.confidence,
                -c.timestamp,
            ))
        else:
            claims.sort(key=lambda c: (-c.confidence, -c.timestamp))

        # Estimate token budget (rough: 1 claim ~ 40 tokens, header ~ 200)
        header_tokens = len("\n".join(lines)) // 4
        remaining = max_tokens - header_tokens
        claims_budget = max(remaining // 40, 0)

        displayed_claims = claims[:claims_budget] if claims_budget > 0 else []
        if displayed_claims:
            lines.append("### Claims")
            for c in displayed_claims:
                status_tag = f" [{c.status.value}]" if c.status != ClaimStatus.ACTIVE else ""
                lines.append(
                    f"- {c.subject.display_name or c.subject.id} "
                    f"{c.predicate.id} "
                    f"{c.object.display_name or c.object.id} "
                    f"(conf={c.confidence:.2f}{status_tag})"
                )
            lines.append("")

        # Dropped claims signal
        total_claims = len(claims)
        shown_claims = len(displayed_claims)
        dropped = total_claims - shown_claims
        if dropped > 0:
            lines.append("### Note")
            lines.append(
                f"Showing {shown_claims} of {total_claims} claims "
                f"(sorted by confidence \u00d7 recency). "
                f"{dropped} claims omitted due to token budget. "
                f"Use `focus` parameter to narrow scope, or increase `max_tokens`."
            )
            # Show entities only in dropped claims
            dropped_entities: set[str] = set()
            for c in claims[shown_claims:]:
                dropped_entities.add(c.subject.id)
                dropped_entities.add(c.object.id)
            shown_entities: set[str] = set()
            for c in displayed_claims:
                shown_entities.add(c.subject.id)
                shown_entities.add(c.object.id)
            unique_dropped = dropped_entities - shown_entities
            if unique_dropped:
                sample = sorted(unique_dropped)[:5]
                suffix = f" (+{len(unique_dropped) - 5} more)" if len(unique_dropped) > 5 else ""
                lines.append(f"Entities only in omitted claims: {', '.join(sample)}{suffix}")
            lines.append("")

        # Frontier
        if state.frontier_entities:
            lines.append("### Frontier (unexplored edges)")
            for fe in state.frontier_entities[:10]:
                lines.append(f"- {fe}")
            lines.append("")

        return "\n".join(lines)

    def list_threads(
        self,
        entity_filter: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """List existing threads, optionally filtered to threads touching an entity."""
        results = []
        # Find all thread entities
        thread_entities = self.db.list_entities(entity_type="thread", limit=limit * 5)

        for te in thread_entities:
            thread_id = te.id
            state = self._load_thread_state(thread_id)
            if state is None:
                continue
            if entity_filter:
                ef_lower = normalize_entity_id(entity_filter)
                if ef_lower not in [normalize_entity_id(s) for s in state.seed_entities]:
                    # Check entity-thread index
                    if ef_lower not in self.db._entity_thread_index.get(ef_lower, set()):
                        if thread_id not in self.db._entity_thread_index.get(ef_lower, set()):
                            continue
            results.append({
                "thread_id": thread_id,
                "seed_entities": state.seed_entities,
                "summary": state.summary or "",
                "last_accessed": state.last_accessed,
                "claim_count": len(state.claim_ids),
                "frontier_count": len(state.frontier_entities),
                "parent_thread_id": state.parent_thread_id,
            })
            if len(results) >= limit:
                break

        return results

    # --- Thread internals ---

    def _resolve_thread_seed(self, seed: "str | list[str]") -> list[str]:
        """Resolve seed input to a list of entity IDs."""
        if isinstance(seed, list):
            return seed

        # NL query
        if seed.startswith("?"):
            query_text = seed[1:].strip()
            results = self.db.search_entities(query_text, top_k=5)
            return [r.id for r in results]

        # Check if it's an existing entity
        entity = self.db._store.get_entity(seed)
        if entity:
            return [seed]

        # Normalized lookup
        normalized = normalize_entity_id(seed)
        entity = self.db._store.get_entity(normalized)
        if entity:
            return [normalized]

        # Fall back to search
        results = self.db.search_entities(seed, top_k=5)
        return [r.id for r in results]

    def _persist_thread_state(
        self,
        state: "ThreadState",
        seed_entities: list[str],
        overwrite_id: str | None = None,
    ) -> None:
        """Persist a thread state as claims in the graph.

        Uses unique source_ids per persist call to avoid content-level dedup.
        On overwrite, the old thread claims remain (append-only) but the latest
        payload wins on load (most recent by timestamp).
        """
        from dataclasses import asdict
        thread_id = overwrite_id or state.thread_id
        state.thread_id = thread_id

        payload = {
            "schema_ref": "topic_thread",
            "data": asdict(state),
        }

        # Use unique source_id per persist to avoid content-level dedup
        persist_id = f"thread_engine_{int(time.time() * 1e9)}"

        # Ingest one claim per seed entity (the thread_seeded_by relationships)
        for i, eid in enumerate(seed_entities[:10]):
            try:
                self.db.ingest(
                    subject=(thread_id, "thread"),
                    predicate=("thread_seeded_by", "threads"),
                    object=(eid, "entity"),
                    provenance={"source_type": "system", "source_id": persist_id},
                    confidence=1.0,
                    payload=payload if i == 0 else None,
                )
            except Exception:
                # Duplicate claim is OK -- thread state is carried by the latest payload
                pass

    def _load_thread_state(self, thread_id: str) -> "ThreadState | None":
        """Load a thread's state from its most recent claim payload."""
        from attestdb.core.types import ThreadState
        claims = self.db.claims_for(thread_id)
        # Find the most recent claim with a thread payload (highest timestamp)
        best = None
        for c in claims:
            if c.payload and c.payload.schema_ref == "topic_thread":
                if best is None or c.timestamp > best.timestamp:
                    best = c
        if best is None:
            return None
        data = best.payload.data
        if isinstance(data, dict):
            return ThreadState(**{
                k: v for k, v in data.items()
                if k in ThreadState.__dataclass_fields__
            })
        return None

    def _mark_threads_stale(self, entity_id: str) -> None:
        """Mark threads touching this entity as stale (narrative needs regeneration)."""
        thread_ids = self.db._entity_thread_index.get(entity_id, set())
        for tid in thread_ids:
            state = self._load_thread_state(tid)
            if state and not state.stale:
                state.stale = True
                self._persist_thread_state(state, state.seed_entities, overwrite_id=tid)
