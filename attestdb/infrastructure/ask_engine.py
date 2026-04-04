"""AskEngine — question-answering subsystem extracted from AttestDB.

V2 pipeline: entity-first resolution → graph-native evidence → focused LLM synthesis.
Replaces slow word-by-word BM25 scanning with three-tier entity extraction.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import AskResult, Citation, EntitySummary, PathResult, claim_from_dict

logger = logging.getLogger(__name__)


def _safe_claims_for(store, eid, pred=None, src=None, min_conf=0.0, limit=0):
    """Call claims_for with backwards-compatible limit parameter."""
    try:
        return store.claims_for(eid, pred, src, min_conf, limit)
    except TypeError:
        # Old wheel without limit param — always cap to prevent OOM
        result = store.claims_for(eid, pred, src, min_conf)
        cap = limit if limit > 0 else 500
        return result[:cap]


# Stop words for candidate generation
_STOP = frozenset({
    "what", "who", "how", "why", "when", "where", "which", "does",
    "the", "are", "for", "and", "that", "this", "with", "from",
    "about", "has", "have", "been", "is", "was", "were", "tell",
    "me", "show", "find", "get", "list", "of", "in", "on", "at",
    "by", "an", "a", "do", "did", "our", "we", "us", "made",
    "being", "last", "recent", "done", "any", "all", "some",
    "much", "many", "can", "could", "would", "should", "will",
    "to", "it", "its", "be", "not", "no", "or", "but", "if",
    "than", "them", "they", "their", "there", "then", "top",
    "best", "most", "also", "just", "only", "very", "more",
    "good", "bad", "new", "old", "like", "use", "used", "using",
    "target", "role", "effect", "type", "cause", "work", "works",
    "related", "involved", "between", "affect", "impact",
    "evidence", "suggest", "prevent", "inhibiting", "inhibit", "inhibition",
    "activate", "activating", "regulate", "know", "known",
})

# High-frequency domain terms too common for useful BM25 on large bio DBs
# (each matches 100K+ entities, BM25 scan takes 5-10s per term)
_BM25_SKIP = frozenset({
    "disease", "gene", "protein", "cell", "drug", "compound",
    "treatment", "therapy", "receptor", "enzyme", "tissue",
    "organ", "syndrome", "disorder", "condition", "mutation",
    "variant", "expression", "level", "factor", "response",
    "inhibition", "activation", "regulation", "suppression",
    "complement", "signaling", "process", "mechanism",
})

# Predicate specificity weights for bridge scoring.
# Causal/mechanistic predicates are more informative than generic associations.
_PRED_WEIGHT = {
    "inhibits": 1.0, "activates": 1.0, "binds": 1.0,
    "upregulates": 0.9, "downregulates": 0.9,
    "causes": 0.9, "prevents": 0.9, "treats": 0.9,
    "regulates": 0.7, "interacts": 0.6, "interacts_with": 0.6,
    "predisposes": 0.7, "contraindicates": 0.7,
    "expressed_in": 0.4, "participates_in": 0.3,
    "associated_with": 0.1, "associates": 0.1,
}


from attestdb.core.vocabulary import compose_predicates as _compose


def _entity_name(raw, fallback: str = "?") -> str:
    """Extract best display name from a raw entity dict."""
    if isinstance(raw, dict):
        return raw.get("display_name") or raw.get("name") or fallback
    if hasattr(raw, "name") and raw.name:
        return raw.name
    return fallback


@dataclass
class ResolvedEntity:
    """An entity resolved from a question, with match metadata."""
    entity_id: str
    name: str
    entity_type: str
    claim_count: int
    match_tier: int  # 1=exact, 2=bm25, 3=llm
    original_mention: str


class AskEngine:
    """Encapsulates the ask() pipeline and its helpers.

    V2 architecture: entity-first resolution → graph evidence → LLM synthesis.
    """

    def __init__(self, db, ops_callback=None):
        self.db = db
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._ops_callback = ops_callback

    # ──────────────────────────────────────────────────────────────────
    # LLM access (unchanged from v1)
    # ──────────────────────────────────────────────────────────────────

    def _get_llm_client(self):
        """Return (client, model) for LLM calls."""
        # Use pre-loaded client if available (avoids httpx deadlock in uvicorn)
        if hasattr(self, '_preloaded_client') and self._preloaded_client:
            return self._preloaded_client, self._preloaded_model
        if hasattr(self.db, '_preloaded_llm_client') and self.db._preloaded_llm_client:
            return self.db._preloaded_llm_client, self.db._preloaded_llm_model

        try:
            from attestdb.core.providers import PROVIDERS, EXTRACTION_FALLBACK_CHAIN
            import os
            from openai import OpenAI

            for provider_name in EXTRACTION_FALLBACK_CHAIN:
                provider = PROVIDERS.get(provider_name)
                if not provider:
                    continue
                key = os.environ.get(provider["env_key"], "")
                if not key:
                    continue
                client = OpenAI(api_key=key, base_url=provider["base_url"])
                model = provider["default_model"]
                logger.info("_get_llm_client: initialized %s/%s", provider_name, model)
                return client, model
        except ImportError:
            logger.debug("_get_llm_client: openai package not available")

        try:
            ext = self.db._get_text_extractor()
            if ext._client and ext._llm_model:
                return ext._client, ext._llm_model
        except Exception as exc:
            logger.debug("_get_llm_client: text extractor unavailable: %s", exc)

        return None, None

    def _llm_call_via_requests(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str | None:
        """LLM call using requests library directly — no httpx, no asyncio deadlock.

        This is the preferred path when running inside uvicorn or any async framework.
        Falls through the provider chain until one succeeds.
        """
        import os
        try:
            import requests as _requests
        except ImportError:
            return None

        try:
            from attestdb.core.providers import PROVIDERS, EXTRACTION_FALLBACK_CHAIN
        except ImportError:
            return None

        for provider_name in EXTRACTION_FALLBACK_CHAIN:
            provider = PROVIDERS.get(provider_name)
            if not provider:
                continue
            key = os.environ.get(provider["env_key"], "")
            if not key:
                continue
            try:
                url = provider["base_url"].rstrip("/") + "/chat/completions"
                resp = _requests.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": provider["default_model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        logger.info("_llm_call_requests: %s/%s succeeded", provider_name, provider["default_model"])
                        # Track token usage
                        usage = data.get("usage", {})
                        self._last_prompt_tokens = usage.get("prompt_tokens", 0)
                        self._last_completion_tokens = usage.get("completion_tokens", 0)
                        return content.strip()
            except Exception as exc:
                logger.debug("_llm_call_requests: %s failed: %s", provider_name, exc)
                continue

        return None

    def _llm_call(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str | None:
        """LLM call with provider fallback. Tries requests first (uvicorn-safe),
        falls back to openai library."""
        self._last_prompt_tokens = 0
        self._last_completion_tokens = 0
        # Prefer requests library — avoids httpx/uvicorn deadlock
        result = self._llm_call_via_requests(prompt, max_tokens, temperature)
        if result:
            self._total_prompt_tokens += self._last_prompt_tokens
            self._total_completion_tokens += self._last_completion_tokens
            return result

        # Fallback: openai library (works from CLI, may deadlock in uvicorn)
        client, model = self._get_llm_client()
        if not client:
            logger.warning("_llm_call: no LLM client available")
            return None

        messages = [{"role": "user", "content": prompt}]
        for param_name, param_kwargs in [
            ("max_completion_tokens", {"max_completion_tokens": max_tokens}),
            ("max_tokens", {"max_tokens": max_tokens}),
        ]:
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages,
                    temperature=temperature, timeout=30,
                    **param_kwargs,
                )
                content = r.choices[0].message.content
                if content and content.strip():
                    if hasattr(r, "usage") and r.usage:
                        self._last_prompt_tokens = getattr(r.usage, "prompt_tokens", 0) or 0
                        self._last_completion_tokens = getattr(r.usage, "completion_tokens", 0) or 0
                    self._total_prompt_tokens += self._last_prompt_tokens
                    self._total_completion_tokens += self._last_completion_tokens
                    return content.strip()
            except Exception:
                continue

        logger.warning("_llm_call: all providers failed (%d char prompt)", len(prompt))
        return None

    # ──────────────────────────────────────────────────────────────────
    # Phase A: Entity Extraction (target < 500ms)
    # ──────────────────────────────────────────────────────────────────

    def _generate_candidates(self, question: str) -> list[str]:
        """Extract candidate entity mention spans from question text.

        Returns candidate strings longest-first: contiguous non-stopword
        subsequences of 1-4 words.
        """
        words = question.split()
        clean_words = []
        for w in words:
            cleaned = w.strip("?.,!\"'()[]{}:;").lower()
            if cleaned and len(cleaned) >= 2 and cleaned not in _STOP:
                clean_words.append((len(clean_words), w.strip("?.,!\"'()[]{}:;")))

        candidates: list[str] = []
        seen: set[str] = set()

        # Generate contiguous spans of content words (max 4), longest first
        for span_len in range(min(4, len(clean_words)), 0, -1):
            for start in range(len(clean_words) - span_len + 1):
                span = clean_words[start:start + span_len]
                text = " ".join(w for _, w in span)
                key = text.lower()
                if key not in seen:
                    candidates.append(text)
                    seen.add(key)

        return candidates

    def _resolve_entity(self, candidate: str) -> ResolvedEntity | None:
        """Try to resolve a candidate mention to a database entity.

        Tier 1: exact match via normalize_entity_id → get_entity.
        Tier 2: targeted BM25 search for the full phrase.
        """
        # Tier 1: Exact match
        normalized = normalize_entity_id(candidate)
        raw = self.db._store.get_entity(normalized)
        if raw:
            cc = raw.get("claim_count", 0) if isinstance(raw, dict) else getattr(raw, "claim_count", 0)
            if cc > 0:
                name = _entity_name(raw, normalized)
                etype = raw.get("entity_type", "") if isinstance(raw, dict) else getattr(raw, "entity_type", "")
                return ResolvedEntity(
                    entity_id=normalized, name=name or normalized,
                    entity_type=etype, claim_count=cc,
                    match_tier=1, original_mention=candidate,
                )

        # Tier 2: Targeted BM25 — single words only, skip high-frequency terms
        if len(candidate.split()) > 1:
            return None
        if candidate.lower() in _BM25_SKIP:
            return None
        hits = self.db.search_entities(candidate, top_k=10)
        # Pick the hit with highest claim count (not first BM25 rank)
        best: ResolvedEntity | None = None
        cand_lower = candidate.lower()
        cand_words = set(cand_lower.split())
        for hit in hits:
            if hit.claim_count > 0:
                hit_name = (hit.name or hit.id).lower()
                if any(w in hit_name for w in cand_words if len(w) >= 3):
                    if best is None or hit.claim_count > best.claim_count:
                        best = ResolvedEntity(
                            entity_id=hit.id, name=hit.name or hit.id,
                            entity_type=hit.entity_type, claim_count=hit.claim_count,
                            match_tier=2, original_mention=candidate,
                        )
        if best:
            return best

        return None

    def _extract_entities_llm(self, question: str) -> list[tuple[str, str]]:
        """Tier 3: LLM extraction of entity names from question text."""
        prompt = (
            "Extract specific named entities (genes, proteins, diseases, drugs, "
            "compounds, organisms, pathways, companies, people) from this question. "
            "Return ONLY a JSON list of [name, type] pairs. No explanation.\n\n"
            f"Question: {question}\n\n"
            'Example: [["KRAS", "gene"], ["heart disease", "disease"]]'
        )
        response = self._llm_call(prompt, max_tokens=150, temperature=0.0)
        if not response:
            return []
        try:
            # Strip markdown code fences if present
            text = response.strip().strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
            parsed = json.loads(text)
            return [(name, t) for name, t in parsed if isinstance(name, str)]
        except (json.JSONDecodeError, ValueError, TypeError):
            return []

    def _extract_question_entities(self, question: str, top_k: int = 10) -> list[ResolvedEntity]:
        """Master entity extraction: Tier 1 (exact) → Tier 2 (BM25) → Tier 3 (LLM).

        Returns resolved entities sorted by match quality and claim count.
        """
        t0 = time.monotonic()
        candidates = self._generate_candidates(question)
        resolved: dict[str, ResolvedEntity] = {}

        # Tier 1+2: Try each candidate with exact match then BM25
        for candidate in candidates:
            if len(resolved) >= top_k:
                break
            entity = self._resolve_entity(candidate)
            if entity and entity.entity_id not in resolved:
                resolved[entity.entity_id] = entity

        tier12_time = time.monotonic() - t0
        logger.info("Entity extraction Tier 1+2: %d entities in %.0fms",
                     len(resolved), tier12_time * 1000)

        # Tier 3: LLM extraction only if we found 0 entities
        # (LLM calls add 5-15s; skip if we already have any entity match)
        if len(resolved) == 0:
            t1 = time.monotonic()
            llm_entities = self._extract_entities_llm(question)
            for name, type_hint in llm_entities:
                if len(resolved) >= top_k:
                    break
                entity = self._resolve_entity(name)
                if entity and entity.entity_id not in resolved:
                    entity.match_tier = 3
                    resolved[entity.entity_id] = entity
            logger.info("Entity extraction Tier 3 (LLM): +%d entities in %.0fms",
                         len(resolved) - len([r for r in resolved.values() if r.match_tier <= 2]),
                         (time.monotonic() - t1) * 1000)

        # Sort: exact matches first, then by claim count
        result = sorted(resolved.values(), key=lambda r: (-r.match_tier == 1, -r.claim_count))
        return result[:top_k]

    # ──────────────────────────────────────────────────────────────────
    # Phase B: Graph-Native Evidence (target < 2s)
    # ──────────────────────────────────────────────────────────────────

    def _classify_question(self, question: str, entities: list[ResolvedEntity]) -> str:
        """Classify question type for evidence strategy selection."""
        q = question.lower()
        relationship_words = {
            "cause", "prevent", "inhibit", "affect", "lead", "connect",
            "interact", "bind", "regulate", "associate", "link", "treat",
            "target", "pathway", "mechanism", "evidence",
        }
        has_rel = any(w in q for w in relationship_words)

        if len(entities) >= 2 and has_rel:
            return "relationship"
        if len(entities) == 1:
            specific = {"bind", "interact", "regulate", "target", "treat", "express", "inhibit"}
            if any(w in q for w in specific):
                return "single"
        return "exploratory"

    def _evidence_single(self, entity: ResolvedEntity) -> tuple[str, list[Citation]]:
        """Evidence for single-entity questions: predicate summary + top claims."""
        lines = [f"## {entity.name} ({entity.entity_type}, {entity.claim_count} claims)"]
        citations: list[Citation] = []

        # Predicate summary (instant — no claim materialization)
        if hasattr(self.db._store, 'entity_predicate_counts'):
            pred_counts = self.db._store.entity_predicate_counts(entity.entity_id)
            if isinstance(pred_counts, list) and pred_counts:
                lines.append("Relationship types:")
                for pred, count in pred_counts[:12]:
                    lines.append(f"  - {pred}: {count} claims")

        # Top claims by confidence — always fetch a small sample for graph citations
        # Rust-side limit=30 prevents full materialization even on 100K+ entities
        raw_claims = _safe_claims_for(self.db._store, entity.entity_id, None, None, 0.3, 30)
        if raw_claims:
            lines.append("\nTop relationships:")
            seen = set()
            for d in raw_claims:
                if not isinstance(d, dict):
                    continue
                c = claim_from_dict(d)
                subj = c.subject.display_name or c.subject.id
                obj = c.object.display_name or c.object.id
                triple = f"{subj} {c.predicate.id} {obj}"
                if triple in seen:
                    continue
                seen.add(triple)
                src = c.provenance.source_type if c.provenance else ""
                lines.append(f"  - {triple} [conf={c.confidence:.2f}, source: {src}]")
                # Evidence text from payload
                if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                    ev = c.payload.data.get("evidence_text", "")
                    if ev:
                        lines.append(f'    Evidence: "{ev}"')
                if len(citations) < 50:
                    citations.append(Citation(
                        claim_id=c.claim_id, subject=c.subject.id,
                        predicate=c.predicate.id, object=c.object.id,
                        confidence=c.confidence,
                        source_id=c.provenance.source_id,
                        source_type=c.provenance.source_type,
                    ))
                if len(seen) >= 25:
                    break

        return "\n".join(lines), citations

    def _evidence_relationship(self, entities: list[ResolvedEntity]) -> tuple[str, list[Citation], list[dict], list[str]]:
        """Evidence for relationship questions using neighborhood intersection.

        Claim-native approach — not graph BFS:
        1. Get each entity's neighbors by predicate (instant via adjacency index)
        2. Intersect neighborhoods — shared neighbors are the bridging evidence
        3. Score by predicate composition semantics
        4. For low-degree pairs, also try direct path finding
        """
        lines: list[str] = []
        citations: list[Citation] = []
        contradictions: list[dict] = []
        gaps: list[str] = []

        # Step 1: Bidirectional neighborhood intersection
        # Sample neighbors from BOTH entities and intersect the sets.
        # This is O(sample_a + sample_b) — much faster than BFS.
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]

                # Get neighbor IDs from Rust adjacency index (instant, no materialization)
                def _get_neighbors(eid: str) -> set[str]:
                    try:
                        return set(self.db._store.neighbors(eid))
                    except (AttributeError, TypeError):
                        raw = _safe_claims_for(self.db._store, eid, None, None, 0.0, 200)
                        return {(d.get("object",{}).get("id","") if d.get("subject",{}).get("id","") == eid else d.get("subject",{}).get("id","")) for d in raw if isinstance(d, dict)}

                neighbors_a = _get_neighbors(a.entity_id)
                neighbors_b = _get_neighbors(b.entity_id)

                # Direct connection?
                if b.entity_id in neighbors_a or a.entity_id in neighbors_b:
                    lines.append(f"\n## Direct connection: {a.name} ↔ {b.name}")

                # Intersect — shared neighbors are bridges
                shared = neighbors_a & neighbors_b
                if not shared:
                    gaps.append(f"No shared neighbors between {a.name} and {b.name} ({len(neighbors_a)}+{len(neighbors_b)} neighbors)")
                    continue

                # Score each bridge using entity_predicate_counts (instant, no claims)
                scored: list[tuple] = []
                for mid_id in shared:
                    mid_raw = self.db._store.get_entity(mid_id)
                    if not mid_raw:
                        continue
                    mid_name = _entity_name(mid_raw, mid_id)
                    mid_cc = mid_raw.get("claim_count", 0) if isinstance(mid_raw, dict) else 0
                    if mid_cc > 200000:
                        continue  # skip mega-hubs

                    # Get bridge entity's predicates (instant from counter table)
                    mid_preds = self.db._store.entity_predicate_counts(mid_id) if hasattr(self.db._store, 'entity_predicate_counts') else []
                    if not isinstance(mid_preds, list) or not mid_preds:
                        continue

                    # Use top 2 predicates as proxy for A→Bridge and Bridge→B
                    pred_a = mid_preds[0][0]
                    pred_b = mid_preds[1][0] if len(mid_preds) > 1 else pred_a
                    weight_a = _PRED_WEIGHT.get(pred_a, 0.2)
                    weight_b = _PRED_WEIGHT.get(pred_b, 0.2)
                    score = weight_a * weight_b

                    composed = _compose(pred_a, pred_b)
                    scored.append((mid_id, mid_name, mid_cc, score, pred_a, pred_b, composed))

                scored.sort(key=lambda x: -x[3])
                n_specific = sum(1 for s in scored if s[3] >= 0.3)
                lines.append(f"\n## Bridges: {a.name} ↔ {b.name} ({len(scored)} shared, {n_specific} with specific predicates)")

                for mid_id, mid_name, mid_cc, score, pred_a, pred_b, composed in scored[:5]:
                    composed_tag = f" → inferred: {a.name} {composed} {b.name}" if composed != "associated_with" else ""
                    lines.append(f"  {a.name} --[{pred_a}]--> {mid_name} --[{pred_b}]--> {b.name}  (score={score:.2f}, {mid_cc} claims){composed_tag}")
                    if len(citations) < 48:
                        citations.append(Citation(
                            claim_id=f"bridge:{a.entity_id}:{mid_id}:leg1",
                            subject=a.entity_id, predicate=pred_a,
                            object=mid_id, confidence=score,
                            source_id="bridge", source_type="causal_composition",
                        ))
                        citations.append(Citation(
                            claim_id=f"bridge:{mid_id}:{b.entity_id}:leg2",
                            subject=mid_id, predicate=pred_b,
                            object=b.entity_id, confidence=score,
                            source_id="bridge", source_type="causal_composition",
                        ))

        # Add predicate summaries for each entity (instant, no materialization)
        for entity in entities[:4]:
            if hasattr(self.db._store, 'entity_predicate_counts'):
                pred_counts = self.db._store.entity_predicate_counts(entity.entity_id)
                if isinstance(pred_counts, list) and pred_counts:
                    lines.append(f"\n## {entity.name} — relationship summary ({entity.claim_count} claims):")
                    for pred, count in pred_counts[:10]:
                        lines.append(f"  - {pred}: {count} claims")
                    # Build summary citations for high-degree entities
                    if entity.claim_count > 500 and len(citations) < 50:
                        for pred, count in pred_counts[:8]:
                            citations.append(Citation(
                                claim_id=f"summary:{entity.entity_id}:{pred}",
                                subject=entity.entity_id, predicate=pred,
                                object=f"{count} targets",
                                confidence=min(0.7, count / 100),
                                source_id="predicate_summary",
                                source_type="aggregate",
                            ))

        return "\n".join(lines), citations, contradictions, gaps

    def _evidence_exploratory(self, entity: ResolvedEntity) -> tuple[str, list[Citation]]:
        """Evidence for exploratory questions: BFS depth-1 + full summary."""
        # Reuse single-entity evidence (it's already comprehensive)
        return self._evidence_single(entity)

    def _assemble_evidence(
        self, entities: list[ResolvedEntity], question_type: str,
    ) -> tuple[str, list[Citation], list[dict], list[str]]:
        """Dispatch to appropriate evidence strategy."""
        if question_type == "relationship" and len(entities) >= 2:
            return self._evidence_relationship(entities)
        elif question_type == "single" and entities:
            text, cites = self._evidence_single(entities[0])
            return text, cites, [], []
        elif entities:
            text, cites = self._evidence_exploratory(entities[0])
            return text, cites, [], []
        return "", [], [], []

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = 10) -> AskResult:
        """Answer a natural-language question using the knowledge graph.

        V2 pipeline: entity extraction → graph evidence → LLM synthesis.
        Target: complex questions < 10s, simple questions < 3s.
        """
        t_start = time.monotonic()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # Phase A: Entity extraction (< 500ms)
        entities = self._extract_question_entities(question, top_k=top_k)
        t_a = time.monotonic()

        if not entities:
            return AskResult(
                answer=None,
                meta={"pipeline": "v2", "phase_a_ms": int((t_a - t_start) * 1000),
                       "n_searched": 0, "n_search_hits": 0,
                       "selected_types": [], "n_clusters": 0,
                       "cluster_sizes": [], "cluster_labels": [],
                       "prompt_tokens": self._total_prompt_tokens,
                       "completion_tokens": self._total_completion_tokens},
            )

        # Phase B: Evidence assembly (< 2s)
        q_type = self._classify_question(question, entities)
        evidence, citations, contradictions, gaps = self._assemble_evidence(entities, q_type)
        t_b = time.monotonic()

        logger.info("Phase A: %.0fms (%d entities), Phase B: %.0fms (%s, %d chars evidence)",
                     (t_a - t_start) * 1000, len(entities),
                     (t_b - t_a) * 1000, q_type, len(evidence))

        # Phase C: LLM synthesis (< 8s)
        gap_note = ""
        if gaps:
            gap_note = (
                "\n\nIMPORTANT: The knowledge graph found gaps — "
                + "; ".join(gaps[:3])
                + ". If no bridging evidence was found, say so clearly. "
                "Do NOT infer connections from predicate summaries alone "
                "(e.g. 'both have inhibits relationships' is not evidence). "
                "State what IS known and what IS NOT."
            )
        prompt = (
            "You are answering questions about a knowledge graph with "
            f"{self.db._store.stats().get('total_claims', 0):,} claims from "
            "30+ curated sources. Below is evidence.\n\n"
            f"{evidence}\n\n"
            "---\n\n"
            f"Question: {question}{gap_note}\n\n"
            "Instructions:\n"
            "- If bridging entities were found, trace the multi-hop chain explicitly.\n"
            "- If no connection was found, say so — this is a knowledge gap, not a failure.\n"
            "- Distinguish direct evidence from inferences.\n"
            "- Answer in 3-8 sentences using specific entity names from the evidence."
        )
        answer = self._llm_call(prompt, max_tokens=1024)
        t_c = time.monotonic()

        logger.info("Phase C (LLM): %.0fms. Total: %.1fs",
                     (t_c - t_b) * 1000, t_c - t_start)

        # Build entity summaries for response
        entity_summaries = []
        for e in entities[:top_k]:
            entity_summaries.append(EntitySummary(
                id=e.entity_id, name=e.name,
                entity_type=e.entity_type,
                claim_count=e.claim_count,
            ))

        return AskResult(
            answer=answer,
            citations=citations,
            contradictions=contradictions,
            gaps=gaps,
            entities=entity_summaries,
            evidence=evidence,
            meta={
                "pipeline": "v2",
                "question_type": q_type,
                "phase_a_ms": int((t_a - t_start) * 1000),
                "phase_b_ms": int((t_b - t_a) * 1000),
                "phase_c_ms": int((t_c - t_b) * 1000),
                "total_ms": int((t_c - t_start) * 1000),
                "n_searched": len(entities),
                "n_search_hits": len(entities),
                "selected_types": sorted({e.entity_type for e in entities if e.entity_type}),
                "n_clusters": 0,
                "cluster_sizes": [],
                "cluster_labels": [],
                "entity_tiers": {e.entity_id: e.match_tier for e in entities},
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
            },
        )
        if self._ops_callback:
            try:
                self._ops_callback(
                    "ask_query",
                    question=question[:200],
                    entity_count=len(entities),
                    prompt_tokens=self._total_prompt_tokens,
                    completion_tokens=self._total_completion_tokens,
                    elapsed_ms=int((t_c - t_start) * 1000),
                )
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────
    # Legacy helpers (kept for tests and attest_db.py delegates)
    # ──────────────────────────────────────────────────────────────────

    _LARGE_ENTITY_THRESHOLD = 200
    _NEIGHBOR_CLAIMS_CAP = 200

    def _label_cluster(self, cluster: list[str], entity_map: dict[str, "EntitySummary"]) -> str:
        """Generate a human-readable label for a cluster of entities."""
        type_counts: dict[str, int] = {}
        for eid in cluster:
            e = entity_map.get(eid)
            if e:
                etype = e.entity_type or "unknown"
                type_counts[etype] = type_counts.get(etype, 0) + 1
        dominant = max(type_counts, key=type_counts.get) if type_counts else "unknown"
        ranked = sorted(
            (entity_map[eid] for eid in cluster if eid in entity_map),
            key=lambda e: -e.claim_count,
        )
        names = [e.name or e.id for e in ranked[:2]]
        if names:
            return f"{dominant} ({', '.join(names)})"
        return dominant

    def _cluster_entities(self, entity_ids: list[str]) -> list[list[str]]:
        """Cluster candidate entities by 2-hop graph connectivity."""
        if len(entity_ids) <= 1:
            return [list(entity_ids)] if entity_ids else []

        neighbors: dict[str, set[str]] = {}
        for eid in entity_ids:
            raw = _safe_claims_for(self.db._store, eid, None, None, 0.0, 200)
            adj: set[str] = set()
            for c in raw:
                if isinstance(c, dict):
                    s = c.get("subject", {}).get("id", "")
                    o = c.get("object", {}).get("id", "")
                    adj.add(o if s == eid else s)
            neighbors[eid] = adj

        cand_adj: dict[str, set[str]] = {eid: set() for eid in entity_ids}
        ids = list(entity_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if b in neighbors[a] or neighbors[a] & neighbors[b]:
                    cand_adj[a].add(b)
                    cand_adj[b].add(a)

        visited: set[str] = set()
        clusters: list[list[str]] = []
        for eid in entity_ids:
            if eid in visited:
                continue
            component: list[str] = []
            queue = deque([eid])
            visited.add(eid)
            while queue:
                node = queue.popleft()
                component.append(node)
                for nb in cand_adj[node]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            clusters.append(component)
        clusters.sort(key=lambda c: -len(c))
        return clusters

    def _gather_evidence(
        self,
        entity_ids: list[str],
        max_rels: int = 60,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build rich evidence from claims for a set of entities (legacy v1)."""
        from attestdb.core.vocabulary import KNOWLEDGE_PRIORITY, knowledge_label, OPPOSITE_PREDICATES

        lines: list[str] = []
        citations: list[Citation] = []
        all_contradictions: list[dict] = []
        all_gaps: list[str] = []
        seen: set[str] = set()
        seen_claim_ids: set[str] = set()
        per_entity = max(max_rels // max(len(entity_ids), 1), 5)

        for eid in entity_ids:
            entity = self.db._store.get_entity(eid)
            if not entity:
                continue
            ename = _entity_name(entity, eid)
            etype = entity.get("entity_type", "entity")
            claim_count = entity.get("claim_count", 0)
            lines.append(f"\n## {ename} ({etype}, {claim_count} claims)")

            budgeted = []

            if claim_count > self._LARGE_ENTITY_THRESHOLD and hasattr(self.db._store, 'entity_predicate_counts'):
                pred_counts = self.db._store.entity_predicate_counts(eid)
                if isinstance(pred_counts, list):
                    for pred, count in pred_counts[:per_entity]:
                        lines.append(f"- {ename} {pred} ({count} claims)")
                    if collect_citations:
                        for pred, count in pred_counts[:10]:
                            if len(citations) >= 50:
                                break
                            citations.append(Citation(
                                claim_id=f"summary:{eid}:{pred}",
                                subject=eid, predicate=pred,
                                object=f"{count} targets",
                                confidence=min(0.7, count / 100),
                                source_id="predicate_summary",
                                source_type="aggregate",
                            ))

            if not budgeted and claim_count <= self._LARGE_ENTITY_THRESHOLD:
                raw_claims = _safe_claims_for(self.db._store, eid, None, None, 0.0, 0)
                raw_claims = [d for d in raw_claims if isinstance(d, dict)]
                raw_claims.sort(key=lambda d: -d.get("confidence", 0))
                budgeted = [claim_from_dict(d) for d in raw_claims[:per_entity]]

            pred_targets: dict[str, list] = {}
            for c in budgeted:
                key = (c.subject.id, c.object.id)
                pred_targets.setdefault(key, []).append(c.predicate.id)

            for c in budgeted:
                subj = c.subject.display_name or c.subject.id
                obj = c.object.display_name or c.object.id
                triple = f"{subj} \u2192 {c.predicate.id} \u2192 {obj}"
                if triple in seen:
                    continue
                seen.add(triple)
                tag = ""
                if c.predicate.id in KNOWLEDGE_PRIORITY:
                    tag = f" \u26a0 {knowledge_label(c.predicate.id).upper()}"
                src = c.provenance.source_type if c.provenance else ""
                ann = f"[conf={c.confidence:.2f}, source: {src}]{tag}"
                lines.append(f"- {triple} {ann}")
                if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                    ev = c.payload.data.get("evidence_text", "")
                    if ev:
                        lines.append(f'    Evidence: "{ev}"')
                if collect_citations and len(citations) < 50:
                    if c.claim_id not in seen_claim_ids:
                        seen_claim_ids.add(c.claim_id)
                        citations.append(Citation(
                            claim_id=c.claim_id, subject=c.subject.id,
                            predicate=c.predicate.id, object=c.object.id,
                            confidence=c.confidence,
                            source_id=c.provenance.source_id if c.provenance else "",
                            source_type=c.provenance.source_type if c.provenance else "",
                        ))

            for (subj, obj), preds in pred_targets.items():
                pred_set = set(preds)
                for p1, p2 in OPPOSITE_PREDICATES.items():
                    if p1 in pred_set and p2 in pred_set:
                        desc = f"{subj} has both '{p1}' and '{p2}' relationship with {obj}"
                        lines.append(f"  \u26a0 Contradiction: {desc}")
                        if collect_citations:
                            all_contradictions.append({
                                "claim_a": "", "claim_b": "",
                                "description": desc, "status": "unresolved",
                            })

        if collect_citations:
            from attestdb.core.vocabulary import BUILT_IN_PREDICATE_TYPES
            for eid in entity_ids:
                entity = self.db._store.get_entity(eid)
                if not entity:
                    continue
                cc = entity.get("claim_count", 0)
                if cc > 0 and hasattr(self.db._store, 'entity_predicate_counts'):
                    preds = self.db._store.entity_predicate_counts(eid)
                    if isinstance(preds, list) and len(preds) <= 2 and cc > 10:
                        ename = _entity_name(entity, eid)
                        all_gaps.append(f"{ename} has {cc} claims but only {len(preds)} relationship types")

        evidence_text = "\n".join(lines)
        if collect_citations:
            return evidence_text, citations, all_contradictions, all_gaps
        return evidence_text

    def _gather_clustered_evidence(
        self,
        clusters: list[list[str]],
        entity_map: dict[str, "EntitySummary"],
        max_rels: int = 60,
        collect_citations: bool = False,
    ) -> str | tuple:
        """Build evidence organized by topic cluster (legacy v1)."""
        clusters = clusters[:5]
        total_entities = sum(len(c) for c in clusters)
        sections: list[str] = []
        all_citations: list[Citation] = []
        all_contradictions: list[dict] = []
        all_gaps: list[str] = []
        for cluster in clusters:
            budget = max(15, int(max_rels * len(cluster) / max(total_entities, 1)))
            if collect_citations:
                evidence, cit, contra, gaps = self._gather_evidence(
                    cluster, max_rels=budget, collect_citations=True,
                )
                all_citations.extend(cit)
                all_contradictions.extend(contra)
                for g in gaps:
                    if g not in all_gaps:
                        all_gaps.append(g)
            else:
                evidence = self._gather_evidence(cluster, max_rels=budget)
            if evidence.strip():
                sections.append(evidence)
        text = "\n\n".join(sections)
        if collect_citations:
            return text, all_citations, all_contradictions, all_gaps
        return text
