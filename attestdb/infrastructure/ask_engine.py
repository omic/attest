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
from attestdb.core.types import AnswerSegment, AskResult, Citation, EntitySummary, PathResult, claim_from_dict

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

    def __init__(self, db, ops_callback=None, entity_filter=None,
                 question_entity_filter=None):
        """
        entity_filter: optional Callable[[str], bool] — called with an entity
        name or id. Claims whose subject passes through _keep_claim are kept;
        used by demos to strip corpus-specific pseudonyms that otherwise leak.

        question_entity_filter: optional Callable[
            [list[ResolvedEntity], str], list[ResolvedEntity]
        ] — called once per ask() with the entities extracted from the
        question and the original question text. Returns the filtered list
        to use downstream. The ask engine does not prescribe how filtering
        happens (LLM classifier, heuristic, or explicit allow-list) — callers
        decide. Common use: drop group-noun anchors like "victims" or
        "employees and associates" from the question's entity list so the
        graph centers on the specific named entity being asked about.
        """
        self.db = db
        self._last_prompt_tokens: int = 0
        self._last_completion_tokens: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._ops_callback = ops_callback
        self._agent_engine = None
        self.entity_filter = entity_filter
        self.question_entity_filter = question_entity_filter

    def _keep_claim(self, c) -> bool:
        """True if the subject passes the optional entity_filter.

        We intentionally do NOT filter the object side: court-doc pseudonyms
        ("Jane", "Individual I") most matter as *subjects* of claims ("Jane
        said X"); as objects they are usually the referent of a real subject's
        claim ("Epstein abused Jane") and dropping them suppresses legitimate
        evidence. The object side also holds non-person identifiers that the
        gibberish filter mis-classifies — e.g. aircraft tail numbers ("N908JE"),
        docket IDs, and short codes — whose presence is essential for
        flight-log, registry, and case-docket claims to surface.
        """
        if self.entity_filter is None:
            return True
        name = getattr(c.subject, "display_name", None) or getattr(c.subject, "id", "")
        return bool(self.entity_filter(name))

    @staticmethod
    def _doc_root(source_id: str) -> str:
        """Strip chunk/page suffixes so EFTA-123#p4 and EFTA-123#p7 collapse."""
        if not source_id:
            return ""
        for sep in ("#", "?", ":p", ":c"):
            idx = source_id.find(sep)
            if idx > 0:
                return source_id[:idx]
        return source_id

    def _score_and_sort_citations(self, citations: list) -> list:
        """Rank citations by corroboration × predicate_weight × source_diversity.

        corroboration_count = distinct doc-ids per (subject, predicate, object)
        predicate_weight    = PREDICATE_META weight (fallback 0.3)
        source_diversity    = distinct source_types sharing the triple
        """
        try:
            from attestdb.core.predicate_salience import PREDICATE_META
        except Exception:
            PREDICATE_META = {}

        groups: dict[tuple[str, str, str], dict] = {}
        for c in citations:
            key = (c.subject or "", c.predicate or "", c.object or "")
            g = groups.setdefault(key, {"docs": set(), "types": set()})
            g["docs"].add(self._doc_root(c.source_id or ""))
            if c.source_type:
                g["types"].add(c.source_type)

        for c in citations:
            key = (c.subject or "", c.predicate or "", c.object or "")
            g = groups[key]
            corrob = max(1, len([d for d in g["docs"] if d]))
            diversity = max(1, len(g["types"]))
            meta = PREDICATE_META.get(c.predicate)
            p_weight = (meta[1] if isinstance(meta, tuple) and len(meta) >= 2 else 0.3) or 0.3
            c.corroboration_count = corrob
            c.score = corrob * p_weight * diversity

        citations.sort(key=lambda x: x.score, reverse=True)
        return citations

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
            from attestdb.intelligence.llm_client import get_llm_client
            client, model = get_llm_client()
            if client is not None:
                logger.info("_get_llm_client: initialized via intelligence layer (model=%s)", model)
                return client, model
        except ImportError:
            logger.debug("_get_llm_client: intelligence layer not available")

        try:
            ext = self.db._get_text_extractor()
            if ext._client and ext._llm_model:
                return ext._client, ext._llm_model
        except Exception as exc:
            logger.debug("_get_llm_client: text extractor unavailable: %s", exc)

        return None, None

    def _parse_answer_envelope(self, raw: str | None) -> tuple[str | None, list[AnswerSegment]]:
        """Parse the LLM's JSON envelope into (prose answer, structured segments).

        Boundary visibility: if parsing succeeds, every span is tagged verified vs
        synthesized. If parsing fails, we degrade honestly — the entire prose
        response is reported as a single ``synthesized`` segment with no claim_ids,
        signaling to callers that we could not establish provenance.
        """
        if not raw:
            return None, []
        text = raw.strip()
        # Strip ```json ... ``` fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2:
                text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        # Locate the outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return raw, [AnswerSegment(kind="synthesized", text=raw, confidence=0.0)]
        try:
            envelope = json.loads(text[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return raw, [AnswerSegment(kind="synthesized", text=raw, confidence=0.0)]

        answer = envelope.get("answer")
        if not isinstance(answer, str):
            answer = raw

        segments: list[AnswerSegment] = []
        raw_segments = envelope.get("segments", [])
        if isinstance(raw_segments, list):
            for s in raw_segments:
                if not isinstance(s, dict):
                    continue
                kind = s.get("kind", "synthesized")
                if kind not in ("verified", "synthesized"):
                    kind = "synthesized"
                seg_text = s.get("text", "")
                if not isinstance(seg_text, str) or not seg_text.strip():
                    continue
                claim_ids = s.get("claim_ids", []) or []
                if not isinstance(claim_ids, list):
                    claim_ids = []
                claim_ids = [str(c) for c in claim_ids if c]
                try:
                    confidence = float(s.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                # If model marked something as verified but produced no claim_ids,
                # downgrade to synthesized — verified MUST cite a claim.
                if kind == "verified" and not claim_ids:
                    kind = "synthesized"
                segments.append(AnswerSegment(
                    kind=kind, text=seg_text.strip(),
                    claim_ids=claim_ids, confidence=confidence,
                ))

        if not segments:
            segments = [AnswerSegment(kind="synthesized", text=answer, confidence=0.0)]
        return answer, segments

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
            "Extract specific named entities from this question. Entity types include: "
            "genes, proteins, diseases, drugs, compounds, organisms, pathways, "
            "companies, people, locations, events, activities, and measurable values. "
            "If the question uses first-person language (I, my, me), include "
            '["user", "person"] in the output. '
            "Return ONLY a JSON list of [name, type] pairs. No explanation.\n\n"
            f"Question: {question}\n\n"
            'Example: [["KRAS", "gene"], ["heart disease", "disease"], '
            '["user", "person"], ["San Francisco", "location"]]'
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

    # First-person words that imply the question is about the "user" entity.
    # The extraction pipeline resolves "I"→"user" at ingest time; the ask
    # engine mirrors this so the same entity is found at query time.
    _FIRST_PERSON = frozenset({
        "i", "my", "me", "myself", "mine",
        "i'm", "i've", "i'd", "i'll",
    })

    def _extract_question_entities(self, question: str, top_k: int = 10) -> list[ResolvedEntity]:
        """Master entity extraction: Tier 1 (exact) → Tier 2 (BM25) → Tier 3 (LLM).

        Returns resolved entities sorted by match quality and claim count.
        """
        t0 = time.monotonic()
        candidates = self._generate_candidates(question)

        # If the question uses first-person language, add "user" as a
        # candidate.  The extraction pipeline resolves "I" → entity "user"
        # (type: person) at ingest time; we do the same at query time so
        # the graph lookup succeeds.  This is data-driven: if "user" has
        # no claims in the graph, _resolve_entity returns None and it's a
        # no-op.
        q_words = {w.strip("?.,!\"'()[]{}:;").lower() for w in question.split()}
        if q_words & self._FIRST_PERSON:
            candidates.insert(0, "user")

        resolved: dict[str, ResolvedEntity] = {}

        # Tier 1+2: Try each candidate with exact match then BM25
        # TODO: aggregate/group-noun candidates ("victims", "employees and
        # associates") can resolve to real aggregate nodes in some corpora and
        # mis-anchor downstream views (graph centering, cross-entity scoring).
        # Filtering belongs at a semantic layer, not as a hardcoded token list:
        # either an LLM classifier pass over resolved candidates, or a
        # data-driven signal (e.g. entity has no external_ids, name is a
        # plural common noun). Leaving this to a follow-up — do not reintroduce
        # a hardcoded domain-specific set here.
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
        result = result[:top_k]

        # Optional caller-provided post-filter. Runs once per question and
        # can prune low-value anchors (e.g. group nouns like "victims" or
        # "employees and associates") that resolve but shouldn't drive
        # downstream evidence or graph centering. The filter is opaque to
        # the engine — callers can use an LLM, heuristic, or anything else.
        if self.question_entity_filter is not None and result:
            try:
                filtered = self.question_entity_filter(result, question)
                if isinstance(filtered, list):
                    # Keep at least one entity so single-entity paths still fire.
                    result = filtered if filtered else result
            except Exception:
                logger.exception("question_entity_filter raised; keeping unfiltered result")
        return result

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

    def _evidence_single(
        self, entity: ResolvedEntity, question: str = "",
    ) -> tuple[str, list[Citation]]:
        """Evidence for single-entity questions: predicate summary + top claims.

        When ``question`` is provided and the entity has many claims, claims
        are ranked by relevance to the question keywords so that the most
        pertinent evidence surfaces instead of arbitrary top-N.
        """
        lines = [f"## {entity.name} ({entity.entity_type}, {entity.claim_count} claims)"]
        citations: list[Citation] = []

        # Predicate summary (instant — no claim materialization)
        if hasattr(self.db._store, 'entity_predicate_counts'):
            pred_counts = self.db._store.entity_predicate_counts(entity.entity_id)
            if isinstance(pred_counts, list) and pred_counts:
                lines.append("Relationship types:")
                for pred, count in pred_counts[:12]:
                    lines.append(f"  - {pred}: {count} claims")

        # Build question keywords for relevance filtering
        q_keywords: set[str] = set()
        if question:
            q_keywords = {
                w.strip("?.,!\"'()[]{}:;").lower()
                for w in question.split()
                if len(w.strip("?.,!\"'()[]{}:;")) >= 3
                and w.strip("?.,!\"'()[]{}:;").lower() not in _STOP
            }

        # Fetch claims — use a larger window whenever we have keyword context
        # so relevance filtering can surface answer-bearing claims that happen
        # to sit past position 30 in the Rust store's default iteration order.
        fetch_limit = 200 if q_keywords else 30
        raw_claims = _safe_claims_for(self.db._store, entity.entity_id, None, None, 0.3, fetch_limit)

        # High-degree predicate-targeted fetch: for entities with many total claims
        # (e.g. an aircraft tail number with 1.5k claims), the default fetch may
        # miss the specific predicate the question is about. If any question
        # keyword matches a predicate name in the entity's predicate_counts, pull
        # an extra bounded sample filtered by that predicate so the answer-bearing
        # claims can't be starved by the limit.
        extra_preds_used: list[str] = []
        if (q_keywords
                and entity.claim_count > 200
                and hasattr(self.db._store, 'entity_predicate_counts')):
            pc = self.db._store.entity_predicate_counts(entity.entity_id) or []
            for pred, _count in pc[:30]:
                ptext = pred.replace("_", " ").lower()
                if any(kw in ptext for kw in q_keywords):
                    extra = _safe_claims_for(
                        self.db._store, entity.entity_id, pred, None, 0.3, 60
                    )
                    if extra:
                        raw_claims = (raw_claims or []) + extra
                        extra_preds_used.append(pred)
                if len(extra_preds_used) >= 3:
                    break

        if raw_claims:
            # Parse claims and detect temporal updates: same (subject, predicate)
            # with different objects across sources → the latest supersedes.
            parsed = []
            for d in raw_claims:
                if not isinstance(d, dict):
                    continue
                parsed.append(claim_from_dict(d))
            # De-dupe by claim_id (predicate-targeted fetch can overlap general fetch)
            _seen_cids: set[str] = set()
            _uniq = []
            for c in parsed:
                if c.claim_id in _seen_cids:
                    continue
                _seen_cids.add(c.claim_id)
                _uniq.append(c)
            parsed = _uniq
            # Strip pseudonyms / noise entities from evidence if a filter
            # is configured. Without this, court-doc placeholders like
            # "jane" or "individual ii" leak into the LLM answer as if
            # they were real named people.
            parsed = [c for c in parsed if self._keep_claim(c)]

            # Relevance ranking: when the entity has many claims and we have
            # question keywords, score each claim by keyword overlap with its
            # predicate, object name, and evidence text.  This ensures the
            # answer-bearing claims surface even on high-degree entities.
            if q_keywords and len(parsed) > 30:
                def _relevance(c):
                    text = " ".join([
                        c.predicate.id.replace("_", " "),
                        c.object.display_name or c.object.id,
                        (c.payload.data.get("evidence_text", "")
                         if c.payload and hasattr(c.payload, "data")
                         and isinstance(c.payload.data, dict) else ""),
                    ]).lower()
                    return sum(1 for kw in q_keywords if kw in text)
                parsed.sort(key=lambda c: (-_relevance(c), -(c.confidence or 0)))

            # Source-diversity interleave: after ranking, re-order so the top
            # claim from each source_type appears before the second-best from any
            # single source_type. Prevents a flood of high-confidence DOJ
            # extractions from crowding out lower-confidence but topically
            # relevant claims (e.g. compiled_flight_log at conf=0.9 losing to
            # DOJ at 0.95). Preserves within-source_type ordering from the
            # relevance sort above.
            if len(parsed) > 10:
                by_stype: dict[str, list] = {}
                for c in parsed:
                    st = c.provenance.source_type if c.provenance else ""
                    by_stype.setdefault(st, []).append(c)
                interleaved = []
                while any(by_stype.values()):
                    for st in list(by_stype.keys()):
                        if by_stype[st]:
                            interleaved.append(by_stype[st].pop(0))
                parsed = interleaved

            # Group by (subject_id, predicate_id) to detect value updates
            by_sp: dict[tuple[str, str], list] = {}
            for c in parsed:
                key = (c.subject.id, c.predicate.id)
                by_sp.setdefault(key, []).append(c)

            # For groups with multiple different objects, mark superseded claims
            superseded_ids: set[str] = set()
            for key, group in by_sp.items():
                objects = {c.object.id for c in group}
                if len(objects) > 1 and len(group) > 1:
                    # Sort by timestamp (ascending) — latest wins
                    sorted_group = sorted(group, key=lambda c: c.timestamp or 0)
                    for older in sorted_group[:-1]:
                        superseded_ids.add(older.claim_id)

            lines.append("\nTop relationships:")
            seen = set()
            for c in parsed:
                subj = c.subject.display_name or c.subject.id
                obj = c.object.display_name or c.object.id
                triple = f"{subj} {c.predicate.id} {obj}"
                if triple in seen:
                    continue
                seen.add(triple)
                src = c.provenance.source_type if c.provenance else ""
                src_id = c.provenance.source_id if c.provenance else ""
                ts = str(c.timestamp) if c.timestamp else ""
                source_info = f"source: {src}"
                if src_id:
                    source_info += f", id: {src_id}"
                if ts:
                    source_info += f", time: {ts}"
                # Annotate superseded claims so the LLM knows which value is current
                if c.claim_id in superseded_ids:
                    lines.append(f"  - [SUPERSEDED] {triple} [conf={c.confidence:.2f}, {source_info}]")
                else:
                    lines.append(f"  - {triple} [conf={c.confidence:.2f}, {source_info}]")
                # Evidence text from payload
                if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                    ev = c.payload.data.get("evidence_text", "")
                    if ev:
                        lines.append(f'    Evidence: "{ev}"')
                if len(citations) < 50:
                    ev_text = ""
                    if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                        ev_text = c.payload.data.get("evidence_text", "") or ""
                    citations.append(Citation(
                        claim_id=c.claim_id, subject=c.subject.id,
                        predicate=c.predicate.id, object=c.object.id,
                        confidence=c.confidence,
                        source_id=c.provenance.source_id,
                        source_type=c.provenance.source_type,
                        evidence_text=ev_text,
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
                lines.append(
                    f"\n## Shared neighbors (structural hints — NOT sourced claims): "
                    f"{a.name} ↔ {b.name} ({len(scored)} shared, {n_specific} with specific predicates). "
                    f"These are graph-structure connections only; do NOT state any composed relationship "
                    f"(e.g. '{a.name} is-X of {b.name}') as fact unless a verified claim above says so."
                )

                # Tighten bridge selection: require a meaningful score and
                # reject pairs where both legs share the same predicate
                # (self-duplicates usually come from bridge entities whose
                # top-2 predicates are dominated by a single relation, which
                # produces nonsense like "X traveled_to M traveled_to Y").
                # Threshold 0.09 = both legs at least default weight (0.3 × 0.3)
                # OR one leg with weight ≥ 0.45. Filter before trimming to 5.
                BRIDGE_MIN_SCORE = 0.09
                filtered_scored = [
                    s for s in scored
                    if s[3] >= BRIDGE_MIN_SCORE and s[4] != s[5]
                ]
                for mid_id, mid_name, mid_cc, score, pred_a, pred_b, composed in filtered_scored[:5]:
                    # Drop the "composed → inferred" tag entirely. Previously we appended a
                    # speculative composite predicate (e.g. "is_attorney_of"), which the LLM
                    # treated as a real fact and hallucinated sentences like "Maxwell is
                    # attorney for Trump." The raw two-hop path is enough context; composition
                    # was a footgun.
                    lines.append(
                        f"  [structural] {a.name} --[{pred_a}]--> {mid_name} "
                        f"--[{pred_b}]--> {b.name}  (score={score:.2f}, {mid_cc} claims)"
                    )
                    if len(citations) < 48:
                        citations.append(Citation(
                            claim_id=f"bridge:{a.entity_id}:{mid_id}:leg1",
                            subject=a.entity_id, predicate=pred_a,
                            object=mid_id, confidence=score,
                            source_id="bridge", source_type="causal_composition",
                            kind="inferred_path",
                        ))
                        citations.append(Citation(
                            claim_id=f"bridge:{mid_id}:{b.entity_id}:leg2",
                            subject=mid_id, predicate=pred_b,
                            object=b.entity_id, confidence=score,
                            source_id="bridge", source_type="causal_composition",
                            kind="inferred_path",
                        ))

        # Add predicate summaries + claim materialization for each entity.
        # Low-degree entities (<=500 claims) get full claim details so the LLM
        # can see actual values (e.g., team_size=4 vs team_size=5).
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
                                kind="predicate_summary",
                            ))

            # Materialize claims for low-degree entities so the LLM sees actual values.
            # Bumped from 30 to 200 so answer-bearing claims (e.g. flight-log entries)
            # don't get truncated by default Rust-store iteration order.
            if entity.claim_count <= 500:
                raw_claims = _safe_claims_for(self.db._store, entity.entity_id, None, None, 0.3, 200)
                if raw_claims:
                    parsed = []
                    for d in raw_claims:
                        if not isinstance(d, dict):
                            continue
                        parsed.append(claim_from_dict(d))
                    parsed = [c for c in parsed if self._keep_claim(c)]

                    # Cross-entity priority: for relationship questions (N≥2 query
                    # entities), boost claims whose OTHER endpoint is one of the
                    # other query entities, or shares a last-name/display-name token
                    # with one. Previously "how is Trump connected to the victims"
                    # surfaced 170 generic procedural claims about "victims"
                    # (Boies-Schiller-represents-victims etc.) while Trump's own
                    # claims that touch Epstein/Maxwell/Jane drowned. With this
                    # boost, direct cross-entity claims sort to the top.
                    other_query_ids = {
                        e.entity_id for e in entities if e.entity_id != entity.entity_id
                    }
                    if other_query_ids:
                        def _cross_entity_score(c) -> int:
                            other = c.object.id if c.subject.id == entity.entity_id else c.subject.id
                            return 1 if other in other_query_ids else 0
                        parsed.sort(
                            key=lambda c: (-_cross_entity_score(c), -(c.confidence or 0))
                        )

                    # Detect temporal updates: same (subject, predicate) with different objects
                    by_sp: dict[tuple[str, str], list] = {}
                    for c in parsed:
                        key = (c.subject.id, c.predicate.id)
                        by_sp.setdefault(key, []).append(c)
                    superseded_ids: set[str] = set()
                    for key, group in by_sp.items():
                        objects = {c.object.id for c in group}
                        if len(objects) > 1 and len(group) > 1:
                            sorted_group = sorted(group, key=lambda c: c.timestamp or 0)
                            for older in sorted_group[:-1]:
                                superseded_ids.add(older.claim_id)

                    lines.append(f"\n  Top claims for {entity.name}:")
                    seen = set()
                    for c in parsed:
                        subj = c.subject.display_name or c.subject.id
                        obj = c.object.display_name or c.object.id
                        triple = f"{subj} {c.predicate.id} {obj}"
                        if triple in seen:
                            continue
                        seen.add(triple)
                        src = c.provenance.source_type if c.provenance else ""
                        src_id = c.provenance.source_id if c.provenance else ""
                        ts = str(c.timestamp) if c.timestamp else ""
                        source_info = f"source: {src}"
                        if src_id:
                            source_info += f", id: {src_id}"
                        if ts:
                            source_info += f", time: {ts}"
                        if c.claim_id in superseded_ids:
                            lines.append(f"  - [SUPERSEDED] {triple} [conf={c.confidence:.2f}, {source_info}]")
                        else:
                            lines.append(f"  - {triple} [conf={c.confidence:.2f}, {source_info}]")
                        if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                            ev = c.payload.data.get("evidence_text", "")
                            if ev:
                                lines.append(f'    Evidence: "{ev}"')
                        if len(citations) < 50:
                            ev_text = ""
                            if c.payload and hasattr(c.payload, 'data') and isinstance(c.payload.data, dict):
                                ev_text = c.payload.data.get("evidence_text", "") or ""
                            citations.append(Citation(
                                claim_id=c.claim_id, subject=c.subject.id,
                                predicate=c.predicate.id, object=c.object.id,
                                confidence=c.confidence,
                                source_id=c.provenance.source_id if c.provenance else "",
                                source_type=c.provenance.source_type if c.provenance else "",
                                evidence_text=ev_text,
                            ))
                        if len(seen) >= 15:
                            break

        return "\n".join(lines), citations, contradictions, gaps

    def _evidence_exploratory(self, entity: ResolvedEntity, question: str = "") -> tuple[str, list[Citation]]:
        """Evidence for exploratory questions: BFS depth-1 + full summary."""
        # Reuse single-entity evidence (it's already comprehensive)
        return self._evidence_single(entity, question=question)

    def _assemble_evidence(
        self, entities: list[ResolvedEntity], question_type: str,
        question: str = "",
    ) -> tuple[str, list[Citation], list[dict], list[str]]:
        """Dispatch to appropriate evidence strategy."""
        if question_type == "relationship" and len(entities) >= 2:
            return self._evidence_relationship(entities)
        elif question_type == "single" and entities:
            text, cites = self._evidence_single(entities[0], question=question)
            return text, cites, [], []
        elif entities:
            # For exploratory questions with multiple entities, gather
            # evidence from ALL resolved entities, not just the first.
            all_text: list[str] = []
            all_cites: list[Citation] = []
            for ent in entities[:5]:
                t, c = self._evidence_exploratory(ent, question=question)
                all_text.append(t)
                all_cites.extend(c)
            return "\n\n".join(all_text), all_cites, [], []
        return "", [], [], []

    # ──────────────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────────────

    def ask(self, question: str, top_k: int = 10, engine: str = "v2") -> AskResult:
        """Answer a natural-language question using the knowledge graph.

        Args:
            question: Natural-language question.
            top_k: Maximum entities to consider.
            engine: Pipeline to use — "v2" (default), "v3" (Agent SDK), or
                    "shadow" (run both, return v2 with v3 comparison in meta).

        Returns:
            AskResult with structured citations, contradictions, and gaps.
        """
        if engine == "v3":
            return self._ask_v3(question, top_k)
        if engine == "shadow":
            return self._ask_shadow(question, top_k)

        return self._ask_v2(question, top_k)

    def _ask_v2(self, question: str, top_k: int = 10) -> AskResult:
        """V2 pipeline: entity extraction → graph evidence → LLM synthesis."""
        t_start = time.monotonic()
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0

        # Phase A: Entity extraction (< 500ms)
        entities = self._extract_question_entities(question, top_k=top_k)
        t_a = time.monotonic()

        if not entities:
            try:
                from attestdb.infrastructure.unanswered_log import log_unanswered
                log_unanswered(
                    self.db, question=question, reason="no_entities",
                    entities=[], pipeline="v2",
                )
            except Exception:
                pass
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
        evidence, citations, contradictions, gaps = self._assemble_evidence(entities, q_type, question=question)
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
            "30+ curated sources. Below is evidence. Each evidence line ends with "
            "[conf=…, source: …]; claim lines also begin with a triple that maps to "
            "the citations array passed back to the caller.\n\n"
            f"{evidence}\n\n"
            "---\n\n"
            f"Question: {question}{gap_note}\n\n"
            "Rules for the answer prose:\n"
            "1. Open with the single most specific, best-corroborated claim. "
            "   Name who, what, when, source. Do NOT open with generic framing "
            "   (\"The record shows several connections...\").\n"
            "2. For relationship questions, list the strongest paths as a short "
            "   bulleted enumeration — one bullet per distinct connection.\n"
            "3. Use exact phrases from the evidence text when available.\n"
            "4. FORBIDDEN hedges — do not use any of these: "
            "\"it can be inferred\", \"the exact nature is not fully specified\", "
            "\"suggests a connection\", \"appears to be associated\", \"may have\", "
            "\"it is worth noting\", \"the record indicates a relationship\". "
            "If a claim is an inference rather than a direct citation, mark its "
            "segment as kind=\"synthesized\" (see output format) and state plainly: "
            "\"Inference (not directly stated): ...\".\n"
            "4a. NO FABRICATION. Every relationship you state (X did Y to Z) MUST be "
            "supported by at least one explicit evidence line above. Do NOT invent "
            "roles (e.g. calling someone an \"attorney\" or \"business partner\" if "
            "no evidence line uses that word for that person). Do NOT conflate "
            "adjacent people (if A is X's lawyer and B met X, do not write that B is "
            "X's lawyer). If the evidence does not state a relationship, write "
            "\"no evidence in the graph of <relation>\" — never paper over a gap.\n"
            "4b. \"synthesized\" segments are ONLY for summarizing or aggregating "
            "multiple evidence lines that say similar things. They are NOT a license "
            "to introduce new relationships absent from the evidence. Every "
            "synthesized segment must cite ≥2 claim_ids from the evidence above.\n"
            "5. If multiple sources support the same claim, say so (e.g. \"3 sources\").\n"
            "6. When multiple claims share the same subject and predicate but "
            "different objects, the most recent supersedes earlier ones.\n"
            "7. If no connection is found, say so in one sentence. Do not pad.\n"
            "8. Keep the whole answer under 8 sentences (or bullets).\n\n"
            "OUTPUT FORMAT — return a single JSON object, nothing else:\n"
            "{\n"
            '  "answer": "<the prose answer following rules 1-8>",\n'
            '  "segments": [\n'
            '    {"kind": "verified",    "text": "<exact span quoted/paraphrased '
            'from a specific evidence claim>", "claim_ids": ["<one or more triples '
            'or claim_ids from the evidence>"], "confidence": 0.0-1.0},\n'
            '    {"kind": "synthesized", "text": "<an inference or aggregation you '
            'made across claims>", "claim_ids": ["<the claims you reasoned over, if '
            'any>"], "confidence": 0.0-1.0}\n'
            "  ]\n"
            "}\n"
            "Every span of the answer that is not a direct claim quote MUST be "
            "kind=\"synthesized\". When in doubt, prefer \"synthesized\". "
            "Use the triples (\"subj pred obj\") shown in the evidence as claim_ids "
            "if you do not have a hash-form claim_id."
        )
        raw = self._llm_call(prompt, max_tokens=1536)
        answer, segments = self._parse_answer_envelope(raw)
        t_c = time.monotonic()

        citations = self._score_and_sort_citations(citations)

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

        try:
            from attestdb.calibration.synthesis_log import log_synthesis_output
            verified = [s for s in segments if s.kind == "verified"]
            if segments:
                agg_conf = sum(s.confidence for s in segments) / len(segments)
            else:
                agg_conf = 0.0
            output_id = log_synthesis_output(
                self.db,
                source_id="ask_engine.v2",
                confidence=agg_conf,
                payload={
                    "kind": "ask",
                    "question": question[:500],
                    "n_segments": len(segments),
                    "n_verified": len(verified),
                    "n_citations": len(citations),
                    "n_gaps": len(gaps),
                    "answer_preview": (answer or "")[:240],
                },
            )
        except Exception:
            output_id = None
            agg_conf = 0.0
            verified = []

        # Demand-driven gap signal: log questions we couldn't answer well.
        # Distinct from blindspots() (structural). Failed compositions =
        # roadmap (Block).
        try:
            from attestdb.infrastructure.unanswered_log import log_unanswered
            entity_ids = [e.entity_id for e in entities[:top_k]]
            unanswered_reason = None
            if not answer or not answer.strip():
                unanswered_reason = "no_answer"
            elif not citations:
                unanswered_reason = "no_evidence"
            elif not verified and agg_conf < 0.4:
                # Pure-synthesis with low confidence — interpretation, not knowledge.
                unanswered_reason = "low_confidence"
            if unanswered_reason:
                log_unanswered(
                    self.db, question=question, reason=unanswered_reason,
                    entities=entity_ids, confidence=agg_conf,
                    n_citations=len(citations), n_gaps=len(gaps),
                    pipeline="v2",
                )
        except Exception:
            pass

        result = AskResult(
            answer=answer,
            citations=citations,
            contradictions=contradictions,
            gaps=gaps,
            entities=entity_summaries,
            evidence=evidence,
            segments=segments,
            meta={
                "pipeline": "v2",
                "output_id": output_id,
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
        self._fire_ops_callback(question, len(entities),
                                int((t_c - t_start) * 1000), "v2")
        return result

    # ── V3: Agent SDK investigation mode ──────────────────────────────

    def _ask_v3(self, question: str, top_k: int = 10) -> AskResult:
        """V3 pipeline: autonomous agent investigation via Anthropic tool-use."""
        if self._agent_engine is None:
            from attestdb.intelligence.ask_agent_engine import AskAgentEngine
            self._agent_engine = AskAgentEngine(self.db)

        agent_result = self._agent_engine.run(question)

        # Extract citations from claims_for tool calls
        citations = self._extract_citations_from_log(agent_result.tool_calls)

        # Extract gaps from find_gaps tool calls
        gaps = self._extract_gaps_from_log(agent_result.tool_calls)

        # Build tool sequence summary for ops log
        tool_sequence = [tc.tool for tc in agent_result.tool_calls]

        result = AskResult(
            answer=agent_result.answer,
            citations=citations,
            gaps=gaps,
            meta={
                "pipeline": "v3",
                "total_ms": agent_result.total_ms,
                "agent_tool_calls": len(agent_result.tool_calls),
                "agent_tool_sequence": tool_sequence,
                "agent_turns": agent_result.turns,
                "agent_model": agent_result.model,
                "agent_actual_model": agent_result.actual_model,
                "prompt_tokens": agent_result.prompt_tokens,
                "completion_tokens": agent_result.completion_tokens,
                "cache_creation_tokens": agent_result.cache_creation_tokens,
                "cache_read_tokens": agent_result.cache_read_tokens,
                "cost_usd": agent_result.cost_usd,
                "cost_limit_hit": agent_result.cost_limit_hit,
                "tool_limit_hit": agent_result.tool_limit_hit,
                "stop_reason": agent_result.stop_reason,
                "request_ids": agent_result.request_ids,
                "service_tier": agent_result.service_tier,
            },
        )
        self._fire_ops_callback(
            question=question,
            entity_count=0,
            elapsed_ms=agent_result.total_ms,
            pipeline="v3",
            prompt_tokens=agent_result.prompt_tokens,
            completion_tokens=agent_result.completion_tokens,
            cost_usd=agent_result.cost_usd,
            model=agent_result.actual_model or agent_result.model,
            tool_sequence=tool_sequence,
        )
        return result

    def _ask_shadow(self, question: str, top_k: int = 10) -> AskResult:
        """Run both v2 and v3, return v2 result with v3 summary in meta."""
        v2_result = self._ask_v2(question, top_k)
        try:
            v3_result = self._ask_v3(question, top_k)
            v2_result.meta["shadow_v3"] = {
                "answer": v3_result.answer,
                "citations_count": len(v3_result.citations),
                "tool_calls": v3_result.meta.get("agent_tool_calls", 0),
                "total_ms": v3_result.meta.get("total_ms", 0),
                "prompt_tokens": v3_result.meta.get("prompt_tokens", 0),
                "completion_tokens": v3_result.meta.get("completion_tokens", 0),
                "cost_usd": v3_result.meta.get("cost_usd", 0.0),
            }
        except Exception as e:
            v2_result.meta["shadow_v3"] = {"error": str(e)}
        return v2_result

    @staticmethod
    def _extract_citations_from_log(tool_calls) -> list[Citation]:
        """Extract Citation objects from claims_for tool call results."""
        citations = []
        for tc in tool_calls:
            if tc.tool != "claims_for":
                continue
            try:
                claims = json.loads(tc.output)
                if not isinstance(claims, list):
                    continue
                for c in claims:
                    citations.append(Citation(
                        claim_id=c.get("claim_id", ""),
                        subject=c.get("subject", ""),
                        predicate=c.get("predicate", ""),
                        object=c.get("object", ""),
                        confidence=c.get("confidence", 0.0),
                        source_id=c.get("source_id", ""),
                        source_type=c.get("source_type", ""),
                    ))
            except (json.JSONDecodeError, TypeError):
                continue
        return citations

    @staticmethod
    def _extract_gaps_from_log(tool_calls) -> list[str]:
        """Extract gap descriptions from find_gaps tool call results."""
        gaps = []
        for tc in tool_calls:
            if tc.tool != "find_gaps":
                continue
            try:
                result = json.loads(tc.output)
                if isinstance(result, list):
                    for g in result:
                        if isinstance(g, str):
                            gaps.append(g)
                        elif isinstance(g, dict):
                            gaps.append(str(g))
            except (json.JSONDecodeError, TypeError):
                continue
        return gaps

    def _fire_ops_callback(self, question: str, entity_count: int,
                           elapsed_ms: int, pipeline: str, *,
                           prompt_tokens: int | None = None,
                           completion_tokens: int | None = None,
                           cost_usd: float | None = None,
                           model: str = "",
                           tool_sequence: list[str] | None = None):
        """Fire ops callback if configured.

        For v2, prompt/completion tokens default to self._total_* counters.
        For v3, callers pass them explicitly from AgentRunResult.
        """
        if self._ops_callback:
            p_tok = prompt_tokens if prompt_tokens is not None else self._total_prompt_tokens
            c_tok = completion_tokens if completion_tokens is not None else self._total_completion_tokens
            if cost_usd is None and (p_tok or c_tok):
                from attestdb.core.providers import estimate_cost as _est
                cost_usd = _est(model or "unknown", p_tok, c_tok)
            try:
                self._ops_callback(
                    "ask_query",
                    question=question[:200],
                    entity_count=entity_count,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    cost_usd=cost_usd or 0.0,
                    model=model,
                    elapsed_ms=elapsed_ms,
                    pipeline=pipeline,
                    tool_sequence=tool_sequence or [],
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
                                kind="predicate_summary",
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
