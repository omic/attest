"""Entity matchers — exact ID, fuzzy name, domain-based, and AI-assisted matching."""

from __future__ import annotations

import difflib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Entity-match-specific cold start defaults (different from global calibration defaults)
_ENTITY_MATCH_AUTO_APPROVE = 0.85
_ENTITY_MATCH_REVIEW = 0.70


@dataclass
class MatchCandidate:
    entity_a: str  # source_id:record_id
    entity_b: str
    confidence: float
    match_method: str  # "exact_id" | "fuzzy_name" | "domain" | "ai"
    reasoning: str


class ExactMatcher:
    """Match records across sources on shared external IDs (DUNS, tax_id, etc.)."""

    def match(
        self,
        records_a: list[dict],
        records_b: list[dict],
        id_fields: list[str],
    ) -> list[MatchCandidate]:
        matches: list[MatchCandidate] = []

        # Build index: field_name -> field_value -> list of (source_id, record_id)
        index_b: dict[str, dict[str, list[dict]]] = {}
        for field_name in id_fields:
            index_b[field_name] = {}
            for rec in records_b:
                val = rec.get(field_name)
                if val is not None and str(val).strip():
                    key = str(val).strip().lower()
                    index_b[field_name].setdefault(key, []).append(rec)

        seen_pairs: set[tuple[str, str]] = set()

        for rec_a in records_a:
            source_a = rec_a.get("_source_id", "unknown")
            rid_a = rec_a.get("_record_id", "unknown")
            key_a = f"{source_a}:{rid_a}"

            for field_name in id_fields:
                val_a = rec_a.get(field_name)
                if val_a is None or not str(val_a).strip():
                    continue
                lookup = str(val_a).strip().lower()

                for rec_b in index_b.get(field_name, {}).get(lookup, []):
                    source_b = rec_b.get("_source_id", "unknown")
                    rid_b = rec_b.get("_record_id", "unknown")
                    key_b = f"{source_b}:{rid_b}"

                    if key_a == key_b:
                        continue

                    pair = (min(key_a, key_b), max(key_a, key_b))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    matches.append(
                        MatchCandidate(
                            entity_a=key_a,
                            entity_b=key_b,
                            confidence=1.0,
                            match_method="exact_id",
                            reasoning=f"Exact match on {field_name}={val_a}",
                        )
                    )

        return matches


class FuzzyNameMatcher:
    """Match records by fuzzy name similarity using difflib."""

    def match(
        self,
        records_a: list[dict],
        records_b: list[dict],
        name_field: str = "name",
        threshold: float = 0.85,
    ) -> list[MatchCandidate]:
        matches: list[MatchCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        for rec_a in records_a:
            name_a = rec_a.get(name_field)
            if not name_a or not str(name_a).strip():
                continue
            name_a = str(name_a).strip()

            source_a = rec_a.get("_source_id", "unknown")
            rid_a = rec_a.get("_record_id", "unknown")
            key_a = f"{source_a}:{rid_a}"

            for rec_b in records_b:
                name_b = rec_b.get(name_field)
                if not name_b or not str(name_b).strip():
                    continue
                name_b = str(name_b).strip()

                source_b = rec_b.get("_source_id", "unknown")
                rid_b = rec_b.get("_record_id", "unknown")
                key_b = f"{source_b}:{rid_b}"

                if key_a == key_b:
                    continue

                pair = (min(key_a, key_b), max(key_a, key_b))
                if pair in seen_pairs:
                    continue

                confidence = self._compute_similarity(name_a, name_b)
                if confidence >= threshold:
                    seen_pairs.add(pair)
                    matches.append(
                        MatchCandidate(
                            entity_a=key_a,
                            entity_b=key_b,
                            confidence=confidence,
                            match_method="fuzzy_name",
                            reasoning=f"Fuzzy name match: '{name_a}' ~ '{name_b}' (score={confidence:.3f})",
                        )
                    )

        return matches

    @staticmethod
    def _compute_similarity(a: str, b: str) -> float:
        """Return max of direct SequenceMatcher ratio and token-sort ratio."""
        # Direct comparison
        seq_ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

        # Token sort: split on whitespace, sort tokens, rejoin
        tokens_a = " ".join(sorted(a.lower().split()))
        tokens_b = " ".join(sorted(b.lower().split()))
        token_sort_ratio = difflib.SequenceMatcher(None, tokens_a, tokens_b).ratio()

        return max(seq_ratio, token_sort_ratio)


class DomainMatcher:
    """Match records by shared email domain."""

    def match(
        self,
        records_a: list[dict],
        records_b: list[dict],
        domain_field: str = "domain",
    ) -> list[MatchCandidate]:
        matches: list[MatchCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        # Build domain index for records_b
        domain_index: dict[str, list[dict]] = {}
        for rec in records_b:
            domain = rec.get(domain_field)
            if domain and str(domain).strip():
                key = str(domain).strip().lower()
                domain_index.setdefault(key, []).append(rec)

        for rec_a in records_a:
            domain_a = rec_a.get(domain_field)
            if not domain_a or not str(domain_a).strip():
                continue
            lookup = str(domain_a).strip().lower()

            source_a = rec_a.get("_source_id", "unknown")
            rid_a = rec_a.get("_record_id", "unknown")
            key_a = f"{source_a}:{rid_a}"

            for rec_b in domain_index.get(lookup, []):
                source_b = rec_b.get("_source_id", "unknown")
                rid_b = rec_b.get("_record_id", "unknown")
                key_b = f"{source_b}:{rid_b}"

                if key_a == key_b:
                    continue

                pair = (min(key_a, key_b), max(key_a, key_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                matches.append(
                    MatchCandidate(
                        entity_a=key_a,
                        entity_b=key_b,
                        confidence=0.80,
                        match_method="domain",
                        reasoning=f"Shared domain: {lookup}",
                    )
                )

        return matches


class AIMatcher:
    """Match ambiguous entity pairs using an LLM (Claude API via OpenAI-compatible endpoint).

    Sends candidate pairs with all available context and receives a match
    probability plus reasoning.  Requires ``ANTHROPIC_API_KEY`` in the
    environment.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def match(
        self,
        records_a: list[dict],
        records_b: list[dict],
        *,
        min_confidence: float = 0.60,
    ) -> list[MatchCandidate]:
        """Score all cross-source pairs via LLM.  Only pairs above *min_confidence* are returned."""
        if not self.available:
            log.debug("AIMatcher: no ANTHROPIC_API_KEY, skipping")
            return []

        try:
            import openai  # noqa: F811
        except ImportError:
            log.debug("AIMatcher: openai package not installed, skipping")
            return []

        client = openai.OpenAI(
            api_key=self._api_key,
            base_url="https://api.anthropic.com/v1/",
        )

        matches: list[MatchCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        for rec_a in records_a:
            source_a = rec_a.get("_source_id", "unknown")
            rid_a = rec_a.get("_record_id", "unknown")
            key_a = f"{source_a}:{rid_a}"

            for rec_b in records_b:
                source_b = rec_b.get("_source_id", "unknown")
                rid_b = rec_b.get("_record_id", "unknown")
                key_b = f"{source_b}:{rid_b}"

                if key_a == key_b:
                    continue
                pair = (min(key_a, key_b), max(key_a, key_b))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                result = self._score_pair(client, rec_a, rec_b)
                if result is not None and result[0] >= min_confidence:
                    matches.append(
                        MatchCandidate(
                            entity_a=key_a,
                            entity_b=key_b,
                            confidence=result[0],
                            match_method="ai",
                            reasoning=result[1],
                        )
                    )

        return matches

    def _score_pair(
        self,
        client: object,
        rec_a: dict,
        rec_b: dict,
    ) -> tuple[float, str] | None:
        """Call the LLM to score a single pair.  Returns (confidence, reasoning) or None on error."""
        skip_keys = {"_source_id", "_record_id"}
        context_a = {k: v for k, v in rec_a.items() if k not in skip_keys}
        context_b = {k: v for k, v in rec_b.items() if k not in skip_keys}

        prompt = (
            "You are an entity-resolution expert. Determine whether these two records "
            "refer to the same real-world entity.\n\n"
            f"Record A: {json.dumps(context_a, default=str)}\n"
            f"Record B: {json.dumps(context_b, default=str)}\n\n"
            "Respond with ONLY a JSON object: "
            '{"confidence": <float 0-1>, "reasoning": "<one sentence>"}'
        )

        try:
            response = client.chat.completions.create(  # type: ignore[union-attr]
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            # Parse JSON from response (handle markdown fences)
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            return (float(data["confidence"]), str(data.get("reasoning", "")))
        except Exception as exc:
            log.debug("AIMatcher: LLM call failed for pair: %s", exc)
            return None


class EntityMatcher:
    """Orchestrator that runs matchers in cascade: exact -> fuzzy -> domain -> AI.

    Unmatched records from each stage flow into the next.
    """

    def __init__(
        self,
        threshold_engine: object | None = None,
        ai_matcher: AIMatcher | None = None,
    ) -> None:
        self.threshold_engine = threshold_engine
        self._exact = ExactMatcher()
        self._fuzzy = FuzzyNameMatcher()
        self._domain = DomainMatcher()
        self._ai = ai_matcher

    def match_entities(
        self,
        records_by_source: dict[str, list[dict]],
        id_fields: list[str] | None = None,
        name_field: str = "name",
        domain_field: str = "domain",
        prediction_log: object | None = None,
    ) -> dict:
        """Run the full matching cascade.

        Returns:
            {"matches": [...], "review_queue": [...], "unmatched": [...]}
        """
        all_records = self._tag_records(records_by_source)
        auto_approve_threshold, review_threshold = self._get_thresholds()

        all_matches: list[MatchCandidate] = []
        matched_keys: set[str] = set()

        sources = list(records_by_source.keys())
        source_records = self._group_by_source(all_records)

        # Stage 1: Exact ID match
        if id_fields:
            self._run_pairwise_exact(sources, source_records, id_fields, all_matches, matched_keys)

        # Stage 2: Fuzzy name (unmatched only)
        unmatched = self._get_unmatched_by_source(all_records, matched_keys)
        self._run_pairwise_fuzzy(unmatched, name_field, all_matches, matched_keys)

        # Stage 3: Domain match (still unmatched only)
        unmatched = self._get_unmatched_by_source(all_records, matched_keys)
        self._run_pairwise_domain(unmatched, domain_field, all_matches, matched_keys)

        # Stage 4: AI match (still unmatched only, if AI matcher configured)
        if self._ai is not None and self._ai.available:
            unmatched = self._get_unmatched_by_source(all_records, matched_keys)
            self._run_pairwise_ai(unmatched, all_matches, matched_keys)

        # Classify and log
        approved, review_queue = self._classify(all_matches, auto_approve_threshold, review_threshold)
        self._log_predictions(all_matches, prediction_log)

        unmatched_keys = sorted(
            f"{rec['_source_id']}:{rec['_record_id']}"
            for rec in all_records
            if f"{rec['_source_id']}:{rec['_record_id']}" not in matched_keys
        )

        return {
            "matches": approved,
            "review_queue": review_queue,
            "unmatched": unmatched_keys,
        }

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _tag_records(records_by_source: dict[str, list[dict]]) -> list[dict]:
        all_records: list[dict] = []
        for source_id, records in records_by_source.items():
            for rec in records:
                tagged = dict(rec)
                tagged.setdefault("_source_id", source_id)
                tagged.setdefault("_record_id", tagged.get("id", str(uuid.uuid4())))
                all_records.append(tagged)
        return all_records

    def _get_thresholds(self) -> tuple[float, float]:
        if self.threshold_engine is not None:
            thresholds = self.threshold_engine.get_thresholds(  # type: ignore[union-attr]
                decision_type="entity_match",
            )
            return thresholds.auto_approve_threshold, thresholds.review_threshold
        # Entity-match cold start defaults: 0.85 auto-approve, 0.70 review
        return _ENTITY_MATCH_AUTO_APPROVE, _ENTITY_MATCH_REVIEW

    @staticmethod
    def _group_by_source(all_records: list[dict]) -> dict[str, list[dict]]:
        source_records: dict[str, list[dict]] = {}
        for rec in all_records:
            source_records.setdefault(rec["_source_id"], []).append(rec)
        return source_records

    @staticmethod
    def _get_unmatched_by_source(
        all_records: list[dict],
        matched_keys: set[str],
    ) -> dict[str, list[dict]]:
        unmatched: dict[str, list[dict]] = {}
        for rec in all_records:
            key = f"{rec['_source_id']}:{rec['_record_id']}"
            if key not in matched_keys:
                unmatched.setdefault(rec["_source_id"], []).append(rec)
        return unmatched

    def _run_pairwise_exact(
        self,
        sources: list[str],
        source_records: dict[str, list[dict]],
        id_fields: list[str],
        all_matches: list[MatchCandidate],
        matched_keys: set[str],
    ) -> None:
        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1 :]:
                for m in self._exact.match(source_records.get(src_a, []), source_records.get(src_b, []), id_fields):
                    matched_keys.add(m.entity_a)
                    matched_keys.add(m.entity_b)
                    all_matches.append(m)

    def _run_pairwise_fuzzy(
        self,
        unmatched: dict[str, list[dict]],
        name_field: str,
        all_matches: list[MatchCandidate],
        matched_keys: set[str],
    ) -> None:
        srcs = list(unmatched.keys())
        for i, src_a in enumerate(srcs):
            for src_b in srcs[i + 1 :]:
                for m in self._fuzzy.match(unmatched.get(src_a, []), unmatched.get(src_b, []), name_field=name_field):
                    matched_keys.add(m.entity_a)
                    matched_keys.add(m.entity_b)
                    all_matches.append(m)

    def _run_pairwise_domain(
        self,
        unmatched: dict[str, list[dict]],
        domain_field: str,
        all_matches: list[MatchCandidate],
        matched_keys: set[str],
    ) -> None:
        srcs = list(unmatched.keys())
        for i, src_a in enumerate(srcs):
            for src_b in srcs[i + 1 :]:
                for m in self._domain.match(unmatched.get(src_a, []), unmatched.get(src_b, []), domain_field=domain_field):
                    matched_keys.add(m.entity_a)
                    matched_keys.add(m.entity_b)
                    all_matches.append(m)

    def _run_pairwise_ai(
        self,
        unmatched: dict[str, list[dict]],
        all_matches: list[MatchCandidate],
        matched_keys: set[str],
    ) -> None:
        srcs = list(unmatched.keys())
        for i, src_a in enumerate(srcs):
            for src_b in srcs[i + 1 :]:
                for m in self._ai.match(unmatched.get(src_a, []), unmatched.get(src_b, [])):  # type: ignore[union-attr]
                    matched_keys.add(m.entity_a)
                    matched_keys.add(m.entity_b)
                    all_matches.append(m)

    @staticmethod
    def _classify(
        all_matches: list[MatchCandidate],
        auto_approve_threshold: float,
        review_threshold: float,
    ) -> tuple[list[MatchCandidate], list[MatchCandidate]]:
        approved: list[MatchCandidate] = []
        review_queue: list[MatchCandidate] = []
        for m in all_matches:
            if m.confidence >= auto_approve_threshold:
                approved.append(m)
            elif m.confidence >= review_threshold:
                review_queue.append(m)
        return approved, review_queue

    @staticmethod
    def _log_predictions(
        all_matches: list[MatchCandidate],
        prediction_log: object | None,
    ) -> None:
        if prediction_log is None:
            return
        now = time.time()
        for m in all_matches:
            try:
                prediction_log.record(  # type: ignore[union-attr]
                    _make_prediction_record(m, now)
                )
            except Exception:
                log.debug("Failed to log prediction for %s <-> %s", m.entity_a, m.entity_b)


def _make_prediction_record(match: MatchCandidate, now: float) -> object:
    """Build a PredictionRecord for a match candidate.

    Import is deferred to avoid hard dependency on calibration module.
    """
    from attestdb.calibration.prediction_log import PredictionRecord

    source_id = match.entity_a.split(":")[0] if ":" in match.entity_a else "unknown"

    return PredictionRecord(
        prediction_id=str(uuid.uuid4()),
        tenant_id="default",
        decision_type="entity_match",
        source_id=source_id,
        field_semantic_type=None,
        predicted_confidence=match.confidence,
        predicted_value=f"{match.entity_a} <-> {match.entity_b}",
        review_outcome="pending",
        corrected_value=None,
        reviewed_at=None,
        reviewed_by=None,
        created_at=now,
    )
