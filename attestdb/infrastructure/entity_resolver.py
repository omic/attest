"""Entity resolution — external ID index, feature-based matching, and duplicate detection."""

from __future__ import annotations

import logging
import math

import jellyfish

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import EntitySummary, entity_summary_from_dict

logger = logging.getLogger(__name__)


# ── Similarity helpers ───────────────────────────────────────────────────


def _trigram_set(s: str) -> set[str]:
    """Character trigrams from a string."""
    s = s.lower()
    if len(s) < 3:
        return {s}
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on character trigram sets."""
    sa, sb = _trigram_set(a), _trigram_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _token_set_ratio(a: str, b: str) -> float:
    """Sorted-token Jaccard — handles word reorderings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _acronym_match(short: str, long: str) -> bool:
    """Check if short is an acronym of long (e.g. IBM / International Business Machines)."""
    short_up = short.strip().upper()
    if len(short_up) < 2 or len(short_up) > 8:
        return False
    if not short_up.isalpha():
        return False
    tokens = long.strip().split()
    if len(tokens) < len(short_up):
        return False
    initials = "".join(t[0].upper() for t in tokens if t)
    return initials == short_up


def _is_acronym_like(s: str) -> bool:
    """Heuristic: all uppercase, short, alphabetic."""
    s = s.strip()
    return len(s) <= 8 and s.isalpha() and s.isupper()


# ── Person name flipping ─────────────────────────────────────────────────

_NAME_HONORIFICS = frozenset({
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.", "prof", "prof.",
    "jr", "jr.", "sr", "sr.", "iii", "iv",
})


def _flip_comma_name(name: str) -> str | None:
    """Flip "Lastname, Firstname [Middle]" into "firstname lastname" form.

    Only fires on single-comma person-style names: one token before the
    comma, 1–3 tokens after (dropping honorifics). Returns the flipped
    string (not normalized), or None if the input doesn't look like a
    comma-first person name.
    """
    if name is None:
        return None
    s = name.strip()
    if s.count(",") != 1:
        return None
    head, _, rest = s.partition(",")
    head_toks = [t for t in head.split() if t]
    rest_toks = [
        t for t in rest.replace(",", " ").split()
        if t and t.lower().strip(".") not in _NAME_HONORIFICS
    ]
    if len(head_toks) != 1 or not (1 <= len(rest_toks) <= 3):
        return None
    return " ".join(rest_toks + head_toks)


# ── Org suffix stripping ────────────────────────────────────────────────

_ORG_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "corporation", "llc", "ltd", "ltd.",
    "co", "co.", "company", "group", "holdings", "plc", "sa", "ag",
    "gmbh", "pty", "nv", "bv", "se", "lp", "llp",
}


def _strip_org_suffix(name: str) -> str:
    """Strip common corporate suffixes for comparison."""
    tokens = name.lower().split()
    while tokens and tokens[-1] in _ORG_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


# ── Blocking key computation ────────────────────────────────────────────


def _compute_blocking_keys(
    entity_id: str,
    display_name: str,
    entity_type: str = "",
    external_ids: dict[str, str] | None = None,
) -> list[str]:
    """Compute multiple blocking keys for an entity.

    Each key maps to a bucket in the blocking index. At query time,
    union the buckets for all keys to get candidate entities.
    """
    keys: list[str] = []
    name = display_name or entity_id

    # 1. Sorted first-3 tokens (handles reorderings)
    tokens = sorted(name.lower().split())
    if tokens:
        keys.append("tok:" + "_".join(tokens[:3]))

    # 2. Double Metaphone (phonetic — handles typos in names)
    for token in name.split()[:3]:
        token_clean = token.strip().lower()
        if len(token_clean) >= 2:
            try:
                codes = jellyfish.metaphone(token_clean)
                if codes:
                    keys.append(f"ph:{codes}")
            except Exception:
                pass

    # 3. Acronym key (extract initials from multi-word names)
    words = name.split()
    if len(words) >= 2:
        initials = "".join(w[0].upper() for w in words if w)
        if 2 <= len(initials) <= 8:
            keys.append(f"acr:{initials}")

    # 4. If the entity IS an acronym, register it as an acronym key
    stripped = name.strip()
    if _is_acronym_like(stripped):
        keys.append(f"acr:{stripped.upper()}")

    # 5. First-token + length-bucket (cheap universal fallback)
    if tokens:
        length_bucket = len(name) // 5
        keys.append(f"fl:{tokens[0]}_{length_bucket}")

    # 6. External ID prefixes
    if external_ids:
        for ns, val in external_ids.items():
            keys.append(f"xid:{ns}:{val}")

    # 7. Org suffix stripped key
    stripped_name = _strip_org_suffix(name.lower())
    if stripped_name and stripped_name != name.lower():
        stripped_tokens = sorted(stripped_name.split())
        if stripped_tokens:
            keys.append("tok:" + "_".join(stripped_tokens[:3]))

    return keys


class EntityResolver:
    """Resolves entity names against existing graph entities.

    Owns a reverse external_id index and a resolution cascade:
    1. Exact match on normalized name (conf=1.0)
    2. External ID match (conf=0.99)
    3. Text search fuzzy match if mode >= "fuzzy" (conf=0.5-0.95)

    Modes:
        "external_ids" — exact name + external ID lookup only
        "fuzzy" — adds text search + token overlap scoring
        "full" — same as fuzzy (reserved for future embedding-based resolution)
    """

    def __init__(self, store, mode: str = "external_ids"):
        self._store = store
        self._mode = mode
        # (namespace, ext_id) -> entity_id
        self._ext_id_index: dict[tuple[str, str], str] = {}
        # Blocking index: blocking_key -> set of entity_ids
        self._blocking_index: dict[str, set[str]] = {}
        self._built = False
        # Optional calibration hooks (set via configure_calibration)
        self._prediction_log = None
        self._review_queue = None
        self._threshold_engine = None

    def configure_calibration(
        self,
        prediction_log=None,
        review_queue=None,
        threshold_engine=None,
    ) -> None:
        """Attach calibration modules for decision logging and review routing."""
        self._prediction_log = prediction_log
        self._review_queue = review_queue
        self._threshold_engine = threshold_engine

    def build_index(self) -> None:
        """Scan all entities and build reverse external_id + blocking indexes.

        When multiple entities share the same (namespace, ext_id), all are tracked
        via _ext_id_collisions for duplicate detection.
        """
        self._ext_id_index.clear()
        self._blocking_index.clear()
        # Track collisions: (namespace, ext_id) -> set of entity_ids
        self._ext_id_collisions: dict[tuple[str, str], set[str]] = {}
        raw_entities = self._store.list_entities()
        entities = [entity_summary_from_dict(d) for d in raw_entities]
        for entity in entities:
            # External ID index
            ext_ids = entity.external_ids or {}
            for namespace, ext_id in ext_ids.items():
                key = (namespace, ext_id)
                if key in self._ext_id_index:
                    self._ext_id_collisions.setdefault(key, set()).add(self._ext_id_index[key])
                    self._ext_id_collisions[key].add(entity.id)
                self._ext_id_index[key] = entity.id

            # Blocking index
            bkeys = _compute_blocking_keys(
                entity.id, entity.name, entity.entity_type, ext_ids,
            )
            for bk in bkeys:
                self._blocking_index.setdefault(bk, set()).add(entity.id)

        self._built = True
        logger.info(
            "EntityResolver: indexed %d external_id mappings, %d blocking keys "
            "from %d entities (%d ext_id collisions)",
            len(self._ext_id_index), len(self._blocking_index),
            len(entities), len(self._ext_id_collisions),
        )

    def register_external_id(self, entity_id: str, namespace: str, ext_id: str) -> None:
        """Incrementally update ext_id index after upsert_entity."""
        self._ext_id_index[(namespace, ext_id)] = entity_id

    def register_entity(
        self, entity_id: str, display_name: str,
        entity_type: str = "", external_ids: dict[str, str] | None = None,
    ) -> None:
        """Incrementally register an entity in the blocking index."""
        bkeys = _compute_blocking_keys(entity_id, display_name, entity_type, external_ids)
        for bk in bkeys:
            self._blocking_index.setdefault(bk, set()).add(entity_id)

    def get_blocking_candidates(
        self, name: str, entity_type: str = "",
        external_ids: dict[str, str] | None = None, max_candidates: int = 50,
    ) -> list[str]:
        """Retrieve candidate entity_ids from the blocking index.

        Computes blocking keys for the query, unions all matching buckets,
        and returns up to max_candidates entity_ids.
        """
        query_keys = _compute_blocking_keys(
            normalize_entity_id(name), name, entity_type, external_ids,
        )
        candidates: set[str] = set()
        for qk in query_keys:
            bucket = self._blocking_index.get(qk, set())
            # Skip oversized buckets (low-information)
            if len(bucket) > 1000:
                continue
            candidates.update(bucket)
            if len(candidates) >= max_candidates * 2:
                break
        # Remove self (normalized query) from candidates
        normalized = normalize_entity_id(name)
        candidates.discard(normalized)
        return list(candidates)[:max_candidates]

    def resolve_by_external_id(self, namespace: str, ext_id: str) -> str | None:
        """Reverse lookup: (namespace, ext_id) -> entity_id."""
        return self._ext_id_index.get((namespace, ext_id))

    def resolve(
        self,
        name: str,
        entity_type: str = "",
        external_ids: dict[str, str] | None = None,
    ) -> tuple[str | None, float]:
        """Resolution cascade.

        Returns (existing_entity_id, confidence) or (None, 0.0).
        """
        normalized = normalize_entity_id(name)

        # 1. Exact match on normalized name
        try:
            raw = self._store.get_entity(normalized)
        except BaseException:
            raw = None  # Store may be temporarily unavailable (e.g., after snapshot)
        if raw is not None:
            return entity_summary_from_dict(raw).id, 1.0

        # 1b. Person comma-first → firstname-first fallback.
        # "Epstein, Jeffrey" → "jeffrey epstein". Only for person-type
        # (or empty — unknown) to avoid corrupting comma-bearing org names.
        if entity_type in ("", "person"):
            flipped = _flip_comma_name(name)
            if flipped is not None:
                flipped_norm = normalize_entity_id(flipped)
                if flipped_norm != normalized:
                    try:
                        raw = self._store.get_entity(flipped_norm)
                    except BaseException:
                        raw = None
                    if raw is not None:
                        return entity_summary_from_dict(raw).id, 0.98

        # 2. External ID match
        if external_ids:
            for namespace, ext_id in external_ids.items():
                resolved = self.resolve_by_external_id(namespace, ext_id)
                if resolved is not None:
                    return resolved, 0.99

        # 3. Fuzzy match via blocking index + BM25 (modes: fuzzy, full)
        if self._mode in ("fuzzy", "full"):
            # Collect candidates from blocking index + BM25
            candidate_ids: set[str] = set()

            # 3a. Blocking index candidates
            blocking_hits = self.get_blocking_candidates(
                name, entity_type, external_ids, max_candidates=50,
            )
            candidate_ids.update(blocking_hits)

            # 3b. BM25 candidates (existing text search)
            if hasattr(self._store, "search_entities"):
                for d in self._store.search_entities(normalized, top_k=20):
                    es = entity_summary_from_dict(d)
                    if es.id != normalized:
                        candidate_ids.add(es.id)

            # Get calibrated thresholds (or defaults)
            accept_threshold, review_threshold = self._get_thresholds(entity_type)

            # Score all unique candidates
            best_id = None
            best_score = 0.0
            for cid in candidate_ids:
                try:
                    raw = self._store.get_entity(cid)
                except BaseException:
                    continue
                if raw is None:
                    continue
                candidate = entity_summary_from_dict(raw)
                if candidate.id == normalized:
                    return candidate.id, 1.0
                # Pass original name for acronym detection (needs case)
                score = self._score_candidate(name, entity_type, candidate)
                if score > best_score:
                    best_score = score
                    best_id = candidate.id

            if best_id is not None:
                if best_score >= accept_threshold:
                    # Auto-merge: high confidence
                    self._log_decision(
                        name, best_id, entity_type, best_score, "auto_merge",
                    )
                    return best_id, best_score
                elif best_score >= review_threshold:
                    # Gray zone: route to review queue, don't merge yet
                    self._log_decision(
                        name, best_id, entity_type, best_score, "review",
                    )
                    self._enqueue_review(
                        name, best_id, entity_type, best_score,
                    )
                    # Don't resolve — let the human decide
                else:
                    # Below review threshold: reject
                    self._log_decision(
                        name, best_id, entity_type, best_score, "reject",
                    )

        return None, 0.0

    def find_duplicates(
        self, min_confidence: float = 0.9,
    ) -> list[tuple[str, str, float]]:
        """Scan existing entities for duplicate pairs.

        Pass 1: External ID collisions (different entity_ids, same ext_id).
        Pass 2: Text similarity via search_entities (if mode >= "fuzzy").
        Returns deduplicated (entity_a, entity_b, confidence) sorted by confidence desc.
        """
        if not self._built:
            self.build_index()

        seen_pairs: set[tuple[str, str]] = set()
        results: list[tuple[str, str, float]] = []

        # Pass 1: External ID collisions from build_index
        collisions = getattr(self, "_ext_id_collisions", {})
        for (_ns, _eid), entity_ids in collisions.items():
            unique_ids = sorted(entity_ids)
            for i in range(len(unique_ids)):
                for j in range(i + 1, len(unique_ids)):
                    pair = (unique_ids[i], unique_ids[j])
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        results.append((pair[0], pair[1], 0.99))

        # Pass 2: Text similarity via blocking buckets (fuzzy/full mode)
        if self._mode in ("fuzzy", "full"):
            # Iterate blocking buckets — only score entities that share a key
            for _bkey, bucket in self._blocking_index.items():
                if len(bucket) < 2 or len(bucket) > 500:
                    continue  # Skip singletons and oversized buckets
                bucket_list = sorted(bucket)
                for i in range(len(bucket_list)):
                    for j in range(i + 1, len(bucket_list)):
                        pair = (bucket_list[i], bucket_list[j])
                        if pair in seen_pairs:
                            continue
                        seen_pairs.add(pair)
                        # Fetch entities for scoring
                        try:
                            raw_a = self._store.get_entity(pair[0])
                            raw_b = self._store.get_entity(pair[1])
                        except BaseException:
                            continue
                        if raw_a is None or raw_b is None:
                            continue
                        entity_a = entity_summary_from_dict(raw_a)
                        entity_b = entity_summary_from_dict(raw_b)
                        score = self._score_candidate(
                            entity_a.id, entity_a.entity_type, entity_b,
                        )
                        if score >= min_confidence:
                            results.append((pair[0], pair[1], score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    @staticmethod
    def _score_candidate(
        query: str,
        query_type: str,
        candidate: EntitySummary,
        query_external_ids: dict[str, str] | None = None,
    ) -> float:
        """Score a candidate entity against a query using a weighted feature ensemble.

        Features: Jaro-Winkler, token-set ratio, trigram Jaccard, acronym match,
        type compatibility, external ID overlap. Type-aware weighting.
        Returns a score in [0.0, 0.95].
        """
        q = query.lower()
        # Best candidate text: try display_name first, fall back to id
        c_name = (candidate.name or "").lower()
        c_id = candidate.id.lower()
        if not q or (not c_name and not c_id):
            return 0.0

        # ── Compute features (best of id and display_name) ──────────

        # Compare against both candidate surfaces
        surfaces = [c_id]
        if c_name and c_name != c_id:
            surfaces.append(c_name)

        jw = max(jellyfish.jaro_winkler_similarity(q, s) for s in surfaces)
        tsr = max(_token_set_ratio(q, s) for s in surfaces)
        tri = max(_trigram_jaccard(q, s) for s in surfaces)

        # Acronym match (bidirectional, use original case)
        acr = 0.0
        c_orig_name = candidate.name or ""
        c_orig_id = candidate.id
        for qa, qb in [
            (query, c_orig_name), (query, c_orig_id),
            (c_orig_name, query), (c_orig_id, query),
        ]:
            if qa and qb and _is_acronym_like(qa) and _acronym_match(qa, qb):
                acr = 1.0
                break

        # Org suffix stripping bonus
        org_bonus = 0.0
        c_primary = c_name if c_name else c_id
        if query_type in ("organization", "company", "org", ""):
            q_stripped = _strip_org_suffix(q)
            c_stripped = _strip_org_suffix(c_primary)
            if q_stripped and c_stripped and q_stripped != q:
                stripped_jw = jellyfish.jaro_winkler_similarity(q_stripped, c_stripped)
                if stripped_jw > jw:
                    org_bonus = stripped_jw - jw

        # Type compatibility
        type_match = 0.0
        if query_type and candidate.entity_type:
            if query_type == candidate.entity_type:
                type_match = 1.0

        # External ID overlap
        ext_id_match = 0.0
        if query_external_ids and candidate.external_ids:
            for ns, val in query_external_ids.items():
                if candidate.external_ids.get(ns) == val:
                    ext_id_match = 1.0
                    break

        # ── Combine ──────────────────────────────────────────────────

        # External ID match is near-deterministic — short-circuit
        if ext_id_match > 0:
            return min(0.95, 0.80 + type_match * 0.10)

        # Acronym match is a strong signal — short-circuit
        if acr > 0:
            return min(0.95, 0.75 + type_match * 0.10)

        # Org suffix stripping: if stripping improves JW, boost it
        jw_effective = jw
        if org_bonus > 0:
            jw_effective = jw + org_bonus

        # Core text similarity: average of top-2 signals.
        # This prevents a single misleading feature (e.g. high JW on
        # short unrelated strings) from producing a false positive.
        signals = sorted([jw_effective, tsr, tri], reverse=True)
        text_sim = (signals[0] + signals[1]) / 2.0

        # Type bonus
        type_bonus = 0.10 if type_match > 0 else 0.0

        # Numeric-mismatch penalty: digits in entity strings usually encode
        # data — addresses ("59th st" vs "69th st"), durations ("18 months"
        # vs "21 months"), dollar amounts ("$10m" vs "$110m"), ages ("under
        # 15" vs "under 16"). When the digit sequences of the two strings
        # differ but the text around them is similar, JW/TSR/trigram rate
        # them high despite being semantically different values. Halve the
        # score in that case so auto-merge leaves them alone. Identical
        # digits (or none in either side) → no penalty.
        def _digit_signature(s: str) -> tuple[str, ...]:
            out, cur = [], []
            for ch in s:
                if ch.isdigit():
                    cur.append(ch)
                else:
                    if cur:
                        out.append("".join(cur))
                        cur = []
            if cur:
                out.append("".join(cur))
            return tuple(out)

        best_surface = c_name if c_name else c_id
        q_digits = _digit_signature(q)
        c_digits = _digit_signature(best_surface)
        digit_penalty = 1.0
        if q_digits and c_digits and q_digits != c_digits:
            digit_penalty = 0.5

        # Final score: text similarity dominates, type is a bonus
        score = (text_sim + type_bonus) * digit_penalty

        # Cap at 0.95 — reserve 1.0 for exact/ext-id matches
        return min(0.95, score)

    # ── Calibration helpers ──────────────────────────────────────────

    def _get_thresholds(self, entity_type: str = "") -> tuple[float, float]:
        """Get (accept_threshold, review_threshold) from calibration or defaults.

        Returns per-entity-type thresholds if calibration is wired,
        otherwise falls back to sensible defaults.
        """
        if self._threshold_engine is not None:
            try:
                cfg = self._threshold_engine.get_thresholds(
                    decision_type="entity_match",
                )
                return cfg.auto_approve_threshold, cfg.review_threshold
            except Exception:
                pass
        # Defaults: accept >= 0.5, review >= 0.4
        return 0.5, 0.4

    def _log_decision(
        self,
        query_name: str,
        candidate_id: str,
        entity_type: str,
        score: float,
        decision: str,
    ) -> None:
        """Log a match decision to the prediction log."""
        if self._prediction_log is None:
            return
        try:
            import time
            import uuid
            from attestdb.calibration.prediction_log import PredictionRecord

            # Map ER decisions to valid prediction outcomes
            outcome_map = {
                "auto_merge": "confirmed",
                "review": "pending",
                "reject": "rejected",
            }
            record = PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                tenant_id="default",
                decision_type="entity_match",
                source_id="entity_resolver",
                field_semantic_type=entity_type or None,
                predicted_confidence=score,
                predicted_value=f"{query_name} <-> {candidate_id}",
                review_outcome=outcome_map.get(decision, "pending"),
                corrected_value=None,
                reviewed_at=None,
                reviewed_by=None,
                created_at=time.time(),
            )
            self._prediction_log.record(record)
        except Exception as e:
            logger.debug("Failed to log ER decision: %s", e)

    def _enqueue_review(
        self,
        query_name: str,
        candidate_id: str,
        entity_type: str,
        score: float,
    ) -> None:
        """Route a gray-zone match to the human review queue."""
        if self._review_queue is None:
            return
        try:
            import time
            import uuid
            from attestdb.review.models import ReviewItem

            item = ReviewItem(
                item_id=str(uuid.uuid4()),
                item_type="entity_match",
                source_id="entity_resolver",
                priority=score,
                status="pending",
                context={
                    "query_name": query_name,
                    "candidate_id": candidate_id,
                    "entity_type": entity_type,
                    "score": score,
                },
                proposed_value=f"merge:{query_name}->{candidate_id}",
                created_at=time.time(),
                tenant_id="default",
            )
            self._review_queue.add(item)
        except Exception as e:
            logger.debug("Failed to enqueue ER review: %s", e)
