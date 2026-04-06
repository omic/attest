"""Cross-source schema alignment.

Groups fields from multiple :class:`SchemaMap` instances into aligned
canonical groups using exact name matching, semantic type matching,
and heuristic (or LLM-assisted) fuzzy matching.  Detects conflicts,
records alignment decisions in the calibration prediction log, and
respects adaptive thresholds for review routing.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from attestdb.discovery.analyzer import SemanticMapping
from attestdb.discovery.sampler import FieldProfile
from attestdb.discovery.schema_map import SchemaMap

logger = logging.getLogger(__name__)

# ── Default confidence thresholds (cold start) ───────────────────────

_AUTO_APPROVE_THRESHOLD = 0.95
_NEEDS_REVIEW_THRESHOLD = 0.60


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class AlignedFieldGroup:
    """A group of fields from different sources that map to the same concept."""

    canonical_name: str
    semantic_type: str
    source_mappings: dict[str, str] = field(default_factory=dict)
    alignment_confidence: float = 0.0
    conflict_type: str = "none"
    transformation_needed: bool = False
    transformation_notes: str = ""
    review_status: str = "needs_review"


@dataclass
class UnifiedSchema:
    """The result of aligning multiple source schemas."""

    tenant_id: str
    aligned_fields: list[AlignedFieldGroup] = field(default_factory=list)
    orphan_fields: list[tuple[str, str]] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "tenant_id": self.tenant_id,
            "aligned_fields": [
                {
                    "canonical_name": g.canonical_name,
                    "semantic_type": g.semantic_type,
                    "source_mappings": dict(g.source_mappings),
                    "alignment_confidence": g.alignment_confidence,
                    "conflict_type": g.conflict_type,
                    "transformation_needed": g.transformation_needed,
                    "transformation_notes": g.transformation_notes,
                    "review_status": g.review_status,
                }
                for g in self.aligned_fields
            ],
            "orphan_fields": [list(pair) for pair in self.orphan_fields],
            "conflicts": list(self.conflicts),
            "source_ids": list(self.source_ids),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UnifiedSchema:
        """Reconstruct from a dict (inverse of ``to_dict``)."""
        aligned = [
            AlignedFieldGroup(
                canonical_name=g["canonical_name"],
                semantic_type=g["semantic_type"],
                source_mappings=g.get("source_mappings", {}),
                alignment_confidence=g.get("alignment_confidence", 0.0),
                conflict_type=g.get("conflict_type", "none"),
                transformation_needed=g.get("transformation_needed", False),
                transformation_notes=g.get("transformation_notes", ""),
                review_status=g.get("review_status", "needs_review"),
            )
            for g in d.get("aligned_fields", [])
        ]
        orphans = [tuple(pair) for pair in d.get("orphan_fields", [])]
        return cls(
            tenant_id=d.get("tenant_id", "default"),
            aligned_fields=aligned,
            orphan_fields=orphans,
            conflicts=d.get("conflicts", []),
            source_ids=d.get("source_ids", []),
            created_at=d.get("created_at", 0.0),
        )

    def to_report(self) -> str:
        """Generate a human-readable alignment report."""
        lines = [
            "Unified Schema Alignment Report",
            f"Tenant: {self.tenant_id}",
            f"Sources: {', '.join(self.source_ids)}",
            "=" * 60,
            f"Aligned field groups: {len(self.aligned_fields)}",
            f"Orphan fields:        {len(self.orphan_fields)}",
            f"Conflicts:            {len(self.conflicts)}",
        ]

        if self.aligned_fields:
            lines.append("")
            lines.append("Aligned Fields:")
            for g in self.aligned_fields:
                sources_str = ", ".join(
                    f"{sid}:{fname}" for sid, fname in sorted(g.source_mappings.items())
                )
                transform_flag = " TRANSFORM" if g.transformation_needed else ""
                lines.append(
                    f"  {g.canonical_name} ({g.semantic_type}) "
                    f"conf={g.alignment_confidence:.2f} "
                    f"conflict={g.conflict_type}{transform_flag} "
                    f"[{g.review_status}]"
                )
                lines.append(f"    sources: {sources_str}")
                if g.transformation_notes:
                    lines.append(f"    transform: {g.transformation_notes}")

        if self.orphan_fields:
            lines.append("")
            lines.append("Orphan Fields:")
            for source_id, fname in self.orphan_fields:
                lines.append(f"  {source_id}: {fname}")

        if self.conflicts:
            lines.append("")
            lines.append("Conflicts:")
            for c in self.conflicts:
                lines.append(f"  {c.get('canonical_name', '?')}: {c.get('description', '')}")

        return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────


def align_schemas(
    schema_maps: list[SchemaMap],
    prediction_log: object | None = None,
    threshold_engine: object | None = None,
    tenant_id: str = "default",
    llm_client: object | None = None,
) -> UnifiedSchema:
    """Align fields across multiple source schemas into a unified schema.

    Three alignment layers are applied in sequence:

    1. **Exact name match** — fields with identical names (case-insensitive).
    2. **Semantic type match** — fields with the same ``semantic_type`` from
       their :class:`SemanticMapping`.
    3. **Fuzzy match** — heuristic similarity, then AI-assisted matching
       (if *llm_client* is provided) on remaining unmatched fields.

    Parameters
    ----------
    schema_maps:
        One :class:`SchemaMap` per data source.
    prediction_log:
        Optional :class:`PredictionLog` — alignment decisions are recorded
        as ``decision_type="field_alignment"`` predictions.
    threshold_engine:
        Optional :class:`ThresholdEngine` — used to set ``review_status``
        on each group.  Falls back to cold-start defaults.
    tenant_id:
        Tenant identifier propagated to all output structures.
    llm_client:
        Optional OpenAI-compatible client for AI-assisted fuzzy matching.
        Must support ``client.chat.completions.create()``.
    """
    if not schema_maps:
        return UnifiedSchema(tenant_id=tenant_id, source_ids=[])

    source_ids = [sm.source_id for sm in schema_maps]
    profile_index, mapping_index = _build_indexes(schema_maps)

    # Track which (source_id, field_name) pairs are still unmatched
    unmatched: set[tuple[str, str]] = {
        (sm.source_id, fp.field_name)
        for sm in schema_maps
        for fp in sm.field_profiles
    }

    aligned_groups: list[AlignedFieldGroup] = []

    # Layer 1: exact name match
    _align_exact_names(schema_maps, profile_index, mapping_index, unmatched, aligned_groups)

    # Layer 2: semantic type match
    _align_semantic_types(mapping_index, profile_index, unmatched, aligned_groups)

    # Layer 3: fuzzy match (heuristic + optional LLM)
    _align_fuzzy(profile_index, mapping_index, unmatched, aligned_groups)

    # Layer 3b: AI-assisted fuzzy match on remaining unmatched
    if llm_client is not None and unmatched:
        _align_llm_fuzzy(
            llm_client, profile_index, mapping_index, unmatched, aligned_groups,
        )

    # Collect orphans + resolve review status
    orphan_fields = sorted(unmatched)
    auto_thresh, review_thresh = _get_thresholds(threshold_engine)
    conflicts = _finalize_groups(aligned_groups, auto_thresh, review_thresh)

    if prediction_log is not None:
        _record_predictions(aligned_groups, source_ids, prediction_log, tenant_id)

    return UnifiedSchema(
        tenant_id=tenant_id,
        aligned_fields=aligned_groups,
        orphan_fields=orphan_fields,
        conflicts=conflicts,
        source_ids=source_ids,
        created_at=time.time(),
    )


# ── Index building ───────────────────────────────────────────────────


def _build_indexes(
    schema_maps: list[SchemaMap],
) -> tuple[dict[str, dict[str, FieldProfile]], dict[str, dict[str, SemanticMapping]]]:
    """Build source_id -> field_name -> profile/mapping lookup dicts."""
    profile_index: dict[str, dict[str, FieldProfile]] = {}
    mapping_index: dict[str, dict[str, SemanticMapping]] = {}

    for sm in schema_maps:
        profile_index[sm.source_id] = {fp.field_name: fp for fp in sm.field_profiles}
        mapping_index[sm.source_id] = {sem.field_name: sem for sem in sm.semantic_mappings}

    return profile_index, mapping_index


# ── Layer runners ────────────────────────────────────────────────────


def _align_exact_names(
    schema_maps: list[SchemaMap],
    profile_index: dict[str, dict[str, FieldProfile]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
    unmatched: set[tuple[str, str]],
    aligned_groups: list[AlignedFieldGroup],
) -> None:
    """Layer 1: group fields with identical names (case-insensitive)."""
    name_groups = _group_by_name(schema_maps)
    for _norm_name, members in name_groups.items():
        if len(members) < 2:
            continue
        conflict = _detect_conflict(members, profile_index, mapping_index)
        group = AlignedFieldGroup(
            canonical_name=_pick_canonical_name(members, profile_index),
            semantic_type=_pick_semantic_type(members, mapping_index),
            source_mappings=dict(members),
            alignment_confidence=_exact_match_confidence(members, profile_index),
            conflict_type=conflict["type"],
            transformation_notes=conflict.get("notes", ""),
        )
        aligned_groups.append(group)
        for pair in members:
            unmatched.discard(pair)


def _align_semantic_types(
    mapping_index: dict[str, dict[str, SemanticMapping]],
    profile_index: dict[str, dict[str, FieldProfile]],
    unmatched: set[tuple[str, str]],
    aligned_groups: list[AlignedFieldGroup],
) -> None:
    """Layer 2: group remaining fields by semantic type."""
    sem_groups = _group_by_semantic_type(unmatched, mapping_index)
    for sem_type, members in sem_groups.items():
        if len(members) < 2 or sem_type == "unknown":
            continue
        conflict = _detect_conflict(members, profile_index, mapping_index)
        group = AlignedFieldGroup(
            canonical_name=_pick_canonical_name(members, profile_index),
            semantic_type=sem_type,
            source_mappings=dict(members),
            alignment_confidence=_semantic_match_confidence(members, mapping_index),
            conflict_type=conflict["type"],
            transformation_notes=conflict.get("notes", ""),
        )
        names = {fname for _, fname in members}
        if len(names) > 1 and group.conflict_type == "none":
            group.conflict_type = "naming"
        aligned_groups.append(group)
        for pair in members:
            unmatched.discard(pair)


def _align_fuzzy(
    profile_index: dict[str, dict[str, FieldProfile]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
    unmatched: set[tuple[str, str]],
    aligned_groups: list[AlignedFieldGroup],
) -> None:
    """Layer 3: heuristic fuzzy matching on remaining fields."""
    fuzzy_pairs = _fuzzy_match(unmatched, profile_index, mapping_index)
    for members, confidence in fuzzy_pairs:
        if len(members) < 2:
            continue
        conflict = _detect_conflict(members, profile_index, mapping_index)
        group = AlignedFieldGroup(
            canonical_name=_pick_canonical_name(members, profile_index),
            semantic_type=_pick_semantic_type(members, mapping_index),
            source_mappings=dict(members),
            alignment_confidence=confidence,
            conflict_type=conflict["type"],
            transformation_notes=conflict.get("notes", ""),
        )
        aligned_groups.append(group)
        for pair in members:
            unmatched.discard(pair)


def _finalize_groups(
    groups: list[AlignedFieldGroup],
    auto_thresh: float,
    review_thresh: float,
) -> list[dict]:
    """Set review_status on each group and collect conflict details."""
    conflicts: list[dict] = []
    for group in groups:
        group.review_status = _classify_review(
            group.alignment_confidence, auto_thresh, review_thresh
        )
        group.transformation_needed = (
            group.conflict_type in ("unit", "type")
            or bool(group.transformation_notes)
        )
        if group.conflict_type != "none":
            conflicts.append({
                "canonical_name": group.canonical_name,
                "conflict_type": group.conflict_type,
                "description": (
                    group.transformation_notes
                    or f"{group.conflict_type} conflict across sources"
                ),
                "sources": list(group.source_mappings.keys()),
            })
    return conflicts


# ── Layer 1: exact name grouping ─────────────────────────────────────


def _group_by_name(
    schema_maps: list[SchemaMap],
) -> dict[str, list[tuple[str, str]]]:
    """Group fields by case-insensitive name across sources.

    Returns ``{normalised_name: [(source_id, original_field_name), ...]}``.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for sm in schema_maps:
        for fp in sm.field_profiles:
            key = fp.field_name.lower().strip()
            groups.setdefault(key, []).append((sm.source_id, fp.field_name))
    return groups


# ── Layer 2: semantic type grouping ──────────────────────────────────


def _group_by_semantic_type(
    unmatched: set[tuple[str, str]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
) -> dict[str, list[tuple[str, str]]]:
    """Group remaining unmatched fields by semantic type."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for sid, fname in unmatched:
        sem = mapping_index.get(sid, {}).get(fname)
        if sem and sem.semantic_type and sem.semantic_type != "unknown":
            groups.setdefault(sem.semantic_type, []).append((sid, fname))
    return groups


# ── Layer 3: fuzzy matching ──────────────────────────────────────────


def _fuzzy_match(
    unmatched: set[tuple[str, str]],
    profile_index: dict[str, dict[str, FieldProfile]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
) -> list[tuple[list[tuple[str, str]], float]]:
    """Attempt heuristic fuzzy matching on remaining unmatched fields.

    Compares pairs across different sources using Jaccard similarity on
    type distributions and fill rate proximity.  Returns groups with
    their confidence score.
    """
    if len(unmatched) < 2:
        return []

    items = sorted(unmatched)
    matched_pairs: list[tuple[list[tuple[str, str]], float]] = []
    consumed: set[tuple[str, str]] = set()

    for i, (sid_a, fname_a) in enumerate(items):
        if (sid_a, fname_a) in consumed:
            continue
        prof_a = profile_index.get(sid_a, {}).get(fname_a)
        if not prof_a:
            continue

        best_match: tuple[str, str] | None = None
        best_score = 0.0

        for j in range(i + 1, len(items)):
            sid_b, fname_b = items[j]
            if sid_b == sid_a:
                continue  # Only match across different sources
            if (sid_b, fname_b) in consumed:
                continue

            prof_b = profile_index.get(sid_b, {}).get(fname_b)
            if not prof_b:
                continue

            score = _heuristic_similarity(prof_a, prof_b)
            if score > best_score:
                best_score = score
                best_match = (sid_b, fname_b)

        if best_match is not None and best_score >= 0.50:
            members = [(sid_a, fname_a), best_match]
            matched_pairs.append((members, best_score))
            consumed.add((sid_a, fname_a))
            consumed.add(best_match)

    return matched_pairs


def _align_llm_fuzzy(
    llm_client: object,
    profile_index: dict[str, dict[str, FieldProfile]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
    unmatched: set[tuple[str, str]],
    aligned_groups: list[AlignedFieldGroup],
) -> None:
    """Layer 3b: AI-assisted fuzzy matching using an LLM.

    Sends pairs of unmatched fields (across different sources) to the LLM
    with their profiles and asks for match probability.
    """
    items = sorted(unmatched)
    if len(items) < 2:
        return

    # Build cross-source pairs (limit to avoid excessive LLM calls)
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for i, (sid_a, fname_a) in enumerate(items):
        for j in range(i + 1, len(items)):
            sid_b, fname_b = items[j]
            if sid_b == sid_a:
                continue
            pairs.append(((sid_a, fname_a), (sid_b, fname_b)))

    if not pairs:
        return

    # Cap at 20 LLM calls to avoid runaway costs
    pairs = pairs[:20]

    consumed: set[tuple[str, str]] = set()
    for (sid_a, fname_a), (sid_b, fname_b) in pairs:
        if (sid_a, fname_a) in consumed or (sid_b, fname_b) in consumed:
            continue

        prof_a = profile_index.get(sid_a, {}).get(fname_a)
        prof_b = profile_index.get(sid_b, {}).get(fname_b)
        if not prof_a or not prof_b:
            continue

        match_result = _llm_match_pair(llm_client, sid_a, prof_a, sid_b, prof_b)
        if match_result is None:
            continue

        probability, reasoning = match_result
        if probability < 0.60:
            continue

        members = [(sid_a, fname_a), (sid_b, fname_b)]
        conflict = _detect_conflict(members, profile_index, mapping_index)
        group = AlignedFieldGroup(
            canonical_name=_pick_canonical_name(members, profile_index),
            semantic_type=_pick_semantic_type(members, mapping_index),
            source_mappings=dict(members),
            alignment_confidence=probability,
            conflict_type=conflict["type"],
            transformation_notes=conflict.get("notes", "") or reasoning,
        )
        aligned_groups.append(group)
        consumed.add((sid_a, fname_a))
        consumed.add((sid_b, fname_b))

    for pair in consumed:
        unmatched.discard(pair)


def _llm_match_pair(
    llm_client: object,
    source_a: str,
    prof_a: FieldProfile,
    source_b: str,
    prof_b: FieldProfile,
) -> tuple[float, str] | None:
    """Ask the LLM whether two fields represent the same concept.

    Returns ``(probability, reasoning)`` or ``None`` on failure.
    """
    import json as _json

    prompt = (
        "You are a data schema expert. Determine if these two fields from "
        "different data sources represent the same business concept.\n\n"
        f"Field A: \"{prof_a.field_name}\" from source \"{source_a}\"\n"
        f"  Type distribution: {prof_a.type_distribution}\n"
        f"  Fill rate: {prof_a.fill_rate:.2f}\n"
        f"  Cardinality: {prof_a.cardinality}\n"
        f"  Samples: {prof_a.value_samples[:5]}\n\n"
        f"Field B: \"{prof_b.field_name}\" from source \"{source_b}\"\n"
        f"  Type distribution: {prof_b.type_distribution}\n"
        f"  Fill rate: {prof_b.fill_rate:.2f}\n"
        f"  Cardinality: {prof_b.cardinality}\n"
        f"  Samples: {prof_b.value_samples[:5]}\n\n"
        'Respond with JSON only: {"match_probability": 0.0-1.0, "reasoning": "..."}'
    )

    try:
        response = llm_client.chat.completions.create(
            model=getattr(llm_client, "_model", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown fences)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = _json.loads(text)
        prob = float(data.get("match_probability", 0.0))
        reasoning = str(data.get("reasoning", ""))
        return (max(0.0, min(1.0, prob)), reasoning)
    except Exception as exc:
        logger.debug("LLM match failed for %s vs %s: %s", prof_a.field_name, prof_b.field_name, exc)
        return None


def _heuristic_similarity(a: FieldProfile, b: FieldProfile) -> float:
    """Score similarity between two field profiles (0-1).

    Components:
    - Jaccard similarity on type distribution keys (weight 0.4)
    - Fill rate proximity (weight 0.3)
    - Name token overlap (weight 0.3)
    """
    # Type distribution Jaccard
    keys_a = set(a.type_distribution.keys())
    keys_b = set(b.type_distribution.keys())
    jaccard = len(keys_a & keys_b) / len(keys_a | keys_b) if keys_a or keys_b else 0.0

    # Fill rate proximity
    fill_sim = 1.0 - abs(a.fill_rate - b.fill_rate)

    # Name token overlap
    tokens_a = set(re.split(r"[_\s]+", a.field_name.lower()))
    tokens_b = set(re.split(r"[_\s]+", b.field_name.lower()))
    name_sim = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a or tokens_b else 0.0

    return 0.4 * jaccard + 0.3 * fill_sim + 0.3 * name_sim


# ── Canonical name selection ─────────────────────────────────────────


def _pick_canonical_name(
    members: list[tuple[str, str]],
    profile_index: dict[str, dict[str, FieldProfile]],
) -> str:
    """Choose the canonical name for a group.

    Prefers the most common field name.  Ties broken by highest fill rate.
    """
    from collections import Counter

    name_counts = Counter(fname for _, fname in members)
    max_count = max(name_counts.values())
    candidates = [name for name, cnt in name_counts.items() if cnt == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Break tie by highest fill rate
    best_name = candidates[0]
    best_fill = 0.0
    for sid, fname in members:
        if fname not in candidates:
            continue
        prof = profile_index.get(sid, {}).get(fname)
        if prof and prof.fill_rate > best_fill:
            best_fill = prof.fill_rate
            best_name = fname
    return best_name


# ── Semantic type selection ──────────────────────────────────────────


def _pick_semantic_type(
    members: list[tuple[str, str]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
) -> str:
    """Pick the best semantic type for a group.

    Prefers the most common non-unknown type, then the one with highest
    confidence.
    """
    from collections import Counter

    types: list[tuple[str, float]] = []
    for sid, fname in members:
        sem = mapping_index.get(sid, {}).get(fname)
        if sem and sem.semantic_type and sem.semantic_type != "unknown":
            types.append((sem.semantic_type, sem.confidence))

    if not types:
        return "unknown"

    type_counts = Counter(t for t, _ in types)
    max_count = max(type_counts.values())
    candidates = [t for t, cnt in type_counts.items() if cnt == max_count]

    if len(candidates) == 1:
        return candidates[0]

    # Break tie by highest confidence
    best_type = candidates[0]
    best_conf = 0.0
    for t, conf in types:
        if t in candidates and conf > best_conf:
            best_conf = conf
            best_type = t
    return best_type


# ── Conflict detection ───────────────────────────────────────────────


def _check_unit_conflict(
    numeric_ranges: list[tuple[float, float]],
    members: list[tuple[str, str]],
) -> dict | None:
    """Check for unit mismatch (e.g., cents vs dollars) across numeric ranges."""
    if len(numeric_ranges) < 2:
        return None
    for i in range(len(numeric_ranges)):
        for j in range(i + 1, len(numeric_ranges)):
            r_a, r_b = numeric_ranges[i], numeric_ranges[j]
            mean_a = (r_a[0] + r_a[1]) / 2 if r_a[1] != r_a[0] else r_a[0]
            mean_b = (r_b[0] + r_b[1]) / 2 if r_b[1] != r_b[0] else r_b[0]
            if mean_a > 0 and mean_b > 0:
                ratio = max(mean_a, mean_b) / min(mean_a, mean_b)
                if ratio > 50:
                    sids = [sid for sid, _ in members]
                    return {
                        "type": "unit",
                        "notes": (
                            f"Numeric range ratio ~{ratio:.0f}x across sources "
                            f"({sids[0]} vs {sids[-1]}); possible unit mismatch "
                            f"(e.g., cents vs dollars)"
                        ),
                    }
    return None


def _detect_conflict(
    members: list[tuple[str, str]],
    profile_index: dict[str, dict[str, FieldProfile]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
) -> dict:
    """Detect the most significant conflict in an aligned group.

    Returns ``{"type": ..., "notes": ...}``.  Priorities:
    unit > type > semantics > naming > none.
    """
    names: set[str] = set()
    sem_types: set[str] = set()
    type_dists: list[dict[str, float]] = []
    numeric_ranges: list[tuple[float, float]] = []

    for sid, fname in members:
        names.add(fname)
        sem = mapping_index.get(sid, {}).get(fname)
        if sem and sem.semantic_type != "unknown":
            sem_types.add(sem.semantic_type)

        prof = profile_index.get(sid, {}).get(fname)
        if prof:
            type_dists.append(prof.type_distribution)
            stats = prof.statistical_summary
            if "min" in stats and "max" in stats:
                numeric_ranges.append((stats["min"], stats["max"]))

    # Unit conflict: numeric ranges differ by ~100x (cents vs dollars)
    unit = _check_unit_conflict(numeric_ranges, members)
    if unit is not None:
        return unit

    # Type conflict: dominant types differ significantly
    if len(type_dists) >= 2:
        dominant_types = [max(td, key=td.get) for td in type_dists if td]
        if len(set(dominant_types)) > 1:
            return {
                "type": "type",
                "notes": f"Type distributions differ: {', '.join(set(dominant_types))}",
            }

    # Semantics conflict: similar but not identical semantic types
    if len(sem_types) > 1:
        prefixes = {t.split(".")[0] for t in sem_types}
        if len(prefixes) == 1:
            return {
                "type": "semantics",
                "notes": f"Similar semantic types: {', '.join(sorted(sem_types))}",
            }
        return {
            "type": "semantics",
            "notes": f"Different semantic types: {', '.join(sorted(sem_types))}",
        }

    # Naming conflict: different names, same semantics
    if len(names) > 1:
        return {
            "type": "naming",
            "notes": f"Different field names: {', '.join(sorted(names))}",
        }

    return {"type": "none", "notes": ""}


# ── Confidence scoring ───────────────────────────────────────────────


def _exact_match_confidence(
    members: list[tuple[str, str]],
    profile_index: dict[str, dict[str, FieldProfile]],
) -> float:
    """Confidence for exact-name matches.

    Base = 0.95.  Penalise by (1 - avg fill_rate) * 0.1.
    """
    fill_rates: list[float] = []
    for sid, fname in members:
        prof = profile_index.get(sid, {}).get(fname)
        if prof:
            fill_rates.append(prof.fill_rate)

    avg_fill = sum(fill_rates) / len(fill_rates) if fill_rates else 0.5
    return min(1.0, 0.95 - (1.0 - avg_fill) * 0.1)


def _semantic_match_confidence(
    members: list[tuple[str, str]],
    mapping_index: dict[str, dict[str, SemanticMapping]],
) -> float:
    """Confidence for semantic-type matches.

    Average of the per-field semantic mapping confidences, capped at 0.90.
    """
    confidences: list[float] = []
    for sid, fname in members:
        sem = mapping_index.get(sid, {}).get(fname)
        if sem:
            confidences.append(sem.confidence)

    avg = sum(confidences) / len(confidences) if confidences else 0.5
    return min(0.90, avg)


# ── Threshold resolution ─────────────────────────────────────────────


def _get_thresholds(threshold_engine: object | None) -> tuple[float, float]:
    """Return (auto_approve_threshold, review_threshold)."""
    if threshold_engine is not None:
        try:
            config = threshold_engine.get_thresholds(
                decision_type="field_alignment",
            )
            return config.auto_approve_threshold, config.review_threshold
        except Exception:
            pass
    return _AUTO_APPROVE_THRESHOLD, _NEEDS_REVIEW_THRESHOLD


def _classify_review(
    confidence: float,
    auto_thresh: float,
    review_thresh: float,
) -> str:
    """Map confidence to a review status string."""
    if confidence >= auto_thresh:
        return "auto_approved"
    if confidence >= review_thresh:
        return "needs_review"
    return "rejected"


# ── Prediction logging ───────────────────────────────────────────────


def _record_predictions(
    groups: list[AlignedFieldGroup],
    source_ids: list[str],
    prediction_log: object,
    tenant_id: str,
) -> None:
    """Record each alignment decision in the prediction log."""
    try:
        from attestdb.calibration.prediction_log import PredictionRecord
    except ImportError:
        logger.debug("calibration module not available; skipping prediction logging")
        return

    for group in groups:
        for source_id in group.source_mappings:
            field_name = group.source_mappings[source_id]
            record = PredictionRecord(
                prediction_id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                decision_type="field_alignment",
                source_id=source_id,
                field_semantic_type=group.semantic_type,
                predicted_confidence=group.alignment_confidence,
                predicted_value=f"{field_name} -> {group.canonical_name}",
                review_outcome="pending",
                corrected_value=None,
                reviewed_at=None,
                reviewed_by=None,
                created_at=time.time(),
            )
            try:
                prediction_log.record(record)
            except Exception as exc:
                logger.warning("Failed to record prediction: %s", exc)
