"""Claim-equivalence grouping: collapse predicate-synonym duplicates.

When two claims describe the same event with different predicates — e.g.
``Epstein wired_money_to Maxwell`` and ``Epstein transferred_money_to Maxwell``
on the same date from the same source doc — they are the same logical fact
expressed with surface-form vocabulary drift. This primitive groups them so
downstream aggregations (money rollups, corroboration counts) don't double
count.

Non-destructive: the raw claim log is untouched. ``EquivalenceGroup`` rows
are returned as derived findings. The optional build script in
``scripts/build_claim_equivalence_groups.py`` can write ``same_claim_as``
meta-claims linking members.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from attestdb.core.predicate_salience import (
    DEFAULT_JOURNALISM_REGISTRY, PredicateSalienceRegistry,
)
from attestdb.core.types import claim_evidence_text

if TYPE_CHECKING:
    from attestdb.core.types import Claim
    from attestdb.infrastructure.attest_db import AttestDB


# ────────────────────────────────────────────────────────────────────
# Result dataclass
# ────────────────────────────────────────────────────────────────────

@dataclass
class EquivalenceGroup:
    group_id: str
    claim_ids: list[str]
    subject: str
    object: Optional[str]
    numeric_value: Optional[float]
    predicate_synonym_group: str
    representative_claim_id: str
    member_predicates: list[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────
# Numeric value extraction
# ────────────────────────────────────────────────────────────────────

_DOLLAR_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m\b|k\b|b\b)?",
    re.IGNORECASE,
)


def _parse_amount(text: str) -> Optional[float]:
    if not text:
        return None
    largest: Optional[float] = None
    for m in _DOLLAR_RE.finditer(str(text)):
        num = m.group(1).replace(",", "")
        try:
            val = float(num)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        if unit.startswith("b"):
            val *= 1_000_000_000
        elif unit.startswith("m"):
            val *= 1_000_000
        elif unit.startswith("k") or unit == "thousand":
            val *= 1_000
        if largest is None or val > largest:
            largest = val
    return largest


def _claim_numeric_value(c: "Claim") -> Optional[float]:
    """Look for a numeric value in the object id, evidence text, and
    common payload keys. None if no parseable amount found.
    """
    amt = _parse_amount(c.object.id)
    if amt is not None:
        return amt
    amt = _parse_amount(claim_evidence_text(c))
    if amt is not None:
        return amt
    try:
        data = c.payload.data if (c.payload and hasattr(c.payload, "data")) else {}
    except Exception:
        data = {}
    if isinstance(data, dict):
        for k in ("amount", "value", "dollars", "figure"):
            if k in data:
                amt = _parse_amount(str(data[k]))
                if amt is not None:
                    return amt
    return None


# ────────────────────────────────────────────────────────────────────
# Grouping
# ────────────────────────────────────────────────────────────────────

_ONE_DAY_NS = 86_400 * 1_000_000_000


def _within_tolerance(a: Optional[float], b: Optional[float], rel_tol: float) -> bool:
    """Both-None → True (non-numeric grouping). Mixed → False. Both
    present → relative tolerance or equal.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom == 0:
        return True
    return abs(a - b) / denom <= rel_tol


def _doc_key(source_id: str) -> str:
    if not source_id:
        return ""
    for tok in source_id.replace(":", " ").replace("/", " ").replace("#", " ").split():
        if not tok.startswith("chunk"):
            return tok
    return source_id.split(":")[0].split("#")[0]


def _group_id_for(claim_ids: list[str]) -> str:
    h = hashlib.sha256()
    for cid in sorted(claim_ids):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
    return "equiv_" + h.hexdigest()[:16]


def _pick_representative(members: list["Claim"]) -> "Claim":
    # Highest confidence, tiebreak by earliest claim_id lexicographically.
    return sorted(members, key=lambda c: (-float(c.confidence or 0.0), c.claim_id))[0]


def find_equivalent_claim_groups(
    db: "AttestDB",
    *,
    predicate_synonyms: Optional[dict[str, set[str]]] = None,
    numeric_tolerance: float = 0.01,
    time_window_days: int = 1,
    registry: PredicateSalienceRegistry = DEFAULT_JOURNALISM_REGISTRY,
) -> list[EquivalenceGroup]:
    """Return groups of claims that describe the same underlying event.

    Two claims belong to the same group iff:
      - they share a subject id,
      - they share an object id OR have numeric values within
        ``numeric_tolerance`` relative,
      - their predicates belong to the same synonym class,
      - their timestamps overlap within ``time_window_days`` (when both
        have timestamps), or they share a source document.

    If ``predicate_synonyms`` is None, the registry's ``synonym_groups()``
    are used.
    """
    synonyms = predicate_synonyms or registry.synonym_groups()
    if not synonyms:
        return []

    # predicate_id → group_name
    pred_to_group: dict[str, str] = {}
    for group_name, preds in synonyms.items():
        for p in preds:
            pred_to_group[p] = group_name

    # Collect candidate claims per synonym group.
    claims_by_group: dict[str, list["Claim"]] = defaultdict(list)
    for pid, group_name in pred_to_group.items():
        try:
            for c in db.claims_for_predicate(pid) or []:
                claims_by_group[group_name].append(c)
        except Exception:
            continue

    window_ns = time_window_days * _ONE_DAY_NS
    results: list[EquivalenceGroup] = []

    for group_name, claims in claims_by_group.items():
        # Bucket by subject first, then compare pairs within each subject.
        by_subject: dict[str, list["Claim"]] = defaultdict(list)
        for c in claims:
            by_subject[c.subject.id].append(c)

        for subject_id, subject_claims in by_subject.items():
            # Union-find style clustering over candidate pairs.
            n = len(subject_claims)
            if n < 2:
                continue
            parent = list(range(n))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            # Pre-extract comparison keys.
            numeric_values = [_claim_numeric_value(c) for c in subject_claims]
            object_ids = [c.object.id for c in subject_claims]
            timestamps = [int(getattr(c, "timestamp", 0) or 0) for c in subject_claims]
            doc_keys = [
                _doc_key(getattr(c.provenance, "source_id", "") or "")
                for c in subject_claims
            ]

            for i in range(n):
                for j in range(i + 1, n):
                    # Object-id match OR numeric match within tolerance.
                    same_object = (
                        object_ids[i] == object_ids[j]
                        and numeric_values[i] is None
                        and numeric_values[j] is None
                    )
                    same_numeric = (
                        numeric_values[i] is not None
                        and numeric_values[j] is not None
                        and _within_tolerance(
                            numeric_values[i], numeric_values[j], numeric_tolerance,
                        )
                    )
                    # (numeric-within-tolerance is handled above; same object
                    # alone is not sufficient when values are far apart.)
                    # If one has a number and one doesn't but object ids match,
                    # treat as the same (e.g. "$20M" object + payload-numeric).
                    if (
                        object_ids[i] == object_ids[j]
                        and (numeric_values[i] is None) != (numeric_values[j] is None)
                    ):
                        same_object = True

                    if not (same_object or same_numeric):
                        continue

                    # Temporal overlap or shared source.
                    ts_i, ts_j = timestamps[i], timestamps[j]
                    if ts_i and ts_j:
                        if abs(ts_i - ts_j) > window_ns:
                            continue
                    else:
                        # Fall back to shared source doc.
                        if not (doc_keys[i] and doc_keys[j] and doc_keys[i] == doc_keys[j]):
                            continue

                    union(i, j)

            # Materialize clusters of size ≥ 2.
            clusters: dict[int, list[int]] = defaultdict(list)
            for idx in range(n):
                clusters[find(idx)].append(idx)

            for idxs in clusters.values():
                if len(idxs) < 2:
                    continue
                members = [subject_claims[k] for k in idxs]
                rep = _pick_representative(members)
                # Determine representative numeric value / object.
                rep_numeric = _claim_numeric_value(rep)
                rep_object = rep.object.id if rep_numeric is None else None
                if rep_numeric is None and all(
                    _claim_numeric_value(m) is None for m in members
                ):
                    rep_object = members[0].object.id
                claim_ids = [m.claim_id for m in members]
                results.append(EquivalenceGroup(
                    group_id=_group_id_for(claim_ids),
                    claim_ids=sorted(claim_ids),
                    subject=subject_id,
                    object=rep_object,
                    numeric_value=rep_numeric,
                    predicate_synonym_group=group_name,
                    representative_claim_id=rep.claim_id,
                    member_predicates=sorted({m.predicate.id for m in members}),
                ))

    # Stable order: by synonym group, then representative claim id.
    results.sort(key=lambda g: (g.predicate_synonym_group, g.representative_claim_id))
    return results


# ────────────────────────────────────────────────────────────────────
# Meta-claim writer (non-destructive)
# ────────────────────────────────────────────────────────────────────

SAME_CLAIM_AS_PREDICATE = "same_claim_as"


def write_same_claim_as_meta(db: "AttestDB", groups: list[EquivalenceGroup]) -> int:
    """Append ``same_claim_as`` meta-claims linking each non-representative
    member to its group representative. Returns number of meta-claims written.
    """
    written = 0
    for g in groups:
        for cid in g.claim_ids:
            if cid == g.representative_claim_id:
                continue
            db.ingest(
                subject=(cid, "claim"),
                predicate=(SAME_CLAIM_AS_PREDICATE, "equivalence"),
                object=(g.representative_claim_id, "claim"),
                provenance={
                    "source_type": "claim_equivalence",
                    "source_id": g.group_id,
                    "method": "find_equivalent_claim_groups",
                },
                payload={
                    "schema_ref": "equivalence.v1",
                    "data": {
                        "synonym_group": g.predicate_synonym_group,
                        "member_predicates": g.member_predicates,
                    },
                },
            )
            written += 1
    return written
