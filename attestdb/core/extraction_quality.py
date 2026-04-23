"""Extraction-quality gate for LLM-extracted claims.

Rejects or flags claims whose evidence doesn't support a first-class
assertion. Common failure modes:

  * parenthetical comparisons ("like in the Cosby case, …")
  * legal citations ("as held in Nadeen Cool v. Vitek, …")
  * evidence that's entirely parenthetical ("(confidential)")

Generic filter — applies to any LLM-extracted corpus (legal briefs,
scientific papers, business docs) where sources cite or compare without
making a first-party assertion.
"""

from __future__ import annotations

from typing import Literal

from attestdb.core.types import ClaimInput

QualityVerdict = Literal["accept", "flag", "reject"]

# Citation markers — plain lowercase substrings, not regex. Order is not
# significant (first match wins at the rule level, not marker level).
_CITATION_MARKERS: tuple[str, ...] = (
    " v. ",
    " vs. ",
    "as held in",
    "like in the",
    "compare ",
    "see also",
    "cf. ",
    "see, e.g.,",
)

_CITATION_PROXIMITY_CHARS = 40
_MIN_TOKENS = 5


def _parenthetical_spans(text: str) -> list[tuple[int, int]]:
    """Return char spans (start_inclusive, end_exclusive) of `( ... )` groups.

    Simple scanner, no nesting handling — unmatched `(` is skipped.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "(":
            close = text.find(")", i + 1)
            if close == -1:
                break
            spans.append((i, close + 1))
            i = close + 1
        else:
            i += 1
    return spans


def _is_entirely_parenthetical(text: str) -> bool:
    """True if `text` (stripped) is wrapped in `( … )` or `— … —`."""
    s = text.strip()
    if len(s) < 2:
        return False
    if s.startswith("(") and s.endswith(")"):
        # Ensure the outer parens match — no mid-string close.
        depth = 0
        for idx, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and idx != len(s) - 1:
                    return False
        return depth == 0
    if s.startswith("—") and s.endswith("—") and len(s) > 2:
        # Em-dash wrap; ensure no interior em-dash closes early.
        return s.count("—") == 2
    return False


def _id_variants(ent_id: str) -> list[str]:
    """Entity IDs are normalized with underscores; evidence text uses spaces.

    Return both the underscore form and a space-separated variant so
    substring checks match canonical IDs against raw text.
    """
    if not ent_id:
        return []
    low = ent_id.lower()
    variants = [low]
    if "_" in low:
        variants.append(low.replace("_", " "))
    return variants


def _id_in_parenthetical_only(text: str, ent_id: str) -> bool:
    """True if `ent_id` appears in `text` AND every occurrence is inside a `( … )` span.

    Case-insensitive substring check. Returns False if ent_id is not
    present at all (i.e. there's nothing to complain about).
    """
    if not ent_id:
        return False
    low_text = text.lower()
    variants = _id_variants(ent_id)
    low_id = next((v for v in variants if v in low_text), None)
    if low_id is None:
        return False
    spans = _parenthetical_spans(text)
    # Find all occurrences; every one must lie inside some paren span.
    start = 0
    any_outside = False
    any_found = False
    while True:
        idx = low_text.find(low_id, start)
        if idx == -1:
            break
        any_found = True
        end = idx + len(low_id)
        in_paren = any(s <= idx and end <= e for s, e in spans)
        if not in_paren:
            any_outside = True
            break
        start = end
    return any_found and not any_outside


def _citation_near_entity(text: str, ent_id: str) -> bool:
    """True if any citation marker appears within _CITATION_PROXIMITY_CHARS of ent_id."""
    if not ent_id:
        return False
    low_text = text.lower()
    variants = _id_variants(ent_id)
    low_id = next((v for v in variants if v in low_text), None)
    if low_id is None:
        return False
    id_idx = low_text.find(low_id)
    id_end = id_idx + len(low_id)
    for marker in _CITATION_MARKERS:
        m_idx = low_text.find(marker)
        while m_idx != -1:
            m_end = m_idx + len(marker)
            # distance is gap between nearest edges
            if m_idx >= id_end:
                dist = m_idx - id_end
            elif id_idx >= m_end:
                dist = id_idx - m_end
            else:
                dist = 0  # overlap
            if dist <= _CITATION_PROXIMITY_CHARS:
                return True
            m_idx = low_text.find(marker, m_idx + 1)
    return False


def _evaluate(evidence_text: str, subj_id: str, obj_id: str) -> tuple[QualityVerdict, str]:
    """Shared evaluator used by both ClaimInput and Claim variants."""
    text = evidence_text or ""

    # Rule 1: too short
    if len(text.split()) < _MIN_TOKENS:
        return "flag", "evidence_too_short"

    # Rule 3 before Rule 2: entity-in-parens (reject) is a stronger signal
    # than "whole evidence is parenthetical" (flag) and should win when both
    # apply (spec test fixture: "(See Cosby v. Constand...)" → reject).
    if _id_in_parenthetical_only(text, subj_id) or _id_in_parenthetical_only(text, obj_id):
        return "reject", "entity_only_in_parenthetical"

    # Rule 2: entirely parenthetical (no identified entity inside)
    if _is_entirely_parenthetical(text):
        return "flag", "parenthetical_only"

    # Rule 4: citation marker near entity
    if _citation_near_entity(text, subj_id) or _citation_near_entity(text, obj_id):
        return "flag", "citation_context"

    return "accept", "ok"


def is_substantive_claim(claim_input: ClaimInput) -> tuple[QualityVerdict, str]:
    """Return (verdict, reason). Reason is a short tag like 'parenthetical_only'."""
    payload = claim_input.payload or {}
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    evidence_text = data.get("evidence_text", "") if isinstance(data, dict) else ""
    subj_id = claim_input.subject[0] if claim_input.subject else ""
    obj_id = claim_input.object[0] if claim_input.object else ""
    return _evaluate(evidence_text, subj_id, obj_id)


def audit_db(db, apply: bool) -> dict:
    """Audit extraction quality of every claim in an open AttestDB.

    Returns {counts, reason_counts, samples, applied}. Does not open/close db.
    With ``apply=True``, writes ``extraction_quality="flag:<reason>"`` into
    each flagged claim's payload (non-destructive — rejected claims untouched).
    """
    from collections import Counter

    from attestdb.core.types import Payload, claim_evidence_text

    counts: Counter = Counter()
    reason_counts: Counter = Counter()
    samples: dict = {"reject": [], "flag": []}
    applied = 0

    for claim in db.iter_claims():
        verdict, reason = is_substantive_claim_obj(claim)
        counts[verdict] += 1
        if verdict == "accept":
            continue
        reason_counts[reason] += 1
        ev = claim_evidence_text(claim)
        if len(samples[verdict]) < 20:
            samples[verdict].append((claim.claim_id, reason, ev[:160]))

        if apply and verdict == "flag":
            if claim.payload is None:
                claim.payload = Payload(schema_ref="", data={})
            if not isinstance(claim.payload.data, dict):
                claim.payload.data = {}
            tag = f"flag:{reason}"
            if claim.payload.data.get("extraction_quality") != tag:
                claim.payload.data["extraction_quality"] = tag
                db._store.insert_claim(claim)
                applied += 1

    return {
        "counts": dict(counts),
        "reason_counts": dict(reason_counts),
        "samples": samples,
        "applied": applied,
    }


def print_audit_report(result: dict, apply: bool) -> None:
    """Format an audit_db() result to stdout."""
    counts = result["counts"]
    reason_counts = result["reason_counts"]
    samples = result["samples"]
    applied = result["applied"]
    total = sum(counts.values())
    print(f"Scanned {total} claims")
    print(f"  accept: {counts.get('accept', 0)}")
    print(f"  flag:   {counts.get('flag', 0)}")
    print(f"  reject: {counts.get('reject', 0)}")
    if reason_counts:
        print("\nBy reason:")
        for reason, n in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {reason}: {n}")
    for verdict in ("reject", "flag"):
        if samples.get(verdict):
            print(f"\nSample {verdict} (up to 20):")
            for cid, reason, ev in samples[verdict]:
                print(f"  [{reason}] {cid} — {ev!r}")
    if apply:
        print(f"\nApplied extraction_quality tag to {applied} claims.")
    else:
        print("\n(dry run; pass --apply to tag flagged claims)")


def is_substantive_claim_obj(claim) -> tuple[QualityVerdict, str]:
    """Variant that accepts a stored Claim (with .payload.data and .subject.id)."""
    from attestdb.core.types import claim_evidence_text

    evidence_text = claim_evidence_text(claim)
    subj = getattr(claim, "subject", None)
    obj = getattr(claim, "object", None)
    subj_id = getattr(subj, "id", "") if subj is not None else ""
    obj_id = getattr(obj, "id", "") if obj is not None else ""
    return _evaluate(evidence_text, subj_id, obj_id)
