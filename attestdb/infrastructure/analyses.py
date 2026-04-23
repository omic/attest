"""Generic narrative analyses over a claim graph.

Each function here is corpus-agnostic. It takes:
- ``db``: an ``AttestDB`` instance
- ``registry``: a ``PredicateSalienceRegistry`` that knows what
  predicates mean in the caller's domain (categories, weights, labels)
- ``quality_filter``: an optional ``EntityQualityFilter`` that decides
  whether an entity is a real named thing vs. a placeholder

…and returns typed dataclasses describing the finding. The demo or
agent layer formats the result for display.

The five primitives map to common narrative questions any claim corpus
faces:

- ``entity_dossier`` — "give me everything that matters about X,
  organized by category, with prosecution-status-style summary."
- ``entities_with_category_gap`` — "who has claims in category A but
  none in category B?" (e.g. accused + not prosecuted; bug-filed +
  not closed; lead-contacted + no reply)
- ``top_claims_by_numeric_value`` — "biggest dollar amounts / largest
  payments / highest-value transactions"
- ``contradicted_assertions`` — "find statements paired with
  contradicting claims about the same fact" (e.g. denials with
  countervailing record; sales claims with negative customer
  feedback)
- ``non_obvious_associates`` — "who appears in the same documents as
  the pivot entity often, but isn't already in the obvious set?"
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from attestdb.core.entity_quality import EntityQualityFilter
from attestdb.core.predicate_salience import (
    DEFAULT_JOURNALISM_REGISTRY, PredicateSalienceRegistry,
)

from attestdb.core.types import claim_evidence_text


# ────────────────────────────────────────────────────────────────────
# Equivalence-group default resolution
# ────────────────────────────────────────────────────────────────────

def _resolve_equivalence_groups(
    db: "AttestDB",
    equivalence_groups,
    *,
    numeric_tolerance: float = 0.05,
    time_window_days: int = 7,
) -> list:
    """Decide which equivalence groups to apply for an aggregator.

    Rules:
      - Explicit ``False`` → dedup disabled (empty list).
      - Env var ``ATTEST_DISABLE_EQUIVALENCE=1`` → globally disabled.
      - ``None`` (default) → compute via ``find_equivalent_claim_groups``.
      - A concrete list → use as-is.
    """
    if equivalence_groups is False:
        return []
    if os.environ.get("ATTEST_DISABLE_EQUIVALENCE") == "1":
        return []
    if equivalence_groups is None:
        from attestdb.infrastructure.claim_equivalence import (
            find_equivalent_claim_groups,
        )
        try:
            return find_equivalent_claim_groups(
                db,
                numeric_tolerance=numeric_tolerance,
                time_window_days=time_window_days,
            )
        except Exception:
            return []
    return list(equivalence_groups)

if TYPE_CHECKING:
    from attestdb.core.types import Claim
    from attestdb.infrastructure.attest_db import AttestDB


# ────────────────────────────────────────────────────────────────────
# Result dataclasses
# ────────────────────────────────────────────────────────────────────

@dataclass
class DossierSection:
    category: str
    label: str
    icon: str
    total: int
    claims: list  # of Claim


@dataclass
class StatusSummary:
    """A status determined by which categories the entity has claims in.
    For example, in the journalism domain: prosecuted (has legal-process
    claims), accused-but-not-prosecuted (has allegation claims but no
    legal-process claims), or quiet (neither).
    """
    label: str
    is_resolved: bool       # True when the "completed" status applies
    detail_actions: list[dict]  # the specific claims supporting the status


@dataclass
class EntityDossier:
    entity_id: str
    label: str
    entity_type: str
    total_claims: int
    sections: list[DossierSection]
    status: Optional[StatusSummary] = None


@dataclass
class CategoryGapFinding:
    entity_id: str
    label: str
    has_count: int           # number of claims in the "has" categories
    has_predicates: list[str]
    sample_claims: list      # top supporting claims by salience
    salience: float
    source_count: int


@dataclass
class NumericFinding:
    amount: float
    amount_label: str        # "$30.0M", "1.2K", etc.
    subject_id: str
    subject_label: str
    object_id: str
    object_label: str
    predicate: str
    verb_phrase: str
    claim: object            # raw claim
    source_count: int = 1    # corroborating doc count after dedupe


@dataclass
class AssertionContradictionFinding:
    subject_id: str
    subject_label: str
    assertion_predicate: str
    assertion_label: str
    assertion_object: str
    assertion_quote: str
    assertion_source: str    # bates / source_id
    contradicting_count: int
    contradicting_sample: list  # of Claim


@dataclass
class AssociateFinding:
    entity_id: str
    label: str
    shared_docs: int
    pivot_doc_total: int
    their_total_claims: int
    specificity: float       # shared_docs / their_total_claims


# ────────────────────────────────────────────────────────────────────
# Primitive 1: entity_dossier
# ────────────────────────────────────────────────────────────────────

def entity_dossier(
    db: "AttestDB",
    entity_id: str,
    *,
    registry: PredicateSalienceRegistry = DEFAULT_JOURNALISM_REGISTRY,
    limit_per_category: int = 8,
    status_categories: Optional[dict[str, list[str]]] = None,
) -> EntityDossier:
    """Return all of an entity's claims, grouped by predicate category
    (per the registry), each section salience-sorted.

    ``status_categories`` (optional) is a mapping of status-key →
    list of predicate ids that mark "this status is established."
    For the journalism case, pass {"prosecuted": ["pleaded_guilty_to",
    "convicted_of", "indicted_for", ...]}; the dossier's ``.status``
    field then reflects whether any of those predicates fired against
    the entity.
    """
    es = db.get_entity(entity_id)
    if es is None:
        return EntityDossier(entity_id=entity_id, label=entity_id, entity_type="",
                             total_claims=0, sections=[], status=None)

    raw_claims = db.claims_for(entity_id, min_confidence=0.0) or []

    # Build per-claim corroboration count via document-level grouping.
    def _doc_key(src: str) -> str:
        if not src:
            return ""
        # Strip chunk suffixes (":chunk_3", "#chunk2") so multiple chunks
        # of the same source doc count as one corroboration source.
        for tok in src.replace(":", " ").replace("/", " ").replace("#", " ").split():
            if not tok.startswith("chunk"):
                return tok
        return src.split(":")[0].split("#")[0]

    corrob: dict[tuple, set[str]] = defaultdict(set)
    for c in raw_claims:
        key = (c.subject.id, c.predicate.id, c.object.id)
        d = _doc_key(getattr(c.provenance, "source_id", "") or "")
        if d:
            corrob[key].add(d)

    # Group by category
    grouped: dict[str, list[tuple[float, object]]] = defaultdict(list)
    for c in raw_claims:
        cat, _w, _lbl = registry.meta(c.predicate.id)
        key = (c.subject.id, c.predicate.id, c.object.id)
        cc = len(corrob.get(key, set()))
        evi = claim_evidence_text(c).strip()
        sal = registry.claim_salience(c.predicate.id, cc, bool(evi), source_diversity=cc)
        grouped[cat].append((sal, c))

    sections: list[DossierSection] = []
    for cat_spec in registry.categories():
        items = grouped.get(cat_spec.key, [])
        items.sort(key=lambda x: -x[0])
        if not items:
            continue
        sections.append(DossierSection(
            category=cat_spec.key,
            label=cat_spec.label,
            icon=cat_spec.icon,
            total=len(items),
            claims=[c for _, c in items[:limit_per_category]],
        ))

    # Status summary
    status: Optional[StatusSummary] = None
    if status_categories:
        # Examine the entity's *outgoing* claims (where it's the subject)
        # for status-defining predicates.
        actions: list[dict] = []
        established: list[str] = []
        for status_label, preds in status_categories.items():
            for c in raw_claims:
                if c.subject.id != entity_id:
                    continue
                if c.predicate.id in preds:
                    actions.append({
                        "status": status_label,
                        "predicate": c.predicate.id,
                        "object": c.object.id,
                    })
                    if status_label not in established:
                        established.append(status_label)
        if established:
            status = StatusSummary(
                label=", ".join(established),
                is_resolved=True,
                detail_actions=actions[:8],
            )
        else:
            status = StatusSummary(
                label="no qualifying status claims in the record",
                is_resolved=False,
                detail_actions=[],
            )

    return EntityDossier(
        entity_id=entity_id,
        label=es.name or entity_id,
        entity_type=es.entity_type or "",
        total_claims=len(raw_claims),
        sections=sections,
        status=status,
    )


# ────────────────────────────────────────────────────────────────────
# Primitive 2: entities_with_category_gap
# ────────────────────────────────────────────────────────────────────

def entities_with_category_gap(
    db: "AttestDB",
    *,
    has_predicates: list[str],
    lacks_predicates: list[str],
    role: str = "subject",
    min_count: int = 2,
    quality_filter: Optional[EntityQualityFilter] = None,
    registry: PredicateSalienceRegistry = DEFAULT_JOURNALISM_REGISTRY,
    exclude_entity_ids: Optional[set[str]] = None,
    skip_entity: Optional[callable] = None,
    top_k: int = 50,
    equivalence_groups: Union[list, None, bool] = None,
) -> list[CategoryGapFinding]:
    """Find entities with ≥``min_count`` claims in ``has_predicates`` and
    zero claims in ``lacks_predicates``.

    ``role`` controls which side of the claim makes the entity an actor:
    - "subject" — entity must be the subject of has_predicates claims
    - "object" — entity must be the object
    - "either" — entity in either role counts

    ``skip_entity(entity_id, claims_for_entity) -> bool`` lets the caller
    apply custom logic (e.g. "skip if entity also appears as a victim").

    Returns ranked list of findings (highest salience first).
    """
    # Resolve equivalence groups (on by default): non-representative members
    # of a synonym-predicate group collapse into the representative so
    # ``paid`` + ``wired_money_to`` about the same (subject, object) don't
    # inflate has_count.
    eg = _resolve_equivalence_groups(db, equivalence_groups)
    _gap_rep_of: dict[str, str] = {}
    for g in eg or []:
        for cid in getattr(g, "claim_ids", []) or []:
            _gap_rep_of[cid] = g.representative_claim_id

    # Bucket allegation-style claims by candidate actor entity.
    by_actor: dict[str, list] = defaultdict(list)
    for pid in has_predicates:
        try:
            for c in db.claims_for_predicate(pid) or []:
                if _gap_rep_of and c.claim_id in _gap_rep_of and _gap_rep_of[c.claim_id] != c.claim_id:
                    continue
                if role in ("subject", "either"):
                    by_actor[c.subject.id].append(c)
                if role in ("object", "either"):
                    if c.object.id != c.subject.id:  # don't double-count self-refs
                        by_actor[c.object.id].append(c)
        except Exception:
            continue

    excluded = exclude_entity_ids or set()
    findings: list[CategoryGapFinding] = []
    for eid, has_claims in by_actor.items():
        if eid in excluded:
            continue
        if len(has_claims) < min_count:
            continue
        try:
            es = db.get_entity(eid)
        except Exception:
            continue
        if quality_filter and not quality_filter.is_substantive(es):
            continue
        # Exclude entities with any "lacks" predicate against them
        # (entity as subject — i.e. the entity took the action).
        try:
            entity_claims = db.claims_for(eid, min_confidence=0.0) or []
        except Exception:
            entity_claims = []
        has_lacks = False
        for ec in entity_claims:
            if ec.subject.id == eid and ec.predicate.id in set(lacks_predicates):
                has_lacks = True
                break
        if has_lacks:
            continue
        if skip_entity and skip_entity(eid, entity_claims):
            continue
        # Tally
        sources = set()
        preds = set()
        for c in has_claims:
            src = getattr(c.provenance, "source_id", "") or ""
            if src:
                sources.add(src)
            preds.add(c.predicate.id)
        salience = sum(
            registry.claim_salience(
                c.predicate.id,
                getattr(c, "corroboration_count", 0) or 0,
                bool(claim_evidence_text(c).strip()),
            )
            for c in has_claims
        )
        # Top sample claims by salience
        sorted_claims = sorted(
            has_claims,
            key=lambda c: -registry.claim_salience(
                c.predicate.id,
                getattr(c, "corroboration_count", 0) or 0,
                bool(claim_evidence_text(c).strip()),
            ),
        )
        findings.append(CategoryGapFinding(
            entity_id=eid,
            label=(es.name if es else eid) or eid,
            has_count=len(has_claims),
            has_predicates=sorted(preds),
            sample_claims=sorted_claims[:3],
            salience=salience,
            source_count=len(sources),
        ))

    findings.sort(key=lambda f: (-f.salience, -f.has_count))
    return findings[:top_k]


# ────────────────────────────────────────────────────────────────────
# Primitive 3: top_claims_by_numeric_value
# ────────────────────────────────────────────────────────────────────

_DOLLAR_RE = re.compile(
    r"\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|m\b|k\b|b\b)?",
    re.IGNORECASE,
)


def _parse_amount(text: str, min_value: float = 1000) -> Optional[float]:
    """Best-effort dollar amount extractor. Returns the largest matching
    amount in ``text``, or None if nothing ≥ ``min_value`` parses.
    """
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
        if val >= min_value and (largest is None or val > largest):
            largest = val
    return largest


def _format_money(v: float) -> str:
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:,.0f}"


def top_claims_by_numeric_value(
    db: "AttestDB",
    *,
    predicate_ids: list[str],
    sources: tuple[str, ...] = ("object", "evidence", "payload"),
    min_value: float = 1000,
    payload_keys: tuple[str, ...] = ("amount", "value", "dollars", "figure"),
    quality_filter: Optional[EntityQualityFilter] = None,
    registry: PredicateSalienceRegistry = DEFAULT_JOURNALISM_REGISTRY,
    top_k: int = 20,
    require_subject_quality: bool = True,
    equivalence_groups: Union[list, None, bool] = None,
) -> list[NumericFinding]:
    """Top-N claims by parsed numeric value across the given predicates.

    Looks at the object id, the evidence_text, and named payload keys
    in priority order (configurable via ``sources``). Dedupes identical
    (subject, predicate, object, amount) tuples reported by multiple
    sources — same payment, multiple docs.

    Synonym-predicate dedup is **on by default**: ``$20M wired_money_to``
    + ``$20M transferred_money_to`` on the same doc collapse into one row,
    with ``source_count`` counting collapsed synonyms. Escape hatches:

      - ``equivalence_groups=False`` disables dedup for this call.
      - env ``ATTEST_DISABLE_EQUIVALENCE=1`` disables globally.
      - An explicit list of ``EquivalenceGroup`` values is used verbatim.
    """
    equivalence_groups = _resolve_equivalence_groups(db, equivalence_groups)
    # Build a claim_id → representative_claim_id map from equivalence groups.
    rep_of: dict[str, str] = {}
    synonym_sources: dict[str, int] = {}
    if equivalence_groups:
        for g in equivalence_groups:
            for cid in getattr(g, "claim_ids", []) or []:
                rep_of[cid] = g.representative_claim_id
            synonym_sources[g.representative_claim_id] = len(g.claim_ids)
    candidates: list[NumericFinding] = []
    for pid in predicate_ids:
        try:
            claims = db.claims_for_predicate(pid) or []
        except Exception:
            continue
        for c in claims:
            # Skip non-representative members of an equivalence group;
            # their representative carries the collapsed tally.
            if rep_of and c.claim_id in rep_of and rep_of[c.claim_id] != c.claim_id:
                continue
            amount: Optional[float] = None
            for src in sources:
                if amount is not None:
                    break
                if src == "object":
                    amount = _parse_amount(c.object.id, min_value=min_value)
                elif src == "evidence":
                    amount = _parse_amount(claim_evidence_text(c), min_value=min_value)
                elif src == "payload":
                    payload_data = {}
                    try:
                        payload_data = (c.payload.data if hasattr(c.payload, "data") else None) or {}
                    except Exception:
                        pass
                    for k in payload_keys:
                        if k in payload_data:
                            amount = _parse_amount(str(payload_data[k]), min_value=min_value)
                            if amount is not None:
                                break
            if amount is None:
                continue
            try:
                sl = db.get_entity(c.subject.id)
                ol = db.get_entity(c.object.id)
            except Exception:
                sl = ol = None
            if require_subject_quality and quality_filter and not quality_filter.is_substantive(sl):
                continue
            initial_source_count = synonym_sources.get(c.claim_id, 1)
            candidates.append(NumericFinding(
                amount=amount,
                amount_label=_format_money(amount),
                subject_id=c.subject.id,
                subject_label=(sl.name if sl else c.subject.id) or c.subject.id,
                object_id=c.object.id,
                object_label=(ol.name if ol else c.object.id) or c.object.id,
                predicate=c.predicate.id,
                verb_phrase=registry.human_label(c.predicate.id),
                claim=c,
                source_count=initial_source_count,
            ))

    # Dedupe (subject, predicate, object, amount)
    # When equivalence grouping collapses synonyms, group by (subject, object,
    # amount) instead — the predicate surface form is no longer discriminating.
    by_key: dict[tuple, NumericFinding] = {}
    for f in candidates:
        if rep_of:
            key = (f.subject_id, f.object_id, f.amount)
        else:
            key = (f.subject_id, f.predicate, f.object_id, f.amount)
        if key not in by_key:
            by_key[key] = f
        else:
            by_key[key].source_count += f.source_count
    deduped = sorted(by_key.values(), key=lambda f: -f.amount)
    return deduped[:top_k]


# ────────────────────────────────────────────────────────────────────
# Primitive 4: contradicted_assertions
# ────────────────────────────────────────────────────────────────────

_CONTRADICTION_STOPWORDS = frozenset({
    "the", "and", "with", "from", "that", "this", "have", "been",
    "about", "into", "over", "your", "their", "them", "they",
    "never", "visit", "visited", "visiting", "island", "estate",
    "ranch", "residence", "home", "property", "place", "there",
})


def _overlap_tokens(text: str) -> set[str]:
    """Return candidate entity tokens — lowercase words ≥4 chars,
    minus common stopwords, with trailing 's' stripped for simple
    plural/possessive folding."""
    out: set[str] = set()
    for tok in re.split(r"[^a-z0-9]+", (text or "").lower()):
        if len(tok) >= 5 and tok.endswith("s"):
            tok = tok[:-1]
        if len(tok) >= 4 and tok not in _CONTRADICTION_STOPWORDS:
            out.add(tok)
    return out


def contradicted_assertions(
    db: "AttestDB",
    *,
    assertion_predicates: list[str],
    travel_or_contact_predicates: tuple[str, ...] = (
        "visited", "traveled_to", "flew_to", "flew_on", "was_on",
        "met_with", "stayed_at", "dined_with", "communicated_with",
        "spoke_with", "called", "emailed", "photographed_with",
        "attended_with", "socialized_with",
    ),
    location_keywords: tuple[str, ...] = (
        "island", "estate", "ranch", "residence", "home", "property",
    ),
    quality_filter: Optional[EntityQualityFilter] = None,
    registry: PredicateSalienceRegistry = DEFAULT_JOURNALISM_REGISTRY,
    top_k: int = 20,
    llm_judge: bool = False,
    llm_judge_provider: Optional[str] = None,
    llm_judge_model: Optional[str] = None,
    llm_judge_min_confidence: float = 0.5,
    # Deprecated alias preserved for back-compat.
    travel_or_visit_predicates: Optional[tuple[str, ...]] = None,
) -> list[AssertionContradictionFinding]:
    """For each claim with an assertion predicate (e.g. ``never_visited``,
    ``denied_visiting``), find any contradicting claim about the same
    subject + the same fact area.

    Heuristic match, in order of specificity:
    1. Counter-claim's object/evidence literally contains the denial object.
    2. Counter predicate is travel/contact AND counter object names a
       place-noun (the original narrow fallback).
    3. Counter predicate is travel/contact AND counter claim shares a
       significant token with the denial object or denial evidence
       (entity-overlap: "never_visited epstein_island" vs
       "flew_on epstein_planes" — both carry the token "epstein").
    """
    if travel_or_visit_predicates is not None:
        travel_or_contact_predicates = tuple(
            set(travel_or_contact_predicates) | set(travel_or_visit_predicates)
        )
    findings: list[AssertionContradictionFinding] = []
    assertion_set = set(assertion_predicates)
    for pid in assertion_predicates:
        try:
            claims = db.claims_for_predicate(pid) or []
        except Exception:
            continue
        for c in claims:
            try:
                sl = db.get_entity(c.subject.id)
            except Exception:
                sl = None
            if quality_filter and not quality_filter.is_substantive(sl):
                continue
            assertion_obj = (c.object.id or "").lower().strip()
            denial_evi = claim_evidence_text(c)
            denial_tokens = _overlap_tokens(assertion_obj) | _overlap_tokens(denial_evi)
            # Never match on the denier's own name — that's trivially present
            denial_tokens -= _overlap_tokens(c.subject.id)
            contradicting: list = []
            try:
                entity_claims = db.claims_for(c.subject.id, min_confidence=0.0) or []
            except Exception:
                entity_claims = []
            for cc in entity_claims:
                if cc.subject.id != c.subject.id:
                    continue
                ccpid = cc.predicate.id
                if ccpid in assertion_set:
                    continue
                obj_l = (cc.object.id or "").lower()
                evi_l = claim_evidence_text(cc).lower()
                if assertion_obj and (assertion_obj in obj_l or assertion_obj in evi_l):
                    contradicting.append(cc)
                elif ccpid in travel_or_contact_predicates and any(
                    k in obj_l for k in location_keywords
                ):
                    contradicting.append(cc)
                elif ccpid in travel_or_contact_predicates and denial_tokens and (
                    denial_tokens & (_overlap_tokens(obj_l) | _overlap_tokens(evi_l))
                ):
                    contradicting.append(cc)
            evi = claim_evidence_text(c).strip()
            src = getattr(c.provenance, "source_id", "") or ""
            findings.append(AssertionContradictionFinding(
                subject_id=c.subject.id,
                subject_label=(sl.name if sl else c.subject.id) or c.subject.id,
                assertion_predicate=c.predicate.id,
                assertion_label=registry.human_label(c.predicate.id),
                assertion_object=c.object.id,
                assertion_quote=evi[:280],
                assertion_source=src,
                contradicting_count=len(contradicting),
                contradicting_sample=contradicting[:3],
            ))
    if llm_judge and findings:
        # Build a flat list of (denial_claim, counter_claim) pairs, remember
        # which finding each pair belongs to, then filter counter-claims by
        # the judge's verdict.
        from attestdb.intelligence.contradiction_judge import judge_contradictions

        # Re-walk to get the denial Claim objects aligned with findings.
        # We already have contradicting_sample on each finding; but that was
        # truncated to 3 — we want to judge whatever we surface.
        pairs: list = []
        pair_owner: list[int] = []
        denial_claims_by_finding: list = []
        # Reconstruct denial claims in the same order as findings.
        # (findings was built by iterating claims_for_predicate in the same
        # order we're about to re-iterate here.)
        denial_iter: list = []
        for pid in assertion_predicates:
            try:
                for c in db.claims_for_predicate(pid) or []:
                    denial_iter.append(c)
            except Exception:
                continue
        # Map denial claim_id → the finding index it produced. Findings are
        # in the same order as denial_iter, minus any filtered by quality.
        # Instead of reconstructing that, we just attach one denial claim per
        # finding by matching subject+predicate+quote — cheap and robust.
        denial_by_finding: dict[int, object] = {}
        for idx, f in enumerate(findings):
            for c in denial_iter:
                if (
                    c.subject.id == f.subject_id
                    and c.predicate.id == f.assertion_predicate
                    and c.object.id == f.assertion_object
                ):
                    denial_by_finding[idx] = c
                    break

        for idx, f in enumerate(findings):
            denial_c = denial_by_finding.get(idx)
            if denial_c is None:
                continue
            for cc in f.contradicting_sample:
                pairs.append((denial_c, cc))
                pair_owner.append(idx)
            denial_claims_by_finding.append(denial_c)

        verdicts = judge_contradictions(
            pairs,
            provider=llm_judge_provider,
            model=llm_judge_model,
        )

        # Regroup surviving counter-claims per finding.
        kept_per_finding: dict[int, list] = defaultdict(list)
        for (pair, owner_idx, verdict) in zip(pairs, pair_owner, verdicts):
            if verdict.contradicts and verdict.confidence >= llm_judge_min_confidence:
                kept_per_finding[owner_idx].append(pair[1])

        filtered: list[AssertionContradictionFinding] = []
        for idx, f in enumerate(findings):
            kept = kept_per_finding.get(idx, [])
            if not kept:
                continue
            filtered.append(AssertionContradictionFinding(
                subject_id=f.subject_id,
                subject_label=f.subject_label,
                assertion_predicate=f.assertion_predicate,
                assertion_label=f.assertion_label,
                assertion_object=f.assertion_object,
                assertion_quote=f.assertion_quote,
                assertion_source=f.assertion_source,
                contradicting_count=len(kept),
                contradicting_sample=kept[:3],
            ))
        findings = filtered

    findings.sort(key=lambda f: (-f.contradicting_count, -len(f.assertion_quote)))
    return findings[:top_k]


# ────────────────────────────────────────────────────────────────────
# Primitive 5: non_obvious_associates
# ────────────────────────────────────────────────────────────────────

def non_obvious_associates(
    db: "AttestDB",
    pivot_entity_id: str,
    *,
    exclude: Optional[set[str]] = None,
    quality_filter: Optional[EntityQualityFilter] = None,
    min_docs: int = 3,
    min_specificity: float = 0.1,
    max_their_claims: int = 800,
    entity_type_filter: Optional[str] = "person",
    max_sources_scanned: int = 300,
    top_k: int = 25,
) -> list[AssociateFinding]:
    """Named entities that co-occur with ``pivot_entity_id`` in many
    source documents but aren't already in the obvious-set.

    - ``min_docs``: how many shared documents required
    - ``min_specificity``: shared_docs / their_total_claims floor
      (pure-cooccurrence with a corpus-wide entity gets diluted)
    - ``max_their_claims``: skip celebrities — entities with too many
      claims would dominate the list without telling you anything new
    """
    db_pivot = db.get_entity(pivot_entity_id)
    if db_pivot is None:
        return []
    obvious = exclude or set()

    pivot_sources: set[str] = set()
    pivot_bates: set[str] = set()
    for c in db.claims_for(pivot_entity_id, min_confidence=0.0) or []:
        sid = getattr(c.provenance, "source_id", "") or ""
        if sid:
            pivot_sources.add(sid)
            # Approximate "doc" by the prefix before any chunk suffix.
            doc = sid.split(":")[0].split("#")[0]
            if doc:
                pivot_bates.add(doc)

    co_docs: dict[str, set[str]] = defaultdict(set)
    for sid in list(pivot_sources)[:max_sources_scanned]:
        try:
            for c in db.claims_by_source_id(sid) or []:
                doc = sid.split(":")[0].split("#")[0]
                for side in (c.subject, c.object):
                    oid = getattr(side, "id", "") or ""
                    if not oid or oid == pivot_entity_id or oid in obvious:
                        continue
                    co_docs[oid].add(doc)
        except Exception:
            continue

    findings: list[AssociateFinding] = []
    for oid, docs in co_docs.items():
        if len(docs) < min_docs:
            continue
        try:
            es = db.get_entity(oid)
        except Exception:
            continue
        if es is None:
            continue
        if entity_type_filter and es.entity_type != entity_type_filter:
            continue
        if quality_filter and not quality_filter.is_substantive(es):
            continue
        if es.claim_count > max_their_claims:
            continue
        specificity = len(docs) / max(1, es.claim_count)
        if specificity < min_specificity:
            continue
        findings.append(AssociateFinding(
            entity_id=oid,
            label=(es.name or oid) or oid,
            shared_docs=len(docs),
            pivot_doc_total=len(pivot_bates),
            their_total_claims=es.claim_count,
            specificity=round(specificity, 3),
        ))
    findings.sort(key=lambda f: (-f.shared_docs * f.specificity, -f.shared_docs))
    return findings[:top_k]
