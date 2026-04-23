"""PredicateStore — self-improving, self-describing predicate system.

Single source of truth for all predicate metadata: identity, constraints,
composition rules, aliases, opposition pairs, and usage statistics.
Replaces the hardcoded constants in vocabulary.py with a living store
that observes, infers, and exports.

Usage:
    store = PredicateStore(db._store)
    store.seed_core()                    # one-time: load standard predicates
    info = store.get("has_metric")       # query
    store.observe("has_metric", "entity", "metric")  # learn
    catalog = store.export_catalog()     # for LLM prompts
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PredicateInfo:
    """Everything the system knows about a predicate."""

    id: str
    description: str = ""
    subject_types: set[str] = field(default_factory=set)
    object_types: set[str] = field(default_factory=set)
    directional: bool = False
    symmetric: bool = False
    opposite: str | None = None
    composition: dict[str, str] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)
    data_patterns: list[str] = field(default_factory=list)
    equivalence_class: str = ""  # "positive", "negative", or ""
    origin: str = "seed"  # "seed" | "domain" | "learned"
    observation_count: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Observation tracking: {(subj_type, obj_type): count}
    _type_observations: dict[tuple[str, str], int] = field(
        default_factory=dict, repr=False,
    )

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for persistence."""
        return {
            "id": self.id,
            "description": self.description,
            "subject_types": sorted(self.subject_types),
            "object_types": sorted(self.object_types),
            "directional": self.directional,
            "symmetric": self.symmetric,
            "opposite": self.opposite,
            "composition": self.composition,
            "aliases": sorted(self.aliases),
            "data_patterns": self.data_patterns,
            "equivalence_class": self.equivalence_class,
            "origin": self.origin,
            "observation_count": self.observation_count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "type_observations": {
                f"{k[0]}:{k[1]}": v for k, v in self._type_observations.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> PredicateInfo:
        """Deserialize from dict."""
        obs = {}
        for k, v in d.get("type_observations", {}).items():
            parts = k.split(":", 1)
            if len(parts) == 2:
                obs[(parts[0], parts[1])] = v
        return cls(
            id=d["id"],
            description=d.get("description", ""),
            subject_types=set(d.get("subject_types", [])),
            object_types=set(d.get("object_types", [])),
            directional=d.get("directional", False),
            symmetric=d.get("symmetric", False),
            opposite=d.get("opposite"),
            composition=d.get("composition", {}),
            aliases=set(d.get("aliases", [])),
            data_patterns=d.get("data_patterns", []),
            equivalence_class=d.get("equivalence_class", ""),
            origin=d.get("origin", "seed"),
            observation_count=d.get("observation_count", 0),
            first_seen=d.get("first_seen", 0.0),
            last_seen=d.get("last_seen", 0.0),
            _type_observations=obs,
        )


class PredicateStore:
    """Self-improving predicate knowledge store.

    Manages all predicate metadata in a single place. Persists to the
    Rust MetadataStore via register_predicate(). Tracks usage observations
    and can infer constraints from empirical evidence.
    """

    def __init__(self, rust_store=None):
        self._store = rust_store
        self._predicates: dict[str, PredicateInfo] = {}
        self._alias_index: dict[str, str] = {}  # alias → canonical pred_id
        self._seeded = False
        if rust_store is not None:
            self._load_from_store()

    # ── Core API ─────────────────────────────────────────────────────

    def get(self, pred_id: str) -> PredicateInfo | None:
        """Get predicate info by ID."""
        return self._predicates.get(pred_id)

    def exists(self, pred_id: str) -> bool:
        """Check if a predicate is registered."""
        return pred_id in self._predicates or pred_id in self._alias_index

    def all(self) -> list[PredicateInfo]:
        """Return all registered predicates."""
        return list(self._predicates.values())

    def register(self, pred_id: str, info: PredicateInfo) -> None:
        """Register or update a predicate."""
        info.id = pred_id
        self._predicates[pred_id] = info
        for alias in info.aliases:
            self._alias_index[alias] = pred_id
        self._persist(pred_id)

    # ── Observation ──────────────────────────────────────────────────

    def observe(self, pred_id: str, subj_type: str, obj_type: str) -> None:
        """Record a successful claim using this predicate.

        Lightweight — increments counters only. Called on every ingestion.
        """
        info = self._predicates.get(pred_id)
        if info is None:
            # Auto-register unknown predicates as learned
            info = PredicateInfo(
                id=pred_id,
                origin="learned",
                first_seen=time.time(),
            )
            self._predicates[pred_id] = info

        now = time.time()
        info.observation_count += 1
        info.last_seen = now
        if info.first_seen == 0.0:
            info.first_seen = now

        key = (subj_type, obj_type)
        info._type_observations[key] = info._type_observations.get(key, 0) + 1

    # ── Normalization ────────────────────────────────────────────────

    def normalize(self, raw_predicate: str) -> str:
        """Normalize a raw predicate string to its canonical form.

        Resolves aliases, strips prefixes/suffixes, handles verb tense.
        Returns the raw predicate if no match is found (does NOT
        default to associated_with — that's a policy decision for
        the caller).
        """
        cleaned = raw_predicate.lower().strip().rstrip(".").replace(" ", "_")

        # Direct match
        if cleaned in self._predicates:
            return cleaned

        # Alias lookup
        canonical = self._alias_index.get(cleaned)
        if canonical:
            return canonical

        # Strip prefixes and retry
        stripped = cleaned
        for prefix in ("is_", "can_", "may_", "potentially_", "directly_"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
        for suffix in ("_in", "_of", "_by", "_to", "_for", "_with"):
            if stripped.endswith(suffix):
                stripped = stripped[:-len(suffix)]

        if stripped in self._predicates:
            return stripped
        canonical = self._alias_index.get(stripped)
        if canonical:
            return canonical

        return cleaned

    # ── Validation ───────────────────────────────────────────────────

    def validate(
        self, pred_id: str, subj_type: str, obj_type: str,
    ) -> tuple[bool, str]:
        """Check if a predicate usage is valid against constraints.

        Returns (valid, reason). If the predicate has no constraints,
        it's always valid.
        """
        info = self._predicates.get(pred_id)
        if info is None:
            return (True, "unknown predicate, no constraints")

        if info.subject_types and subj_type not in info.subject_types:
            return (
                False,
                f"subject_type '{subj_type}' not allowed for '{pred_id}' "
                f"(allowed: {sorted(info.subject_types)})",
            )

        if info.object_types and obj_type not in info.object_types:
            return (
                False,
                f"object_type '{obj_type}' not allowed for '{pred_id}' "
                f"(allowed: {sorted(info.object_types)})",
            )

        return (True, "")

    # ── Composition ──────────────────────────────────────────────────

    def compose(self, pred_ac: str, pred_cb: str) -> str:
        """Compose two predicates along A→C→B into predicted A→B.

        Checks the first predicate's composition rules, then the second's.
        Falls back to same-predicate transitivity, then associated_with.
        """
        info_ac = self._predicates.get(pred_ac)
        if info_ac and pred_cb in info_ac.composition:
            return info_ac.composition[pred_cb]

        # Check if either is non-directional (weak)
        if info_ac and not info_ac.directional:
            return "associated_with"
        info_cb = self._predicates.get(pred_cb)
        if info_cb and not info_cb.directional:
            return "associated_with"

        # Same-predicate transitivity
        if pred_ac == pred_cb:
            return pred_ac

        return "associated_with"

    def get_opposite(self, pred_id: str) -> str | None:
        """Get the opposing predicate, if any."""
        info = self._predicates.get(pred_id)
        return info.opposite if info else None

    def predicates_agree(self, pred_a: str, pred_b: str) -> bool:
        """Check if two predicates have the same directional class."""
        info_a = self._predicates.get(pred_a)
        info_b = self._predicates.get(pred_b)
        if info_a and info_b and info_a.equivalence_class and info_b.equivalence_class:
            return info_a.equivalence_class == info_b.equivalence_class
        return pred_a == pred_b

    def is_causal(self, pred_id: str) -> bool:
        """Check if a predicate is causal/directional and composable."""
        info = self._predicates.get(pred_id)
        return info.directional if info else False

    # ── Inference ────────────────────────────────────────────────────

    def infer_constraints(self, min_observations: int = 20) -> list[dict]:
        """Analyze observations and infer constraints for learned predicates.

        Only updates predicates with origin="learned" and enough observations.
        Returns a list of inference results for review.
        """
        results = []
        for pred_id, info in self._predicates.items():
            if info.observation_count < min_observations:
                continue
            if info.origin == "seed" and info.subject_types:
                continue  # Don't override seed constraints

            total = sum(info._type_observations.values())
            if total < min_observations:
                continue

            # Compute subject_type and object_type distributions
            subj_counts: dict[str, int] = {}
            obj_counts: dict[str, int] = {}
            for (st, ot), count in info._type_observations.items():
                subj_counts[st] = subj_counts.get(st, 0) + count
                obj_counts[ot] = obj_counts.get(ot, 0) + count

            # Crystallize if 95%+ of observations use the same type set
            inferred_subj = {
                st for st, c in subj_counts.items() if c / total >= 0.05
            }
            inferred_obj = {
                ot for ot, c in obj_counts.items() if c / total >= 0.05
            }

            changed = False
            if not info.subject_types and inferred_subj:
                info.subject_types = inferred_subj
                changed = True
            if not info.object_types and inferred_obj:
                info.object_types = inferred_obj
                changed = True

            if changed:
                self._persist(pred_id)
                results.append({
                    "predicate": pred_id,
                    "inferred_subject_types": sorted(inferred_subj),
                    "inferred_object_types": sorted(inferred_obj),
                    "observation_count": total,
                })

        return results

    # ── Export ────────────────────────────────────────────────────────

    def export_catalog(self) -> dict:
        """Export the full predicate catalog for LLM consumption.

        Returns a dict with predicate descriptions, constraints,
        and usage guidance — suitable for inclusion in LLM prompts.
        """
        catalog = {}
        for pred_id, info in sorted(self._predicates.items()):
            if info.description:
                entry: dict = {"description": info.description}
                if info.data_patterns:
                    entry["data_patterns"] = info.data_patterns
                if info.subject_types:
                    entry["subject_types"] = sorted(info.subject_types)
                if info.object_types:
                    entry["object_types"] = sorted(info.object_types)
                if info.observation_count > 0:
                    entry["usage_count"] = info.observation_count
                catalog[pred_id] = entry
        return catalog

    # ── Persistence ──────────────────────────────────────────────────

    def _persist(self, pred_id: str) -> None:
        """Write a single predicate to the Rust MetadataStore."""
        if self._store is None:
            return
        info = self._predicates.get(pred_id)
        if info is None:
            return
        try:
            self._store.register_predicate(pred_id, info.to_dict())
        except Exception as exc:
            logger.debug("Failed to persist predicate %s: %s", pred_id, exc)

    def _load_from_store(self) -> None:
        """Load predicates persisted in PredicateStore format from MetadataStore.

        Only loads entries written by PredicateStore (have "id" key).
        Ignores legacy constraint dicts from domain vocabularies — those
        are handled by the old Rule 9 path or re-registered through the
        PredicateStore API by domain vocabulary modules.
        """
        if self._store is None:
            return
        try:
            constraints = self._store.get_predicate_constraints()
            for pred_id, data in constraints.items():
                if isinstance(data, str):
                    data = json.loads(data)
                # Only load PredicateStore-format entries (have "id" key)
                if isinstance(data, dict) and "id" in data:
                    self._predicates[pred_id] = PredicateInfo.from_dict(data)
                # Skip legacy {subject_types, object_types} dicts
            # Rebuild alias index
            for pred_id, info in self._predicates.items():
                for alias in info.aliases:
                    self._alias_index[alias] = pred_id
        except Exception as exc:
            logger.debug("Failed to load predicates from store: %s", exc)

    def persist_all(self) -> None:
        """Write all predicates to the store. Called on DB close."""
        for pred_id in self._predicates:
            self._persist(pred_id)

    # ── Seeding ──────────────────────────────────────────────────────

    def seed_core(self) -> int:
        """Load core predicates from seed data.

        Called once on database creation. Returns number of predicates seeded.
        Idempotent — skips predicates that already exist.
        """
        if self._seeded:
            return 0

        count = 0
        for pred_id, info in _CORE_PREDICATES.items():
            if pred_id not in self._predicates:
                self._predicates[pred_id] = copy.deepcopy(info)
                for alias in info.aliases:
                    self._alias_index[alias] = pred_id
                # Don't persist during seed — in-memory only until close()
                count += 1

        self._seeded = True
        logger.info("Seeded %d core predicates", count)
        return count


# ── Core predicate seed data ─────────────────────────────────────────
# This replaces STANDARD_PREDICATES, OPPOSITE_PREDICATES,
# PREDICATE_COMPOSITION, PREDICATE_ALIAS_MAP, etc.

def _p(
    pred_id: str,
    description: str,
    *,
    directional: bool = False,
    symmetric: bool = False,
    opposite: str | None = None,
    composition: dict[str, str] | None = None,
    aliases: set[str] | None = None,
    data_patterns: list[str] | None = None,
    equivalence_class: str = "",
    subject_types: set[str] | None = None,
    object_types: set[str] | None = None,
) -> PredicateInfo:
    """Helper to build a PredicateInfo for seed data."""
    return PredicateInfo(
        id=pred_id,
        description=description,
        subject_types=subject_types or set(),
        object_types=object_types or set(),
        directional=directional,
        symmetric=symmetric,
        opposite=opposite,
        composition=composition or {},
        aliases=aliases or set(),
        data_patterns=data_patterns or [],
        equivalence_class=equivalence_class,
        origin="seed",
    )


_CORE_PREDICATES: dict[str, PredicateInfo] = {
    # ── Enterprise / business predicates ──────────────────────────
    "has_metric": _p(
        "has_metric",
        "Numeric measurement, score, amount, percentage, or count",
        data_patterns=["revenue", "amount", "score", "count", "total",
                       "rate", "pct", "percent", "aum", "cvss", "nps",
                       "return", "price", "cost", "fee", "value"],
    ),
    "has_status": _p(
        "has_status",
        "State, status, phase, or condition value",
        data_patterns=["status", "state", "phase", "stage", "rating",
                       "level", "tier", "grade", "risk_rating",
                       "remediation_status", "certification_status"],
    ),
    "assigned_to": _p(
        "assigned_to",
        "Person responsible, owner, or assignee",
        directional=True,
        data_patterns=["assigned_to", "assignee", "owner", "manager",
                       "relationship_manager", "auditor", "rep",
                       "account_manager", "lead"],
    ),
    "classified_as": _p(
        "classified_as",
        "Category, type, classification, or grouping",
        data_patterns=["type", "category", "classification", "industry",
                       "segment", "department", "severity", "kind",
                       "portfolio_type", "contract_type", "sector"],
    ),
    "located_in": _p(
        "located_in",
        "Geographic location, region, or jurisdiction",
        directional=True,
        data_patterns=["country", "region", "city", "location", "geo",
                       "jurisdiction", "governing_law", "state"],
    ),
    "managed_by": _p(
        "managed_by",
        "Management or supervisory relationship",
        directional=True,
        opposite="manages",
        data_patterns=["manager", "supervisor", "team_lead", "director"],
        object_types={"person", "entity"},
    ),
    "manages": _p(
        "manages",
        "Manages or supervises",
        directional=True,
        opposite="managed_by",
    ),
    "described_by": _p(
        "described_by",
        "Free text description, notes, or comments",
        data_patterns=["description", "notes", "comment", "text",
                       "summary", "details", "finding_text", "remarks"],
    ),
    "part_of": _p(
        "part_of",
        "Hierarchical membership — belongs to a group, division, or parent",
        directional=True,
        data_patterns=["department", "division", "section", "parent",
                       "group", "organization", "team"],
    ),
    "reports_to": _p(
        "reports_to",
        "Reporting chain relationship",
        directional=True,
        data_patterns=["reports_to", "direct_report", "supervisor"],
    ),
    "owns": _p(
        "owns",
        "Ownership relationship",
        directional=True,
        data_patterns=["owner", "account_owner", "asset_owner"],
    ),
    "relates_to": _p(
        "relates_to",
        "Generic relationship — use only when no specific predicate fits",
        aliases={"linked_to", "connected_to", "implicated_in",
                 "is_linked_to", "concerned_with", "complicated_by",
                 "resistant_to", "sensitive_to"},
    ),
    "associated_with": _p(
        "associated_with",
        "Weak, non-directional association — prefer a specific predicate",
        aliases={"is_associated_with", "associates"},
    ),

    # ── Causal / regulatory predicates ───────────────────────────
    "activates": _p(
        "activates", "Positively activates or turns on",
        directional=True, equivalence_class="positive",
        opposite="inhibits",
        composition={
            "activates": "activates", "inhibits": "inhibits",
            "promotes": "promotes", "downregulates": "downregulates",
            "upregulates": "upregulates", "regulates": "regulates",
        },
        aliases={"activate"},
    ),
    "inhibits": _p(
        "inhibits", "Negatively inhibits or blocks activity",
        directional=True, equivalence_class="negative",
        opposite="activates",
        composition={
            "activates": "inhibits", "inhibits": "activates",
            "promotes": "inhibits", "downregulates": "upregulates",
            "upregulates": "downregulates", "regulates": "regulates",
        },
        aliases={"inhibit"},
    ),
    "upregulates": _p(
        "upregulates", "Increases expression or activity level",
        directional=True, equivalence_class="positive",
        opposite="downregulates",
        composition={
            "activates": "upregulates", "inhibits": "downregulates",
            "upregulates": "upregulates", "downregulates": "downregulates",
            "regulates": "regulates",
        },
        aliases={"upregulate"},
    ),
    "downregulates": _p(
        "downregulates", "Decreases expression or activity level",
        directional=True, equivalence_class="negative",
        opposite="upregulates",
        composition={
            "activates": "downregulates", "inhibits": "upregulates",
            "downregulates": "upregulates", "upregulates": "downregulates",
            "regulates": "regulates",
        },
        aliases={"downregulate"},
    ),
    "promotes": _p(
        "promotes", "Promotes or facilitates",
        directional=True, equivalence_class="positive",
        opposite="suppresses",
        composition={
            "activates": "promotes", "inhibits": "suppresses",
            "promotes": "promotes", "suppresses": "suppresses",
            "upregulates": "upregulates", "downregulates": "downregulates",
        },
        aliases={"promote", "facilitates", "facilitate_the_spread_of"},
    ),
    "suppresses": _p(
        "suppresses", "Suppresses or inhibits progression",
        directional=True, equivalence_class="negative",
        opposite="promotes",
        composition={
            "activates": "suppresses", "inhibits": "promotes",
            "promotes": "suppresses", "suppresses": "promotes",
            "upregulates": "downregulates", "downregulates": "upregulates",
        },
        aliases={"suppress"},
    ),
    "increases": _p(
        "increases", "Increases quantity or level",
        directional=True, equivalence_class="positive",
        opposite="decreases",
        composition={
            "activates": "activates", "inhibits": "inhibits",
            "upregulates": "upregulates", "downregulates": "downregulates",
        },
        aliases={"increase"},
    ),
    "decreases": _p(
        "decreases", "Decreases quantity or level",
        directional=True, equivalence_class="negative",
        opposite="increases",
        composition={
            "activates": "inhibits", "inhibits": "activates",
            "upregulates": "downregulates", "downregulates": "upregulates",
        },
        aliases={"decrease"},
    ),
    "causes": _p(
        "causes", "Causes or leads to",
        directional=True, equivalence_class="positive",
        opposite="prevents",
        composition={
            "activates": "activates", "inhibits": "inhibits",
            "causes": "causes", "prevents": "prevents",
        },
        aliases={"cause"},
    ),
    "prevents": _p(
        "prevents", "Prevents or blocks from occurring",
        directional=True, equivalence_class="negative",
        opposite="causes",
        composition={
            "activates": "inhibits", "inhibits": "activates",
            "causes": "prevents", "prevents": "causes",
        },
        aliases={"prevent"},
    ),
    "enables": _p(
        "enables", "Enables or makes possible",
        directional=True, equivalence_class="positive",
        opposite="blocks",
        composition={
            "activates": "activates", "inhibits": "inhibits",
            "upregulates": "upregulates", "downregulates": "downregulates",
        },
        aliases={"enable"},
    ),
    "blocks": _p(
        "blocks", "Blocks or prevents",
        directional=True, equivalence_class="negative",
        opposite="enables",
        composition={
            "activates": "inhibits", "inhibits": "activates",
            "upregulates": "downregulates", "downregulates": "upregulates",
        },
        aliases={"block"},
    ),
    "enhances": _p(
        "enhances", "Enhances or amplifies effect",
        directional=True, equivalence_class="positive",
        opposite="reduces",
        aliases={"enhance"},
    ),
    "reduces": _p(
        "reduces", "Reduces or diminishes",
        directional=True, equivalence_class="negative",
        opposite="enhances",
        aliases={"reduce"},
    ),
    "stabilizes": _p(
        "stabilizes", "Stabilizes structure or activity",
        directional=True, equivalence_class="positive",
        opposite="destabilizes",
        composition={
            "activates": "activates", "inhibits": "inhibits",
        },
        aliases={"stabilize"},
    ),
    "destabilizes": _p(
        "destabilizes", "Destabilizes structure or activity",
        directional=True, equivalence_class="negative",
        opposite="stabilizes",
        composition={
            "activates": "inhibits", "inhibits": "activates",
        },
        aliases={"destabilize"},
    ),
    "regulates": _p(
        "regulates", "Regulates (direction unspecified)",
        directional=True,
        composition={
            "activates": "regulates", "inhibits": "regulates",
            "upregulates": "regulates", "downregulates": "regulates",
            "regulates": "regulates",
        },
        aliases={"regulate", "affects", "influences"},
    ),

    # ── Interaction predicates ───────────────────────────────────
    "interacts_with": _p(
        "interacts_with", "Physical or functional interaction",
        symmetric=True,
        aliases={"interacts", "interact"},
    ),
    "binds": _p(
        "binds", "Physical binding interaction",
        symmetric=True,
        aliases={"bind"},
    ),
    "targets": _p(
        "targets", "Targets or acts upon",
        directional=True,
        aliases={"target"},
    ),

    # ── Association predicates ───────────────────────────────────
    "correlates_with": _p(
        "correlates_with", "Statistical correlation",
        symmetric=True,
        aliases={"is_correlated_with"},
    ),
    "contributes_to": _p(
        "contributes_to", "Contributes to a process or outcome",
        directional=True,
    ),

    # ── Spatial / structural predicates ──────────────────────────
    "expressed_in": _p(
        "expressed_in", "Expressed in a tissue or cell type",
        directional=True,
    ),
    "participates_in": _p(
        "participates_in", "Participates in a pathway or process",
        directional=True,
        aliases={"is_involved_in", "involved_in"},
    ),

    # ── Therapeutic / clinical predicates ─────────────────────────
    "treats": _p(
        "treats", "Used to treat a condition",
        directional=True,
        aliases={"treat", "is_used_for"},
    ),
    "biomarker_for": _p(
        "biomarker_for", "Serves as a biomarker for a condition",
        directional=True,
        aliases={"is_a_biomarker_for", "serves_as"},
    ),

    # ── Post-translational predicates ────────────────────────────
    "phosphorylates": _p(
        "phosphorylates", "Adds phosphate group",
        directional=True, opposite="dephosphorylates",
        aliases={"phosphorylate"},
    ),
    "dephosphorylates": _p(
        "dephosphorylates", "Removes phosphate group",
        directional=True, opposite="phosphorylates",
        aliases={"dephosphorylate"},
    ),
    "methylates": _p(
        "methylates", "Adds methyl group",
        directional=True, opposite="demethylates",
        aliases={"methylate"},
    ),
    "demethylates": _p(
        "demethylates", "Removes methyl group",
        directional=True, opposite="methylates",
        aliases={"demethylate"},
    ),

    # ── Semantic predicates ──────────────────────────────────────
    "induces": _p("induces", "Induces a state or process", directional=True, aliases={"induce"}),
    "modulates": _p("modulates", "Modulates activity", directional=True, aliases={"modulate", "is_major_mechanism_for"}),
    "mediates": _p("mediates", "Mediates a process", directional=True, aliases={"mediate"}),
    "encodes": _p("encodes", "Encodes a product", directional=True, aliases={"encode"}),

    # ── Structural predicates (used by engine) ───────────────────
    "caused": _p("caused", "Internal: causal event link"),
    "observed": _p("observed", "Internal: observation event"),
    "derived_from": _p("derived_from", "Derived from another claim", directional=True),
    "contradicts": _p("contradicts", "Contradicts another claim", symmetric=True),
    "same_as": _p("same_as", "Entity equivalence", symmetric=True),
    "not_same_as": _p("not_same_as", "Entity non-equivalence", symmetric=True),
    "retracted": _p("retracted", "Claim has been retracted"),
    "contradiction_resolved": _p("contradiction_resolved", "Contradiction was resolved"),
    "superseded_by": _p("superseded_by", "Superseded by a newer version", directional=True, aliases={"supersedes"}),
    "inquiry": _p("inquiry", "Internal: inquiry tracking"),
    "no_evidence_for": _p("no_evidence_for", "No evidence found for this assertion"),

    # ── Thread predicates ────────────────────────────────────────
    "thread_seeded_by": _p("thread_seeded_by", "Thread was seeded by this claim"),
    "thread_branched_from": _p("thread_branched_from", "Thread branched from another"),
    "thread_extended_by": _p("thread_extended_by", "Thread was extended by this claim"),

    # ── Research predicates ──────────────────────────────────────
    "has_finding": _p("has_finding", "Research finding"),
    "has_gap": _p("has_gap", "Knowledge gap identified"),
    "investigating": _p("investigating", "Currently under investigation"),
    "has_strategy": _p("has_strategy", "Has an investigation strategy"),

    # ── Knowledge predicates ─────────────────────────────────────
    "has_warning": _p("has_warning", "Known warning or issue"),
    "has_vulnerability": _p("has_vulnerability", "Known vulnerability"),
    "has_pattern": _p("has_pattern", "Recognized pattern"),
    "has_decision": _p("has_decision", "Recorded decision"),
    "has_tip": _p("has_tip", "Helpful tip or shortcut"),
    "had_bug": _p("had_bug", "Had a bug (historical)"),
    "has_issue": _p("has_issue", "Has an open issue"),
    "failed_on": _p("failed_on", "Failed on a specific input or scenario"),
    "has_fix": _p("has_fix", "Has a known fix"),
    "resolved": _p("resolved", "Issue was resolved"),
    "had_outcome": _p("had_outcome", "Produced an outcome"),
    "produced_by": _p("produced_by", "Was produced by a process", directional=True),
    "has_next_steps": _p("has_next_steps", "Has planned next steps"),
    "has_goal": _p("has_goal", "Has a defined goal"),
    "has_architecture": _p("has_architecture", "Has an architectural design"),
    "has_trade_off": _p("has_trade_off", "Has a known trade-off"),
    "has_convention": _p("has_convention", "Follows a convention"),
    "depends_on": _p("depends_on", "Depends on another component", directional=True),
    "has_requirement": _p("has_requirement", "Has a requirement"),

    # ── Compliance predicates ────────────────────────────────────
    "requires": _p(
        "requires", "Regulatory requirement link",
        directional=True,
    ),
    "satisfied_by": _p(
        "satisfied_by", "Requirement satisfied by evidence",
        directional=True,
    ),
    "verified_by": _p(
        "verified_by", "Verified by a control or evidence",
        directional=True,
    ),
    "implements": _p(
        "implements", "Implements or fulfills a requirement or standard",
        directional=True,
    ),
    "evidenced_by": _p(
        "evidenced_by", "Evidenced by an artifact",
        directional=True,
        subject_types={"control"},
        object_types={"evidence_artifact"},
    ),

    # ── Connector predicates (Slack, GitHub, Jira, etc.) ─────────
    "posted_in": _p(
        "posted_in", "Posted a message in a channel or thread",
        directional=True,
        data_patterns=["channel", "posted_in", "thread"],
    ),
    "mentioned": _p(
        "mentioned", "Mentioned another person in a message",
        directional=True,
        data_patterns=["mentioned", "mention", "tagged"],
    ),
    "authored_by": _p(
        "authored_by", "Authored or created by a person",
        directional=True,
        data_patterns=["author", "creator", "created_by", "opened_by"],
    ),
    "labeled": _p(
        "labeled", "Tagged with a label or category",
        directional=True,
        data_patterns=["label", "tag", "tagged"],
    ),
    "has_state": _p(
        "has_state", "Current state of an issue, ticket, or item",
        data_patterns=["state", "status", "open", "closed", "resolved"],
    ),
}


# ── Module-level default store ───────────────────────────────────────
# Used by code that needs predicate metadata without a db instance.
# Seeded with core predicates on first access.

_default_store: PredicateStore | None = None


def get_default_store() -> PredicateStore:
    """Get the module-level default PredicateStore, seeded with core predicates.

    This store has no Rust persistence — it's in-memory only. Used by
    functions that need predicate metadata (composition, opposition,
    normalization) without access to a db instance.
    """
    global _default_store
    if _default_store is None:
        _default_store = PredicateStore()
        _default_store.seed_core()
    return _default_store
