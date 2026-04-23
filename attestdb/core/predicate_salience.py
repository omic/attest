"""Predicate salience: register categories, weights, and human labels for
predicates so domain-specific analyses (entity dossier, gap analysis,
narrative leaderboards) can rank and group claims by what matters in the
domain.

The ``PredicateSalienceRegistry`` is the configurable mechanism. The
``DEFAULT_JOURNALISM_REGISTRY`` instance below is populated with the
predicates the Epstein explorer uses and is returned by the module-level
helpers (``predicate_meta``, ``human_label``, ``claim_salience``, etc.)
so existing callers don't need to change.

Other domains build their own registry — a sales corpus might register
``closed_won`` as ``("revenue", 1.0, "closed")`` and ``cc_emailed`` as
``("metadata", 0.0, "was cc'd on")`` — then pass the registry into the
analyses in ``attestdb.infrastructure.analyses``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# ────────────────────────────────────────────────────────────────────
# Default category constants the journalism registry uses.
# Custom registries can register their own category strings — these are
# just convenience aliases so callers can reference them by name.
# ────────────────────────────────────────────────────────────────────

CATEGORY_ALLEGATION = "allegation"
CATEGORY_DENIAL = "denial"
CATEGORY_MONEY = "money"
CATEGORY_LEGAL = "legal"
CATEGORY_RELATIONSHIP = "relationship"
CATEGORY_TRAVEL = "travel"
CATEGORY_METADATA = "metadata"

# Display order + label/icon for the journalism registry (legacy export).
CATEGORY_ORDER = [
    CATEGORY_ALLEGATION, CATEGORY_MONEY, CATEGORY_DENIAL, CATEGORY_LEGAL,
    CATEGORY_RELATIONSHIP, CATEGORY_TRAVEL, CATEGORY_METADATA,
]
CATEGORY_LABELS = {
    CATEGORY_ALLEGATION:  "Allegations",
    CATEGORY_DENIAL:      "Public denials",
    CATEGORY_MONEY:       "Money",
    CATEGORY_LEGAL:       "Legal process",
    CATEGORY_RELATIONSHIP: "Relationships",
    CATEGORY_TRAVEL:      "Travel & locations",
    CATEGORY_METADATA:    "Court paperwork & metadata",
}
CATEGORY_ICONS = {
    CATEGORY_ALLEGATION:  "⚠",
    CATEGORY_DENIAL:      "🚫",
    CATEGORY_MONEY:       "💰",
    CATEGORY_LEGAL:       "⚖",
    CATEGORY_RELATIONSHIP: "🔗",
    CATEGORY_TRAVEL:      "✈",
    CATEGORY_METADATA:    "📄",
}


# ────────────────────────────────────────────────────────────────────
# Registry — the configurable mechanism
# ────────────────────────────────────────────────────────────────────

@dataclass
class CategorySpec:
    key: str
    label: str = ""
    icon: str = ""
    order: int = 100  # lower = displayed first

    def __post_init__(self):
        if not self.label:
            self.label = self.key.replace("_", " ").title()


@dataclass
class PredicateSpec:
    predicate_id: str
    category: str
    weight: float
    label: str
    synonym_group: Optional[str] = None


class PredicateSalienceRegistry:
    """A registry mapping predicate ids to (category, weight, label).

    Build one per domain. Pass into analyses that need to rank or group
    claims by category/weight (entity dossier, gap analysis, etc.).

    Methods are intentionally small so this stays a configuration object,
    not a query layer.
    """

    def __init__(self):
        self._predicates: dict[str, PredicateSpec] = {}
        self._categories: dict[str, CategorySpec] = {}
        self._fallbacks: list[Callable[[str], Optional[Tuple[str, float, str]]]] = []
        self._default_meta: Tuple[str, float, str] = ("relationship", 0.3, "")

    # ── Registration ─────────────────────────────────────────────────

    def register(self, predicate_id: str, category: str, weight: float, label: str) -> None:
        """Add or overwrite a predicate's metadata."""
        self._predicates[predicate_id] = PredicateSpec(predicate_id, category, weight, label)

    def set_synonym_group(self, predicate_id: str, group: str) -> None:
        """Mark ``predicate_id`` as a member of the named synonym group.
        No-op if the predicate isn't registered — strictly additive.
        """
        spec = self._predicates.get(predicate_id)
        if spec is not None:
            spec.synonym_group = group

    def synonym_group_of(self, predicate_id: str) -> Optional[str]:
        spec = self._predicates.get(predicate_id)
        return spec.synonym_group if spec else None

    def synonym_groups(self) -> dict[str, set[str]]:
        """Return {group_name: {predicate_id, ...}} across all registered predicates."""
        out: dict[str, set[str]] = {}
        for spec in self._predicates.values():
            if spec.synonym_group:
                out.setdefault(spec.synonym_group, set()).add(spec.predicate_id)
        return out

    def register_many(self, mapping: dict[str, Tuple[str, float, str]]) -> None:
        """Bulk-register from {predicate_id: (category, weight, label)}."""
        for pid, (cat, w, lbl) in mapping.items():
            self.register(pid, cat, w, lbl)

    def register_category(self, key: str, label: str = "", icon: str = "", order: int = 100) -> None:
        self._categories[key] = CategorySpec(key=key, label=label, icon=icon, order=order)

    def register_fallback(self, fn: Callable[[str], Optional[Tuple[str, float, str]]]) -> None:
        """Attach a function applied to predicates not in the explicit
        registry. First fallback to return non-None wins. The default
        meta is used if all fallbacks return None.
        """
        self._fallbacks.append(fn)

    def set_default(self, category: str, weight: float, label_template: str = "") -> None:
        """Meta returned for predicates with no explicit registration and
        no matching fallback. ``label_template`` of ``""`` falls back to
        the predicate id with underscores → spaces.
        """
        self._default_meta = (category, weight, label_template)

    # ── Query ────────────────────────────────────────────────────────

    def meta(self, predicate_id: str) -> Tuple[str, float, str]:
        """Return (category, weight, human_label) for any predicate id.
        Looks up explicit registrations first, then runs each fallback,
        then returns the default.
        """
        if predicate_id in self._predicates:
            p = self._predicates[predicate_id]
            return (p.category, p.weight, p.label)
        for fn in self._fallbacks:
            result = fn(predicate_id)
            if result is not None:
                return result
        cat, w, lbl_tmpl = self._default_meta
        label = lbl_tmpl or _humanize(predicate_id)
        return (cat, w, label)

    def category_of(self, predicate_id: str) -> str:
        return self.meta(predicate_id)[0]

    def weight_of(self, predicate_id: str) -> float:
        return self.meta(predicate_id)[1]

    def human_label(self, predicate_id: str) -> str:
        return self.meta(predicate_id)[2]

    def is_metadata(self, predicate_id: str) -> bool:
        return self.weight_of(predicate_id) == 0.0

    def categories(self) -> list[CategorySpec]:
        """All registered categories sorted by their order field."""
        return sorted(self._categories.values(), key=lambda c: c.order)

    def category_spec(self, key: str) -> CategorySpec:
        if key in self._categories:
            return self._categories[key]
        # Auto-create a placeholder so consumers can ask about a category
        # they registered predicates for without explicitly naming it.
        return CategorySpec(key=key, label=key.replace("_", " ").title())

    # ── Salience ─────────────────────────────────────────────────────

    def claim_salience(
        self,
        predicate_id: str,
        corroboration_count: int = 0,
        has_evidence_text: bool = False,
        source_diversity: int = 1,
    ) -> float:
        """Salience score combining predicate weight, corroboration,
        evidence presence, and source diversity. Returns 0.0 when the
        predicate's weight is 0 (so metadata sorts to the bottom and can
        be hidden behind a single toggle).
        """
        w = self.weight_of(predicate_id)
        if w == 0.0:
            return 0.0
        corrob_factor = 1.0 + math.log1p(max(0, corroboration_count - 1))
        evidence_factor = 1.0 if has_evidence_text else 0.5
        diversity_factor = 1.0 + 0.3 * math.log1p(max(0, source_diversity - 1))
        return w * corrob_factor * evidence_factor * diversity_factor

    # ── Inspection ──────────────────────────────────────────────────

    def predicates_in(self, category: str) -> list[str]:
        """All registered predicate ids that belong to ``category``."""
        return [p.predicate_id for p in self._predicates.values() if p.category == category]

    def all_predicates(self) -> list[str]:
        return list(self._predicates.keys())


def _humanize(pid: str) -> str:
    return pid.replace("_", " ")


# ────────────────────────────────────────────────────────────────────
# DEFAULT_JOURNALISM_REGISTRY — pre-populated for the Epstein demo and
# similar journalism corpora. Module-level helpers below delegate to
# this instance so existing imports keep working.
# ────────────────────────────────────────────────────────────────────

def _build_journalism_registry() -> PredicateSalienceRegistry:
    r = PredicateSalienceRegistry()

    # Categories with display order + icon
    for key, order in zip(CATEGORY_ORDER, range(len(CATEGORY_ORDER))):
        r.register_category(key, label=CATEGORY_LABELS[key], icon=CATEGORY_ICONS[key], order=order)

    r.register_many({
        # ── Allegations (1.0 / 0.9) ─────────────────────────────────
        "allegedly_sexually_abused": (CATEGORY_ALLEGATION, 1.0, "is alleged to have sexually abused"),
        "sexually_abused":           (CATEGORY_ALLEGATION, 1.0, "allegedly sexually abused"),
        "was_abused_by":             (CATEGORY_ALLEGATION, 1.0, "was allegedly abused by"),
        "facilitated_abuse_by":      (CATEGORY_ALLEGATION, 1.0, "facilitated abuse by"),
        "allegedly_trafficked":      (CATEGORY_ALLEGATION, 1.0, "is alleged to have trafficked"),
        "trafficked":                (CATEGORY_ALLEGATION, 1.0, "allegedly trafficked"),
        "recruited":                 (CATEGORY_ALLEGATION, 0.9, "allegedly recruited"),
        "solicited":                 (CATEGORY_ALLEGATION, 0.9, "allegedly solicited"),
        "groomed":                   (CATEGORY_ALLEGATION, 0.9, "is alleged to have groomed"),
        "is_potential_co_conspirator_of": (CATEGORY_ALLEGATION, 0.85, "is named as a potential co-conspirator of"),
        "accused":                   (CATEGORY_ALLEGATION, 0.85, "accused"),
        "is_charged_with":           (CATEGORY_ALLEGATION, 0.85, "is charged with"),

        # ── Public denials ──────────────────────────────────────────
        "denied":                    (CATEGORY_DENIAL, 0.85, "denied"),
        "denied_visiting":           (CATEGORY_DENIAL, 0.85, "denies visiting"),
        "never_visited":             (CATEGORY_DENIAL, 0.85, "denies ever visiting"),
        "claimed_to_never_have":     (CATEGORY_DENIAL, 0.85, "claims to have never"),

        # ── Money ───────────────────────────────────────────────────
        "paid":                      (CATEGORY_MONEY, 0.9, "paid"),
        "made_payment_of":           (CATEGORY_MONEY, 0.9, "made a payment of"),
        "transferred_money_to":      (CATEGORY_MONEY, 0.9, "transferred money to"),
        "transferred_to":            (CATEGORY_MONEY, 0.9, "transferred funds to"),
        "wired_money_to":            (CATEGORY_MONEY, 0.95, "wired money to"),
        "settled_for":               (CATEGORY_MONEY, 0.9, "settled for"),
        "compensated":               (CATEGORY_MONEY, 0.85, "compensated"),
        "bribed":                    (CATEGORY_MONEY, 1.0, "allegedly bribed"),
        "has_net_worth":             (CATEGORY_MONEY, 0.6, "has reported net worth"),
        "owns_corporation":          (CATEGORY_MONEY, 0.5, "owns the corporation"),
        "owns_property":             (CATEGORY_MONEY, 0.5, "owns property at"),
        "owned":                     (CATEGORY_MONEY, 0.4, "owned"),
        "is_owner_of":               (CATEGORY_MONEY, 0.4, "is the owner of"),
        "has_account_balance":       (CATEGORY_MONEY, 0.4, "has account balance of"),

        # ── Legal process ───────────────────────────────────────────
        "pleaded_guilty_to":         (CATEGORY_LEGAL, 0.85, "pleaded guilty to"),
        "agreed_to_plead_guilty_to": (CATEGORY_LEGAL, 0.85, "agreed to plead guilty to"),
        "convicted_of":              (CATEGORY_LEGAL, 0.85, "was convicted of"),
        "was_convicted_of":          (CATEGORY_LEGAL, 0.85, "was convicted of"),
        "entered_into_npa_with":     (CATEGORY_LEGAL, 0.9,  "entered a non-prosecution agreement with"),
        "sentenced_to_imprisonment": (CATEGORY_LEGAL, 0.85, "was sentenced to imprisonment"),
        "sentenced_to":              (CATEGORY_LEGAL, 0.8,  "was sentenced to"),
        "sentenced_to_fine":         (CATEGORY_LEGAL, 0.7,  "was fined"),
        "faces_potential_sentence":  (CATEGORY_LEGAL, 0.7,  "faces a potential sentence of"),
        "is_incarcerated_in":        (CATEGORY_LEGAL, 0.75, "is incarcerated at"),
        "was_incarcerated_in":       (CATEGORY_LEGAL, 0.75, "was incarcerated at"),
        "was_indicted_in":           (CATEGORY_LEGAL, 0.75, "was indicted in"),
        "was_arrested_on_date":      (CATEGORY_LEGAL, 0.7,  "was arrested on"),
        "indicted_for":              (CATEGORY_LEGAL, 0.8,  "was indicted for"),
        "was_investigated_in":       (CATEGORY_LEGAL, 0.65, "was investigated in"),
        "investigated":              (CATEGORY_LEGAL, 0.6,  "investigated"),
        "conducted_investigation_of":(CATEGORY_LEGAL, 0.6,  "conducted an investigation of"),
        "convicted_on_date":         (CATEGORY_LEGAL, 0.65, "was convicted on"),
        "pleaded_guilty_in":         (CATEGORY_LEGAL, 0.7,  "pleaded guilty in"),
        "filed_civil_suit_against":  (CATEGORY_LEGAL, 0.65, "filed civil suit against"),
        "waived_right_to":           (CATEGORY_LEGAL, 0.55, "waived the right to"),
        "agreed_to_renounce":        (CATEGORY_LEGAL, 0.55, "agreed to renounce"),
        "has_criminal_record":       (CATEGORY_LEGAL, 0.65, "has a criminal record for"),
        "is_registered_as":          (CATEGORY_LEGAL, 0.7,  "is registered as"),
        "is_registered_in":          (CATEGORY_LEGAL, 0.65, "is registered in"),

        # ── Relationships ───────────────────────────────────────────
        "associated_with":           (CATEGORY_RELATIONSHIP, 0.4, "is associated with"),
        "is_associated_with":        (CATEGORY_RELATIONSHIP, 0.4, "is associated with"),
        "had_business_relationship_with": (CATEGORY_RELATIONSHIP, 0.6, "had a business relationship with"),
        "met":                       (CATEGORY_RELATIONSHIP, 0.55, "met"),
        "knows":                     (CATEGORY_RELATIONSHIP, 0.5, "knows"),
        "is_friend_of":              (CATEGORY_RELATIONSHIP, 0.5, "is a friend of"),
        "was_friend_of":             (CATEGORY_RELATIONSHIP, 0.5, "was a friend of"),
        "was_friendly_with":         (CATEGORY_RELATIONSHIP, 0.5, "was friendly with"),
        "socialized_with":           (CATEGORY_RELATIONSHIP, 0.5, "socialized with"),
        "worked_with":               (CATEGORY_RELATIONSHIP, 0.55, "worked with"),
        "is_brother_of":             (CATEGORY_RELATIONSHIP, 0.4, "is the brother of"),
        "has_spouse":                (CATEGORY_RELATIONSHIP, 0.4, "has spouse"),
        "has_family_member":         (CATEGORY_RELATIONSHIP, 0.35, "has family member"),
        "employed":                  (CATEGORY_RELATIONSHIP, 0.55, "employed"),
        "hired":                     (CATEGORY_RELATIONSHIP, 0.55, "hired"),
        "retained":                  (CATEGORY_RELATIONSHIP, 0.5, "retained"),
        "nominated":                 (CATEGORY_RELATIONSHIP, 0.65, "nominated"),
        "sent_package_to":           (CATEGORY_RELATIONSHIP, 0.4, "sent a package to"),

        # ── Travel ──────────────────────────────────────────────────
        "traveled_to":               (CATEGORY_TRAVEL, 0.6, "traveled to"),
        "traveled":                  (CATEGORY_TRAVEL, 0.55, "traveled"),
        "traveled_to_foreign_country": (CATEGORY_TRAVEL, 0.65, "traveled to the foreign country"),
        "flew_on":                   (CATEGORY_TRAVEL, 0.7, "flew on"),
        "was_on":                    (CATEGORY_TRAVEL, 0.55, "was on"),
        "visited":                   (CATEGORY_TRAVEL, 0.6, "visited"),
        "is_located_in":             (CATEGORY_TRAVEL, 0.4, "is located in"),

        # ── Metadata (hidden by default in the UI) ──────────────────
        "is_a":                      (CATEGORY_METADATA, 0.0, "is"),
        "is_listed_in":              (CATEGORY_METADATA, 0.0, "is listed in"),
        "is_subject_of":             (CATEGORY_METADATA, 0.1, "is the subject of"),
        "is_defendant_in":           (CATEGORY_METADATA, 0.2, "is a defendant in"),
        "is_plaintiff_in":           (CATEGORY_METADATA, 0.2, "is a plaintiff in"),
        "is_petitioner_in":          (CATEGORY_METADATA, 0.2, "is the petitioner in"),
        "is_party_to":               (CATEGORY_METADATA, 0.2, "is a party to"),
        "is_victim_in":              (CATEGORY_METADATA, 0.3, "is named as a victim in"),
        "is_witness_in":             (CATEGORY_METADATA, 0.3, "is a witness in"),
        "is_counsel_for":            (CATEGORY_METADATA, 0.1, "is counsel for"),
        "is_attorney_for":           (CATEGORY_METADATA, 0.1, "is attorney for"),
        "is_lawyer_for":             (CATEGORY_METADATA, 0.1, "is lawyer for"),
        "is_represented_by":         (CATEGORY_METADATA, 0.1, "is represented by"),
        "acted_as_counsel_for":      (CATEGORY_METADATA, 0.1, "acted as counsel for"),
        "had_legal_counsel":         (CATEGORY_METADATA, 0.1, "had legal counsel"),
        "has_attorney":              (CATEGORY_METADATA, 0.1, "has attorney"),
        "represented":               (CATEGORY_METADATA, 0.15, "represented"),
        "represents":                (CATEGORY_METADATA, 0.15, "represents"),
        "is_prosecuting":            (CATEGORY_METADATA, 0.2, "is prosecuting"),
        "filed":                     (CATEGORY_METADATA, 0.1, "filed"),
        "filed_on_date":             (CATEGORY_METADATA, 0.1, "filed on"),
        "filed_document":            (CATEGORY_METADATA, 0.1, "filed document"),
        "named_in":                  (CATEGORY_METADATA, 0.2, "is named in"),
        "classified_as":             (CATEGORY_METADATA, 0.1, "is classified as"),
        "authored":                  (CATEGORY_METADATA, 0.2, "authored"),
        "acted_as":                  (CATEGORY_METADATA, 0.2, "acted as"),
        "has_short_title":           (CATEGORY_METADATA, 0.0, "has the title"),
        "has_docket_number":         (CATEGORY_METADATA, 0.0, "has docket number"),
        "has_case_number":           (CATEGORY_METADATA, 0.0, "has case number"),
        "has_age":                   (CATEGORY_METADATA, 0.1, "has age"),
        "has_citizenship":           (CATEGORY_METADATA, 0.1, "has citizenship"),
        "has_residence_in":          (CATEGORY_METADATA, 0.15, "has residence in"),
        "had_residence_in":          (CATEGORY_METADATA, 0.15, "had residence in"),
        "resided_in":                (CATEGORY_METADATA, 0.15, "resided in"),
        "resides_in":                (CATEGORY_METADATA, 0.15, "resides in"),
        "was_born_in":               (CATEGORY_METADATA, 0.15, "was born in"),
        "was_known_as":              (CATEGORY_METADATA, 0.15, "was also known as"),
        "used_alias":                (CATEGORY_METADATA, 0.2, "used the alias"),
        "has_passport_count":        (CATEGORY_METADATA, 0.2, "has passport count"),
        "profession":                (CATEGORY_METADATA, 0.1, "profession"),
        "possessed":                 (CATEGORY_METADATA, 0.3, "possessed"),
        "possesses":                 (CATEGORY_METADATA, 0.3, "possesses"),
        "is_housed_in":              (CATEGORY_METADATA, 0.2, "is housed in"),
        "was":                       (CATEGORY_METADATA, 0.0, "was"),
    })

    # Pattern fallbacks for unregistered predicates following journalism
    # naming conventions.
    def _journalism_fallback(pid: str) -> Optional[Tuple[str, float, str]]:
        if pid.startswith("allegedly_"):
            return (CATEGORY_ALLEGATION, 0.85, _humanize(pid))
        if pid.startswith("denied_") or pid.startswith("never_") or pid.startswith("claimed_to_"):
            return (CATEGORY_DENIAL, 0.8, _humanize(pid))
        if pid.startswith("paid_") or pid.startswith("transferred_") or pid.startswith("wired_"):
            return (CATEGORY_MONEY, 0.85, _humanize(pid))
        if pid.startswith("convicted_") or pid.startswith("sentenced_") or pid.startswith("indicted_") or pid.startswith("pleaded_") or pid.startswith("arrested_"):
            return (CATEGORY_LEGAL, 0.7, _humanize(pid))
        if pid.startswith("traveled_") or pid.startswith("flew_") or pid.startswith("visited_"):
            return (CATEGORY_TRAVEL, 0.55, _humanize(pid))
        if pid.startswith("has_") or pid.startswith("is_listed_") or pid.startswith("classified_") or pid.startswith("filed_"):
            return (CATEGORY_METADATA, 0.05, _humanize(pid))
        if pid == "is_a":
            return (CATEGORY_METADATA, 0.0, "is")
        return None

    r.register_fallback(_journalism_fallback)
    r.set_default(CATEGORY_RELATIONSHIP, 0.3)

    # ── Synonym groups (conservative seed) ───────────────────────────
    # Members collapse into one logical claim when (subject, object, window)
    # match. Only predicates already registered above are tagged.
    _SYNONYM_SEED: dict[str, set[str]] = {
        "finance_transfer": {
            "paid", "made_payment_of", "transferred_money_to",
            "transferred_to", "wired_money_to", "compensated",
        },
        "travel_visit": {
            "visited", "traveled_to", "traveled_to_foreign_country",
            "traveled",
        },
    }
    for group, members in _SYNONYM_SEED.items():
        for pid in members:
            r.set_synonym_group(pid, group)

    return r


DEFAULT_JOURNALISM_REGISTRY: PredicateSalienceRegistry = _build_journalism_registry()


# ────────────────────────────────────────────────────────────────────
# Module-level helpers — preserved for backward compatibility.
# Every existing import (predicate_meta, human_label, claim_salience,
# is_metadata, category_of) goes through DEFAULT_JOURNALISM_REGISTRY.
# ────────────────────────────────────────────────────────────────────

def predicate_meta(pid: str) -> Tuple[str, float, str]:
    return DEFAULT_JOURNALISM_REGISTRY.meta(pid)


def human_label(pid: str) -> str:
    return DEFAULT_JOURNALISM_REGISTRY.human_label(pid)


def category_of(pid: str) -> str:
    return DEFAULT_JOURNALISM_REGISTRY.category_of(pid)


def claim_salience(
    predicate_id: str,
    corroboration_count: int = 0,
    has_evidence_text: bool = False,
    source_diversity: int = 1,
) -> float:
    return DEFAULT_JOURNALISM_REGISTRY.claim_salience(
        predicate_id, corroboration_count, has_evidence_text, source_diversity,
    )


def is_metadata(predicate_id: str) -> bool:
    return DEFAULT_JOURNALISM_REGISTRY.is_metadata(predicate_id)


# ────────────────────────────────────────────────────────────────────
# Per-domain leaderboard predicate sets — also preserved for back-compat
# and used by the journalism investigations. Other domains build their
# own.
# ────────────────────────────────────────────────────────────────────

LEADERBOARD_PREDICATES: dict[str, list[str]] = {
    "money": [
        "paid", "made_payment_of", "transferred_money_to", "transferred_to",
        "wired_money_to", "settled_for", "compensated", "bribed",
    ],
    "allegations": [
        "allegedly_sexually_abused", "sexually_abused", "was_abused_by",
        "facilitated_abuse_by", "allegedly_trafficked", "trafficked",
        "recruited", "solicited", "groomed", "is_potential_co_conspirator_of",
        "accused", "is_charged_with",
    ],
    "denials": [
        "denied", "denied_visiting", "never_visited", "claimed_to_never_have",
    ],
    "flights": [
        "flew_on", "was_on", "traveled_to", "traveled_to_foreign_country",
        "visited",
    ],
}


def leaderboard_categories() -> list[str]:
    return list(LEADERBOARD_PREDICATES.keys())


# ────────────────────────────────────────────────────────────────────
# Legacy export: PREDICATE_META as a dict view of the journalism registry
# for any caller still referencing the old constant.
# ────────────────────────────────────────────────────────────────────

PREDICATE_META: dict[str, Tuple[str, float, str]] = {
    p.predicate_id: (p.category, p.weight, p.label)
    for p in DEFAULT_JOURNALISM_REGISTRY._predicates.values()
}
