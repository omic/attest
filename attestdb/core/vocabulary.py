"""Built-in vocabulary constants and special predicate sets."""

BUILT_IN_ENTITY_TYPES: set[str] = {
    "entity", "event", "metric", "document", "agent", "system",
}

BUILT_IN_PREDICATE_TYPES: set[str] = {
    "relates_to", "caused", "observed", "derived_from", "contradicts",
    "same_as", "not_same_as", "retracted", "contradiction_resolved", "inquiry",
    "no_evidence_for", "superseded_by",
}

# Research predicates — used by multi-agent collaborative research
RESEARCH_PREDICATES: set[str] = {
    "has_finding", "has_gap", "investigating",
    "no_evidence_for", "superseded_by", "has_strategy",
}

# Opposing research predicates — evidence vs absence
OPPOSITE_RESEARCH_PREDICATES: dict[str, str] = {
    "no_evidence_for": "supports",
    "supports": "no_evidence_for",
}

BUILT_IN_SOURCE_TYPES: set[str] = {
    "observation", "computation", "document_extraction",
    "llm_inference", "human_annotation", "chat_extraction",
    "agent",
}

# Predicates with special engine semantics
CONTRADICTION_PREDICATES: set[str] = {"contradicts", "contradiction_resolved"}
SYMMETRIC_PREDICATES: set[str] = {"interacts", "binds", "resembles", "homolog_of", "same_as"}

# Opposing predicate pairs — used for contradiction detection
OPPOSITE_PREDICATES: dict[str, str] = {
    "activates": "inhibits",
    "inhibits": "activates",
    "upregulates": "downregulates",
    "downregulates": "upregulates",
    "promotes": "suppresses",
    "suppresses": "promotes",
    "increases": "decreases",
    "decreases": "increases",
    "causes": "prevents",
    "prevents": "causes",
    "enables": "blocks",
    "blocks": "enables",
    "enhances": "reduces",
    "reduces": "enhances",
    "stabilizes": "destabilizes",
    "destabilizes": "stabilizes",
    "phosphorylates": "dephosphorylates",
    "dephosphorylates": "phosphorylates",
    "methylates": "demethylates",
    "demethylates": "methylates",
}

# Predicate composition rules — (pred_A_to_C, pred_C_to_B) → predicted_A_to_B
# Causal logic: inhibits + promotes → inhibits,
# inhibits + inhibits → activates (double negative), etc.
PREDICATE_COMPOSITION: dict[tuple[str, str], str] = {
    # activates chains
    ("activates", "activates"): "activates",
    ("activates", "inhibits"): "inhibits",
    ("activates", "promotes"): "promotes",
    ("activates", "downregulates"): "downregulates",
    ("activates", "upregulates"): "upregulates",
    # inhibits chains (double negative)
    ("inhibits", "activates"): "inhibits",
    ("inhibits", "inhibits"): "activates",
    ("inhibits", "promotes"): "inhibits",
    ("inhibits", "downregulates"): "upregulates",
    ("inhibits", "upregulates"): "downregulates",
    # promotes chains
    ("promotes", "activates"): "promotes",
    ("promotes", "inhibits"): "suppresses",
    ("promotes", "promotes"): "promotes",
    ("promotes", "suppresses"): "suppresses",
    # suppresses chains (double negative)
    ("suppresses", "activates"): "suppresses",
    ("suppresses", "inhibits"): "promotes",
    ("suppresses", "promotes"): "suppresses",
    ("suppresses", "suppresses"): "promotes",
    # upregulates / downregulates
    ("upregulates", "activates"): "upregulates",
    ("upregulates", "inhibits"): "downregulates",
    ("upregulates", "upregulates"): "upregulates",
    ("upregulates", "downregulates"): "downregulates",
    ("downregulates", "activates"): "downregulates",
    ("downregulates", "inhibits"): "upregulates",
    ("downregulates", "downregulates"): "upregulates",
    ("downregulates", "upregulates"): "downregulates",
    # causes / prevents chains
    ("causes", "activates"): "activates",
    ("causes", "inhibits"): "inhibits",
    ("causes", "causes"): "causes",
    ("causes", "prevents"): "prevents",
    ("prevents", "activates"): "inhibits",
    ("prevents", "inhibits"): "activates",
    ("prevents", "causes"): "prevents",
    ("prevents", "prevents"): "causes",
    # increases / decreases
    ("increases", "activates"): "activates",
    ("increases", "inhibits"): "inhibits",
    ("decreases", "activates"): "inhibits",
    ("decreases", "inhibits"): "activates",
    # enables / blocks
    ("enables", "activates"): "activates",
    ("enables", "inhibits"): "inhibits",
    ("blocks", "activates"): "inhibits",
    ("blocks", "inhibits"): "activates",
}

# Predicates that don't compose meaningfully — produce "associated_with" in compositions
_WEAK_PREDICATES: set[str] = {
    "associated_with", "relates_to", "resembles", "interacts_with",
    "interacts", "coexpressed_with", "same_as", "associates",
    "participates_in", "investigated_in", "regulates", "expresses",
}

# Causal predicates — directional, compose via PREDICATE_COMPOSITION rules
CAUSAL_PREDICATES: set[str] = {
    "activates", "inhibits", "upregulates", "downregulates",
    "promotes", "suppresses", "increases", "decreases",
    "causes", "prevents", "enables", "blocks",
    "enhances", "reduces", "stabilizes", "destabilizes",
    "phosphorylates", "dephosphorylates", "methylates", "demethylates",
}

# Predicate equivalence — semantically close predicates that should match
# when comparing predictions against expected outcomes.
# "upregulates" and "activates" both mean positive regulation;
# "downregulates" and "inhibits" both mean negative regulation.
PREDICATE_EQUIVALENCE: dict[str, str] = {
    "activates": "positive",
    "upregulates": "positive",
    "promotes": "positive",
    "increases": "positive",
    "enables": "positive",
    "enhances": "positive",
    "stabilizes": "positive",
    "inhibits": "negative",
    "downregulates": "negative",
    "suppresses": "negative",
    "decreases": "negative",
    "blocks": "negative",
    "reduces": "negative",
    "destabilizes": "negative",
    "prevents": "negative",
}


def predicates_agree(pred_a: str, pred_b: str) -> bool:
    """Check if two predicates are semantically equivalent (same direction).

    Returns True if both are positive regulation or both are negative.
    Returns False if they oppose or either is unknown.
    """
    class_a = PREDICATE_EQUIVALENCE.get(pred_a)
    class_b = PREDICATE_EQUIVALENCE.get(pred_b)
    if class_a and class_b:
        return class_a == class_b
    return pred_a == pred_b


def compose_predicates(pred_ac: str, pred_cb: str) -> str:
    """Compose two predicates along A→C→B into predicted A→B predicate.

    Lookup in composition table, fall back to same-predicate transitivity,
    then "associated_with".
    """
    if pred_ac in _WEAK_PREDICATES or pred_cb in _WEAK_PREDICATES:
        return "associated_with"
    result = PREDICATE_COMPOSITION.get((pred_ac, pred_cb))
    if result:
        return result
    # Same-predicate transitivity
    if pred_ac == pred_cb:
        return pred_ac
    return "associated_with"


def predict_predicate_from_paths(
    evidence_paths: list[dict],
) -> tuple[str, float]:
    """Weighted majority vote across bridging paths.

    Each path dict should have:
        - ac_predicates: set[str] — predicates from A to bridge C
        - cb_predicates: set[str] — predicates from bridge C to B
        - path_weight: float — weight (e.g. min(confidence_ac, confidence_cb))

    Returns (predicted_predicate, vote_fraction).
    """
    votes: dict[str, float] = {}
    total_weight = 0.0
    for path in evidence_paths:
        ac_preds = path.get("ac_predicates", set())
        cb_preds = path.get("cb_predicates", set())
        weight = path.get("path_weight", 1.0)
        for ac_p in ac_preds:
            for cb_p in cb_preds:
                composed = compose_predicates(ac_p, cb_p)
                votes[composed] = votes.get(composed, 0.0) + weight
                total_weight += weight
    if not votes:
        return ("associated_with", 0.0)
    best = max(votes, key=lambda k: votes[k])
    fraction = votes[best] / total_weight if total_weight > 0 else 0.0
    return (best, fraction)


# ---------------------------------------------------------------------------
# Knowledge predicates — used by learning layer, recall, and retrieval
# ---------------------------------------------------------------------------

# All predicates that represent recorded knowledge (learnings, bugs, fixes, etc.)
KNOWLEDGE_PREDICATES: set[str] = {
    "has_warning", "has_vulnerability",
    "has_pattern", "has_decision",
    "has_tip",
    "had_bug", "has_issue", "failed_on",
    "has_fix", "resolved",
    "has_status",
    "had_outcome", "produced_by",
    "has_next_steps",
    # Strategic predicates
    "has_goal", "has_strategy", "has_architecture",
    "has_trade_off", "has_convention", "depends_on",
    "has_requirement", "supersedes",
}

# Priority ordering for knowledge predicates (lower = surface first).
# Warnings and vulnerabilities are most actionable and should always appear
# before historical bug reports.
KNOWLEDGE_PRIORITY: dict[str, int] = {
    # Tier 0: Urgent — surface immediately
    "has_warning": 0,
    "has_vulnerability": 0,
    # Tier 1: Strategic — goals, architecture, decisions
    "has_goal": 1,
    "has_strategy": 1,
    "has_architecture": 1,
    "has_decision": 1,
    "has_pattern": 1,
    # Tier 2: Constraints — requirements, trade-offs, conventions, dependencies
    "has_requirement": 2,
    "has_trade_off": 2,
    "has_convention": 2,
    "depends_on": 2,
    "has_tip": 2,
    # Tier 3: Problem history
    "had_bug": 3,
    "has_issue": 3,
    "failed_on": 3,
    # Tier 4-5: Resolutions
    "has_fix": 4,
    "supersedes": 4,
    "resolved": 5,
    "has_status": 5,
    # Tier 6+: Session metadata
    "had_outcome": 6,
    "produced_by": 7,
    "has_next_steps": 8,
}


# Display labels for knowledge predicates (replaces string-hacking with replace())
KNOWLEDGE_LABEL: dict[str, str] = {
    "has_warning": "warning",
    "has_vulnerability": "vulnerability",
    "has_goal": "goal",
    "has_strategy": "strategy",
    "has_architecture": "architecture",
    "has_pattern": "pattern",
    "has_decision": "decision",
    "has_requirement": "requirement",
    "has_trade_off": "trade-off",
    "has_convention": "convention",
    "depends_on": "dependency",
    "has_tip": "tip",
    "had_bug": "bug",
    "has_issue": "issue",
    "failed_on": "failed",
    "has_fix": "fix",
    "supersedes": "supersedes",
    "resolved": "resolved",
    "has_status": "status",
    "had_outcome": "outcome",
    "produced_by": "source",
    "has_next_steps": "next_steps",
}

# Priority threshold: predicates at or below this are "high priority"
# (warnings, patterns, decisions, tips) vs "history" (bugs, fixes, status)
KNOWLEDGE_HIGH_PRIORITY_THRESHOLD = 2


def knowledge_label(predicate_id: str) -> str:
    """Human-readable label for a knowledge predicate."""
    return KNOWLEDGE_LABEL.get(predicate_id, predicate_id)


def knowledge_sort_key(predicate_id: str, confidence: float = 0.5) -> tuple:
    """Sort key for knowledge claims: priority first, then confidence descending."""
    return (KNOWLEDGE_PRIORITY.get(predicate_id, 9), -confidence)


# Quantitative payload schemas — schemas whose data has {value, unit}
QUANTITATIVE_SCHEMAS: set[str] = {
    "binding_affinity",
    "expression_data",
    "differential_expression",
    "clinical_outcome",
    "enzyme_activity",
    "latency_measurement",
}
