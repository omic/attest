"""Built-in vocabulary constants and special predicate sets."""

BUILT_IN_ENTITY_TYPES: set[str] = {
    "entity", "event", "metric", "document", "agent", "system",
}

BUILT_IN_PREDICATE_TYPES: set[str] = {
    "relates_to", "caused", "observed", "derived_from", "contradicts",
    "same_as", "not_same_as", "retracted", "contradiction_resolved", "inquiry",
}

BUILT_IN_SOURCE_TYPES: set[str] = {
    "observation", "computation", "document_extraction",
    "llm_inference", "human_annotation", "chat_extraction",
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
    ("promotes", "inhibits"): "inhibits",
    # upregulates / downregulates
    ("upregulates", "activates"): "upregulates",
    ("upregulates", "inhibits"): "downregulates",
    ("downregulates", "activates"): "downregulates",
    ("downregulates", "inhibits"): "upregulates",
}

# Predicates that don't compose meaningfully
_WEAK_PREDICATES: set[str] = {"associated_with", "relates_to", "resembles"}


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


# Quantitative payload schemas — schemas whose data has {value, unit}
QUANTITATIVE_SCHEMAS: set[str] = {
    "binding_affinity",
    "expression_data",
    "differential_expression",
    "clinical_outcome",
    "enzyme_activity",
    "latency_measurement",
}
