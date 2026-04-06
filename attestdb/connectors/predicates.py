"""Predicate vocabulary for connectors.

Centralizes the classification of predicates as directional (composable
for 2-hop prediction via ``predict()``) or attributive (structural,
non-composable).  Connectors use ``predicate_type()`` instead of
hardcoding ``"directional"`` or ``"relates_to"`` strings.
"""

# Directional predicates: asymmetric relationships that are composable
# for causal chain prediction.  A→B + B→C = A→C.
DIRECTIONAL_PREDICATES: frozenset[str] = frozenset({
    "affects_service",
    "assigned_to",
    "authored_by",
    "belongs_to",
    "in_pipeline",
    "mentioned",
    "opened_by",
    "owned_by",
    "posted_in",
    "reported_by",
    "submitted_by",
    "works_at",
})

# Attribute predicates: symmetric or structural properties.
# Not composable — used for filtering and display.
ATTRIBUTE_PREDICATES: frozenset[str] = frozenset({
    "associated_with",
    "categorized_as",
    "closes_on",
    "escalation_policy",
    "has_amount",
    "has_domain",
    "has_priority",
    "has_role",
    "has_size",
    "has_stage",
    "has_state",
    "has_status",
    "has_urgency",
    "in_industry",
    "labeled",
    "tagged",
    "titled",
})


def predicate_type(pred_name: str) -> str:
    """Return ``"directional"`` or ``"relates_to"`` for *pred_name*.

    Directional predicates are composable for 2-hop causal prediction.
    Everything else is structural/attributive.
    """
    if pred_name in DIRECTIONAL_PREDICATES:
        return "directional"
    return "relates_to"
