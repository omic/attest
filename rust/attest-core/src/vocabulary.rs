//! Built-in vocabulary constants and special predicate sets.

use std::collections::HashSet;
use std::sync::LazyLock;

/// Built-in entity types.
pub static BUILT_IN_ENTITY_TYPES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    ["entity", "event", "metric", "document", "agent", "system"]
        .into_iter()
        .collect()
});

/// Built-in predicate types.
pub static BUILT_IN_PREDICATE_TYPES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "relates_to",
        "caused",
        "observed",
        "derived_from",
        "contradicts",
        "same_as",
        "not_same_as",
        "retracted",
        "contradiction_resolved",
        "inquiry",
    ]
    .into_iter()
    .collect()
});

/// Built-in source types.
pub static BUILT_IN_SOURCE_TYPES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "observation",
        "computation",
        "document_extraction",
        "llm_inference",
        "human_annotation",
    ]
    .into_iter()
    .collect()
});

/// Predicates with special engine semantics (alias resolution).
pub static ALIAS_PREDICATES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| ["same_as", "not_same_as"].into_iter().collect());

/// Contradiction-related predicates.
pub static CONTRADICTION_PREDICATES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| ["contradicts", "contradiction_resolved"].into_iter().collect());

/// Quantitative payload schemas — schemas whose data has {value, unit}.
pub static QUANTITATIVE_SCHEMAS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "binding_affinity",
        "expression_data",
        "differential_expression",
        "clinical_outcome",
        "enzyme_activity",
        "latency_measurement",
    ]
    .into_iter()
    .collect()
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_built_in_entity_types() {
        assert!(BUILT_IN_ENTITY_TYPES.contains("entity"));
        assert!(BUILT_IN_ENTITY_TYPES.contains("system"));
        assert!(!BUILT_IN_ENTITY_TYPES.contains("protein"));
    }

    #[test]
    fn test_alias_predicates() {
        assert!(ALIAS_PREDICATES.contains("same_as"));
        assert!(ALIAS_PREDICATES.contains("not_same_as"));
        assert!(!ALIAS_PREDICATES.contains("relates_to"));
    }

    #[test]
    fn test_quantitative_schemas() {
        assert!(QUANTITATIVE_SCHEMAS.contains("binding_affinity"));
        assert_eq!(QUANTITATIVE_SCHEMAS.len(), 6);
    }
}
