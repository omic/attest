//! Built-in vocabulary constants and special predicate sets.

use std::collections::{HashMap, HashSet};
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

/// Opposite predicate pairs — used for contradiction detection.
/// If "activates" and "inhibits" are opposites, a claim asserting both
/// for the same subject/object is a contradiction.
pub static OPPOSITE_PREDICATES: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        let pairs = [
            ("activates", "inhibits"),
            ("upregulates", "downregulates"),
            ("promotes", "suppresses"),
            ("increases", "decreases"),
            ("causes", "prevents"),
            ("enables", "blocks"),
            ("enhances", "reduces"),
            ("stabilizes", "destabilizes"),
            ("phosphorylates", "dephosphorylates"),
            ("methylates", "demethylates"),
        ];
        let mut map = HashMap::new();
        for (a, b) in &pairs {
            map.insert(*a, *b);
            map.insert(*b, *a);
        }
        map
    });

/// Inverse predicate pairs — directional reversals for query-time derivation.
/// If A→causes→B exists, querying B yields B→caused_by→A without storage.
/// Distinct from OPPOSITE_PREDICATES (which are contradictions).
pub static INVERSE_PREDICATES: LazyLock<HashMap<&'static str, &'static str>> =
    LazyLock::new(|| {
        let pairs = [
            ("causes", "caused_by"),
            ("activates", "activated_by"),
            ("inhibits", "inhibited_by"),
            ("upregulates", "upregulated_by"),
            ("downregulates", "downregulated_by"),
            ("promotes", "promoted_by"),
            ("suppresses", "suppressed_by"),
            ("targets", "targeted_by"),
            ("treats", "treated_by"),
            ("phosphorylates", "phosphorylated_by"),
            ("dephosphorylates", "dephosphorylated_by"),
            ("methylates", "methylated_by"),
            ("demethylates", "demethylated_by"),
            ("encodes", "encoded_by"),
            ("induces", "induced_by"),
            ("modulates", "modulated_by"),
            ("mediates", "mediated_by"),
            ("stabilizes", "stabilized_by"),
            ("destabilizes", "destabilized_by"),
        ];
        let mut map = HashMap::new();
        for (a, b) in &pairs {
            map.insert(*a, *b);
            map.insert(*b, *a);
        }
        map
    });

/// Causal predicates — the subset used for predict() and transitive composition.
pub static CAUSAL_PREDICATES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "activates", "inhibits", "upregulates", "downregulates",
        "promotes", "suppresses", "increases", "decreases",
        "causes", "prevents", "enables", "blocks",
        "enhances", "reduces", "regulates",
        "stabilizes", "destabilizes",
        "phosphorylates", "dephosphorylates",
        "methylates", "demethylates",
    ]
    .into_iter()
    .collect()
});

/// Predicate composition table — how two predicates compose along a 2-hop path.
/// (A→pred1→C, C→pred2→B) yields A→composed→B.
pub static PREDICATE_COMPOSITION: LazyLock<HashMap<(&'static str, &'static str), &'static str>> =
    LazyLock::new(|| {
        let entries: Vec<((&str, &str), &str)> = vec![
            // activates chains
            (("activates", "activates"), "activates"),
            (("activates", "inhibits"), "inhibits"),
            (("activates", "upregulates"), "upregulates"),
            (("activates", "downregulates"), "downregulates"),
            // inhibits chains (double negative)
            (("inhibits", "activates"), "inhibits"),
            (("inhibits", "inhibits"), "activates"),
            (("inhibits", "upregulates"), "downregulates"),
            (("inhibits", "downregulates"), "upregulates"),
            // upregulates / downregulates
            (("upregulates", "activates"), "upregulates"),
            (("upregulates", "inhibits"), "downregulates"),
            (("upregulates", "upregulates"), "upregulates"),
            (("upregulates", "downregulates"), "downregulates"),
            (("downregulates", "activates"), "downregulates"),
            (("downregulates", "inhibits"), "upregulates"),
            (("downregulates", "downregulates"), "upregulates"),
            (("downregulates", "upregulates"), "downregulates"),
            // causes / prevents
            (("causes", "activates"), "activates"),
            (("causes", "inhibits"), "inhibits"),
            (("causes", "causes"), "causes"),
            (("causes", "prevents"), "prevents"),
            (("prevents", "activates"), "inhibits"),
            (("prevents", "inhibits"), "activates"),
            (("prevents", "causes"), "prevents"),
            (("prevents", "prevents"), "causes"),
        ];
        entries.into_iter().collect()
    });

/// Predicates that don't compose meaningfully — produce "associated_with".
pub static WEAK_PREDICATES: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "associated_with", "relates_to", "resembles", "interacts_with",
        "interacts", "coexpressed_with", "same_as", "associates",
        "participates_in", "investigated_in", "expresses",
    ]
    .into_iter()
    .collect()
});

/// Compose two predicates along a 2-hop path: A→pred1→C→pred2→B.
/// Returns the inferred predicate for A→B.
pub fn compose_predicates(pred_ac: &str, pred_cb: &str) -> &'static str {
    if WEAK_PREDICATES.contains(pred_ac) || WEAK_PREDICATES.contains(pred_cb) {
        return "associated_with";
    }
    if let Some(&result) = PREDICATE_COMPOSITION.get(&(pred_ac, pred_cb)) {
        return result;
    }
    // Same-predicate transitivity — if both are the same causal predicate,
    // the composition is the predicate itself (A→activates→B→activates→C = A→activates→C).
    // We can't return a &'static str for an arbitrary pred_ac, so we check known ones.
    if pred_ac == pred_cb && CAUSAL_PREDICATES.contains(pred_ac) {
        // The predicate is in CAUSAL_PREDICATES which contains &'static str.
        // Find the matching static reference.
        for &p in CAUSAL_PREDICATES.iter() {
            if p == pred_ac {
                return p;
            }
        }
    }
    "associated_with"
}

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

    #[test]
    fn test_opposite_predicates_bidirectional() {
        assert_eq!(OPPOSITE_PREDICATES.get("activates"), Some(&"inhibits"));
        assert_eq!(OPPOSITE_PREDICATES.get("inhibits"), Some(&"activates"));
        assert_eq!(OPPOSITE_PREDICATES.get("causes"), Some(&"prevents"));
        assert_eq!(OPPOSITE_PREDICATES.get("prevents"), Some(&"causes"));
        // Every entry has a reverse
        for (k, v) in OPPOSITE_PREDICATES.iter() {
            assert_eq!(OPPOSITE_PREDICATES.get(v), Some(k));
        }
    }

    #[test]
    fn test_inverse_predicates_bidirectional() {
        assert_eq!(INVERSE_PREDICATES.get("causes"), Some(&"caused_by"));
        assert_eq!(INVERSE_PREDICATES.get("caused_by"), Some(&"causes"));
        assert_eq!(INVERSE_PREDICATES.get("activates"), Some(&"activated_by"));
        assert_eq!(INVERSE_PREDICATES.get("activated_by"), Some(&"activates"));
        // Every entry has a reverse
        for (k, v) in INVERSE_PREDICATES.iter() {
            assert_eq!(INVERSE_PREDICATES.get(v), Some(k), "inverse of {} should map back", v);
        }
    }

    #[test]
    fn test_inverse_and_opposite_are_distinct() {
        // "activates" opposite is "inhibits" (contradiction)
        // "activates" inverse is "activated_by" (directional reversal)
        assert_ne!(
            OPPOSITE_PREDICATES.get("activates"),
            INVERSE_PREDICATES.get("activates"),
        );
    }

    #[test]
    fn test_compose_predicates_basic() {
        assert_eq!(compose_predicates("activates", "activates"), "activates");
        assert_eq!(compose_predicates("activates", "inhibits"), "inhibits");
        assert_eq!(compose_predicates("inhibits", "inhibits"), "activates"); // double negative
        assert_eq!(compose_predicates("causes", "prevents"), "prevents");
        assert_eq!(compose_predicates("prevents", "prevents"), "causes");
    }

    #[test]
    fn test_compose_predicates_weak() {
        assert_eq!(compose_predicates("associated_with", "activates"), "associated_with");
        assert_eq!(compose_predicates("activates", "relates_to"), "associated_with");
    }

    #[test]
    fn test_compose_predicates_same_predicate_fallback() {
        // Same causal predicate composes to itself
        assert_eq!(compose_predicates("enhances", "enhances"), "enhances");
        assert_eq!(compose_predicates("reduces", "reduces"), "reduces");
    }

    #[test]
    fn test_compose_predicates_unknown() {
        assert_eq!(compose_predicates("activates", "binds"), "associated_with");
    }

    #[test]
    fn test_causal_predicates() {
        assert!(CAUSAL_PREDICATES.contains("activates"));
        assert!(CAUSAL_PREDICATES.contains("inhibits"));
        assert!(CAUSAL_PREDICATES.contains("regulates"));
        assert!(!CAUSAL_PREDICATES.contains("binds")); // interaction, not causal
    }
}
