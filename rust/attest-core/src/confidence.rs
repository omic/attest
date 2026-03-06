//! Confidence computation — Tier 1: source-type weights.

use std::collections::HashMap;
use std::sync::LazyLock;

/// Default weight when source_type is not in the map.
pub const DEFAULT_WEIGHT: f64 = 0.50;

/// Source-type weights — Tier 1. No corroboration. No decay.
static SOURCE_TYPE_WEIGHTS: LazyLock<HashMap<&'static str, f64>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    // Built-in vocabulary
    m.insert("observation", 0.70);
    m.insert("computation", 0.50);
    m.insert("document_extraction", 0.60);
    m.insert("llm_inference", 0.30);
    m.insert("human_annotation", 0.90);
    // Bio vocabulary
    m.insert("experimental", 0.85);
    m.insert("crystallography", 0.95);
    m.insert("mass_spec", 0.80);
    m.insert("docking", 0.40);
    m.insert("literature", 0.65);
    // Bio-specific source types
    m.insert("experimental_crystallography", 1.0);
    m.insert("experimental_cryo_em", 0.95);
    m.insert("experimental_spr", 0.9);
    m.insert("experimental_assay", 0.85);
    m.insert("experimental_mass_spec", 0.85);
    m.insert("computational_alphafold", 0.6);
    m.insert("computational_docking", 0.4);
    m.insert("computational_md", 0.5);
    m.insert("computational_ml_prediction", 0.35);
    m.insert("literature_extraction", 0.50);
    m.insert("database_import", 0.70);
    m.insert("expert_annotation", 0.80);
    // Multi-source loaders
    m.insert("pathway_database", 0.70);
    m
});

/// Source-type weight only. No corroboration. No decay.
pub fn tier1_confidence(source_type: &str) -> f64 {
    SOURCE_TYPE_WEIGHTS
        .get(source_type)
        .copied()
        .unwrap_or(DEFAULT_WEIGHT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_source_types() {
        assert!((tier1_confidence("observation") - 0.70).abs() < f64::EPSILON);
        assert!((tier1_confidence("human_annotation") - 0.90).abs() < f64::EPSILON);
        assert!((tier1_confidence("experimental_crystallography") - 1.0).abs() < f64::EPSILON);
        assert!((tier1_confidence("llm_inference") - 0.30).abs() < f64::EPSILON);
    }

    #[test]
    fn test_unknown_source_type_default() {
        assert!((tier1_confidence("unknown_type") - DEFAULT_WEIGHT).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_weights_in_range() {
        for &weight in SOURCE_TYPE_WEIGHTS.values() {
            assert!((0.0..=1.0).contains(&weight), "Weight {weight} out of range");
        }
    }
}
