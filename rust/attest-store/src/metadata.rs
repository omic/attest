//! Metadata storage: vocabularies, predicate constraints, payload schemas.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Vocabulary definition for a namespace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Vocabulary {
    #[serde(default)]
    pub entity_types: Vec<String>,
    #[serde(default)]
    pub predicate_types: Vec<String>,
    #[serde(default)]
    pub source_types: Vec<String>,
}

/// Metadata store for vocabularies, predicate constraints, and payload schemas.
///
/// Predicate constraints and payload schemas are stored as JSON strings
/// rather than serde_json::Value to ensure bincode round-trip compatibility
/// (bincode does not support deserialize_any, which Value requires).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetadataStore {
    vocabularies: HashMap<String, Vocabulary>,
    predicate_constraints: HashMap<String, String>,
    payload_schemas: HashMap<String, String>,
}

impl MetadataStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        self.vocabularies.insert(namespace.to_string(), vocab);
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        self.predicate_constraints
            .insert(predicate_id.to_string(), constraints.to_string());
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        self.payload_schemas
            .insert(schema_id.to_string(), schema.to_string());
    }

    pub fn vocabularies(&self) -> &HashMap<String, Vocabulary> {
        &self.vocabularies
    }

    pub fn predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        self.predicate_constraints
            .iter()
            .filter_map(|(k, v)| {
                serde_json::from_str(v).ok().map(|val| (k.clone(), val))
            })
            .collect()
    }

    pub fn payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.payload_schemas
            .iter()
            .filter_map(|(k, v)| {
                serde_json::from_str(v).ok().map(|val| (k.clone(), val))
            })
            .collect()
    }
}
