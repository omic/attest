//! Shared types defining the contract between infrastructure and intelligence.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Claim lifecycle status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Active,
    Archived,
    Tombstoned,
    ProvenanceDegraded,
}

impl Default for ClaimStatus {
    fn default() -> Self {
        Self::Active
    }
}

impl ClaimStatus {
    /// String value matching the Python enum.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Active => "active",
            Self::Archived => "archived",
            Self::Tombstoned => "tombstoned",
            Self::ProvenanceDegraded => "provenance_degraded",
        }
    }
}

impl std::str::FromStr for ClaimStatus {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "archived" => Ok(Self::Archived),
            "tombstoned" => Ok(Self::Tombstoned),
            "provenance_degraded" => Ok(Self::ProvenanceDegraded),
            _ => Err(format!("unknown ClaimStatus: {s}")),
        }
    }
}

/// Reference to an entity in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRef {
    pub id: String,
    pub entity_type: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub external_ids: HashMap<String, String>,
}

/// Reference to a predicate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateRef {
    pub id: String,
    pub predicate_type: String,
}

/// Provenance metadata for a claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    pub source_type: String,
    pub source_id: String,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub chain: Vec<String>,
    #[serde(default)]
    pub model_version: Option<String>,
    #[serde(default)]
    pub organization: Option<String>,
}

/// Typed payload attached to a claim.
///
/// `data` is stored as a JSON string for bincode compatibility (bincode does
/// not support `deserialize_any`, which `serde_json::Value` requires). The
/// custom serde impl is transparent — callers still see `serde_json::Value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Payload {
    pub schema_ref: String,
    #[serde(with = "json_value_as_string")]
    pub data: serde_json::Value,
}

/// Serde helper: serialize/deserialize `serde_json::Value` via a JSON string.
/// This allows bincode (which lacks `deserialize_any`) to round-trip the field.
mod json_value_as_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = value.to_string();
        s.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

/// A fully validated claim in the knowledge graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub claim_id: String,
    pub content_id: String,
    pub subject: EntityRef,
    pub predicate: PredicateRef,
    pub object: EntityRef,
    pub confidence: f64,
    pub provenance: Provenance,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub payload: Option<Payload>,
    #[serde(default)]
    pub timestamp: i64,
    #[serde(default)]
    pub status: ClaimStatus,
}

/// Input structure for ingesting a new claim.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClaimInput {
    /// (id, entity_type)
    pub subject: (String, String),
    /// (id, predicate_type)
    pub predicate: (String, String),
    /// (id, entity_type)
    pub object: (String, String),
    pub provenance: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub payload: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub timestamp: Option<i64>,
    #[serde(default)]
    pub external_ids: Option<HashMap<String, HashMap<String, String>>>,
}

/// Summary information about an entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySummary {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    #[serde(default)]
    pub external_ids: HashMap<String, String>,
    #[serde(default)]
    pub claim_count: usize,
}

/// A relationship derived from one or more claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Relationship {
    pub predicate: String,
    pub target: EntitySummary,
    pub confidence: f64,
    #[serde(default = "default_one")]
    pub n_independent_sources: usize,
    #[serde(default)]
    pub source_types: Vec<String>,
    #[serde(default)]
    pub latest_claim_timestamp: i64,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

fn default_one() -> usize {
    1
}

/// A detected contradiction between two claims.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Contradiction {
    pub claim_a: String,
    pub claim_b: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub evidence_a: i64,
    #[serde(default)]
    pub evidence_b: i64,
    #[serde(default = "default_unresolved")]
    pub status: String,
}

fn default_unresolved() -> String {
    "unresolved".to_string()
}

/// A quantitative measurement extracted from a claim payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantitativeClaim {
    pub predicate: String,
    pub target: String,
    pub value: f64,
    pub unit: String,
    #[serde(default)]
    pub metric: String,
    #[serde(default)]
    pub source_type: String,
    #[serde(default)]
    pub confidence: f64,
}

/// The assembled context frame for a focal entity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextFrame {
    pub focal_entity: EntitySummary,
    #[serde(default)]
    pub direct_relationships: Vec<Relationship>,
    #[serde(default)]
    pub quantitative_data: Vec<QuantitativeClaim>,
    #[serde(default)]
    pub contradictions: Vec<Contradiction>,
    #[serde(default)]
    pub knowledge_gaps: Vec<String>,
    #[serde(default)]
    pub narrative: String,
    #[serde(default)]
    pub provenance_summary: HashMap<String, f64>,
    #[serde(default)]
    pub claim_count: usize,
    #[serde(default = "default_confidence_range")]
    pub confidence_range: (f64, f64),
}

fn default_confidence_range() -> (f64, f64) {
    (0.0, 0.0)
}

/// Result of a batch ingestion operation.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchResult {
    #[serde(default)]
    pub ingested: usize,
    #[serde(default)]
    pub duplicates: usize,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claim_status_roundtrip() {
        for status in [
            ClaimStatus::Active,
            ClaimStatus::Archived,
            ClaimStatus::Tombstoned,
            ClaimStatus::ProvenanceDegraded,
        ] {
            let s = status.as_str();
            let parsed: ClaimStatus = s.parse().unwrap();
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn test_claim_status_parse_invalid() {
        assert!("invalid".parse::<ClaimStatus>().is_err());
    }

    #[test]
    fn test_batch_result_default() {
        let br = BatchResult::default();
        assert_eq!(br.ingested, 0);
        assert_eq!(br.duplicates, 0);
        assert!(br.errors.is_empty());
    }
}
