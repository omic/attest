//! Append-only claim log with indexed lookups.
//!
//! The claim log is the source of truth. All indexes can be rebuilt from it.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use attest_core::types::Claim;

/// Append-only claim storage with secondary indexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimLog {
    /// The ordered sequence of claims (append-only).
    claims: Vec<Claim>,

    /// claim_id → index in `claims` vec.
    claim_id_index: HashMap<String, usize>,

    /// content_id → list of indices (for corroboration lookups).
    content_id_index: HashMap<String, Vec<usize>>,

    /// entity_id → list of claim indices where entity is subject or object.
    entity_index: HashMap<String, Vec<usize>>,

    /// Bidirectional adjacency index — maintained on insert.
    /// entity_id → set of neighbor entity_ids.
    #[serde(default)]
    adjacency: HashMap<String, HashSet<String>>,

    /// Timestamp index — (timestamp, claim_index) pairs sorted by timestamp.
    /// Enables O(log n) range queries.
    #[serde(default)]
    timestamp_index: Vec<(i64, usize)>,

    /// Whether the timestamp index is known to be sorted.
    #[serde(default)]
    ts_sorted: bool,

    /// source_id → list of claim indices.
    #[serde(default)]
    source_index: HashMap<String, Vec<usize>>,

    /// predicate_id → list of claim indices.
    #[serde(default)]
    predicate_index: HashMap<String, Vec<usize>>,
}

impl ClaimLog {
    pub fn new() -> Self {
        Self {
            claims: Vec::new(),
            claim_id_index: HashMap::new(),
            content_id_index: HashMap::new(),
            entity_index: HashMap::new(),
            adjacency: HashMap::new(),
            timestamp_index: Vec::new(),
            ts_sorted: true, // empty is sorted
            source_index: HashMap::new(),
            predicate_index: HashMap::new(),
        }
    }

    /// Append a claim. Caller must ensure claim_id uniqueness.
    pub fn append(&mut self, claim: Claim) {
        let idx = self.claims.len();

        // Index by claim_id
        self.claim_id_index
            .insert(claim.claim_id.clone(), idx);

        // Index by content_id
        self.content_id_index
            .entry(claim.content_id.clone())
            .or_default()
            .push(idx);

        // Index by subject and object entity (deduplicate self-referential claims)
        self.entity_index
            .entry(claim.subject.id.clone())
            .or_default()
            .push(idx);
        if claim.object.id != claim.subject.id {
            self.entity_index
                .entry(claim.object.id.clone())
                .or_default()
                .push(idx);
        }

        // Maintain adjacency index (bidirectional, skip self-loops)
        if claim.subject.id != claim.object.id {
            self.adjacency
                .entry(claim.subject.id.clone())
                .or_default()
                .insert(claim.object.id.clone());
            self.adjacency
                .entry(claim.object.id.clone())
                .or_default()
                .insert(claim.subject.id.clone());
        }

        // Maintain source_id index
        self.source_index
            .entry(claim.provenance.source_id.clone())
            .or_default()
            .push(idx);

        // Maintain predicate_id index
        self.predicate_index
            .entry(claim.predicate.id.clone())
            .or_default()
            .push(idx);

        // Maintain timestamp index
        // Track whether insertion preserves sort order
        if self.ts_sorted && !self.timestamp_index.is_empty() {
            // SAFETY: unwrap is safe — the `is_empty()` guard above confirms at least one element
            let last_ts = self.timestamp_index.last().unwrap().0;
            if claim.timestamp < last_ts {
                self.ts_sorted = false;
            }
        }
        self.timestamp_index.push((claim.timestamp, idx));

        self.claims.push(claim);
    }

    /// Check if a claim_id exists.
    pub fn contains(&self, claim_id: &str) -> bool {
        self.claim_id_index.contains_key(claim_id)
    }

    /// Get a claim by claim_id.
    pub fn get(&self, claim_id: &str) -> Option<&Claim> {
        self.claim_id_index
            .get(claim_id)
            .and_then(|&idx| self.claims.get(idx))
    }

    /// Get all claims with a given content_id.
    pub fn by_content_id(&self, content_id: &str) -> Vec<&Claim> {
        self.content_id_index
            .get(content_id)
            .map(|indices| indices.iter().filter_map(|&idx| self.claims.get(idx)).collect())
            .unwrap_or_default()
    }

    /// Count claims involving an entity without allocating a Vec.
    pub fn count_for_entity(&self, entity_id: &str) -> usize {
        self.entity_index
            .get(entity_id)
            .map(|indices| indices.len())
            .unwrap_or(0)
    }

    /// Get all claims involving an entity (as subject or object).
    pub fn for_entity(&self, entity_id: &str) -> Vec<&Claim> {
        self.entity_index
            .get(entity_id)
            .map(|indices| indices.iter().filter_map(|&idx| self.claims.get(idx)).collect())
            .unwrap_or_default()
    }

    /// Get all claims involving any entity in the set.
    pub fn for_entities(&self, entity_ids: &HashSet<String>) -> Vec<&Claim> {
        let mut seen = HashSet::new();
        let mut result = Vec::new();
        for eid in entity_ids {
            if let Some(indices) = self.entity_index.get(eid) {
                for &idx in indices {
                    if seen.insert(idx) {
                        if let Some(claim) = self.claims.get(idx) {
                            result.push(claim);
                        }
                    }
                }
            }
        }
        result
    }

    /// Filter claims for an entity with optional predicate_type, source_type, min_confidence.
    pub fn for_entity_filtered(
        &self,
        entity_ids: &HashSet<String>,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<&Claim> {
        self.for_entities(entity_ids)
            .into_iter()
            .filter(|c| {
                if let Some(pt) = predicate_type {
                    if c.predicate.predicate_type != pt {
                        return false;
                    }
                }
                if let Some(st) = source_type {
                    if c.provenance.source_type != st {
                        return false;
                    }
                }
                if c.confidence < min_confidence {
                    return false;
                }
                true
            })
            .collect()
    }

    // ── Adjacency index ─────────────────────────────────────────────

    /// Get the pre-built bidirectional adjacency list. O(1) — returns a reference.
    pub fn adjacency_list(&self) -> &HashMap<String, HashSet<String>> {
        &self.adjacency
    }

    // ── Timestamp index ─────────────────────────────────────────────

    /// Ensure the timestamp index is sorted (required for binary search).
    /// Called lazily before range queries if non-monotonic timestamps were inserted.
    fn ensure_ts_sorted(&mut self) {
        if !self.ts_sorted {
            self.timestamp_index.sort_by_key(|&(ts, _)| ts);
            self.ts_sorted = true;
        }
    }

    /// Get all claims within a timestamp range [min_ts, max_ts] (inclusive).
    /// Uses binary search on the sorted timestamp index — O(log n + k) where k is result count.
    pub fn in_time_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<&Claim> {
        self.ensure_ts_sorted();
        let start = self.timestamp_index.partition_point(|&(ts, _)| ts < min_ts);
        let end = self.timestamp_index.partition_point(|&(ts, _)| ts <= max_ts);
        self.timestamp_index[start..end]
            .iter()
            .filter_map(|(_, idx)| self.claims.get(*idx))
            .collect()
    }

    /// Get the most recent N claims by timestamp.
    /// Uses the sorted index and iterates from the end — O(n) sort check + O(k) iteration.
    pub fn most_recent(&mut self, n: usize) -> Vec<&Claim> {
        self.ensure_ts_sorted();
        self.timestamp_index
            .iter()
            .rev()
            .take(n)
            .filter_map(|(_, idx)| self.claims.get(*idx))
            .collect()
    }

    /// Rebuild derived indexes (adjacency, timestamp, source, predicate) from claim data.
    /// Called after deserialization when `#[serde(default)]` fields may be empty.
    pub fn rebuild_derived_indexes(&mut self) {
        if self.claims.is_empty() {
            return;
        }

        // Rebuild adjacency (skip self-loops)
        if self.adjacency.is_empty() {
            for claim in &self.claims {
                if claim.subject.id != claim.object.id {
                    self.adjacency
                        .entry(claim.subject.id.clone())
                        .or_default()
                        .insert(claim.object.id.clone());
                    self.adjacency
                        .entry(claim.object.id.clone())
                        .or_default()
                        .insert(claim.subject.id.clone());
                }
            }
        }

        // Rebuild timestamp index
        if self.timestamp_index.is_empty() {
            self.timestamp_index = self
                .claims
                .iter()
                .enumerate()
                .map(|(idx, c)| (c.timestamp, idx))
                .collect();
            self.ts_sorted = false;
        }

        // Rebuild source_id index
        if self.source_index.is_empty() {
            for (idx, claim) in self.claims.iter().enumerate() {
                self.source_index
                    .entry(claim.provenance.source_id.clone())
                    .or_default()
                    .push(idx);
            }
        }

        // Rebuild predicate_id index
        if self.predicate_index.is_empty() {
            for (idx, claim) in self.claims.iter().enumerate() {
                self.predicate_index
                    .entry(claim.predicate.id.clone())
                    .or_default()
                    .push(idx);
            }
        }
    }

    /// Return a slice of all claims.
    pub fn all_claims(&self) -> &[Claim] {
        &self.claims
    }

    /// Total number of claims.
    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    /// Iterate over all claims.
    pub fn iter(&self) -> impl Iterator<Item = &Claim> {
        self.claims.iter()
    }

    /// Get all unique entity IDs that appear as subjects or objects.
    pub fn all_entity_ids(&self) -> HashSet<&str> {
        self.entity_index.keys().map(|s| s.as_str()).collect()
    }

    /// Get all claims with a given source_id.
    pub fn by_source_id(&self, source_id: &str) -> Vec<&Claim> {
        self.source_index
            .get(source_id)
            .map(|indices| indices.iter().filter_map(|&idx| self.claims.get(idx)).collect())
            .unwrap_or_default()
    }

    /// Get all claims with a given predicate_id.
    pub fn by_predicate_id(&self, predicate_id: &str) -> Vec<&Claim> {
        self.predicate_index
            .get(predicate_id)
            .map(|indices| indices.iter().filter_map(|&idx| self.claims.get(idx)).collect())
            .unwrap_or_default()
    }

    /// Count claims by predicate_type.
    pub fn count_by_predicate_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for claim in &self.claims {
            *counts
                .entry(claim.predicate.predicate_type.clone())
                .or_insert(0) += 1;
        }
        counts
    }

    /// Count claims by source_type.
    pub fn count_by_source_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for claim in &self.claims {
            *counts
                .entry(claim.provenance.source_type.clone())
                .or_insert(0) += 1;
        }
        counts
    }
}

impl Default for ClaimLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

    fn make_claim(claim_id: &str, subj: &str, pred: &str, obj: &str) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: format!("{subj}|{pred}|{obj}"),
            subject: EntityRef {
                id: subj.to_string(),
                entity_type: "entity".to_string(),
                display_name: subj.to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: pred.to_string(),
                predicate_type: "relates_to".to_string(),
            },
            object: EntityRef {
                id: obj.to_string(),
                entity_type: "entity".to_string(),
                display_name: obj.to_string(),
                external_ids: Default::default(),
            },
            confidence: 0.7,
            provenance: Provenance {
                source_type: "observation".to_string(),
                source_id: "test".to_string(),
                method: None,
                chain: vec![],
                model_version: None,
                organization: None,
            },
            embedding: None,
            payload: None,
            timestamp: 1000,
            status: ClaimStatus::Active,
        }
    }

    #[test]
    fn test_append_and_get() {
        let mut log = ClaimLog::new();
        log.append(make_claim("c1", "a", "rel", "b"));
        assert!(log.contains("c1"));
        assert!(!log.contains("c2"));
        assert_eq!(log.get("c1").unwrap().claim_id, "c1");
    }

    #[test]
    fn test_by_content_id() {
        let mut log = ClaimLog::new();
        let mut c1 = make_claim("c1", "a", "rel", "b");
        c1.content_id = "shared".to_string();
        let mut c2 = make_claim("c2", "a", "rel", "b");
        c2.content_id = "shared".to_string();
        log.append(c1);
        log.append(c2);
        assert_eq!(log.by_content_id("shared").len(), 2);
    }

    #[test]
    fn test_for_entity() {
        let mut log = ClaimLog::new();
        log.append(make_claim("c1", "a", "rel", "b"));
        log.append(make_claim("c2", "b", "rel", "c"));
        // "b" is in both claims
        assert_eq!(log.for_entity("b").len(), 2);
        // "a" is only in c1
        assert_eq!(log.for_entity("a").len(), 1);
        // "d" is in none
        assert_eq!(log.for_entity("d").len(), 0);
    }

    #[test]
    fn test_for_entities_deduplicates() {
        let mut log = ClaimLog::new();
        log.append(make_claim("c1", "a", "rel", "b"));
        let mut ids = HashSet::new();
        ids.insert("a".to_string());
        ids.insert("b".to_string());
        // c1 involves both a and b, should appear once
        assert_eq!(log.for_entities(&ids).len(), 1);
    }

    #[test]
    fn test_len() {
        let mut log = ClaimLog::new();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        log.append(make_claim("c1", "a", "rel", "b"));
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }
}
