//! Entity storage with claim count tracking.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use attest_core::types::EntitySummary;

/// Tokenize a string for the text index: lowercase, split on non-alphanumeric, min 2 chars.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_string())
        .collect()
}

/// BM25 scoring parameters
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// Compute BM25 score for a document matching a single query term.
/// - tf: term frequency in document
/// - df: document frequency (how many docs contain this term)
/// - dl: document length (number of tokens)
/// - avgdl: average document length across corpus
/// - n: total number of documents
pub fn bm25_score(tf: f64, df: f64, dl: f64, avgdl: f64, n: f64) -> f64 {
    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
    let tf_norm = (tf * (BM25_K1 + 1.0)) / (tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avgdl));
    idf * tf_norm
}

/// Stored entity data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityData {
    pub id: String,
    pub entity_type: String,
    pub display_name: String,
    pub external_ids: HashMap<String, String>,
    pub created_at: i64,
}

impl EntityData {
    /// Convert to an EntitySummary with the given claim count.
    pub fn to_summary(&self, claim_count: usize) -> EntitySummary {
        EntitySummary {
            id: self.id.clone(),
            name: self.display_name.clone(),
            entity_type: self.entity_type.clone(),
            external_ids: self.external_ids.clone(),
            claim_count,
        }
    }
}

/// Entity storage with type-based indexing and text search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityStore {
    /// entity_id → entity data.
    entities: HashMap<String, EntityData>,

    /// entity_type → set of entity_ids.
    type_index: HashMap<String, Vec<String>>,

    /// Inverted text index: lowercase token → set of entity_ids.
    /// Built from display_name and entity_id on upsert.
    #[serde(default)]
    text_index: HashMap<String, Vec<String>>,
}

impl EntityStore {
    pub fn new() -> Self {
        Self {
            entities: HashMap::new(),
            type_index: HashMap::new(),
            text_index: HashMap::new(),
        }
    }

    /// Insert or update an entity. Merges external_ids if entity already exists.
    pub fn upsert(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        if let Some(existing) = self.entities.get_mut(entity_id) {
            // Merge external IDs
            if let Some(new_ids) = external_ids {
                for (k, v) in new_ids {
                    existing.external_ids.insert(k.clone(), v.clone());
                }
            }
            return;
        }

        let display = if display_name.is_empty() {
            entity_id
        } else {
            display_name
        };

        // Index tokens from display_name and entity_id (deduplicated)
        let mut seen = HashSet::new();
        for token in tokenize(display).into_iter().chain(tokenize(entity_id)) {
            if seen.insert(token.clone()) {
                self.text_index
                    .entry(token)
                    .or_default()
                    .push(entity_id.to_string());
            }
        }

        let data = EntityData {
            id: entity_id.to_string(),
            entity_type: entity_type.to_string(),
            display_name: display.to_string(),
            external_ids: external_ids.cloned().unwrap_or_default(),
            created_at: timestamp,
        };

        self.type_index
            .entry(entity_type.to_string())
            .or_default()
            .push(entity_id.to_string());

        self.entities.insert(entity_id.to_string(), data);
    }

    /// Check if an entity exists.
    pub fn contains(&self, entity_id: &str) -> bool {
        self.entities.contains_key(entity_id)
    }

    /// Get entity data.
    pub fn get(&self, entity_id: &str) -> Option<&EntityData> {
        self.entities.get(entity_id)
    }

    /// Get entity as summary (claim_count must be provided by caller from claim_log).
    pub fn get_summary(&self, entity_id: &str, claim_count: usize) -> Option<EntitySummary> {
        self.entities.get(entity_id).map(|e| e.to_summary(claim_count))
    }

    /// List all entities, optionally filtered by type.
    pub fn list(&self, entity_type: Option<&str>) -> Vec<&EntityData> {
        match entity_type {
            Some(et) => self
                .type_index
                .get(et)
                .map(|ids| {
                    ids.iter()
                        .filter_map(|id| self.entities.get(id))
                        .collect()
                })
                .unwrap_or_default(),
            None => self.entities.values().collect(),
        }
    }

    /// Search entities by text query. Returns entities matching any query token,
    /// ranked by BM25 relevance score.
    pub fn search(&self, query: &str, top_k: usize) -> Vec<&EntityData> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Vec::new();
        }

        let n = self.entities.len() as f64;
        if n == 0.0 {
            return Vec::new();
        }

        // Compute average document length (number of indexed tokens per entity).
        // Sum up all posting list lengths; each entity contributes one entry per
        // unique token, so the sum equals total (entity, token) pairs.
        let total_tokens: usize = self.text_index.values().map(|v| v.len()).sum();
        let avgdl = total_tokens as f64 / n;

        // Collect per-entity token hits and document frequencies
        // tf_map: entity_id -> { token -> count }
        let mut tf_map: HashMap<&str, HashMap<&str, usize>> = HashMap::new();
        let mut df_map: HashMap<&str, usize> = HashMap::new();

        for token in &tokens {
            if let Some(entity_ids) = self.text_index.get(token.as_str()) {
                // df = number of distinct entities containing this token
                *df_map.entry(token.as_str()).or_insert(0) = entity_ids.len();
                for eid in entity_ids {
                    *tf_map
                        .entry(eid.as_str())
                        .or_default()
                        .entry(token.as_str())
                        .or_insert(0) += 1;
                }
            }
        }

        // Pre-compute document lengths (number of tokens per entity)
        // We cache this to avoid re-scanning the text_index for each entity.
        let mut dl_cache: HashMap<&str, usize> = HashMap::new();
        for entity_ids in self.text_index.values() {
            for eid in entity_ids {
                *dl_cache.entry(eid.as_str()).or_insert(0) += 1;
            }
        }

        // Compute BM25 score for each matching entity
        let mut scored: Vec<(&str, f64)> = tf_map
            .iter()
            .map(|(&eid, token_counts)| {
                let dl = *dl_cache.get(eid).unwrap_or(&1) as f64;
                let score: f64 = token_counts
                    .iter()
                    .map(|(&tok, &tf)| {
                        let df = *df_map.get(tok).unwrap_or(&1) as f64;
                        bm25_score(tf as f64, df, dl, avgdl, n)
                    })
                    .sum();
                (eid, score)
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .filter_map(|(eid, _)| self.entities.get(eid))
            .collect()
    }

    /// Rebuild the text index from entity data.
    /// Called after deserialization when `#[serde(default)]` text_index may be empty.
    pub fn rebuild_text_index(&mut self) {
        if !self.text_index.is_empty() || self.entities.is_empty() {
            return; // Already populated or nothing to index
        }
        for (entity_id, data) in &self.entities {
            let display = if data.display_name.is_empty() {
                entity_id.as_str()
            } else {
                data.display_name.as_str()
            };
            let mut seen = HashSet::new();
            for token in tokenize(display) {
                if seen.insert(token.clone()) {
                    self.text_index
                        .entry(token)
                        .or_default()
                        .push(entity_id.clone());
                }
            }
            for token in tokenize(entity_id) {
                if seen.insert(token.clone()) {
                    self.text_index
                        .entry(token)
                        .or_default()
                        .push(entity_id.clone());
                }
            }
        }
    }

    /// Total number of entities.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Count entities per type.
    pub fn count_by_type(&self) -> HashMap<String, usize> {
        self.type_index
            .iter()
            .map(|(t, ids)| (t.clone(), ids.len()))
            .collect()
    }
}

impl Default for EntityStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_and_get() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "BRCA1", None, 0);
        assert!(store.contains("brca1"));
        let data = store.get("brca1").unwrap();
        assert_eq!(data.entity_type, "gene");
        assert_eq!(data.display_name, "BRCA1");
    }

    #[test]
    fn test_upsert_merges_external_ids() {
        let mut store = EntityStore::new();
        let mut ids1 = HashMap::new();
        ids1.insert("ncbi".to_string(), "672".to_string());
        store.upsert("brca1", "gene", "BRCA1", Some(&ids1), 0);

        let mut ids2 = HashMap::new();
        ids2.insert("uniprot".to_string(), "P38398".to_string());
        store.upsert("brca1", "gene", "BRCA1", Some(&ids2), 0);

        let data = store.get("brca1").unwrap();
        assert_eq!(data.external_ids.len(), 2);
        assert_eq!(data.external_ids["ncbi"], "672");
        assert_eq!(data.external_ids["uniprot"], "P38398");
    }

    #[test]
    fn test_empty_display_name_uses_id() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "", None, 0);
        assert_eq!(store.get("brca1").unwrap().display_name, "brca1");
    }

    #[test]
    fn test_list_by_type() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "BRCA1", None, 0);
        store.upsert("tp53", "gene", "TP53", None, 0);
        store.upsert("aspirin", "compound", "Aspirin", None, 0);

        assert_eq!(store.list(Some("gene")).len(), 2);
        assert_eq!(store.list(Some("compound")).len(), 1);
        assert_eq!(store.list(None).len(), 3);
    }

    #[test]
    fn test_text_search() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "BRCA1 DNA repair associated", None, 0);
        store.upsert("tp53", "gene", "TP53 tumor protein p53", None, 0);
        store.upsert("aspirin", "compound", "Aspirin anti-inflammatory", None, 0);

        // Single token search
        let results = store.search("dna", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        // Multi-token: "tumor protein" — tp53 should match both tokens
        let results = store.search("tumor protein", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "tp53");

        // Partial match: "repair aspirin" — both entities match one token each
        let results = store.search("repair aspirin", 10);
        assert_eq!(results.len(), 2);

        // No match
        let results = store.search("nonexistent", 10);
        assert_eq!(results.len(), 0);

        // Top-k limiting
        let results = store.search("tp53 brca1 aspirin", 1);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_count_by_type() {
        let mut store = EntityStore::new();
        store.upsert("a", "gene", "A", None, 0);
        store.upsert("b", "gene", "B", None, 0);
        store.upsert("c", "compound", "C", None, 0);
        let counts = store.count_by_type();
        assert_eq!(counts["gene"], 2);
        assert_eq!(counts["compound"], 1);
    }

    #[test]
    fn test_search_single_char_tokens() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "BRCA1 DNA repair", None, 0);
        store.upsert("tp53", "gene", "TP53 tumor protein", None, 0);

        // Single-char query "a" should be filtered out by tokenize() (len < 2)
        let results = store.search("a", 10);
        assert!(results.is_empty(), "single-char token should yield no results");
    }

    #[test]
    fn test_search_numeric_entities() {
        let mut store = EntityStore::new();
        store.upsert("gene123", "gene", "Gene 123 variant", None, 0);
        store.upsert("tp53", "gene", "TP53 tumor protein", None, 0);

        // "123" is 3 chars, should be tokenized and found
        let results = store.search("123", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "gene123");
    }

    #[test]
    fn test_search_top_k_larger_than_matches() {
        let mut store = EntityStore::new();
        store.upsert("brca1", "gene", "BRCA1 DNA repair", None, 0);
        store.upsert("brca2", "gene", "BRCA2 DNA repair", None, 0);

        // top_k=100 but only 2 entities match "dna" — should return 2 without error
        let results = store.search("dna", 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bm25_score_basic() {
        // With tf=1, df=1, dl=avgdl, the score should be positive
        let score = bm25_score(1.0, 1.0, 5.0, 5.0, 10.0);
        assert!(score > 0.0, "BM25 score should be positive for matching terms");

        // Higher TF should give higher score (same doc length)
        let score_tf1 = bm25_score(1.0, 1.0, 5.0, 5.0, 10.0);
        let score_tf3 = bm25_score(3.0, 1.0, 5.0, 5.0, 10.0);
        assert!(score_tf3 > score_tf1, "Higher TF should increase score");

        // Rarer term (lower DF) should give higher score
        let score_common = bm25_score(1.0, 8.0, 5.0, 5.0, 10.0);
        let score_rare = bm25_score(1.0, 1.0, 5.0, 5.0, 10.0);
        assert!(score_rare > score_common, "Rarer terms should score higher");

        // Shorter document should score higher (same TF)
        let score_short = bm25_score(1.0, 1.0, 3.0, 5.0, 10.0);
        let score_long = bm25_score(1.0, 1.0, 10.0, 5.0, 10.0);
        assert!(score_short > score_long, "Shorter docs should score higher for same TF");
    }

    #[test]
    fn test_bm25_ranking_in_entity_store() {
        let mut store = EntityStore::new();
        // Short doc with "brca1" — high term density
        store.upsert("brca1_gene", "gene", "brca1 gene", None, 0);
        // Long doc with "brca1" — lower term density
        store.upsert(
            "brca1_pathway",
            "pathway",
            "brca1 related dna repair signaling pathway",
            None,
            0,
        );

        let results = store.search("brca1", 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "brca1_gene", "short doc should rank first");
        assert_eq!(results[1].id, "brca1_pathway", "long doc should rank second");
    }
}
