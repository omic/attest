//! Storage backend abstraction.
//!
//! Provides two backends:
//! - [`MemoryBackend`]: In-memory store with optional file persistence (original implementation)
//! - [`LmdbBackend`]: File-backed store using LMDB via heed (default for new databases)
//!
//! All implement [`StorageBackend`], and the [`Backend`] enum delegates
//! to whichever variant it holds.

pub mod lmdb;
pub mod memory;
pub mod traits;

pub use memory::MemoryBackend;
pub use self::lmdb::LmdbBackend;
pub use traits::StorageBackend;

use std::collections::{HashMap, HashSet};

use attest_core::errors::AttestError;
use attest_core::types::{Claim, ClaimStatus, EntitySummary};

use crate::metadata::Vocabulary;
use crate::store::StoreStats;

/// Storage backend for RustStore.
pub enum Backend {
    /// In-memory store with optional file persistence via checkpoint + WAL.
    InMemory(Box<MemoryBackend>),
    /// File-backed store using LMDB (default for new databases).
    Lmdb(Box<LmdbBackend>),
}

/// Delegate every `StorageBackend` method to the inner backend.
impl StorageBackend for Backend {
    fn close(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(m) => m.close(),
            Backend::Lmdb(l) => l.close(),
        }
    }

    fn checkpoint(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(m) => m.checkpoint(),
            Backend::Lmdb(l) => l.checkpoint(),
        }
    }

    fn clear_aliases(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(_) => Ok(()),
            Backend::Lmdb(l) => l.clear_aliases(),
        }
    }

    fn purge_source(&mut self, source_id: &str) -> Result<usize, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(0),
            Backend::Lmdb(l) => l.purge_source(source_id),
        }
    }

    fn outgoing_causal_edges(
        &self,
        entity_id: &str,
        causal_predicates: &std::collections::HashSet<String>,
    ) -> Vec<(String, String, f64)> {
        match self {
            Backend::InMemory(_) => Vec::new(),
            Backend::Lmdb(l) => l.outgoing_causal_edges(entity_id, causal_predicates),
        }
    }

    fn compact(&mut self) -> Result<bool, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(false),
            Backend::Lmdb(l) => l.compact(),
        }
    }

    fn schema_version(&self) -> u32 {
        match self {
            Backend::InMemory(_) => 3,
            Backend::Lmdb(l) => l.schema_version(),
        }
    }

    fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        match self {
            Backend::InMemory(m) => m.register_vocabulary(namespace, vocab),
            Backend::Lmdb(l) => l.register_vocabulary(namespace, vocab),
        }
    }

    fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        match self {
            Backend::InMemory(m) => m.register_predicate(predicate_id, constraints),
            Backend::Lmdb(l) => l.register_predicate(predicate_id, constraints),
        }
    }

    fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        match self {
            Backend::InMemory(m) => m.register_payload_schema(schema_id, schema),
            Backend::Lmdb(l) => l.register_payload_schema(schema_id, schema),
        }
    }

    fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        match self {
            Backend::InMemory(m) => m.get_registered_vocabularies(),
            Backend::Lmdb(l) => l.get_registered_vocabularies(),
        }
    }

    fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Backend::InMemory(m) => m.get_predicate_constraints(),
            Backend::Lmdb(l) => l.get_predicate_constraints(),
        }
    }

    fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Backend::InMemory(m) => m.get_payload_schemas(),
            Backend::Lmdb(l) => l.get_payload_schemas(),
        }
    }

    fn resolve(&mut self, entity_id: &str) -> String {
        match self {
            Backend::InMemory(m) => m.resolve(entity_id),
            Backend::Lmdb(l) => l.resolve(entity_id),
        }
    }

    fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        match self {
            Backend::InMemory(m) => m.get_alias_group(entity_id),
            Backend::Lmdb(l) => l.get_alias_group(entity_id),
        }
    }

    fn warm_caches(&self) {
        match self {
            Backend::InMemory(m) => m.warm_caches(),
            Backend::Lmdb(l) => l.warm_caches(),
        }
    }

    fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        match self {
            Backend::InMemory(m) => m.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp),
            Backend::Lmdb(l) => l.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp),
        }
    }

    fn update_display_name(&mut self, entity_id: &str, new_display: &str) -> bool {
        match self {
            Backend::InMemory(m) => m.update_display_name(entity_id, new_display),
            Backend::Lmdb(l) => l.update_display_name(entity_id, new_display),
        }
    }

    fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(m) => {
                for (id, etype, display, ext_ids) in entities {
                    m.upsert_entity(id, etype, display, Some(ext_ids), timestamp);
                }
                Ok(())
            }
            Backend::Lmdb(l) => l.upsert_entities_batch(entities, timestamp),
        }
    }

    fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        match self {
            Backend::InMemory(m) => m.get_entity(entity_id),
            Backend::Lmdb(l) => l.get_entity(entity_id),
        }
    }

    fn get_entities_batch(&self, entity_ids: &[&str]) -> HashMap<String, (String, String)> {
        match self {
            Backend::InMemory(m) => m.get_entities_batch(entity_ids),
            Backend::Lmdb(l) => l.get_entities_batch(entity_ids),
        }
    }

    fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
        offset: usize,
        limit: usize,
    ) -> Vec<EntitySummary> {
        match self {
            Backend::InMemory(m) => m.list_entities(entity_type, min_claims, offset, limit),
            Backend::Lmdb(l) => l.list_entities(entity_type, min_claims, offset, limit),
        }
    }

    fn count_entities(&self, entity_type: Option<&str>, min_claims: usize) -> usize {
        match self {
            Backend::InMemory(m) => m.count_entities(entity_type, min_claims),
            Backend::Lmdb(l) => l.count_entities(entity_type, min_claims),
        }
    }

    fn count_claims(&self) -> usize {
        match self {
            Backend::InMemory(m) => m.count_claims(),
            Backend::Lmdb(l) => l.count_claims(),
        }
    }

    fn insert_claim(&mut self, claim: Claim, checkpoint_interval: u64) -> bool {
        match self {
            Backend::InMemory(m) => m.insert_claim(claim, checkpoint_interval),
            Backend::Lmdb(l) => l.insert_claim(claim, checkpoint_interval),
        }
    }

    fn insert_claims_batch(&mut self, claims: Vec<Claim>, checkpoint_interval: u64) -> usize {
        match self {
            Backend::InMemory(m) => m.insert_claims_batch(claims, checkpoint_interval),
            Backend::Lmdb(l) => l.insert_claims_batch(claims, checkpoint_interval),
        }
    }

    fn claim_exists(&self, claim_id: &str) -> bool {
        match self {
            Backend::InMemory(m) => m.claim_exists(claim_id),
            Backend::Lmdb(l) => l.claim_exists(claim_id),
        }
    }

    fn get_claim(&self, claim_id: &str) -> Option<Claim> {
        match self {
            Backend::InMemory(m) => m.get_claim(claim_id),
            Backend::Lmdb(l) => l.get_claim(claim_id),
        }
    }

    fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_content_id(content_id),
            Backend::Lmdb(l) => l.claims_by_content_id(content_id),
        }
    }

    fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.all_claims(offset, limit),
            Backend::Lmdb(l) => l.all_claims(offset, limit),
        }
    }

    fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_source_id(source_id),
            Backend::Lmdb(l) => l.claims_by_source_id(source_id),
        }
    }

    fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_predicate_id(predicate_id),
            Backend::Lmdb(l) => l.claims_by_predicate_id(predicate_id),
        }
    }

    fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_for(entity_id, predicate_type, source_type, min_confidence),
            Backend::Lmdb(l) => l.claims_for(entity_id, predicate_type, source_type, min_confidence),
        }
    }

    fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        match self {
            Backend::InMemory(m) => m.get_claim_provenance_chain(claim_id),
            Backend::Lmdb(l) => l.get_claim_provenance_chain(claim_id),
        }
    }

    fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        match self {
            Backend::InMemory(m) => m.bfs_claims(entity_id, max_depth),
            Backend::Lmdb(l) => l.bfs_claims(entity_id, max_depth),
        }
    }

    fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        match self {
            Backend::InMemory(m) => m.path_exists(entity_a, entity_b, max_depth),
            Backend::Lmdb(l) => l.path_exists(entity_a, entity_b, max_depth),
        }
    }

    fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        match self {
            Backend::InMemory(m) => m.get_adjacency_list(),
            Backend::Lmdb(l) => l.get_adjacency_list(),
        }
    }

    fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_in_range(min_ts, max_ts),
            Backend::Lmdb(l) => l.claims_in_range(min_ts, max_ts),
        }
    }

    fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.most_recent_claims(n),
            Backend::Lmdb(l) => l.most_recent_claims(n),
        }
    }

    fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        match self {
            Backend::InMemory(m) => m.search_entities(query, top_k),
            Backend::Lmdb(l) => l.search_entities(query, top_k),
        }
    }

    fn stats(&self) -> StoreStats {
        match self {
            Backend::InMemory(m) => m.stats(),
            Backend::Lmdb(l) => l.stats(),
        }
    }

    fn update_claim_status(&mut self, claim_id: &str, status: ClaimStatus) -> Result<bool, AttestError> {
        match self {
            Backend::InMemory(m) => m.update_claim_status(claim_id, status),
            Backend::Lmdb(l) => l.update_claim_status(claim_id, status),
        }
    }

    fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError> {
        match self {
            Backend::InMemory(m) => m.update_claim_status_batch(updates),
            Backend::Lmdb(l) => l.update_claim_status_batch(updates),
        }
    }

    fn set_include_retracted(&mut self, include: bool) {
        match self {
            Backend::InMemory(m) => m.set_include_retracted(include),
            Backend::Lmdb(l) => l.set_include_retracted(include),
        }
    }

    fn set_namespace_filter(&mut self, namespaces: Vec<String>) {
        match self {
            Backend::InMemory(m) => m.set_namespace_filter(namespaces),
            Backend::Lmdb(l) => l.set_namespace_filter(namespaces),
        }
    }

    fn get_namespace_filter(&self) -> &[String] {
        match self {
            Backend::InMemory(m) => m.get_namespace_filter(),
            Backend::Lmdb(l) => l.get_namespace_filter(),
        }
    }

    fn predicate_counts(&self) -> HashMap<String, u64> {
        match self {
            Backend::InMemory(m) => m.predicate_counts(),
            Backend::Lmdb(l) => l.predicate_counts(),
        }
    }

    fn backfill_pred_id_counts(&mut self) -> Result<usize, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(0),
            Backend::Lmdb(l) => l.backfill_pred_id_counts(),
        }
    }

    fn backfill_claim_summaries(&mut self) -> Result<u64, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(0),
            Backend::Lmdb(l) => l.backfill_claim_summaries(),
        }
    }

    fn backfill_analytics_counters(&mut self) -> Result<u64, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(0),
            Backend::Lmdb(l) => l.backfill_analytics_counters(),
        }
    }

    fn find_contradictions(
        &self,
        pred_a: &str,
        pred_b: &str,
    ) -> Vec<(String, String, u64, u64, f64, f64)> {
        match self {
            Backend::InMemory(m) => m.find_contradictions(pred_a, pred_b),
            Backend::Lmdb(l) => l.find_contradictions(pred_a, pred_b),
        }
    }

    fn gap_analysis(
        &self,
        entity_type: &str,
        expected_predicates: &[&str],
        limit: usize,
    ) -> Vec<(String, Vec<String>)> {
        match self {
            Backend::InMemory(m) => m.gap_analysis(entity_type, expected_predicates, limit),
            Backend::Lmdb(l) => l.gap_analysis(entity_type, expected_predicates, limit),
        }
    }

    fn entity_predicate_counts(&self, entity_id: &str) -> Vec<(String, u64)> {
        match self {
            Backend::InMemory(m) => m.entity_predicate_counts(entity_id),
            Backend::Lmdb(l) => l.entity_predicate_counts(entity_id),
        }
    }

    fn entity_source_counts(&self, entity_id: &str) -> Vec<(String, u64, f64)> {
        match self {
            Backend::InMemory(m) => m.entity_source_counts(entity_id),
            Backend::Lmdb(l) => l.entity_source_counts(entity_id),
        }
    }

    fn find_single_source_entities(&self, min_claims: u64) -> Vec<String> {
        match self {
            Backend::InMemory(m) => m.find_single_source_entities(min_claims),
            Backend::Lmdb(l) => l.find_single_source_entities(min_claims),
        }
    }

    fn set_bulk_load_mode(&mut self, enabled: bool) {
        match self {
            Backend::InMemory(_) => {} // no-op
            Backend::Lmdb(l) => l.set_bulk_load_mode(enabled),
        }
    }

    fn rebuild_counters(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(_) => Ok(()),
            Backend::Lmdb(l) => l.rebuild_counters(),
        }
    }

    fn merge_from(&mut self, source_path: &str) -> Result<usize, AttestError> {
        match self {
            Backend::InMemory(_) => Err(AttestError::Provenance(
                "merge_from is only supported on LMDB backends".to_string()
            )),
            Backend::Lmdb(l) => l.merge_from(source_path),
        }
    }

    fn source_id_counts(&self) -> HashMap<String, u64> {
        match self {
            Backend::InMemory(_) => HashMap::new(),
            Backend::Lmdb(l) => l.source_id_counts(),
        }
    }
}
