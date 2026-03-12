//! Storage backend abstraction.
//!
//! Provides two backends:
//! - [`MemoryBackend`]: In-memory store with optional file persistence (original implementation)
//! - [`RedbBackend`]: File-backed store using redb B+ trees
//!
//! Both implement [`StorageBackend`], and the [`Backend`] enum delegates
//! to whichever variant it holds.

pub mod memory;
pub mod migration;
pub mod redb;
pub mod tables;
pub mod traits;

pub use memory::MemoryBackend;
pub use self::redb::RedbBackend;
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
    /// File-backed store using redb B+ trees.
    File(Box<RedbBackend>),
}

/// Delegate every `StorageBackend` method to the inner backend.
impl StorageBackend for Backend {
    fn close(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(m) => m.close(),
            Backend::File(r) => r.close(),
        }
    }

    fn checkpoint(&mut self) -> Result<(), AttestError> {
        match self {
            Backend::InMemory(m) => m.checkpoint(),
            Backend::File(r) => r.checkpoint(),
        }
    }

    fn compact(&mut self) -> Result<bool, AttestError> {
        match self {
            Backend::InMemory(_) => Ok(false),
            Backend::File(r) => r.compact(),
        }
    }

    fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        match self {
            Backend::InMemory(m) => m.register_vocabulary(namespace, vocab),
            Backend::File(r) => r.register_vocabulary(namespace, vocab),
        }
    }

    fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        match self {
            Backend::InMemory(m) => m.register_predicate(predicate_id, constraints),
            Backend::File(r) => r.register_predicate(predicate_id, constraints),
        }
    }

    fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        match self {
            Backend::InMemory(m) => m.register_payload_schema(schema_id, schema),
            Backend::File(r) => r.register_payload_schema(schema_id, schema),
        }
    }

    fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        match self {
            Backend::InMemory(m) => m.get_registered_vocabularies(),
            Backend::File(r) => r.get_registered_vocabularies(),
        }
    }

    fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Backend::InMemory(m) => m.get_predicate_constraints(),
            Backend::File(r) => r.get_predicate_constraints(),
        }
    }

    fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        match self {
            Backend::InMemory(m) => m.get_payload_schemas(),
            Backend::File(r) => r.get_payload_schemas(),
        }
    }

    fn resolve(&mut self, entity_id: &str) -> String {
        match self {
            Backend::InMemory(m) => m.resolve(entity_id),
            Backend::File(r) => r.resolve(entity_id),
        }
    }

    fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        match self {
            Backend::InMemory(m) => m.get_alias_group(entity_id),
            Backend::File(r) => r.get_alias_group(entity_id),
        }
    }

    fn warm_caches(&self) {
        match self {
            Backend::InMemory(m) => m.warm_caches(),
            Backend::File(r) => r.warm_caches(),
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
            Backend::File(r) => r.upsert_entity(entity_id, entity_type, display_name, external_ids, timestamp),
        }
    }

    fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) {
        match self {
            Backend::InMemory(m) => {
                for (id, etype, display, ext_ids) in entities {
                    m.upsert_entity(id, etype, display, Some(ext_ids), timestamp);
                }
            }
            Backend::File(r) => r.upsert_entities_batch(entities, timestamp),
        }
    }

    fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        match self {
            Backend::InMemory(m) => m.get_entity(entity_id),
            Backend::File(r) => r.get_entity(entity_id),
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
            Backend::File(r) => r.list_entities(entity_type, min_claims, offset, limit),
        }
    }

    fn count_entities(&self, entity_type: Option<&str>, min_claims: usize) -> usize {
        match self {
            Backend::InMemory(m) => m.count_entities(entity_type, min_claims),
            Backend::File(r) => r.count_entities(entity_type, min_claims),
        }
    }

    fn count_claims(&self) -> usize {
        match self {
            Backend::InMemory(m) => m.count_claims(),
            Backend::File(r) => r.count_claims(),
        }
    }

    fn insert_claim(&mut self, claim: Claim, checkpoint_interval: u64) -> bool {
        match self {
            Backend::InMemory(m) => m.insert_claim(claim, checkpoint_interval),
            Backend::File(r) => r.insert_claim(claim, checkpoint_interval),
        }
    }

    fn insert_claims_batch(&mut self, claims: Vec<Claim>, checkpoint_interval: u64) -> usize {
        match self {
            Backend::InMemory(m) => m.insert_claims_batch(claims, checkpoint_interval),
            Backend::File(r) => r.insert_claims_batch(claims, checkpoint_interval),
        }
    }

    fn claim_exists(&self, claim_id: &str) -> bool {
        match self {
            Backend::InMemory(m) => m.claim_exists(claim_id),
            Backend::File(r) => r.claim_exists(claim_id),
        }
    }

    fn get_claim(&self, claim_id: &str) -> Option<Claim> {
        match self {
            Backend::InMemory(m) => m.get_claim(claim_id),
            Backend::File(r) => r.get_claim(claim_id),
        }
    }

    fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_content_id(content_id),
            Backend::File(r) => r.claims_by_content_id(content_id),
        }
    }

    fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.all_claims(offset, limit),
            Backend::File(r) => r.all_claims(offset, limit),
        }
    }

    fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_source_id(source_id),
            Backend::File(r) => r.claims_by_source_id(source_id),
        }
    }

    fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_by_predicate_id(predicate_id),
            Backend::File(r) => r.claims_by_predicate_id(predicate_id),
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
            Backend::File(r) => r.claims_for(entity_id, predicate_type, source_type, min_confidence),
        }
    }

    fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        match self {
            Backend::InMemory(m) => m.get_claim_provenance_chain(claim_id),
            Backend::File(r) => r.get_claim_provenance_chain(claim_id),
        }
    }

    fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        match self {
            Backend::InMemory(m) => m.bfs_claims(entity_id, max_depth),
            Backend::File(r) => r.bfs_claims(entity_id, max_depth),
        }
    }

    fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        match self {
            Backend::InMemory(m) => m.path_exists(entity_a, entity_b, max_depth),
            Backend::File(r) => r.path_exists(entity_a, entity_b, max_depth),
        }
    }

    fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        match self {
            Backend::InMemory(m) => m.get_adjacency_list(),
            Backend::File(r) => r.get_adjacency_list(),
        }
    }

    fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.claims_in_range(min_ts, max_ts),
            Backend::File(r) => r.claims_in_range(min_ts, max_ts),
        }
    }

    fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        match self {
            Backend::InMemory(m) => m.most_recent_claims(n),
            Backend::File(r) => r.most_recent_claims(n),
        }
    }

    fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        match self {
            Backend::InMemory(m) => m.search_entities(query, top_k),
            Backend::File(r) => r.search_entities(query, top_k),
        }
    }

    fn stats(&self) -> StoreStats {
        match self {
            Backend::InMemory(m) => m.stats(),
            Backend::File(r) => r.stats(),
        }
    }

    fn update_claim_status(&mut self, claim_id: &str, status: ClaimStatus) -> Result<bool, AttestError> {
        match self {
            Backend::InMemory(m) => m.update_claim_status(claim_id, status),
            Backend::File(r) => r.update_claim_status(claim_id, status),
        }
    }

    fn update_claim_status_batch(&mut self, updates: &[(String, ClaimStatus)]) -> Result<usize, AttestError> {
        match self {
            Backend::InMemory(m) => m.update_claim_status_batch(updates),
            Backend::File(r) => r.update_claim_status_batch(updates),
        }
    }

    fn set_include_retracted(&mut self, include: bool) {
        match self {
            Backend::InMemory(m) => m.set_include_retracted(include),
            Backend::File(r) => r.set_include_retracted(include),
        }
    }

    fn set_namespace_filter(&mut self, namespaces: Vec<String>) {
        match self {
            Backend::InMemory(m) => m.set_namespace_filter(namespaces),
            Backend::File(r) => r.set_namespace_filter(namespaces),
        }
    }

    fn get_namespace_filter(&self) -> &[String] {
        match self {
            Backend::InMemory(m) => m.get_namespace_filter(),
            Backend::File(r) => r.get_namespace_filter(),
        }
    }
}
