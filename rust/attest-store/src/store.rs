//! RustStore — drop-in replacement for KuzuStore.
//!
//! In-memory store backed by append-only claim log with persistence
//! to a single file. Implements the full 22-method KuzuStore interface.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::normalization::normalize_entity_id;
use attest_core::types::{Claim, EntitySummary};
use attest_core::vocabulary::ALIAS_PREDICATES;

use crate::claim_log::ClaimLog;
use crate::entity_store::EntityStore;
use crate::file_format;
use crate::metadata::{MetadataStore, Vocabulary};
use crate::union_find::UnionFind;
use crate::wal::{Wal, WalEntry};

/// Serializable snapshot of the entire store state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct StoreState {
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
}

/// Store statistics.
#[derive(Debug, Clone)]
pub struct StoreStats {
    pub total_claims: usize,
    pub entity_count: usize,
    pub entity_types: HashMap<String, usize>,
    pub predicate_types: HashMap<String, usize>,
    pub source_types: HashMap<String, usize>,
}

/// In-memory claim-native store with file persistence.
///
/// Concurrency model: single-writer via advisory file lock.
/// The lock is acquired on `new()` and released on `close()`.
/// Attempting to open the same database from another process will fail
/// with `AttestError::Provenance("database is locked ...")`.
/// Default: auto-checkpoint after this many WAL entries.
const DEFAULT_CHECKPOINT_INTERVAL: u64 = 1000;

pub struct RustStore {
    db_path: PathBuf,
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
    /// Held for the lifetime of the store to enforce single-writer.
    _lock_file: Option<std::fs::File>,
    /// Write-ahead log for crash durability (None for in-memory stores).
    wal: Option<Wal>,
    /// Auto-checkpoint after this many WAL entries. 0 = disabled.
    checkpoint_interval: u64,
}

impl RustStore {
    /// Open or create a store at the given path.
    ///
    /// Acquires an exclusive file lock. If the database file exists but
    /// has a checksum mismatch (crash during write), the store starts
    /// empty with a warning — the corrupted file is not trusted.
    pub fn new(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);

        // Acquire exclusive lock
        let lock_file = file_format::acquire_lock(&path)
            .map_err(|e| AttestError::Provenance(format!("failed to lock database: {e}")))?;

        let mut store = if path.exists() && file_format::is_attest_file(&path) {
            match file_format::read_store::<StoreState>(&path) {
                Ok(state) => {
                    let mut claims = state.claims;
                    let mut entities = state.entities;
                    let mut aliases = state.aliases;
                    // Rebuild indexes that may be empty from older file formats
                    claims.rebuild_derived_indexes();
                    entities.rebuild_text_index();
                    aliases.rebuild_groups();
                    Self {
                        db_path: path.clone(),
                        claims,
                        entities,
                        metadata: state.metadata,
                        aliases,
                        _lock_file: Some(lock_file),
                        wal: None,
                        checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
                    }
                }
                Err(file_format::FileError::ChecksumMismatch { expected, actual }) => {
                    // Crash recovery: checkpoint is corrupted, WAL replay below
                    // may still recover data
                    log::warn!(
                        "Database file corrupted (checksum {actual:#010x} != {expected:#010x}). \
                         Starting with empty store, will attempt WAL replay."
                    );
                    Self {
                        db_path: path.clone(),
                        claims: ClaimLog::new(),
                        entities: EntityStore::new(),
                        metadata: MetadataStore::new(),
                        aliases: UnionFind::new(),
                        _lock_file: Some(lock_file),
                        wal: None,
                        checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
                    }
                }
                Err(e) => {
                    file_format::release_lock(&lock_file);
                    return Err(AttestError::Provenance(
                        format!("failed to open database: {e}"),
                    ));
                }
            }
        } else {
            Self {
                db_path: path.clone(),
                claims: ClaimLog::new(),
                entities: EntityStore::new(),
                metadata: MetadataStore::new(),
                aliases: UnionFind::new(),
                _lock_file: Some(lock_file),
                wal: None,
                checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            }
        };

        // Replay WAL entries (claims recorded after last checkpoint)
        match Wal::read_entries(&path) {
            Ok(entries) => {
                let count = entries.len();
                for entry in entries {
                    match entry {
                        WalEntry::InsertClaim(claim) => {
                            store.insert_claim_no_wal(claim);
                        }
                    }
                }
                if count > 0 {
                    log::info!("Replayed {count} WAL entries");
                }
            }
            Err(e) => {
                log::warn!("Failed to read WAL: {e}");
            }
        }

        // Open WAL for future writes
        store.wal = Some(
            Wal::open(&path)
                .map_err(|e| AttestError::Provenance(format!("failed to open WAL: {e}")))?
        );

        Ok(store)
    }

    /// Create a purely in-memory store (no file path, no persistence, no lock, no WAL).
    pub fn in_memory() -> Self {
        Self {
            db_path: PathBuf::new(),
            claims: ClaimLog::new(),
            entities: EntityStore::new(),
            metadata: MetadataStore::new(),
            aliases: UnionFind::new(),
            _lock_file: None,
            wal: None,
            checkpoint_interval: 0,
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    ///
    /// Writes a full checkpoint (`.attest` file), truncates the WAL,
    /// then releases the lock.
    pub fn close(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self._lock_file.is_none() {
            return Ok(()); // in-memory or already closed
        }
        let state = StoreState {
            claims: self.claims.clone(),
            entities: self.entities.clone(),
            metadata: self.metadata.clone(),
            aliases: self.aliases.clone(),
        };
        let write_result = file_format::write_store(&self.db_path, &state);

        // Truncate WAL after successful checkpoint
        if write_result.is_ok() {
            if let Some(ref mut wal) = self.wal {
                if let Err(e) = wal.truncate() {
                    log::warn!("Failed to truncate WAL after checkpoint: {e}");
                }
            }
        }

        // Always release the lock, even if write failed — holding it wedges the DB
        if let Some(lock) = self._lock_file.take() {
            file_format::release_lock(&lock);
        }
        self.wal = None;

        write_result
            .map_err(|e| AttestError::Provenance(format!("failed to save database: {e}")))
    }

    /// Write a full checkpoint without releasing the lock.
    ///
    /// Flushes all in-memory state to the `.attest` file and truncates
    /// the WAL. The store remains open for further operations.
    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self._lock_file.is_none() {
            return Ok(());
        }
        let state = StoreState {
            claims: self.claims.clone(),
            entities: self.entities.clone(),
            metadata: self.metadata.clone(),
            aliases: self.aliases.clone(),
        };
        file_format::write_store(&self.db_path, &state)
            .map_err(|e| AttestError::Provenance(format!("checkpoint failed: {e}")))?;

        if let Some(ref mut wal) = self.wal {
            wal.truncate()
                .map_err(|e| AttestError::Provenance(format!("WAL truncate failed: {e}")))?;
        }
        Ok(())
    }

    // ── Metadata ───────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        self.metadata.register_vocabulary(namespace, vocab);
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        self.metadata.register_predicate(predicate_id, constraints);
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        self.metadata.register_payload_schema(schema_id, schema);
    }

    pub fn get_registered_vocabularies(&self) -> &HashMap<String, Vocabulary> {
        self.metadata.vocabularies()
    }

    pub fn get_predicate_constraints(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.predicate_constraints()
    }

    pub fn get_payload_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.metadata.payload_schemas()
    }

    // ── Alias resolution ───────────────────────────────────────────────

    /// Resolve entity ID through alias chain.
    pub fn resolve(&mut self, entity_id: &str) -> String {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.find(&canonical)
    }

    /// Get all entity IDs that resolve to the same canonical.
    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.get_group(&canonical)
    }

    // ── Cache management ───────────────────────────────────────────────

    /// No-op in Rust — everything is already in memory.
    pub fn warm_caches(&self) {}

    // ── Entity CRUD ────────────────────────────────────────────────────

    pub fn upsert_entity(
        &mut self,
        entity_id: &str,
        entity_type: &str,
        display_name: &str,
        external_ids: Option<&HashMap<String, String>>,
        timestamp: i64,
    ) {
        self.entities
            .upsert(entity_id, entity_type, display_name, external_ids, timestamp);
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        let claim_count = self.claims.count_for_entity(entity_id);
        self.entities.get_summary(entity_id, claim_count)
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> Vec<EntitySummary> {
        self.entities
            .list(entity_type)
            .into_iter()
            .filter_map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                if count >= min_claims {
                    Some(EntitySummary {
                        id: e.id.clone(),
                        name: e.display_name.clone(),
                        entity_type: e.entity_type.clone(),
                        external_ids: e.external_ids.clone(),
                        claim_count: count,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim. Caller must ensure entities exist.
    ///
    /// The claim is appended to the WAL and fsynced BEFORE being applied
    /// to in-memory state. Returns `false` if the claim_id already exists.
    pub fn insert_claim(&mut self, claim: Claim) -> bool {
        if self.claims.contains(&claim.claim_id) {
            return false;
        }

        // Write to WAL first (durability before visibility)
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.append_claim(&claim) {
                log::error!("WAL write failed for claim {}: {e}", claim.claim_id);
                // WAL failure = durability failure. Still apply in-memory
                // so the current session isn't broken, but log loudly.
            }
        }

        self.apply_claim(claim);

        // Auto-checkpoint if interval reached
        if self.checkpoint_interval > 0 {
            if let Some(ref wal) = self.wal {
                if wal.entry_count() >= self.checkpoint_interval {
                    if let Err(e) = self.checkpoint() {
                        log::warn!("Auto-checkpoint failed: {e}");
                    }
                }
            }
        }

        true
    }

    /// Insert a claim without WAL write (used during WAL replay).
    fn insert_claim_no_wal(&mut self, claim: Claim) -> bool {
        if self.claims.contains(&claim.claim_id) {
            return false;
        }
        self.apply_claim(claim);
        true
    }

    /// Apply a claim to in-memory state (indexes + aliases).
    fn apply_claim(&mut self, claim: Claim) {
        let pred_id = claim.predicate.id.clone();
        let subj_id = claim.subject.id.clone();
        let obj_id = claim.object.id.clone();

        self.claims.append(claim);

        if ALIAS_PREDICATES.contains(pred_id.as_str()) {
            if pred_id == "same_as" {
                self.aliases.union(&subj_id, &obj_id);
            } else if pred_id == "not_same_as" {
                self.aliases.split(&subj_id);
            }
        }
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        self.claims.contains(claim_id)
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        self.claims
            .by_content_id(content_id)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn claims_for(
        &mut self,
        entity_id: &str,
        predicate_type: Option<&str>,
        source_type: Option<&str>,
        min_confidence: f64,
    ) -> Vec<Claim> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);
        self.claims
            .for_entity_filtered(&aliases, predicate_type, source_type, min_confidence)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get the provenance chain for a claim.
    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        self.claims
            .get(claim_id)
            .map(|c| c.provenance.chain.clone())
            .unwrap_or_default()
    }

    // ── Graph traversal ────────────────────────────────────────────────

    /// BFS traversal collecting claims at each hop depth.
    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);
        let mut visited: HashSet<String> = aliases.clone();
        let mut frontier: HashSet<String> = aliases;
        let mut results: Vec<(Claim, usize)> = Vec::new();

        for hop in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }

            let claims = self.claims.for_entities(&frontier);
            let mut next_frontier = HashSet::new();

            for claim in claims {
                results.push((claim.clone(), hop));
                for eid in [&claim.subject.id, &claim.object.id] {
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
            }

            frontier = next_frontier;
        }

        results
    }

    /// Check if a path exists between two entities within max_depth hops.
    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        let ra = self.resolve(entity_a);
        let rb = self.resolve(entity_b);

        if ra == rb {
            return true;
        }

        // BFS from ra
        let mut visited = HashSet::new();
        visited.insert(ra.clone());
        let mut frontier = HashSet::new();
        frontier.insert(ra);

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let claims = self.claims.for_entities(&frontier);
            let mut next_frontier = HashSet::new();

            for claim in claims {
                for eid in [&claim.subject.id, &claim.object.id] {
                    if *eid == rb {
                        return true;
                    }
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
            }

            frontier = next_frontier;
        }

        false
    }

    /// Get the bidirectional adjacency list. O(1) — returns a clone of the
    /// maintained index (updated on every insert_claim).
    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        self.claims.adjacency_list().clone()
    }

    /// Get claims within a timestamp range [min_ts, max_ts] (inclusive).
    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        self.claims
            .in_time_range(min_ts, max_ts)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get the most recent N claims by timestamp.
    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        self.claims.most_recent(n).into_iter().cloned().collect()
    }

    /// Search entities by text query. Returns entity summaries matching the query,
    /// ranked by relevance (number of matching tokens).
    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        self.entities
            .search(query, top_k)
            .into_iter()
            .map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                EntitySummary {
                    id: e.id.clone(),
                    name: e.display_name.clone(),
                    entity_type: e.entity_type.clone(),
                    external_ids: e.external_ids.clone(),
                    claim_count: count,
                }
            })
            .collect()
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            total_claims: self.claims.len(),
            entity_count: self.entities.len(),
            entity_types: self.entities.count_by_type(),
            predicate_types: self.claims.count_by_predicate_type(),
            source_types: self.claims.count_by_source_type(),
        }
    }
}

impl Drop for RustStore {
    fn drop(&mut self) {
        if self._lock_file.is_some() {
            // close() was never called — but WAL has durable entries.
            // Best-effort: try to checkpoint before releasing lock.
            log::warn!(
                "RustStore dropped without calling close() — \
                 attempting best-effort checkpoint."
            );
            if let Err(e) = self.checkpoint() {
                log::error!(
                    "Best-effort checkpoint in Drop failed: {e}. \
                     WAL entries are still durable and will replay on next open."
                );
            }
            self.wal = None;
            if let Some(lock) = self._lock_file.take() {
                file_format::release_lock(&lock);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use attest_core::types::*;

    fn make_claim(
        claim_id: &str,
        subj: &str,
        pred: &str,
        obj: &str,
        source_type: &str,
    ) -> Claim {
        Claim {
            claim_id: claim_id.to_string(),
            content_id: attest_core::compute_content_id(subj, pred, obj),
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
                source_type: source_type.to_string(),
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

    fn setup_store() -> RustStore {
        let mut store = RustStore::in_memory();

        store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
        store.upsert_entity("tp53", "gene", "TP53", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "experimental"));
        store.insert_claim(make_claim("c2", "tp53", "inhibits", "aspirin", "literature"));

        store
    }

    #[test]
    fn test_entity_upsert_and_get() {
        let store = setup_store();
        let entity = store.get_entity("brca1").unwrap();
        assert_eq!(entity.name, "BRCA1");
        assert_eq!(entity.entity_type, "gene");
        assert_eq!(entity.claim_count, 1); // brca1 is in c1
    }

    #[test]
    fn test_claim_exists() {
        let store = setup_store();
        assert!(store.claim_exists("c1"));
        assert!(!store.claim_exists("c99"));
    }

    #[test]
    fn test_claims_by_content_id() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let c1 = make_claim("c1", "a", "rel", "b", "obs");
        let content_id = c1.content_id.clone();
        store.insert_claim(c1);

        let mut c2 = make_claim("c2", "a", "rel", "b", "exp");
        c2.content_id.clone_from(&content_id);
        store.insert_claim(c2);

        assert_eq!(store.claims_by_content_id(&content_id).len(), 2);
    }

    #[test]
    fn test_claims_for() {
        let mut store = setup_store();
        let claims = store.claims_for("brca1", None, None, 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");

        // tp53 is in both claims
        let claims = store.claims_for("tp53", None, None, 0.0);
        assert_eq!(claims.len(), 2);
    }

    #[test]
    fn test_claims_for_with_filters() {
        let mut store = setup_store();
        // Filter by source_type
        let claims = store.claims_for("tp53", None, Some("experimental"), 0.0);
        assert_eq!(claims.len(), 1);
        assert_eq!(claims[0].claim_id, "c1");
    }

    #[test]
    fn test_list_entities() {
        let store = setup_store();
        assert_eq!(store.list_entities(Some("gene"), 0).len(), 2);
        assert_eq!(store.list_entities(Some("compound"), 0).len(), 1);
        assert_eq!(store.list_entities(None, 0).len(), 3);
    }

    #[test]
    fn test_list_entities_min_claims() {
        let store = setup_store();
        // tp53 has 2 claims, brca1 and aspirin have 1 each
        let high = store.list_entities(None, 2);
        assert_eq!(high.len(), 1);
        assert_eq!(high[0].id, "tp53");
    }

    #[test]
    fn test_alias_resolution() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("gene_a", "gene", "A", None, 0);
        store.upsert_entity("gene_a_alias", "gene", "A alias", None, 0);
        store.upsert_entity("gene_b", "gene", "B", None, 0);

        // Create same_as claim
        let alias_claim = Claim {
            claim_id: "alias1".to_string(),
            content_id: "alias_content".to_string(),
            subject: EntityRef {
                id: "gene_a".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A".to_string(),
                external_ids: Default::default(),
            },
            predicate: PredicateRef {
                id: "same_as".to_string(),
                predicate_type: "same_as".to_string(),
            },
            object: EntityRef {
                id: "gene_a_alias".to_string(),
                entity_type: "gene".to_string(),
                display_name: "A alias".to_string(),
                external_ids: Default::default(),
            },
            confidence: 1.0,
            provenance: Provenance {
                source_type: "human_annotation".to_string(),
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
        };
        store.insert_claim(alias_claim);

        // Both should resolve to the same root
        let r1 = store.resolve("gene_a");
        let r2 = store.resolve("gene_a_alias");
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_bfs_claims() {
        let mut store = setup_store();
        // From brca1, depth 1: should find c1 (brca1->tp53)
        let results = store.bfs_claims("brca1", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.claim_id, "c1");
        assert_eq!(results[0].1, 1); // hop 1

        // From brca1, depth 2: hop 1 finds c1, hop 2 re-finds c1 (via tp53) + c2
        // No cross-hop dedup — matches KuzuStore behavior
        let results = store.bfs_claims("brca1", 2);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_path_exists() {
        let mut store = setup_store();
        assert!(store.path_exists("brca1", "tp53", 1));
        assert!(store.path_exists("brca1", "aspirin", 2));
        assert!(!store.path_exists("brca1", "aspirin", 1)); // too far in 1 hop
    }

    #[test]
    fn test_adjacency_list() {
        let store = setup_store();
        let adj = store.get_adjacency_list();
        assert!(adj["brca1"].contains("tp53"));
        assert!(adj["tp53"].contains("brca1")); // bidirectional
        assert!(adj["tp53"].contains("aspirin"));
    }

    #[test]
    fn test_stats() {
        let store = setup_store();
        let stats = store.stats();
        assert_eq!(stats.total_claims, 2);
        assert_eq!(stats.entity_count, 3);
        assert_eq!(stats.entity_types["gene"], 2);
        assert_eq!(stats.entity_types["compound"], 1);
    }

    #[test]
    fn test_get_claim_provenance_chain() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut claim = make_claim("c1", "a", "rel", "b", "obs");
        claim.provenance.chain = vec!["upstream1".to_string(), "upstream2".to_string()];
        store.insert_claim(claim);

        let chain = store.get_claim_provenance_chain("c1");
        assert_eq!(chain, vec!["upstream1", "upstream2"]);

        // Non-existent claim returns empty
        assert!(store.get_claim_provenance_chain("nonexistent").is_empty());
    }

    #[test]
    fn test_warm_caches_noop() {
        let store = setup_store();
        store.warm_caches(); // Should not panic
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = std::env::temp_dir().join("attest_store_test.attest");
        let _ = std::fs::remove_file(&dir);

        // Create and populate
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            store.upsert_entity("tp53", "gene", "TP53", None, 0);
            store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "exp"));
            store.close().unwrap();
        }

        // Reopen and verify
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert!(store.claim_exists("c1"));
            assert_eq!(store.get_entity("brca1").unwrap().name, "BRCA1");
            assert_eq!(store.stats().total_claims, 1);
        }

        std::fs::remove_file(&dir).unwrap();
    }

    #[test]
    fn test_file_locking() {
        let dir = std::env::temp_dir().join("attest_lock_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        // First open succeeds
        let mut store1 = RustStore::new(dir.to_str().unwrap()).unwrap();

        // Second open should fail with lock error
        let result = RustStore::new(dir.to_str().unwrap());
        assert!(result.is_err());
        match result {
            Err(e) => {
                let err_msg = format!("{e}");
                assert!(err_msg.contains("lock"), "Error should mention lock: {err_msg}");
            }
            Ok(_) => panic!("Expected lock error"),
        }

        // After close, lock is released
        store1.close().unwrap();
        let mut store2 = RustStore::new(dir.to_str().unwrap()).unwrap();
        store2.close().unwrap();

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
    }

    #[test]
    fn test_crash_recovery_corrupted_file() {
        let dir = std::env::temp_dir().join("attest_crash_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        // Write a valid store
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "a", "obs"));
            store.close().unwrap();
        }

        // Corrupt the file by flipping a byte in the data section
        {
            let mut data = std::fs::read(&dir).unwrap();
            // Flip a byte in the data section (after 32-byte header)
            if data.len() > 40 {
                data[40] ^= 0xFF;
            }
            std::fs::write(&dir, &data).unwrap();
        }

        // Opening should succeed (crash recovery) but start empty
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 0, "Corrupted file should start empty");
            assert_eq!(store.stats().entity_count, 0);
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
    }

    #[test]
    fn test_atomic_write_leaves_no_tmp() {
        let dir = std::env::temp_dir().join("attest_atomic_test.attest");
        let tmp = dir.with_extension("attest.tmp");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(&tmp);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));

        let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.close().unwrap();

        // Temp file should not exist after successful close
        assert!(!tmp.exists(), "Temp file should be cleaned up after close");
        // Main file should exist
        assert!(dir.exists(), "Main file should exist after close");

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
    }

    #[test]
    fn test_claims_in_range() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        let mut c1 = make_claim("c1", "a", "rel", "b", "obs");
        c1.timestamp = 1000;
        let mut c2 = make_claim("c2", "a", "rel", "b", "obs");
        c2.claim_id = "c2".to_string();
        c2.timestamp = 2000;
        let mut c3 = make_claim("c3", "a", "rel", "b", "obs");
        c3.claim_id = "c3".to_string();
        c3.timestamp = 3000;

        store.insert_claim(c1);
        store.insert_claim(c2);
        store.insert_claim(c3);

        assert_eq!(store.claims_in_range(1500, 2500).len(), 1);
        assert_eq!(store.claims_in_range(1000, 3000).len(), 3);
        assert_eq!(store.claims_in_range(4000, 5000).len(), 0);
    }

    #[test]
    fn test_most_recent_claims() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);

        for i in 0..5 {
            let mut c = make_claim(&format!("c{i}"), "a", "rel", "b", "obs");
            c.timestamp = (i + 1) * 1000;
            store.insert_claim(c);
        }

        let recent = store.most_recent_claims(2);
        assert_eq!(recent.len(), 2);
        assert!(recent[0].timestamp >= recent[1].timestamp);
    }

    #[test]
    fn test_search_entities() {
        let mut store = RustStore::in_memory();
        store.upsert_entity("brca1", "gene", "BRCA1 DNA repair", None, 0);
        store.upsert_entity("tp53", "gene", "TP53 tumor protein", None, 0);
        store.upsert_entity("aspirin", "compound", "Aspirin", None, 0);

        // Search for "BRCA" — should find brca1
        let results = store.search_entities("brca1", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        // Search for "protein" — should find tp53
        let results = store.search_entities("protein", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "tp53");

        // Search for "gene" — should find entities with "gene" in ID
        let results = store.search_entities("gene", 10);
        // "gene" appears as token in "brca1" → no, "gene" only in type not display_name
        // Actually EntityStore tokenizes entity_id and display_name
        // "brca1" tokens: ["brca1"], "BRCA1 DNA repair" tokens: ["brca1", "dna", "repair"]
        // "gene" doesn't appear in any of those, so 0 results
        assert_eq!(results.len(), 0);

        // Search for "repair" — should find brca1
        let results = store.search_entities("repair", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "brca1");

        // Search for "aspirin"
        let results = store.search_entities("aspirin", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "aspirin");
    }

    #[test]
    fn test_adjacency_maintained() {
        // Verify adjacency is maintained on insert, not rebuilt
        let mut store = RustStore::in_memory();
        store.upsert_entity("a", "entity", "A", None, 0);
        store.upsert_entity("b", "entity", "B", None, 0);
        store.upsert_entity("c", "entity", "C", None, 0);

        store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));
        let adj = store.get_adjacency_list();
        assert!(adj["a"].contains("b"));
        assert!(adj["b"].contains("a"));
        assert_eq!(adj.len(), 2); // only a and b

        store.insert_claim(make_claim("c2", "b", "rel", "c", "obs"));
        let adj = store.get_adjacency_list();
        assert!(adj["b"].contains("c"));
        assert!(adj["c"].contains("b"));
        assert_eq!(adj.len(), 3); // a, b, c
    }

    #[test]
    fn test_metadata_registration() {
        let mut store = RustStore::in_memory();

        store.register_vocabulary(
            "bio",
            Vocabulary {
                entity_types: vec!["gene".to_string(), "protein".to_string()],
                predicate_types: vec!["binds_to".to_string()],
                source_types: vec!["experimental".to_string()],
            },
        );

        let vocabs = store.get_registered_vocabularies();
        assert!(vocabs.contains_key("bio"));
        assert!(vocabs["bio"].entity_types.contains(&"gene".to_string()));

        store.register_predicate(
            "binds_to",
            serde_json::json!({
                "subject_types": ["protein"],
                "object_types": ["protein", "compound"]
            }),
        );
        assert!(store.get_predicate_constraints().contains_key("binds_to"));
    }

    #[test]
    fn test_wal_crash_recovery() {
        // Simulate: write claims → crash (no close) → reopen → data recovered
        let dir = std::env::temp_dir().join("attest_wal_recovery_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

        // Phase 1: write claims, then DROP without close (simulates crash)
        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("brca1", "gene", "BRCA1", None, 0);
            store.upsert_entity("tp53", "gene", "TP53", None, 0);
            store.insert_claim(make_claim("c1", "brca1", "binds_to", "tp53", "exp"));
            store.insert_claim(make_claim("c2", "tp53", "inhibits", "brca1", "lit"));
            // Intentionally no close() — Drop will attempt best-effort checkpoint
            // but even if that fails, WAL entries are durable
        }

        // Phase 2: reopen — should recover via WAL replay
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 2, "Should recover 2 claims from WAL/checkpoint");
            assert!(store.claim_exists("c1"));
            assert!(store.claim_exists("c2"));
            let mut store = store;
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }

    #[test]
    fn test_checkpoint_without_close() {
        let dir = std::env::temp_dir().join("attest_checkpoint_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);
            store.insert_claim(make_claim("c1", "a", "rel", "b", "obs"));

            // Checkpoint should persist without closing
            store.checkpoint().unwrap();

            // Insert more after checkpoint
            store.insert_claim(make_claim("c2", "b", "rel", "a", "obs"));
            store.close().unwrap();
        }

        // Reopen — should have both claims
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 2);
            let mut store = store;
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }

    #[test]
    fn test_auto_checkpoint() {
        let dir = std::env::temp_dir().join("attest_auto_cp_test.attest");
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);

        {
            let mut store = RustStore::new(dir.to_str().unwrap()).unwrap();
            // Set very low interval for testing
            store.checkpoint_interval = 3;

            store.upsert_entity("a", "entity", "A", None, 0);
            store.upsert_entity("b", "entity", "B", None, 0);

            // Insert 5 claims — should trigger auto-checkpoint at claim 3
            for i in 0..5 {
                let mut c = make_claim(&format!("c{i}"), "a", "rel", "b", "obs");
                c.timestamp = (i + 1) * 1000;
                store.insert_claim(c);
            }

            // The .attest file should exist (auto-checkpoint wrote it)
            assert!(dir.exists(), "Auto-checkpoint should have written .attest file");

            store.close().unwrap();
        }

        // Verify all 5 claims
        {
            let store = RustStore::new(dir.to_str().unwrap()).unwrap();
            assert_eq!(store.stats().total_claims, 5);
            let mut store = store;
            store.close().unwrap();
        }

        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("attest.lock"));
        crate::wal::Wal::remove(&dir);
    }
}
