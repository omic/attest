//! In-memory storage backend with optional file persistence.
//!
//! This is the original RustStore implementation extracted into a backend.
//! Used for `:memory:` databases and file-backed stores (pending redb migration).

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
use crate::store::StoreStats;
use crate::union_find::UnionFind;
use crate::wal::{Wal, WalEntry};

/// Serializable snapshot of the entire store state (owned, for deserialization).
#[derive(Debug, Clone, serde::Deserialize)]
struct StoreState {
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
}

/// Borrowed view of store state for serialization (avoids cloning).
#[derive(Debug, serde::Serialize)]
struct StoreStateRef<'a> {
    claims: &'a ClaimLog,
    entities: &'a EntityStore,
    metadata: &'a MetadataStore,
    aliases: &'a UnionFind,
}

/// Default: auto-checkpoint after this many WAL entries.
pub(crate) const DEFAULT_CHECKPOINT_INTERVAL: u64 = 1000;

/// In-memory storage backend with optional file persistence via checkpoint + WAL.
pub struct MemoryBackend {
    db_path: PathBuf,
    claims: ClaimLog,
    entities: EntityStore,
    metadata: MetadataStore,
    aliases: UnionFind,
    lock_file: Option<std::fs::File>,
    wal: Option<Wal>,
}

impl MemoryBackend {
    /// Create a purely in-memory backend (no file path, no persistence).
    pub fn new_empty() -> Self {
        Self {
            db_path: PathBuf::new(),
            claims: ClaimLog::new(),
            entities: EntityStore::new(),
            metadata: MetadataStore::new(),
            aliases: UnionFind::new(),
            lock_file: None,
            wal: None,
        }
    }

    /// Open or create a file-backed memory backend.
    ///
    /// Acquires an exclusive file lock. If the database file exists but
    /// has a checksum mismatch (crash during write), the store starts
    /// empty with a warning — the corrupted file is not trusted.
    pub fn open(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);

        // Acquire exclusive lock
        let lock_file = file_format::acquire_lock(&path)
            .map_err(|e| AttestError::Provenance(format!("failed to lock database: {e}")))?;

        let mut backend = if path.exists() && file_format::is_attest_file(&path) {
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
                        lock_file: Some(lock_file),
                        wal: None,
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
                        lock_file: Some(lock_file),
                        wal: None,
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
                lock_file: Some(lock_file),
                wal: None,
            }
        };

        // Replay WAL entries (claims recorded after last checkpoint)
        match Wal::read_entries(&path) {
            Ok(entries) => {
                let count = entries.len();
                for entry in entries {
                    match entry {
                        WalEntry::InsertClaim(claim) => {
                            backend.insert_claim_no_wal(claim);
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
        backend.wal = Some(
            Wal::open(&path)
                .map_err(|e| AttestError::Provenance(format!("failed to open WAL: {e}")))?
        );

        Ok(backend)
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    /// Persist state and close. Releases the file lock.
    pub fn close(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self.lock_file.is_none() {
            return Ok(()); // in-memory or already closed
        }
        let state = StoreStateRef {
            claims: &self.claims,
            entities: &self.entities,
            metadata: &self.metadata,
            aliases: &self.aliases,
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
        if let Some(lock) = self.lock_file.take() {
            file_format::release_lock(&lock);
        }
        self.wal = None;

        write_result
            .map_err(|e| AttestError::Provenance(format!("failed to save database: {e}")))
    }

    /// Write a full checkpoint without releasing the lock.
    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        if self.db_path.as_os_str().is_empty() || self.lock_file.is_none() {
            return Ok(());
        }
        let state = StoreStateRef {
            claims: &self.claims,
            entities: &self.entities,
            metadata: &self.metadata,
            aliases: &self.aliases,
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

    /// Get raw entity data (used by migration to preserve created_at timestamps).
    pub(crate) fn get_entity_data(&self, entity_id: &str) -> Option<&crate::entity_store::EntityData> {
        self.entities.get(entity_id)
    }

    // ── Cache management ───────────────────────────────────────────────

    /// No-op in memory backend — everything is already in memory.
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
        offset: usize,
        limit: usize,
    ) -> Vec<EntitySummary> {
        let filtered = self.entities
            .list(entity_type)
            .into_iter()
            .filter_map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                if count >= min_claims {
                    Some(e.to_summary(count))
                } else {
                    None
                }
            });
        if limit == 0 {
            filtered.skip(offset).collect()
        } else {
            filtered.skip(offset).take(limit).collect()
        }
    }

    /// Count entities matching the given filter without materializing results.
    pub fn count_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> usize {
        if min_claims == 0 {
            match entity_type {
                Some(_) => self.entities.list(entity_type).len(),
                None => self.entities.len(),
            }
        } else {
            self.entities
                .list(entity_type)
                .into_iter()
                .filter(|e| self.claims.count_for_entity(&e.id) >= min_claims)
                .count()
        }
    }

    /// Count total claims.
    pub fn count_claims(&self) -> usize {
        self.claims.len()
    }

    // ── Claim operations ───────────────────────────────────────────────

    /// Insert a validated claim with WAL write.
    /// Returns `false` if the claim_id already exists.
    pub fn insert_claim(&mut self, claim: Claim, checkpoint_interval: u64) -> bool {
        if self.claims.contains(&claim.claim_id) {
            return false;
        }

        // Write to WAL first (durability before visibility)
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.append_claim(&claim) {
                log::error!("WAL write failed for claim {}: {e}", claim.claim_id);
                return false;
            }
        }

        self.apply_claim(claim);

        // Auto-checkpoint if interval reached
        if checkpoint_interval > 0 {
            if let Some(ref wal) = self.wal {
                if wal.entry_count() >= checkpoint_interval {
                    if let Err(e) = self.checkpoint() {
                        log::warn!("Auto-checkpoint failed: {e}");
                    }
                }
            }
        }

        true
    }

    /// Batch-insert claims with a single WAL sync.
    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>, checkpoint_interval: u64) -> usize {
        let mut inserted = 0usize;
        for claim in claims {
            if self.claims.contains(&claim.claim_id) {
                continue;
            }

            // Write to WAL without fsync
            if let Some(ref mut wal) = self.wal {
                if let Err(e) = wal.append_claim_no_sync(&claim) {
                    log::error!("WAL write failed for claim {}: {e}", claim.claim_id);
                    continue; // Skip claim — not durable
                }
            }

            self.apply_claim(claim);
            inserted += 1;
        }

        // Single sync + checkpoint at end of batch
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.sync() {
                log::error!("WAL sync failed after batch insert: {e}");
            }
        }
        if checkpoint_interval > 0 {
            if let Some(ref wal) = self.wal {
                if wal.entry_count() >= checkpoint_interval {
                    if let Err(e) = self.checkpoint() {
                        log::warn!("Auto-checkpoint failed after batch: {e}");
                    }
                }
            }
        }

        inserted
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

    pub fn all_claims(&self, offset: usize, limit: usize) -> Vec<Claim> {
        let all = self.claims.all_claims();
        if limit == 0 {
            all.iter().skip(offset).cloned().collect()
        } else {
            all.iter().skip(offset).take(limit).cloned().collect()
        }
    }

    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        self.claims
            .by_source_id(source_id)
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        self.claims
            .by_predicate_id(predicate_id)
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

    /// Get the bidirectional adjacency list.
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

    /// Search entities by text query.
    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        self.entities
            .search(query, top_k)
            .into_iter()
            .map(|e| {
                let count = self.claims.count_for_entity(&e.id);
                e.to_summary(count)
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

impl Drop for MemoryBackend {
    fn drop(&mut self) {
        if self.lock_file.is_some() {
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
            if let Some(lock) = self.lock_file.take() {
                file_format::release_lock(&lock);
            }
        }
    }
}
