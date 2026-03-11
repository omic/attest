//! File-backed storage backend using redb B+ trees.
//!
//! Opens a redb database at the given path. All claim/entity data lives on disk,
//! paged into memory by the OS on demand. Union-find and metadata stay in-memory
//! (tiny) and are flushed to a META_BLOBS table on close/checkpoint.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::path::PathBuf;

use attest_core::errors::AttestError;
use attest_core::normalization::normalize_entity_id;
use attest_core::types::{Claim, EntitySummary};
use attest_core::vocabulary::ALIAS_PREDICATES;
use redb::{Database, ReadableMultimapTable, ReadableTable, ReadableTableMetadata, TableDefinition};

use crate::entity_store::{tokenize, EntityData};
use crate::metadata::{MetadataStore, Vocabulary};
use crate::store::StoreStats;
use crate::union_find::UnionFind;

use super::tables::*;

/// META_BLOBS keys for in-memory state.
const META_KEY_ALIASES: &str = "aliases";
const META_KEY_METADATA: &str = "metadata";

/// Convert any error into an AttestError.
fn redb_err<E: Display>(msg: &str, e: E) -> AttestError {
    AttestError::Provenance(format!("{msg}: {e}"))
}

/// File-backed storage backend using redb B+ trees.
pub struct RedbBackend {
    db: Option<Database>,
    #[allow(dead_code)]
    db_path: PathBuf,
    next_seq: u64,
    aliases: UnionFind,
    metadata: MetadataStore,
    aliases_dirty: bool,
    metadata_dirty: bool,
}

impl RedbBackend {
    /// Open or create a redb-backed store at the given path.
    pub fn open(db_path: &str) -> Result<Self, AttestError> {
        let path = PathBuf::from(db_path);

        let db = match Database::create(&path) {
            Ok(db) => db,
            Err(e) => {
                let msg = e.to_string();
                // redb's lock error → translate so tests can match on "lock"
                if msg.contains("locked") || msg.contains("DatabaseAlreadyOpen") {
                    return Err(AttestError::Provenance(
                        format!("failed to lock database: {msg}"),
                    ));
                }
                // Corruption → delete and recreate
                if msg.contains("corrupt") || msg.contains("magic") || msg.contains("invalid") {
                    log::warn!("Database file corrupted ({msg}). Recreating.");
                    let _ = std::fs::remove_file(&path);
                    Database::create(&path)
                        .map_err(|e2| redb_err("failed to recreate database", e2))?
                } else {
                    return Err(redb_err("failed to open database", e));
                }
            }
        };

        // Ensure all tables exist
        {
            let txn = db.begin_write().map_err(|e| redb_err("begin_write", e))?;
            txn.open_table(CLAIMS).map_err(|e| redb_err("create table", e))?;
            txn.open_table(CLAIM_ID_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(CONTENT_ID_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(ENTITY_CLAIMS_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(ADJACENCY_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(TIMESTAMP_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(SOURCE_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(PREDICATE_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_table(ENTITIES).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(ENTITY_TYPE_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_multimap_table(TEXT_IDX).map_err(|e| redb_err("create table", e))?;
            txn.open_table(META_BLOBS).map_err(|e| redb_err("create table", e))?;
            txn.open_table(PRED_TYPE_COUNTS).map_err(|e| redb_err("create table", e))?;
            txn.open_table(SRC_TYPE_COUNTS).map_err(|e| redb_err("create table", e))?;
            txn.open_table(ENTITY_TYPE_COUNTS).map_err(|e| redb_err("create table", e))?;
            txn.commit().map_err(|e| redb_err("commit", e))?;
        }

        // Load next_seq, aliases, metadata in a single read transaction
        let (next_seq, aliases, metadata) = {
            let txn = db.begin_read().map_err(|e| redb_err("begin_read", e))?;

            let next_seq = {
                let table = txn.open_table(CLAIMS).map_err(|e| redb_err("open table", e))?;
                let last = table.last().map_err(|e| redb_err("last", e))?;
                match last {
                    Some((k, _)) => k.value() + 1,
                    None => 0,
                }
            };

            let (aliases, metadata) = {
                let table = txn.open_table(META_BLOBS).map_err(|e| redb_err("open table", e))?;

                let aliases = match table.get(META_KEY_ALIASES).map_err(|e| redb_err("get", e))? {
                    Some(data) => {
                        bincode::deserialize(data.value())
                            .unwrap_or_else(|e| {
                                log::warn!("Failed to deserialize aliases: {e}");
                                UnionFind::new()
                            })
                    }
                    None => UnionFind::new(),
                };

                let metadata = match table.get(META_KEY_METADATA).map_err(|e| redb_err("get", e))? {
                    Some(data) => {
                        bincode::deserialize(data.value())
                            .unwrap_or_else(|e| {
                                log::warn!("Failed to deserialize metadata: {e}");
                                MetadataStore::new()
                            })
                    }
                    None => MetadataStore::new(),
                };

                (aliases, metadata)
            };

            (next_seq, aliases, metadata)
        };

        Ok(Self {
            db: Some(db),
            db_path: path,
            next_seq,
            aliases,
            metadata,
            aliases_dirty: false,
            metadata_dirty: false,
        })
    }

    /// Get reference to the open database.
    fn db(&self) -> Result<&Database, AttestError> {
        self.db.as_ref().ok_or_else(|| {
            AttestError::Provenance("database is closed".to_string())
        })
    }

    /// Begin a read transaction.
    fn read_txn(&self) -> Result<redb::ReadTransaction, AttestError> {
        self.db()?.begin_read().map_err(|e| redb_err("begin_read", e))
    }

    /// Begin a write transaction.
    fn write_txn(&self) -> Result<redb::WriteTransaction, AttestError> {
        self.db()?.begin_write().map_err(|e| redb_err("begin_write", e))
    }

    /// Increment a u64 counter in the given table.
    fn increment_counter(
        txn: &redb::WriteTransaction,
        table_def: TableDefinition<&str, u64>,
        key: &str,
    ) -> Result<(), AttestError> {
        let mut table = txn.open_table(table_def)
            .map_err(|e| redb_err("open table", e))?;
        let current = table.get(key).ok().flatten()
            .map(|v| v.value()).unwrap_or(0);
        table.insert(key, current + 1)
            .map_err(|e| redb_err("insert", e))?;
        Ok(())
    }

    // ── Lifecycle ──────────────────────────────────────────────────────

    pub fn close(&mut self) -> Result<(), AttestError> {
        self.flush_in_memory_state()?;
        self.db = None;
        Ok(())
    }

    pub fn checkpoint(&mut self) -> Result<(), AttestError> {
        self.flush_in_memory_state()
    }

    /// Flush dirty aliases/metadata to META_BLOBS.
    fn flush_in_memory_state(&mut self) -> Result<(), AttestError> {
        if !self.aliases_dirty && !self.metadata_dirty {
            return Ok(());
        }
        let db = self.db()?;
        let txn = db.begin_write().map_err(|e| redb_err("begin_write", e))?;
        {
            let mut table = txn.open_table(META_BLOBS)
                .map_err(|e| redb_err("open table", e))?;
            if self.aliases_dirty {
                let bytes = bincode::serialize(&self.aliases)
                    .map_err(|e| redb_err("serialize aliases", e))?;
                table.insert(META_KEY_ALIASES, bytes.as_slice())
                    .map_err(|e| redb_err("insert", e))?;
                self.aliases_dirty = false;
            }
            if self.metadata_dirty {
                let bytes = bincode::serialize(&self.metadata)
                    .map_err(|e| redb_err("serialize metadata", e))?;
                table.insert(META_KEY_METADATA, bytes.as_slice())
                    .map_err(|e| redb_err("insert", e))?;
                self.metadata_dirty = false;
            }
        }
        txn.commit().map_err(|e| redb_err("commit", e))?;
        Ok(())
    }

    // ── Metadata ───────────────────────────────────────────────────────

    pub fn register_vocabulary(&mut self, namespace: &str, vocab: Vocabulary) {
        self.metadata.register_vocabulary(namespace, vocab);
        self.metadata_dirty = true;
    }

    pub fn register_predicate(&mut self, predicate_id: &str, constraints: serde_json::Value) {
        self.metadata.register_predicate(predicate_id, constraints);
        self.metadata_dirty = true;
    }

    pub fn register_payload_schema(&mut self, schema_id: &str, schema: serde_json::Value) {
        self.metadata.register_payload_schema(schema_id, schema);
        self.metadata_dirty = true;
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

    pub fn resolve(&mut self, entity_id: &str) -> String {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.find(&canonical)
    }

    pub fn get_alias_group(&mut self, entity_id: &str) -> HashSet<String> {
        let canonical = normalize_entity_id(entity_id);
        self.aliases.get_group(&canonical)
    }

    /// Apply alias side-effects for a claim. Call AFTER transaction commit.
    fn apply_alias_predicate(&mut self, claim: &Claim) {
        if ALIAS_PREDICATES.contains(claim.predicate.id.as_str()) {
            if claim.predicate.id == "same_as" {
                self.aliases.union(&claim.subject.id, &claim.object.id);
                self.aliases_dirty = true;
            } else if claim.predicate.id == "not_same_as" {
                self.aliases.split(&claim.subject.id);
                self.aliases_dirty = true;
            }
        }
    }

    // ── Cache management ───────────────────────────────────────────────

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
        let txn = match self.write_txn() {
            Ok(t) => t,
            Err(e) => { log::warn!("upsert_entity: {e}"); return; }
        };
        {
            // Open ENTITIES once — used for both read check and write
            let mut entities = match txn.open_table(ENTITIES) {
                Ok(t) => t,
                Err(e) => { log::warn!("upsert_entity open table: {e}"); return; }
            };

            // Check if entity exists → merge external_ids only
            let existing: Option<EntityData> = entities.get(entity_id).ok().flatten()
                .and_then(|data| bincode::deserialize(data.value()).ok());

            if let Some(mut entity) = existing {
                if let Some(new_ids) = external_ids {
                    for (k, v) in new_ids {
                        entity.external_ids.insert(k.clone(), v.clone());
                    }
                    if let Ok(bytes) = bincode::serialize(&entity) {
                        let _ = entities.insert(entity_id, bytes.as_slice());
                    }
                }
                drop(entities);
                let _ = txn.commit();
                return;
            }

            let display = if display_name.is_empty() {
                entity_id
            } else {
                display_name
            };

            let data = EntityData {
                id: entity_id.to_string(),
                entity_type: entity_type.to_string(),
                display_name: display.to_string(),
                external_ids: external_ids.cloned().unwrap_or_default(),
                created_at: timestamp,
            };
            let bytes = match bincode::serialize(&data) {
                Ok(b) => b,
                Err(_) => return,
            };
            let _ = entities.insert(entity_id, bytes.as_slice());
            drop(entities);

            // Update ENTITY_TYPE_IDX
            if let Ok(mut type_idx) = txn.open_multimap_table(ENTITY_TYPE_IDX) {
                let _ = type_idx.insert(entity_type, entity_id);
            }

            // Update TEXT_IDX
            if let Ok(mut text_idx) = txn.open_multimap_table(TEXT_IDX) {
                let mut seen = HashSet::new();
                for token in tokenize(display).into_iter().chain(tokenize(entity_id)) {
                    if seen.insert(token.clone()) {
                        let _ = text_idx.insert(token.as_str(), entity_id);
                    }
                }
            }

            // Update ENTITY_TYPE_COUNTS
            let _ = Self::increment_counter(&txn, ENTITY_TYPE_COUNTS, entity_type);
        }
        let _ = txn.commit();
    }

    /// Upsert multiple entities in a single write transaction.
    pub fn upsert_entities_batch(
        &mut self,
        entities: &[(String, String, String, HashMap<String, String>)],
        timestamp: i64,
    ) {
        if entities.is_empty() {
            return;
        }
        let txn = match self.write_txn() {
            Ok(t) => t,
            Err(e) => { log::warn!("upsert_entities_batch: {e}"); return; }
        };

        let mut entities_table = match txn.open_table(ENTITIES) {
            Ok(t) => t,
            Err(e) => { log::warn!("upsert_entities_batch open: {e}"); return; }
        };
        let mut type_idx = txn.open_multimap_table(ENTITY_TYPE_IDX).ok();
        let mut text_idx = txn.open_multimap_table(TEXT_IDX).ok();

        for (entity_id, entity_type, display_name, external_ids) in entities {
            let existing: Option<EntityData> = entities_table.get(entity_id.as_str()).ok().flatten()
                .and_then(|data| bincode::deserialize(data.value()).ok());

            if let Some(mut entity) = existing {
                if !external_ids.is_empty() {
                    for (k, v) in external_ids {
                        entity.external_ids.insert(k.clone(), v.clone());
                    }
                    if let Ok(bytes) = bincode::serialize(&entity) {
                        let _ = entities_table.insert(entity_id.as_str(), bytes.as_slice());
                    }
                }
                continue;
            }

            let display = if display_name.is_empty() {
                entity_id.as_str()
            } else {
                display_name.as_str()
            };

            let data = EntityData {
                id: entity_id.clone(),
                entity_type: entity_type.clone(),
                display_name: display.to_string(),
                external_ids: external_ids.clone(),
                created_at: timestamp,
            };
            if let Ok(bytes) = bincode::serialize(&data) {
                let _ = entities_table.insert(entity_id.as_str(), bytes.as_slice());
            }

            if let Some(ref mut idx) = type_idx {
                let _ = idx.insert(entity_type.as_str(), entity_id.as_str());
            }
            if let Some(ref mut idx) = text_idx {
                let mut seen = HashSet::new();
                for token in tokenize(display).into_iter().chain(tokenize(entity_id)) {
                    if seen.insert(token.clone()) {
                        let _ = idx.insert(token.as_str(), entity_id.as_str());
                    }
                }
            }
            let _ = Self::increment_counter(&txn, ENTITY_TYPE_COUNTS, entity_type);
        }

        drop(entities_table);
        drop(type_idx);
        drop(text_idx);
        let _ = txn.commit();
    }

    pub fn get_entity(&self, entity_id: &str) -> Option<EntitySummary> {
        let txn = self.read_txn().ok()?;
        let entities = txn.open_table(ENTITIES).ok()?;
        let data_bytes = entities.get(entity_id).ok()??;
        let entity: EntityData = bincode::deserialize(data_bytes.value()).ok()?;

        let claim_count = txn.open_multimap_table(ENTITY_CLAIMS_IDX).ok()
            .and_then(|idx| idx.get(entity_id).ok())
            .map(|iter| iter.count())
            .unwrap_or(0);

        Some(entity.to_summary(claim_count))
    }

    pub fn list_entities(
        &self,
        entity_type: Option<&str>,
        min_claims: usize,
    ) -> Vec<EntitySummary> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let claims_idx = txn.open_multimap_table(ENTITY_CLAIMS_IDX).ok();

        let count_claims = |eid: &str| -> usize {
            claims_idx.as_ref()
                .and_then(|idx| idx.get(eid).ok())
                .map(|iter| iter.count())
                .unwrap_or(0)
        };

        match entity_type {
            Some(et) => {
                // Use type index → look up each entity by ID
                let entity_ids: Vec<String> = match txn.open_multimap_table(ENTITY_TYPE_IDX) {
                    Ok(idx) => {
                        idx.get(et).ok()
                            .map(|iter| iter.filter_map(|v| {
                                v.ok().map(|v| v.value().to_string())
                            }).collect())
                            .unwrap_or_default()
                    }
                    Err(_) => return Vec::new(),
                };
                let entities_table = match txn.open_table(ENTITIES) {
                    Ok(t) => t,
                    Err(_) => return Vec::new(),
                };
                entity_ids.into_iter().filter_map(|eid| {
                    let data_bytes = entities_table.get(eid.as_str()).ok()??;
                    let entity: EntityData = bincode::deserialize(data_bytes.value()).ok()?;
                    let cc = count_claims(&eid);
                    (cc >= min_claims).then(|| entity.to_summary(cc))
                }).collect()
            }
            None => {
                // Scan ENTITIES directly — no intermediate ID collection
                let entities_table = match txn.open_table(ENTITIES) {
                    Ok(t) => t,
                    Err(_) => return Vec::new(),
                };
                entities_table.iter().ok()
                    .map(|iter| {
                        iter.filter_map(|entry| {
                            let (k, v) = entry.ok()?;
                            let entity: EntityData = bincode::deserialize(v.value()).ok()?;
                            let cc = count_claims(k.value());
                            (cc >= min_claims).then(|| entity.to_summary(cc))
                        }).collect()
                    })
                    .unwrap_or_default()
            }
        }
    }

    // ── Claim operations ───────────────────────────────────────────────

    pub fn insert_claim(&mut self, claim: Claim, _checkpoint_interval: u64) -> bool {
        let txn = match self.write_txn() {
            Ok(t) => t,
            Err(_) => return false,
        };

        // Check existence within the write transaction (no separate read txn)
        {
            let table = match txn.open_table(CLAIM_ID_IDX) {
                Ok(t) => t,
                Err(_) => return false,
            };
            if table.get(claim.claim_id.as_str()).ok().flatten().is_some() {
                return false;
            }
        }

        let seq = self.next_seq;
        if self.write_claim_to_txn(&txn, seq, &claim).is_err() {
            return false;
        }

        match txn.commit() {
            Ok(_) => {
                self.next_seq = seq + 1;
                // Apply alias mutation AFTER successful commit
                self.apply_alias_predicate(&claim);
                true
            }
            Err(_) => false,
        }
    }

    pub fn insert_claims_batch(&mut self, claims: Vec<Claim>, _checkpoint_interval: u64) -> usize {
        // Dedup within the batch first (cheap, in-memory)
        let mut seen = HashSet::new();
        let unique_claims: Vec<Claim> = claims.into_iter()
            .filter(|c| seen.insert(c.claim_id.clone()))
            .collect();

        if unique_claims.is_empty() {
            return 0;
        }

        // Chunk into multiple transactions to limit dirty page memory.
        // Without chunking, 15M claims in one txn can require 20+ GB of dirty pages,
        // stalling indefinitely on spinning disks.
        const CHUNK_SIZE: usize = 100_000;
        let mut total_inserted = 0usize;

        for chunk in unique_claims.chunks(CHUNK_SIZE) {
            let txn = match self.write_txn() {
                Ok(t) => t,
                Err(_) => break,
            };

            // Filter against existing data within this transaction
            let new_in_chunk: Vec<&Claim> = {
                let table = match txn.open_table(CLAIM_ID_IDX) {
                    Ok(t) => t,
                    Err(_) => break,
                };
                chunk.iter().filter(|c| {
                    table.get(c.claim_id.as_str()).ok().flatten().is_none()
                }).collect()
            };

            if new_in_chunk.is_empty() {
                // Drop txn without commit (read-only)
                continue;
            }

            let mut seq = self.next_seq;
            let mut chunk_inserted = 0usize;
            for claim in &new_in_chunk {
                if self.write_claim_to_txn(&txn, seq, claim).is_ok() {
                    seq += 1;
                    chunk_inserted += 1;
                }
            }

            match txn.commit() {
                Ok(_) => {
                    self.next_seq = seq;
                    for claim in &new_in_chunk {
                        self.apply_alias_predicate(claim);
                    }
                    total_inserted += chunk_inserted;
                }
                Err(_) => break,
            }
        }

        total_inserted
    }

    /// Write a single claim into all tables within an existing write transaction.
    fn write_claim_to_txn(
        &self,
        txn: &redb::WriteTransaction,
        seq: u64,
        claim: &Claim,
    ) -> Result<(), AttestError> {
        let bytes = bincode::serialize(claim)
            .map_err(|e| redb_err("serialize claim", e))?;

        // CLAIMS: seq → bytes
        {
            let mut table = txn.open_table(CLAIMS)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(seq, bytes.as_slice())
                .map_err(|e| redb_err("insert", e))?;
        }
        // CLAIM_ID_IDX
        {
            let mut table = txn.open_table(CLAIM_ID_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.claim_id.as_str(), seq)
                .map_err(|e| redb_err("insert", e))?;
        }
        // CONTENT_ID_IDX
        {
            let mut table = txn.open_multimap_table(CONTENT_ID_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.content_id.as_str(), seq)
                .map_err(|e| redb_err("insert", e))?;
        }
        // ENTITY_CLAIMS_IDX (subject + object)
        {
            let mut table = txn.open_multimap_table(ENTITY_CLAIMS_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.subject.id.as_str(), seq)
                .map_err(|e| redb_err("insert", e))?;
            if claim.object.id != claim.subject.id {
                table.insert(claim.object.id.as_str(), seq)
                    .map_err(|e| redb_err("insert", e))?;
            }
        }
        // ADJACENCY_IDX (bidirectional, skip self-loops)
        if claim.subject.id != claim.object.id {
            let mut table = txn.open_multimap_table(ADJACENCY_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.subject.id.as_str(), claim.object.id.as_str())
                .map_err(|e| redb_err("insert", e))?;
            table.insert(claim.object.id.as_str(), claim.subject.id.as_str())
                .map_err(|e| redb_err("insert", e))?;
        }
        // TIMESTAMP_IDX
        {
            let mut table = txn.open_multimap_table(TIMESTAMP_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.timestamp, seq)
                .map_err(|e| redb_err("insert", e))?;
        }
        // SOURCE_IDX
        {
            let mut table = txn.open_multimap_table(SOURCE_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.provenance.source_id.as_str(), seq)
                .map_err(|e| redb_err("insert", e))?;
        }
        // PREDICATE_IDX
        {
            let mut table = txn.open_multimap_table(PREDICATE_IDX)
                .map_err(|e| redb_err("open table", e))?;
            table.insert(claim.predicate.id.as_str(), seq)
                .map_err(|e| redb_err("insert", e))?;
        }
        // Stats counters
        Self::increment_counter(txn, PRED_TYPE_COUNTS, claim.predicate.predicate_type.as_str())?;
        Self::increment_counter(txn, SRC_TYPE_COUNTS, claim.provenance.source_type.as_str())?;
        Ok(())
    }

    fn claim_exists_internal(&self, claim_id: &str) -> Result<bool, AttestError> {
        let txn = self.read_txn()?;
        let table = txn.open_table(CLAIM_ID_IDX)
            .map_err(|e| redb_err("open table", e))?;
        Ok(table.get(claim_id).map_err(|e| redb_err("get", e))?.is_some())
    }

    pub fn claim_exists(&self, claim_id: &str) -> bool {
        self.claim_exists_internal(claim_id).unwrap_or(false)
    }

    /// Load a claim by seq number.
    fn load_claim_by_seq(&self, txn: &redb::ReadTransaction, seq: u64) -> Option<Claim> {
        let table = txn.open_table(CLAIMS).ok()?;
        let data = table.get(seq).ok()??;
        bincode::deserialize(data.value()).ok()
    }

    /// Load claims for a set of seq numbers.
    fn load_claims_by_seqs(&self, txn: &redb::ReadTransaction, seqs: &[u64]) -> Vec<Claim> {
        let table = match txn.open_table(CLAIMS) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        seqs.iter().filter_map(|&seq| {
            table.get(seq).ok()?.map(|data| {
                bincode::deserialize::<Claim>(data.value()).ok()
            }).flatten()
        }).collect()
    }

    /// Collect seq numbers from a str→u64 multimap entry.
    fn collect_seqs_str(
        txn: &redb::ReadTransaction,
        table_def: redb::MultimapTableDefinition<&str, u64>,
        key: &str,
    ) -> Vec<u64> {
        match txn.open_multimap_table(table_def) {
            Ok(table) => {
                match table.get(key) {
                    Ok(iter) => iter.filter_map(|v| v.ok().map(|v| v.value())).collect(),
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        }
    }

    /// Look up claims via a str→u64 multimap index.
    fn claims_by_str_index(
        &self,
        table_def: redb::MultimapTableDefinition<&str, u64>,
        key: &str,
    ) -> Vec<Claim> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let seqs = Self::collect_seqs_str(&txn, table_def, key);
        self.load_claims_by_seqs(&txn, &seqs)
    }

    pub fn claims_by_content_id(&self, content_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(CONTENT_ID_IDX, content_id)
    }

    pub fn all_claims(&self) -> Vec<Claim> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let table = match txn.open_table(CLAIMS) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let capacity = table.len().unwrap_or(0) as usize;
        let mut claims = Vec::with_capacity(capacity);
        if let Ok(iter) = table.iter() {
            for entry in iter {
                if let Ok((_, v)) = entry {
                    if let Ok(claim) = bincode::deserialize::<Claim>(v.value()) {
                        claims.push(claim);
                    }
                }
            }
        }
        claims
    }

    pub fn claims_by_source_id(&self, source_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(SOURCE_IDX, source_id)
    }

    pub fn claims_by_predicate_id(&self, predicate_id: &str) -> Vec<Claim> {
        self.claims_by_str_index(PREDICATE_IDX, predicate_id)
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

        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let claims = self.claims_for_entities(&txn, &aliases);

        // Apply filters
        claims.into_iter().filter(|c| {
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
            c.confidence >= min_confidence
        }).collect()
    }

    pub fn get_claim_provenance_chain(&self, claim_id: &str) -> Vec<String> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let seq = match txn.open_table(CLAIM_ID_IDX) {
            Ok(table) => match table.get(claim_id) {
                Ok(Some(v)) => v.value(),
                _ => return Vec::new(),
            },
            Err(_) => return Vec::new(),
        };
        self.load_claim_by_seq(&txn, seq)
            .map(|c| c.provenance.chain)
            .unwrap_or_default()
    }

    // ── Graph traversal ────────────────────────────────────────────────

    /// Get neighbor entity IDs from the adjacency index.
    fn neighbors(&self, txn: &redb::ReadTransaction, entity_id: &str) -> HashSet<String> {
        match txn.open_multimap_table(ADJACENCY_IDX) {
            Ok(table) => {
                match table.get(entity_id) {
                    Ok(iter) => iter
                        .filter_map(|v| v.ok().map(|v| v.value().to_string()))
                        .collect(),
                    Err(_) => HashSet::new(),
                }
            }
            Err(_) => HashSet::new(),
        }
    }

    /// Get claims for a set of entity IDs (deduplicated by seq).
    fn claims_for_entities(
        &self,
        txn: &redb::ReadTransaction,
        entity_ids: &HashSet<String>,
    ) -> Vec<Claim> {
        let mut seen_seqs = HashSet::new();
        let mut all_seqs = Vec::new();
        for eid in entity_ids {
            let seqs = Self::collect_seqs_str(txn, ENTITY_CLAIMS_IDX, eid.as_str());
            for seq in seqs {
                if seen_seqs.insert(seq) {
                    all_seqs.push(seq);
                }
            }
        }
        self.load_claims_by_seqs(txn, &all_seqs)
    }

    pub fn bfs_claims(&mut self, entity_id: &str, max_depth: usize) -> Vec<(Claim, usize)> {
        let resolved = self.resolve(entity_id);
        let aliases = self.get_alias_group(&resolved);

        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut visited: HashSet<String> = aliases.clone();
        let mut frontier: HashSet<String> = aliases;
        let mut results: Vec<(Claim, usize)> = Vec::new();

        for hop in 1..=max_depth {
            if frontier.is_empty() {
                break;
            }

            let claims = self.claims_for_entities(&txn, &frontier);
            let mut next_frontier = HashSet::new();

            for claim in claims {
                for eid in [&claim.subject.id, &claim.object.id] {
                    if !visited.contains(eid) {
                        next_frontier.insert(eid.clone());
                        visited.insert(eid.clone());
                    }
                }
                results.push((claim, hop));
            }

            frontier = next_frontier;
        }

        results
    }

    pub fn path_exists(&mut self, entity_a: &str, entity_b: &str, max_depth: usize) -> bool {
        let ra = self.resolve(entity_a);
        let rb = self.resolve(entity_b);

        if ra == rb {
            return true;
        }

        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return false,
        };

        let mut visited = HashSet::new();
        visited.insert(ra.clone());
        let mut frontier = HashSet::new();
        frontier.insert(ra);

        for _ in 0..max_depth {
            if frontier.is_empty() {
                break;
            }

            let mut next_frontier = HashSet::new();
            for eid in &frontier {
                let neighbors = self.neighbors(&txn, eid);
                for n in neighbors {
                    if n == rb {
                        return true;
                    }
                    if !visited.contains(&n) {
                        next_frontier.insert(n.clone());
                        visited.insert(n);
                    }
                }
            }

            frontier = next_frontier;
        }

        false
    }

    pub fn get_adjacency_list(&self) -> HashMap<String, HashSet<String>> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };
        let table = match txn.open_multimap_table(ADJACENCY_IDX) {
            Ok(t) => t,
            Err(_) => return HashMap::new(),
        };

        let mut result: HashMap<String, HashSet<String>> = HashMap::new();
        if let Ok(iter) = table.iter() {
            for entry in iter {
                if let Ok((k, values)) = entry {
                    let neighbors: HashSet<String> = values
                        .filter_map(|v| v.ok().map(|v| v.value().to_string()))
                        .collect();
                    if !neighbors.is_empty() {
                        result.entry(k.value().to_string())
                            .or_default()
                            .extend(neighbors);
                    }
                }
            }
        }
        result
    }

    // ── Temporal queries ───────────────────────────────────────────────

    pub fn claims_in_range(&mut self, min_ts: i64, max_ts: i64) -> Vec<Claim> {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let table = match txn.open_multimap_table(TIMESTAMP_IDX) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        let mut seqs = Vec::new();
        if let Ok(range) = table.range(min_ts..=max_ts) {
            for entry in range {
                if let Ok((_, values)) = entry {
                    for v in values {
                        if let Ok(v) = v {
                            seqs.push(v.value());
                        }
                    }
                }
            }
        }
        self.load_claims_by_seqs(&txn, &seqs)
    }

    pub fn most_recent_claims(&mut self, n: usize) -> Vec<Claim> {
        if n == 0 {
            return Vec::new();
        }
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Use the CLAIMS table (ordered by seq, higher seq = more recent insertion).
        // This matches the in-memory implementation which returns claims in
        // reverse insertion order. For timestamp ordering, we load then sort.
        let table = match txn.open_table(CLAIMS) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Iterate claims in reverse seq order, take first n
        let mut claims: Vec<Claim> = Vec::with_capacity(n);
        if let Ok(range) = table.range::<u64>(..) {
            for entry in range.rev() {
                if claims.len() >= n {
                    break;
                }
                if let Ok((_, v)) = entry {
                    if let Ok(claim) = bincode::deserialize::<Claim>(v.value()) {
                        claims.push(claim);
                    }
                }
            }
        }
        // Sort by timestamp descending for consistent ordering
        claims.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        claims
    }

    // ── Text search ────────────────────────────────────────────────────

    pub fn search_entities(&self, query: &str, top_k: usize) -> Vec<EntitySummary> {
        let tokens = tokenize(query);
        if tokens.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let text_idx = match txn.open_multimap_table(TEXT_IDX) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Count matching tokens per entity
        let mut scores: HashMap<String, usize> = HashMap::new();
        for token in &tokens {
            if let Ok(iter) = text_idx.get(token.as_str()) {
                for v in iter {
                    if let Ok(v) = v {
                        *scores.entry(v.value().to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort by score descending, take top_k
        let mut ranked: Vec<(String, usize)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        let entities_table = match txn.open_table(ENTITIES) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let claims_idx = txn.open_multimap_table(ENTITY_CLAIMS_IDX).ok();

        ranked.into_iter().take(top_k).filter_map(|(eid, _)| {
            let data_bytes = entities_table.get(eid.as_str()).ok()??;
            let entity: EntityData = bincode::deserialize(data_bytes.value()).ok()?;
            let claim_count = claims_idx.as_ref()
                .and_then(|idx| idx.get(eid.as_str()).ok())
                .map(|iter| iter.count())
                .unwrap_or(0);
            Some(entity.to_summary(claim_count))
        }).collect()
    }

    // ── Stats ──────────────────────────────────────────────────────────

    pub fn stats(&self) -> StoreStats {
        let txn = match self.read_txn() {
            Ok(t) => t,
            Err(_) => return StoreStats::default(),
        };

        let total_claims = txn.open_table(CLAIMS).ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0) as usize;

        let entity_count = txn.open_table(ENTITIES).ok()
            .and_then(|t| t.len().ok())
            .unwrap_or(0) as usize;

        let entity_types = Self::read_counts(&txn, ENTITY_TYPE_COUNTS);
        let predicate_types = Self::read_counts(&txn, PRED_TYPE_COUNTS);
        let source_types = Self::read_counts(&txn, SRC_TYPE_COUNTS);

        StoreStats {
            total_claims,
            entity_count,
            entity_types,
            predicate_types,
            source_types,
        }
    }

    fn read_counts(
        txn: &redb::ReadTransaction,
        table_def: TableDefinition<&str, u64>,
    ) -> HashMap<String, usize> {
        match txn.open_table(table_def) {
            Ok(table) => {
                table.iter().ok()
                    .map(|iter| {
                        iter.filter_map(|entry| {
                            entry.ok().map(|(k, v)| {
                                (k.value().to_string(), v.value() as usize)
                            })
                        }).collect()
                    })
                    .unwrap_or_default()
            }
            Err(_) => HashMap::new(),
        }
    }
}

impl Drop for RedbBackend {
    fn drop(&mut self) {
        if self.db.is_some() {
            log::warn!(
                "RedbBackend dropped without calling close() — \
                 attempting best-effort flush."
            );
            if let Err(e) = self.flush_in_memory_state() {
                log::error!("Best-effort flush in Drop failed: {e}");
            }
            self.db = None;
        }
    }
}
