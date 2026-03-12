//! redb table definitions for the file-backed storage backend.

use redb::{MultimapTableDefinition, TableDefinition};

// Primary claim storage: seq → bincode(Claim)
pub const CLAIMS: TableDefinition<u64, &[u8]> = TableDefinition::new("claims");

// Claim indexes
pub const CLAIM_ID_IDX: TableDefinition<&str, u64> = TableDefinition::new("claim_id_idx");
pub const CONTENT_ID_IDX: MultimapTableDefinition<&str, u64> =
    MultimapTableDefinition::new("content_id_idx");
pub const ENTITY_CLAIMS_IDX: MultimapTableDefinition<&str, u64> =
    MultimapTableDefinition::new("entity_claims_idx");
pub const ADJACENCY_IDX: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("adjacency_idx");
pub const TIMESTAMP_IDX: MultimapTableDefinition<i64, u64> =
    MultimapTableDefinition::new("timestamp_idx");
pub const SOURCE_IDX: MultimapTableDefinition<&str, u64> =
    MultimapTableDefinition::new("source_idx");
pub const PREDICATE_IDX: MultimapTableDefinition<&str, u64> =
    MultimapTableDefinition::new("predicate_idx");

pub const NAMESPACE_IDX: MultimapTableDefinition<&str, u64> =
    MultimapTableDefinition::new("namespace_idx");

// Entity storage: entity_id → bincode(EntityData)
pub const ENTITIES: TableDefinition<&str, &[u8]> = TableDefinition::new("entities");
pub const ENTITY_TYPE_IDX: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("entity_type_idx");
pub const TEXT_IDX: MultimapTableDefinition<&str, &str> =
    MultimapTableDefinition::new("text_idx");

// In-memory state blobs (loaded on open, flushed on close/checkpoint)
pub const META_BLOBS: TableDefinition<&str, &[u8]> = TableDefinition::new("meta_blobs");

// Status overrides: claim_id → u8 status code (0=active, 1=archived, 2=tombstoned, 3=provenance_degraded)
pub const STATUS_OVERRIDES: TableDefinition<&str, u8> = TableDefinition::new("status_overrides");

// Stats counters (maintained on insert for O(1) stats)
pub const PRED_TYPE_COUNTS: TableDefinition<&str, u64> =
    TableDefinition::new("pred_type_counts");
pub const SRC_TYPE_COUNTS: TableDefinition<&str, u64> =
    TableDefinition::new("src_type_counts");
pub const ENTITY_TYPE_COUNTS: TableDefinition<&str, u64> =
    TableDefinition::new("entity_type_counts");
