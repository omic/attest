//! attest-store: Storage engine for Attest.
//!
//! Provides a claim store with pluggable backends:
//! - In-memory backend with file persistence (original implementation)
//! - redb B+ tree backend for disk-efficient storage (Phase 2)
//!
//! # Architecture
//!
//! - [`ClaimLog`] — Append-only claim storage with secondary indexes
//! - [`EntityStore`] — Entity CRUD with type indexing
//! - [`UnionFind`] — Alias resolution via union-find with path compression
//! - [`MetadataStore`] — Vocabulary, predicate constraint, and schema registry
//! - [`RustStore`] — Top-level API with backend dispatch
//!
//! # Persistence
//!
//! Store state is serialized to a single file with a verified header
//! (magic bytes, version, CRC32 checksum). See [`file_format`].

pub mod backend;
pub mod claim_log;
pub mod entity_store;
pub mod file_format;
pub mod metadata;
pub mod store;
pub mod union_find;
pub mod wal;

pub use metadata::Vocabulary;
pub use store::{RustStore, StoreStats};
