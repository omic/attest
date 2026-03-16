//! attest-store: Storage engine for Attest.
//!
//! Provides a claim store with pluggable backends:
//! - In-memory backend (`:memory:` databases)
//! - LMDB backend via heed (default for file-backed databases)
//!
//! # Architecture
//!
//! - [`ClaimLog`] — Append-only claim storage with secondary indexes
//! - [`EntityStore`] — Entity CRUD with type indexing
//! - [`UnionFind`] — Alias resolution via union-find with path compression
//! - [`MetadataStore`] — Vocabulary, predicate constraint, and schema registry
//! - [`RustStore`] — Top-level API with backend dispatch

pub mod backend;
pub mod claim_log;
pub mod entity_store;
pub mod file_format;
pub mod journal;
pub mod metadata;
pub mod store;
pub mod union_find;
pub mod wal;

pub use metadata::Vocabulary;
pub use store::{RustStore, StoreStats};
