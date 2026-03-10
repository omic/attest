//! Storage backend abstraction.
//!
//! Provides two backends:
//! - [`MemoryBackend`]: In-memory store with optional file persistence (original implementation)
//! - [`RedbBackend`]: File-backed store using redb B+ trees

pub mod memory;
pub mod migration;
pub mod redb;
pub mod tables;

pub use memory::MemoryBackend;
pub use self::redb::RedbBackend;

/// Storage backend for RustStore.
pub enum Backend {
    /// In-memory store with optional file persistence via checkpoint + WAL.
    InMemory(MemoryBackend),
    /// File-backed store using redb B+ trees.
    File(RedbBackend),
}
