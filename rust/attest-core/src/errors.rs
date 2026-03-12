//! Exception hierarchy for Attest.

use std::fmt;

/// Base error type for all Attest errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestError {
    /// Missing or invalid provenance.
    Provenance(String),
    /// Unknown entity, predicate, or source type.
    Vocabulary(String),
    /// Payload doesn't match registered schema.
    SchemaValidation(String),
    /// Subject/object types don't match predicate constraints.
    PredicateConstraint(String),
    /// Content-addressed claim ID already exists.
    DuplicateClaim(String),
    /// Provenance chain contains cycles.
    CircularProvenance(String),
    /// Embedding dimensionality doesn't match database config.
    Dimensionality(String),
    /// Entity not found in the database.
    EntityNotFound(String),
    /// Attempted write on a read-only store.
    ReadOnly,
    /// Generic validation error.
    Validation(String),
}

impl fmt::Display for AttestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provenance(msg) => write!(f, "ProvenanceError: {msg}"),
            Self::Vocabulary(msg) => write!(f, "VocabularyError: {msg}"),
            Self::SchemaValidation(msg) => write!(f, "SchemaValidationError: {msg}"),
            Self::PredicateConstraint(msg) => write!(f, "PredicateConstraintError: {msg}"),
            Self::DuplicateClaim(msg) => write!(f, "DuplicateClaimError: {msg}"),
            Self::CircularProvenance(msg) => write!(f, "CircularProvenanceError: {msg}"),
            Self::Dimensionality(msg) => write!(f, "DimensionalityError: {msg}"),
            Self::EntityNotFound(msg) => write!(f, "EntityNotFoundError: {msg}"),
            Self::ReadOnly => write!(f, "ReadOnlyError: store is read-only"),
            Self::Validation(msg) => write!(f, "ValidationError: {msg}"),
        }
    }
}

impl std::error::Error for AttestError {}

/// Convenience type alias for Attest operations.
pub type Result<T> = std::result::Result<T, AttestError>;
