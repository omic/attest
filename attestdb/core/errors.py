"""Exception hierarchy for Attest."""


class AttestError(Exception):
    """Base exception for all Attest errors."""


class ProvenanceError(AttestError):
    """Missing or invalid provenance."""


class VocabularyError(AttestError):
    """Unknown entity, predicate, or source type."""


class SchemaValidationError(AttestError):
    """Payload doesn't match registered schema."""


class PredicateConstraintError(AttestError):
    """Subject/object types don't match predicate constraints."""


class DuplicateClaimError(AttestError):
    """Content-addressed claim ID already exists."""


class CircularProvenanceError(AttestError):
    """Provenance chain contains cycles."""


class DimensionalityError(AttestError):
    """Embedding dimensionality doesn't match database config."""


class EntityNotFoundError(AttestError):
    """Entity not found in the database."""


class ClaimRejectedByQualityGateError(AttestError):
    """Claim rejected by a quality gate callback during ingestion."""
