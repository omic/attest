"""Claim identity functions — locked, shared between Phase 1 and Phase 2.

SHA-256 of pipe-delimited canonical inputs, UTF-8 encoded.
"""

import hashlib


def compute_claim_id(
    subject_canonical: str,
    predicate_id: str,
    object_canonical: str,
    source_id: str,
    source_type: str,
    timestamp: int,
) -> str:
    """Compute globally unique claim ID."""
    payload = (
        f"{subject_canonical}|{predicate_id}|{object_canonical}"
        f"|{source_id}|{source_type}|{timestamp}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_content_id(
    subject_canonical: str,
    predicate_id: str,
    object_canonical: str,
) -> str:
    """Compute corroboration grouping key."""
    payload = f"{subject_canonical}|{predicate_id}|{object_canonical}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_chain_hash(prev_chain_hash: str, claim_id: str) -> str:
    """Compute Merkle chain hash for tamper-evident append-only log.

    Each entry's hash incorporates the previous entry's hash, creating
    a verifiable chain.  Tampering with any earlier entry invalidates
    all subsequent hashes.
    """
    payload = f"{prev_chain_hash}|{claim_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
