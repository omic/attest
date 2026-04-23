"""Shared extraction-output dataclasses.

Lives in ``intelligence/`` but is intentionally OSS-shipped alongside
``heuristic_extractor.py``: the rule-based OSS extractor and the
LLM-backed enterprise extractor must produce the same result types so
callers can swap between them. Keeping the types in a standalone
module means OSS users never pull in ``text_extractor.py`` (which
imports LLM SDKs) just to import a result type.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from attestdb.core.types import ClaimInput


@dataclass
class ExtractedClaim:
    """Raw extractor output (LLM or rule-based) before validation."""

    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    confidence: float = 0.7
    evidence_text: str = ""
    #: Optional ISO-8601 date (YYYY-MM-DD) the claim's underlying event
    #: occurred, when the supporting text identifies one. Empty string
    #: when unknown. Passed through ``_validate_and_convert`` to
    #: ``ClaimInput.timestamp`` (nanoseconds). Distinct from ingest
    #: time, which the ingestion layer stamps separately.
    event_date: str = ""


@dataclass
class DocumentEntity:
    """One named person (or named pseudonym) discovered in a document,
    along with the role mentions and aliases they can appear under.

    Used by the two-pass catalog-aware extraction flow: a cheap first
    LLM pass builds the catalog, then the claims-extraction prompt
    consults it to resolve role mentions ('the pilot', 'the houseman')
    to the named individual before emitting claims.
    """

    name: str
    aliases: list[str] = field(default_factory=list)
    role: str | None = None


@dataclass
class ExtractionResult:
    """Validated claims + rejected candidates + warnings + stats from one extraction pass."""

    claims: list[ClaimInput] = field(default_factory=list)
    rejected: list[tuple[ExtractedClaim, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    raw_count: int = 0
    discovered_entity_types: set[str] = field(default_factory=set)
    discovered_predicates: set[str] = field(default_factory=set)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Document catalog built during the two-pass extraction flow, if
    #: enabled. Empty for single-pass runs. Consumers that want to feed
    #: alias information into entity-resolution (e.g. via same_as
    #: claims) read this.
    catalog: list[DocumentEntity] = field(default_factory=list)

    @property
    def n_valid(self) -> int:
        return len(self.claims)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)
