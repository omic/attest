"""HeuristicExtractor — pattern-based claim extraction without LLM.

Extracts (subject, predicate, object) triples from text using regex patterns
derived from registered vocabularies. Works offline — no API keys required.

Best for text containing explicit relational statements like:
- "BRCA1 binds RAD51"
- "Metformin treats Type 2 Diabetes"
- "auth-service depends on redis-cache"

For nuanced or implicit claims, use TextExtractor with an LLM.
"""

from __future__ import annotations

import logging
import re

from attestdb.core.normalization import normalize_entity_id
from attestdb.core.types import ClaimInput
from attestdb.intelligence.text_extractor import ExtractedClaim, ExtractionResult

logger = logging.getLogger(__name__)

# Natural-language surface forms for known predicates.
# Maps predicate_id → list of regex fragments that match in running text.
# Fragments are case-insensitive; captured groups before/after are entity spans.
_DEFAULT_PREDICATE_PATTERNS: dict[str, list[str]] = {
    # Bio
    "binds": [r"binds(?: to)?", r"bound to"],
    "inhibits": [r"inhibits", r"is an? inhibitor of", r"blocks"],
    "activates": [r"activates", r"is an? activator of"],
    "upregulates": [r"upregulates", r"up-regulates"],
    "downregulates": [r"downregulates", r"down-regulates"],
    "associated_with": [
        r"(?:is |are )?associated with",
        r"(?:is |are )?linked to",
        r"(?:is |are )?implicated in",
        r"(?:is |are )?related to",
    ],
    "treats": [r"treats", r"is (?:a |first-line )?treatment for"],
    "interacts": [r"interacts with"],
    "expressed_in": [r"(?:is |are )?expressed in"],
    "phosphorylates": [r"phosphorylates"],
    "participates_in": [r"participates in", r"(?:is |are )?involved in"],
    "regulates": [r"regulates"],
    # DevOps
    "depends_on": [r"depends on", r"requires"],
    "triggers": [r"triggers", r"fires"],
    "mitigates": [r"mitigates"],
    "monitors": [r"monitors"],
    "owns": [r"owns", r"is owned by"],
    "deploys": [r"deploys"],
    # ML
    "trained_on": [r"(?:was |is )?trained on"],
    "evaluated_on": [r"(?:was |is )?evaluated on"],
    "outperforms": [r"outperforms"],
    "uses_feature": [r"uses (?:the )?feature"],
    "derived_from": [r"(?:is |was )?derived from"],
    # Generic
    "caused": [r"causes", r"caused"],
    "contradicts": [r"contradicts"],
}

# Entity span pattern: captures noun phrases (capitalized words, optionally
# with hyphens, numbers, and connecting words).
# Examples: "BRCA1", "Type 2 Diabetes", "auth-service", "PI3K/AKT pathway"
# Uses a non-greedy match of capitalized words connected by spaces.
_ENTITY_RE = r"((?:[A-Z][A-Za-z0-9/\-]*(?:\s+(?:of|the|[A-Z0-9][A-Za-z0-9/\-]*|\d+))*))"


class HeuristicExtractor:
    """Pattern-based claim extraction — no LLM required.

    Builds regex patterns from predicate surface forms and matches them
    against input text. Entity boundaries are detected via capitalization
    heuristics.

    Args:
        predicates: Valid predicate IDs. If None, uses all built-in patterns.
        entity_types: Valid entity types for type inference.
        default_entity_type: Entity type to assign when type can't be inferred.
        predicate_constraints: {predicate_id: {subject_types: [...], object_types: [...]}}
            Used to infer entity types from predicate context.
        predicate_patterns: Additional {predicate_id: [regex_fragments]} to
            extend or override default patterns.
        confidence: Default confidence for heuristic extractions.
    """

    def __init__(
        self,
        predicates: set[str] | list[str] | None = None,
        entity_types: set[str] | list[str] | None = None,
        default_entity_type: str = "entity",
        predicate_constraints: dict[str, dict] | None = None,
        predicate_patterns: dict[str, list[str]] | None = None,
        known_entities: dict[str, str] | None = None,
        confidence: float = 0.6,
    ):
        self._entity_types = set(entity_types) if entity_types else set()
        self._default_entity_type = default_entity_type
        self._predicate_constraints = predicate_constraints or {}
        self._confidence = confidence

        # Known entity dictionary: {normalized_name: entity_type}
        # Enables matching lowercase entities like "redis", "kubernetes"
        self._known_entities: dict[str, str] = known_entities or {}

        # Build pattern table: merge defaults with overrides
        patterns = dict(_DEFAULT_PREDICATE_PATTERNS)
        if predicate_patterns:
            patterns.update(predicate_patterns)

        # Filter to requested predicates
        if predicates:
            pred_set = set(predicates)
            patterns = {k: v for k, v in patterns.items() if k in pred_set}

        # Compile regex patterns: {predicate_id: compiled_regex}
        # Entity pattern is case-sensitive (relies on capitalization heuristics).
        # Predicate fragment is wrapped in (?i:...) for case-insensitive matching.
        self._patterns: list[tuple[str, re.Pattern]] = []
        self._known_patterns: list[tuple[str, re.Pattern]] = []
        for pred_id, fragments in patterns.items():
            for frag in fragments:
                full = _ENTITY_RE + r"\s+(?i:" + frag + r")\s+" + _ENTITY_RE
                try:
                    self._patterns.append((pred_id, re.compile(full)))
                except re.error as e:
                    logger.warning("Invalid pattern for %s: %s", pred_id, e)

        # Build known-entity patterns (case-insensitive entity matching)
        if self._known_entities:
            _KNOWN_ENTITY_RE = self._build_known_entity_re()
            for pred_id, fragments in patterns.items():
                for frag in fragments:
                    full = _KNOWN_ENTITY_RE + r"\s+(?i:" + frag + r")\s+" + _KNOWN_ENTITY_RE
                    try:
                        self._known_patterns.append((pred_id, re.compile(full, re.IGNORECASE)))
                    except re.error:
                        pass

    def _build_known_entity_re(self) -> str:
        """Build a regex alternation matching any known entity name."""
        # Sort longest first to prefer longer matches
        names = sorted(self._known_entities.keys(), key=len, reverse=True)
        escaped = [re.escape(n) for n in names if len(n) >= 2]
        if not escaped:
            return r"(\b\w+\b)"
        return r"(" + "|".join(escaped) + r")"

    def extract(
        self,
        text: str,
        source_type: str = "chat_extraction",
        source_id: str = "",
        method: str = "heuristic_extraction_v1",
    ) -> ExtractionResult:
        """Extract structured claims from text using pattern matching.

        Args:
            text: The text to extract claims from.
            source_type: Provenance source_type for extracted claims.
            source_id: Provenance source_id.
            method: Provenance method tag.

        Returns:
            ExtractionResult with validated ClaimInputs.
        """
        result = ExtractionResult()
        seen: set[tuple[str, str, str]] = set()  # deduplicate

        for sentence in _split_sentences(text):
            for pred_id, pattern in self._patterns:
                for match in pattern.finditer(sentence):
                    subj_raw = match.group(1).strip()
                    obj_raw = match.group(2).strip()

                    if not subj_raw or not obj_raw:
                        continue

                    # Skip if entity is too short (likely false positive)
                    if len(subj_raw) < 2 or len(obj_raw) < 2:
                        continue

                    subj_norm = normalize_entity_id(subj_raw)
                    obj_norm = normalize_entity_id(obj_raw)

                    # Deduplicate
                    key = (subj_norm, pred_id, obj_norm)
                    if key in seen:
                        continue
                    seen.add(key)

                    result.raw_count += 1

                    # Infer entity types from predicate constraints
                    subj_type, obj_type = self._infer_types(pred_id)

                    claim_input = ClaimInput(
                        subject=(subj_norm, subj_type),
                        predicate=(pred_id, pred_id),
                        object=(obj_norm, obj_type),
                        provenance={
                            "source_type": source_type,
                            "source_id": source_id,
                            "method": method,
                        },
                        confidence=self._confidence,
                        payload={"schema": "", "data": {"evidence_text": sentence.strip()}},
                    )
                    result.claims.append(claim_input)

        # Pass 2: Known-entity matching (case-insensitive)
        if self._known_patterns:
            for sentence in _split_sentences(text):
                for pred_id, pattern in self._known_patterns:
                    for match in pattern.finditer(sentence):
                        subj_raw = match.group(1).strip()
                        obj_raw = match.group(2).strip()
                        if not subj_raw or not obj_raw or len(subj_raw) < 2 or len(obj_raw) < 2:
                            continue

                        subj_norm = normalize_entity_id(subj_raw)
                        obj_norm = normalize_entity_id(obj_raw)

                        key = (subj_norm, pred_id, obj_norm)
                        if key in seen:
                            continue
                        seen.add(key)
                        result.raw_count += 1

                        # Use known entity types
                        subj_type = self._known_entities.get(subj_norm, self._default_entity_type)
                        obj_type = self._known_entities.get(obj_norm, self._default_entity_type)
                        # Also try predicate constraints
                        if (
                            subj_type == self._default_entity_type
                            or obj_type == self._default_entity_type
                        ):
                            inf_s, inf_o = self._infer_types(pred_id)
                            if subj_type == self._default_entity_type:
                                subj_type = inf_s
                            if obj_type == self._default_entity_type:
                                obj_type = inf_o

                        claim_input = ClaimInput(
                            subject=(subj_norm, subj_type),
                            predicate=(pred_id, pred_id),
                            object=(obj_norm, obj_type),
                            provenance={
                                "source_type": source_type,
                                "source_id": source_id,
                                "method": method,
                            },
                            confidence=self._confidence,
                            payload={"schema": "", "data": {"evidence_text": sentence.strip()}},
                        )
                        result.claims.append(claim_input)

        logger.info(
            "Heuristic extraction: %d claims from text (%d chars)",
            result.n_valid, len(text),
        )
        return result

    def _infer_types(self, pred_id: str) -> tuple[str, str]:
        """Infer subject/object types from predicate constraints.

        If the predicate has constraints, use the first allowed type.
        Otherwise, fall back to default_entity_type.
        """
        constraints = self._predicate_constraints.get(pred_id)
        if not constraints:
            return self._default_entity_type, self._default_entity_type

        subj_types = constraints.get("subject_types", [])
        obj_types = constraints.get("object_types", [])
        subj_type = subj_types[0] if subj_types else self._default_entity_type
        obj_type = obj_types[0] if obj_types else self._default_entity_type
        return subj_type, obj_type

    def extract_and_ingest(
        self,
        text: str,
        db,
        source_type: str = "chat_extraction",
        source_id: str = "",
        method: str = "heuristic_extraction_v1",
        curator=None,
    ) -> ExtractionResult:
        """Extract claims from text and ingest into DB.

        Same interface as TextExtractor.extract_and_ingest().
        """
        result = self.extract(text, source_type, source_id, method)

        ingested_claims = []
        for claim_input in result.claims:
            if curator is not None:
                decision = curator.triage(claim_input)
                if decision == "skip":
                    result.rejected.append(
                        (ExtractedClaim(
                            subject=claim_input.subject[0],
                            subject_type=claim_input.subject[1],
                            predicate=claim_input.predicate[0],
                            object=claim_input.object[0],
                            object_type=claim_input.object[1],
                        ), "curator: skip"),
                    )
                    continue

            try:
                db.ingest(
                    subject=claim_input.subject,
                    predicate=claim_input.predicate,
                    object=claim_input.object,
                    provenance=claim_input.provenance,
                    confidence=claim_input.confidence,
                    payload=claim_input.payload,
                )
                ingested_claims.append(claim_input)
            except Exception as e:
                result.warnings.append(f"Ingestion failed: {e}")

        result.claims = ingested_claims
        return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple split on period/newline boundaries."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip()]
