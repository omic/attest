"""Unstructured Source Pipeline — extract structured claims from emails, Slack, and documents.

Converts free-text from emails, Slack messages, QBR documents, and other
unstructured sources into structured (subject, predicate, object) claims
suitable for ingestion into AttestDB.

Uses LLM extraction via the provider fallback chain when available, with
rule-based pattern matching as a fallback.  Optionally resolves extracted
entity names against an EntityIndex for linking.

Confidence is intentionally lower than structured sources: default 0.4,
capped at 0.7 for all unstructured claims.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field

from attestdb.core.providers import EXTRACTION_FALLBACK_CHAIN, PROVIDERS
from attestdb.core.providers import load_env_file as _load_env_file

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_MAX_UNSTRUCTURED_CONFIDENCE = 0.7
_DEFAULT_CONFIDENCE = 0.4
_SNIPPET_MAX_LEN = 200

# ── Source-type prompts ──────────────────────────────────────────────

_SOURCE_PROMPTS: dict[str, str] = {
    "email": (
        "Extract business claims from this email. Look for: customer sentiment, "
        "risk signals, revenue mentions, relationship updates."
    ),
    "slack": (
        "Extract claims from this Slack message. Look for: escalations, blockers, "
        "customer feedback, action items."
    ),
    "document": (
        "Extract key business claims from this document. Look for: metrics, "
        "health scores, risk assessments, strategic plans."
    ),
    "qbr": (
        "Extract key business claims from this document. Look for: metrics, "
        "health scores, risk assessments, strategic plans."
    ),
    "generic": "Extract structured claims from this text.",
}

_SYSTEM_PROMPT = """\
You are a structured claim extractor. Given unstructured text, extract factual \
claims as (subject, predicate, object) triples.

Respond with EXACTLY a JSON array of objects. Each object must have:
- "subject": entity name (company, person, product, metric)
- "predicate": claim type (e.g. "risk.escalation", "satisfaction.sentiment", \
"satisfaction.nps", "revenue.amount", "relationship.status", "action.blocker")
- "object": value (e.g. "at risk", "positive", "45", "$1.2M")
- "confidence": float 0.0-1.0 (how certain you are this claim is present)
- "source_snippet": the exact excerpt (max 200 chars) from which this claim was derived

Example output:
[{"subject": "Acme Corp", "predicate": "risk.escalation", "object": "at risk", \
"confidence": 0.8, "source_snippet": "Acme account is at risk of churning"}]

Only extract claims that are explicitly supported by the text. Do not infer \
or speculate.\
"""


# ── Rule-based extraction patterns ───────────────────────────────────

_RULE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, predicate, default_object)
    (re.compile(r"(?:is\s+)?at\s+risk", re.IGNORECASE), "risk.escalation", "at risk"),
    (re.compile(r"risk\s+of\s+churn", re.IGNORECASE), "risk.escalation", "churn risk"),
    (re.compile(r"escalat(?:ed|ing|ion)", re.IGNORECASE), "risk.escalation", "escalated"),
    (re.compile(r"block(?:ed|er|ing)", re.IGNORECASE), "action.blocker", "blocked"),
    (re.compile(r"NPS\s+(?:score\s+)?(?:of\s+)?(\d+)", re.IGNORECASE), "satisfaction.nps", ""),
    (re.compile(r"CSAT\s+(?:score\s+)?(?:of\s+)?(\d+)", re.IGNORECASE), "satisfaction.csat", ""),
    (re.compile(r"health\s*score\s+(?:of\s+)?(\d+)", re.IGNORECASE), "satisfaction.health", ""),
    (re.compile(r"churn(?:ed|ing)?", re.IGNORECASE), "churn.status", "churned"),
    (re.compile(r"renew(?:ed|al|ing)", re.IGNORECASE), "renewal.status", "renewed"),
    (re.compile(r"(?:very\s+)?(?:un)?happy|dissatisfied|frustrated", re.IGNORECASE), "satisfaction.sentiment", "negative"),
    (re.compile(r"(?:very\s+)?(?:happy|satisfied|pleased|delighted)", re.IGNORECASE), "satisfaction.sentiment", "positive"),
    (re.compile(r"\$[\d,.]+[MmKkBb]?", re.IGNORECASE), "revenue.amount", ""),
]

# Patterns for extracting entity names from sentences.
_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    # "the Acme account" / "Acme Corp" / "Acme"
    re.compile(r"(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+(?:Corp|Inc|LLC|Ltd|Co|Account|account))?)\s+(?:account\s+)?(?:is|has|was|are)", re.UNICODE),
    # "Customer: Acme" or "Account: Acme Corp"
    re.compile(r"(?:customer|account|client|company)[:\s]+([A-Z][A-Za-z0-9\s]+?)(?:\s+(?:is|has|was|—|-|\n))", re.IGNORECASE),
]


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class UnstructuredSource:
    """An unstructured text source to extract claims from."""

    source_id: str
    source_type: str  # "email" | "slack" | "document" | "qbr" | "generic"
    content: str
    metadata: dict = field(default_factory=dict)  # sender, timestamp, channel, filename, etc.
    tenant_id: str = "default"


@dataclass
class ExtractionConfig:
    """Configuration for claim extraction from unstructured text."""

    source_type: str = "generic"
    entity_context: list[str] = field(default_factory=list)  # known entity names for linking
    claim_types: list[str] = field(default_factory=list)  # expected claim types
    default_confidence: float = _DEFAULT_CONFIDENCE


@dataclass
class UnstructuredClaim:
    """A claim extracted from unstructured text."""

    subject: str  # entity name extracted from text
    predicate: str  # claim type
    object: str  # value
    confidence: float
    source_snippet: str  # text excerpt (max 200 chars)
    linked_entity_id: str | None = None  # resolved entity ID, if matched


# ── Public API ───────────────────────────────────────────────────────


def extract_claims(
    source: UnstructuredSource,
    config: ExtractionConfig | None = None,
    entity_index: object | None = None,
) -> list[UnstructuredClaim]:
    """Extract structured claims from an unstructured source.

    Tries LLM extraction first (using the provider fallback chain).
    Falls back to rule-based extraction when no LLM is available.

    If *entity_index* is provided (must have a ``.search(name)`` method),
    each claim's subject is matched against the index to populate
    ``linked_entity_id``.
    """
    if config is None:
        config = ExtractionConfig(source_type=source.source_type)

    # Try LLM extraction first
    client, model = _get_llm_client()
    if client is not None:
        claims = _extract_with_llm(client, model, source, config)
    else:
        claims = _extract_rule_based(source, config)

    # Cap confidence for unstructured sources
    for claim in claims:
        claim.confidence = min(claim.confidence, _MAX_UNSTRUCTURED_CONFIDENCE)

    # Attempt entity linking
    if entity_index is not None:
        for claim in claims:
            claim.linked_entity_id = _link_entity(claim.subject, entity_index)

    return claims


def claims_to_claim_inputs(
    claims: list[UnstructuredClaim],
    source_id: str,
) -> list[dict]:
    """Convert UnstructuredClaim list to ClaimInput-compatible dicts.

    Each dict matches the fields expected by ``ClaimInput``, with
    provenance recording ``source_type="unstructured"`` and the
    given *source_id*.
    """
    results: list[dict] = []
    for claim in claims:
        results.append({
            "subject": (claim.subject, "entity"),
            "predicate": (claim.predicate, "unstructured_extraction"),
            "object": (claim.object, "value"),
            "confidence": claim.confidence,
            "provenance": {
                "source_type": "unstructured",
                "source_id": source_id,
                "method": "unstructured_extraction",
                "labels": {
                    "source_snippet": claim.source_snippet[:_SNIPPET_MAX_LEN],
                    **({"linked_entity_id": claim.linked_entity_id}
                       if claim.linked_entity_id else {}),
                },
            },
        })
    return results


# ── LLM extraction ──────────────────────────────────────────────────


def _get_llm_client() -> tuple[object | None, str]:
    """Walk the extraction fallback chain and return the first available client.

    Returns ``(client, model_name)`` or ``(None, "")`` if no provider is
    available.
    """
    for provider_name in EXTRACTION_FALLBACK_CHAIN:
        provider = PROVIDERS.get(provider_name)
        if not provider:
            continue

        api_key = os.environ.get(provider["env_key"])
        if not api_key:
            env_vars = _load_env_file(".env")
            api_key = env_vars.get(provider["env_key"])

        if not api_key:
            continue

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=provider["base_url"],
            )
            model = provider["default_model"]
            logger.info(
                "Unstructured extractor using provider=%s, model=%s",
                provider_name, model,
            )
            return client, model
        except ImportError:
            logger.debug("openai package not installed")
            return None, ""
        except Exception as exc:
            logger.warning("Failed to init %s client: %s", provider_name, exc)
            continue

    logger.info("No LLM provider available; using rule-based extraction")
    return None, ""


def _extract_with_llm(
    client: object,
    model: str,
    source: UnstructuredSource,
    config: ExtractionConfig,
) -> list[UnstructuredClaim]:
    """Extract claims using an LLM call."""
    source_prompt = _SOURCE_PROMPTS.get(
        config.source_type, _SOURCE_PROMPTS["generic"]
    )

    user_prompt_parts = [source_prompt, "", source.content]

    if config.entity_context:
        user_prompt_parts.append(
            f"\nKnown entities: {', '.join(config.entity_context)}"
        )
    if config.claim_types:
        user_prompt_parts.append(
            f"\nExpected claim types: {', '.join(config.claim_types)}"
        )

    user_prompt = "\n".join(user_prompt_parts)

    try:
        response = client.chat.completions.create(  # type: ignore[union-attr]
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        return _parse_llm_response(content, config)
    except Exception as exc:
        logger.warning("LLM extraction failed: %s; falling back to rules", exc)
        return _extract_rule_based(source, config)


def _parse_llm_response(
    content: str,
    config: ExtractionConfig,
) -> list[UnstructuredClaim]:
    """Parse the JSON array returned by the LLM."""
    # Strip markdown fences if present
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence line and closing fence
        lines = content.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        content = "\n".join(lines)

    # Try to find a JSON array
    bracket_start = content.find("[")
    bracket_end = content.rfind("]")
    if bracket_start == -1 or bracket_end == -1:
        logger.warning("No JSON array found in LLM response")
        return []

    json_str = content[bracket_start : bracket_end + 1]

    try:
        items = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON: %s", exc)
        return []

    if not isinstance(items, list):
        logger.warning("LLM response was not a JSON array")
        return []

    claims: list[UnstructuredClaim] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()

        if not subject or not predicate or not obj:
            continue

        confidence = config.default_confidence
        try:
            raw_conf = float(item.get("confidence", config.default_confidence))
            confidence = min(raw_conf, _MAX_UNSTRUCTURED_CONFIDENCE)
        except (TypeError, ValueError):
            pass

        snippet = str(item.get("source_snippet", ""))[:_SNIPPET_MAX_LEN]

        claims.append(UnstructuredClaim(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            source_snippet=snippet,
        ))

    return claims


# ── Rule-based extraction ────────────────────────────────────────────


def _extract_rule_based(
    source: UnstructuredSource,
    config: ExtractionConfig,
) -> list[UnstructuredClaim]:
    """Extract claims using regex patterns when no LLM is available."""
    claims: list[UnstructuredClaim] = []
    text = source.content

    # Split into sentences for snippet extraction
    sentences = re.split(r"[.!?\n]+", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 5:
            continue

        for pattern, predicate, default_obj in _RULE_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue

            # Determine the object value
            if match.groups():
                # Pattern has a capture group (e.g., NPS score number)
                obj = match.group(1)
            elif default_obj:
                obj = default_obj
            else:
                obj = match.group(0)

            # Try to extract the entity subject
            subject = _extract_subject(sentence, config.entity_context)
            if not subject:
                continue

            snippet = sentence[:_SNIPPET_MAX_LEN]

            claims.append(UnstructuredClaim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=config.default_confidence,
                source_snippet=snippet,
            ))
            # Only one claim per pattern per sentence
            break

    return claims


def _extract_subject(sentence: str, entity_context: list[str]) -> str:
    """Try to extract an entity name from a sentence.

    Checks entity_context names first (exact substring match),
    then falls back to regex patterns.
    """
    # Check known entity names first
    for entity in entity_context:
        if entity.lower() in sentence.lower():
            return entity

    # Try regex patterns
    for pattern in _ENTITY_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip()

    return ""


# ── Entity linking ───────────────────────────────────────────────────


def _link_entity(
    subject: str,
    entity_index: object,
) -> str | None:
    """Attempt to resolve a subject name to a UnifiedEntity ID.

    Returns the entity_id of the best match, or None if no match found.
    """
    try:
        results = entity_index.search(subject, limit=1)  # type: ignore[union-attr]
        if results:
            return results[0].entity_id
    except Exception as exc:
        logger.debug("Entity linking failed for '%s': %s", subject, exc)
    return None
