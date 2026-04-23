"""Unstructured extraction pipeline — LLM + rule-based claim extraction.

Extracts structured (subject, predicate, object) claims from free text
using the LLM provider fallback chain, with rule-based pattern matching
as a fallback when no LLM is available.

Confidence is capped at 0.7 for all unstructured claims.

Supports batch processing with configurable concurrency and exponential
backoff for rate limit handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from attestdb.core.providers import EXTRACTION_FALLBACK_CHAIN, PROVIDERS
from attestdb.core.providers import load_env_file as _load_env_file
from attestdb.extraction.entity_linker import EntityLinker
from attestdb.extraction.prompt_templates import get_content_prompt, get_system_prompt

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_MAX_UNSTRUCTURED_CONFIDENCE = 0.7
_DEFAULT_CONFIDENCE = 0.4
_SNIPPET_MAX_LEN = 200
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


# ── Dataclass ────────────────────────────────────────────────────────


@dataclass
class ExtractedClaim:
    """A claim extracted from unstructured content."""

    subject: str
    predicate: str
    object: str
    confidence: float
    source_snippet: str
    content_type: str
    linked_entity_id: str | None = None
    raw_type: str = ""


@dataclass
class BatchResult:
    """Result of batch extraction across multiple documents."""

    claims: list[ExtractedClaim] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    documents_processed: int = 0


# ── Rule-based extraction patterns ───────────────────────────────────

_RULE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"(?:is\s+)?at\s+risk", re.IGNORECASE), "risk.escalation", "at risk"),
    (re.compile(r"risk\s+of\s+churn", re.IGNORECASE), "risk.escalation", "churn risk"),
    (re.compile(r"escalat(?:ed|ing|ion)", re.IGNORECASE), "risk.escalation", "escalated"),
    (re.compile(r"block(?:ed|er|ing)", re.IGNORECASE), "action.blocker", "blocked"),
    (re.compile(r"NPS\s+(?:score\s+)?(?:of\s+)?(?:dropped\s+to\s+)?(\d+)", re.IGNORECASE), "satisfaction.nps", ""),
    (re.compile(r"CSAT\s+(?:score\s+)?(?:of\s+)?(\d+)", re.IGNORECASE), "satisfaction.csat", ""),
    (re.compile(r"health\s*score(?:\s+of)?\s*:?\s*(\d+)", re.IGNORECASE), "satisfaction.health", ""),
    (re.compile(r"churn(?:ed|ing)?", re.IGNORECASE), "churn.status", "churned"),
    (re.compile(r"renew(?:ed|al|ing)", re.IGNORECASE), "renewal.status", "renewed"),
    (re.compile(r"(?:champion|key\s+contact)\s+(?:left|departed|leaving|resigned|quit)", re.IGNORECASE),
     "relationship.champion_change", "champion departed"),
    (re.compile(r"(?:the\s+)?champion\s+(?:has\s+)?left", re.IGNORECASE),
     "relationship.champion_change", "champion departed"),
    (re.compile(r"push(?:ed)?\s+back\b.*?(\d+\s+(?:month|week|day)s?)", re.IGNORECASE), "timeline.delay", ""),
    (re.compile(r"delay(?:ed)?\s+(?:by\s+)?(\d+\s+(?:month|week|day)s?)", re.IGNORECASE), "timeline.delay", ""),
    (re.compile(r"go-?live\s+(?:delayed|pushed|moved)", re.IGNORECASE), "timeline.delay", "go-live delayed"),
    (re.compile(r"(?:very\s+)?(?:un)?happy|dissatisfied|frustrated|furious", re.IGNORECASE),
     "satisfaction.sentiment", "negative"),
    (re.compile(r"(?:very\s+)?(?:happy|satisfied|pleased|delighted)", re.IGNORECASE),
     "satisfaction.sentiment", "positive"),
    (re.compile(r"\$[\d,.]+[MmKkBb]?", re.IGNORECASE), "revenue.amount", ""),
]

_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+(?:Corp|Inc|LLC|Ltd|Co|Account|account))?)"
        r"\s+(?:account\s+)?(?:is|has|was|are)",
        re.UNICODE,
    ),
    re.compile(
        r"(?:customer|account|client|company)[:\s]+([A-Z][A-Za-z0-9\s]+?)"
        r"(?:\s+(?:is|has|was|\u2014|-|\n))",
        re.IGNORECASE,
    ),
    # "about Acme's renewal" / "Acme Corp NPS" / "Acme just pushed back"
    re.compile(
        r"(?:about|for|from|—)\s+([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)?)"
        r"(?:'s|\s+(?:NPS|CSAT|health|just|account))",
        re.UNICODE,
    ),
    # "the Acme Corp champion/account/team"
    re.compile(
        r"(?:the\s+)?([A-Z][A-Za-z0-9]+(?:\s+(?:Corp|Inc|LLC|Ltd))?)"
        r"\s+(?:champion|account|team|contact|rep)",
        re.UNICODE,
    ),
    # Document headers: "Review — TechStart Inc" or "Report: Acme Corp"
    re.compile(
        r"(?:review|report|summary|brief)\s*[—\-:]\s*"
        r"([A-Z][A-Za-z0-9]+(?:\s+[A-Z]?[A-Za-z0-9]+)*)",
        re.IGNORECASE,
    ),
]


# ── Main class ───────────────────────────────────────────────────────


class UnstructuredExtractor:
    """Extract structured claims from unstructured text.

    Parameters
    ----------
    entity_index:
        Optional entity index (must have ``.search(name, limit=N)`` method)
        for linking extracted mentions to known entities.
    prediction_log:
        Optional ``PredictionLog`` instance for recording extraction decisions.
    """

    def __init__(
        self,
        entity_index: object | None = None,
        prediction_log: object | None = None,
        max_cost_usd: float = 0.0,
    ) -> None:
        self._entity_index = entity_index
        self._linker = EntityLinker(entity_index) if entity_index is not None else None
        self._prediction_log = prediction_log
        self._max_cost_usd = max_cost_usd  # 0 = unbounded
        self._total_cost_usd = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._last_reported_dollar = 0  # for $N increment logging
        self._budget_exceeded = False

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost_usd, 4)

    @property
    def budget_exceeded(self) -> bool:
        return self._budget_exceeded

    def usage_summary(self) -> dict:
        return {
            "cost_usd": round(self._total_cost_usd, 4),
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "max_cost_usd": self._max_cost_usd,
            "budget_exceeded": self._budget_exceeded,
        }

    def extract_claims(
        self,
        content: str,
        content_type: str,
        metadata: dict | None = None,
    ) -> list[ExtractedClaim]:
        """Extract structured claims from *content*.

        Parameters
        ----------
        content:
            Raw text to extract claims from.
        content_type:
            ``"email"`` | ``"slack_message"`` | ``"document"`` |
            ``"qbr"`` | ``"meeting_notes"`` | ``"generic"``.
        metadata:
            Optional context: ``author``, ``timestamp``, ``recipients``,
            ``channel``, ``subject``, ``entity_context``,
            ``owner_entities``, ``sender_role``.
        """
        if not content or not content.strip():
            return []

        metadata = metadata or {}

        # Budget check — once exceeded, fall back to rule-based for free
        if self._budget_exceeded:
            return self._extract_rule_based(content, content_type)

        # Try LLM extraction first
        client, model = _get_llm_client()
        if client is not None:
            claims = self._extract_with_llm(client, model, content, content_type, metadata)
        else:
            claims = self._extract_rule_based(content, content_type)

        # Cap confidence
        for claim in claims:
            claim.confidence = min(claim.confidence, _MAX_UNSTRUCTURED_CONFIDENCE)

        # Entity linking (pass metadata for context disambiguation)
        if self._linker is not None:
            for claim in claims:
                link = self._linker.link_single(claim.subject, metadata)
                if link.resolved_entity_id is not None:
                    claim.linked_entity_id = link.resolved_entity_id

        return claims

    def batch_extract(
        self,
        documents: list[dict],
        max_concurrency: int = 5,
    ) -> BatchResult:
        """Extract claims from multiple documents in parallel.

        Parameters
        ----------
        documents:
            Each dict must have ``content`` and ``content_type``.
            Optional: ``metadata``, ``source_id``.
        max_concurrency:
            Maximum parallel LLM calls.

        Returns
        -------
        BatchResult with all claims, errors, and timing.
        """
        result = BatchResult()
        t0 = time.monotonic()

        def _process_one(doc: dict) -> list[ExtractedClaim]:
            return self.extract_claims(
                content=doc["content"],
                content_type=doc["content_type"],
                metadata=doc.get("metadata"),
            )

        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {pool.submit(_process_one, doc): i for i, doc in enumerate(documents)}
            for future in as_completed(futures):
                try:
                    claims = future.result()
                    result.claims.extend(claims)
                    result.documents_processed += 1
                except Exception as exc:
                    idx = futures[future]
                    result.errors.append(f"Document {idx}: {exc}")

        result.elapsed = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _extract_with_llm(
        self,
        client: object,
        model: str,
        content: str,
        content_type: str,
        metadata: dict,
    ) -> list[ExtractedClaim]:
        entity_context = metadata.get("entity_context")

        system_prompt = get_system_prompt()
        user_prompt = get_content_prompt(content_type, content, entity_context)

        # Add metadata context to prompt when available
        meta_parts: list[str] = []
        if metadata.get("author"):
            meta_parts.append(f"Author: {metadata['author']}")
        if metadata.get("subject"):
            meta_parts.append(f"Subject: {metadata['subject']}")
        if metadata.get("channel"):
            meta_parts.append(f"Channel: {metadata['channel']}")
        if metadata.get("recipients"):
            meta_parts.append(f"Recipients: {', '.join(metadata['recipients'])}")
        if meta_parts:
            user_prompt += "\n\nContext:\n" + "\n".join(meta_parts)

        # Retry with exponential backoff
        for attempt in range(_MAX_RETRIES):
            try:
                response = client.chat.completions.create(  # type: ignore[union-attr]
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1024,
                    temperature=0.1,
                )
                raw_content = response.choices[0].message.content or ""
                self._record_usage(response, model)
                return self._parse_llm_response(raw_content, content_type)
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    backoff = _BASE_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "LLM extraction attempt %d failed: %s; retrying in %.1fs",
                        attempt + 1, exc, backoff,
                    )
                    time.sleep(backoff)
                else:
                    logger.warning("LLM extraction failed after %d attempts: %s", _MAX_RETRIES, exc)
                    return self._extract_rule_based(content, content_type)

        return []  # unreachable but satisfies type checker

    def _record_usage(self, response: object, model: str) -> None:
        """Track cumulative cost from an OpenAI-compatible response. Trip budget if over."""
        from attestdb.core.providers import estimate_cost

        usage = getattr(response, "usage", None)
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = estimate_cost(model, prompt, completion)

        self._total_prompt_tokens += prompt
        self._total_completion_tokens += completion
        self._total_cost_usd += cost

        # Dollar-increment logging (every whole $1 passed)
        current_dollar = int(self._total_cost_usd)
        if current_dollar > self._last_reported_dollar:
            logger.info(
                "LLM extraction spend: $%.2f (%d prompt + %d completion tokens)",
                self._total_cost_usd, self._total_prompt_tokens, self._total_completion_tokens,
            )
            self._last_reported_dollar = current_dollar

        # Budget trip
        if self._max_cost_usd > 0 and self._total_cost_usd >= self._max_cost_usd:
            if not self._budget_exceeded:
                logger.warning(
                    "LLM extraction budget hit: $%.2f >= $%.2f — falling back to rule-based",
                    self._total_cost_usd, self._max_cost_usd,
                )
                self._budget_exceeded = True

    def _parse_llm_response(
        self,
        content: str,
        content_type: str,
    ) -> list[ExtractedClaim]:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            content = "\n".join(lines)

        bracket_start = content.find("[")
        bracket_end = content.rfind("]")
        if bracket_start == -1 or bracket_end == -1:
            return []

        try:
            items = json.loads(content[bracket_start : bracket_end + 1])
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            return []

        claims: list[ExtractedClaim] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            if not subject or not predicate or not obj:
                continue

            confidence = _DEFAULT_CONFIDENCE
            try:
                confidence = min(float(item.get("confidence", _DEFAULT_CONFIDENCE)),
                                 _MAX_UNSTRUCTURED_CONFIDENCE)
            except (TypeError, ValueError):
                pass

            claims.append(ExtractedClaim(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=confidence,
                source_snippet=str(item.get("source_snippet", ""))[:_SNIPPET_MAX_LEN],
                content_type=content_type,
                raw_type=str(item.get("raw_type", content_type)),
            ))

        return claims

    # ------------------------------------------------------------------
    # Rule-based extraction
    # ------------------------------------------------------------------

    def _extract_rule_based(
        self,
        content: str,
        content_type: str,
    ) -> list[ExtractedClaim]:
        claims: list[ExtractedClaim] = []
        sentences = re.split(r"[.!?\n]+", content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue

            for pattern, predicate, default_obj in _RULE_PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue

                if match.groups():
                    obj = match.group(1)
                elif default_obj:
                    obj = default_obj
                else:
                    obj = match.group(0)

                subject = _extract_subject(sentence, content)
                if not subject:
                    continue

                claims.append(ExtractedClaim(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence=_DEFAULT_CONFIDENCE,
                    source_snippet=sentence[:_SNIPPET_MAX_LEN],
                    content_type=content_type,
                    raw_type=content_type,
                ))
                break  # one claim per sentence

        return claims


# ── Helpers ──────────────────────────────────────────────────────────


def _get_llm_client() -> tuple[object | None, str]:
    """Get an LLM client via the intelligence layer, if installed.

    Falls back to ``(None, "")`` — which the extractor handles by
    routing to its rule-based path — when ``attestdb.intelligence`` is
    not available or no provider is configured.
    """
    try:
        from attestdb.intelligence.llm_client import get_llm_client
    except ImportError:
        return None, ""
    return get_llm_client()


def _extract_subject(sentence: str, full_content: str = "") -> str:
    """Try to extract an entity name from a sentence using regex patterns."""
    # Try sentence-level patterns first
    for pattern in _ENTITY_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return match.group(1).strip()

    # Fall back to full content scan for entity names
    if full_content and full_content != sentence:
        for pattern in _ENTITY_PATTERNS:
            match = pattern.search(full_content)
            if match:
                return match.group(1).strip()

    return ""
