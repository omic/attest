"""Entity linker — resolve extracted mention text to known entity IDs.

Resolution strategy (in order):
1. Exact alias match against the entity index (confidence 1.0)
2. Fuzzy search via the index's search method (confidence 0.7)
3. Context-based disambiguation when metadata provides owner context
4. Unresolved — flagged, not silently dropped (confidence 0.0)

Handles ambiguous references like "the account", "John's team",
"that big deal we discussed" via metadata context and pattern matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Patterns that indicate an ambiguous entity reference
_AMBIGUOUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:the|that|this)\s+(?:account|customer|client|company|deal)\b", re.IGNORECASE),
    re.compile(r"\b(?:the|that|this)\s+(?:big|main|key|important)\s+\w+\b", re.IGNORECASE),
    re.compile(r"\b\w+'s\s+(?:team|account|deal|project)\b", re.IGNORECASE),
    re.compile(r"\b(?:they|them|their)\b", re.IGNORECASE),
]


@dataclass
class EntityLink:
    """Result of resolving a single mention against the entity index."""

    mention_text: str
    resolved_entity_id: str | None
    confidence: float
    resolution_method: str  # "exact" | "fuzzy" | "context" | "unresolved"
    context_used: str = ""  # what context enabled resolution (if any)
    flagged: bool = False  # True when mention was unresolvable


class EntityLinker:
    """Resolve extracted entity mentions against an existing entity index.

    Parameters
    ----------
    entity_index:
        An object that supports ``.search(name, limit=N)`` returning results
        with ``entity_id`` and ``canonical_name`` attributes, and optionally
        ``aliases``.
    """

    def __init__(self, entity_index: object) -> None:
        self._index = entity_index

    def link(
        self,
        mentions: list[str],
        metadata: dict | None = None,
    ) -> list[EntityLink]:
        """Resolve a batch of mention strings to entity links."""
        return [self.link_single(m, metadata) for m in mentions]

    def link_single(
        self,
        mention: str,
        metadata: dict | None = None,
    ) -> EntityLink:
        """Resolve a single mention string.

        Tries exact → fuzzy → context → unresolved (flagged).
        """
        mention = mention.strip()
        if not mention:
            return EntityLink(
                mention_text=mention,
                resolved_entity_id=None,
                confidence=0.0,
                resolution_method="unresolved",
                flagged=True,
            )

        # Strategy 1: Exact alias match
        exact_id = self._try_exact(mention)
        if exact_id is not None:
            return EntityLink(
                mention_text=mention,
                resolved_entity_id=exact_id,
                confidence=1.0,
                resolution_method="exact",
            )

        # Strategy 2: Fuzzy search
        fuzzy_id = self._try_fuzzy(mention)
        if fuzzy_id is not None:
            return EntityLink(
                mention_text=mention,
                resolved_entity_id=fuzzy_id,
                confidence=0.7,
                resolution_method="fuzzy",
            )

        # Strategy 3: Context-based disambiguation
        if metadata:
            ctx_id, ctx_desc = self._try_context(mention, metadata)
            if ctx_id is not None:
                return EntityLink(
                    mention_text=mention,
                    resolved_entity_id=ctx_id,
                    confidence=0.6,
                    resolution_method="context",
                    context_used=ctx_desc,
                )

        # Strategy 4: Unresolved — flag it
        return EntityLink(
            mention_text=mention,
            resolved_entity_id=None,
            confidence=0.0,
            resolution_method="unresolved",
            flagged=True,
        )

    # ------------------------------------------------------------------
    # Resolution strategies
    # ------------------------------------------------------------------

    def _try_exact(self, mention: str) -> str | None:
        """Attempt exact alias match. Returns entity_id or None."""
        try:
            results = self._index.search(mention, limit=5)  # type: ignore[union-attr]
            if not results:
                return None
            top = results[0]
            entity_id = getattr(top, "entity_id", None)
            if entity_id is None:
                return None
            # Check canonical name
            canonical = getattr(top, "canonical_name", "")
            if canonical and canonical.lower() == mention.lower():
                return entity_id
            # Check aliases
            aliases = getattr(top, "aliases", set())
            for alias in aliases:
                if alias.lower() == mention.lower():
                    return entity_id
            return None
        except Exception as exc:
            logger.debug("Exact match failed for '%s': %s", mention, exc)
            return None

    def _try_fuzzy(self, mention: str) -> str | None:
        """Attempt fuzzy search. Returns entity_id of top result or None."""
        # Skip fuzzy for ambiguous references — they need context, not search
        if _is_ambiguous_reference(mention):
            return None
        try:
            results = self._index.search(mention, limit=1)  # type: ignore[union-attr]
            if results:
                return getattr(results[0], "entity_id", None)
        except Exception as exc:
            logger.debug("Fuzzy search failed for '%s': %s", mention, exc)
        return None

    def _try_context(self, mention: str, metadata: dict) -> tuple[str | None, str]:
        """Use metadata context to disambiguate an ambiguous mention.

        Supported context keys:
        - ``owner_entities``: list of entity IDs the author owns/manages
        - ``sender_role``: author's role (e.g. "csm", "ae")
        - ``recent_entities``: entities mentioned earlier in conversation

        Returns (entity_id, description) or (None, "").
        """
        # If mention is an ambiguous reference like "the account"
        if not _is_ambiguous_reference(mention):
            return None, ""

        # Try owner_entities first (CSM who owns Acme → "the account" = Acme)
        owner_entities = metadata.get("owner_entities", [])
        if len(owner_entities) == 1:
            return owner_entities[0], f"sole owned entity by {metadata.get('sender_role', 'author')}"

        # Try recent_entities (most recently mentioned entity in conversation)
        recent = metadata.get("recent_entities", [])
        if len(recent) == 1:
            return recent[0], "sole recently mentioned entity"

        return None, ""


def _is_ambiguous_reference(mention: str) -> bool:
    """Return True if the mention is an ambiguous pronoun-like reference."""
    for pattern in _AMBIGUOUS_PATTERNS:
        if pattern.search(mention):
            return True
    return False
