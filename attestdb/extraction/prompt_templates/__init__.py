"""Content-type-specific prompt templates for LLM claim extraction.

Each template module provides a versioned prompt with few-shot examples
and content-type-specific instructions.
"""

from __future__ import annotations

from attestdb.extraction.prompt_templates.email_template import PROMPT as _EMAIL
from attestdb.extraction.prompt_templates.slack_template import PROMPT as _SLACK
from attestdb.extraction.prompt_templates.document_template import PROMPT as _DOC
from attestdb.extraction.prompt_templates.meeting_notes_template import PROMPT as _MEETING

_SYSTEM_PROMPT = """\
You are a structured claim extractor. Given unstructured text, extract factual \
claims as (subject, predicate, object) triples.

Respond with EXACTLY a JSON array of objects. Each object must have:
- "subject": entity name (company, person, product, metric)
- "predicate": claim type using dotted taxonomy (e.g. "risk.escalation", \
"satisfaction.nps", "revenue.amount", "relationship.champion_change", \
"action.blocker", "timeline.delay", "commitment.deadline")
- "object": value (e.g. "at risk", "positive", "45", "$1.2M", "3 months delay")
- "confidence": float 0.0-1.0 (how certain you are this claim is present in the text)
- "source_snippet": the exact excerpt (max 200 chars) from which this claim was derived

Only extract claims explicitly supported by the text. Do not infer or speculate. \
Prefer specific predicate types over generic ones.\
"""

_CONTENT_PROMPTS: dict[str, str] = {
    "email": _EMAIL,
    "slack_message": _SLACK,
    "document": _DOC,
    "qbr": _DOC,  # QBRs use the document template
    "meeting_notes": _MEETING,
    "generic": (
        "Extract structured claims from this text. Look for: factual "
        "assertions, metrics, status indicators, relationships between "
        "entities, and any quantitative data."
    ),
}


def get_system_prompt() -> str:
    """Return the base system prompt for JSON array claim extraction."""
    return _SYSTEM_PROMPT


def get_content_prompt(
    content_type: str,
    content: str,
    entity_context: list[str] | None = None,
) -> str:
    """Return a content-type-specific user prompt.

    Parameters
    ----------
    content_type:
        One of ``"email"``, ``"slack_message"``, ``"document"``, ``"qbr"``,
        ``"meeting_notes"``, or ``"generic"``.
    content:
        The raw text to extract claims from.
    entity_context:
        Optional known entity names the LLM should prefer when resolving mentions.
    """
    instruction = _CONTENT_PROMPTS.get(content_type, _CONTENT_PROMPTS["generic"])
    parts = [instruction, "", content]
    if entity_context:
        parts.append(f"\nKnown entities: {', '.join(entity_context)}")
    return "\n".join(parts)
