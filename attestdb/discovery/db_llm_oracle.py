"""LLM oracle for schema compiler — resolves ambiguous predicates and entity types.

Called only for low-confidence inferences (< 0.8). The rule engine handles
the deterministic 80-90%; this module resolves the ambiguous tail via
constrained classification over catalog candidates.

Three resolvers: TableTypeResolver, FkPredicateResolver, DisplayNameResolver.
Each accepts a compact context object and returns a structured decision.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Confidence threshold — only call LLM below this
LLM_THRESHOLD = 0.8


@dataclass
class LLMDecision:
    """Result from an LLM oracle call."""

    chosen: str
    confidence: float
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    needs_new: bool = False
    proposed_new: str | None = None
    source: str = "llm"  # "llm" or "rule" (passthrough)


def _get_client():
    """Get an OpenAI-compatible LLM client via the intelligence layer.

    Delegates to ``attestdb.intelligence.llm_client.get_llm_client`` so the
    SDK dependency lives on the enterprise side of the directory
    boundary. Returns ``(None, "")`` on OSS-only installs.
    """
    try:
        from attestdb.intelligence.llm_client import get_llm_client
    except ImportError:
        return None, ""
    return get_llm_client()


def _llm_call(client, model: str, system: str, user: str) -> dict | None:
    """Make a single LLM call, parse JSON response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON: %s", exc)
        return None
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None


# ======================================================================
# FK Predicate Resolver
# ======================================================================

_FK_SYSTEM_PROMPT = """You are a knowledge graph predicate selector. Given a foreign key relationship between two database tables, select the best predicate from the candidate list.

Rules:
- Choose from the provided candidates whenever possible.
- Only set needs_new_predicate to true if NO candidate fits the relationship semantics.
- Confidence should reflect how well the predicate captures the relationship meaning.
- Respond with valid JSON only, no other text."""

_FK_USER_TEMPLATE = """FK: {source_table}.{fk_column} -> {target_table}.{target_column}

Source table "{source_table}" ({source_type}):
  Columns: {source_columns}
  Sample: {source_sample}

Target table "{target_table}" ({target_type}):
  Columns: {target_columns}
  Sample: {target_sample}

Rule engine top guess: {rule_guess} (confidence: {rule_confidence})

Candidate predicates: {candidates}

Respond with JSON:
{{"predicate": "chosen_predicate", "confidence": 0.0-1.0, "reasoning": "one sentence", "alternatives": ["alt1", "alt2"], "needs_new_predicate": false}}"""


def resolve_fk_predicate(
    source_table: str,
    target_table: str,
    fk_column: str,
    target_column: str,
    source_type: str,
    target_type: str,
    source_columns: list[str],
    target_columns: list[str],
    source_sample: dict | None,
    target_sample: dict | None,
    rule_guess: str,
    rule_confidence: float,
    candidates: list[str],
) -> LLMDecision:
    """Resolve an ambiguous FK predicate via LLM."""
    if rule_confidence >= LLM_THRESHOLD:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="rule engine confident",
            source="rule",
        )

    client, model = _get_client()
    if not client:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="no LLM available, using rule engine guess",
            source="rule",
        )

    # Build compact sample strings
    src_sample_str = _compact_sample(source_sample) if source_sample else "N/A"
    tgt_sample_str = _compact_sample(target_sample) if target_sample else "N/A"

    user_msg = _FK_USER_TEMPLATE.format(
        source_table=source_table,
        target_table=target_table,
        fk_column=fk_column,
        target_column=target_column,
        source_type=source_type,
        target_type=target_type,
        source_columns=", ".join(source_columns[:15]),
        target_columns=", ".join(target_columns[:15]),
        source_sample=src_sample_str,
        target_sample=tgt_sample_str,
        rule_guess=rule_guess,
        rule_confidence=rule_confidence,
        candidates=", ".join(candidates[:10]),
    )

    result = _llm_call(client, model, _FK_SYSTEM_PROMPT, user_msg)
    if not result:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="LLM call failed, using rule engine guess",
            source="rule",
        )

    chosen = result.get("predicate", rule_guess)
    llm_conf = float(result.get("confidence", 0.5))

    # Fuse confidence: Bayesian combination
    fused = _fuse_confidence(rule_confidence, llm_conf, chosen == rule_guess)

    return LLMDecision(
        chosen=chosen,
        confidence=fused,
        reasoning=result.get("reasoning", ""),
        alternatives=result.get("alternatives", []),
        needs_new=result.get("needs_new_predicate", False),
        proposed_new=chosen if result.get("needs_new_predicate") else None,
        source="llm",
    )


# ======================================================================
# Table Type Resolver
# ======================================================================

_TYPE_SYSTEM_PROMPT = """You are a database table classifier. Given a table's structure, classify it into one of the provided entity types.

Rules:
- Choose from the provided type list whenever possible.
- Only set needs_new_type to true if NO existing type fits.
- Use FK neighbors to disambiguate ambiguous names (e.g., "accounts" could be CRM, GL, or auth).
- Respond with valid JSON only, no other text."""

_TYPE_USER_TEMPLATE = """Table: {table_name}
Columns: {columns}
FK neighbors: {fk_neighbors}
Row count: {row_count}
Sample row: {sample}

Rule engine guess: {rule_guess} (confidence: {rule_confidence})

Available types: {type_list}

Respond with JSON:
{{"type": "chosen_type", "confidence": 0.0-1.0, "reasoning": "one sentence", "rejected": ["type1", "type2"], "needs_new_type": false}}"""


def resolve_table_type(
    table_name: str,
    columns: list[str],
    column_types: list[str],
    fk_neighbors: list[str],
    row_count: int,
    sample_row: dict | None,
    rule_guess: str,
    rule_confidence: float,
    available_types: list[str] | None = None,
) -> LLMDecision:
    """Resolve an ambiguous table entity type via LLM."""
    if rule_confidence >= LLM_THRESHOLD:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="rule engine confident",
            source="rule",
        )

    client, model = _get_client()
    if not client:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="no LLM available",
            source="rule",
        )

    if available_types is None:
        available_types = [
            "person", "company", "organization", "project", "invoice",
            "product", "ticket", "collection", "media", "category",
            "capability", "entity",
        ]

    col_desc = ", ".join(
        f"{c} ({t})" for c, t in zip(columns[:20], column_types[:20])
    )
    sample_str = _compact_sample(sample_row) if sample_row else "N/A"

    user_msg = _TYPE_USER_TEMPLATE.format(
        table_name=table_name,
        columns=col_desc,
        fk_neighbors=", ".join(fk_neighbors[:10]),
        row_count=row_count,
        sample=sample_str,
        rule_guess=rule_guess,
        rule_confidence=rule_confidence,
        type_list=", ".join(available_types),
    )

    result = _llm_call(client, model, _TYPE_SYSTEM_PROMPT, user_msg)
    if not result:
        return LLMDecision(
            chosen=rule_guess,
            confidence=rule_confidence,
            reasoning="LLM call failed",
            source="rule",
        )

    chosen = result.get("type", rule_guess)
    llm_conf = float(result.get("confidence", 0.5))
    fused = _fuse_confidence(rule_confidence, llm_conf, chosen == rule_guess)

    return LLMDecision(
        chosen=chosen,
        confidence=fused,
        reasoning=result.get("reasoning", ""),
        alternatives=result.get("rejected", []),
        needs_new=result.get("needs_new_type", False),
        proposed_new=chosen if result.get("needs_new_type") else None,
        source="llm",
    )


# ======================================================================
# Display Name Resolver
# ======================================================================

_DISPLAY_SYSTEM_PROMPT = """You are a database display name selector. Given a table's columns, choose the best column(s) to use as the human-readable display name for entities in this table.

Rules:
- Choose from actual column names only.
- For person tables, prefer FirstName + LastName concatenation.
- For tables without obvious name columns, choose the best business identifier.
- Respond with valid JSON only, no other text."""

_DISPLAY_USER_TEMPLATE = """Table: {table_name} ({entity_type})
Columns: {columns}
Sample row: {sample}

Rule engine guess: {rule_guess}

Respond with JSON:
{{"display_columns": ["col1", "col2"], "template": "{{col1}} {{col2}}", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""


def resolve_display_name(
    table_name: str,
    entity_type: str,
    columns: list[str],
    column_types: list[str],
    sample_row: dict | None,
    rule_guess: list[str],
    rule_confidence: float,
) -> LLMDecision:
    """Resolve an ambiguous display column choice via LLM."""
    if rule_confidence >= LLM_THRESHOLD:
        return LLMDecision(
            chosen="|".join(rule_guess),
            confidence=rule_confidence,
            reasoning="rule engine confident",
            source="rule",
        )

    client, model = _get_client()
    if not client:
        return LLMDecision(
            chosen="|".join(rule_guess),
            confidence=rule_confidence,
            reasoning="no LLM available",
            source="rule",
        )

    col_desc = ", ".join(
        f"{c} ({t})" for c, t in zip(columns[:20], column_types[:20])
    )
    sample_str = _compact_sample(sample_row) if sample_row else "N/A"

    user_msg = _DISPLAY_USER_TEMPLATE.format(
        table_name=table_name,
        entity_type=entity_type,
        columns=col_desc,
        sample=sample_str,
        rule_guess=rule_guess,
    )

    result = _llm_call(client, model, _DISPLAY_SYSTEM_PROMPT, user_msg)
    if not result:
        return LLMDecision(
            chosen="|".join(rule_guess),
            confidence=rule_confidence,
            reasoning="LLM call failed",
            source="rule",
        )

    display_cols = result.get("display_columns", rule_guess)
    llm_conf = float(result.get("confidence", 0.5))

    # Validate columns exist
    valid_cols = [c for c in display_cols if c in columns]
    if not valid_cols:
        valid_cols = rule_guess

    return LLMDecision(
        chosen="|".join(valid_cols),
        confidence=llm_conf,
        reasoning=result.get("reasoning", ""),
        source="llm",
    )


# ======================================================================
# Helpers
# ======================================================================

def _compact_sample(row: dict | None) -> str:
    """Format a sample row compactly for prompts."""
    if not row:
        return "N/A"
    parts = []
    for k, v in list(row.items())[:10]:
        val = str(v)[:40] if v is not None else "NULL"
        parts.append(f"{k}={val}")
    return "{" + ", ".join(parts) + "}"


def _fuse_confidence(
    rule_conf: float, llm_conf: float, agreement: bool,
) -> float:
    """Bayesian fusion of rule and LLM confidence.

    If they agree, confidence rises. If they disagree, it drops.
    """
    if agreement:
        # Bayesian: C = (Cr * Cl) / (Cr * Cl + (1-Cr) * (1-Cl))
        numerator = rule_conf * llm_conf
        denominator = numerator + (1 - rule_conf) * (1 - llm_conf)
        return round(numerator / denominator, 3) if denominator > 0 else 0.5
    else:
        # Disagreement — use the higher one but penalize
        return round(max(rule_conf, llm_conf) * 0.7, 3)
