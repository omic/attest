"""Extraction pipeline — structured claim extraction from unstructured content.

Provides LLM-powered and rule-based extraction, entity linking against
an existing entity index, and claim normalization for ingestion.
"""

from __future__ import annotations

from attestdb.extraction.claim_normalizer import ClaimNormalizer
from attestdb.extraction.entity_linker import EntityLink, EntityLinker
from attestdb.extraction.prompt_templates import get_content_prompt, get_system_prompt
from attestdb.extraction.unstructured_pipeline import (
    BatchResult,
    ExtractedClaim,
    UnstructuredExtractor,
)

__all__ = [
    "BatchResult",
    "ClaimNormalizer",
    "EntityLink",
    "EntityLinker",
    "ExtractedClaim",
    "UnstructuredExtractor",
    "get_content_prompt",
    "get_system_prompt",
]
