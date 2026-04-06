"""SchemaMap — persistent, versioned representation of a discovered schema.

Combines field profiles, semantic mappings, and deprecation flags into a
single serialisable artefact that can be saved/loaded and reviewed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from attestdb.discovery.analyzer import SemanticMapping
from attestdb.discovery.sampler import FieldProfile

logger = logging.getLogger(__name__)


@dataclass
class SchemaMap:
    """Full discovered schema for a data source."""

    source_id: str
    source_type: str
    tenant_id: str
    discovered_at: float = field(default_factory=time.time)
    field_profiles: list[FieldProfile] = field(default_factory=list)
    semantic_mappings: list[SemanticMapping] = field(default_factory=list)
    deprecated_fields: list[str] = field(default_factory=list)
    schema_version: int = 1

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "tenant_id": self.tenant_id,
            "discovered_at": self.discovered_at,
            "schema_version": self.schema_version,
            "deprecated_fields": list(self.deprecated_fields),
            "field_profiles": [
                {
                    "field_name": fp.field_name,
                    "parent_object": fp.parent_object,
                    "fill_rate": fp.fill_rate,
                    "type_distribution": dict(fp.type_distribution),
                    "cardinality": fp.cardinality,
                    "value_samples": list(fp.value_samples),
                    "statistical_summary": dict(fp.statistical_summary),
                    "is_pii": fp.is_pii,
                    "tenant_id": fp.tenant_id,
                }
                for fp in self.field_profiles
            ],
            "semantic_mappings": [
                {
                    "field_name": sm.field_name,
                    "semantic_type": sm.semantic_type,
                    "confidence": sm.confidence,
                    "reasoning": sm.reasoning,
                    "review_status": sm.review_status,
                    "tenant_id": sm.tenant_id,
                }
                for sm in self.semantic_mappings
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SchemaMap:
        """Reconstruct a SchemaMap from a dict (inverse of ``to_dict``)."""
        field_profiles = [
            FieldProfile(
                field_name=fp["field_name"],
                parent_object=fp["parent_object"],
                fill_rate=fp["fill_rate"],
                type_distribution=fp.get("type_distribution", {}),
                cardinality=fp.get("cardinality", 0),
                value_samples=fp.get("value_samples", []),
                statistical_summary=fp.get("statistical_summary", {}),
                is_pii=fp.get("is_pii", False),
                tenant_id=fp.get("tenant_id", "default"),
            )
            for fp in d.get("field_profiles", [])
        ]

        semantic_mappings = [
            SemanticMapping(
                field_name=sm["field_name"],
                semantic_type=sm["semantic_type"],
                confidence=sm["confidence"],
                reasoning=sm.get("reasoning", ""),
                review_status=sm.get("review_status", "unmapped"),
                tenant_id=sm.get("tenant_id", "default"),
            )
            for sm in d.get("semantic_mappings", [])
        ]

        return cls(
            source_id=d["source_id"],
            source_type=d["source_type"],
            tenant_id=d.get("tenant_id", "default"),
            discovered_at=d.get("discovered_at", 0.0),
            schema_version=d.get("schema_version", 1),
            field_profiles=field_profiles,
            semantic_mappings=semantic_mappings,
            deprecated_fields=d.get("deprecated_fields", []),
        )

    # ── Reporting ─────────────────────────────────────────────────────

    def to_review_report(self) -> str:
        """Generate a human-readable summary for review."""
        total = len(self.field_profiles)
        auto_mapped = sum(
            1 for m in self.semantic_mappings if m.review_status == "auto_mapped"
        )
        needs_review = sum(
            1 for m in self.semantic_mappings if m.review_status == "needs_review"
        )
        unmapped = sum(
            1 for m in self.semantic_mappings if m.review_status == "unmapped"
        )

        lines = [
            f"Schema Discovery Report: {self.source_id} ({self.source_type})",
            f"Tenant: {self.tenant_id}  |  Version: {self.schema_version}",
            "=" * 60,
            f"Total fields:    {total}",
            f"Auto-mapped:     {auto_mapped}",
            f"Needs review:    {needs_review}",
            f"Unmapped:        {unmapped}",
            f"Deprecated:      {len(self.deprecated_fields)}",
        ]

        if self.deprecated_fields:
            lines.append("")
            lines.append("Deprecated fields:")
            for fname in self.deprecated_fields:
                lines.append(f"  - {fname}")

        # Show sample mappings (up to 10)
        mapped = [m for m in self.semantic_mappings if m.semantic_type != "unknown"]
        if mapped:
            lines.append("")
            lines.append("Sample mappings:")
            for m in mapped[:10]:
                lines.append(
                    f"  {m.field_name} -> {m.semantic_type} "
                    f"(conf={m.confidence:.2f}, {m.review_status})"
                )
            if len(mapped) > 10:
                lines.append(f"  ... and {len(mapped) - 10} more")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write the schema map as JSON to *path*."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("SchemaMap saved to %s", path)

    @classmethod
    def load(cls, path: str) -> SchemaMap:
        """Read a schema map from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info("SchemaMap loaded from %s", path)
        return cls.from_dict(data)
