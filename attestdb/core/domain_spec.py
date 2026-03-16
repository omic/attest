"""Domain vocabulary specification for entity naming conventions.

A DomainSpec describes how to resolve human-readable display names
for each entity type in a domain. Used by bulk loaders, the
DisplayNameResolver, and ingestion quality checks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EntityTypeSpec:
    """How to resolve display names for one entity type."""

    entity_type: str  # e.g., "gene"
    id_pattern: str  # regex matching opaque IDs, e.g., r"^Gene_\d+$"
    display_source: str  # "reference_file" | "source_field" | "identity"
    reference_file: str | None = None  # URL for downloadable reference
    field_mapping: dict | None = None  # column positions: {"entrez_id": 1, "symbol": 2}
    fallback: str = "identity"  # what to do if resolution fails

    def is_opaque(self, name: str) -> bool:
        """Check if a display name matches the opaque ID pattern."""
        return bool(re.match(self.id_pattern, name))


@dataclass
class DomainSpec:
    """Domain vocabulary specification.

    A JSON document describing a domain's entity types, predicate types,
    and naming conventions. Loaded once per domain.
    """

    name: str  # "bio", "sales", "security"
    version: str  # semver
    entity_types: list[EntityTypeSpec] = field(default_factory=list)
    predicate_types: list[str] = field(default_factory=list)
    predicate_constraints: dict | None = None  # {pred: {subject_types, object_types}}

    @classmethod
    def from_file(cls, path: Path | str) -> DomainSpec:
        """Load a DomainSpec from a JSON file."""
        path = Path(path)
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> DomainSpec:
        """Construct a DomainSpec from a dictionary."""
        entity_types = [
            EntityTypeSpec(
                entity_type=et["entity_type"],
                id_pattern=et.get("id_pattern", ".*"),
                display_source=et.get("display_source", "identity"),
                reference_file=et.get("reference_file"),
                field_mapping=et.get("field_mapping"),
                fallback=et.get("fallback", "identity"),
            )
            for et in data.get("entity_types", [])
        ]
        return cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            entity_types=entity_types,
            predicate_types=data.get("predicate_types", []),
            predicate_constraints=data.get("predicate_constraints"),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "entity_types": [
                {
                    "entity_type": et.entity_type,
                    "id_pattern": et.id_pattern,
                    "display_source": et.display_source,
                    **({"reference_file": et.reference_file} if et.reference_file else {}),
                    **({"field_mapping": et.field_mapping} if et.field_mapping else {}),
                    "fallback": et.fallback,
                }
                for et in self.entity_types
            ],
            "predicate_types": self.predicate_types,
            **({"predicate_constraints": self.predicate_constraints}
               if self.predicate_constraints else {}),
        }

    def get_entity_spec(self, entity_type: str) -> EntityTypeSpec | None:
        """Get the spec for a given entity type."""
        for et in self.entity_types:
            if et.entity_type == entity_type:
                return et
        return None

    def looks_opaque(self, name: str, entity_type: str | None = None) -> bool:
        """Check if a display name looks like an opaque ID.

        If entity_type is given, checks only that type's pattern.
        Otherwise checks all entity type patterns.
        """
        if entity_type:
            spec = self.get_entity_spec(entity_type)
            return spec.is_opaque(name) if spec else False
        return any(et.is_opaque(name) for et in self.entity_types)
