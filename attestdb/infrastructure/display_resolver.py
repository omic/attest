"""Display name resolver — maps opaque entity IDs to human-readable names.

Deterministic: downloads reference files once, then pure dict lookups.
No LLM calls, no per-claim cost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from attestdb.core.domain_spec import DomainSpec, EntityTypeSpec

logger = logging.getLogger(__name__)


class DisplayNameResolver:
    """Resolve opaque entity IDs to human-readable display names.

    Uses a DomainSpec to determine which entity types need resolution
    and how to resolve them (reference file download, source field, etc.).

    Example::

        from attestdb.core.domain_specs import load_spec
        spec = load_spec("bio")
        resolver = DisplayNameResolver(spec, cache_dir="/tmp/attestdb_cache")
        resolver.load()
        resolver.resolve("Gene_1017", "gene")  # -> "CDK2"
    """

    def __init__(self, domain_spec: DomainSpec, cache_dir: str | Path):
        self._spec = domain_spec
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Per-entity-type lookup tables: {entity_type: {opaque_id: display_name}}
        self._lookups: dict[str, dict[str, str]] = {}

    def load(self) -> None:
        """Download reference files and build lookup tables.

        Only downloads files for entity types with display_source="reference_file".
        """
        for et_spec in self._spec.entity_types:
            if et_spec.display_source == "reference_file" and et_spec.reference_file:
                self._load_reference(et_spec)

    def _load_reference(self, et_spec: EntityTypeSpec) -> None:
        """Load a reference file for an entity type."""
        ref = et_spec.reference_file
        if not ref:
            return

        if ref.startswith("ncbi://gene_info"):
            self._load_ncbi_gene_info(et_spec)
        else:
            logger.debug(
                "Unsupported reference_file scheme for %s: %s",
                et_spec.entity_type, ref,
            )

    def _load_ncbi_gene_info(self, et_spec: EntityTypeSpec) -> None:
        """Load NCBI gene_info for gene symbol resolution.

        Delegates to GeneIDMapper.load_gene_info() for the actual download,
        then builds the lookup table.
        """
        from attestdb.infrastructure.id_mapper import GeneIDMapper

        mapper = GeneIDMapper(str(self._cache_dir))
        mapper.load_gene_info()

        lookup = {}
        for entrez_id, symbol in mapper._entrez_to_symbol.items():  # noqa: SLF001
            lookup[f"Gene_{entrez_id}"] = symbol
        self._lookups[et_spec.entity_type] = lookup
        logger.info(
            "DisplayNameResolver: loaded %d gene symbols from NCBI gene_info",
            len(lookup),
        )

    def resolve(self, entity_id: str, entity_type: str) -> str:
        """Return human-readable display name, or entity_id if unresolvable.

        Pure dict lookup — no network calls after load().
        """
        lookup = self._lookups.get(entity_type)
        if lookup:
            resolved = lookup.get(entity_id)
            if resolved:
                return resolved
        return entity_id

    def is_opaque(self, name: str, entity_type: str | None = None) -> bool:
        """Check if a display name matches an opaque ID pattern.

        Delegates to the underlying DomainSpec.
        """
        return self._spec.looks_opaque(name, entity_type)

    def is_readable(self, name: str, entity_type: str | None = None) -> bool:
        """Check if a display name passes quality standards.

        A name is readable if it does NOT match an opaque ID pattern.
        """
        return not self.is_opaque(name, entity_type)

    @property
    def spec(self) -> DomainSpec:
        """The underlying domain specification."""
        return self._spec

    @property
    def loaded_types(self) -> list[str]:
        """Entity types with loaded lookup tables."""
        return list(self._lookups.keys())
