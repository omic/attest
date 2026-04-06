"""Unified entity index — SQLite-backed registry of resolved entities."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class UnifiedEntity:
    entity_id: str
    canonical_name: str
    entity_type: str
    source_records: dict[str, str]  # source_id -> record_id
    aliases: set[str]
    confidence: float
    needs_review: bool
    active: bool
    tenant_id: str
    created_at: float
    updated_at: float


class EntityIndex:
    """SQLite-backed index of unified entities across source systems."""

    def __init__(self, db_path: str, tenant_id: str = "default") -> None:
        self.db_path = db_path
        self.tenant_id = tenant_id
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_records TEXT NOT NULL DEFAULT '{}',
                aliases TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                needs_review INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                tenant_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_tenant_type_name
            ON entities (tenant_id, entity_type, canonical_name)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_active
            ON entities (tenant_id, active)
        """)
        self._conn.commit()

    def create(
        self,
        name: str,
        entity_type: str,
        source_id: str,
        record_id: str,
        confidence: float = 1.0,
        needs_review: bool = False,
    ) -> UnifiedEntity:
        """Create a new unified entity."""
        now = time.time()
        entity_id = str(uuid.uuid4())
        source_records = {source_id: record_id}
        aliases: set[str] = {name}

        self._conn.execute(
            """
            INSERT INTO entities (
                entity_id, canonical_name, entity_type, source_records, aliases,
                confidence, needs_review, active, tenant_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                entity_id,
                name,
                entity_type,
                json.dumps(source_records),
                json.dumps(sorted(aliases)),
                confidence,
                int(needs_review),
                self.tenant_id,
                now,
                now,
            ),
        )
        self._conn.commit()
        log.debug("Created entity %s: %s (%s)", entity_id, name, entity_type)

        return UnifiedEntity(
            entity_id=entity_id,
            canonical_name=name,
            entity_type=entity_type,
            source_records=source_records,
            aliases=aliases,
            confidence=confidence,
            needs_review=needs_review,
            active=True,
            tenant_id=self.tenant_id,
            created_at=now,
            updated_at=now,
        )

    def resolve(self, source_id: str, record_id: str) -> UnifiedEntity | None:
        """Find entity by source record.

        Scans source_records JSON for a match. Returns None if not found.
        """
        rows = self._conn.execute(
            "SELECT * FROM entities WHERE tenant_id = ? AND active = 1",
            (self.tenant_id,),
        ).fetchall()

        for row in rows:
            sr = json.loads(row["source_records"])
            if sr.get(source_id) == record_id:
                return self._row_to_entity(row)

        return None

    def merge(self, entity_a_id: str, entity_b_id: str) -> UnifiedEntity:
        """Merge entity_b into entity_a.

        Combines source_records and aliases. Picks the canonical name from
        the entity with higher confidence.
        """
        a = self.get(entity_a_id)
        b = self.get(entity_b_id)
        if a is None:
            raise ValueError(f"Entity not found: {entity_a_id}")
        if b is None:
            raise ValueError(f"Entity not found: {entity_b_id}")

        # Merge source records
        merged_sources = dict(a.source_records)
        merged_sources.update(b.source_records)

        # Merge aliases
        merged_aliases = a.aliases | b.aliases

        # Pick canonical name from higher confidence entity
        canonical = a.canonical_name if a.confidence >= b.confidence else b.canonical_name
        merged_aliases.add(a.canonical_name)
        merged_aliases.add(b.canonical_name)

        # Use max confidence
        merged_confidence = max(a.confidence, b.confidence)

        now = time.time()
        self._conn.execute(
            """
            UPDATE entities SET
                canonical_name = ?,
                source_records = ?,
                aliases = ?,
                confidence = ?,
                updated_at = ?
            WHERE entity_id = ?
            """,
            (
                canonical,
                json.dumps(merged_sources),
                json.dumps(sorted(merged_aliases)),
                merged_confidence,
                now,
                entity_a_id,
            ),
        )

        # Deactivate entity_b
        self._conn.execute(
            "UPDATE entities SET active = 0, updated_at = ? WHERE entity_id = ?",
            (now, entity_b_id),
        )
        self._conn.commit()

        log.info("Merged entity %s into %s", entity_b_id, entity_a_id)

        return UnifiedEntity(
            entity_id=entity_a_id,
            canonical_name=canonical,
            entity_type=a.entity_type,
            source_records=merged_sources,
            aliases=merged_aliases,
            confidence=merged_confidence,
            needs_review=a.needs_review,
            active=True,
            tenant_id=self.tenant_id,
            created_at=a.created_at,
            updated_at=now,
        )

    def split(self, entity_id: str, record_ids_to_split: list[str]) -> UnifiedEntity:
        """Split source records off into a new entity.

        The specified record_ids (by source_id) are removed from the original
        entity and placed into a new entity. Returns the new entity.
        """
        original = self.get(entity_id)
        if original is None:
            raise ValueError(f"Entity not found: {entity_id}")

        # Partition source records
        remaining_sources: dict[str, str] = {}
        split_sources: dict[str, str] = {}
        for source_id, record_id in original.source_records.items():
            if source_id in record_ids_to_split:
                split_sources[source_id] = record_id
            else:
                remaining_sources[source_id] = record_id

        if not split_sources:
            raise ValueError("No matching source records to split")

        now = time.time()

        # Update the original entity
        self._conn.execute(
            """
            UPDATE entities SET source_records = ?, updated_at = ?
            WHERE entity_id = ?
            """,
            (json.dumps(remaining_sources), now, entity_id),
        )

        # Create the new entity
        new_entity_id = str(uuid.uuid4())
        self._conn.execute(
            """
            INSERT INTO entities (
                entity_id, canonical_name, entity_type, source_records, aliases,
                confidence, needs_review, active, tenant_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)
            """,
            (
                new_entity_id,
                original.canonical_name,
                original.entity_type,
                json.dumps(split_sources),
                json.dumps(sorted(original.aliases)),
                original.confidence,
                self.tenant_id,
                now,
                now,
            ),
        )
        self._conn.commit()

        log.info("Split %d records from entity %s into %s", len(split_sources), entity_id, new_entity_id)

        return UnifiedEntity(
            entity_id=new_entity_id,
            canonical_name=original.canonical_name,
            entity_type=original.entity_type,
            source_records=split_sources,
            aliases=set(original.aliases),
            confidence=original.confidence,
            needs_review=True,
            active=True,
            tenant_id=self.tenant_id,
            created_at=now,
            updated_at=now,
        )

    def update_canonical_name(self, entity_id: str, new_name: str) -> None:
        """Update the canonical name and add old name to aliases."""
        entity = self.get(entity_id)
        if entity is None:
            raise ValueError(f"Entity not found: {entity_id}")

        aliases = entity.aliases
        aliases.add(entity.canonical_name)
        aliases.add(new_name)

        now = time.time()
        self._conn.execute(
            """
            UPDATE entities SET canonical_name = ?, aliases = ?, updated_at = ?
            WHERE entity_id = ?
            """,
            (new_name, json.dumps(sorted(aliases)), now, entity_id),
        )
        self._conn.commit()

    def deactivate(self, entity_id: str) -> None:
        """Mark an entity as inactive."""
        now = time.time()
        self._conn.execute(
            "UPDATE entities SET active = 0, updated_at = ? WHERE entity_id = ?",
            (now, entity_id),
        )
        self._conn.commit()
        log.info("Deactivated entity %s", entity_id)

    def search(
        self,
        name: str,
        entity_type: str | None = None,
        limit: int = 20,
    ) -> list[UnifiedEntity]:
        """Case-insensitive LIKE search on canonical_name."""
        where_clauses = ["tenant_id = ?", "active = 1", "canonical_name LIKE ?"]
        params: list[str | int] = [self.tenant_id, f"%{name}%"]

        if entity_type is not None:
            where_clauses.append("entity_type = ?")
            params.append(entity_type)

        where = " AND ".join(where_clauses)
        params.append(limit)

        rows = self._conn.execute(
            f"""
            SELECT * FROM entities
            WHERE {where}
            ORDER BY confidence DESC, canonical_name
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [self._row_to_entity(r) for r in rows]

    def get(self, entity_id: str) -> UnifiedEntity | None:
        """Get entity by ID."""
        row = self._conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()

        if row is None:
            return None
        return self._row_to_entity(row)

    def count(self, entity_type: str | None = None, active_only: bool = True) -> int:
        """Count entities, optionally filtered."""
        where_clauses = ["tenant_id = ?"]
        params: list[str] = [self.tenant_id]

        if active_only:
            where_clauses.append("active = 1")
        if entity_type is not None:
            where_clauses.append("entity_type = ?")
            params.append(entity_type)

        where = " AND ".join(where_clauses)

        row = self._conn.execute(
            f"SELECT COUNT(*) AS cnt FROM entities WHERE {where}",
            params,
        ).fetchone()

        return row["cnt"] if row else 0

    def add_source_record(
        self,
        entity_id: str,
        source_id: str,
        record_id: str,
        alias: str | None = None,
    ) -> None:
        """Add a source record to an existing entity."""
        entity = self.get(entity_id)
        if entity is None:
            raise ValueError(f"Entity not found: {entity_id}")

        entity.source_records[source_id] = record_id
        if alias:
            entity.aliases.add(alias)

        now = time.time()
        self._conn.execute(
            """
            UPDATE entities SET source_records = ?, aliases = ?, updated_at = ?
            WHERE entity_id = ?
            """,
            (
                json.dumps(entity.source_records),
                json.dumps(sorted(entity.aliases)),
                now,
                entity_id,
            ),
        )
        self._conn.commit()

    def get_all_claims(self, entity_id: str, store: object) -> list:
        """Return all claims from the store for every source record of this entity.

        Parameters
        ----------
        entity_id:
            The unified entity to look up.
        store:
            Any object with a ``claims_for(entity, pred, source, min_conf, limit)``
            method (e.g. a ``RustStore`` or ``AttestDB``).

        Returns
        -------
        list
            Claim objects (dicts or dataclasses, depending on the store).
        """
        entity = self.get(entity_id)
        if entity is None:
            return []

        all_claims: list = []
        seen_ids: set[str] = set()
        for source_id, record_id in entity.source_records.items():
            try:
                claims = store.claims_for(record_id, None, source_id, 0.0, 0)  # type: ignore[union-attr]
            except Exception:
                # Also try with the canonical name / aliases
                claims = []
                try:
                    for alias in entity.aliases:
                        for claim in store.claims_for(alias, None, source_id, 0.0, 0):  # type: ignore[union-attr]
                            claims.append(claim)
                except Exception:
                    log.debug("Failed to fetch claims for %s:%s", source_id, record_id)
                    continue

            for claim in claims:
                cid = claim.get("claim_id") if isinstance(claim, dict) else getattr(claim, "claim_id", None)
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_claims.append(claim)

        return all_claims

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> UnifiedEntity:
        return UnifiedEntity(
            entity_id=row["entity_id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            source_records=json.loads(row["source_records"]),
            aliases=set(json.loads(row["aliases"])),
            confidence=row["confidence"],
            needs_review=bool(row["needs_review"]),
            active=bool(row["active"]),
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
