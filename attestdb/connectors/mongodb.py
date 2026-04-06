"""MongoDB connector — maps documents to claims via field mapping.

Reads documents from a MongoDB collection and yields claim dicts.  The
mapping specifies which document fields map to subject, predicate, and
object.  Supports dot notation for nested fields (e.g. ``"metadata.author"``).

Usage::

    from attestdb.connectors.mongodb import MongoDBConnector

    conn = MongoDBConnector(
        uri="mongodb://localhost:27017",
        database="knowledge",
        collection="interactions",
        mapping={
            "subject": "gene_name",
            "predicate": "relation_type",
            "object": "target_name",
            "subject_type": "gene",
            "object_type": "protein",
        },
        query={"organism": "human"},
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import StructuredConnector

logger = logging.getLogger(__name__)


class MongoDBConnector(StructuredConnector):
    """Import claims from a MongoDB collection with field mapping."""

    name = "mongodb"

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str,
        mapping: dict,
        query: dict | None = None,
        source_type: str = "mongodb",
        max_items: int = 10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.uri = uri
        self.database = database
        self.collection = collection
        self.mapping = mapping
        self.query = query or {}
        self.source_type = source_type
        self.max_items = max_items

    @staticmethod
    def _get_nested(doc: dict, dotted_key: str):
        """Resolve a dot-notation key against a nested document."""
        val = doc
        for part in dotted_key.split("."):
            val = val[part]
        return val

    def fetch(self) -> Iterator[dict]:
        try:
            import pymongo
        except ImportError:
            raise ImportError("pip install pymongo for MongoDB support")

        client = pymongo.MongoClient(self.uri)
        try:
            db = client[self.database]
            coll = db[self.collection]
            cursor = coll.find(self.query, batch_size=1000).limit(self.max_items)
            for doc in cursor:
                try:
                    yield self._doc_to_claim(doc)
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("mongodb: skipping document: %s", exc)
        finally:
            client.close()

    def _doc_to_claim(self, doc: dict) -> dict:
        m = self.mapping
        doc_id = str(doc.get("_id", ""))
        claim = {
            "subject": (
                str(self._get_nested(doc, m["subject"])),
                m.get("subject_type", "entity"),
            ),
            "predicate": (
                str(self._get_nested(doc, m["predicate"])),
                m.get("predicate_type", "relates_to"),
            ),
            "object": (
                str(self._get_nested(doc, m["object"])),
                m.get("object_type", "entity"),
            ),
            "provenance": {
                "source_type": self.source_type,
                "source_id": f"mongodb:{self.database}.{self.collection}:{doc_id}",
            },
        }
        if "confidence" in m:
            try:
                claim["confidence"] = float(self._get_nested(doc, m["confidence"]))
            except (KeyError, TypeError, ValueError):
                pass
        return claim
