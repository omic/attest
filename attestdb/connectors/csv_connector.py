"""CSV/TSV connector — maps rows to claims via column mapping.

Reads a local CSV or TSV file and yields claim dicts.  The mapping
specifies which columns map to subject, predicate, and object.

Usage::

    from attestdb.connectors.csv_connector import CSVConnector

    conn = CSVConnector(
        path="data/interactions.csv",
        mapping={
            "subject": "gene",
            "predicate": "relation",
            "object": "target",
            "subject_type": "gene",
            "object_type": "protein",
        },
    )
    result = conn.run(db)
"""

from __future__ import annotations

import csv
import logging
from typing import Iterator

from attestdb.connectors.base import QueryConnector

logger = logging.getLogger(__name__)


class CSVConnector(QueryConnector):
    """Import claims from a CSV or TSV file with column mapping."""

    name = "csv"

    def __init__(
        self,
        path: str,
        mapping: dict,
        delimiter: str = ",",
        source_type: str = "csv_import",
        confidence: float | None = None,
        **kwargs: object,
    ) -> None:
        # CSV doesn't use a query string — pass path as the query
        # for source_id generation in _row_to_claim.
        super().__init__(
            query=path, mapping=mapping,
            source_type=source_type, **kwargs,
        )
        self.path = path
        self.delimiter = delimiter
        self.default_confidence = confidence
        self._row_num = 0

    def _open_cursor(self) -> Iterator[dict]:
        self._row_num = 0
        with open(self.path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            for row in reader:
                self._row_num += 1
                yield row

    def _row_to_claim(self, row: dict) -> dict:
        """Override to add per-row source_id and default confidence."""
        m = self.mapping
        claim: dict = {
            "subject": (
                str(row[m["subject"]]),
                m.get("subject_type", "entity"),
            ),
            "predicate": (
                str(row[m["predicate"]]),
                m.get("predicate_type", "relates_to"),
            ),
            "object": (
                str(row[m["object"]]),
                m.get("object_type", "entity"),
            ),
            "provenance": {
                "source_type": self.source_type,
                "source_id": f"csv:{self.path}:row{self._row_num}",
            },
        }
        if "confidence" in m and m["confidence"] in row:
            claim["confidence"] = float(row[m["confidence"]])
        elif self.default_confidence is not None:
            claim["confidence"] = self.default_confidence
        return claim
