"""Airtable connector — imports records as knowledge claims.

Uses the Airtable REST API to fetch records from a table and maps
field names to claim components using a mapping dict.

Usage::

    from attestdb.connectors.airtable import AirtableConnector

    conn = AirtableConnector(
        token=os.environ["AIRTABLE_API_KEY"],
        base_id="appXXXXXXXXXXXXXX",
        table_name="Issues",
        mapping={
            "subject": "Name",
            "subject_type": "entity",
            "predicate": "Relationship",
            "predicate_type": "relates_to",
            "object": "Target",
            "object_type": "entity",
        },
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Iterator

from attestdb.connectors.base import ConnectorResult, StructuredConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)

AIRTABLE_API = "https://api.airtable.com/v0"


class AirtableConnector(StructuredConnector):
    """Sync connector that imports Airtable records as structured claims."""

    name = "airtable"

    def __init__(
        self,
        token: str,
        base_id: str,
        table_name: str,
        mapping: dict,
        source_type: str = "airtable",
        view: str | None = None,
        formula: str | None = None,
        max_items: int = 1000,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the Airtable connector")
        if not mapping:
            raise ValueError("mapping dict is required (keys: subject, predicate, object)")
        self._token = token
        self._base_id = base_id
        self._table_name = table_name
        self._mapping = mapping
        self._source_type = source_type
        self._view = view
        self._formula = formula
        self._max_items = max_items
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts from Airtable records."""
        url = f"{AIRTABLE_API}/{self._base_id}/{self._table_name}"
        offset: str | None = None
        count = 0
        pages = 0

        while count < self._max_items and pages < self._MAX_PAGES:
            params: dict[str, Any] = {
                "pageSize": min(100, self._max_items - count),
            }
            if offset:
                params["offset"] = offset
            if self._view:
                params["view"] = self._view
            if self._formula:
                params["filterByFormula"] = self._formula

            resp = self._request_with_retry("GET", url, params=params)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("records", [])
            if not records:
                break

            for record in records:
                claim = self._record_to_claim(record)
                if claim:
                    yield claim
                    count += 1
                if count >= self._max_items:
                    break

            offset = data.get("offset")
            if not offset:
                break

            pages += 1
            time.sleep(0.2)  # Airtable rate limit: 5 req/sec

    def _record_to_claim(self, record: dict) -> dict | None:
        """Map an Airtable record to a claim dict."""
        record_id = record.get("id", "")
        fields = record.get("fields", {})
        m = self._mapping

        subject = fields.get(m.get("subject", ""))
        predicate = m.get("predicate_value") or fields.get(m.get("predicate", ""))
        obj = fields.get(m.get("object", ""))

        if not subject or not predicate or not obj:
            return None

        # Handle linked records (Airtable returns lists for linked fields)
        if isinstance(subject, list):
            subject = subject[0] if subject else None
        if isinstance(obj, list):
            obj = obj[0] if obj else None
        if not subject or not obj:
            return None

        subject_type = m.get("subject_type", "entity")
        predicate_type = m.get("predicate_type", "relates_to")
        object_type = m.get("object_type", "entity")

        source_id = f"{self._source_type}:{record_id}"

        result: dict = {
            "subject": (str(subject), subject_type),
            "predicate": (str(predicate), predicate_type),
            "object": (str(obj), object_type),
            "provenance": {
                "source_type": self._source_type,
                "source_id": source_id,
            },
        }

        # Optional confidence from a field
        confidence_field = m.get("confidence")
        if confidence_field and confidence_field in fields:
            try:
                result["confidence"] = float(fields[confidence_field])
            except (ValueError, TypeError):
                pass

        return result
