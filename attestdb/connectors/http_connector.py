"""Generic HTTP/Webhook connector — imports JSON from any URL as claims.

Fetches JSON from an arbitrary HTTP endpoint and maps fields to claim
components using a simple dot-notation mapping dict.

Usage::

    from attestdb.connectors.http_connector import HTTPConnector

    conn = HTTPConnector(
        url="https://api.example.com/data",
        headers={"X-Api-Key": "secret"},
        items_path="data.results",
        mapping={
            "subject": "name",
            "subject_type": "entity",
            "predicate": "relation",
            "predicate_type": "relates_to",
            "object": "target",
            "object_type": "entity",
        },
        source_type="example_api",
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


def _resolve_path(obj: Any, path: str) -> Any:
    """Resolve a dot-separated path against a dict/list structure.

    >>> _resolve_path({"a": {"b": [1, 2]}}, "a.b")
    [1, 2]
    """
    if not path:
        return obj
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        elif isinstance(obj, (list, tuple)) and key.isdigit():
            obj = obj[int(key)]
        else:
            return None
        if obj is None:
            return None
    return obj


class HTTPConnector(StructuredConnector):
    """Generic connector that fetches JSON from any HTTP endpoint."""

    name = "http"

    def __init__(
        self,
        url: str,
        method: str = "GET",
        headers: dict | None = None,
        params: dict | None = None,
        body: dict | None = None,
        mapping: dict | None = None,
        source_type: str = "http_import",
        auth_token: str | None = None,
        items_path: str = "",
        next_url_path: str | None = None,
        offset_param: str | None = None,
        page_size: int = 100,
        max_items: int = 10_000,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the HTTP connector")
        if not mapping:
            raise ValueError("mapping dict is required (keys: subject, predicate, object)")
        self._url = url
        self._method = method.upper()
        self._params = params or {}
        self._body = body
        self._mapping = mapping
        self._source_type = source_type
        self._items_path = items_path
        self._next_url_path = next_url_path
        self._offset_param = offset_param
        self._page_size = page_size
        self._max_items = max_items
        self._session = requests.Session()
        if auth_token:
            self._session.headers["Authorization"] = f"Bearer {auth_token}"
        if headers:
            self._session.headers.update(headers)

    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts by fetching JSON and applying the mapping."""
        url = self._url
        count = 0
        pages = 0
        offset = 0

        while count < self._max_items and pages < self._MAX_PAGES:
            params = dict(self._params)
            if self._offset_param:
                params[self._offset_param] = offset

            req_kwargs: dict[str, Any] = {"params": params}
            if self._body is not None:
                req_kwargs["json"] = self._body

            resp = self._request_with_retry(self._method, url, **req_kwargs)
            resp.raise_for_status()
            data = resp.json()

            # Extract items array from response
            items = _resolve_path(data, self._items_path) if self._items_path else data
            if not isinstance(items, list):
                # Single object — wrap it
                if items is not None:
                    items = [items]
                else:
                    break

            if not items:
                break

            for item in items:
                claim = self._map_item(item, count)
                if claim:
                    yield claim
                    count += 1
                if count >= self._max_items:
                    break

            # Pagination
            if self._next_url_path:
                next_url = _resolve_path(data, self._next_url_path)
                if not next_url:
                    break
                url = next_url
            elif self._offset_param:
                offset += len(items)
                if len(items) < self._page_size:
                    break
            else:
                # No pagination configured — single request
                break

            pages += 1
            time.sleep(0.1)

    def _map_item(self, item: dict, index: int) -> dict | None:
        """Apply the mapping to extract a claim dict from a single item."""
        m = self._mapping

        subject = _resolve_path(item, m.get("subject", ""))
        predicate = _resolve_path(item, m.get("predicate", ""))
        obj = _resolve_path(item, m.get("object", ""))

        if not subject or not predicate or not obj:
            return None

        subject_type = m.get("subject_type", "entity")
        predicate_type = m.get("predicate_type", "relates_to")
        object_type = m.get("object_type", "entity")

        # Allow dynamic types from item fields
        if subject_type.startswith("$."):
            subject_type = str(_resolve_path(item, subject_type[2:]) or "entity")
        if object_type.startswith("$."):
            object_type = str(_resolve_path(item, object_type[2:]) or "entity")

        source_id_field = m.get("source_id")
        if source_id_field:
            source_id = f"{self._source_type}:{_resolve_path(item, source_id_field)}"
        else:
            source_id = f"{self._source_type}:{index}"

        confidence = m.get("confidence")
        if isinstance(confidence, str):
            confidence = _resolve_path(item, confidence)

        result: dict = {
            "subject": (str(subject), subject_type),
            "predicate": (str(predicate), predicate_type),
            "object": (str(obj), object_type),
            "provenance": {
                "source_type": self._source_type,
                "source_id": source_id,
            },
        }
        if confidence is not None:
            try:
                result["confidence"] = float(confidence)
            except (ValueError, TypeError):
                pass
        return result
