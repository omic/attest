"""Sync Gmail connector — fetches email threads and ingests via text extraction.

Uses the Gmail REST API with a pre-obtained OAuth2 access token.
Threads are fetched in full, plain-text bodies are extracted from MIME
payloads recursively, and each thread is passed to ``db.ingest_text()``.

Usage::

    from attestdb.connectors.gmail import GmailConnector

    connector = GmailConnector(access_token="ya29.xxx", query="label:inbox")
    result = connector.run(db)
"""

from __future__ import annotations

import base64
import logging
import time
from typing import Iterator

from attestdb.connectors.base import TextConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"

# Maximum size of a base64-encoded body part to decode (25 MB).
_MAX_BODY_BYTES = 25 * 1024 * 1024


class GmailConnector(TextConnector):
    """Sync connector that ingests Gmail threads via text extraction."""

    name = "gmail"

    def __init__(
        self,
        access_token: str,
        query: str = "",
        max_results: int = 100,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the Gmail connector")
        self.access_token = access_token
        self.query = query
        self.max_results = max_results
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {access_token}"

    # ------------------------------------------------------------------
    # TextConnector contract
    # ------------------------------------------------------------------

    def fetch_texts(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, text)`` for each Gmail thread."""
        thread_ids = self._list_threads()
        logger.info(
            "gmail: found %d thread(s) matching query %r",
            len(thread_ids), self.query,
        )

        for thread_id in thread_ids:
            try:
                thread_data = self._get_thread(thread_id)
                body_text = self._extract_text(thread_data)
                yield (f"gmail:{thread_id}", body_text)
            except Exception as exc:
                logger.warning(
                    "gmail: error fetching thread %s: %s",
                    thread_id, exc,
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _list_threads(self) -> list[str]:
        """Return up to ``max_results`` thread IDs matching ``self.query``."""
        thread_ids: list[str] = []
        page_token: str | None = None

        while len(thread_ids) < self.max_results:
            params: dict[str, str | int] = {
                "maxResults": min(100, self.max_results - len(thread_ids)),
            }
            if self.query:
                params["q"] = self.query
            if page_token:
                params["pageToken"] = page_token

            resp = self._request_with_retry(
                "GET", f"{GMAIL_API}/users/me/threads", params=params
            )
            resp.raise_for_status()
            data = resp.json()

            for t in data.get("threads", []):
                thread_ids.append(t["id"])

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.1)

        return thread_ids[: self.max_results]

    def _get_thread(self, thread_id: str) -> dict:
        """Fetch a single thread with full MIME payload."""
        time.sleep(0.1)
        resp = self._request_with_retry(
            "GET",
            f"{GMAIL_API}/users/me/threads/{thread_id}",
            params={"format": "full"},
        )
        resp.raise_for_status()
        return resp.json()

    def _extract_text(self, thread_data: dict) -> str:
        """Concatenate plain-text bodies from all messages in a thread."""
        parts: list[str] = []
        for message in thread_data.get("messages", []):
            payload = message.get("payload", {})
            text = self._extract_text_from_payload(payload)
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _extract_text_from_payload(self, payload: dict) -> str:
        """Recursively extract text/plain content from a MIME payload."""
        mime_type = payload.get("mimeType", "")

        # Leaf part with text/plain body data
        if mime_type == "text/plain":
            body = payload.get("body", {})
            data = body.get("data", "")
            if data:
                if len(data) > _MAX_BODY_BYTES:
                    logger.warning(
                        "gmail: skipping oversized body part (%d bytes)",
                        len(data),
                    )
                    return ""
                return base64.urlsafe_b64decode(data).decode(
                    "utf-8", errors="replace",
                )
            return ""

        # Multipart — recurse into sub-parts
        sub_parts = payload.get("parts", [])
        texts: list[str] = []
        for part in sub_parts:
            text = self._extract_text_from_payload(part)
            if text:
                texts.append(text)
        return "\n".join(texts)
