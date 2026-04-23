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

from attestdb.connectors.base import HybridConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

GMAIL_API = "https://gmail.googleapis.com/gmail/v1"

# Maximum size of a base64-encoded body part to decode (25 MB).
_MAX_BODY_BYTES = 25 * 1024 * 1024


class GmailConnector(HybridConnector):
    """Sync connector that ingests Gmail threads as structural + text claims.

    Structural pass: extracts sender/recipient relationships with email
    addresses as external_ids for cross-connector entity resolution.
    Text pass: extracts knowledge claims from email body text.
    """

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
        self._thread_cache: list[dict] = []

    # ------------------------------------------------------------------
    # HybridConnector contract
    # ------------------------------------------------------------------

    def fetch(self) -> Iterator[dict]:
        """Yield structural claims from email headers (sender, recipients)."""
        import email.utils

        self._thread_cache = []
        thread_ids = self._list_threads()
        logger.info(
            "gmail: found %d thread(s) matching query %r",
            len(thread_ids), self.query,
        )

        for thread_id in thread_ids:
            try:
                thread_data = self._get_thread(thread_id)
                self._thread_cache.append(thread_data)

                for message in thread_data.get("messages", []):
                    headers = self._extract_headers(message)
                    sid = f"gmail:{thread_id}"

                    # Parse sender
                    from_header = headers.get("from", "")
                    sender_pairs = email.utils.getaddresses([from_header])
                    if not sender_pairs:
                        continue
                    sender_name, sender_email = sender_pairs[0]
                    if not sender_email:
                        continue
                    sender_name = sender_name or sender_email.split("@")[0]

                    pl: dict = {"schema_ref": "gmail/thread", "data": {
                        "thread_id": thread_id,
                        "sender_email": sender_email,
                    }}

                    # sender authored this thread
                    yield self._make_claim(
                        sender_name, "person", "authored",
                        f"gmail:{thread_id}", "email_thread", sid,
                        subj_ext={"email": sender_email},
                        payload=pl,
                    )

                    # Parse recipients (To + Cc)
                    for hdr_name in ("to", "cc"):
                        recip_header = headers.get(hdr_name, "")
                        if not recip_header:
                            continue
                        for recip_name, recip_email in email.utils.getaddresses([recip_header]):
                            if not recip_email:
                                continue
                            recip_name = recip_name or recip_email.split("@")[0]
                            yield self._make_claim(
                                sender_name, "person", "sent_email_to",
                                recip_name, "person", sid,
                                subj_ext={"email": sender_email},
                                obj_ext={"email": recip_email},
                                payload=pl,
                            )

            except Exception as exc:
                logger.warning("gmail: error fetching thread %s: %s", thread_id, exc)

    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, text)`` for each cached Gmail thread."""
        for thread_data in self._thread_cache:
            thread_id = thread_data.get("id", "")
            body_text = self._extract_text(thread_data)
            if body_text and body_text.strip():
                yield (f"gmail:{thread_id}", body_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_headers(message: dict) -> dict[str, str]:
        """Extract headers from a Gmail message payload as a lowercase dict."""
        headers: dict[str, str] = {}
        payload = message.get("payload", {})
        for h in payload.get("headers", []):
            name = h.get("name", "").lower()
            value = h.get("value", "")
            if name and value:
                headers[name] = value
        return headers

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
