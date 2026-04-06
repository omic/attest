"""Notion connector — fetches pages and extracts text for claim ingestion."""

from __future__ import annotations

import logging
import time
from typing import Iterator

from attestdb.connectors.base import TextConnector

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"


class NotionConnector(TextConnector):
    """Sync Notion connector that ingests page text via ``db.ingest_text()``."""

    name = "notion"

    def __init__(
        self,
        api_key: str,
        database_id: str | None = None,
        max_pages: int = 100,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError(
                "pip install requests for the Notion connector"
            )
        self._api_key = api_key
        self._database_id = database_id
        self._max_pages = max_pages
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # TextConnector contract
    # ------------------------------------------------------------------

    def fetch_texts(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, text)`` for each Notion page."""
        pages = self._list_pages()
        for page_id, title in pages:
            logger.info(
                "Fetching page %s (%s)",
                title or "(untitled)", page_id,
            )
            try:
                text = self._get_page_text(page_id)
            except Exception as exc:
                logger.warning("notion: page %s: %s", page_id, exc)
                continue
            yield (f"notion:{page_id}", text)
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Notion API helpers
    # ------------------------------------------------------------------

    def _list_pages(self) -> list[tuple[str, str]]:
        """Return list of (page_id, title) tuples, up to *max_pages*."""
        pages: list[tuple[str, str]] = []
        start_cursor: str | None = None

        while len(pages) < self._max_pages:
            if self._database_id:
                url = f"{NOTION_API}/databases/{self._database_id}/query"
                body: dict = {}
            else:
                url = f"{NOTION_API}/search"
                body = {
                    "filter": {"property": "object", "value": "page"},
                }

            if start_cursor:
                body["start_cursor"] = start_cursor
            body["page_size"] = min(100, self._max_pages - len(pages))

            resp = self._request_with_retry(
                "POST", url, json=body, timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for page in data.get("results", []):
                page_id = page["id"]
                title = ""
                for prop in page.get("properties", {}).values():
                    if prop.get("type") == "title":
                        parts = prop.get("title", [])
                        title = "".join(
                            t.get("plain_text", "") for t in parts
                        )
                        break
                pages.append((page_id, title))

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")
            time.sleep(0.1)

        return pages

    _TEXT_BLOCK_TYPES = frozenset({
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
    })

    def _get_page_text(self, page_id: str) -> str:
        """Return concatenated plain text from supported block types."""
        lines: list[str] = []
        start_cursor: str | None = None

        for _ in range(self._MAX_PAGES):
            params: dict = {"page_size": 100}
            if start_cursor:
                params["start_cursor"] = start_cursor

            resp = self._request_with_retry(
                "GET",
                f"{NOTION_API}/blocks/{page_id}/children",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            for block in data.get("results", []):
                btype = block.get("type", "")
                if btype not in self._TEXT_BLOCK_TYPES:
                    continue
                rich_text = block.get(btype, {}).get("rich_text", [])
                text = "".join(
                    rt.get("plain_text", "") for rt in rich_text
                )
                if text.strip():
                    lines.append(text.strip())

            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")
            time.sleep(0.1)

        return "\n".join(lines)
