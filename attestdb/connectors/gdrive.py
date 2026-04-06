"""Google Drive connector — lists files from Drive, downloads/exports text,
and ingests each file through ``db.ingest_text()``.

Extends the Google auth pattern from the GDocs connector to handle all
file types in Drive: Google-native formats (Docs, Sheets, Slides) are
exported as plain text; text-like files are downloaded directly; .docx
files have their XML extracted.

Usage::

    from attestdb.connectors.gdrive import GDriveConnector

    conn = GDriveConnector(
        access_token="ya29....",
        folder_id="1abc...",  # optional — restrict to folder
        max_files=50,
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Iterator

from attestdb.connectors.base import Connector, ConnectorResult

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"

_FOLDER_MIME = "application/vnd.google-apps.folder"
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
_GOOGLE_SHEET_MIME = "application/vnd.google-apps.spreadsheet"
_GOOGLE_SLIDES_MIME = "application/vnd.google-apps.presentation"

# Google-native formats that can be exported to text/plain
_EXPORTABLE_MIMES = {_GOOGLE_DOC_MIME, _GOOGLE_SHEET_MIME, _GOOGLE_SLIDES_MIME}

# Raw-downloadable text-like formats
_TEXT_MIMES = {
    "text/plain", "text/markdown", "text/csv", "text/html",
    "text/xml", "application/json",
}


class GDriveConnector(Connector):
    """Sync connector that lists files from Google Drive, downloads their
    text content, and ingests via ``db.ingest_text()``."""

    name = "gdrive"

    def __init__(
        self,
        access_token: str,
        folder_id: str | None = None,
        mime_types: list[str] | None = None,
        max_files: int = 50,
        extraction: str = "heuristic",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError("pip install requests for the Google Drive connector")
        self._token = access_token
        self._folder_id = folder_id
        self._mime_types = mime_types
        self._max_files = max_files
        self._extraction = extraction
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {access_token}"

    def fetch(self) -> Iterator[dict]:
        raise NotImplementedError("Use run() for text connectors")

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)

        files = self._list_files()
        logger.info("%s: found %d files", self.name, len(files))

        for file_id, name, mime in files:
            if mime == _FOLDER_MIME:
                result.claims_skipped += 1
                continue

            try:
                text = self._get_text(file_id, mime)
            except Exception as exc:
                result.errors.append(f"gdrive/{file_id} ({name}): {exc}")
                time.sleep(0.1)
                continue

            if not text or not text.strip():
                result.claims_skipped += 1
                time.sleep(0.1)
                continue

            source_id = f"gdrive/{file_id}"
            try:
                if name.strip():
                    db.ingest(
                        subject=(name.strip(), "document"),
                        predicate=("covers_topic", "covers_topic"),
                        object=(name.strip(), "topic"),
                        provenance={"source_type": "document", "source_id": source_id},
                        confidence=1.0,
                    )
                    result.claims_ingested += 1

                if self._extraction != "none":
                    er = db.ingest_text(text, source_id=source_id)
                    result.claims_ingested += er.n_valid
            except Exception as exc:
                result.errors.append(f"gdrive/{file_id} ingest: {exc}")

            time.sleep(0.1)

        self._finalize_result(result, start)
        return result

    # ── helpers ──────────────────────────────────────────────────────

    def _list_files(self) -> list[tuple[str, str, str]]:
        """List files (paginated). Returns (file_id, name, mimeType)."""
        files: list[tuple[str, str, str]] = []
        page_token: str | None = None

        q_parts = ["mimeType != 'application/vnd.google-apps.folder'"]
        if self._folder_id:
            q_parts.append(f"'{self._folder_id}' in parents")
        if self._mime_types:
            mime_filter = " or ".join(f"mimeType='{m}'" for m in self._mime_types)
            q_parts = [f"({mime_filter})"]
            if self._folder_id:
                q_parts.append(f"'{self._folder_id}' in parents")

        while len(files) < self._max_files:
            params: dict = {
                "q": " and ".join(q_parts),
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": min(100, self._max_files - len(files)),
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token

            resp = self._request_with_retry("GET", DRIVE_FILES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            for f in data.get("files", []):
                files.append((f["id"], f.get("name", ""), f.get("mimeType", "")))
                if len(files) >= self._max_files:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            time.sleep(0.1)

        return files

    def _get_text(self, file_id: str, mime: str) -> str:
        """Get text content for a file by ID based on its MIME type."""
        if mime in _EXPORTABLE_MIMES:
            return self._export_text(file_id)
        if mime in _TEXT_MIMES:
            return self._download(file_id)
        # Best effort: try downloading as text
        return self._download(file_id)

    def _export_text(self, file_id: str) -> str:
        """Export a Google-native file to text/plain."""
        resp = self._request_with_retry(
            "GET", f"{DRIVE_FILES_URL}/{file_id}/export",
            params={"mimeType": "text/plain"},
        )
        resp.raise_for_status()
        return resp.text

    def _download(self, file_id: str) -> str:
        """Download raw file content."""
        resp = self._request_with_retry(
            "GET", f"{DRIVE_FILES_URL}/{file_id}",
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        resp.raise_for_status()
        return resp.text
