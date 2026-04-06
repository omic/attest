"""Google Docs connector — lists documents via Drive API, fetches content
via Docs API, and ingests each document through ``db.ingest_text()``.

Usage::

    from attestdb.connectors.gdocs import GDocsConnector

    connector = GDocsConnector(access_token="ya29....")
    result = connector.run(db)
"""

from __future__ import annotations

import io
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import TYPE_CHECKING, Iterator

from attestdb.connectors.base import Connector, ConnectorResult

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
DOCS_GET_URL = "https://docs.googleapis.com/v1/documents/{doc_id}"

# mimeTypes that the Docs API can read natively
_GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
# Folder mime — skip these
_FOLDER_MIME = "application/vnd.google-apps.folder"
# Types we can export as text/plain via Drive export endpoint
_EXPORTABLE_MIMES = {_GOOGLE_DOC_MIME}
# Maximum uncompressed size for a .docx XML entry (50 MB).
# Guards against zip bombs that expand far beyond the compressed size.
_MAX_DOCX_UNCOMPRESSED = 50 * 1024 * 1024
# Types we download raw via alt=media (text-like)
_TEXT_MIMES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "text/xml",
    "application/json",
}
# Types that Google Drive can export to text/plain
_DRIVE_EXPORTABLE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.oasis.opendocument.text",  # .odt
    "application/rtf",
}


class GDocsConnector(Connector):
    """Sync connector that lists Google Docs via the Drive API and fetches
    their content via the Docs API.

    Each document with non-empty text is ingested through
    ``db.ingest_text(text, source_id="gdocs:{doc_id}")``.
    """

    name = "gdocs"

    def __init__(
        self,
        access_token: str,
        max_docs: int = 50,
        doc_ids: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if requests is None:
            raise ImportError(
                "The 'requests' package is required for GDocsConnector. "
                "Install it with: pip install requests"
            )
        self.access_token = access_token
        self.max_docs = max_docs
        self.doc_ids = doc_ids
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self.access_token}"}
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self) -> Iterator[dict]:
        """Not supported — use ``run()`` for text connectors."""
        raise NotImplementedError("Use run() for text connectors")

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """List Google Docs, fetch content, and ingest via ``db.ingest_text()``.

        Returns a :class:`ConnectorResult` summarising the run.
        """
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)

        if self.doc_ids:
            docs = self._resolve_doc_ids(self.doc_ids)
        else:
            docs = self._list_docs()
        logger.info("%s: found %d documents", self.name, len(docs))

        for doc_id, title, mime_type in docs:
            if mime_type == _FOLDER_MIME:
                logger.debug("%s: skipping folder %s (%s)", self.name, doc_id, title)
                result.claims_skipped += 1
                continue

            try:
                text = self._get_doc_text(doc_id, mime_type)
            except Exception as exc:
                msg = f"Failed to fetch doc {doc_id} ({title}): {exc}"
                logger.warning(msg)
                result.errors.append(msg)
                time.sleep(0.1)
                continue

            if not text or not text.strip():
                logger.debug("%s: skipping empty doc %s (%s)", self.name, doc_id, title)
                result.claims_skipped += 1
                time.sleep(0.1)
                continue

            source_id = f"gdocs:{doc_id}"
            try:
                # Anchor claim: link the document to its title so
                # the doc is discoverable by name/topic searches.
                if title.strip():
                    db.ingest(
                        subject=(title.strip(), "document"),
                        predicate=("covers_topic", "covers_topic"),
                        object=(title.strip(), "topic"),
                        provenance={"source_type": "gdocs", "source_id": source_id},
                        confidence=1.0,
                    )
                    result.claims_ingested += 1

                extract_result = db.ingest_text(text, source_id=source_id)
                result.claims_ingested += extract_result.n_valid
                result.prompt_tokens += getattr(extract_result, "prompt_tokens", 0)
                result.completion_tokens += getattr(extract_result, "completion_tokens", 0)
                logger.debug("%s: ingested doc %s (%s)", self.name, doc_id, title)
            except Exception as exc:
                msg = f"Failed to ingest doc {doc_id} ({title}): {exc}"
                logger.warning(msg)
                result.errors.append(msg)

            time.sleep(0.1)

        self._finalize_result(result, start)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _list_docs(self) -> list[tuple[str, str, str]]:
        """List Google Docs from the Drive API (paginated).

        Includes files from Shared Drives.
        Returns a list of ``(doc_id, title, mimeType)`` tuples, up to
        ``self.max_docs``.
        """
        docs: list[tuple[str, str, str]] = []
        page_token: str | None = None

        while len(docs) < self.max_docs:
            params: dict[str, str | int | bool] = {
                "q": "mimeType='application/vnd.google-apps.document'",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": min(100, self.max_docs - len(docs)),
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
            }
            if page_token is not None:
                params["pageToken"] = page_token

            resp = self._request_with_retry("GET", DRIVE_FILES_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            for f in data.get("files", []):
                docs.append((f["id"], f.get("name", ""), f.get("mimeType", "")))
                if len(docs) >= self.max_docs:
                    break

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.1)

        return docs

    def _resolve_doc_ids(self, doc_ids: list[str]) -> list[tuple[str, str, str]]:
        """Fetch metadata for specific doc IDs via the Drive API.

        Returns a list of ``(doc_id, title, mimeType)`` tuples.
        """
        docs: list[tuple[str, str, str]] = []
        for doc_id in doc_ids:
            try:
                url = f"{DRIVE_FILES_URL}/{doc_id}"
                resp = self._request_with_retry(
                    "GET",
                    url,
                    params={"fields": "id,name,mimeType", "supportsAllDrives": "true"},
                )
                resp.raise_for_status()
                data = resp.json()
                docs.append((data["id"], data.get("name", ""), data.get("mimeType", "")))
            except Exception as exc:
                logger.warning("Failed to resolve doc ID %s: %s", doc_id, exc)
            time.sleep(0.1)
        return docs

    def _get_doc_text(self, doc_id: str, mime_type: str = _GOOGLE_DOC_MIME) -> str:
        """Fetch a file by ID and return its plain-text content.

        Dispatches based on ``mime_type``:
        - Native Google Docs → Docs API structured extraction
        - .docx / .odt / .rtf → Drive export as text/plain
        - text/* / .json → Drive download (alt=media)
        - Other → Drive export as text/plain (best effort)
        """
        if mime_type == _GOOGLE_DOC_MIME:
            url = DOCS_GET_URL.format(doc_id=doc_id)
            resp = self._request_with_retry("GET", url)
            resp.raise_for_status()
            body = resp.json().get("body", {})
            return self._extract_text(body)

        if mime_type in _TEXT_MIMES:
            return self._drive_download(doc_id)

        if mime_type in _DRIVE_EXPORTABLE_MIMES:
            return self._drive_download_docx(doc_id)

        # Best effort: try download as text
        return self._drive_download(doc_id)

    def _drive_export_text(self, doc_id: str) -> str:
        """Export a file as text/plain via the Drive export endpoint."""
        url = f"{DRIVE_FILES_URL}/{doc_id}/export"
        resp = self._request_with_retry(
            "GET",
            url,
            params={"mimeType": "text/plain", "supportsAllDrives": "true"},
        )
        resp.raise_for_status()
        return resp.text

    def _drive_download(self, doc_id: str) -> str:
        """Download a file's raw content via Drive alt=media."""
        url = f"{DRIVE_FILES_URL}/{doc_id}"
        resp = self._request_with_retry(
            "GET",
            url,
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        resp.raise_for_status()
        return resp.text

    def _drive_download_docx(self, doc_id: str) -> str:
        """Download a .docx file and extract text from its XML content."""
        url = f"{DRIVE_FILES_URL}/{doc_id}"
        resp = self._request_with_retry(
            "GET",
            url,
            params={"alt": "media", "supportsAllDrives": "true"},
        )
        resp.raise_for_status()
        return self._extract_docx_text(resp.content)

    @staticmethod
    def _extract_docx_text(data: bytes) -> str:
        """Extract plain text from a .docx file (ZIP of XML)."""
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            info = zf.getinfo("word/document.xml")
            if info.file_size > _MAX_DOCX_UNCOMPRESSED:
                raise ValueError(
                    f"word/document.xml uncompressed size ({info.file_size} bytes) "
                    f"exceeds limit ({_MAX_DOCX_UNCOMPRESSED} bytes)"
                )
            xml_content = zf.read("word/document.xml")

        root = ET.fromstring(xml_content)
        # Word XML namespace
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        parts: list[str] = []
        for para in root.iter(f"{{{ns['w']}}}p"):
            texts = [t.text for t in para.iter(f"{{{ns['w']}}}t") if t.text]
            if texts:
                parts.append("".join(texts))
        return "\n".join(parts)

    @staticmethod
    def _extract_text(body: dict) -> str:
        """Recursively traverse the Docs API body structure and extract
        plain text from paragraph → elements → textRun → content.
        """
        parts: list[str] = []
        for structural_element in body.get("content", []):
            paragraph = structural_element.get("paragraph")
            if paragraph is None:
                # Handle tables — each cell contains structural elements
                table = structural_element.get("table")
                if table is not None:
                    for row in table.get("tableRows", []):
                        for cell in row.get("tableCells", []):
                            # Each cell has its own content array
                            cell_text = GDocsConnector._extract_text(
                                {"content": cell.get("content", [])}
                            )
                            if cell_text.strip():
                                parts.append(cell_text)
                continue

            for element in paragraph.get("elements", []):
                text_run = element.get("textRun")
                if text_run is not None:
                    content = text_run.get("content", "")
                    parts.append(content)

        return "".join(parts)
