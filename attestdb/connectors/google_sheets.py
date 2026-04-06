"""Google Sheets connector — maps spreadsheet rows to claims via column mapping.

Supports two authentication modes:

1. **API key** — for published (public) sheets, no OAuth needed.
2. **Service account** — for private sheets, using a credentials JSON file.

Usage::

    from attestdb.connectors.google_sheets import GoogleSheetsConnector

    # Public sheet with API key
    conn = GoogleSheetsConnector(
        spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
        sheet_name="Sheet1",
        api_key="AIza...",
        mapping={
            "subject": "Gene",
            "predicate": "Relation",
            "object": "Disease",
            "subject_type": "gene",
            "object_type": "disease",
        },
    )
    result = conn.run(db)

    # Private sheet with service account
    conn = GoogleSheetsConnector(
        spreadsheet_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
        credentials_path="service_account.json",
        mapping={...},
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import StructuredConnector

logger = logging.getLogger(__name__)


class GoogleSheetsConnector(StructuredConnector):
    """Import claims from a Google Sheets spreadsheet with column mapping."""

    name = "google_sheets"

    def __init__(
        self,
        spreadsheet_id: str,
        mapping: dict,
        sheet_name: str = "Sheet1",
        credentials_path: str | None = None,
        api_key: str | None = None,
        source_type: str = "google_sheets",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.spreadsheet_id = spreadsheet_id
        self.mapping = mapping
        self.sheet_name = sheet_name
        self.credentials_path = credentials_path
        self.api_key = api_key
        self.source_type = source_type
        if not credentials_path and not api_key:
            raise ValueError("Provide either credentials_path or api_key")

    def _build_service(self):
        """Build the Google Sheets API v4 service resource."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "pip install google-api-python-client google-auth for Google Sheets support"
            )

        if self.api_key:
            return build("sheets", "v4", developerKey=self.api_key)

        # Service account path
        try:
            from google.oauth2.service_account import Credentials
        except ImportError:
            raise ImportError(
                "pip install google-auth for service account support"
            )

        SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
        creds = Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES
        )
        return build("sheets", "v4", credentials=creds)

    def fetch(self) -> Iterator[dict]:
        service = self._build_service()
        result = (
            service.spreadsheets()
            .values()
            .get(spreadsheetId=self.spreadsheet_id, range=self.sheet_name)
            .execute()
        )
        rows = result.get("values", [])
        if len(rows) < 2:
            return  # No data (header only or empty)

        headers = rows[0]
        m = self.mapping

        for row_num, row in enumerate(rows[1:], start=2):
            try:
                # Pad short rows with empty strings
                padded = row + [""] * (len(headers) - len(row))
                doc = dict(zip(headers, padded))

                claim = {
                    "subject": (
                        str(doc[m["subject"]]),
                        m.get("subject_type", "entity"),
                    ),
                    "predicate": (
                        str(doc[m["predicate"]]),
                        m.get("predicate_type", "relates_to"),
                    ),
                    "object": (
                        str(doc[m["object"]]),
                        m.get("object_type", "entity"),
                    ),
                    "provenance": {
                        "source_type": self.source_type,
                        "source_id": f"gsheets:{self.spreadsheet_id}:{self.sheet_name}:row{row_num}",
                    },
                }
                if "confidence" in m and m["confidence"] in doc:
                    claim["confidence"] = float(doc[m["confidence"]])
                yield claim
            except (KeyError, ValueError, IndexError) as exc:
                logger.warning("google_sheets: skipping row %d: %s", row_num, exc)
