"""S3 connector — reads JSONL or CSV files from an S3 bucket and maps to claims.

Iterates objects under a prefix, downloads each, and parses lines according
to the chosen format.

Usage::

    from attestdb.connectors.s3 import S3Connector

    conn = S3Connector(
        bucket="my-data-lake",
        prefix="claims/2026/",
        file_format="jsonl",
        mapping={
            "subject": "entity_a",
            "predicate": "relation",
            "object": "entity_b",
            "subject_type": "gene",
            "object_type": "protein",
        },
    )
    result = conn.run(db)
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Iterator

from attestdb.connectors.base import StructuredConnector

logger = logging.getLogger(__name__)


class S3Connector(StructuredConnector):
    """Import claims from S3 objects (JSONL or CSV) with field mapping."""

    name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        file_format: str = "jsonl",
        mapping: dict | None = None,
        source_type: str = "s3",
        max_items: int = 10000,
        region: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bucket = bucket
        self.prefix = prefix
        if file_format not in ("jsonl", "csv"):
            raise ValueError(f"Unsupported file_format: {file_format!r} (use 'jsonl' or 'csv')")
        self.file_format = file_format
        self.mapping = mapping or {}
        self.source_type = source_type
        self.max_items = max_items
        self.region = region

    def fetch(self) -> Iterator[dict]:
        try:
            import boto3
        except ImportError:
            raise ImportError("pip install boto3 for S3 support")

        client_kwargs: dict = {"service_name": "s3"}
        if self.region:
            client_kwargs["region_name"] = self.region
        s3 = boto3.client(**client_kwargs)

        count = 0
        continuation_token = None

        while count < self.max_items:
            list_kwargs: dict = {"Bucket": self.bucket, "Prefix": self.prefix}
            if continuation_token:
                list_kwargs["ContinuationToken"] = continuation_token
            resp = s3.list_objects_v2(**list_kwargs)

            for obj in resp.get("Contents", []):
                if count >= self.max_items:
                    return
                key = obj["Key"]
                if key.endswith("/"):
                    continue  # skip directory markers
                try:
                    body = s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                    text = body.decode("utf-8")
                except Exception as exc:
                    logger.warning("s3: failed to read %s: %s", key, exc)
                    continue

                if self.file_format == "jsonl":
                    for claim in self._parse_jsonl(text, key):
                        if count >= self.max_items:
                            return
                        yield claim
                        count += 1
                else:
                    for claim in self._parse_csv(text, key):
                        if count >= self.max_items:
                            return
                        yield claim
                        count += 1

            if resp.get("IsTruncated"):
                continuation_token = resp.get("NextContinuationToken")
            else:
                break

    def _parse_jsonl(self, text: str, key: str) -> Iterator[dict]:
        m = self.mapping
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                claim = {
                    "subject": (
                        str(doc[m["subject"]]) if m.get("subject") else str(doc.get("subject", "")),
                        m.get("subject_type", "entity"),
                    ),
                    "predicate": (
                        str(doc[m["predicate"]]) if m.get("predicate") else str(doc.get("predicate", "")),
                        m.get("predicate_type", "relates_to"),
                    ),
                    "object": (
                        str(doc[m["object"]]) if m.get("object") else str(doc.get("object", "")),
                        m.get("object_type", "entity"),
                    ),
                    "provenance": {
                        "source_type": self.source_type,
                        "source_id": f"s3:{self.bucket}/{key}:line{line_num}",
                    },
                }
                if "confidence" in m and m["confidence"] in doc:
                    claim["confidence"] = float(doc[m["confidence"]])
                yield claim
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("s3: skipping %s line %d: %s", key, line_num, exc)

    def _parse_csv(self, text: str, key: str) -> Iterator[dict]:
        m = self.mapping
        reader = csv.DictReader(io.StringIO(text))
        for row_num, row in enumerate(reader, start=1):
            try:
                claim = {
                    "subject": (str(row[m["subject"]]), m.get("subject_type", "entity")),
                    "predicate": (str(row[m["predicate"]]), m.get("predicate_type", "relates_to")),
                    "object": (str(row[m["object"]]), m.get("object_type", "entity")),
                    "provenance": {
                        "source_type": self.source_type,
                        "source_id": f"s3:{self.bucket}/{key}:row{row_num}",
                    },
                }
                if "confidence" in m and m["confidence"] in row:
                    claim["confidence"] = float(row[m["confidence"]])
                yield claim
            except (KeyError, ValueError) as exc:
                logger.warning("s3: skipping %s row %d: %s", key, row_num, exc)
