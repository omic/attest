"""Base connector interface for Attest data source integrations.

Four pattern-specific base classes enforce orchestration contracts:

- **StructuredConnector** — ``fetch()`` yields claim dicts, sealed ``run()``
  batches and flushes them.
- **TextConnector** — ``fetch_texts()`` yields ``(source_id, text)`` pairs,
  sealed ``run()`` calls ``db.ingest_text()`` for each.
- **HybridConnector** — ``fetch()`` for structural claims + ``fetch_bodies()``
  for text extraction, sealed ``run()`` does both passes.
- **QueryConnector** — extends StructuredConnector; ``_open_cursor()`` returns
  row iterator, shared ``_row_to_claim()`` handles column mapping.

Usage::

    class MyConnector(StructuredConnector):
        name = "my_source"

        def fetch(self) -> Iterator[dict]:
            yield {
                "subject": ("ticket-1", "ticket"),
                "predicate": ("assigned_to", "directional"),
                "object": ("alice", "person"),
                "provenance": {"source_type": "my_source", "source_id": "..."},
                "confidence": 1.0,
            }
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from attestdb.infrastructure.attest_db import AttestDB

logger = logging.getLogger(__name__)

# Defaults for _request_with_retry
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0  # seconds
_RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


@dataclass
class ConnectorResult:
    """Summary returned after a connector run."""

    connector_name: str = ""
    claims_ingested: int = 0
    claims_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Connector(abc.ABC):
    """Abstract base class for all data source connectors.

    Provides shared infrastructure: retry, batching, checkpoints,
    finalization, search.  Subclasses should inherit from one of the
    pattern-specific bases (StructuredConnector, TextConnector,
    HybridConnector, QueryConnector) rather than from Connector directly.

    Direct subclassing is reserved for connectors with genuinely unique
    orchestration (e.g. SlackConnector).
    """

    name: str = ""
    _MAX_PAGES: int = 10_000  # Safety limit for paginated API calls

    def __init__(self, **kwargs: object) -> None:
        # Accept and ignore unknown kwargs so subclasses can be constructed
        # with extra options without breaking the base.
        self._db_path: str | None = str(kwargs["db_path"]) if "db_path" in kwargs else None
        self._session: object | None = None  # Set by HTTP subclasses
        self._checkpoint: dict = {}
        self._load_checkpoint()

    def close(self) -> None:
        """Release resources (e.g. HTTP session)."""
        if hasattr(self, "_session") and self._session is not None:
            if hasattr(self._session, "close"):
                self._session.close()
            self._session = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_checkpoint(self) -> None:
        """Load checkpoint from TokenStore key ``"{name}:checkpoint"``."""
        if not self._db_path or not self.name:
            return
        try:
            from attestdb.connectors.token_store import TokenStore
            ts = TokenStore(self._db_path)
            data = ts.get_token(f"{self.name}:checkpoint")
            if data:
                self._checkpoint = data
        except Exception:
            pass  # No cryptography, no file, etc.

    def _save_checkpoint(self) -> None:
        """Persist ``self._checkpoint`` to TokenStore."""
        if not self._db_path or not self.name or not self._checkpoint:
            return
        try:
            from attestdb.connectors.token_store import TokenStore
            ts = TokenStore(self._db_path)
            ts.save_token(f"{self.name}:checkpoint", dict(self._checkpoint))
        except Exception as e:
            logger.warning("Failed to save checkpoint for %s: %s", self.name, e)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        **kwargs: object,
    ):
        """Issue an HTTP request with exponential backoff on transient errors.

        Parameters
        ----------
        method:
            HTTP method (``"GET"``, ``"POST"``, etc.).
        url:
            Request URL.
        max_retries:
            Maximum number of retries after the initial attempt.
        backoff_base:
            Base delay in seconds; doubled on each retry.
        **kwargs:
            Passed through to ``self._session.request()``.

        Returns the ``requests.Response`` on success (caller should still
        call ``resp.raise_for_status()`` if desired).

        Raises
        ------
        RuntimeError
            If ``self._session`` is not set.
        requests.RequestException
            After all retries are exhausted.
        """
        if self._session is None:
            raise RuntimeError(f"{self.name}: no HTTP session initialised")

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._session.request(method, url, **kwargs)
                if resp.status_code not in _RETRIABLE_STATUS_CODES:
                    return resp
                # Retriable HTTP status — fall through to backoff
                last_exc = None
                if attempt < max_retries:
                    delay = backoff_base * (2 ** attempt)
                    logger.warning(
                        "%s: HTTP %d from %s, retrying in %.1fs (attempt %d/%d)",
                        self.name, resp.status_code, url, delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)
                else:
                    # Final attempt — return the response as-is so the
                    # caller can inspect or raise_for_status().
                    return resp
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    delay = backoff_base * (2 ** attempt)
                    logger.warning(
                        "%s: request to %s failed (%s), retrying in %.1fs (attempt %d/%d)",
                        self.name, url, exc, delay, attempt + 1, max_retries,
                    )
                    time.sleep(delay)

        # Should only reach here on connection-level errors
        raise last_exc  # type: ignore[misc]

    @property
    def supports_search(self) -> bool:
        """Whether this connector supports query-based search."""
        return False

    def search(self, query: str) -> str:
        """Search the data source for evidence matching *query*.

        Returns text suitable for claim extraction (max ~4000 chars).
        Default implementation returns empty string. Override in
        connectors that support search (Slack, GitHub, Jira, Confluence).
        """
        return ""

    @abc.abstractmethod
    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """Execute the connector and ingest data into *db*.

        Each pattern-specific subclass provides a sealed implementation.
        Direct ``Connector`` subclasses (e.g. Slack) implement their own.
        """

    def _finalize_result(self, result: ConnectorResult, start: float) -> None:
        """Set elapsed time and log a summary line.

        Called at the end of ``run()`` (both fetch-based and text-based
        connectors) to avoid duplicating the same 7 lines across every
        connector.
        """
        result.elapsed_seconds = round(time.monotonic() - start, 3)
        logger.info(
            "%s: ingested %d, skipped %d, errors %d in %.1fs",
            self.name,
            result.claims_ingested,
            result.claims_skipped,
            len(result.errors),
            result.elapsed_seconds,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _flush(db: AttestDB, batch: list[dict], result: ConnectorResult) -> None:
        from attestdb.core.types import ClaimInput

        inputs = []
        for d in batch:
            try:
                inputs.append(
                    ClaimInput(
                        subject=d["subject"],
                        predicate=d["predicate"],
                        object=d["object"],
                        provenance=d["provenance"],
                        confidence=d.get("confidence"),
                        embedding=d.get("embedding"),
                        payload=d.get("payload"),
                        timestamp=d.get("timestamp"),
                        external_ids=d.get("external_ids"),
                    )
                )
            except (KeyError, TypeError) as exc:
                result.errors.append(str(exc))

        if inputs:
            br = db.ingest_batch(inputs)
            result.claims_ingested += br.ingested
            result.claims_skipped += br.duplicates
            result.errors.extend(br.errors)

    def _make_claim(
        self,
        subj: str,
        subj_type: str,
        pred: str,
        obj: str,
        obj_type: str,
        source_id: str,
        *,
        confidence: float = 1.0,
        subj_ext: dict[str, str] | None = None,
        obj_ext: dict[str, str] | None = None,
    ) -> dict:
        """Build a claim dict with optional external_ids.

        Uses ``predicate_type()`` from ``attestdb.connectors.predicates``
        to classify the predicate automatically.
        """
        from attestdb.connectors.predicates import predicate_type

        d: dict = {
            "subject": (subj, subj_type),
            "predicate": (pred, predicate_type(pred)),
            "object": (obj, obj_type),
            "provenance": {
                "source_type": self.name,
                "source_id": source_id,
            },
            "confidence": confidence,
        }
        ext: dict = {}
        if subj_ext:
            ext["subject"] = subj_ext
        if obj_ext:
            ext["object"] = obj_ext
        if ext:
            d["external_ids"] = ext
        return d


# ======================================================================
# Pattern-specific base classes
# ======================================================================


class StructuredConnector(Connector):
    """Base for connectors that yield structured claim dicts.

    Subclasses implement ``fetch()`` to yield claim dicts.  ``run()``
    handles batching, flushing, finalization, and checkpoint persistence.

    Used by: PagerDuty, Airtable, Google Sheets, S3, Elasticsearch,
    MongoDB, HTTP connector.
    """

    @abc.abstractmethod
    def fetch(self) -> Iterator[dict]:
        """Yield claim dicts with ``subject``, ``predicate``, ``object``,
        ``provenance`` keys.

        Optional keys: ``confidence``, ``embedding``, ``payload``,
        ``timestamp``, ``external_ids``.
        """

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """Fetch structured claims, batch, and ingest."""
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)
        batch: list[dict] = []

        for claim_dict in self.fetch():
            batch.append(claim_dict)
            if len(batch) >= batch_size:
                self._flush(db, batch, result)
                batch = []

        if batch:
            self._flush(db, batch, result)

        self._finalize_result(result, start)
        self._save_checkpoint()
        return result


class TextConnector(Connector):
    """Base for connectors that extract unstructured text for ingestion.

    Subclasses implement ``fetch_texts()`` to yield ``(source_id, text)``
    pairs.  ``run()`` handles empty-text filtering, ``db.ingest_text()``
    calls, error accumulation, finalization, and checkpoint persistence.

    Used by: Gmail, Confluence, Notion, Teams, Zoho, GDocs.
    """

    @abc.abstractmethod
    def fetch_texts(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, text)`` pairs for text extraction.

        The subclass handles API pagination, HTML stripping, MIME
        decoding, etc.  Empty-text filtering and error handling are
        handled by ``run()``.
        """

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """Fetch texts and ingest via ``db.ingest_text()``."""
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)

        for source_id, text in self.fetch_texts():
            if not text or not text.strip():
                result.claims_skipped += 1
                continue
            try:
                extraction_result = db.ingest_text(
                    text, source_id=source_id,
                )
                result.claims_ingested += extraction_result.n_valid
                result.prompt_tokens += getattr(
                    extraction_result, "prompt_tokens", 0,
                )
                result.completion_tokens += getattr(
                    extraction_result, "completion_tokens", 0,
                )
            except Exception as exc:
                result.errors.append(f"{source_id}: {exc}")

        self._finalize_result(result, start)
        self._save_checkpoint()
        return result


class HybridConnector(Connector):
    """Base for connectors that produce both structural claims and text.

    Subclasses implement ``fetch()`` for structural claim dicts and
    ``fetch_bodies()`` for ``(source_id, text)`` pairs.  ``run()``
    executes both passes: structural first, then text extraction.

    Used by: GitHub, Salesforce, HubSpot, Linear, ServiceNow, Zendesk,
    Jira.
    """

    def __init__(
        self, *, extraction: str = "heuristic", **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._extraction = extraction

    @abc.abstractmethod
    def fetch(self) -> Iterator[dict]:
        """Yield structural claim dicts (same contract as
        ``StructuredConnector.fetch()``)."""

    @abc.abstractmethod
    def fetch_bodies(self) -> Iterator[tuple[str, str]]:
        """Yield ``(source_id, text)`` pairs for text extraction.

        Called after ``fetch()`` completes.  Connectors that collect
        bodies during ``fetch()`` (via ``self._bodies``) can simply
        return ``iter(self._bodies)``.
        """

    def run(self, db: AttestDB, *, batch_size: int = 500) -> ConnectorResult:
        """Structural pass then text extraction pass."""
        start = time.monotonic()
        result = ConnectorResult(connector_name=self.name)

        # --- Structural pass ---
        batch: list[dict] = []
        for claim_dict in self.fetch():
            batch.append(claim_dict)
            if len(batch) >= batch_size:
                self._flush(db, batch, result)
                batch = []
        if batch:
            self._flush(db, batch, result)

        # --- Text extraction pass ---
        if self._extraction != "none":
            for source_id, body in self.fetch_bodies():
                if not body or not body.strip():
                    continue
                try:
                    er = db.ingest_text(body, source_id=source_id)
                    result.claims_ingested += er.n_valid
                    result.prompt_tokens += getattr(
                        er, "prompt_tokens", 0,
                    )
                    result.completion_tokens += getattr(
                        er, "completion_tokens", 0,
                    )
                except Exception as exc:
                    result.errors.append(f"{source_id}: {exc}")

        self._finalize_result(result, start)
        self._save_checkpoint()
        return result


class QueryConnector(StructuredConnector):
    """Base for connectors that map database/file query results to claims.

    Subclasses implement ``_open_cursor()`` to return an iterator of
    dict-like rows.  ``fetch()`` and ``_row_to_claim()`` are concrete —
    the column mapping dict drives the translation.

    Used by: Postgres, MySQL, MSSQL, SQLite, CSV.
    """

    def __init__(
        self,
        query: str,
        mapping: dict,
        source_type: str = "database_import",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.query = query
        self.mapping = mapping
        self.source_type = source_type

    @abc.abstractmethod
    def _open_cursor(self) -> Iterator[dict]:
        """Open a data source and return an iterator of dict-like rows.

        The implementation handles driver import, connection, cursor
        creation, and query execution.
        """

    def fetch(self) -> Iterator[dict]:
        """Iterate rows from ``_open_cursor()`` and map each to a claim."""
        for row in self._open_cursor():
            try:
                yield self._row_to_claim(row)
            except (KeyError, ValueError) as exc:
                logger.warning("%s: skipping row: %s", self.name, exc)

    def _row_to_claim(self, row: dict) -> dict:
        """Map a single row to a claim dict using ``self.mapping``."""
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
                "source_id": f"{self.name}:{self.query[:80]}",
            },
        }
        if "confidence" in m and m["confidence"] in row:
            claim["confidence"] = float(row[m["confidence"]])
        return claim
