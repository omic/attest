"""Continuous ingestion scheduler for connectors.

Runs connectors on a recurring interval in background threads, using the
existing checkpoint system for incremental fetching. Each connector gets
its own thread so a slow or failing connector doesn't block others.

Usage::

    db = attestdb.quickstart("my.db")
    db.sync("slack", interval=300, token="xoxb-...")
    db.sync("github", interval=600, token="ghp_...")
    print(db.sync_status())
    db.sync_stop_all()

The scheduler follows the same threading pattern as SnapshotManager:
``threading.Event().wait(timeout)`` for interruptible sleep.
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from attestdb.connectors.base import Connector, ConnectorResult
    from attestdb.infrastructure.attest_db import AttestDB

from attestdb.infrastructure.event_bus import EventType

logger = logging.getLogger(__name__)


@dataclass
class SyncHandle:
    """Status handle for a scheduled connector."""

    name: str
    interval: float
    status: str = "running"  # running | paused | stopped | error
    last_run: datetime | None = None
    last_result: ConnectorResult | None = None
    next_run: datetime | None = None
    error_count: int = 0
    total_runs: int = 0
    total_claims: int = 0
    last_error: str | None = None


class ConnectorScheduler:
    """Background scheduler that runs connectors on intervals.

    Each connector runs in its own daemon thread. Failures are logged and
    retried with exponential backoff (capped at the configured interval).
    Random jitter (default 10%) prevents thundering herd when multiple
    connectors share the same interval.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handles: dict[str, SyncHandle] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._connectors: dict[str, Connector] = {}
        self._dbs: dict[str, AttestDB] = {}

    def schedule(
        self,
        connector: Connector,
        db: AttestDB,
        interval: float = 300.0,
        jitter: float = 0.1,
        run_immediately: bool = True,
    ) -> SyncHandle:
        """Schedule a connector for periodic execution.

        Args:
            connector: The connector instance to run.
            db: AttestDB instance to ingest into.
            interval: Seconds between runs (default: 300 = 5 min).
            jitter: Random jitter fraction (0.1 = +/- 10%).
            run_immediately: Run once before waiting for the first interval.

        Returns:
            SyncHandle for monitoring and control.
        """
        name = connector.name
        with self._lock:
            if name in self._handles and self._handles[name].status == "running":
                logger.warning("Connector %s already scheduled — stopping old one", name)
                self._stop_one(name)

            handle = SyncHandle(name=name, interval=interval)
            self._handles[name] = handle
            self._connectors[name] = connector
            self._dbs[name] = db

            stop_event = threading.Event()
            self._stop_events[name] = stop_event

            thread = threading.Thread(
                target=self._loop,
                args=(name, interval, jitter, run_immediately),
                daemon=True,
                name=f"attest-sync-{name}",
            )
            self._threads[name] = thread
            thread.start()

        return handle

    def stop(self, name: str) -> None:
        """Stop a scheduled connector."""
        with self._lock:
            ev = self._stop_events.pop(name, None)
            if ev:
                ev.set()
            thread = self._threads.pop(name, None)
            if name in self._handles:
                self._handles[name].status = "stopped"
            self._connectors.pop(name, None)
            self._dbs.pop(name, None)
        # Join OUTSIDE the lock to avoid deadlock
        if thread and thread.is_alive():
            thread.join(timeout=5.0)

    def stop_all(self) -> None:
        """Stop all scheduled connectors."""
        threads_to_join: list[threading.Thread] = []
        with self._lock:
            for name in list(self._handles):
                ev = self._stop_events.pop(name, None)
                if ev:
                    ev.set()
                thread = self._threads.pop(name, None)
                if thread:
                    threads_to_join.append(thread)
                if name in self._handles:
                    self._handles[name].status = "stopped"
                self._connectors.pop(name, None)
                self._dbs.pop(name, None)
        # Join OUTSIDE the lock to avoid deadlock
        for thread in threads_to_join:
            if thread.is_alive():
                thread.join(timeout=5.0)

    def pause(self, name: str) -> None:
        """Pause a connector (thread keeps running but skips execution)."""
        with self._lock:
            if name in self._handles:
                self._handles[name].status = "paused"

    def resume(self, name: str) -> None:
        """Resume a paused connector."""
        with self._lock:
            if name in self._handles and self._handles[name].status == "paused":
                self._handles[name].status = "running"

    def run_now(self, name: str) -> None:
        """Trigger an immediate run of a connector (wakes the sleep)."""
        with self._lock:
            ev = self._stop_events.get(name)
            if ev and name in self._handles:
                self._handles[name]._run_now = True  # type: ignore[attr-defined]
                ev.set()  # wake the wait()
                # Re-create the event so subsequent waits work
                # (the thread will reset it after waking)

    def status(self) -> list[dict]:
        """Return status of all scheduled connectors."""
        with self._lock:
            result = []
            for name, handle in self._handles.items():
                result.append({
                    "name": handle.name,
                    "interval": handle.interval,
                    "status": handle.status,
                    "last_run": handle.last_run.isoformat() if handle.last_run else None,
                    "next_run": handle.next_run.isoformat() if handle.next_run else None,
                    "total_runs": handle.total_runs,
                    "total_claims": handle.total_claims,
                    "error_count": handle.error_count,
                    "last_error": handle.last_error,
                })
            return result

    def _stop_one(self, name: str) -> None:
        """Stop a connector (must hold self._lock). Signals stop but does NOT join."""
        ev = self._stop_events.pop(name, None)
        if ev:
            ev.set()
        self._threads.pop(name, None)
        if name in self._handles:
            self._handles[name].status = "stopped"
        self._connectors.pop(name, None)
        self._dbs.pop(name, None)

    def _loop(
        self, name: str, interval: float, jitter: float, run_immediately: bool,
    ) -> None:
        """Background loop for one connector."""
        stop_event = self._stop_events.get(name)
        if not stop_event:
            return

        backoff = 0.0

        if not run_immediately:
            # Wait for first interval before running
            wait_time = self._jittered(interval, jitter)
            with self._lock:
                handle = self._handles.get(name)
                if handle:
                    handle.next_run = datetime.now(timezone.utc) + timedelta(seconds=wait_time)
            if stop_event.wait(wait_time):
                # Stop was requested — check if it's a run_now or real stop
                if not self._check_run_now(name):
                    return

        while not stop_event.is_set():
            handle = self._handles.get(name)
            if not handle:
                return

            # Skip if paused
            if handle.status == "paused":
                if stop_event.wait(0.1):
                    if not self._check_run_now(name):
                        return
                continue

            # Run the connector
            connector = self._connectors.get(name)
            db = self._dbs.get(name)
            if not connector or not db:
                return

            try:
                logger.info("Sync: running %s", name)
                result = connector.run(db)
                now = datetime.now(timezone.utc)

                with self._lock:
                    handle.last_run = now
                    handle.last_result = result
                    handle.total_runs += 1
                    handle.total_claims += result.claims_ingested
                    handle.status = "running"
                    if result.errors:
                        handle.last_error = result.errors[-1]
                    backoff = 0.0  # Reset backoff on success

                logger.info(
                    "Sync %s: ingested %d, skipped %d in %.1fs",
                    name, result.claims_ingested, result.claims_skipped,
                    result.elapsed_seconds,
                )

                # Fire event
                db._fire(
                    EventType.SYNC_COMPLETED,
                    connector_name=name,
                    claims_ingested=result.claims_ingested,
                )

            except Exception as exc:
                with self._lock:
                    handle.error_count += 1
                    handle.last_error = str(exc)
                    handle.status = "error"
                    # Exponential backoff: 30s, 60s, 120s, ... capped at interval
                    backoff = min(interval, max(30.0, backoff * 2))

                logger.warning(
                    "Sync %s failed (attempt %d): %s — backing off %.0fs",
                    name, handle.error_count, exc, backoff,
                )

            # Wait for next interval (with jitter and backoff)
            wait_time = self._jittered(interval + backoff, jitter)
            with self._lock:
                if handle:
                    handle.next_run = datetime.now(timezone.utc) + timedelta(seconds=wait_time)

            if stop_event.wait(wait_time):
                if not self._check_run_now(name):
                    return
                # run_now was requested — loop again immediately
                # Reset the event for the next wait
                stop_event.clear()

    def _check_run_now(self, name: str) -> bool:
        """Check if the wakeup was a run_now (return True) or a real stop (return False)."""
        handle = self._handles.get(name)
        if handle and getattr(handle, "_run_now", False):
            handle._run_now = False  # type: ignore[attr-defined]
            stop_event = self._stop_events.get(name)
            if stop_event:
                stop_event.clear()
            return True
        return False

    @staticmethod
    def _jittered(base: float, jitter: float) -> float:
        """Add random jitter to a base interval."""
        if jitter <= 0:
            return base
        delta = base * jitter
        return base + random.uniform(-delta, delta)
