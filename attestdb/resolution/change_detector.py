"""Change detector — classify records from a connector into creates, updates, and tombstones.

Also provides ``SyncManager`` for scheduled per-source synchronization
with high-water mark tracking.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ChangeSet:
    creates: list[dict] = field(default_factory=list)
    updates: list[dict] = field(default_factory=list)
    tombstones: list[str] = field(default_factory=list)  # record_ids no longer present
    sync_timestamp: float = 0.0
    source_id: str = ""


def detect_changes(
    connector: object,
    source_id: str,
    last_sync_timestamp: float | None = None,
    known_records: dict[str, dict] | None = None,
) -> ChangeSet:
    """Pull records from a connector and classify changes.

    Args:
        connector: Any object with a ``fetch()`` method that returns
            ``list[dict]``. May also have ``fetch_since(timestamp)``
            for incremental sync.
        source_id: Identifier for the source system.
        last_sync_timestamp: If set and connector supports ``fetch_since``,
            only fetch records changed since this time.
        known_records: Dict of record_id -> last-known record dict. Used
            for full-diff detection when incremental fetch is unavailable.

    Returns:
        ChangeSet with classified creates, updates, and tombstones.
    """
    now = time.time()
    changeset = ChangeSet(sync_timestamp=now, source_id=source_id)

    # Fetch records from connector
    records: list[dict]
    if last_sync_timestamp is not None and hasattr(connector, "fetch_since"):
        records = connector.fetch_since(last_sync_timestamp)  # type: ignore[union-attr]
        log.debug("Fetched %d records since %.0f from %s", len(records), last_sync_timestamp, source_id)
    elif hasattr(connector, "fetch"):
        records = connector.fetch()  # type: ignore[union-attr]
        log.debug("Fetched %d records (full) from %s", len(records), source_id)
    else:
        log.warning("Connector has no fetch() method, returning empty changeset")
        return changeset

    if known_records is None:
        # No prior state: everything is a create
        changeset.creates = records
        return changeset

    # Build set of current record IDs
    current_ids: set[str] = set()
    for rec in records:
        rid = str(rec.get("record_id", rec.get("id", "")))
        if not rid:
            continue
        current_ids.add(rid)

        if rid not in known_records:
            changeset.creates.append(rec)
        else:
            # Check if record changed (simple dict comparison)
            old = known_records[rid]
            if _record_changed(old, rec):
                changeset.updates.append(rec)

    # Tombstones: known records not in current fetch
    for rid in known_records:
        if rid not in current_ids:
            changeset.tombstones.append(rid)

    log.info(
        "Change detection for %s: %d creates, %d updates, %d tombstones",
        source_id,
        len(changeset.creates),
        len(changeset.updates),
        len(changeset.tombstones),
    )

    return changeset


def _record_changed(old: dict, new: dict) -> bool:
    """Compare two record dicts, ignoring internal keys (_source_id, etc.)."""
    skip_keys = {"_source_id", "_record_id", "_sync_timestamp"}
    for key in set(old.keys()) | set(new.keys()):
        if key in skip_keys:
            continue
        if old.get(key) != new.get(key):
            return True
    return False


# ---------------------------------------------------------------------------
# SyncManager — scheduled per-source sync with high-water mark
# ---------------------------------------------------------------------------


@dataclass
class SyncSchedule:
    """Per-source sync configuration."""

    source_id: str
    interval_minutes: int = 15
    last_sync_timestamp: float = 0.0
    last_sync_success: bool = True
    change_count_last: int = 0


class SyncManager:
    """Manages per-source sync schedules and high-water marks.

    State is persisted to a JSON sidecar file at ``{state_path}``.
    """

    def __init__(self, state_path: str) -> None:
        self._state_path = state_path
        self._schedules: dict[str, SyncSchedule] = {}
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule_sync(self, source_id: str, interval_minutes: int = 15) -> SyncSchedule:
        """Register or update a sync schedule for a source."""
        if source_id in self._schedules:
            self._schedules[source_id].interval_minutes = interval_minutes
        else:
            self._schedules[source_id] = SyncSchedule(
                source_id=source_id,
                interval_minutes=interval_minutes,
            )
        self._save_state()
        return self._schedules[source_id]

    def get_schedule(self, source_id: str) -> SyncSchedule | None:
        return self._schedules.get(source_id)

    def sources_due(self) -> list[str]:
        """Return source IDs that are due for sync (elapsed >= interval)."""
        now = time.time()
        due: list[str] = []
        for sid, sched in self._schedules.items():
            elapsed = (now - sched.last_sync_timestamp) / 60.0
            if elapsed >= sched.interval_minutes:
                due.append(sid)
        return due

    def record_sync(
        self,
        source_id: str,
        sync_timestamp: float,
        success: bool = True,
        change_count: int = 0,
    ) -> None:
        """Record a completed sync attempt, advancing the high-water mark on success."""
        sched = self._schedules.get(source_id)
        if sched is None:
            sched = SyncSchedule(source_id=source_id)
            self._schedules[source_id] = sched

        sched.last_sync_success = success
        sched.change_count_last = change_count
        if success:
            sched.last_sync_timestamp = sync_timestamp
        self._save_state()

    def get_high_water_mark(self, source_id: str) -> float:
        """Return the last successful sync timestamp for a source (0.0 if never synced)."""
        sched = self._schedules.get(source_id)
        if sched is None:
            return 0.0
        return sched.last_sync_timestamp

    def run_sync(
        self,
        source_id: str,
        connector: object,
        known_records: dict[str, dict] | None = None,
    ) -> ChangeSet:
        """Run a single sync for *source_id*: detect changes and advance high-water mark."""
        hwm = self.get_high_water_mark(source_id)
        last_ts = hwm if hwm > 0 else None

        changeset = detect_changes(
            connector,
            source_id,
            last_sync_timestamp=last_ts,
            known_records=known_records,
        )

        change_count = len(changeset.creates) + len(changeset.updates) + len(changeset.tombstones)
        self.record_sync(source_id, changeset.sync_timestamp, success=True, change_count=change_count)

        return changeset

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path) as f:
                data = json.load(f)
            for entry in data:
                self._schedules[entry["source_id"]] = SyncSchedule(**entry)
        except Exception:
            log.debug("Failed to load sync state from %s", self._state_path, exc_info=True)

    def _save_state(self) -> None:
        entries = []
        for sched in self._schedules.values():
            entries.append({
                "source_id": sched.source_id,
                "interval_minutes": sched.interval_minutes,
                "last_sync_timestamp": sched.last_sync_timestamp,
                "last_sync_success": sched.last_sync_success,
                "change_count_last": sched.change_count_last,
            })
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp, self._state_path)
