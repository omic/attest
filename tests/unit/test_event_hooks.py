"""Tests for AttestDB event hook system."""

import tempfile

import pytest

from attestdb.infrastructure.attest_db import AttestDB


@pytest.fixture()
def db(tmp_path):
    db = AttestDB(str(tmp_path / "test"), embedding_dim=None)
    yield db
    db.close()


def _make_claim(subject="A", predicate="relates_to", obj="B", source_id="s1"):
    return dict(
        subject=(subject, "entity"),
        predicate=(predicate, "relation"),
        object=(obj, "entity"),
        provenance={"source_type": "test", "source_id": source_id},
    )


class TestOnOffFire:
    def test_on_registers_callback(self, db):
        called = []
        db.on("test_event", lambda **kw: called.append(kw))
        db._fire("test_event", key="value")
        assert called == [{"key": "value"}]

    def test_off_removes_callback(self, db):
        called = []
        cb = lambda **kw: called.append(kw)
        db.on("test_event", cb)
        db.off("test_event", cb)
        db._fire("test_event", key="value")
        assert called == []

    def test_multiple_callbacks(self, db):
        results = []
        db.on("evt", lambda **kw: results.append("a"))
        db.on("evt", lambda **kw: results.append("b"))
        db._fire("evt")
        assert results == ["a", "b"]

    def test_callback_error_does_not_propagate(self, db):
        """A failing callback should not crash _fire or prevent subsequent callbacks."""
        def bad(**kw):
            raise ValueError("boom")

        results = []
        db.on("evt", bad)
        db.on("evt", lambda **kw: results.append("ok"))
        db._fire("evt")
        assert results == ["ok"]


class TestIngestEvents:
    def test_claim_ingested_fires(self, db):
        events = []
        db.on("claim_ingested", lambda **kw: events.append(kw))
        claim_id = db.ingest(**_make_claim())
        assert len(events) == 1
        assert events[0]["claim_id"] == claim_id

    def test_corroboration_fires_on_second_claim(self, db):
        events = []
        db.on("claim_corroborated", lambda **kw: events.append(kw))
        # First claim — no corroboration
        db.ingest(**_make_claim(source_id="s1"))
        assert len(events) == 0
        # Second claim with same content (different source) — corroboration
        db.ingest(**_make_claim(source_id="s2"))
        assert len(events) == 1
        assert events[0]["count"] == 2

    def test_callback_error_does_not_crash_ingest(self, db):
        def bad_hook(**kw):
            raise RuntimeError("hook failed")

        db.on("claim_ingested", bad_hook)
        # Should not raise
        claim_id = db.ingest(**_make_claim())
        assert claim_id  # ingest still returned


    def test_inquiry_matched_fires(self, db):
        events = []
        # Register inquiry first, then subscribe
        db.ingest_inquiry("Does A relate to B?", ("A", "entity"), ("B", "entity"))
        db.on("inquiry_matched", lambda **kw: events.append(kw))
        # Ingest a claim that matches the inquiry
        claim_id = db.ingest(**_make_claim(subject="A", obj="B"))
        assert len(events) >= 1
        assert events[-1]["claim_id"] == claim_id


class TestRetractEvents:
    def test_source_retracted_fires(self, db):
        events = []
        db.on("source_retracted", lambda **kw: events.append(kw))
        db.ingest(**_make_claim(source_id="src1"))
        db.retract("src1", "test reason")
        assert len(events) == 1
        assert events[0]["source_id"] == "src1"
        assert events[0]["reason"] == "test reason"
        assert len(events[0]["claim_ids"]) >= 1


class TestSnapshotEvents:
    def test_snapshot_created_fires(self, db, tmp_path):
        events = []
        db.on("snapshot_created", lambda **kw: events.append(kw))
        dest = str(tmp_path / "snap")
        db.snapshot(dest)
        assert len(events) == 1
        assert events[0]["dest_path"] == dest


class TestInquiryCreatedEvents:
    def test_inquiry_created_fires(self, db):
        events = []
        db.on("inquiry_created", lambda **kw: events.append(kw))
        claim_id = db.ingest_inquiry(
            "Does A relate to B?", ("A", "entity"), ("B", "entity")
        )
        assert len(events) == 1
        assert events[0]["claim_id"] == claim_id
        assert events[0]["question"] == "Does A relate to B?"
