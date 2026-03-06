"""Shared pytest fixtures."""

import os
import tempfile

import pytest


@pytest.fixture
def backend():
    """The storage backend being tested."""
    return "rust"


@pytest.fixture
def tmp_db_path(tmp_path):
    """Temporary database path."""
    return str(tmp_path / "test.attest")


@pytest.fixture
def tmp_index_path(tmp_path):
    """Temporary embedding index path."""
    return str(tmp_path / "test.usearch")


@pytest.fixture
def make_store(tmp_path):
    """Factory fixture: returns a function that creates a RustStore.

    Usage:
        store = make_store()  # uses default path
        store = make_store("custom.attest")  # uses custom filename
    """
    stores = []

    def _make(name=None):
        from attest_rust import RustStore
        path = str(tmp_path / (name or "test.attest"))
        store = RustStore(path)
        stores.append(store)
        return store

    yield _make

    for s in stores:
        try:
            s.close()
        except Exception:
            pass


@pytest.fixture
def make_db(tmp_path):
    """Factory fixture: returns a function that creates AttestDB.

    Usage:
        db = make_db()
        db = make_db(embedding_dim=4)
    """
    dbs = []

    def _make(name="test", embedding_dim=768, strict=False):
        from attestdb.infrastructure.attest_db import AttestDB
        path = str(tmp_path / name)
        db = AttestDB(path, embedding_dim=embedding_dim, strict=strict)
        dbs.append(db)
        return db

    yield _make

    for db in dbs:
        try:
            db.close()
        except Exception:
            pass
