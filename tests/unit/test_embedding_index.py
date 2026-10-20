"""Unit tests for EmbeddingIndex — search, get, save/load, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from attestdb.core.errors import DimensionalityError
from attestdb.infrastructure.embedding_index import EmbeddingIndex


@pytest.fixture()
def idx():
    return EmbeddingIndex(ndim=8)


class TestAdd:
    def test_add_and_len(self, idx):
        vec = np.random.randn(8).astype(np.float32)
        idx.add("claim-1", vec)
        assert len(idx) == 1

    def test_duplicate_add_is_noop(self, idx):
        vec = np.random.randn(8).astype(np.float32)
        idx.add("claim-1", vec)
        idx.add("claim-1", vec)
        assert len(idx) == 1

    def test_wrong_dim_raises(self, idx):
        with pytest.raises(DimensionalityError):
            idx.add("claim-1", np.ones(16))


class TestGet:
    def test_get_returns_stored_embedding(self, idx):
        vec = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        idx.add("claim-1", vec)
        result = idx.get("claim-1")
        assert result is not None
        np.testing.assert_allclose(result, vec, atol=1e-5)


class TestSearch:
    def test_search_returns_results(self, idx):
        # Add 3 known vectors
        idx.add("a", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        idx.add("c", np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32))

        # Search for vector closest to "a"
        results = idx.search(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), top_k=3)
        assert len(results) == 3
        # Results are (claim_id, distance) tuples
        ids = [r[0] for r in results]
        assert "a" in ids


class TestSaveLoad:
    def test_save_load_roundtrip(self, idx, tmp_path):
        vecs = {
            "claim-1": np.random.randn(8).astype(np.float32),
            "claim-2": np.random.randn(8).astype(np.float32),
            "claim-3": np.random.randn(8).astype(np.float32),
        }
        for cid, v in vecs.items():
            idx.add(cid, v)

        path = str(tmp_path / "test.usearch")
        idx.save(path)

        # Load into fresh index
        idx2 = EmbeddingIndex(ndim=8)
        idx2.load(path)
        assert len(idx2) == 3

        # Verify stored embeddings survive roundtrip (usearch quantizes, so use loose tolerance)
        for cid, v in vecs.items():
            retrieved = idx2.get(cid)
            assert retrieved is not None
            np.testing.assert_allclose(retrieved, v, atol=0.01)

    def test_search_after_load(self, idx, tmp_path):
        idx.add("a", np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        idx.add("b", np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32))
        path = str(tmp_path / "test.usearch")
        idx.save(path)

        idx2 = EmbeddingIndex.load_from(path)
        results = idx2.search(np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32), top_k=2)
        assert len(results) == 2
        # Closest to [1,0,...] should be "a"
        assert results[0][0] == "a"


def _mark_saved_files(path):
    import os
    from pathlib import Path

    marker = 1_600_000_000_000_000_000
    for file in (Path(path), Path(str(path) + ".json")):
        os.utime(file, ns=(marker, marker))
    return marker


def test_unchanged_index_keeps_saved_files_untouched(tmp_path):
    from pathlib import Path

    index = EmbeddingIndex(ndim=4)
    index.add("a", [1, 0, 0, 0])
    path = str(tmp_path / "index.usearch")
    index.save(path)
    marker = _mark_saved_files(path)
    index.save(path)
    assert Path(path).stat().st_mtime_ns == marker
    assert Path(path + ".json").stat().st_mtime_ns == marker
    loaded = EmbeddingIndex.load_from(path)
    loaded.save(path)
    assert Path(path).stat().st_mtime_ns == marker


def test_changed_index_and_model_are_persisted(tmp_path):
    index = EmbeddingIndex(ndim=4)
    path = str(tmp_path / "index.usearch")
    index.add("a", [1, 0, 0, 0])
    index.save(path)
    index.add("b", [0, 1, 0, 0])
    index.save(path)
    assert len(EmbeddingIndex.load_from(path)) == 2
    index.remove("a")
    index.model = "caller-model-v1"
    index.save(path)
    loaded = EmbeddingIndex.load_from(path)
    assert len(loaded) == 1 and loaded.model == "caller-model-v1"
    assert loaded.search([0, 1, 0, 0], top_k=1)[0][0] == "b"


def test_missing_sidecar_and_new_destination_are_written(tmp_path):
    from pathlib import Path

    index = EmbeddingIndex(ndim=4)
    index.add("a", [1, 0, 0, 0])
    path = str(tmp_path / "index.usearch")
    index.save(path)
    Path(path + ".json").unlink()
    index.save(path)
    assert len(EmbeddingIndex.load_from(path)) == 1
    other = str(tmp_path / "other.usearch")
    index.save(other)
    assert len(EmbeddingIndex.load_from(other)) == 1
