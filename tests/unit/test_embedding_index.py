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
