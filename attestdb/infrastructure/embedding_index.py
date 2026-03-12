"""Embedding index using usearch for semantic similarity search."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
from usearch.index import Index

from attestdb.core.errors import DimensionalityError

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """HNSW embedding index with bidirectional claim_id <-> integer key mapping."""

    def __init__(self, ndim: int = 768):
        self._ndim = ndim
        self._index = Index(ndim=ndim, metric="cos")
        self._id_to_key: dict[str, int] = {}
        self._key_to_id: dict[int, str] = {}
        self._next_key = 0

    @property
    def ndim(self) -> int:
        return self._ndim

    def __len__(self) -> int:
        return len(self._id_to_key)

    def add(self, claim_id: str, embedding: list[float] | np.ndarray) -> None:
        vec = np.asarray(embedding, dtype=np.float32)
        if vec.shape[0] != self._ndim:
            raise DimensionalityError(
                f"Expected {self._ndim}-dim embedding, got {vec.shape[0]}-dim"
            )
        if claim_id in self._id_to_key:
            return  # Already indexed
        key = self._next_key
        self._next_key += 1
        self._id_to_key[claim_id] = key
        self._key_to_id[key] = claim_id
        self._index.add(key, vec)

    def remove(self, claim_id: str) -> bool:
        """Remove a claim_id from the index. Returns True if it was present."""
        key = self._id_to_key.pop(claim_id, None)
        if key is None:
            return False
        self._key_to_id.pop(key, None)
        if hasattr(self._index, "remove"):
            try:
                self._index.remove(key)
            except Exception as e:
                logger.warning("Failed to remove key %d from usearch index: %s", key, e)
        return True

    def get(self, claim_id: str) -> np.ndarray | None:
        """Retrieve the stored embedding for a claim_id. Returns None if not found."""
        key = self._id_to_key.get(claim_id)
        if key is None:
            return None
        try:
            return np.asarray(self._index.get(key), dtype=np.float32)
        except Exception as e:
            logger.warning("Failed to retrieve embedding for %s: %s", claim_id, e)
            return None

    def search(
        self, query: list[float] | np.ndarray, top_k: int = 10
    ) -> list[tuple[str, float]]:
        if len(self._id_to_key) == 0:
            return []
        vec = np.asarray(query, dtype=np.float32)
        if vec.shape[0] != self._ndim:
            raise DimensionalityError(
                f"Expected {self._ndim}-dim query, got {vec.shape[0]}-dim"
            )
        actual_k = min(top_k, len(self._id_to_key))
        matches = self._index.search(vec, actual_k)
        results = []
        for key, dist in zip(matches.keys, matches.distances):
            key = int(key)
            if key in self._key_to_id:
                results.append((self._key_to_id[key], float(dist)))
        return results

    def save(self, path: str) -> None:
        self._index.save(path)
        sidecar = path + ".json"
        with open(sidecar, "w") as f:
            json.dump(
                {
                    "ndim": self._ndim,
                    "id_to_key": self._id_to_key,
                    "next_key": self._next_key,
                },
                f,
            )

    def load(self, path: str) -> None:
        self._index.load(path)
        sidecar = path + ".json"
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                data = json.load(f)
            self._ndim = data["ndim"]
            self._id_to_key = data["id_to_key"]
            self._next_key = data["next_key"]
        else:
            logger.warning("Embedding index sidecar missing at %s", sidecar)
        # Always rebuild reverse mapping from id_to_key
        self._key_to_id = {int(v): k for k, v in self._id_to_key.items()}

    @classmethod
    def load_from(cls, path: str) -> EmbeddingIndex:
        sidecar = path + ".json"
        ndim = 768
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                data = json.load(f)
            ndim = data["ndim"]
        idx = cls(ndim=ndim)
        idx.load(path)
        return idx
