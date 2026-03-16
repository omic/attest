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


class MultiSpaceEmbeddingIndex:
    """Multiple named HNSW indexes for independent embedding spaces.

    Wraps multiple EmbeddingIndex instances, one per named space.
    Backward compatible: a single "default" space behaves identically
    to the original EmbeddingIndex.

    File layout:
        {base_path}.usearch.manifest.json     # {"spaces": {"default": 768, ...}}
        {base_path}.usearch.default            # HNSW binary
        {base_path}.usearch.default.json       # sidecar (id_to_key)
        {base_path}.usearch.structural
        {base_path}.usearch.structural.json
    """

    def __init__(self, spaces: dict[str, int] | None = None):
        """Initialize with a dict of {space_name: ndim}.

        If spaces is None or empty, creates a single "default" space with 768 dims.
        """
        if not spaces:
            spaces = {"default": 768}
        self._spaces: dict[str, EmbeddingIndex] = {}
        self._dims: dict[str, int] = dict(spaces)
        for name, ndim in spaces.items():
            self._spaces[name] = EmbeddingIndex(ndim=ndim)

    @property
    def space_names(self) -> list[str]:
        return list(self._spaces.keys())

    @property
    def ndim(self) -> int:
        """Dimension of the default space (backward compat)."""
        return self._dims.get("default", 768)

    def __len__(self) -> int:
        """Total entries across all spaces."""
        return sum(len(idx) for idx in self._spaces.values())

    def _get_space(self, space: str) -> EmbeddingIndex:
        if space not in self._spaces:
            raise ValueError(f"Unknown embedding space: {space!r}. Available: {self.space_names}")
        return self._spaces[space]

    def add(self, claim_id: str, embedding, space: str = "default") -> None:
        self._get_space(space).add(claim_id, embedding)

    def remove(self, claim_id: str, space: str | None = None) -> bool:
        """Remove from a specific space, or all spaces if space is None."""
        if space is not None:
            return self._get_space(space).remove(claim_id)
        removed = False
        for idx in self._spaces.values():
            if idx.remove(claim_id):
                removed = True
        return removed

    def get(self, claim_id: str, space: str = "default"):
        return self._get_space(space).get(claim_id)

    def search(self, query, top_k: int = 10, space: str = "default") -> list[tuple[str, float]]:
        return self._get_space(space).search(query, top_k)

    def space_len(self, space: str) -> int:
        return len(self._get_space(space))

    def save(self, base_path: str) -> None:
        """Save all spaces and the manifest."""
        manifest = {"spaces": self._dims}
        manifest_path = base_path + ".usearch.manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        for name, idx in self._spaces.items():
            if len(idx) > 0:
                idx.save(base_path + f".usearch.{name}")

    def load(self, base_path: str) -> None:
        """Load all spaces listed in the manifest."""
        for name, idx in self._spaces.items():
            space_path = base_path + f".usearch.{name}"
            if os.path.exists(space_path):
                idx.load(space_path)

    @classmethod
    def load_from(cls, base_path: str) -> MultiSpaceEmbeddingIndex:
        """Load from manifest, auto-migrating old single-file format."""
        manifest_path = base_path + ".usearch.manifest.json"
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            spaces = manifest.get("spaces", {"default": 768})
            idx = cls(spaces=spaces)
            idx.load(base_path)
            return idx

        # Auto-migrate: old .usearch file -> .usearch.default
        old_path = base_path + ".usearch"
        if os.path.exists(old_path):
            new_path = base_path + ".usearch.default"
            os.rename(old_path, new_path)
            old_sidecar = old_path + ".json"
            if os.path.exists(old_sidecar):
                os.rename(old_sidecar, new_path + ".json")
            # Read ndim from the migrated sidecar
            ndim = 768
            new_sidecar = new_path + ".json"
            if os.path.exists(new_sidecar):
                with open(new_sidecar) as f:
                    data = json.load(f)
                ndim = data.get("ndim", 768)
            idx = cls(spaces={"default": ndim})
            idx.load(base_path)
            # Write manifest for future loads
            idx.save(base_path)
            return idx

        # Nothing exists yet
        return cls()
