"""Disk-based LLM response cache.

Caches raw LLM responses keyed by sha256(prompt) before any parsing.
Satisfies the hard rule: cache expensive API results before analysis.

Usage:
    cache = LLMCache("data/llm_cache")
    cached = cache.get(prompt_text)
    if cached is None:
        response = llm_client.complete(prompt_text)
        cache.put(prompt_text, response, model="groq/...", tokens={"prompt": 100, "completion": 50})
        # Now parse response
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCache:
    """Disk-based cache for raw LLM responses.

    Keyed by sha256 of prompt text. Stores raw response JSON
    alongside metadata (timestamp, model, token counts).
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prompt: str) -> dict | None:
        """Look up cached response. Returns None on miss."""
        path = self._path(self._key(prompt))
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache read error for %s: %s", path.name, exc)
            self._misses += 1
            return None

    def put(
        self,
        prompt: str,
        response: str | dict,
        *,
        model: str = "",
        tokens: dict | None = None,
        source_id: str = "",
    ) -> Path:
        """Write raw response to cache. Returns the cache file path."""
        key = self._key(prompt)
        path = self._path(key)
        payload = {
            "timestamp": time.time(),
            "model": model,
            "tokens": tokens or {},
            "source_id": source_id,
            "prompt_hash": key,
            "prompt_length": len(prompt),
            "response": response,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @property
    def stats(self) -> dict:
        """Return hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }
