"""Elasticsearch connector — maps search hits to claims via field mapping.

Uses the scroll API to paginate through large result sets efficiently.

Usage::

    from attestdb.connectors.elasticsearch import ElasticsearchConnector

    conn = ElasticsearchConnector(
        hosts="http://localhost:9200",
        index="publications",
        mapping={
            "subject": "author",
            "predicate": "published_in",
            "object": "journal",
            "subject_type": "person",
            "object_type": "journal",
        },
        query={"bool": {"must": [{"term": {"status": "published"}}]}},
    )
    result = conn.run(db)
"""

from __future__ import annotations

import logging
from typing import Iterator

from attestdb.connectors.base import StructuredConnector

logger = logging.getLogger(__name__)


class ElasticsearchConnector(StructuredConnector):
    """Import claims from an Elasticsearch index with field mapping."""

    name = "elasticsearch"

    def __init__(
        self,
        hosts: list[str] | str,
        index: str,
        mapping: dict,
        query: dict | None = None,
        source_type: str = "elasticsearch",
        max_items: int = 10000,
        scroll: str = "2m",
        size: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hosts = [hosts] if isinstance(hosts, str) else hosts
        self.index = index
        self.mapping = mapping
        self.query = query or {"match_all": {}}
        self.source_type = source_type
        self.max_items = max_items
        self.scroll = scroll
        self.size = size

    def fetch(self) -> Iterator[dict]:
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError("pip install elasticsearch for Elasticsearch support")

        es = Elasticsearch(self.hosts)
        try:
            resp = es.search(
                index=self.index,
                query=self.query,
                scroll=self.scroll,
                size=self.size,
            )
            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]
            count = 0

            while hits and count < self.max_items:
                for hit in hits:
                    if count >= self.max_items:
                        break
                    try:
                        yield self._hit_to_claim(hit)
                        count += 1
                    except (KeyError, ValueError, TypeError) as exc:
                        logger.warning("elasticsearch: skipping hit: %s", exc)

                if count >= self.max_items:
                    break
                resp = es.scroll(scroll_id=scroll_id, scroll=self.scroll)
                scroll_id = resp.get("_scroll_id")
                hits = resp["hits"]["hits"]

            # Clean up server-side scroll context
            if scroll_id:
                try:
                    es.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    pass
        finally:
            es.close()

    def _hit_to_claim(self, hit: dict) -> dict:
        m = self.mapping
        source = hit["_source"]
        claim = {
            "subject": (
                str(source[m["subject"]]),
                m.get("subject_type", "entity"),
            ),
            "predicate": (
                str(source[m["predicate"]]),
                m.get("predicate_type", "relates_to"),
            ),
            "object": (
                str(source[m["object"]]),
                m.get("object_type", "entity"),
            ),
            "provenance": {
                "source_type": self.source_type,
                "source_id": f"elasticsearch:{self.index}:{hit['_id']}",
            },
        }
        if "confidence" in m and m["confidence"] in source:
            claim["confidence"] = float(source[m["confidence"]])
        return claim
