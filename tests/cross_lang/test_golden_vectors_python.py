"""Python-side golden vector verification.

Reads golden_vectors.json and verifies that the Python implementations of
normalization, hashing, chain hashing, and confidence produce identical
outputs to the recorded vectors. This closes the cross-language contract
loop: vectors are generated from Python, verified by Rust AND back-verified
by Python.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from attestdb.core.confidence import tier1_confidence
from attestdb.core.hashing import compute_chain_hash, compute_claim_id, compute_content_id
from attestdb.core.normalization import normalize_entity_id

VECTORS_PATH = Path(__file__).parent / "golden_vectors.json"


@pytest.fixture(scope="module")
def vectors():
    with open(VECTORS_PATH) as f:
        return json.load(f)


class TestNormalizationVectors:
    def test_all_normalization_vectors(self, vectors):
        for v in vectors["normalization"]:
            result = normalize_entity_id(v["input"])
            assert result == v["expected"], (
                f"Normalization mismatch for {v['description']!r}: "
                f"input={v['input']!r}, expected={v['expected']!r}, got={result!r}"
            )

    def test_vector_count(self, vectors):
        assert len(vectors["normalization"]) == 59


class TestHashingVectors:
    def test_all_claim_id_vectors(self, vectors):
        for v in vectors["hashing"]:
            if v["type"] == "claim_id":
                args = v["args"]
                result = compute_claim_id(*args)
                assert result == v["expected"], (
                    f"claim_id mismatch for {v['description']!r}: "
                    f"expected={v['expected']!r}, got={result!r}"
                )

    def test_all_content_id_vectors(self, vectors):
        for v in vectors["hashing"]:
            if v["type"] == "content_id":
                result = compute_content_id(*v["args"])
                assert result == v["expected"], (
                    f"content_id mismatch for {v['description']!r}: "
                    f"expected={v['expected']!r}, got={result!r}"
                )

    def test_hashing_vector_count(self, vectors):
        assert len(vectors["hashing"]) == 20


class TestChainHashVectors:
    def test_all_chain_hash_vectors(self, vectors):
        for v in vectors["chain_hash"]:
            result = compute_chain_hash(v["prev_chain_hash"], v["claim_id"])
            assert result == v["expected"], (
                f"chain_hash mismatch for {v['description']!r}: "
                f"expected={v['expected']!r}, got={result!r}"
            )

    def test_chain_hash_vector_count(self, vectors):
        assert len(vectors["chain_hash"]) == 13


class TestConfidenceVectors:
    def test_all_confidence_vectors(self, vectors):
        for v in vectors["confidence"]:
            result = tier1_confidence(v["source_type"])
            assert result == pytest.approx(v["expected"]), (
                f"Confidence mismatch for {v['description']!r}: "
                f"source_type={v['source_type']!r}, expected={v['expected']}, got={result}"
            )

    def test_confidence_vector_count(self, vectors):
        assert len(vectors["confidence"]) == 26


class TestVectorIntegrity:
    def test_total_vector_count(self, vectors):
        total = (
            len(vectors["normalization"])
            + len(vectors["hashing"])
            + len(vectors["chain_hash"])
            + len(vectors["confidence"])
        )
        assert total == 118
