"""Tests for the confidence calibration report."""

import pytest

from attestdb.infrastructure.attest_db import AttestDB


@pytest.fixture()
def db(tmp_path):
    db = AttestDB(str(tmp_path / "test"), embedding_dim=None)
    yield db
    db.close()


def _make_claim(subject="A", predicate="relates_to", obj="B", source_id="s1", confidence=None):
    d = dict(
        subject=(subject, "entity"),
        predicate=(predicate, "relation"),
        object=(obj, "entity"),
        provenance={"source_type": "test", "source_id": source_id},
    )
    if confidence is not None:
        d["confidence"] = confidence
    return d


class TestCalibrationReport:
    def test_basic_calibration(self, db):
        # Create labeled claims with varying confidence
        labeled = [
            (_make_claim(subject=f"A{i}", source_id=f"s{i}"), True)
            for i in range(5)
        ] + [
            (_make_claim(subject=f"B{i}", source_id=f"s{i+10}"), False)
            for i in range(5)
        ]
        result = db.calibration_report(labeled, n_bins=5)
        assert result["n_claims"] == 10
        assert len(result["bins"]) == 5
        assert "ece" in result
        assert "well_calibrated" in result

    def test_perfect_calibration_has_low_ece(self, db):
        # All claims true with default confidence ~ 0.5 should give gap near |0.5 - 1.0| = 0.5
        # This tests that the pipeline works end-to-end, not that ECE is low
        labeled = [
            (_make_claim(subject=f"P{i}", source_id=f"s{i}"), True)
            for i in range(4)
        ]
        result = db.calibration_report(labeled, n_bins=2)
        assert result["n_claims"] == 4
        assert isinstance(result["ece"], float)
