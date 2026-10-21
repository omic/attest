"""Execute the public synthetic example using the installed engine."""

import importlib.util
from pathlib import Path


def test_evidence_withdrawal_reopens_coverage_gap(tmp_path):
    path = Path(__file__).resolve().parents[2] / "examples/compliance_evidence.py"
    spec = importlib.util.spec_from_file_location("compliance_reference", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    result = demo.run_demo(str(tmp_path / "compliance.attest"))
    assert result["uncovered_before"] == list(demo.CONTROLS)
    assert result["uncovered_after_evidence"] == [demo.CONTROLS[1]]
    assert result["uncovered_after_withdrawal"] == list(demo.CONTROLS)
