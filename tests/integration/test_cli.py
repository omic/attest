"""Integration tests for the CLI (python -m attestdb)."""

import json
import subprocess
import sys

import pytest


def _run_cli(*args, timeout=30):
    """Run the attestdb CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "attestdb"] + list(args),
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIHelp:
    def test_help(self):
        rc, out, _ = _run_cli("--help")
        assert rc == 0
        assert "knowledge you can verify" in out.lower()
        assert "ingest" in out
        assert "stats" in out
        assert "query" in out


class TestCLIIngest:
    def test_ingest_json_heuristic(self, tmp_path):
        """CLI ingest with heuristic extraction."""
        json_file = tmp_path / "chat.json"
        json_file.write_text(json.dumps([
            {"role": "user", "content": "What is BRCA1?"},
            {"role": "assistant", "content": (
                "BRCA1 binds RAD51 for DNA repair. Olaparib inhibits PARP1."
            )},
        ]))

        db_path = str(tmp_path / "cli_test.db")
        rc, out, err = _run_cli(
            "ingest", str(json_file),
            "--db", db_path,
            "--extraction", "heuristic",
        )
        assert rc == 0
        assert "ingested" in out.lower() or "extracted" in out.lower()
        assert "Database totals" in out


class TestCLIStats:
    def test_stats(self, tmp_path):
        """CLI stats on a fresh DB."""
        # Create a DB first
        json_file = tmp_path / "chat.json"
        json_file.write_text(json.dumps([
            {"role": "user", "content": "What is BRCA1?"},
            {"role": "assistant", "content": "BRCA1 binds RAD51 for homologous recombination DNA repair. Olaparib inhibits PARP1."},
        ]))
        db_path = str(tmp_path / "stats_test.db")
        _run_cli("ingest", str(json_file), "--db", db_path, "--extraction", "heuristic")

        rc, out, _ = _run_cli("stats", db_path)
        assert rc == 0
        assert "Claims:" in out


class TestCLIQuery:
    def test_query_entity(self, tmp_path):
        """CLI query after ingestion."""
        json_file = tmp_path / "chat.json"
        json_file.write_text(json.dumps([
            {"role": "user", "content": "What is BRCA1?"},
            {"role": "assistant", "content": (
                "BRCA1 binds RAD51 for homologous recombination DNA repair. "
                "Olaparib inhibits PARP1 in cancer cells with BRCA mutations."
            )},
        ]))
        db_path = str(tmp_path / "query_test.db")
        _run_cli("ingest", str(json_file), "--db", db_path, "--extraction", "heuristic")

        rc, out, _ = _run_cli("query", db_path, "brca1")
        assert rc == 0
        assert "brca1" in out.lower()
