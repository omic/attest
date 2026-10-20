"""Public memory persists through the real store; installer preserves client config."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib

import pytest

from attestdb import AttestDB, brain, mcp_install, mcp_server
from attestdb.mcp_tools_learning import _SUBJECT_BYTES, recall, remember


def test_real_mcp_protocol_remembers_across_server_restarts(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {key: value for key, value in os.environ.items()
           if not key.endswith("_API_KEY") and not key.startswith("AWS_")}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-I", "-m", "attestdb.mcp_server", "--db", str(tmp_path / "protocol.attest")],
        env=env,
    )

    async def exchange(name, arguments):
        async with asyncio.timeout(30):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    assert name in {tool.name for tool in tools.tools}
                    result = await session.call_tool(name, arguments)
                    assert not result.isError
                    return "\n".join(item.text for item in result.content if item.type == "text")

    saved = asyncio.run(exchange("attest_learned", {
        "subject": "protocol.py", "description": "Check session expiry", "type": "fix",
    }))
    assert "claim_id" in saved
    recalled = asyncio.run(exchange("attest_check_file", {"path": "protocol.py"}))
    assert "Check session expiry" in recalled
    assert "agent_reported" in recalled


def test_memory_survives_reopen_and_keeps_negative_results(tmp_path):
    path = str(tmp_path / "brain.attest")
    db = AttestDB(path, embedding_dim=None)
    try:
        result = remember(db, "auth.py", "Expired sessions must be rejected", "bug")
        remember(db, "auth.py", "Retrying the same expired token did not help", "negative")
    finally:
        db.close()
    db = AttestDB(path, embedding_dim=None)
    try:
        records = recall(db, "auth.py")
        assert {r["kind"] for r in records} == {"bug", "negative"}
        assert result["claim_id"] in {r["claim_id"] for r in records}
        assert all(0 < r["confidence"] <= 1 for r in records)
        assert all(r["verification"] == "agent_reported" for r in records)
        assert recall(db, "unrelated.py") == []
    finally:
        db.close()


def test_default_mcp_exposes_brain_tools_and_stores_session(tmp_path):
    db = AttestDB(str(tmp_path / "brain.attest"), embedding_dim=None)
    mcp_server.configure(db)
    try:
        expected = {"attest_learned", "attest_negative_result", "attest_check_file",
                    "attest_get_prior_approaches", "attest_session_end"}
        assert expected <= {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
        mcp_server.attest_session_end("partial", "Investigated login", "Check expiry", ["auth.py"])
        rows = mcp_server.attest_check_file("auth.py")
        assert rows[0]["outcome"] == "partial"
        assert rows[0]["next_steps"] == "Check expiry"
    finally:
        db.close()
        mcp_server.configure(None)


def test_file_recall_distinguishes_case_preserved_paths(tmp_path):
    db = AttestDB(str(tmp_path / "case.attest"), embedding_dim=None)
    try:
        remember(db, "Auth.py", "Uppercase file finding", "tip")
        remember(db, "auth.py", "Lowercase file finding", "tip")
        rows = recall(db, "auth.py", exact=True)
        assert [row["description"] for row in rows] == ["Lowercase file finding"]
    finally:
        db.close()


@pytest.mark.parametrize("subject,description,kind", [("", "text", "bug"),
                         ("topic", "", "bug"), ("topic", "text", "verified_truth")])
def test_invalid_memory_is_rejected_before_db_use(subject, description, kind):
    with pytest.raises(ValueError):
        remember(None, subject, description, kind)


def _isolate_installer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = {}
    monkeypatch.setattr(mcp_install, "TOOLS", tools)
    monkeypatch.setattr(brain, "TOOLS", tools)
    monkeypatch.setattr(mcp_install, "DEFAULT_MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(mcp_install, "DEFAULT_MEMORY_DB", tmp_path / "memory/brain.attest")
    monkeypatch.setattr(brain, "DEFAULT_MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(brain, "DEFAULT_MEMORY_DB", tmp_path / "memory/brain.attest")
    monkeypatch.setattr(mcp_install, "_load_cloud_config", lambda: None)


def test_codex_install_is_idempotent_and_preserves_toml(tmp_path, monkeypatch):
    _isolate_installer(tmp_path, monkeypatch)
    path = tmp_path / ".codex/config.toml"
    path.parent.mkdir()
    original = '# My settings\nnotify = ["my-notifier"]\n[mcp_servers.other]\ncommand = "other"\n'
    path.write_text(original)
    monkeypatch.setitem(mcp_install.TOOLS, "codex", {
        "name": "Codex", "detect": lambda: True, "config_paths": {"user": lambda: path},
    })
    brain.install("codex")
    first = path.read_text()
    brain.install("codex")
    assert path.read_text() == first
    data = tomllib.loads(first)
    assert data["notify"] == ["my-notifier"]
    assert data["mcp_servers"]["other"]["command"] == "other"
    assert data["mcp_servers"]["brain"]["args"][-1].endswith("brain.attest")
    assert "# My settings" in first
    assert not (tmp_path / ".mcp.json").exists()
    brain.uninstall("codex")
    remaining = tomllib.loads(path.read_text())
    assert "brain" not in remaining["mcp_servers"]
    assert remaining["notify"] == ["my-notifier"]
    assert remaining["mcp_servers"]["other"]["command"] == "other"


def test_malformed_config_is_not_overwritten(tmp_path, monkeypatch):
    _isolate_installer(tmp_path, monkeypatch)
    path = tmp_path / "client.json"
    path.write_text('{"invalid":')
    monkeypatch.setitem(mcp_install.TOOLS, "claude", {
        "name": "Claude", "detect": lambda: True, "config_paths": {"user": lambda: path},
    })
    with pytest.raises(json.JSONDecodeError):
        brain.install("claude")
    assert path.read_text() == '{"invalid":'


def test_codex_migrates_only_owned_notify_and_keeps_other_client_hooks(tmp_path, monkeypatch):
    _isolate_installer(tmp_path, monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('notify = ["attest-mcp", "hook", "--agent", "codex"]\n')
    hooks = tmp_path / "claude-hooks.json"
    hooks.write_text(json.dumps({"hooks": {"Stop": [{
        "hooks": [{"command": "attest-mcp stop"}]}]}}))
    original = hooks.read_text()
    monkeypatch.setitem(mcp_install.TOOLS, "claude", {
        "hooks_path": {"user": lambda: hooks},
    })
    monkeypatch.setitem(mcp_install.TOOLS, "codex", {
        "name": "Codex", "detect": lambda: True, "config_paths": {"user": lambda: path},
    })
    brain.install("codex")
    assert "notify" not in tomllib.loads(path.read_text())
    brain.uninstall("codex")
    assert hooks.read_text() == original


def test_claude_install_preserves_other_servers_without_hooks(tmp_path, monkeypatch):
    _isolate_installer(tmp_path, monkeypatch)
    path = tmp_path / "claude.json"
    path.write_text(json.dumps({"mcpServers": {"existing": {"command": "keep"}}}))
    monkeypatch.setitem(mcp_install.TOOLS, "claude", {
        "name": "Claude", "detect": lambda: True, "config_paths": {"user": lambda: path},
    })
    brain.install("claude")
    assert set(json.loads(path.read_text())["mcpServers"]) == {"brain", "existing"}
    assert not (tmp_path / ".claude/settings.json").exists()
    brain.uninstall("claude")
    assert json.loads(path.read_text())["mcpServers"] == {"existing": {"command": "keep"}}


LONG_SUMMARY = " ".join(f"step {i}: checked the token refresh path and its retries"
                         for i in range(16))


def test_a_session_with_a_long_summary_is_stored_and_recalled(tmp_path):
    """An entity id is an LMDB key of at most 511 bytes: a summary of about 800 characters was
    rejected by the storage engine, and attest_session_end failed."""
    assert len(LONG_SUMMARY) > 800
    db = AttestDB(str(tmp_path / "long.attest"), embedding_dim=None)
    mcp_server.configure(db)
    try:
        mcp_server.attest_session_end("success", LONG_SUMMARY, "Ship it", ["auth.py"])

        by_summary = mcp_server.attest_get_prior_approaches(LONG_SUMMARY)
        by_file = mcp_server.attest_check_file("auth.py")

        assert [row["subject"] for row in by_summary] == [LONG_SUMMARY]
        assert by_file[0]["description"] == LONG_SUMMARY
    finally:
        db.close()
        mcp_server.configure(None)


def test_a_long_subject_is_recalled_exactly_and_by_topic(tmp_path):
    db = AttestDB(str(tmp_path / "subject.attest"), embedding_dim=None)
    try:
        remember(db, LONG_SUMMARY, "Retries hide the expiry bug", "bug")
        remember(db, LONG_SUMMARY + " again", "A second, distinct subject", "tip")

        exact = recall(db, LONG_SUMMARY, exact=True)

        assert [row["description"] for row in exact] == ["Retries hide the expiry bug"]
        assert recall(db, LONG_SUMMARY)[0]["subject"] == LONG_SUMMARY
    finally:
        db.close()


def test_a_short_subject_keeps_the_entity_id_it_always_had(tmp_path):
    db = AttestDB(str(tmp_path / "short.attest"), embedding_dim=None)
    try:
        remember(db, "auth.py", "Tokens expire after an hour", "warning")

        assert [c.subject.id for c in db.claims_for("auth.py")] == ["auth.py"]
    finally:
        db.close()


@pytest.mark.parametrize("length", [_SUBJECT_BYTES, _SUBJECT_BYTES + 1, 2000])
def test_a_subject_of_any_length_around_the_limit_is_stored(tmp_path, length):
    db = AttestDB(str(tmp_path / f"edge{length}.attest"), embedding_dim=None)
    try:
        subject = "a" * length
        remember(db, subject, "Stored", "tip")

        assert [row["subject"] for row in recall(db, subject, exact=True)] == [subject]
    finally:
        db.close()
