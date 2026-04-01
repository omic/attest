"""Local configuration management for AttestDB CLI.

Stores API keys, trial state, telemetry preference in ``~/.attestdb/config.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".attestdb"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _ensure_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load() -> dict[str, Any]:
    """Load config as a flat dict. TOML-like but stored as JSON for zero deps."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict[str, Any]) -> None:
    _ensure_dir()
    CONFIG_FILE.write_text(json.dumps(data, indent=2) + "\n")


def get(key: str, default: Any = None) -> Any:
    """Get a config value."""
    return _load().get(key, default)


def set_key(key: str, value: Any) -> None:
    """Set a config value."""
    data = _load()
    data[key] = value
    _save(data)


def delete(key: str) -> None:
    """Remove a config value."""
    data = _load()
    data.pop(key, None)
    _save(data)


def get_api_key() -> str | None:
    """Get the stored API key (from trial or login)."""
    return get("api_key")


def get_api_endpoint() -> str:
    """Get the API endpoint."""
    return get("api_endpoint", "https://api.attestdb.com")


def get_trial_info() -> dict[str, Any] | None:
    """Get trial information if active."""
    return get("trial")


def set_trial_info(info: dict[str, Any]) -> None:
    """Store trial information."""
    set_key("trial", info)


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled (default: not set = not asked yet)."""
    return get("telemetry_enabled", False)


def telemetry_asked() -> bool:
    """Check if user has been asked about telemetry."""
    return get("telemetry_asked", False)


def default_db_path() -> str:
    """Default local database path."""
    return str(Path.home() / ".attest" / "memory.attest")
