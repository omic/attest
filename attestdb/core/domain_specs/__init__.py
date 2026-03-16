"""Domain spec registry — list and load bundled domain specifications."""

from __future__ import annotations

from pathlib import Path

_SPECS_DIR = Path(__file__).parent


def list_specs() -> list[str]:
    """List available domain spec names (scans directory for .json files)."""
    return sorted(p.stem for p in _SPECS_DIR.glob("*.json"))


def load_spec(name: str):
    """Load a bundled domain spec by name.

    Returns a DomainSpec instance.

    Raises FileNotFoundError if spec doesn't exist.
    """
    from attestdb.core.domain_spec import DomainSpec

    path = _SPECS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Domain spec '{name}' not found at {path}")
    return DomainSpec.from_file(path)


def spec_path(name: str) -> Path:
    """Return path to a spec file (for writing new specs)."""
    return _SPECS_DIR / f"{name}.json"
