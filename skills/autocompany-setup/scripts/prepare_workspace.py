"""Create a private, secret-free AutoCompany workspace; never claim connectivity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FABRICS = {
    "memory",
    "engineering",
    "security-compliance",
    "sales-marketing",
    "research",
    "support-incidents",
    "supplier-assurance",
}

KIT_VERSION = json.loads((Path(__file__).resolve().parents[1] / "kit-manifest.json").read_text())[
    "version"
]


def prepare(root: Path, company: str, fabrics: list[str], goal: str = "") -> dict:
    if not company.strip() or not fabrics or set(fabrics) - FABRICS:
        raise ValueError("Company name and known Fabric selections are required")
    manifest = {
        "schema_version": 1,
        "onboarding_kit_version": KIT_VERSION,
        "company": company.strip(),
        "goal": goal,
        "fabrics": {
            name: {
                "connection": "not_checked",
                "owner": None,
                "service_url": None,
                "credential_reference": None,
            }
            for name in sorted(set(fabrics))
        },
        "automation": {
            "state": "paused",
            "daily_invocation_limit": 2,
            "human_review_required": True,
        },
        "setup_state": "workspace_created",
    }
    root.mkdir(parents=True, exist_ok=True)
    target = root / "company.json"
    if target.is_symlink():
        raise ValueError("Company manifest must not be a symbolic link")
    if target.exists():
        existing = json.loads(target.read_text())
        if existing != manifest:
            raise ValueError("Existing workspace differs; review it before changing configuration")
        return existing
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--fabrics", nargs="+", choices=sorted(FABRICS), required=True)
    parser.add_argument("--goal", default="")
    arguments = parser.parse_args()
    print(
        json.dumps(
            prepare(arguments.root, arguments.company, arguments.fabrics, arguments.goal), indent=2
        )
    )


if __name__ == "__main__":
    main()
