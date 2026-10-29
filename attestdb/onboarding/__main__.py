"""Export the installed shared onboarding kit for the user's existing client."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import export_skill, kit_version, skill_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--directory", type=Path, help="Create a new portable skill folder")
    target.add_argument("--zip", type=Path, help="Create a new portable skill ZIP")
    args = parser.parse_args()
    try:
        if args.directory is not None:
            export_skill(args.directory)
        else:
            content = skill_archive()
            with args.zip.open("xb") as stream:
                stream.write(content)
        print(f"AutoCompany onboarding kit {kit_version()}: {args.directory or args.zip}")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Onboarding export failed: {exc}\n")


if __name__ == "__main__":
    main()
