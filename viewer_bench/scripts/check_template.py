#!/usr/bin/env python3
"""Validate APE Studio habitat folder template (soft contract)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _habitat import HABITAT, REQUIRED_DIRS, REQUIRED_FILES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also require ape.project.yaml key paths to exist",
    )
    args = parser.parse_args()
    missing: list[str] = []
    for rel in REQUIRED_DIRS:
        if not (HABITAT / rel).is_dir():
            missing.append(f"DIR  {rel}")
    for rel in REQUIRED_FILES:
        if not (HABITAT / rel).is_file():
            missing.append(f"FILE {rel}")

    # models/* with src/
    models = HABITAT / "models"
    model_ids = [
        p.name
        for p in models.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ] if models.is_dir() else []
    if not model_ids:
        missing.append("DIR  models/<id>/ (at least one model lineage)")
    for mid in model_ids:
        if not (models / mid / "src").is_dir():
            missing.append(f"DIR  models/{mid}/src")

    if args.strict:
        yaml_path = HABITAT / "ape.project.yaml"
        text = yaml_path.read_text(encoding="utf-8") if yaml_path.is_file() else ""
        for key, rel in [
            ("skills_door", "APE/skills/catalog.md"),
            ("libraries_door", "APE/libraries/catalog.md"),
            ("reporting_playbook", "APE/instructions/reporting.md"),
            ("session_postmortem", "APE/instructions/session-postmortem.md"),
            ("session_start", "scripts/start.ps1"),
            ("session_finish", "scripts/finish.ps1"),
        ]:
            if key in text and not (HABITAT / rel).is_file():
                missing.append(f"MANIFEST {rel} (from {key})")

    print(f"habitat: {HABITAT}")
    print(f"models:  {', '.join(model_ids) or '(none)'}")
    if missing:
        print("TEMPLATE FAIL")
        for m in missing:
            print(f"  - {m}")
        return 1
    print("TEMPLATE OK")
    return 0


if __name__ == "__main__":
    # allow `python scripts/check_template.py` 
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
