"""``python -m apeGmsh.studio.profile`` — ADR 0096 S4 / S5.

Sidecar profiler. Not an MCP tool. Reads ``.apegmsh/mcp_calls.jsonl``
and/or a transcript JSONL. Prints counts, heuristic tokens (chars/4),
``src_search_rate``, and an ``other_gloss`` breakdown (schema 2) when
``other`` is non-zero. ``--skill-budget`` gates skill file size.
``--promote`` prints S5 eligibility and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeGmsh.studio.profile",
        description=(
            "Token-budget profiler (ADR 0096 S4). Not an MCP tool. "
            "src_search_rate is the defect metric."
        ),
    )
    parser.add_argument(
        "--mcp",
        type=Path,
        default=None,
        help="MCP ledger JSONL (default: .apegmsh/mcp_calls.jsonl if present).",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        help="Cursor / agent JSONL transcript to classify.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )
    parser.add_argument(
        "--skill-budget",
        action="store_true",
        help="Fail if skills/apegmsh/ exceeds character caps (CI hygiene).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Print S5 MCP-lookup-miss eligibility (kind=miss, bar=3). "
            "Writes nothing (no skill/MCP mutation)."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repo / model root (default: cwd).",
    )
    args = parser.parse_args(argv)

    root = args.root if args.root is not None else Path.cwd()
    from apeGmsh.studio._paths import mcp_calls_path
    from apeGmsh.studio._profile import (
        build_report,
        format_promote_report,
        format_report,
        promote_report,
        skill_budget_violations,
    )

    mcp_path = args.mcp
    auto_mcp = mcp_calls_path(root)
    if mcp_path is None and not args.skill_budget and not args.promote:
        if args.transcript is None and auto_mcp.is_file():
            mcp_path = auto_mcp

    if args.skill_budget:
        violations = skill_budget_violations(root)
        if violations:
            print("SKILL BUDGET:", file=sys.stderr)
            for line in violations:
                print(f"  {line}", file=sys.stderr)
            return 1
        if args.mcp is None and args.transcript is None and not args.promote:
            print("skill-budget: ok")
            return 0

    if args.promote:
        ledger = mcp_path if mcp_path is not None else auto_mcp
        promo = promote_report(ledger if ledger.is_file() else None)
        if args.json:
            print(json.dumps(promo, indent=2))
        else:
            sys.stdout.write(format_promote_report(promo))
        return 0

    if mcp_path is None and args.transcript is None:
        print(
            "error: no .apegmsh/mcp_calls.jsonl and no --transcript",
            file=sys.stderr,
        )
        return 2

    report = build_report(mcp_path=mcp_path, transcript_path=args.transcript)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        sys.stdout.write(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
