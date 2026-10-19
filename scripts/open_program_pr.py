"""Open a program PR: the documented way (lesson #2200, slice #2211).

    python scripts/open_program_pr.py --title T --body-file F [--label X ...]
    python scripts/open_program_pr.py --count     # print the WIP count, exit 0

Refuses when the body has no ``Slice: #<n>`` line, when a ``gh`` read fails, or
when the program WIP count is 8 or more. The count is
``program_probe.wip_count``: open ``program`` PRs, except those labelled
``human-gate`` that are green and approved (maintainer ruling on #2200).
Always adds the ``program`` label, and always uses ``--base main``.
``--count`` lists each counted and excluded PR with its reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_probe = _load("program_probe")
# Same regex as land_pr check 9; one copy (land_pr.py::SLICE_REF).
SLICE_REF = _probe._land_pr.SLICE_REF
DEFAULT_REPO = _probe.DEFAULT_REPO
WIP_CAP = _probe.WIP_CAP
Runner = Callable[[list[str]], str]


def read_count(gh: Runner, repo: str) -> tuple[list, list]:
    """Counted and excluded open program PRs; any ``gh`` failure propagates."""
    prs = json.loads(gh(["pr", "list", "--repo", repo, "--state", "open",
                         "--label", "program", "--limit", _probe.LIMIT,
                         "--json", _probe.PR_FIELDS]))
    return _probe.wip_count(prs)


def describe(counted: list, excluded: list) -> str:
    lines = [f"WIP count: {len(counted)} of {WIP_CAP} (cap)"]
    lines += [f"  counted  #{p['number']} {p['title']}: {why}" for p, why in counted]
    lines += [f"  excluded #{p['number']} {p['title']}: {why}" for p, why in excluded]
    return "\n".join(lines)


def main(argv: list[str] | None = None, gh: Runner = _probe.run_gh) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPO))
    ap.add_argument("--title")
    ap.add_argument("--body-file", type=Path)
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--count", action="store_true",
                    help="print the WIP count and exit 0 (opens nothing)")
    args = ap.parse_args(argv)
    try:
        counted, excluded = read_count(gh, args.repo)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"refused: could not read the open program PRs ({exc})", file=sys.stderr)
        return 1
    report = describe(counted, excluded)
    if args.count:
        print(report)
        return 0
    if not args.title or args.body_file is None:
        ap.error("--title and --body-file are required unless --count")
    body = args.body_file.read_text(encoding="utf-8")
    if not SLICE_REF.search(body):
        print("refused: the body has no `Slice: #<n>` line", file=sys.stderr)
        return 1
    if len(counted) >= WIP_CAP:
        print(f"refused: WIP is full\n{report}", file=sys.stderr)
        return 1
    cmd = ["pr", "create", "--repo", args.repo, "--base", "main",
           "--body-file", str(args.body_file), "--title", args.title,
           "--label", "program"]
    for label in args.label:
        if label != "program":
            cmd += ["--label", label]
    if args.draft:
        cmd.append("--draft")
    try:
        print(gh(cmd).strip())
    except RuntimeError as exc:
        print(f"gh pr create failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
