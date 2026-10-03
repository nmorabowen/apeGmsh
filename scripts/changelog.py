#!/usr/bin/env python
"""Changelog fragments: validate them and fold them into CHANGELOG.md.

Each PR adds one file ``changelog.d/<slug>.md`` holding exactly one
CHANGELOG section (one ``### `` header plus body). Two PRs then never
touch the same lines and never conflict (lessons #1219, and #1267 with
#1279: GitHub ignored the union driver and flagged every pair).

    python scripts/changelog.py --check      # exit 1 on any violation
    python scripts/changelog.py --check --base HEAD^1   # also: PR leaves CHANGELOG.md alone
    python scripts/changelog.py --assemble   # fold fragments in, delete them

``--assemble`` is a release-time / maintainer housekeeping step, never
run in CI. See ``internal_docs/changelog_workflow.md``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_MARK = "FRAGMENTS ARE ASSEMBLED BELOW THIS COMMENT"
SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*\.md$")
NOT_FRAGMENTS = {"README.md"}


def fragment_paths(root: Path) -> list[Path]:
    """Fragment files, sorted by name (the README is not a fragment)."""
    d = root / "changelog.d"
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.iterdir() if p.is_file() and p.name not in NOT_FRAGMENTS
    )


def _unfenced(lines: list[str]):
    """Yield ``(index, line)`` for lines outside fenced code blocks."""
    in_fence = False
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            yield i, ln


def check_fragment(path: Path) -> list[str]:
    """Problems with one fragment file (empty list when it is fine)."""
    name = path.name
    if not SLUG_RE.match(name):
        return [f"changelog.d/{name}: name must be lowercase kebab-case "
                f"ending in .md (e.g. 2026-09-29-my-change.md)"]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if not text.endswith("\n") or text.endswith("\n\n"):
        problems.append(f"changelog.d/{name}: must end with exactly one newline")
    lines = text.splitlines()
    if not lines or not lines[0].startswith("### "):
        problems.append(f"changelog.d/{name}: first line must be a '### ' header")
    if not [ln for ln in lines[1:] if ln.strip()]:
        problems.append(f"changelog.d/{name}: header has no body")
    heads = [i for i, ln in _unfenced(lines) if ln.startswith("### ")]
    if len(heads) != 1:
        problems.append(
            f"changelog.d/{name}: needs exactly one '### ' header, found {len(heads)}")
    if any(ln.startswith(("# ", "## ")) for _, ln in _unfenced(lines)):
        problems.append(f"changelog.d/{name}: must not contain '# ' or '## ' headers")
    if len(lines) > 1 and lines[1].strip():
        problems.append(f"changelog.d/{name}: needs a blank line after the header")
    return problems


def _anchor_index(lines: list[str]) -> list[int]:
    return [i for i, ln in enumerate(lines) if ANCHOR_MARK in ln]


def check_changelog(lines: list[str]) -> list[str]:
    """Structure violations of CHANGELOG.md."""
    problems: list[str] = []
    headers = [i for i, ln in enumerate(lines) if ln.startswith("## Unreleased")]
    if len(headers) != 1:
        problems.append(
            f"CHANGELOG.md: expected exactly 1 '## Unreleased' line, found {len(headers)}")
    anchors = _anchor_index(lines)
    if len(anchors) != 1:
        problems.append(
            f"CHANGELOG.md: expected exactly 1 entry anchor, found {len(anchors)} "
            f"at lines {[i + 1 for i in anchors]}")
    elif len(headers) == 1:
        first = next(
            (i for i in range(headers[0] + 1, len(lines)) if lines[i].strip()),
            None)
        if first != anchors[0]:
            problems.append(
                f"CHANGELOG.md line {(first or 0) + 1} sits between the "
                f"'## Unreleased' header and the anchor (line {anchors[0] + 1})")
    for i, ln in enumerate(lines):
        if ln.startswith(("<<<<<<<", ">>>>>>>")) or ln == "=======":
            problems.append(f"CHANGELOG.md line {i + 1}: merge conflict marker")
    for i, ln in _unfenced(lines):
        if i and ln.startswith("### ") and lines[i - 1].strip():
            problems.append(
                f"CHANGELOG.md line {i + 1}: '### ' header with no blank "
                f"line before it (a union merge dropped the separator)")
    return problems


def check(root: Path = REPO_ROOT) -> list[str]:
    """All problems in ``root``: fragments and CHANGELOG.md."""
    problems: list[str] = []
    for p in fragment_paths(root):
        problems += check_fragment(p)
    cl = root / "CHANGELOG.md"
    if cl.exists():
        problems += check_changelog(cl.read_text(encoding="utf-8").splitlines())
    else:
        problems.append("CHANGELOG.md is missing")
    return problems


def check_pr_diff(base: str, root: Path = REPO_ROOT) -> list[str]:
    """A PR must leave CHANGELOG.md alone (#1267 and #1279, 2026-10-03).

    GitHub ignored the union merge driver when it computed mergeability,
    so a PR that edited CHANGELOG.md went CONFLICTING the moment any other
    CHANGELOG.md edit landed, and concurrent sessions rebased each other in
    turn. Two diffs may touch the file: an assemble (it deletes fragments)
    and a change to this tool (it may restructure the file).
    """
    out = subprocess.run(
        ["git", "diff", "--name-status", base, "HEAD"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    rows = [ln.split("\t") for ln in out.splitlines() if ln.strip()]
    paths = {r[-1] for r in rows}
    if "CHANGELOG.md" not in paths:
        return []
    assembles = any(
        r[0] == "D" and r[-1].startswith("changelog.d/")
        and r[-1] != "changelog.d/README.md" for r in rows)
    if assembles or "scripts/changelog.py" in paths:
        return []
    return ["CHANGELOG.md: a PR must not edit it. Move your section to "
            "changelog.d/<slug>.md and restore CHANGELOG.md from main "
            "(git checkout origin/main -- CHANGELOG.md). Two PRs that both "
            "edit it conflict on GitHub."]


def assemble(root: Path = REPO_ROOT) -> list[str]:
    """Insert every fragment below the anchor comment, then delete them.

    Returns the names folded in; an empty list means nothing to do.
    """
    frags = fragment_paths(root)
    if not frags:
        return []
    problems = check(root)
    if problems:
        raise ValueError("refusing to assemble:\n  " + "\n  ".join(problems))
    cl = root / "CHANGELOG.md"
    lines = cl.read_text(encoding="utf-8").splitlines()
    end = _anchor_index(lines)[0]
    while "-->" not in lines[end]:
        end += 1
    sections: list[str] = []
    for p in frags:
        sections += [""] + p.read_text(encoding="utf-8").rstrip("\n").splitlines()
    lines[end + 1:end + 1] = sections
    cl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in frags:
        p.unlink()
    return [p.name for p in frags]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--assemble", action="store_true")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--base", metavar="REF",
                    help="with --check: also fail if the diff REF..HEAD edits "
                         "CHANGELOG.md (CI passes HEAD^1 of the PR merge ref)")
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.root)
        if args.base:
            problems += check_pr_diff(args.base, args.root)
        for p in problems:
            print(p, file=sys.stderr)
        return 1 if problems else 0
    try:
        done = assemble(args.root)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"assembled {len(done)} fragment(s)" if done else "nothing to assemble")
    return 0


if __name__ == "__main__":
    sys.exit(main())
