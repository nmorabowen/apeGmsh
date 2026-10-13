"""Regenerate the quirk rows of internal_docs/program/gates.md from scripts/check_quirks.py.

    python scripts/gates_registry.py           # rewrite the file
    python scripts/gates_registry.py --check   # exit 1 on drift, touching nothing

A quirk row carries the rule name, its purpose (the opening clause of its RULES text),
the legitimate way to change it, and the waiver count (the `# apegmsh-lint: <rule>-ok`
comments in the scanned tree). Lock, ratchet and label rows are kept as written.
Owner and Review are read from the existing file for a rule that already has a row;
a new row takes the chain tag of the commit that introduced the rule (`git log -S`),
or `?`. `--check` compares every column except Owner, so it needs no git history.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATES = Path("internal_docs/program/gates.md")
PURPOSE_CAP = 150
OWNER = 1


def _load_quirks():
    spec = importlib.util.spec_from_file_location("check_quirks", ROOT / "scripts" / "check_quirks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def purpose(text: str) -> str:
    """The opening clause of a rule's text: up to the first ': ' or '. ', capped."""
    cut = min((i for i in (text.find(": "), text.find(". ")) if i > 0), default=len(text))
    clause = text[:cut].strip()
    if len(clause) > PURPOSE_CAP:
        clause = clause[: PURPOSE_CAP - 1].rstrip() + "…"
    return clause.replace("|", "/")


def waiver_counts(quirks, root: Path) -> dict[str, int]:
    """`# apegmsh-lint: <rule>-ok` comments per rule, over the files check_quirks scans."""
    counts = {rule: 0 for rule in quirks.RULES}
    for path in quirks._python_files(root):
        if path.name in ("check_quirks.py", "test_check_quirks.py"):
            continue  # they quote the waiver syntax in fixtures and docs
        text = quirks._read_source(path)
        if text is None or "apegmsh-lint" not in text.lower():
            continue
        for comment in quirks._comments(text).values():
            match = quirks.WAIVER.search(comment)
            if match and match.group("rule") in counts:
                counts[match.group("rule")] += 1
    return counts


def introduced_by(rule: str, root: Path) -> str:
    """The chain tag in the subject of the commit that added the rule, else '?'."""
    try:
        out = subprocess.run(
            ["git", "log", "--reverse", "--format=%s", "-S", f'"{rule}"', "--", "scripts/check_quirks.py"],
            cwd=root, capture_output=True, text=True, timeout=60, check=False,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return "?"
    match = re.match(r"\[([A-Z]\w*?)(?:-\w+)?\]", out[0]) if out else None
    return match.group(1) if match else "?"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split(" | ")]


def quirk_row(quirks, rule: str, waivers: int, owner: str, review: str) -> list[str]:
    if rule in quirks.NO_WAIVER:
        change = "change the rule in `scripts/check_quirks.py` with its self-test; it has no per-site waiver"
    else:
        change = (
            f"fix the code, or waive one site with `# apegmsh-lint: {rule}-ok <reason>`; "
            "change the rule in `scripts/check_quirks.py` with its self-test"
        )
    return [f"quirk `{rule}`", owner, purpose(quirks.RULES[rule]), change, str(waivers), review]


def build(root: Path, existing: str, use_git: bool) -> str:
    quirks = _load_quirks()
    lines = existing.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.startswith("| Gate "))
    sep = header + 1
    table_end = next((i for i in range(sep + 1, len(lines)) if not lines[i].startswith("|")), len(lines))
    body = lines[sep + 1 : table_end]
    old = {_cells(ln)[0]: _cells(ln) for ln in body}
    nxt = re.search(r"Next review:\s*(\d{4}-\d{2}-\d{2})", existing)
    default_review = nxt.group(1) if nxt else "?"
    counts = waiver_counts(quirks, root)

    rows = []
    for rule in quirks.RULES:
        prior = old.get(f"quirk `{rule}`")
        owner = prior[OWNER] if prior else (introduced_by(rule, root) if use_git else "?")
        review = prior[5] if prior else default_review
        rows.append("| " + " | ".join(quirk_row(quirks, rule, counts[rule], owner, review)) + " |")

    out: list[str] = []
    placed = False
    for ln in body:
        if _cells(ln)[0].startswith("quirk `"):
            if not placed:
                out += rows
                placed = True
        else:
            out.append(ln)
    if not placed:
        out = rows + out
    return "\n".join(lines[: sep + 1] + out + lines[table_end:]) + "\n"


def _compare_form(text: str) -> dict[str, list[str]]:
    rows = {}
    for ln in text.splitlines():
        if ln.startswith("| ") and not ln.startswith("| Gate ") and not ln.startswith("|---"):
            cells = _cells(ln)
            rows[cells[0]] = cells[:OWNER] + cells[OWNER + 1 :]
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if gates.md differs from check_quirks.py")
    args = parser.parse_args(argv)
    path = ROOT / GATES
    existing = path.read_text(encoding="utf-8")
    if not args.check:
        path.write_text(build(ROOT, existing, use_git=True), encoding="utf-8", newline="\n")
        print(f"wrote {GATES}")
        return 0
    have = _compare_form(existing)
    want = _compare_form(build(ROOT, existing, use_git=False))
    if have == want:
        print("gates.md is current")
        return 0
    for gate in want:
        if gate not in have:
            print(f"missing row: {gate}")
        elif have[gate] != want[gate]:
            print(f"changed row: {gate}")
    for gate in have:
        if gate not in want and gate.startswith("quirk `"):
            print(f"stale row: {gate}")
    print("gates.md drifted from check_quirks.py: run `python scripts/gates_registry.py`")
    return 1


if __name__ == "__main__":
    sys.exit(main())
