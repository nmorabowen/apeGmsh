"""App wall lint: ADR 0112 D4, the app never imports or spawns apeGmsh.

The app in `apeGmshViewer/` reads the artifacts and the schema documents
only. This lint fails on any import from, or subprocess into, `apeGmsh`.
There are no waivers: D4 is a hard wall.

Rules (scans only `git ls-files apeGmshViewer`, so node_modules and build
output are never read; comment-only lines are skipped):

- W1 py-import:         `.py` files, `import apeGmsh...` / `from apeGmsh... import`.
- W2 module-specifier:  `.ts .tsx .mts .cts .js .mjs .cjs` files, every specifier of
                        `import ... from`, `export ... from`, `import(...)`, `require(...)`
                        that is bare `apeGmsh`/`apegmsh` (any case, any subpath) or
                        relative and resolving outside `apeGmshViewer/`.
- W3 subprocess:        a statement calling spawn/exec/fork/subprocess.*/os.system that
                        also mentions python, `py`, a `.py` path or apeGmsh (not
                        `apeGmshViewer` alone).

    python scripts/check_app_wall.py              # this checkout
    python scripts/check_app_wall.py --root DIR   # another git tree

Output: `path:line: W<n> <rule> -- <text>`, exit 1 on any finding. Stdlib only.
Self-test: `tests/test_check_app_wall.py`; CI runs this in `static-gates`.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = "apeGmshViewer"
JS_EXTS = {".ts", ".tsx", ".mts", ".cts", ".js", ".mjs", ".cjs"}

PY_IMPORT = re.compile(r"^\s*(?:import\s+apeGmsh\b|from\s+apeGmsh\b[\w.]*\s+import\b)")
SPEC_PATTERNS = [
    re.compile(r"""\b(?:import|export)\b[^;'"`]*?\bfrom\s*['"]([^'"]+)['"]"""),
    re.compile(r"""\bimport\s*['"]([^'"]+)['"]"""),
    re.compile(r"""\bimport\s*\(\s*['"]([^'"]+)['"]"""),
    re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]"""),
]
SPAWN = re.compile(
    r"\b(?:utilityProcess\.fork|subprocess\.\w+|os\.system"
    r"|spawnSync|spawn|execSync|execFileSync|execFile|exec|fork)\s*\("
)
MENTIONS = re.compile(r"python|\bpy\b|\.py\b|apegmsh(?!viewer)", re.IGNORECASE)


def _tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", APP],
        cwd=root, capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p]


def _is_comment(line: str, is_py: bool) -> bool:
    s = line.strip()
    if is_py:
        return s.startswith("#")
    return s.startswith(("//", "/*", "*"))


def _statement(lines: list[str], i: int) -> str:
    """The call's statement: from line i until its parentheses balance."""
    depth, parts = 0, []
    for j in range(i, min(i + 12, len(lines))):
        parts.append(lines[j])
        depth += lines[j].count("(") - lines[j].count(")")
        if depth <= 0:
            break
    return "\n".join(parts)


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for rel in _tracked(root):
        ext = Path(rel).suffix.lower()
        is_py = ext == ".py"
        if not is_py and ext not in JS_EXTS:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        blanked = [("" if _is_comment(ln, is_py) else ln) for ln in lines]

        def add(n: int, rule: str, name: str, why: str) -> None:
            findings.append(f"{rel}:{n}: {rule} {name} — {why.strip()[:120]}")

        if is_py:
            for n, ln in enumerate(blanked, 1):
                if PY_IMPORT.match(ln):
                    add(n, "W1", "py-import", ln)
        else:
            body = "\n".join(blanked)
            for pat in SPEC_PATTERNS:
                for m in pat.finditer(body):
                    spec = m.group(1)
                    n = body.count("\n", 0, m.start(1)) + 1
                    low = spec.lower()
                    if low == "apegmsh" or low.startswith("apegmsh/"):
                        add(n, "W2", "module-specifier", f"bare specifier '{spec}'")
                    elif spec.startswith("."):
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
                        if target != APP and not target.startswith(APP + "/"):
                            add(n, "W2", "module-specifier", f"'{spec}' resolves outside {APP}/")
        for i, ln in enumerate(blanked):
            if ln and SPAWN.search(ln) and MENTIONS.search(_statement(blanked, i)):
                add(i + 1, "W3", "subprocess", ln)
    return sorted(set(findings))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=REPO)
    root = ap.parse_args(argv).root.resolve()
    if not _tracked(root):
        print(f"check_app_wall: no tracked files under {APP}/, nothing to check")
        return 0
    findings = scan(root)
    for f in findings:
        print(f)
    if findings:
        print(f"check_app_wall: {len(findings)} finding(s); ADR 0112 D4 has no waivers", file=sys.stderr)
        return 1
    print("check_app_wall: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
