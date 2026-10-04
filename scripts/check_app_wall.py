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
- W4 panel-import:      ADR 0112 D6. A module under `apeGmshViewer/src/panels/<a>` may
                        import only `src/state/store`, `src/state/selectors`,
                        `src/state/types`, `src/ui/*` and its own directory
                        `src/panels/<a>/*` (an allow-list; `import type` counts). Any
                        other specifier, relative or bare, fails: another panel, the
                        viewport, the reader, the BlobStore, effects, the loader, a
                        package. A module under `src/ui/` may import only `src/ui/*`
                        and `src/state/types`, so ui/ cannot re-export what a panel
                        may not reach. Panels dispatch events and read selectors.
                        `npm run lint` (dependency-cruiser) holds the same rules over
                        the same graph.

Specifiers are read from the source with comments stripped by a scanner that
knows strings and block comments (an `import\n * as x from '...'` spanning lines
is one import; an import inside a comment is none).

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

PANELS = f"{APP}/src/panels/"
UI = f"{APP}/src/ui/"
# The resolved import targets a panel may reach (W4): exact modules and prefixes.
PANEL_ALLOW_MODULES = {f"{APP}/src/state/store", f"{APP}/src/state/selectors", f"{APP}/src/state/types"}
PANEL_ALLOW_PREFIXES = (UI,)
# What a ui/ module may reach: ui/ itself and the state types.
UI_ALLOW_MODULES = {f"{APP}/src/state/types"}


def _panel_of(path: str) -> str | None:
    """The panel name `<a>` of a path under src/panels/, else None."""
    if not path.startswith(PANELS):
        return None
    first = path[len(PANELS):].split("/", 1)[0]
    return first.rsplit(".", 1)[0] if "." in first else first


def _stem(target: str) -> str:
    return target.rsplit(".", 1)[0] if Path(target).suffix in JS_EXTS else target


def _panel_violation(rel: str, spec: str, target: str | None) -> str | None:
    """Why a panel or ui module may not import `spec` (resolved to `target` when relative), or None."""
    if rel.startswith(UI):
        if target is None:
            return f"ui module imports the package '{spec}'; ui/ imports ui/ and the state types only"
        if target.startswith(UI) or _stem(target) in UI_ALLOW_MODULES:
            return None
        return f"ui module imports '{target[len(APP) + 1:]}'; ui/ imports ui/ and the state types only"
    me = _panel_of(rel)
    if me is None:
        return None
    if target is None:
        return f"panel '{me}' imports the package '{spec}'; panels import the store, selectors, types and ui/ only"
    if _stem(target) in PANEL_ALLOW_MODULES or target.startswith(PANEL_ALLOW_PREFIXES):
        return None
    if target.startswith(f"{PANELS}{me}/"):
        return None
    other = _panel_of(target)
    what = f"panel '{other}'" if other is not None else f"'{target[len(APP) + 1:]}'"
    return f"panel '{me}' imports {what}; panels import the store, selectors, types and ui/ only"


def _tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", APP],
        cwd=root, capture_output=True, check=True,
    ).stdout.decode("utf-8", "replace")
    return [p for p in out.split("\0") if p]


def _is_comment(line: str) -> bool:
    return line.strip().startswith("#")


_SPECIFIER_CONTEXT = re.compile(r"(?:\bfrom|\bimport|\b(?:require|import)\s*\()\s*$")


def _strip_js_comments(text: str, blank_strings: bool = False) -> str:
    """Blank `//` and `/* */` comments (newlines kept, so offsets map to lines).

    With `blank_strings`, every string that is not a module specifier is
    blanked too (for the specifier rules); without it, strings stay whole (the
    subprocess rule reads the command names in them).
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "'\"`":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            # A specifier keeps its text (`from '...'`, `import '...'`,
            # `require('...')`, `import('...')`); any other string is blanked,
            # so an import written inside a string is not an import.
            if not blank_strings or _SPECIFIER_CONTEXT.search("".join(out)[-40:]):
                out.append(text[i:j])
            else:
                out.append(c + "".join("\n" if ch == "\n" else " " for ch in text[i + 1:j - 1]) + (text[j - 1] if j - 1 > i else ""))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


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
        if is_py:
            blanked = [("" if _is_comment(ln) else ln) for ln in text.splitlines()]
            specs = blanked
        else:
            blanked = _strip_js_comments(text).splitlines()
            specs = _strip_js_comments(text, blank_strings=True).splitlines()

        def add(n: int, rule: str, name: str, why: str) -> None:
            findings.append(f"{rel}:{n}: {rule} {name} — {why.strip()[:120]}")

        if is_py:
            for n, ln in enumerate(blanked, 1):
                if PY_IMPORT.match(ln):
                    add(n, "W1", "py-import", ln)
        else:
            body = "\n".join(specs)
            for pat in SPEC_PATTERNS:
                for m in pat.finditer(body):
                    spec = m.group(1)
                    n = body.count("\n", 0, m.start(1)) + 1
                    low = spec.lower()
                    target = None
                    if low == "apegmsh" or low.startswith("apegmsh/"):
                        add(n, "W2", "module-specifier", f"bare specifier '{spec}'")
                    elif spec.startswith("."):
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(rel), spec))
                        if target != APP and not target.startswith(APP + "/"):
                            add(n, "W2", "module-specifier", f"'{spec}' resolves outside {APP}/")
                    why = _panel_violation(rel, spec, target)
                    if why:
                        add(n, "W4", "panel-import", f"{why}: '{spec}'")
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
