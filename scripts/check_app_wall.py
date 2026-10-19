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
- W5 h5-open:           lesson #2260. Under `apeGmshViewer/src/` and
                        `apeGmshViewer/scripts/`, an h5wasm `File` is reached only in
                        `src/reader/h5open.ts`. h5wasm hands HDF5 each file's id
                        rounded to a double, and HDF5 dedupes open files by that
                        id: two NTFS files whose ids differ below 2^53's resolution
                        read as one, silently. `openH5` makes HDF5 compare exact
                        ids. Outside the opener the rule flags: a `.File` member
                        read in a value position (`new h5.File(`, `new (h5 as
                        unknown as T).File(`, `new h5!.File(`, `const C = h5.File`,
                        `(await import(...)).File`; a type position such as
                        `: h5.File`, `<h5.File>` or `as h5.File` is allowed); `File`
                        imported (not `import type`) or re-exported from an
                        `h5wasm` specifier; `File` destructured in a
                        `const/let/var {...} =`; the string literal `"File"`
                        (`h5["File"]`, `Reflect.get(h5, "File")`); and
                        `Reflect.construct`. Tests are not scanned (they write
                        fixtures with h5wasm directly).

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
# W5: the one file that may reach an h5wasm `File`, and where the rule looks.
H5_OPENER = f"{APP}/src/reader/h5open.ts"
H5_SCOPE = (f"{APP}/src/", f"{APP}/scripts/")
# A `.File` / `?.File` member access (not `...File`, not `.Files`).
H5_MEMBER = re.compile(r"(?<!\.)\??\.(?!\.)\s*File(?![\w$])")
# `import { File } from 'h5wasm...'` / `export { File } from 'h5wasm...'` (value imports only).
H5_IMPORT = re.compile(r"""\b(import|export)\s+(type\s+)?(?:[\w$]+\s*,\s*)?\{([^}]*)\}\s*from\s*['"]h5wasm(?:/[^'"]*)?['"]""")
H5_DESTRUCTURE = re.compile(r"\b(?:const|let|var)\s*\{[^{}]*?(?<![\w$])File(?![\w$])[^{}]*\}\s*=")
H5_FILE_STRING = re.compile(r"""(['"`])File\1""")
H5_REFLECT = re.compile(r"\bReflect\s*\.\s*construct\b")
# Words after which `X.File` is a type, not a value read.
TYPE_WORDS = {"typeof", "keyof", "as", "satisfies", "implements", "is"}
IDENT = re.compile(r"[\w$]")


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


def _skip_ws_back(text: str, j: int) -> int:
    while j >= 0 and text[j].isspace():
        j -= 1
    return j


def _match_back(text: str, j: int) -> int:
    """Index of the `(`/`[` that opens the closer at text[j], or -1."""
    pairs = {")": "(", "]": "["}
    stack = [text[j]]
    for k in range(j - 1, -1, -1):
        c = text[k]
        if c in ")]":
            stack.append(c)
        elif c in "([":
            if pairs[stack[-1]] != c:
                return -1
            stack.pop()
            if not stack:
                return k
    return -1


def _chain_start(text: str, i: int) -> int:
    """Start of the member-expression chain whose last access (`.` or `?.`) begins at i.

    Walks back over identifiers, `!`, balanced `(...)`/`[...]` (calls, casts,
    `(await import(...))`) and earlier `.`/`?.` links.
    """
    while True:
        j = _skip_ws_back(text, i - 1)
        while j >= 0 and text[j] == "!":
            j -= 1
        start = j + 1
        while j >= 0 and text[j] in ")]":
            k = _match_back(text, j)
            if k < 0:
                return start
            start, j = k, k - 1
        while j >= 0 and IDENT.match(text[j]):
            start, j = j, j - 1
        k = _skip_ws_back(text, start - 1)
        if k >= 1 and text[k] == "." and text[k - 1] != ".":
            i = k - 1 if text[k - 1] == "?" else k
            continue
        return start


def _innermost_opener(text: str, j: int) -> int:
    """Index of the innermost unclosed `(`, `[` or `{` before j, or -1."""
    depth = 0
    for k in range(j - 1, -1, -1):
        c = text[k]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            if depth == 0:
                return k
            depth -= 1
    return -1


def _open_angles(seg: str) -> int:
    seg = seg.replace("=>", "  ").replace(">=", "  ").replace("<=", "  ")
    return seg.count("<") - seg.count(">")


def _has_ternary(seg: str) -> bool:
    """A conditional-operator `?` in seg (not `?.`, `??`, or an optional `x?:`/`x?)`/`x?,`)."""
    for k, c in enumerate(seg):
        if c != "?" or (k > 0 and seg[k - 1] == "?"):
            continue
        rest = seg[k + 1:].lstrip()
        if rest and rest[0] not in ".?:),=":
            return True
    return False


def _is_object_literal(text: str, brace: int) -> bool:
    j = _skip_ws_back(text, brace - 1)
    if j < 0:
        return False
    if text[j] in "=(,[:?" and not text[j - 1: j + 1] == "=>":
        return True
    return text[max(0, j - 5): j + 1] == "return"


def _is_type_position(text: str, start: int) -> bool:
    """Whether the expression starting at `start` sits where TypeScript reads a type."""
    j = _skip_ws_back(text, start - 1)
    if j < 0:
        return False
    if IDENT.match(text[j]):
        k = j
        while k >= 0 and IDENT.match(text[k]):
            k -= 1
        return text[k + 1:j + 1] in TYPE_WORDS
    c = text[j]
    if c in "|&":
        return j == 0 or text[j - 1] != c  # `a | T` is a union; `a || b` is a value
    if c == "<":
        return True
    if c in ",:":
        opener = _innermost_opener(text, j)
        seg_start = max(opener, text.rfind(";", 0, j)) + 1
        seg = text[seg_start:j]
        if _open_angles(seg) > 0:
            return True  # inside `<...>`: a type argument
        if c == ",":
            return False
        if opener >= 0 and text[opener] == "{" and _is_object_literal(text, opener):
            return False  # `{ k: h5.File }`: an object-literal value
        return not _has_ternary(seg)  # `c ? a : h5.File` is a value; `x: h5.File` an annotation
    return False


def _h5_findings(body: str, kept: str) -> list[tuple[int, str]]:
    """W5 sites in one file: (line, why). `body` has strings blanked, `kept` keeps them."""
    line = lambda text, pos: text.count("\n", 0, pos) + 1  # noqa: E731
    out: list[tuple[int, str]] = []
    for m in H5_MEMBER.finditer(body):
        start = _chain_start(body, m.start())
        if not _is_type_position(body, start):
            out.append((line(body, start), f"a value read of `{' '.join(body[start:m.end()].split())}`"))
    for m in H5_IMPORT.finditer(body):
        if m.group(2):
            continue
        for item in m.group(3).split(","):
            item = item.strip()
            if not item.startswith("type ") and item.split(" as ")[0].strip() == "File":
                out.append((line(body, m.start()), f"`File` {m.group(1)}ed from h5wasm"))
    for m in H5_DESTRUCTURE.finditer(body):
        out.append((line(body, m.start()), "`File` destructured"))
    for m in H5_FILE_STRING.finditer(kept):
        out.append((line(kept, m.start()), f"the string {m.group(0)} (a computed `File` access)"))
    for m in H5_REFLECT.finditer(body):
        out.append((line(body, m.start()), "Reflect.construct"))
    return out


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
            if rel.startswith(H5_SCOPE) and rel != H5_OPENER:
                for n, why in _h5_findings(body, "\n".join(blanked)):
                    add(n, "W5", "h5-open", f"open HDF5 through reader/h5open.ts openH5 (lesson #2260): {why}")
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
