"""Prototype drift check: do the code paths/symbols maintainer docs cite resolve?

Scans AGENTS.md, the maintainer task guides (.claude/skills/apegmsh-*/SKILL.md
except the derived user skill) and the non-ADR architecture docs. For every
backticked token that looks like a repo path (``x/y.py``, ``x/y.md``), with an
optional ``:line`` or ``::symbol`` suffix, check that the file exists (resolved
against a few roots) and, for ``::symbol``/``.symbol`` forms, that the symbol is
defined in that file (AST). Prints a summary per doc.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
DOCS = [ROOT / "AGENTS.md"]
DOCS += [p for p in (ROOT / ".claude" / "skills").glob("apegmsh-*/SKILL.md")
         if "helper" not in str(p)]
ARCH = ROOT / "src/apeGmsh/opensees/architecture"
DOCS += sorted(ARCH.glob("*.md"))
BASES = [ROOT, ROOT / "src/apeGmsh", ROOT / "src/apeGmsh/opensees", ARCH, ROOT / "src"]
ALIASES = {"arch/": "src/apeGmsh/opensees/architecture/"}
TOK = re.compile(r"`([^`\s]+?\.(?:py|md|json|toml|yml))(?:(?::|::)([\w.]+))?`")
LINK = re.compile(r"\]\(([^)#\s]+?\.(?:py|md))(?:#[^)]*)?\)")

_defs_cache = {}


def defs(path):
    if path not in _defs_cache:
        names = set()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            names.add(t.id)
        except Exception:
            pass
        _defs_cache[path] = names
    return _defs_cache[path]


def resolve(tok, doc):
    for a, b in ALIASES.items():
        if tok.startswith(a):
            tok = b + tok[len(a):]
    cands = [doc.parent / tok] + [b / tok for b in BASES]
    for c in cands:
        try:
            if c.is_file():
                return c
        except OSError:
            pass
    # suffix match anywhere in the tracked tree (e.g. `build.py`, `session/_host.py`)
    tail = "/" + tok.lstrip("./")
    for f in ALLFILES:
        if f.endswith(tail):
            return ROOT / f
    return None


import subprocess
ALLFILES = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                          text=True).stdout.splitlines()
tot_ok = tot_bad = 0
rows = []
for doc in DOCS:
    text = doc.read_text(encoding="utf-8", errors="replace")
    toks = [(m.group(1), m.group(2)) for m in TOK.finditer(text)]
    toks += [(m.group(1), None) for m in LINK.finditer(text)]
    ok, bad = 0, []
    for tok, sym in toks:
        if tok.startswith(("http", "~", "C:", "<")) or any(c in tok for c in "*{<[$"):
            continue
        p = resolve(tok, doc)
        if p is None:
            bad.append(tok)
            continue
        if sym and not sym.isdigit() and p.suffix == ".py":
            if sym.split(".")[-1] not in defs(p):
                bad.append(f"{tok}::{sym}")
                continue
        ok += 1
    tot_ok += ok
    tot_bad += len(bad)
    rows.append((doc.relative_to(ROOT).as_posix(), ok, bad))
for rel, ok, bad in rows:
    if bad or ok:
        uniq = sorted(set(bad))
        print(f"{rel}: {ok} ok, {len(bad)} unresolved"
              + (f"  e.g. {', '.join(uniq[:5])}" if uniq else ""))
print(f"TOTAL: {tot_ok} resolve, {tot_bad} do not ({tot_bad / max(1, tot_ok + tot_bad):.0%})")
