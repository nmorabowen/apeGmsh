"""verify_move: prove a code move changed no def or class body.

For every module-level ``def`` / ``async def`` / ``class`` (and every def or
class nested in a class body) under the given paths, take the normalized
``ast.dump`` (no line or column numbers; docstrings kept) keyed by the
qualified name *within its module* (``Cls.method``, never the module path).
The multiset of ``(qualname, dump)`` must be identical between the base and
the head, so a def that moved between files still matches. A class entry
covers its header (decorators, bases, keywords) and its non-def statements
(docstring, class attributes); its members are separate entries.

    python scripts/verify_move.py --base origin/main --head WORKTREE \\
        src/apeGmsh/opensees/apesees.py src/apeGmsh/opensees/build/

Paths are files, directories (recursed for ``*.py``) or globs, and must cover
both the source and the destination of the move. ``--head`` is a git revision
or ``WORKTREE`` (the files on disk). Revisions are read with ``git show``,
never checked out. Exit 0 when identical, 1 when defs were added, removed or
changed (2: see ``--class-map``). ``--json`` prints machine-readable output.

What it cannot see (so a green run is necessary, not sufficient):
  * import-time registration order (decorators run in a new order, a
    registry fills in a different order);
  * relative imports are compared by resolved absolute module, taken from the
    file's path under ``src/`` (an ``__init__.py`` is its own package). A file
    outside ``src/``, or a ``..`` that climbs past the top, keeps the raw form.
    Outside ``src/`` a def moved to a different package depth with the same
    dots can therefore pass although it now resolves to a different module;
  * dynamic module-relative code is not resolved:
    ``importlib.import_module('.x', __package__)``, ``__name__``,
    ``__module__`` and ``__file__``. A cross-package move can change what
    these evaluate to;
  * free-name resolution: a moved body whose globals or imports no longer
    resolve. That is ruff F821's job;
  * module-level statements. Imports and assignments are reported as an
    informational delta and never fail the run;
  * defs nested in functions (they are part of the enclosing body, so they
    are compared), and defs under module-level ``if`` / ``try`` (treated as
    module-level statements, not compared as defs).

``--class-map OLD=NEW[,NEW...]`` (repeatable, opt-in) is for methods moved out
of a class into mixins, where the qualname changes (``apeSees.eigen`` ->
``_ModalMixin.eigen``):
  * a removed ``OLD.m`` and an added ``NEW.m`` with an identical dump count as
    a match (``moved: OLD.m -> NEW.m``) only when NEW is one of OLD's direct
    bases in the head (by simple name), so attribute lookup still reaches it.
    Otherwise, or on a changed body, or when ``m`` is added under two linked
    NEW classes, it stays removed/added and fails. A class cannot be mapped to
    itself (``A=A`` is a usage error);
  * a moved body that is class-sensitive (a private name-mangled identifier
    ``__x``, a zero-argument ``super()``, or ``__class__``) is refused and
    reported as ``class-sensitive: OLD.m (reason)``;
  * class-header entries (an added NEW class, a changed OLD class: bases,
    decorators, class-body statements) do not fail, but only in the module
    files that took part in a moved method. They print under
    ``class headers (review by hand):`` and the run exits **2**
    (``--json``: ``"ok": false, "needs_review": true``), never 0, so a header
    change always forces a human pass. A header of an unmapped class fails.
    Exit 0 in map mode means moves only, zero exempted headers. The OK line
    reads ``OK -- N defs, M moved``, not "identical multiset".
Without ``--class-map`` the behaviour and output are unchanged.

Same qualified name in two modules is legal (a multiset counts both); a
qualname whose dump changed is reported as "changed", never matched by a
rename heuristic.

Stdlib only.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict

DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _git(repo: str, *args: str) -> str:
    res = subprocess.run(["git", "-C", repo, *args], capture_output=True,
                         text=True, encoding="utf-8")
    if res.returncode != 0:
        raise SystemExit(f"verify_move: git {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout


def _matches(path: str, spec: str) -> bool:
    spec = spec.replace("\\", "/").rstrip("/")
    if any(c in spec for c in "*?["):
        return fnmatch.fnmatch(path, spec) or fnmatch.fnmatch(path, spec + "/*")
    return path == spec or path.startswith(spec + "/")


def list_files(repo: str, rev: str, specs: list[str]) -> list[str]:
    if rev == "WORKTREE":
        out: set[str] = set()
        for spec in specs:
            for hit in glob.glob(os.path.join(repo, spec), recursive=True):
                if os.path.isdir(hit):
                    for root, _dirs, names in os.walk(hit):
                        out.update(os.path.relpath(os.path.join(root, n), repo)
                                   for n in names)
                else:
                    out.add(os.path.relpath(hit, repo))
        return sorted(p.replace("\\", "/") for p in out if p.endswith(".py"))
    names = _git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
    return sorted(n for n in names
                  if n.endswith(".py") and any(_matches(n, s) for s in specs))


def read_file(repo: str, rev: str, path: str) -> str:
    if rev == "WORKTREE":
        with open(os.path.join(repo, path), encoding="utf-8") as fh:
            return fh.read()
    return _git(repo, "show", f"{rev}:{path}")


def _header_dump(node: ast.ClassDef) -> str:
    """Class header plus non-def body statements (member defs excluded)."""
    clone = ast.ClassDef(
        name=node.name, bases=node.bases, keywords=node.keywords,
        body=[s for s in node.body if not isinstance(s, DEFS)] or [ast.Pass()],
        decorator_list=node.decorator_list,
        **({"type_params": node.type_params} if hasattr(node, "type_params") else {}),
    )
    return ast.dump(clone, include_attributes=False)


def module_package(path: str) -> list[str] | None:
    """Dotted *package* parts of a repo path under ``src/``, else None.

    ``src/a/b/c.py`` is module ``a.b.c`` in package ``a.b``; ``src/a/b/__init__.py``
    is the package ``a.b`` itself. A path outside ``src/`` cannot be resolved.
    """
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2 or parts[0] != "src" or not parts[-1].endswith(".py"):
        return None
    # Both ``c.py`` and ``__init__.py`` live in the package named by the
    # directory, so the package is the directory parts either way.
    return parts[1:-1] or None


class _AbsoluteImports(ast.NodeTransformer):
    """Rewrite ``from .x import y`` to ``level=0`` with the resolved module."""

    def __init__(self, package: list[str]):
        self.package = package

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level > 0:
            up = node.level - 1
            if up < len(self.package):
                base = self.package[: len(self.package) - up]
                tail = node.module.split(".") if node.module else []
                node.module = ".".join(base + tail)
                node.level = 0
        return node


def collect(source: str, path: str):
    """Return ([(qualname, dump, path)], [module-level statement dumps])."""
    tree = ast.parse(source, filename=path)
    package = module_package(path)
    if package is not None:
        tree = _AbsoluteImports(package).visit(tree)
    defs: list[tuple[str, str, str]] = []
    module_level: list[str] = []

    def walk(body, prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                q = prefix + node.name
                defs.append((q, _header_dump(node), path))
                walk(node.body, q + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append((prefix + node.name,
                             ast.dump(node, include_attributes=False), path))
            elif not prefix:
                module_level.append(ast.dump(node, include_attributes=False))

    walk(tree.body, "")
    return defs, module_level


def gather(repo: str, rev: str, specs: list[str]):
    defs: list[tuple[str, str, str]] = []
    module_level: Counter = Counter()
    for f in list_files(repo, rev, specs):
        d, m = collect(read_file(repo, rev, f), f)
        defs.extend(d)
        module_level.update(m)
    return defs, module_level


def head_bases(repo: str, rev: str, specs: list[str]) -> dict[str, set[str]]:
    """qualname -> simple names of its direct bases, over the head files."""
    out: dict[str, set[str]] = defaultdict(set)

    def walk(body, prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                q = prefix + node.name
                out[q].update(b.id if isinstance(b, ast.Name) else b.attr
                              for b in node.bases
                              if isinstance(b, (ast.Name, ast.Attribute)))
                walk(node.body, q + ".")

    for f in list_files(repo, rev, specs):
        walk(ast.parse(read_file(repo, rev, f), filename=f).body, "")
    return out


_SUPER0 = re.compile(r"Call\(func=Name\(id='super', ctx=Load\(\)\)"
                     r"(?:, args=\[\])?(?:, keywords=\[\])?\)")


def class_sensitive(dump: str) -> str | None:
    """Why a body depends on the class it sits in, or None."""
    if _SUPER0.search(dump):
        return "zero-argument super()"
    names = set(re.findall(r"(?:id|attr|arg|name)='([^']*)'", dump))
    if "__class__" in names:
        return "__class__"
    for n in sorted(names):
        if n.startswith("__") and not n.endswith("__"):
            return f"private name-mangled {n}"
    return None


def _diff(rd: str, ad: str) -> str:
    return "\n".join(difflib.unified_diff(
        rd.replace(", ", ",\n").splitlines(),
        ad.replace(", ", ",\n").splitlines(),
        "base", "head", lineterm="", n=1))


def _is_header(dump: str) -> bool:
    return dump.startswith("ClassDef(")


def _match_moves(removed_by_q, added_by_q, class_map, bases, sensitive):
    """Pair ``OLD.rest`` removals with ``NEW.rest`` additions of equal dump.

    Consumes the matched entries from both dicts and returns the moves.
    """
    moved = []
    for old in sorted(class_map):
        news = [n for n in class_map[old] if n in bases.get(old, ())]
        for q in sorted(removed_by_q):
            if not q.startswith(old + "."):
                continue
            rest = q[len(old) + 1:]
            for entry in list(removed_by_q[q]):
                d, path = entry
                hits = [n for n in news
                        if any(ad == d for ad, _ in added_by_q.get(f"{n}.{rest}", []))]
                if len(hits) != 1:
                    continue  # no match, or a name collision: stays a FAIL
                why = class_sensitive(d)
                if why:
                    sensitive.append({"name": q, "reason": why})
                    continue
                new_q = f"{hits[0]}.{rest}"
                for a_entry in added_by_q[new_q]:
                    if a_entry[0] == d:
                        added_by_q[new_q].remove(a_entry)
                        break
                removed_by_q[q].remove(entry)
                moved.append({"from": q, "to": new_q, "base_file": path,
                              "head_file": a_entry[1]})
    return moved


def compare(base_defs, head_defs, class_map=None, bases=None):
    b = Counter((q, d) for q, d, _ in base_defs)
    h = Counter((q, d) for q, d, _ in head_defs)
    where_b: dict = {}
    where_h: dict = {}
    for q, d, p in base_defs:
        where_b.setdefault((q, d), p)
    for q, d, p in head_defs:
        where_h.setdefault((q, d), p)
    removed_by_q: dict = defaultdict(list)
    added_by_q: dict = defaultdict(list)
    for (q, d), n in (b - h).items():
        removed_by_q[q].extend([(d, where_b[(q, d)])] * n)
    for (q, d), n in (h - b).items():
        added_by_q[q].extend([(d, where_h[(q, d)])] * n)
    sensitive: list = []
    moved = (_match_moves(removed_by_q, added_by_q, class_map, bases or {},
                          sensitive) if class_map else [])
    mapped = set(class_map or ()) | {n for v in (class_map or {}).values() for n in v}
    # exemption is keyed by (module path, qualname): only the files that took
    # part in a move may carry an exempt header.
    ok_paths: dict[str, set[str]] = defaultdict(set)
    for m in moved:
        ok_paths[m["from"].split(".")[0]].add(m["base_file"])
        ok_paths[m["to"].split(".")[0]].add(m["head_file"])

    def exempt(q, path):
        return q in mapped and path in ok_paths.get(q, ())
    headers = []
    changed, removed, added = [], [], []
    for q in sorted(set(removed_by_q) | set(added_by_q)):
        rs, as_ = removed_by_q.get(q, []), added_by_q.get(q, [])
        pairs = min(len(rs), len(as_))
        for (rd, rp), (ad, ap) in zip(rs[:pairs], as_[:pairs]):
            if exempt(q, ap) and _is_header(rd) and _is_header(ad):
                headers.append({"kind": "changed", "name": q, "file": ap,
                                "text": _diff(rd, ad)})
                continue
            diff = _diff(rd, ad)
            changed.append({"name": q, "base_file": rp, "head_file": ap,
                            "diff": diff})
        removed += [{"name": q, "file": p} for _, p in rs[pairs:]]
        for d, p in as_[pairs:]:
            if exempt(q, p) and _is_header(d):
                headers.append({"kind": "added", "name": q, "file": p,
                                "text": d.replace(", ", ",\n")})
            else:
                added.append({"name": q, "file": p})
    if class_map is None:
        return sum(h.values()), removed, added, changed
    return sum(h.values()), removed, added, changed, moved, headers, sensitive


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prove a move changed no def or class body.")
    ap.add_argument("paths", nargs="+", help="files, dirs or globs (source and destination)")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--head", default="WORKTREE", help="git rev or WORKTREE")
    ap.add_argument("--repo", default=".", help="repository root (default: cwd)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--class-map", action="append", default=[],
                    metavar="OLD=NEW[,NEW...]",
                    help="treat methods moved from class OLD into the NEW "
                         "classes as moves (repeatable)")
    a = ap.parse_args(argv)
    class_map: dict[str, list[str]] | None = None
    if a.class_map:
        class_map = {}
        for spec in a.class_map:
            old, sep, news = spec.partition("=")
            names = [n for n in news.split(",") if n]
            if not (sep and old and names):
                ap.error(f"--class-map expects OLD=NEW[,NEW...], got {spec!r}")
            if old in names:
                ap.error(f"--class-map: {old} cannot be mapped to itself")
            class_map.setdefault(old, []).extend(names)
    base_defs, base_mod = gather(a.repo, a.base, a.paths)
    head_defs, head_mod = gather(a.repo, a.head, a.paths)
    moved: list = []
    headers: list = []
    sensitive: list = []
    if class_map is None:
        total, removed, added, changed = compare(base_defs, head_defs)
    else:
        total, removed, added, changed, moved, headers, sensitive = compare(
            base_defs, head_defs, class_map,
            head_bases(a.repo, a.head, a.paths))
    delta = {"removed": sum((base_mod - head_mod).values()),
             "added": sum((head_mod - base_mod).values())}
    ok = not (removed or added or changed)
    review = ok and bool(headers)
    info = (f"module-level delta (informational): "
            f"-{delta['removed']} +{delta['added']} statements")

    def print_extra() -> None:
        for m in moved:
            print(f"  moved:   {m['from']} -> {m['to']}  "
                  f"({m['base_file']} -> {m['head_file']})")
        if headers:
            print("class headers (review by hand):")
            for hd in headers:
                print(f"  {hd['kind']}: {hd['name']}  ({hd['file']})")
                for line in hd["text"].splitlines():
                    print("    " + line)

    if a.json:
        out = {"ok": ok and not review, "defs": total, "removed": removed,
               "added": added, "changed": changed,
               "module_level_delta": delta}
        if class_map is not None:
            out["moved"] = moved
            out["class_headers"] = headers
            out["class_sensitive"] = sensitive
            out["needs_review"] = review
        print(json.dumps(out, indent=2))
    elif ok and review:
        print(f"verify_move: NEEDS REVIEW — {total} defs, {len(moved)} moved, "
              f"{len(headers)} class headers exempted")
        print_extra()
        print(info)
    elif ok:
        if class_map is None:
            print(f"verify_move: OK — {total} defs, identical multiset")
        else:
            print(f"verify_move: OK — {total} defs, {len(moved)} moved")
        print_extra()
        print(info)
    else:
        print("verify_move: FAIL")
        print_extra()
        for sv in sensitive:
            print(f"  class-sensitive: {sv['name']} ({sv['reason']})")
        for r in removed:
            print(f"  removed: {r['name']}  ({r['file']})")
        for r in added:
            print(f"  added:   {r['name']}  ({r['file']})")
        for c in changed:
            print(f"  changed: {c['name']}  ({c['base_file']} -> {c['head_file']})")
            for line in c["diff"].splitlines():
                print("    " + line)
        print(info)
    return 2 if review else (0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
