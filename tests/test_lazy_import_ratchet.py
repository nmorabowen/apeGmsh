"""C4 ratchet: the lazy share of cross-package imports may not grow.

Program link C4 (#1967); PROGRAM.md section 8 lists the lazy cross-package
import share as a structure KPI and makes it a hard gate (maintainer,
2026-10-14, #1202).

Census definition (the KPI session's, stable, do not change):

* Walk ``src/apeGmsh/**/*.py`` with ``ast``.  A file's module name is its
  path under ``src/`` (``__init__.py`` names its package).  Its *package* is
  the second dotted component (``apeGmsh.mesh.x`` is ``mesh``,
  ``apeGmsh._kernel`` is ``_kernel``); ``apeGmsh/__init__.py`` is the root
  package ``""``.
* Every ``import apeGmsh...`` and ``from apeGmsh... import`` is a site;
  relative imports are resolved against the importing module first.  The
  imported package is the second component of the imported module, except
  that ``from apeGmsh import a, b`` counts one site per name, each in
  package ``a``, ``b`` (``import apeGmsh`` is the root).
* A site is *cross-package* when the imported package differs from the
  importer's.  Only cross-package sites are counted.
* A site is **lazy** when it sits inside a function body (``def``,
  ``async def``, ``lambda``) at any depth, and **eager** otherwise (module
  level, including class bodies and ``try``/``if`` blocks there).
* The body of an ``if TYPE_CHECKING:`` (or ``typing.TYPE_CHECKING``) is
  excluded at any depth; its ``else`` branch runs and is counted.

Measured with this census: 377 lazy / 174 eager at ``07f757e0`` and
482 / 186 at ``6c792de6`` (#1202, comment 6173426133).

The ratchet holds the *share*: with baseline ``B = BASELINE_LAZY /
BASELINE_TOTAL``, it asserts ``lazy * BASELINE_TOTAL <= BASELINE_LAZY *
total`` in integers.  A shrink passes and prints the pair to adopt; lowering
the baseline is optional.  Raising it is a maintainer gate (PROGRAM.md
section 4, ``human-gate``), never a fix for a red test: hoist the import to
module level, or move the code into the package it imports.

On a failure the message names the lazy sites missing from the committed
snapshot ``tests/lazy_import_snapshot.txt``, keyed on (file, enclosing
qualname, imported module), never on line numbers.  Regenerate the snapshot
together with a lowered baseline::

    python tests/test_lazy_import_ratchet.py --write-snapshot
"""
from __future__ import annotations

import ast
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SNAPSHOT = Path(__file__).resolve().with_name("lazy_import_snapshot.txt")

#: lazy / total cross-package import sites at the C4 landing commit (main 27a95f769).
BASELINE_LAZY = 519
BASELINE_TOTAL = 710

ROOT = "apeGmsh"


class Site(NamedTuple):
    """One cross-package import site."""

    file: str        # path under src/, forward slashes
    qualname: str    # enclosing function qualname, "<module>" when eager
    module: str      # absolute imported module
    lazy: bool

    def key(self) -> str:
        return f"{self.file} | {self.qualname} | {self.module}"


def module_of(rel: str) -> tuple[str, bool]:
    """``apeGmsh/x/y.py`` -> (``apeGmsh.x.y``, is_package)."""
    parts = rel[:-3].split("/")
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(parts), is_pkg


def package_of(module: str) -> str:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else ""


def _resolve(importer: str, is_pkg: bool, level: int, name: str | None) -> str:
    base = importer.split(".") if is_pkg else importer.split(".")[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    if not base:
        raise ValueError(f"relative import beyond the top level in {importer}")
    return ".".join(base + ([name] if name else []))


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _targets(node: ast.Import | ast.ImportFrom, importer: str,
             is_pkg: bool) -> Iterator[str]:
    """Absolute apeGmsh modules one import statement brings in, one per site."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == ROOT or alias.name.startswith(ROOT + "."):
                yield alias.name
        return
    module = (_resolve(importer, is_pkg, node.level, node.module)
              if node.level else (node.module or ""))
    if module == ROOT:
        for alias in node.names:
            yield ROOT if alias.name == "*" else f"{ROOT}.{alias.name}"
    elif module.startswith(ROOT + "."):
        yield module


def sites_in(rel: str, source: str) -> Iterator[Site]:
    """Every cross-package import site in one file."""
    importer, is_pkg = module_of(rel)
    pkg = package_of(importer)
    tree = ast.parse(source, filename=rel)

    def walk(node: ast.AST, scope: tuple[str, ...], lazy: bool) -> Iterator[Site]:
        for child in ast.iter_child_nodes(node):
            yield from walk_one(child, scope, lazy)

    def walk_one(child: ast.AST, scope: tuple[str, ...], lazy: bool) -> Iterator[Site]:
        if isinstance(child, ast.If) and _is_type_checking(child.test):
            for sub in child.orelse:
                yield from walk_one(sub, scope, lazy)
            return
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            for target in _targets(child, importer, is_pkg):
                if package_of(target) != pkg:
                    yield Site(rel, ".".join(scope) if lazy else "<module>",
                               target, lazy)
            return
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from walk(child, scope + (child.name,), True)
        elif isinstance(child, ast.Lambda):
            yield from walk(child, scope + ("<lambda>",), True)
        elif isinstance(child, ast.ClassDef):
            yield from walk(child, scope + (child.name,), lazy)
        else:
            yield from walk(child, scope, lazy)

    yield from walk(tree, (), False)


def census(files: Iterable[tuple[str, str]]) -> list[Site]:
    """Sites over (path under src/, source) pairs."""
    out: list[Site] = []
    for rel, source in files:
        out.extend(sites_in(rel, source))
    return out


def tree_files(src: Path = SRC) -> Iterator[tuple[str, str]]:
    for p in sorted((src / ROOT).rglob("*.py")):
        yield p.relative_to(src).as_posix(), p.read_text(encoding="utf-8")


def counts(sites: Iterable[Site]) -> tuple[int, int]:
    """(lazy, total)."""
    lazy = total = 0
    for s in sites:
        total += 1
        lazy += s.lazy
    return lazy, total


def snapshot_lines(sites: Iterable[Site]) -> list[str]:
    return sorted(s.key() for s in sites if s.lazy)


def read_snapshot(path: Path = SNAPSHOT) -> Counter[str]:
    return Counter(ln for ln in path.read_text(encoding="utf-8").splitlines()
                   if ln and not ln.startswith("#"))


def new_lazy(sites: Iterable[Site], snapshot: Counter[str]) -> list[str]:
    """Lazy site keys beyond their count in the snapshot."""
    extra = Counter(snapshot_lines(sites)) - snapshot
    return sorted(extra.elements())


def verdict(lazy: int, total: int, base_lazy: int, base_total: int,
            added: list[str]) -> tuple[bool, str | None]:
    """(ok, message): a higher share fails; a lower one passes and names the pair."""
    share = f"{lazy}/{total} = {100 * lazy / total:.2f}%" if total else "0/0"
    base = f"{base_lazy}/{base_total} = {100 * base_lazy / base_total:.2f}%"
    if lazy * base_total > base_lazy * total:
        listing = "\n".join(f"  + {k}" for k in added) or "  (none beyond the snapshot)"
        return False, (
            f"lazy cross-package import share {share} > baseline {base}. "
            "Hoist the import to module level, or move the code into the package "
            "it imports. Do not raise the baseline (a maintainer gate).\n"
            f"Lazy sites not in tests/lazy_import_snapshot.txt "
            f"(file | enclosing qualname | imported module):\n{listing}")
    if lazy * base_total < base_lazy * total:
        return True, (f"lazy cross-package import share {share} < baseline {base}. "
                      f"Shrinking is welcome: set BASELINE_LAZY = {lazy}, "
                      f"BASELINE_TOTAL = {total} and rewrite the snapshot with "
                      "`python tests/test_lazy_import_ratchet.py --write-snapshot`.")
    return True, None


def test_lazy_share_does_not_grow() -> None:
    sites = census(tree_files())
    lazy, total = counts(sites)
    added = new_lazy(sites, read_snapshot()) if lazy * BASELINE_TOTAL > BASELINE_LAZY * total else []
    ok, msg = verdict(lazy, total, BASELINE_LAZY, BASELINE_TOTAL, added)
    print(f"lazy cross-package imports: {lazy} lazy / {total - lazy} eager"
          f" ({100 * lazy / total:.2f}%)")
    if ok and msg:
        print(msg)
    assert ok, msg


def main(argv: list[str]) -> int:
    sites = census(tree_files())
    lazy, total = counts(sites)
    if "--write-snapshot" in argv:
        SNAPSHOT.write_text(
            "# Lazy cross-package import sites: file | enclosing qualname | imported module.\n"
            "# Written by `python tests/test_lazy_import_ratchet.py --write-snapshot`.\n"
            + "".join(f"{ln}\n" for ln in snapshot_lines(sites)),
            encoding="utf-8")
    print(f"{lazy} lazy / {total - lazy} eager / {total} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
