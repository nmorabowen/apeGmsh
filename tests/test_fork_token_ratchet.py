"""K2 ratchet: the hub packages may not grow Ladruno fork tokens.

``internal_docs/plan_expert_panel_2026-09.md`` K2: ``_kernel``, ``core``,
``mesh`` and ``results`` must not accumulate fork-specific code.  New fork
logic belongs behind the fork seam (``results/readers/_ladruno*.py``,
``opensees/``).  The count is keyed by package glob, never by filename, so
it survives file splits and moves inside a package.

Token definition: an AST node whose text contains ``ladruno``
case-insensitively, among
  * identifiers: ``Name.id``, ``Attribute.attr``, def/class names,
    ``arg`` names, ``keyword`` names, import ``alias`` names;
  * string constants that are not docstrings (f-string literal parts
    included).
Comments and docstrings are excluded: prose may name the fork, code may not
grow a dependency on it.

The assertion is ``current <= baseline`` per package.  A baseline may only
shrink.  Raising one is a maintainer gate, never a fix for a red test.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "apeGmsh"

# package glob -> pinned count (measured 2026-10-04 at origin/main 38b482bb).
BASELINE: dict[str, int] = {
    "_kernel/**": 8,
    "core/**": 3,
    "mesh/**": 4,
    "results/**": 37,
}

# The fork seam: readers for the fork's own output format.
EXEMPT_GLOBS = ("results/readers/_ladruno*.py",)

_TOKEN = re.compile("ladruno", re.IGNORECASE)
_DOC_OWNERS = (
    ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef,
)


def _doc_constants(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                ids.add(id(first.value))
    return ids


def count_fork_tokens(source: str) -> int:
    tree = ast.parse(source)
    docs = _doc_constants(tree)
    n = 0
    for node in ast.walk(tree):
        texts: list[str] = []
        if isinstance(node, ast.Name):
            texts = [node.id]
        elif isinstance(node, ast.Attribute):
            texts = [node.attr]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            texts = [node.name]
        elif isinstance(node, ast.arg):
            texts = [node.arg]
        elif isinstance(node, ast.keyword) and node.arg:
            texts = [node.arg]
        elif isinstance(node, ast.alias):
            texts = [node.name, node.asname or ""]
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docs):
            texts = [node.value]
        n += sum(1 for t in texts if _TOKEN.search(t))
    return n


def _exempt(path: Path) -> bool:
    rel = path.relative_to(SRC)
    return any(rel.match(g) for g in EXEMPT_GLOBS)


def count_package(glob: str) -> int:
    total = 0
    for path in sorted(SRC.glob(glob.replace("**", "**/*.py"))):
        if path.suffix == ".py" and not _exempt(path):
            total += count_fork_tokens(path.read_text(encoding="utf-8"))
    return total


@pytest.mark.parametrize("glob", sorted(BASELINE))
def test_fork_tokens_do_not_grow(glob: str) -> None:
    current = count_package(glob)
    assert current <= BASELINE[glob], (
        f"{glob}: {current} fork tokens > baseline {BASELINE[glob]}. "
        "Fork logic belongs behind the fork seam (results/readers/_ladruno*.py "
        "or opensees/). Do not raise the baseline."
    )


def test_counter_semantics() -> None:
    src = (
        '"""Ladruno in a docstring does not count."""\n'
        "# Ladruno in a comment does not count\n"
        "from x import LadrunoBrick as B\n"   # alias name -> 1
        "y = 'ladruno_fork'\n"                 # string -> 1
        "def LadrunoThing(ladruno_flag=1):\n"  # def + arg -> 2
        "    return o.ladruno\n"               # attribute -> 1
    )
    assert count_fork_tokens(src) == 5
