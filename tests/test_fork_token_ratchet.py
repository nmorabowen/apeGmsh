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
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
SRC = _ROOT / "src" / "apeGmsh"


def _quirks():
    """scripts/check_quirks.py, which owns the token definition (rule ``fork-token-string``)."""
    spec = importlib.util.spec_from_file_location(
        "_check_quirks_for_ratchet", _ROOT / "scripts" / "check_quirks.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


_Q = _quirks()

# package glob -> pinned count (measured 2026-10-04 at origin/main 38b482bb).
BASELINE: dict[str, int] = {
    "_kernel/**": 8,
    "core/**": 3,
    "mesh/**": 4,
    "results/**": 37,
}

# The fork seam: readers for the fork's own output format.
EXEMPT_GLOBS = _Q.FORK_SEAM_GLOBS

# One definition of a fork token, shared with the ``fork-token-string`` lint rule.
_TOKEN = _Q.FORK_TOKEN
_doc_constants = _Q.doc_constant_ids


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
