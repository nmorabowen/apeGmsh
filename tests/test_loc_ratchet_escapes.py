"""X-f3 guards (#1845): close two escape routes from the shrink-only LOC ratchet.

``tests/test_loc_ratchet.py`` (X4-b, #1807) freezes the retiring surfaces, but
the #1843 review found two ways to shrink a frozen count without shrinking
the code.  This module closes both; it does not edit the ratchet (a BASELINE
file under land_pr check 8) and reuses its LOC definition.

1. **A new sibling package.**  Code moved out of ``viewers/`` into a new
   ``src/apeGmsh/<name>/`` lowers the ``viewers`` count, and nothing counts
   the new package.  Every top-level package must therefore be a directory
   the ratchet counts (a ``_DIRS`` entry of the ratchet; a ``BASELINE`` key
   such as ``section_gui`` or ``from_recorders`` is a surface name, not a
   counted directory) or be listed in ``PACKAGES_BASELINE``, the set present
   on main when this guard landed.
   A package is a direct child directory of ``src/apeGmsh/`` holding at least
   one ``.py`` file at any depth (so namespace packages count too).

2. **Thinning ``Results.from_recorders`` into a helper.**  The ratchet counts
   only the method's own span, so moving its body into ``_helper`` in the same
   module shrinks the count.  ``closure_loc`` counts the method plus the
   same-module defs it names, followed transitively through exactly these
   reference forms (any reference, not only a call): a bare name of a
   module-level function or class, and ``self.``/``cls.``/``Results.`` or
   ``type(self).``/``type(cls).<method>`` on a ``Results`` method.  Spans run from ``lineno`` to
   ``end_lineno`` (decorators excluded, as in the ratchet), and a line shared
   by two spans counts once.  The sum may not exceed
   ``FROM_RECORDERS_CLOSURE_BASELINE``.

Both pins are baselines.  Adding a package to ``PACKAGES_BASELINE`` or raising
``FROM_RECORDERS_CLOSURE_BASELINE`` is a maintainer gate (PROGRAM section 4,
``human-gate`` label), never a fix for a red test.  Removing a stale package
name or lowering the number only tightens the guard.

Known limits (not followed or not counted; review of #1884):

- a local alias of the class (``k = cls; k._x()``; ``from_native`` already
  binds ``cls`` to a local this way), ``getattr(cls, "_x")`` and other
  dynamic lookups, and module-level lambda or dispatch tables;
- code moved out of ``Results.py`` into another module (for example
  ``results/writers/_cache.py``): the closure is same-module only;
- a new top-level ``.py`` module under ``src/apeGmsh/`` (route 1 checks
  packages only);
- a new sibling file inside ``sections/`` or ``core/``: the ``section_gui``
  and ``rebar`` surfaces are file lists there, so code moved into a new file
  beside them escapes;
- this file is not yet in land_pr's ``BASELINE`` tuple (follow-up G4), so
  editing its pins does not by itself trip the human gate.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.test_loc_ratchet import _DIRS, SRC, from_recorders_loc, loc

_RESULTS = "results/Results.py"
_GATE = ("A new or re-added pin is a maintainer gate (PROGRAM.md section 4, "
         "`human-gate` label); never add one to make this test pass.")

#: Top-level packages under src/apeGmsh/ that are not ratchet surfaces,
#: measured on origin/main 1acff395 (2026-10-13).  Shrink-only.
PACKAGES_BASELINE: frozenset[str] = frozenset({
    "_internal", "_kernel", "assembly", "assess", "core", "cuts", "fem",
    "interop", "mesh", "opensees", "parts", "results", "sections", "viz",
})

#: Non-blank lines of Results.from_recorders plus its same-module closure,
#: measured on origin/main 1acff395 (2026-10-13).  Shrink-only.
FROM_RECORDERS_CLOSURE_BASELINE = 143


# -- route 1: new sibling package ---------------------------------------------

def top_level_packages(root: Path = SRC) -> set[str]:
    """Direct child directories of ``root`` holding at least one ``.py`` file."""
    return {d.name for d in root.iterdir()
            if d.is_dir() and d.name != "__pycache__" and any(d.rglob("*.py"))}


#: Top-level directories the ratchet counts as a whole (its ``_DIRS``).  A
#: ``BASELINE`` key alone is not enough: ``section_gui`` and ``from_recorders``
#: name file-level surfaces, so a package with that name would go uncounted.
RATCHET_PACKAGES: frozenset[str] = frozenset(d for dirs in _DIRS.values() for d in dirs)


def package_findings(root: Path = SRC) -> list[str]:
    found = top_level_packages(root)
    covered = RATCHET_PACKAGES
    msgs = [f"src/apeGmsh/{name}/ is a new top-level package that no LOC ratchet "
            "counts: code moved here from a frozen surface would escape it. Put the "
            "code in an existing package, or ask the maintainer to add "
            f"{name!r} to PACKAGES_BASELINE. {_GATE}"
            for name in sorted(found - covered - PACKAGES_BASELINE)]
    msgs += [f"PACKAGES_BASELINE lists {name!r}, which is no longer a package: "
             "delete the entry (a tightening, no ruling needed), or a later PR "
             "could re-create it unchecked."
             for name in sorted(PACKAGES_BASELINE - found)]
    msgs += [f"{name!r} is both a ratchet surface and in PACKAGES_BASELINE: "
             "remove it from PACKAGES_BASELINE."
             for name in sorted(PACKAGES_BASELINE & covered)]
    return msgs


def test_no_unpinned_top_level_package() -> None:
    msgs = package_findings()
    assert not msgs, "\n".join(msgs)


def test_new_package_fails_and_names_the_gate(tmp_path: Path) -> None:
    for name in sorted(PACKAGES_BASELINE | RATCHET_PACKAGES):
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "data_only").mkdir()  # no .py: not a package
    assert package_findings(tmp_path) == []
    (tmp_path / "_tmp_escape" / "sub").mkdir(parents=True)  # namespace package
    (tmp_path / "_tmp_escape" / "sub" / "m.py").write_text("x = 1\n", encoding="utf-8")
    (msg,) = package_findings(tmp_path)
    assert "_tmp_escape" in msg and "maintainer gate" in msg and "human-gate" in msg


@pytest.mark.parametrize("name", ["section_gui", "from_recorders"])
def test_file_level_surface_name_is_not_a_covered_package(tmp_path: Path, name: str) -> None:
    """A BASELINE key that names files, not a directory, does not admit a package."""
    assert name not in RATCHET_PACKAGES and name not in PACKAGES_BASELINE
    for pkg in sorted(PACKAGES_BASELINE | RATCHET_PACKAGES | {name}):
        (tmp_path / pkg).mkdir()
        (tmp_path / pkg / "__init__.py").write_text("", encoding="utf-8")
    (msg,) = package_findings(tmp_path)
    assert f"src/apeGmsh/{name}/ is a new top-level package" in msg


# -- route 2: from_recorders thinned into a same-module helper -----------------

def _defs(tree: ast.Module) -> tuple[dict[str, list[ast.AST]], dict[str, list[ast.AST]]]:
    """(module-level functions and classes, ``Results`` methods) by name."""
    module: dict[str, list[ast.AST]] = {}
    methods: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module.setdefault(node.name, []).append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Results":
            for fn in node.body:
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.setdefault(fn.name, []).append(fn)
    module.pop("Results", None)  # the class itself is reached method by method
    if "from_recorders" not in methods:
        raise AssertionError("Results.from_recorders not found")
    return module, methods


def _is_results_ref(node: ast.expr) -> bool:
    """``self``, ``cls``, ``Results`` or ``type(self)`` / ``type(cls)``."""
    if isinstance(node, ast.Name):
        return node.id in {"self", "cls", "Results"}
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "type" and len(node.args) == 1
            and isinstance(node.args[0], ast.Name) and node.args[0].id in {"self", "cls"})


def closure(source: str) -> list[ast.AST]:
    """``Results.from_recorders`` and every same-module def it reaches."""
    module, methods = _defs(ast.parse(source))
    seen: dict[int, ast.AST] = {}
    todo: list[ast.AST] = list(methods["from_recorders"])
    while todo:
        node = todo.pop()
        if id(node) in seen:
            continue
        seen[id(node)] = node
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in module:
                todo += module[sub.id]
            elif (isinstance(sub, ast.Attribute) and sub.attr in methods
                  and _is_results_ref(sub.value)):
                todo += methods[sub.attr]
    return sorted(seen.values(), key=lambda n: n.lineno)


def closure_loc(source: str) -> int:
    lines = source.splitlines()
    rows = {i for n in closure(source) for i in range(n.lineno - 1, n.end_lineno)}
    return loc("\n".join(lines[i] for i in sorted(rows)))


def test_from_recorders_closure_does_not_grow() -> None:
    source = (SRC / _RESULTS).read_text(encoding="utf-8")
    current = closure_loc(source)
    names = ", ".join(n.name for n in closure(source))
    if current < FROM_RECORDERS_CLOSURE_BASELINE:
        print(f"from_recorders closure: {current} LOC < baseline "
              f"{FROM_RECORDERS_CLOSURE_BASELINE}; lower it to {current}.")
    assert current <= FROM_RECORDERS_CLOSURE_BASELINE, (
        f"Results.from_recorders plus its same-module closure ({names}) is {current} "
        f"LOC > baseline {FROM_RECORDERS_CLOSURE_BASELINE}. The surface is frozen; "
        "moving its body into a helper in Results.py does not shrink it. The "
        "closure also counts from_native, a shared live callee: if the growth is "
        "legitimate work there (not code moved out of from_recorders), ask the "
        "maintainer to rule on the baseline (PROGRAM.md section 4, `human-gate` "
        "label) rather than raising it in your PR.")


_THIN = '''\
def _helper(n):
    """moved"""
    a = n + 1
    return a

class Results:
    def from_native(self):
        return _Box(1)

    @classmethod
    def from_recorders(cls, n):
        b = _helper(n)
        f = cls._ref
        return cls.from_native(Results._ref)

    def _ref(self):
        return type(self).unused

    def unused(self):
        return 0

class _Box:
    def __init__(self, v):
        self.v = v

def _unreached():
    return 1
'''


def test_closure_follows_references_and_skips_the_unreached() -> None:
    names = [n.name for n in closure(_THIN)]
    assert names == ["_helper", "from_native", "from_recorders", "_ref", "unused", "_Box"]
    assert from_recorders_loc(_THIN) == 4  # the ratchet sees the thin method only
    # _helper 4, from_native 2, from_recorders 4, _ref 2 (reached twice), unused 2,
    # _Box 3; neither _unreached nor the `class Results:` line counts
    assert closure_loc(_THIN) == 4 + 2 + 4 + 2 + 2 + 3


def test_moving_lines_into_a_helper_keeps_them_counted() -> None:
    """The guard-bites proof on a patched copy of the real source."""
    source = (SRC / _RESULTS).read_text(encoding="utf-8")
    moved = ("        out_dir = Path(output_dir)\n"
             "        cache_dir = _cache.resolve_cache_root(cache_root)\n"
             "\n"
             "        source_files = _cache.list_source_files(\n"
             "            spec, out_dir, file_format=file_format, stage_id=stage_id,\n"
             "        )\n")
    assert source.count(moved) == 1 and loc(moved) == 5
    helper = ("\n\ndef _x_f3_helper(spec, output_dir, cache_root, file_format, stage_id):\n"
              "    from .writers import _cache\n"
              + "".join(line[4:] + "\n" for line in moved.splitlines())
              + "    return out_dir, cache_dir, source_files\n")
    call = ("        out_dir, cache_dir, source_files = _x_f3_helper(\n"
            "            spec, output_dir, cache_root, file_format, stage_id)\n")
    thinned = source.replace(moved, call)
    thinned = thinned.replace("\n\nclass Results", helper + "\n\nclass Results", 1)
    ast.parse(thinned)
    # the ratchet's own span drops by 3 (5 lines out, a 2-line call in) ...
    assert from_recorders_loc(thinned) == from_recorders_loc(source) - 3
    # ... the closure keeps the 5 moved lines, plus the helper's def, import and return
    assert closure_loc(thinned) == closure_loc(source) - 3 + 5 + 3
    assert closure_loc(thinned) > FROM_RECORDERS_CLOSURE_BASELINE
    assert "_x_f3_helper" in [n.name for n in closure(thinned)]


@pytest.mark.parametrize("ref", ["_helper(n)", "cls._ref", "Results._ref", "type(cls)._ref"])
def test_each_reference_form_is_followed(ref: str) -> None:
    src = ("def _helper(n):\n    return n\n\n"
           "class Results:\n"
           "    def from_recorders(cls, n):\n"
           f"        return {ref}\n\n"
           "    def _ref(self):\n        return 1\n")
    assert len(closure(src)) == 2
