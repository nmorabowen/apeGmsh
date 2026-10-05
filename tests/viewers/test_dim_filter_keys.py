"""Digit dim-filter keys must not be VTK ``add_key_event``.

An ``add_key_event`` binding fires only while the VTK viewport has
keyboard focus, so with focus in a dock those keypresses never reach it
(measured 2026-09-25; internal_docs/viewer_lessons.md, "Key bindings in
a VTK-hosted window"). The same law applies to ResultsViewer Esc and the
section-builder F7/F8/F9 shortcuts. The 0/1/2/3/4 contract is
``QShortcut`` / ``add_shortcut(..., application=True)``.
"""
from __future__ import annotations

import ast
from pathlib import Path

VIEWERS = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "viewers"


def _unit(class_name: str) -> list[Path]:
    """The module defining ``class_name`` (found by AST walk, not by
    filename) plus its package siblings if it was split into a package.
    A missing or ambiguous class raises, so the guard cannot pass
    vacuously after a hub move."""
    found = []
    for p in sorted(VIEWERS.rglob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        if any(isinstance(n, ast.ClassDef) and n.name == class_name
               for n in tree.body):
            found.append(p)
    assert len(found) == 1, (
        f"expected exactly one class {class_name!r} under {VIEWERS}, "
        f"found {[str(f) for f in found]}"
    )
    f = found[0]
    pkg = f.parent if f.name == "__init__.py" else f.with_suffix("")
    files = [f]
    if pkg.is_dir():
        files += sorted(q for q in pkg.rglob("*.py") if q != f)
    return files


def _lines(unit: list[Path]) -> list[int]:
    return [ln for p in unit for ln in _digit_add_key_event_lines(p)]


def _src(unit: list[Path]) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in unit)


_DIGIT_KEYS = frozenset({"0", "1", "2", "3", "4"})


def _digit_add_key_event_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None)
        if name != "add_key_event" or not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and arg0.value in _DIGIT_KEYS:
            hits.append(node.lineno)
    return hits


def test_model_viewer_dim_keys_are_not_vtk_events() -> None:
    unit = _unit("ModelViewer")
    assert _lines(unit) == []
    src = _src(unit)
    assert "application=True" in src
    assert 'win.add_shortcut' in src


def test_mesh_viewer_dim_keys_are_not_vtk_events() -> None:
    unit = _unit("MeshViewer")
    assert _lines(unit) == []
    src = _src(unit)
    assert "application=True" in src


def test_results_viewer_dim_keys_are_not_vtk_events() -> None:
    unit = _unit("ResultsViewer")
    assert _lines(unit) == []
    src = _src(unit)
    assert "ApplicationShortcut" in src
    assert "_results_filter.toggle" in src
