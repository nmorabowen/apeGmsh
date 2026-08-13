"""Digit dim-filter keys must not be VTK ``add_key_event``.

VTK's QtInteractor swallows those keypresses (the same law as
ResultsViewer Esc and the section-builder F7/F8/F9 shortcuts). The
0/1/2/3/4 contract is ``QShortcut`` / ``add_shortcut(..., application=True)``.
"""
from __future__ import annotations

import ast
from pathlib import Path

VIEWERS = Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "viewers"
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
    path = VIEWERS / "model_viewer.py"
    assert _digit_add_key_event_lines(path) == []
    src = path.read_text(encoding="utf-8")
    assert "application=True" in src
    assert 'win.add_shortcut' in src


def test_mesh_viewer_dim_keys_are_not_vtk_events() -> None:
    path = VIEWERS / "mesh_viewer.py"
    assert _digit_add_key_event_lines(path) == []
    src = path.read_text(encoding="utf-8")
    assert "application=True" in src


def test_results_viewer_dim_keys_are_not_vtk_events() -> None:
    path = VIEWERS / "results_viewer.py"
    assert _digit_add_key_event_lines(path) == []
    src = path.read_text(encoding="utf-8")
    assert "ApplicationShortcut" in src
    assert "_results_filter.toggle" in src
