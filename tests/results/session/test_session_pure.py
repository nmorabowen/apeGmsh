"""Scope guard — the session package never touches the old ontology.

Walks every source file under ``src/apeGmsh/results/session/`` and
asserts none imports ``apeGmsh.viewers.diagrams`` (the ontology ADR
0098 replaces) nor any Qt / VTK / pyvista module (the session is IR;
Qt is a client). AST-based, same idiom as ``test_scene_ir_pure.py``,
so an aliased or re-namespaced import is caught at review time.

The recorded layering exception (S0 decision 1) — importing
``apeGmsh.viewers.core.selection`` and ``apeGmsh.viewers.scene_ir`` —
is deliberately NOT forbidden here.
"""
from __future__ import annotations

import ast
from pathlib import Path

SESSION_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "apeGmsh"
    / "results"
    / "session"
)

FORBIDDEN_ROOTS = frozenset({
    "vtk", "vtkmodules", "pyvista", "pyvistaqt",
    "PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy",
})


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def _offending_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in FORBIDDEN_ROOTS:
                    offenders.append((node.lineno, alias.name))
                if "diagrams" in alias.name:
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Relative imports can climb to apeGmsh.viewers.* (level 3),
            # so 'diagrams' is checked regardless of level — and against
            # the imported NAMES too, or `from apeGmsh.viewers import
            # diagrams` slips through (probe H28).
            if "diagrams" in module:
                offenders.append((node.lineno, module))
            for alias in node.names:
                if "diagrams" in alias.name:
                    offenders.append(
                        (node.lineno, f"{module}.{alias.name}")
                    )
            if not node.level and module and _root(module) in FORBIDDEN_ROOTS:
                offenders.append((node.lineno, module))
    return offenders


def test_session_dir_exists() -> None:
    assert SESSION_DIR.is_dir(), (
        f"results/session/ not found at {SESSION_DIR}; update the path "
        "constant if the package moved."
    )


def test_session_imports_no_diagrams_qt_or_vtk() -> None:
    files = sorted(p for p in SESSION_DIR.rglob("*.py") if p.is_file())
    assert files, "No session source files found — test path is wrong."

    leaks: list[tuple[Path, int, str]] = []
    for path in files:
        for lineno, module in _offending_imports(path):
            leaks.append((path, lineno, module))

    if leaks:
        msg = "\n".join(
            f"  {p.name}:{lno}  →  {mod!r}" for p, lno, mod in sorted(leaks)
        )
        raise AssertionError(
            "results/session/ must import neither viewers.diagrams nor "
            f"any Qt/VTK module (ADR 0098 S0 scope guard).\n"
            f"Found {len(leaks)} forbidden import(s):\n{msg}"
        )


def test_guard_catches_each_forbidden_import_shape(tmp_path) -> None:
    """Mutation test of the guard itself — a green guard proves nothing
    until a forbidden import actually trips it. Every shape a retirement
    sweep could realistically reintroduce must be caught."""
    bad_sources = [
        "import apeGmsh.viewers.diagrams\n",
        "from apeGmsh.viewers import diagrams\n",              # probe H28
        "from apeGmsh.viewers.diagrams import _styles\n",
        "from apeGmsh.viewers.diagrams._kinds import KindDef\n",
        "import apeGmsh.viewers.diagrams as dg\n",
        "from ...viewers.diagrams import _styles\n",           # relative climb
        "import vtk\n",
        "import pyvista\n",
        "from pyvistaqt import QtInteractor\n",
        "from PyQt5 import QtWidgets\n",
        "import PySide6.QtWidgets\n",
        "from qtpy.QtCore import QObject\n",
    ]
    probe = tmp_path / "probe.py"
    misses = []
    for src in bad_sources:
        probe.write_text(src, encoding="utf-8")
        if not _offending_imports(probe):
            misses.append(src.strip())
    assert not misses, f"Guard misses forbidden import shapes: {misses}"

    # And the recorded layering exception stays ALLOWED:
    probe.write_text(
        "from apeGmsh.viewers.core.selection import SelectionState\n"
        "from apeGmsh.viewers.scene_ir import SelectionTarget\n",
        encoding="utf-8",
    )
    assert _offending_imports(probe) == []
