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
            # so 'diagrams' is checked regardless of level.
            if "diagrams" in module:
                offenders.append((node.lineno, module))
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
