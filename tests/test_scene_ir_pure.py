"""INV-1 acceptance test — ``scene_ir`` imports no vtk/pyvista.

Walks every Python source file under ``src/apeGmsh/viewers/scene_ir/``
and asserts none of them import ``vtk`` / ``vtkmodules`` / ``pyvista``
/ ``pyvistaqt``.  This is the render-side mirror of
``test_viewers_pure_h5_consumer.py`` and enforces INV-1 of
[ADR 0042](../architecture/decisions/0042-render-backend-seam.md):
the scene IR is a pure value vocabulary, constructible and assertable
with no GPU and no render context.

The check is structural (AST-based) so a re-namespaced or aliased
import is caught at PR review, not at runtime.
"""
from __future__ import annotations

import ast
from pathlib import Path

SCENE_IR_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
    / "scene_ir"
)

FORBIDDEN_ROOTS = frozenset({"vtk", "vtkmodules", "pyvista", "pyvistaqt"})


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def _scene_ir_files() -> list[Path]:
    return sorted(p for p in SCENE_IR_DIR.rglob("*.py") if p.is_file())


def _collect_offending_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root(alias.name) in FORBIDDEN_ROOTS:
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative imports never reach vtk/pyvista
            module = node.module or ""
            if module and _root(module) in FORBIDDEN_ROOTS:
                offenders.append((node.lineno, module))
    return offenders


def test_scene_ir_dir_exists() -> None:
    assert SCENE_IR_DIR.is_dir(), (
        f"scene_ir/ not found at {SCENE_IR_DIR}; update the path constant "
        "if the package moved."
    )


def test_scene_ir_imports_no_vtk_or_pyvista() -> None:
    files = _scene_ir_files()
    assert files, "No scene_ir source files found — test path is wrong."

    leaks: list[tuple[Path, int, str]] = []
    for path in files:
        for lineno, module in _collect_offending_imports(path):
            leaks.append((path, lineno, module))

    if leaks:
        root = SCENE_IR_DIR.parent.parent.parent.parent  # repo root
        msg = "\n".join(
            f"  {p.relative_to(root)}:{lno}  →  {mod!r}"
            for p, lno, mod in sorted(
                (p, lno, mod) for p, lno, mod in leaks
            )
        )
        raise AssertionError(
            "scene_ir/ must import neither vtk nor pyvista (ADR 0042 "
            f"INV-1).\nFound {len(leaks)} forbidden import(s):\n{msg}"
        )
