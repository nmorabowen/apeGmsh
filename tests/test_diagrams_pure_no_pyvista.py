"""INV-2 acceptance test — ``diagrams/`` imports no vtk/pyvista.

Walks every Python source file under ``src/apeGmsh/viewers/diagrams/``
and asserts none of them import ``vtk`` / ``vtkmodules`` / ``pyvista``
/ ``pyvistaqt``.  This is INV-2 of
[ADR 0042](../src/apeGmsh/opensees/architecture/decisions/0042-render-backend-seam.md):
after the R-B diagram migrations, the domain layer emits ``SceneLayer``
value types through a ``RenderBackend`` and never touches a render
backend's API directly — so it must be constructible and testable with
no GPU and no render context.

The render-side sibling of ``test_scene_ir_pure.py`` (INV-1) and
``test_viewers_pure_h5_consumer.py`` (read-side). The check is
structural (AST-based) so a re-namespaced or aliased import is caught
at PR review, not at runtime.

Relative imports are deliberately ignored: a diagram may import the
``cellblocks_from_grid`` bridge via ``from ..backends.pyvista_qt import
...`` (a diagrams→backends coupling, not a pyvista import). INV-2 polices
only direct ``vtk`` / ``pyvista`` imports; the pyvista-free submesh
accessor that would retire that bridge import is separate later cleanup.
"""
from __future__ import annotations

import ast
from pathlib import Path

DIAGRAMS_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "apeGmsh"
    / "viewers"
    / "diagrams"
)

FORBIDDEN_ROOTS = frozenset({"vtk", "vtkmodules", "pyvista", "pyvistaqt"})


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def _diagram_files() -> list[Path]:
    return sorted(p for p in DIAGRAMS_DIR.rglob("*.py") if p.is_file())


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


def test_diagrams_dir_exists() -> None:
    assert DIAGRAMS_DIR.is_dir(), (
        f"diagrams/ not found at {DIAGRAMS_DIR}; update the path constant "
        "if the package moved."
    )


def test_diagrams_import_no_vtk_or_pyvista() -> None:
    files = _diagram_files()
    assert files, "No diagrams source files found — test path is wrong."

    leaks: list[tuple[Path, int, str]] = []
    for path in files:
        for lineno, module in _collect_offending_imports(path):
            leaks.append((path, lineno, module))

    if leaks:
        root = DIAGRAMS_DIR.parent.parent.parent.parent  # repo root
        msg = "\n".join(
            f"  {p.relative_to(root)}:{lno}  →  {mod!r}"
            for p, lno, mod in sorted(
                (p, lno, mod) for p, lno, mod in leaks
            )
        )
        raise AssertionError(
            "diagrams/ must import neither vtk nor pyvista (ADR 0042 "
            f"INV-2).\nFound {len(leaks)} forbidden import(s):\n{msg}"
        )
