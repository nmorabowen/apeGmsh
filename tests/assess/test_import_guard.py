"""ADR 0094 INV-1 — ``apeGmsh.assess`` must not import viewers, gmsh,
or (Amendment 2) ``apeGmsh.opensees``.

Walks every ``.py`` under ``src/apeGmsh/assess/`` and flags
``import gmsh`` / ``from gmsh …`` / ``import apeGmsh.viewers`` /
``from apeGmsh.viewers …`` / ``import apeGmsh.opensees`` /
``from apeGmsh.opensees …``. Modeled on
``tests/test_viewers_pure_h5_consumer.py``; the positive-control
pattern follows ``tests/opensees/h5/test_model_data_ast_guard.py``.
"""
from __future__ import annotations

import ast
from pathlib import Path


ASSESS_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "apeGmsh" / "assess"
)


def _assess_python_files() -> list[Path]:
    return sorted(p for p in ASSESS_DIR.rglob("*.py") if p.is_file())


def _is_forbidden(module: str) -> bool:
    if module == "gmsh" or module.startswith("gmsh."):
        return True
    if module == "apeGmsh.viewers" or module.startswith("apeGmsh.viewers."):
        return True
    if module == "apeGmsh.opensees" or module.startswith("apeGmsh.opensees."):
        return True
    return False


def collect_offending_imports(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for every forbidden import in ``source``."""
    tree = ast.parse(source, filename="<input>")
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            module = node.module or ""
            if _is_forbidden(module):
                offenders.append((node.lineno, module))
            for alias in node.names:
                full = f"{module}.{alias.name}" if module else alias.name
                if _is_forbidden(full):
                    offenders.append((node.lineno, full))
    return offenders


def test_assess_dir_exists() -> None:
    assert ASSESS_DIR.is_dir(), (
        f"assess/ not found at {ASSESS_DIR}. If the package has moved, "
        f"update the path constant in this test."
    )


def test_assess_has_no_viewers_or_gmsh_imports() -> None:
    files = _assess_python_files()
    assert files, "No assess source files found — test path is wrong."

    leaks: list[tuple[Path, int, str]] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for lineno, module in collect_offending_imports(source):
            leaks.append((path, lineno, module))

    assert not leaks, (
        "apeGmsh.assess must not import apeGmsh.viewers or gmsh "
        "(ADR 0094 INV-1). Offenders:\n"
        + "\n".join(
            f"  {path.relative_to(ASSESS_DIR.parent.parent)}:{ln}  {mod}"
            for path, ln, mod in leaks
        )
    )


def test_positive_control_catches_gmsh_import() -> None:
    src = "import gmsh\nfrom gmsh import model\n"
    found = collect_offending_imports(src)
    assert any(mod == "gmsh" for _, mod in found), found
    assert any(mod.startswith("gmsh") for _, mod in found), found


def test_positive_control_catches_viewers_import() -> None:
    src = (
        "import apeGmsh.viewers\n"
        "from apeGmsh.viewers import ResultsViewer\n"
        "from apeGmsh.viewers.render import render_pack\n"
    )
    found = collect_offending_imports(src)
    modules = {mod for _, mod in found}
    assert "apeGmsh.viewers" in modules, found
    assert any(m.startswith("apeGmsh.viewers") for m in modules), found


def test_positive_control_catches_opensees_import() -> None:
    src = (
        "import apeGmsh.opensees\n"
        "from apeGmsh.opensees import OpenSeesModel\n"
        "from apeGmsh.opensees.opensees_model import OpenSeesModel\n"
    )
    found = collect_offending_imports(src)
    modules = {mod for _, mod in found}
    assert "apeGmsh.opensees" in modules, found
    assert any(m.startswith("apeGmsh.opensees") for m in modules), found


def test_inv9_assess_does_not_read_all_steps() -> None:
    """INV-9: extrema use time=-1; never nodes.get(..., time=None)."""
    for path in _assess_python_files():
        source = path.read_text(encoding="utf-8")
        assert "time=None" not in source, path
        assert "time = None" not in source, path
        assert "node_envelope" not in source, path
