"""Guard: viewer color state must flow from the active Palette.

This suite fails if anyone re-introduces a module-level color constant
outside ``viewers/ui/theme.py``. Keeping a single source of truth is
what made theme switching coherent — the moment a snapshot constant
creeps back, "first render uses stale color" bugs return.

Exceptions (semantically not palette colors):
- ``PARTITION_COLORS`` in ``mesh_scene.py`` — used for partition
  coloring; a distinct concept from the viewport palette.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


VIEWERS_ROOT = Path(__file__).resolve().parents[1] / "src" / "apeGmsh" / "viewers"

# Allowed color-constant names that are semantically NOT palette colors
# (they encode domain meaning like partition id or constraint kind).
_ALLOWED_CONSTANTS = {
    "PARTITION_COLORS",       # mesh_scene.py — partition id coloring
    "_CONSTRAINT_COLORS",     # constraints_tab.py — constraint-kind legend
    "_FALLBACK_COLOR",        # constraints_tab.py — unknown-kind fallback
    "_FALLBACK_RGB",          # color_mode_controller.py — unknown-entity fallback
}

# Files that are legitimately allowed to define palette color constants
# (theme.py is the single source of truth).
_ALLOWED_FILES = {"theme.py"}

# Hex pattern: #RGB or #RRGGBB string literals assigned at module scope.
_MODULE_HEX_CONST = re.compile(
    r'^[A-Z_][A-Z_0-9]*\s*=\s*["\']#[0-9A-Fa-f]{3,8}["\']', re.MULTILINE,
)
# np.array tuple/list RGB constants at module scope.
_MODULE_RGB_CONST = re.compile(
    r'^[A-Z_][A-Z_0-9]*\s*=\s*np\.array\(\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+',
    re.MULTILINE,
)


def test_idle_colors_constant_is_gone():
    """IDLE_COLORS was the back-compat snapshot; it must not return."""
    from apeGmsh.viewers.core import color_manager
    assert not hasattr(color_manager, "IDLE_COLORS")


def test_default_mesh_rgb_constant_is_gone():
    """DEFAULT_MESH_RGB was the hardcoded steel blue fill; it must not return."""
    from apeGmsh.viewers.scene import mesh_scene
    assert not hasattr(mesh_scene, "DEFAULT_MESH_RGB")


def test_interaction_colors_come_from_palette():
    """ColorManager's hover/pick/hidden must read from THEME.current."""
    from apeGmsh.viewers.core import color_manager as cm
    # These module-level RGB constants should no longer exist.
    for name in ("PICK_RGB", "HOVER_RGB", "HIDDEN_RGB"):
        assert not hasattr(cm, name), (
            f"{name} is a stale snapshot — source interaction colors "
            f"from Palette.hover_rgb / pick_rgb / hidden_rgb instead."
        )


@pytest.mark.parametrize("py_file", [
    p for p in VIEWERS_ROOT.rglob("*.py")
    if p.name not in _ALLOWED_FILES
])
def test_no_module_level_color_constants(py_file):
    """Scan viewer modules for module-level hex/RGB constants."""
    text = py_file.read_text(encoding="utf-8")
    offenders: list[str] = []

    for match in _MODULE_HEX_CONST.finditer(text):
        name = match.group().split("=")[0].strip()
        if name not in _ALLOWED_CONSTANTS:
            offenders.append(f"{name} (hex)")

    for match in _MODULE_RGB_CONST.finditer(text):
        name = match.group().split("=")[0].strip()
        if name not in _ALLOWED_CONSTANTS:
            offenders.append(f"{name} (np.array RGB)")

    assert not offenders, (
        f"{py_file.relative_to(VIEWERS_ROOT.parent.parent)} defines "
        f"module-level color constants: {offenders}. "
        f"Source from Palette instead."
    )
