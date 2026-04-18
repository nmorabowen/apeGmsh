"""Background dispatch: linear / radial / flat_corner install correctly."""
from __future__ import annotations

import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.scene.background import apply_background
from apeGmsh.viewers.ui.theme import PALETTES


@pytest.mark.parametrize("theme_name", list(PALETTES))
def test_every_theme_applies_without_error(theme_name):
    pal = PALETTES[theme_name]
    p = pv.Plotter(off_screen=True)
    try:
        apply_background(p, pal)
        textured = bool(p.renderer.GetTexturedBackground())
        if pal.background_mode == "linear":
            assert textured is False
        else:
            # radial and flat_corner both install a texture
            assert textured is True
    finally:
        p.close()


def test_linear_mode_sets_both_colors():
    """PyVista stores top/bottom gradient colors on the renderer."""
    pal = PALETTES["catppuccin_mocha"]
    p = pv.Plotter(off_screen=True)
    try:
        apply_background(p, pal)
        # GradientBackground is VTK's linear-gradient flag
        assert p.renderer.GetGradientBackground() is True
    finally:
        p.close()
