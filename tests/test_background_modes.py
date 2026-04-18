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


def test_textured_modes_clear_gradient_flag():
    """Radial/flat_corner install a texture — VTK linear-gradient flag must be off."""
    pal = PALETTES["neutral_studio"]
    p = pv.Plotter(off_screen=True)
    try:
        apply_background(p, pal)
        assert p.renderer.GetGradientBackground() is False
        assert bool(p.renderer.GetTexturedBackground()) is True
    finally:
        p.close()
