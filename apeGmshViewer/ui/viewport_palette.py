"""
viewport_palette — push a ``Palette`` onto a PyVista plotter.

PyVista's default plotter exposes a linear vertical gradient via
``set_background(bottom_color, top=top_color)``. That is a faithful
match for ``background_mode == "linear"`` and an acceptable
approximation for ``"radial"`` (VTK's default render window does
not natively render a radial gradient, so the linear approximation
reads as a vignette-adjacent cue without custom shader work).
``"flat_corner"`` falls back to a single flat color; the corner
falloff effect is a future enhancement.

Anti-aliasing, axis-widget styling, and mesh-actor colors that read
palette values live at the renderer/main-window level; this helper
is intentionally scoped to *viewport state the plotter owns globally*
so it can be called from a theme observer without touching actors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apeGmshViewer.ui.theme import Palette


def apply_palette_to_plotter(plotter: Any, palette: "Palette") -> None:
    """Apply the background region of ``palette`` to ``plotter``.

    Safe to call repeatedly — each call fully re-sets the background.
    """
    mode = palette.background_mode
    if mode == "flat_corner":
        plotter.set_background(palette.bg_top)
    else:
        # "linear" and "radial" — bg_top goes at top, bg_bottom at bottom
        # (matches the ``Palette`` docstring semantics).
        plotter.set_background(palette.bg_bottom, top=palette.bg_top)
