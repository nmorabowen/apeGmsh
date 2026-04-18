"""Viewport background modes — linear / radial / flat_corner.

PyVista's ``set_background`` handles linear vertical gradient natively.
Radial is rendered as a VTK textured background (procedurally-generated
RGB image). ``flat_corner`` uses a very-soft falloff (Paper theme) so
the model doesn't blend into the UI chrome edge.

This module is the single dispatch point for ``palette.background_mode``
so viewers don't branch on the mode themselves.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv
    from apeGmsh.viewers.ui.theme import Palette


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _radial_gradient_texture(
    center_hex: str,
    edge_hex: str,
    size: int = 1024,
    falloff_exp: float = 1.0,
    dither: float = 1.0,
):
    """Build a VTK texture with a center→edge radial gradient.

    ``falloff_exp`` < 1 pulls the edge color inward (stronger vignette);
    >= 1 keeps the bright center wider (softer). Paper uses a high
    exponent so only the extreme corners darken.

    ``dither`` is the amplitude (in 8-bit RGB units) of uniform noise
    added before quantization. Without it, narrow gradients like the
    Neutral Studio #2a→#02 range show visible banding in flat screen
    regions; one unit of uniform noise turns the bands into imperceptible
    per-pixel noise.
    """
    import vtk
    from vtk.util import numpy_support

    c = np.array(_hex_to_rgb(center_hex), dtype=np.float32)
    e = np.array(_hex_to_rgb(edge_hex), dtype=np.float32)

    ax = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    X, Y = np.meshgrid(ax, ax)
    r = np.clip(np.sqrt(X ** 2 + Y ** 2) / np.sqrt(2.0), 0.0, 1.0)
    t = r ** falloff_exp

    img_f = (
        c[None, None, :] * (1.0 - t[:, :, None])
        + e[None, None, :] * t[:, :, None]
    )
    if dither > 0.0:
        rng = np.random.default_rng(seed=0)
        img_f = img_f + rng.uniform(
            -dither * 0.5, dither * 0.5, size=img_f.shape,
        ).astype(np.float32)
    img = np.clip(img_f, 0.0, 255.0).astype(np.uint8)

    flat = img.reshape(-1, 3)
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True)
    vtk_arr.SetName("RGB")

    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(size, size, 1)
    vtk_img.GetPointData().SetScalars(vtk_arr)

    tex = vtk.vtkTexture()
    tex.SetInputData(vtk_img)
    tex.InterpolateOn()
    return tex


def apply_background(plotter: "pv.Plotter", palette: "Palette") -> None:
    """Apply *palette*'s background_mode to the renderer.

    Safe to call repeatedly (e.g. on live theme switch) — clears any
    prior textured background before installing the new mode.
    """
    renderer = plotter.renderer
    mode = palette.background_mode

    if mode == "linear":
        try:
            renderer.SetTexturedBackground(False)
        except Exception:
            pass
        plotter.set_background(palette.bg_top, top=palette.bg_bottom)
        return

    # Both radial modes install a texture; the only difference is the
    # falloff exponent (tighter vignette vs. corner-only darkening).
    falloff_exp = 1.0 if mode == "radial" else 2.5
    tex = _radial_gradient_texture(
        palette.bg_top, palette.bg_bottom, falloff_exp=falloff_exp,
    )
    # Fallback solid color if the texture somehow fails at draw time.
    rgb01 = tuple(c / 255.0 for c in _hex_to_rgb(palette.bg_top))
    renderer.SetBackground(*rgb01)
    renderer.SetGradientBackground(False)
    renderer.SetBackgroundTexture(tex)
    renderer.SetTexturedBackground(True)
