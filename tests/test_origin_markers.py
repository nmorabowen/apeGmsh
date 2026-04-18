"""Tests for the origin-marker overlay (reference-point glyphs)."""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.scene.origin_markers import build_origin_markers


@pytest.fixture
def plotter():
    p = pv.Plotter(off_screen=True)
    yield p
    p.close()


def test_empty_points_returns_none(plotter):
    glyph, label = build_origin_markers(
        plotter, [], origin_shift=np.zeros(3), model_diagonal=1.0,
    )
    assert glyph is None
    assert label is None


def test_single_origin_at_world_zero(plotter):
    glyph, _ = build_origin_markers(
        plotter, [(0.0, 0.0, 0.0)],
        origin_shift=np.zeros(3),
        model_diagonal=10.0,
        show_coords=False,
    )
    assert glyph is not None


def test_multiple_points_builds_glyph(plotter):
    points = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0)]
    glyph, _ = build_origin_markers(
        plotter, points,
        origin_shift=np.zeros(3),
        model_diagonal=10.0,
        show_coords=False,
    )
    assert glyph is not None


def test_origin_shift_is_subtracted_before_render(plotter):
    """Marker at world (10,0,0) with shift (10,0,0) should render at (0,0,0)."""
    # We build the glyph and inspect its bounds — center should land near
    # (0,0,0) in render space because the shift cancels the world coord.
    glyph, _ = build_origin_markers(
        plotter, [(10.0, 0.0, 0.0)],
        origin_shift=np.array([10.0, 0.0, 0.0]),
        model_diagonal=10.0,
        show_coords=False,
    )
    assert glyph is not None
    bounds = glyph.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    assert abs(cx) < 1e-6
    assert abs(cy) < 1e-6
    assert abs(cz) < 1e-6


def test_palette_has_origin_marker_color():
    from apeGmsh.viewers.ui.theme import PALETTES
    for pal in PALETTES.values():
        assert pal.origin_marker_color.startswith("#")
        assert len(pal.origin_marker_color) == 7  # #RRGGBB


# ──────────────────────────────────────────────────────────────────────
# Live overlay manager
# ──────────────────────────────────────────────────────────────────────

def test_overlay_add_remove_toggle(plotter):
    from apeGmsh.viewers.overlays.origin_markers_overlay import OriginMarkerOverlay
    overlay = OriginMarkerOverlay(
        plotter,
        origin_shift=np.zeros(3),
        model_diagonal=10.0,
        points=[(0.0, 0.0, 0.0)],
        show_coords=False,
    )
    assert len(overlay.points) == 1

    overlay.add((1.0, 2.0, 3.0))
    assert overlay.points[-1] == (1.0, 2.0, 3.0)
    assert len(overlay.points) == 2

    overlay.remove(0)
    assert len(overlay.points) == 1
    assert overlay.points[0] == (1.0, 2.0, 3.0)

    overlay.set_visible(False)
    assert overlay.visible is False
    overlay.set_visible(True)
    assert overlay.visible is True

    overlay.clear()
    assert overlay.points == []


def test_overlay_remove_out_of_range_is_noop(plotter):
    from apeGmsh.viewers.overlays.origin_markers_overlay import OriginMarkerOverlay
    overlay = OriginMarkerOverlay(
        plotter,
        origin_shift=np.zeros(3),
        model_diagonal=1.0,
        points=[(0.0, 0.0, 0.0)],
        show_coords=False,
    )
    overlay.remove(5)  # Out of range — must not raise
    assert len(overlay.points) == 1


def test_overlay_set_size_grows_glyph(plotter):
    from apeGmsh.viewers.overlays.origin_markers_overlay import OriginMarkerOverlay
    overlay = OriginMarkerOverlay(
        plotter,
        origin_shift=np.zeros(3),
        model_diagonal=10.0,
        points=[(0.0, 0.0, 0.0)],
        show_coords=False,
        size=10.0,
    )
    small = overlay._glyph_actor.GetBounds()
    small_extent = small[1] - small[0]

    overlay.set_size(30.0)
    big = overlay._glyph_actor.GetBounds()
    big_extent = big[1] - big[0]

    assert big_extent > small_extent * 2.0  # tripled size → roughly 3× extent


def test_overlay_set_origin_shift_rerenders(plotter):
    from apeGmsh.viewers.overlays.origin_markers_overlay import OriginMarkerOverlay
    overlay = OriginMarkerOverlay(
        plotter,
        origin_shift=np.zeros(3),
        model_diagonal=1.0,
        points=[(10.0, 0.0, 0.0)],
        show_coords=False,
    )
    overlay.set_origin_shift(np.array([10.0, 0.0, 0.0]))
    assert overlay._glyph_actor is not None
    bounds = overlay._glyph_actor.GetBounds()
    cx = 0.5 * (bounds[0] + bounds[1])
    assert abs(cx) < 1e-6  # (10,0,0) - (10,0,0) = render at (0,0,0)
