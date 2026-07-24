"""ADR 0045 — results dim-filter visual ghost-hide, verified on real pixels.

The layered model itself (ElementVisibility + apply_dim_filter) is covered
headlessly in test_element_visibility.py. This closes the part that was
otherwise GPU-eyeball-gated: that VTK actually *paints* the dim-hide — that
toggling a dimension through ``apply_dim_filter`` drops the painted-cell pixel
count and restoring all dims brings it back, using the exact in-place-mutate +
``Modified()`` + ``render()`` flow the live ResultsViewer drives.

Skips cleanly where there is no offscreen GL context (same guard as
test_box_select_frustum_smoke.py), so it verifies where it can and never
blocks CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.element_visibility import (
    ElementVisibility,
    apply_dim_filter,
)

CELL_DIM = np.array([1, 1, 1, 1, 1, 3, 3, 3, 3, 3], dtype=np.int8)


def _foreground_px(plotter) -> int:
    """Pixels that differ from the background corner colour."""
    img = plotter.screenshot(return_img=True)
    if img is None:
        return 0
    bg = img[0, 0].astype(int)
    return int((np.abs(img.astype(int) - bg).sum(axis=2) > 20).sum())


@pytest.fixture
def rendered_grid():
    """Row of 10 hex cells (mixed cell_dim [1]*5 + [3]*5) on a live
    offscreen plotter, mesh added once so renders reuse the mapper."""
    pv = pytest.importorskip("pyvista")
    try:
        grid = pv.ImageData(dimensions=(11, 2, 2)).cast_to_unstructured_grid()
        assert grid.n_cells == 10
        grid.cell_data["vals"] = np.arange(grid.n_cells, dtype=float)
        p = pv.Plotter(off_screen=True, window_size=(400, 200))
        p.add_mesh(grid, scalars="vals", show_edges=True)
        p.view_xy()
        p.render()
    except Exception:                    # pragma: no cover - no GL context
        pytest.skip("no offscreen render context")
    n_all = _foreground_px(p)
    if n_all == 0:                       # GL ran but produced an empty frame
        p.close()
        pytest.skip("offscreen GL produced an empty frame")

    # Capability probe. Some software / offscreen GL rasterisers paint the
    # grid but ignore ``vtkGhostType`` cell hiding, so the pixel-level
    # assertions below could never be satisfied (hiding a cell changes
    # nothing on screen). Confirm the renderer actually honours a
    # hide-everything before trusting the frame; skip cleanly where it does
    # not — the hide logic itself is covered headlessly in
    # test_element_visibility.py, so this only gates the "does VTK *paint*
    # the hide" leg. Restore all-visible before yielding.
    probe = ElementVisibility(grid)
    probe.hide(np.arange(grid.n_cells))
    p.render()
    honours_hide = _foreground_px(p) < n_all * 0.5
    probe.show_all()
    p.render()
    if not honours_hide:
        p.close()
        pytest.skip("offscreen renderer ignores vtkGhostType cell hiding")

    yield grid, p
    p.close()


def test_dim_hide_drops_then_restores_pixels(rendered_grid):
    grid, p = rendered_grid
    ev = ElementVisibility(grid)
    n_all = _foreground_px(p)

    # Hide dim 1 (the left half of the row).
    apply_dim_filter(ev, CELL_DIM, active=[3], all_dims=[1, 3])
    p.render()
    n_hidden = _foreground_px(p)

    # Re-activate all dims → dim layer cleared, full row repainted.
    apply_dim_filter(ev, CELL_DIM, active=[1, 3], all_dims=[1, 3])
    p.render()
    n_restored = _foreground_px(p)

    assert n_hidden < n_all * 0.85, (n_all, n_hidden)
    assert n_restored >= n_all * 0.95, (n_all, n_restored)


def test_manual_hide_survives_dim_restore_on_screen(rendered_grid):
    """Layered model on real pixels: a manual hide stays painted-out after
    the dim filter is cleared back to all-active (show_all-style reveal of
    the dim layer must not un-hide manually-hidden cells)."""
    grid, p = rendered_grid
    ev = ElementVisibility(grid)
    n_all = _foreground_px(p)

    ev.hide([0, 1, 2, 3])                # manual hide of 4 cells
    apply_dim_filter(ev, CELL_DIM, active=[3], all_dims=[1, 3])
    apply_dim_filter(ev, CELL_DIM, active=[1, 3], all_dims=[1, 3])  # restore dims
    p.render()
    n_after = _foreground_px(p)

    # Dims all active again, yet 4 cells stay hidden by the manual layer.
    assert 0 < n_after < n_all * 0.95, (n_all, n_after)
