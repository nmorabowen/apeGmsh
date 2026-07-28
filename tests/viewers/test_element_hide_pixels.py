"""Per-element hide must change the picture, not just the array.

``ElementVisibility`` wrote ghost bit 0x01 believing it was
``HIDDENCELL`` for its whole life. VTK's ``HIDDENCELL`` is 0x20; 0x01 is
``DUPLICATECELL``, which the mapper renders. So manual hide / isolate,
the 0/1/2/3/4 dim filter and stage-activation masks all set a bit
nothing read, and the model never changed on screen.

Every test the feature had asserted the ghost *array*, so they all
passed against a feature that did nothing — the same vacuously-green
shape ADR 0081 and ADR 0083 keep running into. These assert pixels.
"""
from __future__ import annotations

import numpy as np
import pytest


_PAINTED = 12


@pytest.fixture
def hide_rig():
    """Offscreen plotter + a hex grid rendered through a real mapper."""
    import pyvista as pv

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(240, 240))
    except Exception:                                   # pragma: no cover
        pytest.skip("no offscreen render context")
    plotter.background_color = "black"
    grid = pv.ImageData(dimensions=(9, 9, 9)).cast_to_unstructured_grid()
    plotter.add_mesh(grid, color="white")
    plotter.camera_position = "xy"
    plotter.reset_camera()
    yield plotter, grid
    plotter.close()


def _painted(plotter) -> int:
    # Plotter.render() is lazy — it would silently measure nothing.
    plotter.render_window.Render()
    img = np.asarray(
        plotter.screenshot(return_img=True, transparent_background=False),
    )
    return int((img.astype(np.int32).sum(axis=2) > _PAINTED).sum())


def test_the_hide_bit_is_vtks_hide_bit():
    """The constant must be what VTK calls HIDDENCELL, not a guess."""
    from vtkmodules.vtkCommonDataModel import vtkDataSetAttributes

    from apeGmsh.viewers.core.element_visibility import HIDDENCELL

    assert HIDDENCELL == vtkDataSetAttributes.HIDDENCELL
    assert HIDDENCELL != vtkDataSetAttributes.DUPLICATECELL


def test_hiding_cells_changes_the_render(hide_rig):
    """The regression that matters: hide() must remove pixels.

    Cells are chosen by centre-x so the hidden half faces the camera —
    hiding the *back* half of a solid block changes no silhouette and
    would pass no matter what bit was written.
    """
    from apeGmsh.viewers.core.element_visibility import ElementVisibility

    plotter, grid = hide_rig
    before = _painted(plotter)
    assert before > 0, "the block must render before anything is hidden"

    centres = np.asarray(grid.cell_centers().points)[:, 0]
    left = np.flatnonzero(centres < np.median(centres))

    ev = ElementVisibility(grid)
    ev.hide(left)
    after = _painted(plotter)

    assert after < before * 0.9, (
        f"hide() left the picture unchanged ({before} -> {after} px) — "
        f"the ghost bit is not the one the mapper reads"
    )

    ev.show_all()
    assert _painted(plotter) == pytest.approx(before, rel=0.01), (
        "show_all() must put the geometry back"
    )


def test_hidden_cells_are_hidden_for_the_box_pick_too(hide_rig):
    """Box-pick reads the same bit the renderer does.

    These were separate literals (0x01 in the pick path, 0x20 in the
    backend), so a cell could be invisible and still selectable.
    """
    from apeGmsh.viewers.core.element_visibility import HIDDENCELL

    _plotter, grid = hide_rig
    from apeGmsh.viewers.core.element_visibility import ElementVisibility

    ev = ElementVisibility(grid)
    ev.hide([0, 1, 2])

    ghosts = np.asarray(grid.cell_data["vtkGhostType"])
    hidden = (ghosts & HIDDENCELL).astype(bool)
    assert hidden[:3].all() and not hidden[3:].any()
    # And the byte is *pure* 0x20 — the backend's render-verified note
    # says even 0x21 fails to hide 0-D vertex cells.
    assert set(np.unique(ghosts).tolist()) <= {0, HIDDENCELL}
