"""Render-surface fast path — O(surface) animation steps, same picture.

``vtkDataSetMapper`` re-runs its O(volume) surface extraction on every
input MTime bump, even for a scalars-only update (measured 67 ms/step
at 1M tets vs 1.1 ms on a pre-extracted surface). The fast path renders
the extraction's output and scatters updates onto it. These tests pin
the four contracts that make that safe:

* **F-PICK** — a picked render-surface cell resolves to the correct
  VOLUME cell id before the ``PickHit`` crosses the seam (dropping the
  map silently picks the wrong element, it does not fail).
* **F-SLICE** — ``handle.dataset`` stays the volumetric grid: slicing
  it yields filled polygon cells, not the line loops a surface slice
  degrades to (the ADR 0083 S3 cut-face contract).
* **F-INPLACE** — an in-place scalar update changes rendered pixels
  with actor AND mapper-input identity preserved.
* **F-PARITY / F-VIS** — the surface renders pixel-identical to the
  volumetric mapper, including after a visibility-mask change (hidden
  faces dropped, interior faces revealed by re-extraction).
* **F-SUBSTRATE** — the off-seam substrate pair maps one shared
  surface; deform steps scatter onto it; ghost changes re-extract it.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
    VisibilityMask,
)


# =====================================================================
# Helpers
# =====================================================================


def _tet_grid(n_side: int = 5):
    """A uniform tet box (multiple boundary + interior cells)."""
    import pyvista as pv
    from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

    img = pv.ImageData(dimensions=(n_side, n_side, n_side))
    tri = vtkDataSetTriangleFilter()
    tri.SetInputData(img.cast_to_unstructured_grid())
    tri.Update()
    return pv.UnstructuredGrid(tri.GetOutput())



def _tet_layer(grid, layer_id: str = "tets", *, scale: float = 1.0,
               hidden=(), wireframe: bool = False) -> MeshLayer:
    pts = np.asarray(grid.points, dtype=np.float64)
    v = pts[:, 2].copy() * scale
    return MeshLayer(
        layer_id=layer_id,
        points=PointSet(pts),
        cells=CellBlocks({"tetra": grid.cells_dict[10]}),
        fields=(ScalarField(name="v", values=v, location="point"),),
        color=ColorSpec(
            mode="by_array", array_name="v",
            lut=LutSpec(name="jet", vmin=0.0, vmax=float(pts[:, 2].max())),
        ),
        visibility=VisibilityMask(hidden_cells=frozenset(hidden)),
        wireframe=wireframe,
    )


@pytest.fixture
def offscreen_plotter():
    import pyvista as pv

    try:
        plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    except Exception:                                   # pragma: no cover
        pytest.skip("no offscreen render context")
    plotter.background_color = "black"
    yield plotter
    plotter.close()


@pytest.fixture
def backend(offscreen_plotter):
    from apeGmsh.viewers.backends import PyVistaQtBackend

    return PyVistaQtBackend(offscreen_plotter)


def _pin_camera(plotter) -> None:
    camera = plotter.camera
    camera.parallel_projection = True
    camera.position = (10.0, 6.0, 14.0)
    camera.focal_point = (2.0, 2.0, 2.0)
    camera.up = (0.0, 1.0, 0.0)
    camera.parallel_scale = 4.0
    camera.clipping_range = (0.1, 100.0)


def _frame(plotter) -> np.ndarray:
    _pin_camera(plotter)
    plotter.render()
    return np.asarray(
        plotter.screenshot(return_img=True, transparent_background=False),
    ).copy()


def _volumetric_frame(grid, *, hidden=(), wireframe=False) -> np.ndarray:
    """Reference: the same grid through the plain volumetric mapper."""
    import pyvista as pv

    from apeGmsh.viewers.backends.pyvista_qt import apply_visibility_mask

    g = grid.copy()
    g.point_data["v"] = np.asarray(g.points[:, 2], dtype=np.float64)
    apply_visibility_mask(g, VisibilityMask(hidden_cells=frozenset(hidden)))
    plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    plotter.background_color = "black"
    kwargs = dict(
        scalars="v", cmap="jet",
        clim=(0.0, float(np.asarray(g.points)[:, 2].max())),
        show_scalar_bar=False,
    )
    if wireframe:
        kwargs["style"] = "wireframe"
    plotter.add_mesh(g, **kwargs)
    img = _frame(plotter)
    plotter.close()
    return img


# =====================================================================
# F-PICK — surface cell id -> volume cell id, inside the backend
# =====================================================================


class _FakePicker:
    """Duck-typed picker driving ``PyVistaPickBackend._resolve``."""

    def __init__(self, dataset, cell_id: int) -> None:
        self._dataset = dataset
        self._cell_id = cell_id

    def Pick(self, *_args) -> None:
        pass

    def GetViewProp(self):
        return object()

    def GetPickPosition(self):
        return (0.0, 0.0, 0.0)

    def GetCellId(self) -> int:
        return self._cell_id

    def GetDataSet(self):
        return self._dataset


class _StubPlotter:
    renderer = None


def test_picked_surface_cell_resolves_to_volume_cell_id(backend):
    """Constraint 2: the picker's surface-triangle id must cross the
    seam as the VOLUMETRIC cell id — indexing ``cell_dim`` /
    ``cell_to_element_id`` with a raw surface id resolves the WRONG
    element rather than failing."""
    from apeGmsh.viewers.backends._pyvista_pick import PyVistaPickBackend

    grid = _tet_grid()
    handle = backend.add_layer(_tet_layer(grid))
    assert handle.render_surface is not None, "fast path not taken"

    # The picker hands back ids in the render surface's cell space, and
    # the mapping below is what has to hold. Asserting WHICH object the
    # mapper reports as its input is not portable — see the in-place
    # test — so this test asserts the mapping, not the plumbing.

    cell_ids = np.asarray(handle.surf_cell_ids)
    # Pick a surface cell whose volume id differs from its own index —
    # the case where a dropped mapping is silently wrong.
    differing = np.nonzero(cell_ids != np.arange(cell_ids.size))[0]
    assert differing.size, "test mesh must interleave surface/volume ids"
    surf_id = int(differing[-1])

    pick = PyVistaPickBackend(_StubPlotter())
    hit = pick._resolve(_FakePicker(handle.render_surface, surf_id), 0, 0)
    assert hit is not None
    assert hit.cell_id == int(cell_ids[surf_id])
    assert hit.cell_id != surf_id
    assert hit.cell_id < grid.n_cells


def test_pick_passthrough_without_render_surface(backend):
    """Datasets not stamped by the fast path keep raw cell ids — the
    mesh viewer's own extracted surfaces already resolve in surface-id
    space and must not be double-mapped."""
    from apeGmsh.viewers.backends._pyvista_pick import PyVistaPickBackend

    grid = _tet_grid()
    pick = PyVistaPickBackend(_StubPlotter())
    hit = pick._resolve(_FakePicker(grid, 7), 0, 0)
    assert hit is not None and hit.cell_id == 7


# =====================================================================
# F-SLICE — handle.dataset stays volumetric
# =====================================================================


def test_handle_dataset_slices_to_filled_polygons(backend):
    """Constraint 1: ``slice_layer`` (the ADR 0083 S3 cut face) slices
    ``handle.dataset``, which must stay the VOLUMETRIC grid — slicing
    the render surface silently degrades the filled cap to line loops.
    """
    from apeGmsh.viewers.scene_ir import ClipPlaneSpec

    grid = _tet_grid()
    handle = backend.add_layer(_tet_layer(grid))
    assert handle.render_surface is not None

    cut = backend.slice_layer(
        handle,
        ClipPlaneSpec(origin=(2.0, 2.0, 2.0), normal=(1.0, 0.0, 0.0)),
        layer_id="cut",
    )
    assert cut is not None, "slice missed — dataset is not the volume"
    filled = sum(
        conn.shape[0]
        for token, conn in cut.cells.blocks.items()
        if token in ("triangle", "quad", "polygon")
    )
    lines = cut.cells.blocks.get("line")
    assert filled > 0, "cut face lost its filled polygons"
    assert lines is None or lines.shape[0] == 0, (
        "cut face degraded to lines — slice_layer sliced a surface"
    )
    # The cut carries the layer's scalars for the shared LUT.
    assert cut.field_named("v") is not None

    # And the render surface really would degrade — the trap is real.
    surf_cut = handle.render_surface.slice(
        normal="x", origin=(2.0, 2.0, 2.0),
    )
    assert surf_cut.n_faces_strict == 0


# =====================================================================
# F-INPLACE — scalar update: new pixels, same actor, same mapper input
# =====================================================================


def test_inplace_scalar_update_repaints_without_rebuilding(backend):
    grid = _tet_grid()
    handle = backend.add_layer(_tet_layer(grid))
    actor0 = handle.actor
    surface0 = handle.render_surface
    before = _frame(backend.plotter)

    backend.update_layer(handle, _tet_layer(grid, scale=0.25))
    after = _frame(backend.plotter)

    assert handle.actor is actor0, "in-place path rebuilt the actor"
    assert handle.render_surface is surface0
    # The load-bearing guarantee is that the mapper renders what the
    # fast path writes. Assert that as BEHAVIOUR, not as object
    # identity: `mapper.GetInput()` came back as a different C++ object
    # than `render_surface` on Linux CI while the render still updated,
    # so an identity check tested the plumbing and failed on a platform
    # difference rather than on a defect.
    assert not np.array_equal(before, after), "update did not repaint"

    # Same guarantee, stated directly: a write straight through
    # `render_surface` must reach the screen. If the mapper ever held a
    # detached copy, this is the assertion that would catch it.
    surface0.point_data["v"] = np.full(surface0.n_points, 99.0)
    surface0.Modified()
    assert not np.array_equal(after, _frame(backend.plotter)), (
        "a write through render_surface did not reach the mapper"
    )


# =====================================================================
# F-PARITY / F-VIS — pixel parity with the volumetric mapper
# =====================================================================


def test_render_surface_matches_volumetric_render(backend):
    grid = _tet_grid()
    backend.add_layer(_tet_layer(grid))
    assert np.array_equal(_frame(backend.plotter), _volumetric_frame(grid))


def test_wireframe_layer_matches_volumetric_render(
    backend, requires_gl_wireframe_exact,
):
    grid = _tet_grid()
    handle = backend.add_layer(_tet_layer(grid, wireframe=True))
    assert handle.render_surface is not None
    assert np.array_equal(
        _frame(backend.plotter), _volumetric_frame(grid, wireframe=True),
    )


def test_visibility_mask_change_matches_volumetric_render(backend):
    """Constraint 5: hiding cells through ``set_visibility`` must drop
    their faces AND reveal the interior exactly as the volumetric
    mapper does — scattering ghosts cannot do that; only the
    re-extraction path can."""
    grid = _tet_grid()
    hidden = tuple(range(0, grid.n_cells, 3))
    handle = backend.add_layer(_tet_layer(grid))
    surface0 = handle.render_surface
    actor0 = handle.actor

    backend.set_visibility(
        handle, VisibilityMask(hidden_cells=frozenset(hidden)),
    )
    after = _frame(backend.plotter)

    assert handle.actor is actor0, "visibility change must not re-add"
    assert handle.render_surface is not surface0, "mask change must re-extract"
    assert np.array_equal(after, _volumetric_frame(grid, hidden=hidden))

    # Same mask again: no re-extraction (cheap steady-state).
    surface1 = handle.render_surface
    backend.update_layer(
        handle,
        _tet_layer(grid, scale=0.5, hidden=hidden),
    )
    assert handle.render_surface is surface1


def test_update_layer_visibility_change_matches_volumetric(backend):
    """The same mask-change contract through ``update_layer``'s
    in-place path (the scrubber's route)."""
    grid = _tet_grid()
    hidden = tuple(range(0, grid.n_cells, 4))
    handle = backend.add_layer(_tet_layer(grid))
    backend.update_layer(handle, _tet_layer(grid, hidden=hidden))
    assert np.array_equal(
        _frame(backend.plotter), _volumetric_frame(grid, hidden=hidden),
    )


def test_line_layers_bypass_the_surface_path(backend):
    """1-D layers stay on the direct volumetric mapper (extraction is
    already O(n_cells) there; lines gain nothing)."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    layer = MeshLayer(
        layer_id="lines",
        points=PointSet(pts),
        cells=CellBlocks({"line": np.array([[0, 1], [1, 2]])}),
        color=ColorSpec(mode="solid", solid_rgb=(1.0, 0.0, 0.0)),
    )
    handle = backend.add_layer(layer)
    assert handle.render_surface is None
    assert handle.actor.GetMapper().GetInput() is handle.dataset


# =====================================================================
# F-SUBSTRATE — the off-seam pair
# =====================================================================


class _Palette:
    substrate_color = "#808080"
    substrate_edge_color = "#c0c0c0"


class _Prefs:
    mesh_surface_opacity = 1.0
    mesh_line_width = 1.0
    point_size = 6.0


class _Scene:
    def __init__(self, grid) -> None:
        self.grid = grid
        self.model_diagonal = 3.46


def test_substrate_pair_maps_one_shared_render_surface(offscreen_plotter):
    from apeGmsh.viewers.results_viewer import add_substrate_actors

    scene = _Scene(_tet_grid())
    fill, wf = add_substrate_actors(
        offscreen_plotter, scene, palette=_Palette(), prefs=_Prefs(),
    )
    rs = scene.render_surface
    assert rs is not None
    assert fill.GetMapper().GetInput() is rs.surface
    assert wf.GetMapper().GetInput() is rs.surface
    # The volumetric grid remains the untouched source of truth.
    assert scene.grid.n_cells > rs.surface.n_cells


def test_substrate_deform_scatter_follows_the_grid(offscreen_plotter):
    from apeGmsh.viewers.results_viewer import (
        add_substrate_actors, sync_render_surface_points,
    )

    scene = _Scene(_tet_grid())
    add_substrate_actors(
        offscreen_plotter, scene, palette=_Palette(), prefs=_Prefs(),
        reset_camera=True,
    )
    before = _frame(offscreen_plotter)

    deformed = np.asarray(scene.grid.points, dtype=np.float64) * 1.3
    scene.grid.points = deformed
    sync_render_surface_points(scene, deformed)
    after = _frame(offscreen_plotter)

    rs = scene.render_surface
    assert np.allclose(
        np.asarray(rs.surface.points), deformed[rs.point_ids],
    )
    assert not np.array_equal(before, after), "deform step did not repaint"


def test_substrate_ghost_change_reextracts_and_swaps(offscreen_plotter):
    from apeGmsh.viewers.backends.pyvista_qt import refresh_render_surface
    from apeGmsh.viewers.results_viewer import add_substrate_actors

    scene = _Scene(_tet_grid())
    fill, wf = add_substrate_actors(
        offscreen_plotter, scene, palette=_Palette(), prefs=_Prefs(),
    )
    rs = scene.render_surface
    surface0 = rs.surface

    # No ghost change -> memcmp short-circuit, no re-extraction.
    assert refresh_render_surface(scene.grid, rs, (fill, wf)) is False
    assert rs.surface is surface0

    # Hide a boundary cell with the true HIDDENCELL bit -> re-extract,
    # its faces drop, and both mappers swap to the new surface.
    hidden_cell = int(np.asarray(rs.cell_ids)[0])
    ghosts = np.zeros(scene.grid.n_cells, dtype=np.uint8)
    ghosts[hidden_cell] = 0x20
    scene.grid.cell_data["vtkGhostType"] = ghosts

    assert refresh_render_surface(scene.grid, rs, (fill, wf)) is True
    assert rs.surface is not surface0
    assert hidden_cell not in np.asarray(rs.cell_ids)
    assert fill.GetMapper().GetInput() is rs.surface
    assert wf.GetMapper().GetInput() is rs.surface
