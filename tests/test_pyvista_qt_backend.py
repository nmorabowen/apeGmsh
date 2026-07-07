"""Headless tests for PyVistaQtBackend's pure translation layer.

Covers the data-only path (ADR 0042, Phase R-A): IR -> pyvista grid,
RGB array, visibility bitmask, and Protocol conformance.  These build
``pyvista.UnstructuredGrid`` objects and mutate their arrays without an
OpenGL context, so they run on the GPU-less CI / sandbox.  The
plotter-driving methods (``add_layer`` and friends) need a render
context and are verified on the desktop viewer instead.
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.backends import (  # noqa: E402
    PyVistaQtBackend,
    apply_visibility_mask,
    mesh_layer_to_grid,
)
from apeGmsh.viewers.backends.pyvista_qt import TOKEN_TO_VTK  # noqa: E402
from apeGmsh.viewers.scene_ir import (  # noqa: E402
    CellBlocks,
    ColorSpec,
    MeshLayer,
    PointSet,
    RenderBackend,
    ScalarField,
    VisibilityMask,
)


def _two_block_layer(**kw) -> MeshLayer:
    # 4 points: one triangle (0,1,2) + one tetra (0,1,2,3)
    pts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    return MeshLayer(
        layer_id="m",
        points=PointSet(pts),
        cells=CellBlocks(
            {
                "triangle": np.array([[0, 1, 2]]),
                "tetra": np.array([[0, 1, 2, 3]]),
            }
        ),
        **kw,
    )


def test_grid_has_expected_topology() -> None:
    grid = mesh_layer_to_grid(_two_block_layer())
    assert grid.n_points == 4
    assert grid.n_cells == 2


def test_unknown_token_fails_loud() -> None:
    layer = MeshLayer(
        layer_id="bad",
        points=PointSet(np.zeros((3, 3))),
        cells=CellBlocks({"banana": np.array([[0, 1, 2]])}),
    )
    with pytest.raises(ValueError, match="unknown cell token"):
        mesh_layer_to_grid(layer)


def test_scalar_field_attaches_by_location() -> None:
    layer = _two_block_layer(
        fields=(
            ScalarField("q", np.array([1.0, 2.0]), location="cell"),
            ScalarField("u", np.arange(4.0), location="point"),
        )
    )
    grid = mesh_layer_to_grid(layer)
    assert "q" in grid.cell_data and grid.cell_data["q"].shape == (2,)
    assert "u" in grid.point_data and grid.point_data["u"].shape == (4,)


def test_per_entity_rgb_becomes_uint8_colors() -> None:
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # float [0,1], 2 cells
    layer = _two_block_layer(
        color=ColorSpec(mode="per_entity_rgb", entity_rgb=rgb)
    )
    grid = mesh_layer_to_grid(layer)
    colors = grid.cell_data["colors"]
    assert colors.dtype == np.uint8
    assert colors.shape == (2, 3)
    assert colors[0, 0] == 255 and colors[1, 1] == 255


def test_visibility_mask_sets_ghost_bit() -> None:
    grid = mesh_layer_to_grid(_two_block_layer())
    apply_visibility_mask(grid, VisibilityMask(hidden_cells=frozenset({1})))
    ghost = grid.cell_data["vtkGhostType"]
    assert ghost[0] == 0
    # Must be the pure HIDDENCELL byte (0x20): DUPLICATECELL (0x01)
    # leaves vertex cells visible, and even 0x21 fails for them — see
    # _GHOST_HIDDEN_CELL in backends/pyvista_qt.py.
    assert ghost[1] == 0x20


def test_visibility_mask_ignores_out_of_range() -> None:
    grid = mesh_layer_to_grid(_two_block_layer())
    # cell 99 doesn't exist — must not raise, must not set anything
    apply_visibility_mask(grid, VisibilityMask(hidden_cells=frozenset({99})))
    assert grid.cell_data["vtkGhostType"].sum() == 0


def test_backend_satisfies_protocol() -> None:
    backend = PyVistaQtBackend(plotter=None)
    assert isinstance(backend, RenderBackend)
    assert backend.supports_picking() is True


def test_token_map_covers_canonical_tokens() -> None:
    assert set(TOKEN_TO_VTK) >= {
        "vertex", "line", "triangle", "quad",
        "tetra", "hexahedron", "wedge", "pyramid",
    }


# --- Render-path integration (needs an offscreen GL context) -------------


@pytest.fixture
def offscreen():
    try:
        p = pv.Plotter(off_screen=True)
    except Exception:  # pragma: no cover - depends on GL availability
        pytest.skip("no offscreen render context available")
    yield p
    p.close()


def test_backend_add_and_remove_mesh_layer(offscreen) -> None:
    from apeGmsh.viewers.scene_ir import VisibilityMask

    backend = PyVistaQtBackend(offscreen)
    handle = backend.add_layer(_two_block_layer())
    assert handle.actor is not None
    backend.set_visibility(handle, VisibilityMask(hidden_cells=frozenset({0})))
    backend.remove_layer(handle)
    assert handle.actor is None


def test_backend_add_glyph_layer(offscreen) -> None:
    from apeGmsh.viewers.scene_ir import GlyphLayer

    backend = PyVistaQtBackend(offscreen)
    layer = GlyphLayer(
        layer_id="g",
        positions=PointSet(np.array([[0, 0, 0], [1, 0, 0]], dtype=float)),
        kind="arrow",
        orientations=np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
        scales=np.array([1.0, 2.0]),
    )
    handle = backend.add_layer(layer)
    assert handle.actor is not None
    backend.set_layer_visible(handle, False)
    backend.remove_layer(handle)
    assert handle.actor is None


def test_glyph_update_preserves_camera(offscreen) -> None:
    """A per-step glyph rebuild must NOT reframe the model window.

    Glyph layers have no in-place fast path, so ``update_layer`` removes
    and re-adds the actor every animation step.  ``add_mesh`` would reset
    the camera to refit the (differently-scaled) glyph bounds, making the
    whole window appear to zoom as the user scrubs time.  Regression for
    that rescale.
    """
    from apeGmsh.viewers.scene_ir import GlyphLayer

    def _layer(scale: float) -> GlyphLayer:
        return GlyphLayer(
            layer_id="g",
            positions=PointSet(
                np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
            ),
            kind="arrow",
            orientations=np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
            scales=np.array([scale, scale]),
        )

    backend = PyVistaQtBackend(offscreen)
    handle = backend.add_layer(_layer(1.0))
    offscreen.reset_camera()
    before = offscreen.camera_position

    # Next step: glyphs ten times larger -> very different bounds.
    backend.update_layer(handle, _layer(10.0))
    after = offscreen.camera_position

    np.testing.assert_allclose(
        np.array(after.to_list()), np.array(before.to_list())
    )
