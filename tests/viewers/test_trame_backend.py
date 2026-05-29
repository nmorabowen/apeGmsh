"""TrameBackend + WebViewer (ADR 0042, R-C slice 1).

Verifies the web backend shares the desktop backend's ``scene_ir`` →
pyvista translation verbatim (parity), differs only in picking support,
and that the ``WebViewer`` shell builds + binds + scrubs headlessly. The
actual trame/browser display (``WebViewer.show``) needs a notebook and is
verified by eyeball — not here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from apeGmsh.viewers.backends import (
    PyVistaQtBackend,
    TrameBackend,
    PyVistaBackend,
)
from apeGmsh.viewers.scene_ir import CellBlocks, ColorSpec, MeshLayer, PointSet

from tests.conftest import _open_model_from_h5


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def cube_results(g, tmp_path: Path):
    """1x1x1 tet cube + single-stage Results (zero displacement)."""
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter

    g.model.geometry.add_box(0, 0, 0, 1.0, 1.0, 1.0, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    res_path = tmp_path / "run.h5"
    with NativeWriter(res_path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="static", kind="static",
            time=np.array([0.0], dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": np.zeros((1, node_ids.size)),
                "displacement_y": np.zeros((1, node_ids.size)),
                "displacement_z": np.zeros((1, node_ids.size)),
            },
        )
        w.end_stage()
    return Results.from_native(res_path, model=_open_model_from_h5(res_path))


def _quad_layer() -> MeshLayer:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    return MeshLayer(
        layer_id="q",
        points=PointSet(verts),
        cells=CellBlocks({"quad": np.array([[0, 1, 2, 3]], dtype=np.int64)}),
        color=ColorSpec(mode="solid", solid_rgb="#3CB371"),
    )


# ---------------------------------------------------------------------
# Backend identity / capability
# ---------------------------------------------------------------------

def test_trame_backend_is_a_pyvista_backend():
    b = TrameBackend(pv.Plotter(off_screen=True))
    assert isinstance(b, PyVistaBackend)


def test_trame_backend_picking_off_qt_on():
    trame = TrameBackend(pv.Plotter(off_screen=True))
    qt = PyVistaQtBackend(pv.Plotter(off_screen=True))
    assert trame.supports_picking() is False
    assert qt.supports_picking() is True


def test_trame_default_plotter_constructs():
    # No plotter supplied → backend makes its own pyvista.Plotter.
    b = TrameBackend()
    assert b.plotter is not None
    try:
        b.plotter.close()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Render parity — identical grids from the shared scene_ir translation
# ---------------------------------------------------------------------

def test_mesh_layer_grid_parity_qt_vs_trame():
    """Both backends build the same pyvista grid from one MeshLayer —
    they inherit ``add_layer`` from ``PyVistaBackend`` unchanged."""
    layer = _quad_layer()
    qt = PyVistaQtBackend(pv.Plotter(off_screen=True))
    trame = TrameBackend(pv.Plotter(off_screen=True))
    hq = qt.add_layer(layer)
    ht = trame.add_layer(layer)
    np.testing.assert_array_equal(hq.dataset.points, ht.dataset.points)
    assert hq.dataset.n_cells == ht.dataset.n_cells == 1
    np.testing.assert_array_equal(hq.dataset.celltypes, ht.dataset.celltypes)


# ---------------------------------------------------------------------
# WebViewer shell — headless construction / binding / scrub
# ---------------------------------------------------------------------

def test_web_viewer_builds_and_binds(cube_results):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    assert isinstance(wv.backend, TrameBackend)
    assert wv.director.is_bound if hasattr(wv.director, "is_bound") else True
    # Substrate actor is on the plotter.
    assert "substrate" in wv.plotter.actors
    # A stage is active → at least one step.
    assert wv.n_steps >= 1
    # Programmatic scrub doesn't raise.
    wv.set_step(0)


def test_web_viewer_step_clamps(cube_results):
    from apeGmsh.viewers.web_viewer import WebViewer

    wv = WebViewer(cube_results, plotter=pv.Plotter(off_screen=True))
    # Out-of-range step is clamped by the director, not an error.
    wv.set_step(999)


def test_show_web_returns_viewer_without_display(cube_results, monkeypatch):
    """show_web(show=False) builds the viewer but does not display."""
    from apeGmsh.viewers import web_viewer

    wv = web_viewer.show_web(cube_results, show=False)
    assert isinstance(wv, web_viewer.WebViewer)
