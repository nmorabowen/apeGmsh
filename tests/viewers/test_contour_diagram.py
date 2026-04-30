"""ContourDiagram — attach, step update, runtime style adjustments.

Uses a real Results built via NativeWriter + a headless pyvista
Plotter (``off_screen=True``) so we can exercise the full attach
+ update_to_step + render path without a Qt window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    ContourDiagram,
    ContourStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def results_with_known_disp(g, tmp_path: Path):
    """Native HDF5 with predictable displacement_z per node + step.

    For step ``t`` and node ID ``nid``: ``disp_z = nid + t * 1000``.
    Lets per-step assertions check that the right values land on the
    submesh in the right positions.
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 4
    values = np.zeros((n_steps, n_nodes), dtype=np.float64)
    for t in range(n_steps):
        values[t] = node_ids + t * 1000.0

    path = tmp_path / "known_disp.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={"displacement_z": values},
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    """Off-screen pyvista plotter for attach/update tests."""
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# Construction
# =====================================================================

def _make_spec(component="displacement_z", pg=None) -> DiagramSpec:
    sel = SlabSelector(
        component=component,
        pg=(pg,) if pg else None,
    )
    return DiagramSpec(
        kind="contour",
        selector=sel,
        style=ContourStyle(),
    )


def test_construction_requires_contour_style(results_with_known_disp):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad_spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="ContourStyle"):
        ContourDiagram(bad_spec, results_with_known_disp)


def test_construction_rejects_wrong_kind(results_with_known_disp):
    spec = DiagramSpec(
        kind="not_contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    with pytest.raises(ValueError, match="kind="):
        ContourDiagram(spec, results_with_known_disp)


# =====================================================================
# Attach
# =====================================================================

def test_attach_requires_scene(results_with_known_disp, headless_plotter):
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, results_with_known_disp.fem)


def test_attach_unrestricted_paints_all_nodes(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    submesh = diagram._submesh
    assert submesh is not None
    assert submesh.n_points == scene.grid.n_points
    # Step 0 values: node_id + 0*1000 == node_id
    scalars = np.asarray(submesh.point_data["_contour"])
    fem_ids_in_submesh = scene.node_ids[
        np.asarray(submesh.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]
    np.testing.assert_array_equal(scalars, fem_ids_in_submesh.astype(np.float64))


def test_attach_with_pg_paints_only_pg_nodes(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(
        _make_spec(pg="Body"), results_with_known_disp,
    )
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    # All nodes are in Body for this single-volume fixture, so submesh
    # contains every node — but the resolved IDs go through PG resolution.
    assert diagram._submesh is not None
    assert diagram._submesh.n_points > 0


def test_attach_initial_clim_auto_fits(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    clim = diagram.current_clim()
    assert clim is not None
    lo, hi = clim
    # Step 0 values are the node IDs themselves; clim should bracket them
    assert lo <= np.asarray(scene.node_ids).min()
    assert hi >= np.asarray(scene.node_ids).max()


def test_attach_explicit_clim_used_verbatim(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(clim=(-5.0, 5.0)),
    )
    diagram = ContourDiagram(spec, results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    assert diagram.current_clim() == (-5.0, 5.0)


# =====================================================================
# Step update — values land in the right places
# =====================================================================

def test_update_to_step_scatters_correctly(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    fem_ids_in_submesh = scene.node_ids[
        np.asarray(diagram._submesh.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]

    for step in (1, 2, 3, 0):    # also test going back to 0
        diagram.update_to_step(step)
        scalars = np.asarray(diagram._submesh.point_data["_contour"])
        expected = fem_ids_in_submesh.astype(np.float64) + step * 1000.0
        np.testing.assert_array_equal(scalars, expected)


# =====================================================================
# Runtime style adjustments
# =====================================================================

def test_set_clim_updates_state(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_clim(0.0, 100.0)
    assert diagram.current_clim() == (0.0, 100.0)


def test_set_clim_collapses_equal_bounds(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_clim(5.0, 5.0)
    lo, hi = diagram.current_clim()
    assert lo == 5.0
    assert hi > lo


def test_autofit_clim_at_current_step(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.update_to_step(2)    # values are node_id + 2000
    fitted = diagram.autofit_clim_at_current_step()
    assert fitted is not None
    lo, hi = fitted
    expected_min = float(np.asarray(scene.node_ids).min()) + 2000.0
    expected_max = float(np.asarray(scene.node_ids).max()) + 2000.0
    assert lo == expected_min
    assert hi == expected_max


def test_set_opacity_records_runtime_value(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_opacity(0.5)
    assert diagram._runtime_opacity == 0.5


def test_set_cmap_records_runtime_value(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_cmap("plasma")
    assert diagram._runtime_cmap == "plasma"


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    assert diagram.is_attached
    assert diagram._submesh is not None

    diagram.detach()
    assert not diagram.is_attached
    assert diagram._submesh is None
    assert diagram._scalar_array is None


def test_detach_then_reattach_works(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    diagram.detach()
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    assert diagram.is_attached
    assert diagram._submesh is not None
