"""DeformedShapeDiagram — attach, step warp, scale slider.

Verifies the warp math: ``deformed_points == base_points +
scale * displacement_at_step``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DeformedShapeDiagram,
    DeformedShapeStyle,
    DiagramSpec,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def results_with_displacement_xyz(g, tmp_path: Path):
    """Native HDF5 with displacement_x/y/z per node + step.

    Per-axis values:
      disp_x = nid + t * 1.0
      disp_y = nid + t * 2.0
      disp_z = nid + t * 3.0
    """
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 3
    base = np.broadcast_to(
        node_ids.astype(np.float64), (n_steps, n_nodes),
    )
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    dx = base + t * 1.0
    dy = base + t * 2.0
    dz = base + t * 3.0

    path = tmp_path / "disp_xyz.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components={
                "displacement_x": dx,
                "displacement_y": dy,
                "displacement_z": dz,
            },
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# Construction
# =====================================================================

def _make_spec(scale=1.0, show_undeformed=True) -> DiagramSpec:
    return DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),    # display label
        style=DeformedShapeStyle(
            scale=scale, show_undeformed=show_undeformed,
        ),
    )


def test_construction_requires_deformed_style(results_with_displacement_xyz):
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad_spec = DiagramSpec(
        kind="deformed_shape",
        selector=SlabSelector(component="displacement_x"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="DeformedShapeStyle"):
        DeformedShapeDiagram(bad_spec, results_with_displacement_xyz)


# =====================================================================
# Attach + warp math
# =====================================================================

def test_attach_unrestricted_warps_all_points(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(scale=1.0), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )

    deformed = diagram._deformed_grid
    assert deformed is not None
    base = diagram._base_points

    # Step 0: disp_x = nid + 0 = nid for each node
    fem_ids = scene.node_ids[
        np.asarray(deformed.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]
    expected_disp_x = fem_ids.astype(np.float64)
    expected_disp_y = fem_ids.astype(np.float64)
    expected_disp_z = fem_ids.astype(np.float64)

    actual = np.asarray(deformed.points)
    expected = base + np.column_stack(
        [expected_disp_x, expected_disp_y, expected_disp_z]
    )
    np.testing.assert_allclose(actual, expected)


def test_step_change_remixes_displacement(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(scale=1.0), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    base = diagram._base_points
    fem_ids = scene.node_ids[
        np.asarray(diagram._deformed_grid.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]

    for t in (1, 2, 0):
        diagram.update_to_step(t)
        actual = np.asarray(diagram._deformed_grid.points)
        # disp_x = nid + t*1, disp_y = nid + t*2, disp_z = nid + t*3
        expected = base + np.column_stack([
            fem_ids + t * 1.0,
            fem_ids + t * 2.0,
            fem_ids + t * 3.0,
        ])
        np.testing.assert_allclose(actual, expected)


# =====================================================================
# Scale slider
# =====================================================================

def test_set_scale_re_warps_in_place(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(scale=1.0), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    base = diagram._base_points
    fem_ids = scene.node_ids[
        np.asarray(diagram._deformed_grid.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ]

    # Move to step 2 first, then change scale
    diagram.update_to_step(2)
    diagram.set_scale(10.0)

    actual = np.asarray(diagram._deformed_grid.points)
    expected = base + 10.0 * np.column_stack([
        fem_ids + 2.0,
        fem_ids + 4.0,
        fem_ids + 6.0,
    ])
    np.testing.assert_allclose(actual, expected)


def test_current_scale_uses_runtime_override(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(scale=2.0), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )

    assert diagram.current_scale() == 2.0
    diagram.set_scale(5.0)
    assert diagram.current_scale() == 5.0


def test_zero_scale_yields_undeformed(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(scale=0.0), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    np.testing.assert_allclose(
        np.asarray(diagram._deformed_grid.points),
        diagram._base_points,
    )


# =====================================================================
# Undeformed reference toggle
# =====================================================================

def test_undeformed_actor_present_by_default(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(show_undeformed=True), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    assert diagram._undeformed_actor is not None
    assert len(diagram._actors) == 2


def test_undeformed_actor_absent_when_disabled(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(show_undeformed=False), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    assert diagram._undeformed_actor is None
    assert len(diagram._actors) == 1


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(
    results_with_displacement_xyz, headless_plotter,
):
    scene = build_fem_scene(results_with_displacement_xyz.fem)
    diagram = DeformedShapeDiagram(
        _make_spec(), results_with_displacement_xyz,
    )
    diagram.attach(
        headless_plotter, results_with_displacement_xyz.fem, scene,
    )
    diagram.detach()
    assert diagram._deformed_grid is None
    assert diagram._base_points is None
    assert not diagram.is_attached
