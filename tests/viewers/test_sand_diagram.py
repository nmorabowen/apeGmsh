"""SandDiagram — grain cloud inside solid volumes.

Drives the diagram against an off-screen plotter on a small tet-meshed
cube with synthetic displacement steps: registration, seeding
(count / containment / determinism), interpolation correctness
(partition of unity), per-step updates, deform-follow, the
value-weighted density mask, and the no-solids fail-loud path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.backends import PyVistaQtBackend
from apeGmsh.viewers.diagrams import (
    DiagramSpec, NoDataError, SandDiagram, SandStyle, SlabSelector,
)
from apeGmsh.viewers.diagrams._kinds import kind_def
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


N_STEPS = 5
STEP_INCREMENT = 0.1     # displacement_x grows by this per step, per node


@pytest.fixture
def sand_results(g, tmp_path: Path):
    """Tet-meshed cube with displacement_x/y/z across 5 steps."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    base = np.broadcast_to(
        node_ids.astype(np.float64), (N_STEPS, len(node_ids)),
    )
    t = np.arange(N_STEPS, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_x": base + t * STEP_INCREMENT,
        "displacement_y": base + t * 0.2,
        "displacement_z": base + t * 0.3,
    }
    path = tmp_path / "sand.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(N_STEPS, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids, components=components,
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _make_diagram(results, style: SandStyle):
    spec = DiagramSpec(
        kind="sand",
        selector=SlabSelector(component="displacement_x"),
        style=style,
    )
    return SandDiagram(spec, results)


def _attach(results, style: SandStyle):
    plotter = pv.Plotter(off_screen=True)
    plotter.window_size = (320, 240)
    scene = build_fem_scene(results.fem)
    diagram = _make_diagram(results, style)
    diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    return plotter, scene, diagram


# =====================================================================
# Registration
# =====================================================================

def test_kind_registered():
    entry = kind_def("sand")
    assert entry is not None
    assert entry.label == "Sand volume plot"
    assert entry.diagram_class is SandDiagram
    assert entry.style_class is SandStyle
    assert entry.data_topology == "nodes"
    assert entry.in_catalog


def test_occludes_substrate():
    # Grains are strictly interior — an opaque substrate fill would
    # hide the whole diagram, so the viewer must drop the fill.
    assert SandDiagram.occludes_substrate is True


def test_wrong_style_type_raises(sand_results):
    from apeGmsh.viewers.diagrams import ContourStyle
    spec = DiagramSpec(
        kind="sand",
        selector=SlabSelector(component="displacement_x"),
        style=ContourStyle(),
    )
    with pytest.raises(TypeError, match="SandStyle"):
        SandDiagram(spec, sand_results)


# =====================================================================
# Seeding
# =====================================================================

def test_grain_count_matches_target(sand_results):
    plotter, _scene, diagram = _attach(
        sand_results, SandStyle(target_points=500, seed=1),
    )
    try:
        assert diagram._points is not None
        assert diagram._points.n_points == 500
        assert diagram._values is not None
        assert diagram._values.size == 500
    finally:
        diagram.detach()
        plotter.close()


def test_grains_inside_model_bounds(sand_results):
    plotter, scene, diagram = _attach(
        sand_results, SandStyle(target_points=400, seed=2),
    )
    try:
        pts = np.asarray(diagram._points.coords, dtype=np.float64)
        lo = np.asarray(scene.grid.points).min(axis=0) - 1e-6
        hi = np.asarray(scene.grid.points).max(axis=0) + 1e-6
        assert (pts >= lo).all() and (pts <= hi).all()
    finally:
        diagram.detach()
        plotter.close()


def test_seed_reproducible(sand_results):
    style = SandStyle(target_points=300, seed=42)
    p1, _s1, d1 = _attach(sand_results, style)
    p2, _s2, d2 = _attach(sand_results, style)
    try:
        np.testing.assert_array_equal(d1._points.coords, d2._points.coords)
        np.testing.assert_array_equal(d1._values, d2._values)
    finally:
        d1.detach()
        p1.close()
        d2.detach()
        p2.close()


# =====================================================================
# Interpolation
# =====================================================================

def test_values_bounded_by_nodal_range(sand_results):
    """Linear-tet interpolation is convex — grain values must lie
    within the nodal min/max of the component at step 0."""
    plotter, _scene, diagram = _attach(
        sand_results, SandStyle(target_points=400, seed=3),
    )
    try:
        node_ids = np.asarray(sand_results.fem.nodes.ids, dtype=np.float64)
        lo, hi = node_ids.min(), node_ids.max()   # step-0 field == node id
        vals = np.asarray(diagram._values)
        assert (vals >= lo - 1e-9).all() and (vals <= hi + 1e-9).all()
    finally:
        diagram.detach()
        plotter.close()


def test_update_to_step_shifts_by_partition_of_unity(sand_results):
    """The synthetic field adds a CONSTANT per step, so every grain's
    interpolated value must shift by exactly that constant — this
    pins both the partition of unity of the weights and the per-step
    read alignment."""
    plotter, _scene, diagram = _attach(
        sand_results, SandStyle(target_points=400, seed=4),
    )
    try:
        v0 = np.asarray(diagram._values).copy()
        diagram.update_to_step(4)
        v4 = np.asarray(diagram._values)
        np.testing.assert_allclose(
            v4 - v0, np.full_like(v0, 4 * STEP_INCREMENT), atol=1e-9,
        )
        # The rendered layer carries the new values.
        field = diagram._layer.field_named("displacement_x")
        np.testing.assert_allclose(field.values, v4)
    finally:
        diagram.detach()
        plotter.close()


# =====================================================================
# Deform-follow
# =====================================================================

def test_sync_substrate_points_translates_grains(sand_results):
    plotter, scene, diagram = _attach(
        sand_results, SandStyle(target_points=300, seed=5),
    )
    try:
        before = np.asarray(diagram._points.coords, dtype=np.float64).copy()
        shifted = np.asarray(scene.grid.points, dtype=np.float64) + [1.0, 0.0, 0.0]
        diagram.sync_substrate_points(shifted, scene)
        after = np.asarray(diagram._points.coords, dtype=np.float64)
        np.testing.assert_allclose(
            after - before, np.tile([1.0, 0.0, 0.0], (before.shape[0], 1)),
            atol=1e-5,   # PointSet pins float32
        )
        # Reset (None) returns to the reference configuration.
        diagram.sync_substrate_points(None, scene)
        np.testing.assert_allclose(
            np.asarray(diagram._points.coords), before, atol=1e-5,
        )
    finally:
        diagram.detach()
        plotter.close()


# =====================================================================
# Value-weighted density
# =====================================================================

def test_weight_by_value_hides_some_grains(sand_results):
    plotter, _scene, diagram = _attach(
        sand_results,
        SandStyle(
            target_points=400, seed=6,
            weight_by_value=True, density_floor=0.0,
        ),
    )
    try:
        hidden = diagram._layer.visibility.hidden_cells
        n = diagram._points.n_points
        # Field values span the node-id range, so normalized |value|
        # is < 1 for most grains — some must hide, but not all.
        assert 0 < len(hidden) < n
    finally:
        diagram.detach()
        plotter.close()


def test_set_point_size_live(sand_results):
    """Runtime grain-size override re-emits the layer AND reaches the
    actor property (the in-place backend path must push it — the
    dataset carries no point size)."""
    plotter, _scene, diagram = _attach(
        sand_results, SandStyle(target_points=200, seed=8, point_size=5.0),
    )
    try:
        assert diagram._layer.point_size == pytest.approx(5.0)
        diagram.set_point_size(12.0)
        assert diagram.current_point_size() == pytest.approx(12.0)
        assert diagram._layer.point_size == pytest.approx(12.0)
        assert float(diagram._handle.actor.prop.point_size) == pytest.approx(12.0)
    finally:
        diagram.detach()
        plotter.close()


def test_uniform_mode_hides_nothing(sand_results):
    plotter, _scene, diagram = _attach(
        sand_results, SandStyle(target_points=200, seed=7),
    )
    try:
        assert diagram._layer.visibility.hidden_cells == frozenset()
    finally:
        diagram.detach()
        plotter.close()


# =====================================================================
# Fail-loud paths
# =====================================================================

def test_no_solid_elements_raises(g, tmp_path: Path):
    """A shell-only model has nothing to fill — NoDataError, not a
    silent empty layer."""
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="plate")
    g.physical.add_surface("plate", name="Plate")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    path = tmp_path / "shell.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
    results = Results.from_native(path, model=_open_model_from_h5(path))

    plotter = pv.Plotter(off_screen=True)
    scene = build_fem_scene(results.fem)
    diagram = _make_diagram(results, SandStyle(target_points=100))
    try:
        with pytest.raises(NoDataError, match="3-D solid"):
            diagram.attach(PyVistaQtBackend(plotter), results.fem, scene)
    finally:
        plotter.close()


def test_missing_component_raises(sand_results):
    plotter = pv.Plotter(off_screen=True)
    scene = build_fem_scene(sand_results.fem)
    spec = DiagramSpec(
        kind="sand",
        selector=SlabSelector(component="no_such_component"),
        style=SandStyle(target_points=100),
    )
    diagram = SandDiagram(spec, sand_results)
    try:
        with pytest.raises(NoDataError, match="no_such_component"):
            diagram.attach(PyVistaQtBackend(plotter), sand_results.fem, scene)
    finally:
        plotter.close()
