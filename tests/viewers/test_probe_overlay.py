"""ProbeOverlay — point + line + plane data path tests.

Exercises the programmatic probe API (``probe_at_point``,
``probe_along_line``, ``probe_with_plane``) against a real Results +
FEMSceneData. Interactive picking is skipped (needs a Qt event loop).
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
    ResultsDirector,
    SlabSelector,
)
from apeGmsh.viewers.overlays.probe_overlay import (
    LineProbeResult,
    PlaneProbeResult,
    PointProbeResult,
    ProbeOverlay,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixture: small results with one displacement component
# =====================================================================

@pytest.fixture
def results_for_probes(g, tmp_path: Path):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    n_nodes = len(fem.nodes.ids)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    n_steps = 3
    base = np.broadcast_to(node_ids.astype(np.float64), (n_steps, n_nodes))
    t = np.arange(n_steps, dtype=np.float64).reshape(-1, 1)
    components = {
        "displacement_z": base + t * 0.1,
    }

    path = tmp_path / "probes.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_nodes(
            sid, "partition_0",
            node_ids=node_ids,
            components=components,
        )
        w.end_stage()
    return Results.from_native(path)


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


@pytest.fixture
def overlay_with_diagram(results_for_probes, headless_plotter):
    """ProbeOverlay with one ContourDiagram already attached so probes
    have at least one component to read."""
    scene = build_fem_scene(results_for_probes.fem)
    director = ResultsDirector(results_for_probes)
    director.bind_plotter(headless_plotter, scene=scene)

    # Add a diagram so probes have a component to read
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(),
    )
    diagram = ContourDiagram(spec, results_for_probes)
    director.registry.add(diagram)

    overlay = ProbeOverlay(headless_plotter, scene, director)
    return overlay, scene, director, diagram


# =====================================================================
# probe_at_point
# =====================================================================

def test_probe_at_point_snaps_to_nearest_node(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    # Pick a point very close to the first FEM node
    target_idx = 0
    target_coord = np.asarray(scene.grid.points)[target_idx]
    pos = target_coord + np.array([0.001, 0.0, 0.0])

    result = overlay.probe_at_point(pos)
    assert isinstance(result, PointProbeResult)
    assert result.closest_node_id == int(scene.node_ids[target_idx])
    np.testing.assert_allclose(result.closest_coord, target_coord)
    assert result.distance < 0.01


def test_probe_at_point_returns_active_components(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    pos = np.asarray(scene.grid.points)[0]
    result = overlay.probe_at_point(pos)
    assert "displacement_z" in result.field_values


def test_probe_at_point_value_matches_directors_read(overlay_with_diagram):
    overlay, scene, director, _ = overlay_with_diagram
    target_idx = 5
    pos = np.asarray(scene.grid.points)[target_idx]
    result = overlay.probe_at_point(pos)

    # Cross-check via the Director directly
    expected = director.read_at_pick(
        result.closest_node_id, ["displacement_z"],
    )
    assert result.field_values["displacement_z"] == pytest.approx(
        expected["displacement_z"]
    )


def test_probe_at_point_uses_current_step(overlay_with_diagram):
    overlay, scene, director, _ = overlay_with_diagram
    target_idx = 2
    pos = np.asarray(scene.grid.points)[target_idx]

    director.set_step(0)
    r0 = overlay.probe_at_point(pos)
    director.set_step(2)
    r2 = overlay.probe_at_point(pos)

    assert r0.step_index == 0
    assert r2.step_index == 2
    # Step 2 = step 0 + 0.2  (per fixture)
    diff = r2.field_values["displacement_z"] - r0.field_values["displacement_z"]
    assert diff == pytest.approx(0.2, abs=1e-9)


def test_probe_at_point_appends_to_history(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    pos = np.asarray(scene.grid.points)[0]
    overlay.probe_at_point(pos)
    overlay.probe_at_point(pos)
    assert len(overlay.point_results) == 2


# =====================================================================
# probe_along_line
# =====================================================================

def test_probe_along_line_returns_n_samples(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    a = np.asarray(scene.grid.points)[0]
    b = np.asarray(scene.grid.points)[-1]
    result = overlay.probe_along_line(a, b, n_samples=10)
    assert isinstance(result, LineProbeResult)
    assert result.n_samples == 10
    assert result.positions.shape == (10, 3)
    assert result.closest_node_ids.shape == (10,)


def test_probe_along_line_arc_length_monotonic(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    result = overlay.probe_along_line(a, b, n_samples=5)
    diffs = np.diff(result.arc_length)
    assert np.all(diffs >= 0)
    assert result.arc_length[0] == 0.0
    assert result.total_length == pytest.approx(1.0)


def test_probe_along_line_includes_active_component(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    a = np.asarray(scene.grid.points)[0]
    b = np.asarray(scene.grid.points)[-1]
    result = overlay.probe_along_line(a, b, n_samples=5)
    assert "displacement_z" in result.field_values
    assert result.field_values["displacement_z"].shape == (5,)


def test_probe_along_line_clamps_n_samples(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    a = np.asarray(scene.grid.points)[0]
    b = np.asarray(scene.grid.points)[-1]
    result = overlay.probe_along_line(a, b, n_samples=1)
    # Implementation enforces minimum of 2
    assert result.n_samples == 2


# =====================================================================
# probe_with_plane
# =====================================================================

def test_probe_with_plane_returns_slice(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    result = overlay.probe_with_plane(normal="x")
    assert isinstance(result, PlaneProbeResult)
    assert result.n_points > 0
    assert result.slice_mesh is not None


def test_probe_with_plane_axis_normals(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    for axis, expected in [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ]:
        result = overlay.probe_with_plane(normal=axis)
        if result is not None:
            np.testing.assert_allclose(result.normal, expected)


def test_probe_with_plane_explicit_normal(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    # Diagonal slice
    result = overlay.probe_with_plane(normal=(1.0, 1.0, 0.0))
    assert result is not None
    # Normalized
    np.testing.assert_allclose(np.linalg.norm(result.normal), 1.0)


def test_probe_with_plane_unknown_axis_raises(overlay_with_diagram):
    overlay, _, _, _ = overlay_with_diagram
    with pytest.raises(ValueError, match="Unknown normal"):
        overlay.probe_with_plane(normal="diagonal")


def test_probe_with_plane_zero_normal_returns_none(overlay_with_diagram):
    overlay, _, _, _ = overlay_with_diagram
    assert overlay.probe_with_plane(normal=(0.0, 0.0, 0.0)) is None


# =====================================================================
# clear()
# =====================================================================

def test_clear_drops_all_history(overlay_with_diagram):
    overlay, scene, _, _ = overlay_with_diagram
    pos = np.asarray(scene.grid.points)[0]
    overlay.probe_at_point(pos)
    overlay.probe_along_line(pos, pos + 0.1, n_samples=5)
    overlay.probe_with_plane(normal="x")

    overlay.clear()

    assert len(overlay.point_results) == 0
    assert len(overlay.line_results) == 0
    assert len(overlay.plane_results) == 0
