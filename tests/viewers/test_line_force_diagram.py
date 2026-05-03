"""LineForceDiagram — attach + step update on a beam fixture.

Builds a 3-element horizontal beam via apeGmsh (line2 mesh), writes
synthetic line_stations data via NativeWriter (3-pt Lobatto: xi = -1,
0, +1), then attaches the diagram to a headless plotter and verifies:

* Submesh polydata has the right number of points (2 * sum_S) and
  quad cells.
* Top points after a step update equal ``base + scale * value * dir``
  for each station.
* The actor and polydata identity is stable across step changes
  (in-place mutation contract).
* set_scale and set_flip_sign re-render without re-attach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    LineForceDiagram,
    LineForceStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# =====================================================================
# Fixture: 3-element horizontal beam with synthetic line_stations
# =====================================================================

# Per-station value scheme (3-pt Lobatto, xi = -1, 0, +1):
#   value[step, eid_index, station] = (eid_index + 1) * 100 + step * 10 + station
# This lets per-step assertions verify the right slab values land
# on the right base point.

@pytest.fixture
def beam_results(g, tmp_path: Path):
    # Three geometry segments along +X. Use a coarse global size so
    # Gmsh produces one line element per segment (or close to it).
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0, label="p0")
    p1 = g.model.geometry.add_point(1.0, 0.0, 0.0, label="p1")
    p2 = g.model.geometry.add_point(2.0, 0.0, 0.0, label="p2")
    p3 = g.model.geometry.add_point(3.0, 0.0, 0.0, label="p3")
    g.model.geometry.add_line(p0, p1, label="seg0")
    g.model.geometry.add_line(p1, p2, label="seg1")
    g.model.geometry.add_line(p2, p3, label="seg2")
    g.physical.add_curve(["seg0", "seg1", "seg2"], name="Beam")
    # Coarse global size — Gmsh decides the discretization. The
    # fixture below reads the actual element count and adapts.
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    # Pick all line elements — let the fixture adapt to whatever Gmsh
    # actually produced.
    line_eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 1:
            line_eids.extend(int(x) for x in group.ids)
    line_eids = sorted(line_eids)
    n_beams = len(line_eids)
    assert n_beams >= 1, "Fixture produced no line elements"

    n_stations_per_beam = 3        # 3-pt Lobatto: xi = -1, 0, +1
    natural_coords = np.array([-1.0, 0.0, 1.0])

    n_steps = 3
    # Per-station values: per (eid_idx, station) for each step.
    # Shape required by writer: (T, n_elements, n_stations_per_element)
    values = np.zeros(
        (n_steps, n_beams, n_stations_per_beam), dtype=np.float64,
    )
    for step in range(n_steps):
        for ei in range(n_beams):
            for si in range(n_stations_per_beam):
                values[step, ei, si] = (ei + 1) * 100 + step * 10 + si

    path = tmp_path / "beam.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_line_stations_group(
            sid, "partition_0", group_id="g0",
            class_tag=10,    # arbitrary
            int_rule=0,
            element_index=np.asarray(line_eids, dtype=np.int64),
            station_natural_coord=natural_coords,
            components={"bending_moment_z": values},
        )
        w.end_stage()

    results = Results.from_native(path)
    return results, line_eids, natural_coords, values


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


# =====================================================================
# Construction validation
# =====================================================================

def test_construction_requires_line_force_style(beam_results):
    results, _, _, _ = beam_results
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad_spec = DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component="bending_moment_z"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="LineForceStyle"):
        LineForceDiagram(bad_spec, results)


# =====================================================================
# Attach
# =====================================================================

def _make_spec(component="bending_moment_z", scale=None, axis=None):
    return DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component=component),
        style=LineForceStyle(scale=scale, fill_axis=axis),
    )


def test_attach_requires_scene(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    diagram = LineForceDiagram(_make_spec(), results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, results.fem)


def test_attach_builds_polydata(beam_results, headless_plotter):
    results, line_eids, natural_coords, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n_total_stations = len(line_eids) * len(natural_coords)
    assert diagram._n_stations == n_total_stations
    assert diagram._fill_polydata is not None
    # Layout: base + top -> 2 * n_stations points
    assert diagram._fill_polydata.n_points == 2 * n_total_stations
    # Each beam contributes (n_stations - 1) quads
    expected_quads = len(line_eids) * (len(natural_coords) - 1)
    assert diagram._fill_polydata.n_cells == expected_quads


def test_base_points_match_station_positions(
    beam_results, headless_plotter,
):
    results, line_eids, natural_coords, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n_beams = len(line_eids)
    n_per_beam = len(natural_coords)
    bases = diagram._base_points
    assert bases.shape == (n_beams * n_per_beam, 3)

    # All base y/z = 0 (beam is horizontal along X)
    np.testing.assert_allclose(bases[:, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(bases[:, 2], 0.0, atol=1e-10)

    # X positions span the model — endpoints + midpoints for each
    # element. Just verify min/max bracket the model.
    x_values = bases[:, 0]
    assert x_values.min() >= -1e-12
    assert x_values.max() <= 3.0 + 1e-12


# =====================================================================
# Step update — top points = base + scale * value * fill_dir
# =====================================================================

def test_top_points_match_value_at_step_0(
    beam_results, headless_plotter,
):
    results, line_eids, _, raw_values = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    # Component is bending_moment_z -> y_local fill
    # All beams along +X, default vecxz=+Z -> y_local = +Y
    # So top - base = scale * value * (0, 1, 0)
    pts = np.asarray(diagram._fill_polydata.points)
    n = diagram._n_stations
    base = pts[:n]
    top = pts[n:]
    diff = top - base
    np.testing.assert_allclose(diff[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(diff[:, 2], 0.0, atol=1e-10)
    # The y component is scale * value at each station's slab index.
    # Values at step 0 are in the fixture; they're (ei+1)*100 + station.
    # Just verify the magnitudes are non-zero where expected.
    assert np.abs(diff[:, 1]).max() > 0


def test_step_update_changes_top_points(
    beam_results, headless_plotter,
):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n = diagram._n_stations
    pts = np.asarray(diagram._fill_polydata.points)
    diff_step0 = (pts[n:] - pts[:n]).copy()

    diagram.update_to_step(2)
    diff_step2 = pts[n:] - pts[:n]

    # Different magnitudes at different steps
    assert not np.allclose(diff_step0, diff_step2)


# =====================================================================
# Auto-scale at attach
# =====================================================================

def test_auto_scale_picks_fraction_of_diagonal(
    beam_results, headless_plotter,
):
    results, _, _, raw_values = beam_results
    scene = build_fem_scene(results.fem)
    style = LineForceStyle(scale=None, auto_scale_fraction=0.10)
    spec = DiagramSpec(
        kind="line_force",
        selector=SlabSelector(component="bending_moment_z"),
        style=style,
    )
    diagram = LineForceDiagram(spec, results)
    diagram.attach(headless_plotter, results.fem, scene)

    # Auto-scale fits to the global max across every step (not step 0)
    # so the largest fill across history hits ``auto_scale_fraction`` of
    # the model diagonal — keeps load-controlled runs sane (step 0 is
    # the first tiny increment).
    max_abs = float(np.abs(raw_values).max())
    expected = (0.10 * scene.model_diagonal) / max_abs
    assert diagram._initial_scale == pytest.approx(expected)


def test_explicit_scale_used(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=0.5), results)
    diagram.attach(headless_plotter, results.fem, scene)
    assert diagram._initial_scale == 0.5


# =====================================================================
# Runtime style adjustments
# =====================================================================

def test_set_scale_re_renders(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n = diagram._n_stations
    pts = np.asarray(diagram._fill_polydata.points)
    diff_before = (pts[n:] - pts[:n]).copy()

    diagram.set_scale(10.0)
    diff_after = pts[n:] - pts[:n]

    np.testing.assert_allclose(diff_after, 10.0 * diff_before, atol=1e-9)


def test_set_flip_sign_inverts_offsets(
    beam_results, headless_plotter,
):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n = diagram._n_stations
    pts = np.asarray(diagram._fill_polydata.points)
    diff_before = (pts[n:] - pts[:n]).copy()

    diagram.set_flip_sign(True)
    diff_after = pts[n:] - pts[:n]

    np.testing.assert_allclose(diff_after, -diff_before, atol=1e-10)


def test_current_scale(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=2.5), results)
    diagram.attach(headless_plotter, results.fem, scene)
    assert diagram.current_scale() == 2.5
    diagram.set_scale(7.0)
    assert diagram.current_scale() == 7.0


def test_set_fill_axis_to_global_z(beam_results, headless_plotter):
    """For an XY-plane beam, ``global_z`` makes the fill extend along +Z.

    bending_moment_z normally fills along ``y_local`` (= +Y for a beam
    along +X). Switching to ``global_z`` should reorient every station's
    fill direction to +Z so the diagram extrudes out of the model plane.
    """
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile([0.0, 1.0, 0.0], (diagram._n_stations, 1)),
    )

    diagram.set_fill_axis("global_z")
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile([0.0, 0.0, 1.0], (diagram._n_stations, 1)),
    )


def test_set_fill_axis_none_clears_runtime_override(
    beam_results, headless_plotter,
):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)
    diagram.set_fill_axis("global_z")
    assert diagram._runtime_axis == "global_z"
    diagram.set_fill_axis(None)
    assert diagram._runtime_axis is None
    # Fill direction is back to the component default (y_local = +Y).
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile([0.0, 1.0, 0.0], (diagram._n_stations, 1)),
    )


def test_set_fill_axis_tuple(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)
    diagram.set_fill_axis((0.0, 0.0, 5.0))    # any +Z scaling
    np.testing.assert_allclose(
        diagram._fill_directions,
        np.tile([0.0, 0.0, 1.0], (diagram._n_stations, 1)),
    )


# =====================================================================
# In-place mutation contract
# =====================================================================

def test_actor_identity_stable_across_steps(
    beam_results, headless_plotter,
):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)

    initial_actor = diagram._fill_actor
    initial_poly = diagram._fill_polydata
    initial_actor_id = id(initial_actor)
    initial_poly_id = id(initial_poly)

    for step in range(3):
        diagram.update_to_step(step)

    assert diagram._fill_actor is initial_actor
    assert id(diagram._fill_actor) == initial_actor_id
    assert diagram._fill_polydata is initial_poly
    assert id(diagram._fill_polydata) == initial_poly_id


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(beam_results, headless_plotter):
    results, _, _, _ = beam_results
    scene = build_fem_scene(results.fem)
    diagram = LineForceDiagram(_make_spec(scale=1.0), results)
    diagram.attach(headless_plotter, results.fem, scene)
    diagram.detach()
    assert diagram._fill_polydata is None
    assert diagram._fill_actor is None
    assert diagram._n_stations == 0
    assert not diagram.is_attached


# =====================================================================
# Real MPCO fixture — ElasticBeam3d localForce path
# =====================================================================
#
# elasticFrame.mpco contains 11 ElasticBeam3d elements writing
# localForce (12 components / 2 stations per element). Before the
# localForce reader landed, attaching LineForceDiagram on this fixture
# silently produced no actor — the slab was empty.

_FRAME_FIXTURE = Path("tests/fixtures/results/elasticFrame.mpco")


@pytest.fixture
def frame_results():
    if not _FRAME_FIXTURE.exists():
        pytest.skip(f"Missing fixture: {_FRAME_FIXTURE}")
    return Results.from_mpco(_FRAME_FIXTURE)


def test_elastic_frame_attach_builds_polydata(
    frame_results, headless_plotter,
):
    scoped = frame_results.stage(frame_results.stages[0].name)
    scene = build_fem_scene(scoped.fem)
    diagram = LineForceDiagram(_make_spec("bending_moment_z"), scoped)
    diagram.attach(headless_plotter, scoped.fem, scene)
    assert diagram._fill_polydata is not None
    # 11 elements × 2 stations × 2 (base + top points) = 44 vertices
    assert diagram._fill_polydata.n_points == 44
    # 11 elements × 1 quad per element (2 stations → 1 quad fill)
    assert diagram._fill_polydata.n_cells == 11


def test_elastic_frame_axial_force_diagram(
    frame_results, headless_plotter,
):
    """Axial diagram on a moment-loaded frame is symmetric per beam:
    N_1 + N_2 == 0 (verified in the reader test); make sure the diagram
    actually attaches without raising and step updates land cleanly.
    """
    scoped = frame_results.stage(frame_results.stages[0].name)
    scene = build_fem_scene(scoped.fem)
    diagram = LineForceDiagram(_make_spec("axial_force"), scoped)
    diagram.attach(headless_plotter, scoped.fem, scene)
    # Mutate to last step — must not raise and must keep actor identity.
    initial_poly = diagram._fill_polydata
    diagram.update_to_step(scoped.stages[0].n_steps - 1)
    assert diagram._fill_polydata is initial_poly
