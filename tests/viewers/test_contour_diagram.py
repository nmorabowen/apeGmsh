"""ContourDiagram — attach, step update, runtime style adjustments.

Uses a real Results built via NativeWriter + a headless pyvista
Plotter (``off_screen=True``) so we can exercise the full attach
+ update_to_step + render path without a Qt window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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

from tests.conftest import _open_model_from_h5


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
    return Results.from_native(path, model=_open_model_from_h5(path))


# headless_plotter is a shared fixture in tests/viewers/conftest.py
# (yields a PyVistaQtBackend, ADR 0042 R-B.final).


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

    layer = diagram._layer
    assert layer is not None
    assert layer.points.n_points == scene.grid.n_points
    # Step 0 values: node_id + 0*1000 == node_id. The emitted point
    # ScalarField is in submesh-point order, row-aligned with the FEM
    # ids the diagram cached for that order (_fem_ids_to_read).
    field = layer.field_named("displacement_z")
    assert field is not None and field.location == "point"
    fem_ids_in_submesh = diagram._fem_ids_to_read
    np.testing.assert_array_equal(
        np.asarray(field.values), fem_ids_in_submesh.astype(np.float64),
    )


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
    assert diagram._layer is not None
    assert diagram._layer.points.n_points > 0


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

    fem_ids_in_submesh = diagram._fem_ids_to_read

    for step in (1, 2, 3, 0):    # also test going back to 0
        diagram.update_to_step(step)
        field = diagram._layer.field_named("displacement_z")
        expected = fem_ids_in_submesh.astype(np.float64) + step * 1000.0
        np.testing.assert_array_equal(np.asarray(field.values), expected)


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
    assert diagram._layer is not None

    diagram.detach()
    assert not diagram.is_attached
    assert diagram._layer is None
    assert diagram._handle is None
    assert diagram._scalar_values is None


def test_detach_then_reattach_works(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    diagram.detach()
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    assert diagram.is_attached
    assert diagram._layer is not None


# =====================================================================
# LUT mirror (plan 06 step 2)
# =====================================================================


def test_lut_is_none_before_attach(results_with_known_disp):
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    assert diagram.lut is None


def test_attach_builds_lut_from_style_defaults(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    spec = DiagramSpec(
        kind="contour",
        selector=SlabSelector(component="displacement_z"),
        style=ContourStyle(cmap="plasma", clim=(-5.0, 5.0)),
    )
    diagram = ContourDiagram(spec, results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "displacement_z"
    assert lut.preset == "plasma"
    assert lut.range == (-5.0, 5.0)


def test_attach_lut_picks_up_autofit_clim(
    results_with_known_disp, headless_plotter,
):
    """When the style leaves clim=None, the LUT should pick up the
    auto-fitted range from step 0 (not the dummy (0, 1))."""
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    lut = diagram.lut
    clim = diagram.current_clim()
    assert lut.range == clim


def test_set_cmap_routes_through_lut(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    # Runtime override mirrors so it survives a detach/re-attach round.
    assert diagram._runtime_cmap == "turbo"


def test_set_clim_routes_through_lut(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_lut_change_updates_actor_mapper(
    results_with_known_disp, headless_plotter,
):
    """Mutating the LUT directly should push state to the mapper."""
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._handle.actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


def test_detach_clears_lut(
    results_with_known_disp, headless_plotter,
):
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_lut_changes_after_detach_are_noops(
    results_with_known_disp, headless_plotter,
):
    """A LUT instance held externally must not raise after the diagram
    has detached — the disconnect in detach() should prevent the
    callback from firing on a torn-down actor."""
    scene = build_fem_scene(results_with_known_disp.fem)
    diagram = ContourDiagram(_make_spec(), results_with_known_disp)
    diagram.attach(headless_plotter, results_with_known_disp.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    # No assertion needed — this must simply not raise.
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)
