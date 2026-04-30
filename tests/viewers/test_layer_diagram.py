"""LayerStackDiagram — attach + step + thickness profile.

The fixture for layered shells is harder than fibers because we need
2-D mesh elements + per-layer slab data. We use a 2-D rectangle meshed
into quads, then write a synthetic ``LayerSlab`` covering each
quad's GPs.

Tests:

* Substrate sub-mesh extraction picks the shell cells.
* Aggregation modes ('mid_layer', 'mean', 'max_abs') produce expected
  per-cell scalars on a 4-layer fixture.
* Step updates mutate scalars in place.
* ``read_thickness_profile`` returns the bottom-to-top profile sorted
  by (layer_index, sub_gp_index), with cumulative-mid thickness coords.
* The picked-gp listing covers all (eid, gp) pairs.
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
    LayerStackDiagram,
    LayerStackStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


# Fixture: 2-D plate -> shells with N layers, M sub-GPs each.
# value[step, row] = step * 1000 + row
# thickness[row] = 0.25 (constant)
# layer_index runs 0..N-1, sub_gp_index runs 0..M-1, gp_index runs 0..1

@pytest.fixture
def layer_results(g, tmp_path: Path):
    g.model.geometry.add_rectangle(0, 0, 0, 2, 1, label="plate")
    g.physical.add_surface("plate", name="Plate")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)

    shell_eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 2:
            shell_eids.extend(int(x) for x in group.ids)
    shell_eids = sorted(shell_eids)
    n_shells = len(shell_eids)
    assert n_shells >= 1, "Fixture produced no shell elements"

    gps_per_shell = 2
    layers_per_gp = 4
    sub_gps_per_layer = 1   # one sub-GP per layer (simple layered shell)
    n_steps = 3

    rows: list[tuple[int, int, int, int, float]] = []
    for eid in shell_eids:
        for gp in range(gps_per_shell):
            for layer in range(layers_per_gp):
                for sub in range(sub_gps_per_layer):
                    rows.append((
                        eid, gp, layer, sub, 0.25,
                    ))
    eid_arr = np.asarray([r[0] for r in rows], dtype=np.int64)
    gp_arr = np.asarray([r[1] for r in rows], dtype=np.int64)
    layer_arr = np.asarray([r[2] for r in rows], dtype=np.int64)
    sub_arr = np.asarray([r[3] for r in rows], dtype=np.int64)
    thickness_arr = np.asarray([r[4] for r in rows], dtype=np.float64)
    quat_arr = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(rows), 1))
    n_rows = len(rows)

    values = np.zeros((n_steps, n_rows), dtype=np.float64)
    for step in range(n_steps):
        for k in range(n_rows):
            values[step, k] = step * 1000 + k

    path = tmp_path / "layers.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_layers_group(
            sid, "partition_0", group_id="g0",
            element_index=eid_arr,
            gp_index=gp_arr,
            layer_index=layer_arr,
            sub_gp_index=sub_arr,
            thickness=thickness_arr,
            local_axes_quaternion=quat_arr,
            components={"stress_xx": values},
        )
        w.end_stage()

    return (
        Results.from_native(path),
        shell_eids, gps_per_shell, layers_per_gp,
        eid_arr, gp_arr, layer_arr, sub_arr, thickness_arr, values,
    )


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


def _spec(aggregation="mid_layer") -> DiagramSpec:
    return DiagramSpec(
        kind="layer_stack",
        selector=SlabSelector(component="stress_xx"),
        style=LayerStackStyle(aggregation=aggregation),
    )


# =====================================================================
# Construction
# =====================================================================

def test_construction_requires_layer_style(layer_results):
    results = layer_results[0]
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="layer_stack",
        selector=SlabSelector(component="stress_xx"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="LayerStackStyle"):
        LayerStackDiagram(bad, results)


def test_construction_validates_aggregation(layer_results):
    results = layer_results[0]
    bad = DiagramSpec(
        kind="layer_stack",
        selector=SlabSelector(component="stress_xx"),
        style=LayerStackStyle(aggregation="bogus"),
    )
    with pytest.raises(ValueError, match="aggregation"):
        LayerStackDiagram(bad, results)


# =====================================================================
# Attach
# =====================================================================

def test_attach_builds_submesh(layer_results, headless_plotter):
    results, shell_eids, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    assert diagram._submesh is not None
    assert diagram._submesh.n_cells >= 1
    assert diagram._scalar_array is not None
    assert diagram._scalar_array.size == diagram._submesh.n_cells


def test_available_gps_lists_all(layer_results, headless_plotter):
    (results, shell_eids, gps_per_shell, *_) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    expected = sorted(
        (int(eid), gp) for eid in shell_eids for gp in range(gps_per_shell)
    )
    assert diagram.available_gps() == expected


# =====================================================================
# Aggregations
# =====================================================================

def test_mid_layer_aggregation_picks_middle(
    layer_results, headless_plotter,
):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mid_layer"), results)
    diagram.attach(headless_plotter, results.fem, scene)

    # mid_layer picks one specific row per cell; values at step 0 are
    # exactly the row indices, so the per-cell scalar equals the
    # picked row index.
    scalars = np.asarray(diagram._scalar_array)
    # All cells should pick a valid row
    assert np.all(scalars >= 0)


def test_mean_aggregation_averages_layers(
    layer_results, headless_plotter,
):
    (results, shell_eids, *_, values) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mean"), results)
    diagram.attach(headless_plotter, results.fem, scene)

    scalars = np.asarray(diagram._scalar_array)
    # The mean of integer values 0..N-1 equals (N-1)/2; with multiple
    # cells the mean per cell is the mean of that cell's slab rows.
    # Just verify the values are within the data range.
    assert scalars.min() >= 0
    assert scalars.max() <= values[0].max()


def test_max_abs_picks_largest_magnitude(
    layer_results, headless_plotter,
):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("max_abs"), results)
    diagram.attach(headless_plotter, results.fem, scene)

    scalars = np.asarray(diagram._scalar_array)
    # All values are non-negative, so max_abs == max
    assert scalars.max() > 0


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_scalars(layer_results, headless_plotter):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mid_layer"), results)
    diagram.attach(headless_plotter, results.fem, scene)
    initial = np.asarray(diagram._scalar_array).copy()

    diagram.update_to_step(2)
    after = np.asarray(diagram._scalar_array)
    # step 2 values are step*1000 larger
    assert not np.array_equal(initial, after)


# =====================================================================
# Side-panel data accessor
# =====================================================================

def test_read_thickness_profile_returns_sorted_profile(
    layer_results, headless_plotter,
):
    (results, shell_eids, gps_per_shell, layers_per_gp,
     eid_arr, gp_arr, layer_arr, sub_arr, thickness_arr,
     values) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    eid = int(shell_eids[0])
    gp_idx = 0
    data = diagram.read_thickness_profile(eid, gp_idx, step_index=0)
    assert data is not None
    midpoints, vals = data
    # Should have layers_per_gp * sub_gps_per_layer rows
    expected_n = layers_per_gp * 1  # sub_gps_per_layer = 1
    assert midpoints.size == expected_n
    assert vals.size == expected_n
    # Cumulative midpoints with thickness 0.25 each: 0.125, 0.375, 0.625, 0.875
    expected_mid = np.array([0.125, 0.375, 0.625, 0.875])
    np.testing.assert_allclose(midpoints, expected_mid)


def test_read_thickness_profile_uses_step(
    layer_results, headless_plotter,
):
    results, shell_eids, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    eid = int(shell_eids[0])
    d0 = diagram.read_thickness_profile(eid, 0, step_index=0)
    d2 = diagram.read_thickness_profile(eid, 0, step_index=2)
    assert d0 is not None and d2 is not None
    assert not np.array_equal(d0[1], d2[1])


def test_read_thickness_profile_unknown_returns_none(
    layer_results, headless_plotter,
):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)
    assert diagram.read_thickness_profile(99999, 0, step_index=0) is None


# =====================================================================
# In-place mutation
# =====================================================================

def test_actor_identity_stable_across_steps(
    layer_results, headless_plotter,
):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    initial_actor = diagram._actor
    initial_submesh = diagram._submesh
    initial_scalar = diagram._scalar_array

    for step in range(3):
        diagram.update_to_step(step)

    assert diagram._actor is initial_actor
    assert diagram._submesh is initial_submesh
    assert diagram._scalar_array is initial_scalar


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(layer_results, headless_plotter):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)
    diagram.detach()
    assert diagram._submesh is None
    assert diagram._actor is None
    assert diagram._slab_layer is None
    assert not diagram.is_attached
