"""LayerStackDiagram — emits a per-cell-scalar MeshLayer through the backend.

The fixture for layered shells is harder than fibers because we need
2-D mesh elements + per-layer slab data. We use a 2-D rectangle meshed
into quads, then write a synthetic ``LayerSlab`` covering each
quad's GPs.

Split coverage (ADR 0042 R-B Wave 2 #3):

* Emission + aggregation + side-panel accessors via the shared recording
  stub ``backend`` fixture (no GL): the emitted ``MeshLayer`` carries a
  per-cell ``ScalarField``; aggregation modes ('mid_layer', 'mean',
  'max_abs') produce the expected per-cell scalars; ``available_gps`` /
  ``read_thickness_profile`` are unchanged.
* Render integration via a real offscreen ``PyVistaQtBackend``
  (``pv_backend`` fixture): scalar bar on the plotter, mapper scalar
  range, in-place actor stability, the actor is non-pickable.
* A round-trip guard that a mixed tri+quad cell order stays aligned with
  its per-cell scalar through ``cellblocks_from_grid`` →
  ``mesh_layer_to_grid`` (the regrouping the diagram permutes around).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    LayerStackDiagram,
    LayerStackStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene_ir import MeshLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


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
        Results.from_native(path, model=_open_model_from_h5(path)),
        shell_eids, gps_per_shell, layers_per_gp,
        eid_arr, gp_arr, layer_arr, sub_arr, thickness_arr, values,
    )


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
# Attach + emission (recording stub backend)
# =====================================================================

def _emitted_field(diagram):
    layer = diagram._layer
    return layer.field_named(layer.color.array_name)


def test_attach_emits_cell_scalar_layer(layer_results, backend):
    results, shell_eids, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    layer = diagram._layer
    assert isinstance(layer, MeshLayer)
    assert layer.cells.n_cells >= 1
    assert layer.pickable is False
    # Per-cell ScalarField sized to the submesh.
    assert layer.color.mode == "by_array"
    field = _emitted_field(diagram)
    assert field is not None and field.location == "cell"
    assert field.values.size == layer.cells.n_cells
    assert diagram._cell_values.size == layer.cells.n_cells
    # Scalar bar registered on the backend.
    assert diagram._handle.layer_id in backend.scalar_bars


def test_attach_carries_style_opacity(layer_results, backend):
    results = layer_results[0]
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="layer_stack",
        selector=SlabSelector(component="stress_xx"),
        style=LayerStackStyle(aggregation="mid_layer", opacity=0.5),
    )
    diagram = LayerStackDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)
    assert diagram._layer.opacity == pytest.approx(0.5)


def test_available_gps_lists_all(layer_results, backend):
    (results, shell_eids, gps_per_shell, *_) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    expected = sorted(
        (int(eid), gp) for eid in shell_eids for gp in range(gps_per_shell)
    )
    assert diagram.available_gps() == expected


def test_set_visible_routes_through_backend(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    diagram.set_visible(False)
    assert diagram._handle.visible is False
    diagram.set_visible(True)
    assert diagram._handle.visible is True


# =====================================================================
# Aggregations (on the emitted per-cell scalar)
# =====================================================================

def test_mid_layer_aggregation_picks_middle(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mid_layer"), results)
    diagram.attach(backend, results.fem, scene)

    # mid_layer picks one specific row per cell; values at step 0 are
    # exactly the row indices, so the per-cell scalar equals the
    # picked row index — and is mirrored onto the emitted ScalarField.
    scalars = np.asarray(diagram._cell_values)
    assert np.all(scalars >= 0)
    np.testing.assert_array_equal(
        np.asarray(_emitted_field(diagram).values), scalars,
    )


def test_mean_aggregation_averages_layers(layer_results, backend):
    (results, shell_eids, *_, values) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mean"), results)
    diagram.attach(backend, results.fem, scene)

    scalars = np.asarray(diagram._cell_values)
    assert scalars.min() >= 0
    assert scalars.max() <= values[0].max()


def test_max_abs_picks_largest_magnitude(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("max_abs"), results)
    diagram.attach(backend, results.fem, scene)

    scalars = np.asarray(diagram._cell_values)
    # All values are non-negative, so max_abs == max
    assert scalars.max() > 0


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_scalars(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mid_layer"), results)
    diagram.attach(backend, results.fem, scene)
    initial = np.asarray(diagram._cell_values).copy()

    diagram.update_to_step(2)
    after = np.asarray(diagram._cell_values)
    # step 2 values are step*1000 larger.
    assert not np.array_equal(initial, after)
    # The emitted layer's ScalarField reflects the new step.
    np.testing.assert_array_equal(
        np.asarray(_emitted_field(diagram).values), after,
    )


def test_handle_stable_across_steps(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    initial_handle = diagram._handle
    for step in range(3):
        diagram.update_to_step(step)
    assert diagram._handle is initial_handle


# =====================================================================
# Side-panel data accessor (unchanged — uses the stub backend)
# =====================================================================

def test_read_thickness_profile_returns_sorted_profile(layer_results, backend):
    (results, shell_eids, gps_per_shell, layers_per_gp,
     eid_arr, gp_arr, layer_arr, sub_arr, thickness_arr,
     values) = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

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


def test_read_thickness_profile_uses_step(layer_results, backend):
    results, shell_eids, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    eid = int(shell_eids[0])
    d0 = diagram.read_thickness_profile(eid, 0, step_index=0)
    d2 = diagram.read_thickness_profile(eid, 0, step_index=2)
    assert d0 is not None and d2 is not None
    assert not np.array_equal(d0[1], d2[1])


def test_read_thickness_profile_unknown_returns_none(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram.read_thickness_profile(99999, 0, step_index=0) is None


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    layer_id = diagram._handle.layer_id
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert diagram._slab_layer is None
    assert not diagram.is_attached
    assert layer_id in backend.removed
    assert layer_id not in backend.scalar_bars


# =====================================================================
# LUT mirror (diagram-side state, recording stub backend)
# =====================================================================


def test_layer_lut_is_none_before_attach(layer_results):
    results, *_ = layer_results
    diagram = LayerStackDiagram(_spec(), results)
    assert diagram.lut is None


def test_layer_attach_builds_lut_from_style(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="layer_stack",
        selector=SlabSelector(component="stress_xx"),
        style=LayerStackStyle(
            aggregation="mid_layer", cmap="plasma", clim=(-5.0, 5.0),
        ),
    )
    diagram = LayerStackDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "stress_xx"
    assert lut.preset == "plasma"
    assert lut.range == (-5.0, 5.0)


def test_layer_attach_lut_picks_up_autofit_clim(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    clim = diagram.current_clim()
    assert lut.range == clim


def test_layer_set_cmap_routes_through_lut(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    assert diagram._runtime_cmap == "turbo"


def test_layer_set_clim_routes_through_lut(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_layer_lut_change_pushes_colorspec_through_backend(
    layer_results, backend,
):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    color = backend.colors[diagram._handle.layer_id]
    assert color.mode == "by_array"
    assert color.lut.vmin == pytest.approx(100.0)
    assert color.lut.vmax == pytest.approx(200.0)


def test_layer_detach_clears_lut(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_layer_lut_changes_after_detach_are_noops(layer_results, backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(backend, results.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)


# =====================================================================
# Render integration (real offscreen PyVistaQtBackend)
# =====================================================================

def test_scalar_bar_appears_on_plotter(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    assert "stress_xx" in pv_backend.plotter.scalar_bars


def test_actor_is_non_pickable(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    assert diagram._handle.actor.GetPickable() == 0


def test_actor_identity_stable_across_steps(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    initial_actor = diagram._handle.actor
    initial_dataset = diagram._handle.dataset

    for step in range(3):
        diagram.update_to_step(step)

    # In-place fast path: topology unchanged, so the actor + dataset
    # are reused rather than re-added.
    assert diagram._handle.actor is initial_actor
    assert diagram._handle.dataset is initial_dataset


def test_step_update_recolors_dataset_in_place(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec("mid_layer"), results)
    diagram.attach(pv_backend, results.fem, scene)
    diagram.update_to_step(2)
    ds = diagram._handle.dataset
    np.testing.assert_array_equal(
        np.asarray(ds.cell_data["stress_xx"]),
        np.asarray(diagram._cell_values),
    )


def test_detach_removes_scalar_bar(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    for _ in range(3):
        diagram = LayerStackDiagram(_spec(), results)
        diagram.attach(pv_backend, results.fem, scene)
        diagram.detach()
    assert "stress_xx" not in (pv_backend.plotter.scalar_bars or {})


def test_set_show_and_fmt_live(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.set_show_scalar_bar(False)
    assert "stress_xx" not in pv_backend.plotter.scalar_bars

    diagram.set_show_scalar_bar(True)
    assert "stress_xx" in pv_backend.plotter.scalar_bars

    diagram.set_fmt("%.5f")
    bar = pv_backend.plotter.scalar_bars["stress_xx"]
    assert bar.GetLabelFormat() == "%.5f"


def test_layer_lut_change_updates_actor_mapper(layer_results, pv_backend):
    results, *_ = layer_results
    scene = build_fem_scene(results.fem)
    diagram = LayerStackDiagram(_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._handle.actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


# =====================================================================
# Mixed tri+quad cell-order alignment (round-trip guard)
# =====================================================================

def test_mixed_celltype_scalar_stays_aligned_through_round_trip():
    """A per-cell scalar reordered into ``cellblocks_from_grid`` grouped
    order (the diagram's ``group_to_orig`` permutation) must land on the
    correct cell after ``mesh_layer_to_grid`` rebuilds the grid — i.e.
    the value on each rebuilt cell matches the original cell with the
    same connectivity. Guards the regrouping the diagram permutes around.
    """
    import pyvista as pv

    from apeGmsh.viewers.backends.pyvista_qt import (
        cellblocks_from_grid,
        mesh_layer_to_grid,
    )
    from apeGmsh.viewers.scene_ir import (
        ColorSpec,
        LutSpec,
        PointSet,
        ScalarField,
    )

    pts = np.random.default_rng(0).random((12, 3))
    VTK_TRI, VTK_QUAD = 5, 9
    # Interleaved original order: quad, tri, quad, tri.
    conns = [
        (VTK_QUAD, [0, 1, 2, 3]),
        (VTK_TRI, [4, 5, 6]),
        (VTK_QUAD, [7, 8, 9, 10]),
        (VTK_TRI, [5, 6, 11]),
    ]
    cell_arr: list[int] = []
    ctypes: list[int] = []
    for ct, conn in conns:
        cell_arr.append(len(conn))
        cell_arr.extend(conn)
        ctypes.append(ct)
    submesh = pv.UnstructuredGrid(
        np.array(cell_arr), np.array(ctypes), pts,
    )
    # A distinct per-ORIGINAL-cell value.
    orig_values = np.array([10.0, 20.0, 30.0, 40.0])

    # Diagram's permutation: group original cells by cells_dict order.
    celltypes = np.asarray(submesh.celltypes, dtype=np.int64)
    group_to_orig = np.concatenate(
        [np.where(celltypes == t)[0] for t in submesh.cells_dict]
    ).astype(np.int64)
    grouped_values = orig_values[group_to_orig]

    cells = cellblocks_from_grid(submesh)
    layer = MeshLayer(
        layer_id="mix",
        points=PointSet(pts),
        cells=cells,
        fields=(ScalarField("v", grouped_values, "cell"),),
        color=ColorSpec(mode="by_array", array_name="v", lut=LutSpec()),
    )
    grid = mesh_layer_to_grid(layer)

    # For each rebuilt cell, the connectivity identifies its original
    # cell; the scalar must equal that original cell's value.
    orig_conn_to_value = {
        tuple(sorted(conn)): orig_values[i]
        for i, (_ct, conn) in enumerate(conns)
    }
    rebuilt = grid.cells_dict
    rebuilt_vals = np.asarray(grid.cell_data["v"])
    # Walk rebuilt cells in grid order and match connectivity.
    cell_idx = 0
    for _ct, conn_block in rebuilt.items():
        for row in conn_block:
            key = tuple(sorted(int(x) for x in row))
            assert rebuilt_vals[cell_idx] == pytest.approx(
                orig_conn_to_value[key]
            )
            cell_idx += 1
    assert cell_idx == orig_values.size
