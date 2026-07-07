"""FiberSectionDiagram — emits a point-cloud MeshLayer through the backend.

Builds a 2-element beam, writes synthetic fiber data via NativeWriter
(2 GPs per beam, 4 fibers per GP), then verifies (ADR 0042 R-B Wave 2 #2):

* Emission + side-panel data accessors via the shared recording stub
  ``backend`` fixture (no GL): the emitted ``MeshLayer`` carries one
  vertex cell per fiber, a point ``ScalarField``, point-cloud render
  attributes, and ``pickable=False``; ``available_gps`` /
  ``read_section_at_gp`` are unchanged.
* Render integration via a real offscreen ``PyVistaQtBackend``
  (``pv_backend`` fixture): scalar bar on the plotter, mapper scalar
  range, in-place actor stability across steps, the actor is
  non-pickable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh.results import Results
from apeGmsh.results.writers import NativeWriter
from apeGmsh.viewers.diagrams import (
    DiagramSpec,
    FiberSectionDiagram,
    FiberSectionStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene_ir import MeshLayer
from apeGmsh.viewers.scene.fem_scene import build_fem_scene

from tests.conftest import _open_model_from_h5


# Per-fiber fixture scheme (n_fibers_per_gp = 4):
#
#   value[step, row] = step * 1000 + row
#
# row index = (eid_local * gps_per_beam + gp) * fibers_per_gp + fiber_local
# y[row]    = fiber_local - 1.5      (centred around 0)
# z[row]    = (eid_local * gps_per_beam + gp) % 4 - 1.5
# area[row] = 1.0
# material  = fiber_local % 2

@pytest.fixture
def fiber_results(g, tmp_path: Path):
    # Two line segments
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0, label="p0")
    p1 = g.model.geometry.add_point(1.0, 0.0, 0.0, label="p1")
    p2 = g.model.geometry.add_point(2.0, 0.0, 0.0, label="p2")
    g.model.geometry.add_line(p0, p1, label="seg0")
    g.model.geometry.add_line(p1, p2, label="seg1")
    g.physical.add_curve(["seg0", "seg1"], name="Beam")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    line_eids: list[int] = []
    for group in fem.elements:
        if group.element_type.dim == 1:
            line_eids.extend(int(x) for x in group.ids)
    line_eids = sorted(line_eids)
    n_beams = len(line_eids)
    assert n_beams >= 2, f"Fixture expected >= 2 line elements, got {n_beams}"

    gps_per_beam = 2
    fibers_per_gp = 4
    n_steps = 3

    # Build flat per-fiber arrays
    rows: list[tuple[int, int, float, float, float, int]] = []
    # (element_id, gp, y, z, area, material_tag)
    for ei, eid in enumerate(line_eids):
        for gp in range(gps_per_beam):
            for fk in range(fibers_per_gp):
                rows.append((
                    eid, gp,
                    float(fk - 1.5),
                    float(((ei * gps_per_beam + gp) % 4) - 1.5),
                    1.0,
                    fk % 2,
                ))
    eid_arr = np.asarray([r[0] for r in rows], dtype=np.int64)
    gp_arr = np.asarray([r[1] for r in rows], dtype=np.int64)
    y_arr = np.asarray([r[2] for r in rows], dtype=np.float64)
    z_arr = np.asarray([r[3] for r in rows], dtype=np.float64)
    area_arr = np.asarray([r[4] for r in rows], dtype=np.float64)
    mat_arr = np.asarray([r[5] for r in rows], dtype=np.int64)
    n_rows = len(rows)

    values = np.zeros((n_steps, n_rows), dtype=np.float64)
    for step in range(n_steps):
        for k in range(n_rows):
            values[step, k] = step * 1000 + k

    path = tmp_path / "fibers.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(n_steps, dtype=np.float64),
        )
        w.write_fibers_group(
            sid, "partition_0", group_id="g0",
            section_tag=10,
            section_class="FiberSection",
            element_index=eid_arr,
            gp_index=gp_arr,
            y=y_arr, z=z_arr, area=area_arr,
            material_tag=mat_arr,
            components={"fiber_stress": values},
        )
        w.end_stage()

    return (
        Results.from_native(path, model=_open_model_from_h5(path)),
        line_eids, gps_per_beam, fibers_per_gp,
        eid_arr, gp_arr, y_arr, z_arr, area_arr, values,
    )


def _make_spec() -> DiagramSpec:
    return DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=FiberSectionStyle(),
    )


# =====================================================================
# Construction
# =====================================================================

def test_construction_requires_fiber_style(fiber_results):
    results = fiber_results[0]
    from apeGmsh.viewers.diagrams._styles import DiagramStyle
    bad = DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=DiagramStyle(),
    )
    with pytest.raises(TypeError, match="FiberSectionStyle"):
        FiberSectionDiagram(bad, results)


# =====================================================================
# Attach + emission (recording stub backend)
# =====================================================================

def test_attach_requires_scene(fiber_results, backend):
    results = fiber_results[0]
    diagram = FiberSectionDiagram(_make_spec(), results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(backend, results.fem)


def test_attach_emits_point_cloud_layer(fiber_results, backend):
    (results, line_eids, gps_per_beam, fibers_per_gp,
     *_, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    n_total = len(line_eids) * gps_per_beam * fibers_per_gp
    layer = diagram._layer
    assert isinstance(layer, MeshLayer)
    assert layer.points.n_points == n_total
    # One vertex cell per fiber.
    assert set(layer.cells.blocks) == {"vertex"}
    assert layer.cells.n_cells == n_total
    # Point-cloud render attributes; decorative (non-pickable). Flat
    # points, NOT sphere billboards — billboards draw nothing on some
    # GL stacks (see FiberSectionDiagram._build_layer).
    assert layer.point_size is not None
    assert layer.render_points_as_spheres is False
    assert layer.pickable is False
    # Coloured by the fiber value (point ScalarField).
    assert layer.color.mode == "by_array"
    field = layer.field_named(layer.color.array_name)
    assert field is not None and field.location == "point"
    np.testing.assert_array_equal(np.asarray(field.values), values[0])
    # Scalar bar registered on the backend.
    assert diagram._handle.layer_id in backend.scalar_bars


def test_attach_carries_style_opacity(fiber_results, backend):
    results = fiber_results[0]
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=FiberSectionStyle(opacity=0.3),
    )
    diagram = FiberSectionDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)
    assert diagram._layer.opacity == pytest.approx(0.3)


def test_attach_carries_style_point_size(fiber_results, backend):
    results = fiber_results[0]
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=FiberSectionStyle(point_size=4.0),
    )
    diagram = FiberSectionDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)
    assert diagram._layer.point_size == pytest.approx(4.0)


def test_set_point_size_live(fiber_results, backend):
    """Runtime override re-emits the layer with the new dot size."""
    results = fiber_results[0]
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram.current_point_size() == pytest.approx(10.0)
    diagram.set_point_size(22.0)
    assert diagram.current_point_size() == pytest.approx(22.0)
    assert diagram._layer.point_size == pytest.approx(22.0)


def test_set_point_size_reaches_actor(fiber_results, pv_backend):
    """The in-place update path must push the size onto the actor
    property — the dataset carries no point size."""
    results = fiber_results[0]
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    try:
        diagram.set_point_size(17.0)
        assert float(diagram._handle.actor.prop.point_size) == pytest.approx(17.0)
    finally:
        diagram.detach()


def test_legacy_style_dict_without_point_size_deserializes():
    """Sessions saved before ``point_size`` existed carry only the
    (deprecated, never-consumed) ``point_size_fraction`` — they must
    still restore, defaulting the new field."""
    from apeGmsh.viewers.diagrams._session import deserialize_spec
    spec = deserialize_spec({
        "kind": "fiber_section",
        "selector": {"component": "fiber_stress"},
        "style": {
            "cmap": "coolwarm", "clim": None, "opacity": 1.0,
            "point_size_fraction": 0.005, "show_scalar_bar": True,
            "fmt": "%.3g", "panel_marker_scale": 60.0,
            "panel_show_areas": True,
        },
        "stage_id": None, "visible": True, "label": None,
    })
    assert isinstance(spec.style, FiberSectionStyle)
    assert spec.style.point_size == pytest.approx(10.0)


def test_available_gps_lists_all_pairs(fiber_results, backend):
    (results, line_eids, gps_per_beam, *_) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    expected = sorted(
        (int(eid), gp) for eid in line_eids for gp in range(gps_per_beam)
    )
    assert diagram.available_gps() == expected


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_scalars(fiber_results, backend):
    results, _, _, _, *_, values = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.update_to_step(2)
    field = diagram._layer.field_named(diagram._layer.color.array_name)
    np.testing.assert_array_equal(np.asarray(field.values), values[2])


def test_handle_stable_across_steps(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    initial_handle = diagram._handle
    for step in range(3):
        diagram.update_to_step(step)
    assert diagram._handle is initial_handle


def test_set_visible_routes_through_backend(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    diagram.set_visible(False)
    assert diagram._handle.visible is False
    diagram.set_visible(True)
    assert diagram._handle.visible is True


# =====================================================================
# Side-panel data accessor (unchanged — uses the stub backend)
# =====================================================================

def test_read_section_at_gp_returns_correct_subset(fiber_results, backend):
    (results, line_eids, gps_per_beam, fibers_per_gp,
     eid_arr, gp_arr, y_arr, z_arr, area_arr, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    # Pick the first beam, gp 1
    eid = int(line_eids[0])
    gp_idx = 1
    data = diagram.read_section_at_gp(eid, gp_idx, step_index=0)
    assert data is not None
    y_out, z_out, area_out, val_out = data

    expected_mask = (eid_arr == eid) & (gp_arr == gp_idx)
    np.testing.assert_array_equal(y_out, y_arr[expected_mask])
    np.testing.assert_array_equal(z_out, z_arr[expected_mask])
    np.testing.assert_array_equal(area_out, area_arr[expected_mask])
    np.testing.assert_array_equal(val_out, values[0][expected_mask])


def test_read_section_at_gp_uses_step(fiber_results, backend):
    (results, line_eids, *_, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    data_step0 = diagram.read_section_at_gp(int(line_eids[0]), 0, step_index=0)
    data_step2 = diagram.read_section_at_gp(int(line_eids[0]), 0, step_index=2)
    assert data_step0 is not None and data_step2 is not None
    assert not np.array_equal(data_step0[3], data_step2[3])


def test_read_section_at_gp_unknown_returns_none(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    assert diagram.read_section_at_gp(99999, 0, step_index=0) is None


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    layer_id = diagram._handle.layer_id
    diagram.detach()
    assert diagram._layer is None
    assert diagram._handle is None
    assert diagram._slab_y is None
    assert not diagram.is_attached
    assert layer_id in backend.removed
    assert layer_id not in backend.scalar_bars


# =====================================================================
# LUT mirror (diagram-side state, recording stub backend)
# =====================================================================


def test_fiber_lut_is_none_before_attach(fiber_results):
    results, *_ = fiber_results
    diagram = FiberSectionDiagram(_make_spec(), results)
    assert diagram.lut is None


def test_fiber_attach_builds_lut_from_style(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    spec = DiagramSpec(
        kind="fiber_section",
        selector=SlabSelector(component="fiber_stress"),
        style=FiberSectionStyle(cmap="plasma", clim=(-5.0, 5.0)),
    )
    diagram = FiberSectionDiagram(spec, results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    assert lut is not None
    assert lut.array_name == "fiber_stress"
    assert lut.preset == "plasma"
    assert lut.range == (-5.0, 5.0)


def test_fiber_attach_lut_picks_up_autofit_clim(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    lut = diagram.lut
    clim = diagram.current_clim()
    assert lut.range == clim


def test_fiber_set_cmap_routes_through_lut(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_cmap("turbo")
    assert diagram.lut.preset == "turbo"
    assert diagram._runtime_cmap == "turbo"


def test_fiber_set_clim_routes_through_lut(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.set_clim(-2.0, 7.0)
    assert diagram.lut.range == (-2.0, 7.0)
    assert diagram.current_clim() == (-2.0, 7.0)


def test_fiber_lut_change_pushes_colorspec_through_backend(
    fiber_results, backend,
):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    color = backend.colors[diagram._handle.layer_id]
    assert color.mode == "by_array"
    assert color.lut.vmin == pytest.approx(100.0)
    assert color.lut.vmax == pytest.approx(200.0)


def test_fiber_detach_clears_lut(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    assert diagram.lut is not None
    diagram.detach()
    assert diagram.lut is None


def test_fiber_lut_changes_after_detach_are_noops(fiber_results, backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)
    held_lut = diagram.lut
    diagram.detach()
    held_lut.set_preset("magma")
    held_lut.set_range(0.0, 1.0)


# =====================================================================
# Render integration (real offscreen PyVistaQtBackend)
# =====================================================================

def test_scalar_bar_appears_on_plotter(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    assert "fiber_stress" in pv_backend.plotter.scalar_bars


def test_actor_is_non_pickable(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    assert diagram._handle.actor.GetPickable() == 0


def test_actor_identity_stable_across_steps(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    initial_actor = diagram._handle.actor
    initial_dataset = diagram._handle.dataset

    for step in range(3):
        diagram.update_to_step(step)

    # In-place fast path: topology unchanged, so the actor + dataset
    # are reused rather than re-added.
    assert diagram._handle.actor is initial_actor
    assert diagram._handle.dataset is initial_dataset


def test_step_update_recolors_dataset_in_place(fiber_results, pv_backend):
    results, _, _, _, *_, values = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)
    diagram.update_to_step(2)
    ds = diagram._handle.dataset
    np.testing.assert_array_equal(
        np.asarray(ds.point_data["fiber_stress"]), values[2],
    )


def test_detach_removes_scalar_bar(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    for _ in range(3):
        diagram = FiberSectionDiagram(_make_spec(), results)
        diagram.attach(pv_backend, results.fem, scene)
        diagram.detach()
    assert "fiber_stress" not in (pv_backend.plotter.scalar_bars or {})


def test_set_show_and_fmt_live(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.set_show_scalar_bar(False)
    assert "fiber_stress" not in pv_backend.plotter.scalar_bars

    diagram.set_show_scalar_bar(True)
    assert "fiber_stress" in pv_backend.plotter.scalar_bars

    diagram.set_fmt("%.2e")
    bar = pv_backend.plotter.scalar_bars["fiber_stress"]
    assert bar.GetLabelFormat() == "%.2e"


def test_fiber_lut_change_updates_actor_mapper(fiber_results, pv_backend):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.lut.set_range(100.0, 200.0)
    mapper = diagram._handle.actor.GetMapper()
    sr = mapper.GetScalarRange()
    assert sr[0] == pytest.approx(100.0)
    assert sr[1] == pytest.approx(200.0)


def test_fiber_fmt_survives_lut_change(fiber_results, pv_backend):
    """ScalarColorSupport unification: the bar refresh on a LUT change
    passes the runtime fmt (previously only the contour did, so a
    ``set_fmt`` here was lost on the next colormap change)."""
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(pv_backend, results.fem, scene)

    diagram.set_fmt("%.2e")
    diagram.lut.set_preset("magma")   # triggers the bar refresh
    bar = pv_backend.plotter.scalar_bars["fiber_stress"]
    assert bar.GetLabelFormat() == "%.2e"


# =====================================================================
# Station natural coordinates — true positions vs inferred fallback
# =====================================================================
#
# The beams in the fixture run along global +x (unit length each), so
# with the default vecxz frame a fiber's world position is
# ``(x_i + (1 + ξ)/2 · L,  y_fiber,  z_fiber)`` — the x coordinate IS
# the station. ``fiber_results`` writes no station dataset (the
# pre-station file shape), so positions fall back to the uniform
# spread; ``fiber_results_stations`` writes a deliberately NON-uniform
# rule that the uniform inference cannot reproduce.

_TRUE_XI = (-0.7, 0.4)   # non-uniform 2-station rule (≠ uniform -1/+1)


@pytest.fixture
def fiber_results_stations(g, tmp_path: Path):
    """Same shape as ``fiber_results`` but WITH true station coords."""
    p0 = g.model.geometry.add_point(0.0, 0.0, 0.0, label="p0")
    p1 = g.model.geometry.add_point(1.0, 0.0, 0.0, label="p1")
    g.model.geometry.add_line(p0, p1, label="seg0")
    g.physical.add_curve(["seg0"], name="Beam")
    g.mesh.sizing.set_global_size(10.0)
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)

    line_eids = sorted(
        int(x)
        for group in fem.elements if group.element_type.dim == 1
        for x in group.ids
    )
    fibers_per_gp = 2
    fiber_y = [-0.1, 0.1]
    rows_per_beam = len(_TRUE_XI) * fibers_per_gp

    eid_arr = np.repeat(line_eids, rows_per_beam).astype(np.int64)
    gp_arr = np.tile(
        np.repeat(np.arange(len(_TRUE_XI)), fibers_per_gp), len(line_eids),
    ).astype(np.int64)
    xi_arr = np.tile(
        np.repeat(np.asarray(_TRUE_XI), fibers_per_gp), len(line_eids),
    ).astype(np.float64)
    y_arr = np.tile(np.asarray(fiber_y), len(_TRUE_XI) * len(line_eids))
    z_arr = np.zeros_like(y_arr)
    area_arr = np.ones_like(y_arr)
    mat_arr = np.ones(eid_arr.size, dtype=np.int64)
    values = np.arange(
        2 * eid_arr.size, dtype=np.float64,
    ).reshape(2, eid_arr.size)

    path = tmp_path / "fibers_stations.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="dyn", kind="transient",
            time=np.arange(2, dtype=np.float64),
        )
        w.write_fibers_group(
            sid, "partition_0", group_id="g0",
            section_tag=10, section_class="FiberSection",
            element_index=eid_arr, gp_index=gp_arr,
            y=y_arr, z=z_arr, area=area_arr, material_tag=mat_arr,
            components={"fiber_stress": values},
            station_natural_coord=xi_arr,
        )
        w.end_stage()

    results = Results.from_native(path, model=_open_model_from_h5(path))
    return results, np.asarray(line_eids), xi_arr, y_arr


def _endpoints_x_by_eid(fem) -> dict[int, tuple[float, float]]:
    """``eid → (x_i, x_j)`` for the dim-1 elements (members run along +x)."""
    nid_to_x = {
        int(n): float(c[0])
        for n, c in zip(fem.nodes.ids, np.asarray(fem.nodes.coords))
    }
    out: dict[int, tuple[float, float]] = {}
    for group in fem.elements:
        if group.element_type.dim != 1:
            continue
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for k, eid in enumerate(np.asarray(group.ids, dtype=np.int64)):
            out[int(eid)] = (
                nid_to_x[int(conn[k, 0])], nid_to_x[int(conn[k, 1])],
            )
    return out


def test_world_positions_use_true_station_coords(
    fiber_results_stations, backend,
):
    results, line_eids, xi_arr, y_arr = fiber_results_stations
    # Round-trip sanity: the slab carries what the writer was given.
    slab = results.elements.fibers.get(component="fiber_stress")
    np.testing.assert_allclose(slab.station_natural_coord, xi_arr)
    slab_eid = np.asarray(slab.element_index, dtype=np.int64)

    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(backend, results.fem, scene)

    pts = np.asarray(diagram._points.coords, dtype=np.float64)
    # Members run along +x: each fiber's x is its element's TRUE
    # station, x_i + (1 + ξ)/2 · (x_j − x_i) — not the uniform spread
    # (which would land at the member ends for a 2-station rule).
    ends = _endpoints_x_by_eid(results.fem)
    expected_x = np.array([
        ends[int(e)][0]
        + (1.0 + xi) / 2.0 * (ends[int(e)][1] - ends[int(e)][0])
        for e, xi in zip(slab_eid, xi_arr)
    ])
    np.testing.assert_allclose(pts[:, 0], expected_x, atol=1e-6)
    np.testing.assert_allclose(pts[:, 1], y_arr, atol=1e-6)
    # And the true rule is genuinely different from the uniform one.
    uniform_x = np.array([
        ends[int(e)][0] if xi < 0 else ends[int(e)][1]
        for e, xi in zip(slab_eid, xi_arr)
    ])
    assert np.abs(pts[:, 0] - uniform_x).max() > 0.01


def test_missing_station_coords_falls_back_and_warns(
    fiber_results, backend, monkeypatch,
):
    (results, line_eids, gps_per_beam, fibers_per_gp, *_) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)

    calls: list[tuple] = []
    monkeypatch.setattr(
        "apeGmsh.viewers._log.log_action",
        lambda *a, **k: calls.append((a, k)),
    )
    diagram.attach(backend, results.fem, scene)

    # The pre-station file has no coords → uniform inference, loudly.
    assert any(a[1] == "fiber_station_xi_inferred" for a, _ in calls)
    # Uniform 2-GP spread: every station lands exactly at a member end.
    pts = np.asarray(diagram._points.coords, dtype=np.float64)
    end_xs = {
        round(x, 6)
        for pair in _endpoints_x_by_eid(results.fem).values()
        for x in pair
    }
    assert set(np.round(pts[:, 0], 6).tolist()) <= end_xs
