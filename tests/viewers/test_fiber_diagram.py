"""FiberSectionDiagram — attach + step + side-panel data accessors.

Builds a 2-element beam, writes synthetic fiber data via NativeWriter
(2 GPs per beam, 4 fibers per GP), then verifies:

* The 3-D dot cloud has one point per fiber.
* Step updates mutate the scalar in place.
* ``available_gps()`` returns all (eid, gp) pairs.
* ``read_section_at_gp`` returns the right (y, z, area, value) tuple
  for a specific (eid, gp) pair, with values from the active step.
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
    FiberSectionDiagram,
    FiberSectionStyle,
    SlabSelector,
)
from apeGmsh.viewers.scene.fem_scene import build_fem_scene


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
        Results.from_native(path),
        line_eids, gps_per_beam, fibers_per_gp,
        eid_arr, gp_arr, y_arr, z_arr, area_arr, values,
    )


@pytest.fixture
def headless_plotter():
    plotter = pv.Plotter(off_screen=True)
    yield plotter
    plotter.close()


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
# Attach
# =====================================================================

def test_attach_requires_scene(fiber_results, headless_plotter):
    results = fiber_results[0]
    diagram = FiberSectionDiagram(_make_spec(), results)
    with pytest.raises(RuntimeError, match="FEMSceneData"):
        diagram.attach(headless_plotter, results.fem)


def test_attach_builds_dot_cloud(fiber_results, headless_plotter):
    (results, line_eids, gps_per_beam, fibers_per_gp,
     *_, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    n_total = len(line_eids) * gps_per_beam * fibers_per_gp
    assert diagram._cloud is not None
    assert diagram._cloud.n_points == n_total
    assert diagram._scalar_array is not None
    np.testing.assert_array_equal(
        np.asarray(diagram._scalar_array), values[0],
    )


def test_available_gps_lists_all_pairs(fiber_results, headless_plotter):
    (results, line_eids, gps_per_beam, *_) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    expected = sorted(
        (int(eid), gp) for eid in line_eids for gp in range(gps_per_beam)
    )
    assert diagram.available_gps() == expected


# =====================================================================
# Step update
# =====================================================================

def test_step_update_changes_scalars(fiber_results, headless_plotter):
    results, _, _, _, *_, values = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    diagram.update_to_step(2)
    np.testing.assert_array_equal(
        np.asarray(diagram._scalar_array), values[2],
    )


# =====================================================================
# Side-panel data accessor
# =====================================================================

def test_read_section_at_gp_returns_correct_subset(
    fiber_results, headless_plotter,
):
    (results, line_eids, gps_per_beam, fibers_per_gp,
     eid_arr, gp_arr, y_arr, z_arr, area_arr, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

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


def test_read_section_at_gp_uses_step(fiber_results, headless_plotter):
    (results, line_eids, *_, values) = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    data_step0 = diagram.read_section_at_gp(int(line_eids[0]), 0, step_index=0)
    data_step2 = diagram.read_section_at_gp(int(line_eids[0]), 0, step_index=2)
    assert data_step0 is not None and data_step2 is not None
    assert not np.array_equal(data_step0[3], data_step2[3])


def test_read_section_at_gp_unknown_returns_none(
    fiber_results, headless_plotter,
):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)

    assert diagram.read_section_at_gp(99999, 0, step_index=0) is None


# =====================================================================
# In-place mutation
# =====================================================================

def test_actor_identity_stable_across_steps(fiber_results, headless_plotter):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)
    initial_actor = diagram._actor
    initial_cloud = diagram._cloud
    initial_scalar = diagram._scalar_array

    for step in range(3):
        diagram.update_to_step(step)

    assert diagram._actor is initial_actor
    assert diagram._cloud is initial_cloud
    assert diagram._scalar_array is initial_scalar


# =====================================================================
# Detach
# =====================================================================

def test_detach_clears_state(fiber_results, headless_plotter):
    results, *_ = fiber_results
    scene = build_fem_scene(results.fem)
    diagram = FiberSectionDiagram(_make_spec(), results)
    diagram.attach(headless_plotter, results.fem, scene)
    diagram.detach()
    assert diagram._cloud is None
    assert diagram._actor is None
    assert diagram._slab_y is None
    assert not diagram.is_attached
