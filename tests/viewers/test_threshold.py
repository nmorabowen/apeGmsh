"""Scalar THRESHOLD — the catalog's first region tool (ADR 0084 D1).

Threshold is not a diagram kind: it is a scalar-driven hide layer on the
existing :class:`ElementVisibility` mechanism, so these tests assert on
the two things that actually define the feature — **which cell ids end
up hidden**, and **that the set follows the time cursor** — rather than
on call counts.

Everything here runs in the default headless lane: real
``ElementVisibility`` over real ``pyvista`` grids, the real
``GeometryManager``/``PumpSet``, and thin fakes only where the
production code already takes a callback (the value reader). No Qt, no
GL, no window.

The reference mesh is a strip of four quads sharing edges::

    1---3---5---7---9      cell k spans columns k and k+1
    |c0 |c1 |c2 |c3 |      node n sits in column n // 2
    0---2---4---6---8

so a node's scalar is its column index and cell ``k`` carries the node
values ``{k, k+1}``. That makes every expected hidden set checkable by
hand, and it puts a **straddling** cell on each side of any interior
range — the case the all-nodes rule must hide.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv
from vtk import VTK_LINE, VTK_QUAD, VTK_TETRA, VTK_TRIANGLE

from apeGmsh.viewers._pump_set import PumpSet
from apeGmsh.viewers.core.element_visibility import (
    LAYER_DIM,
    LAYER_MANUAL,
    LAYER_THRESHOLD,
    ElementVisibility,
)
from apeGmsh.viewers.core.threshold_controller import (
    TOPOLOGY_GAUSS,
    ThresholdController,
    ThresholdSettings,
    cells_with_all_nodes_in_range,
    compute_hidden_mask,
)
from apeGmsh.viewers.diagrams._geometries import GeometryManager


# =====================================================================
# Meshes and fakes
# =====================================================================


def _quad_strip(n_cells: int = 4) -> pv.UnstructuredGrid:
    """The 4-quad strip from the module docstring."""
    n_cols = n_cells + 1
    pts = np.zeros((2 * n_cols, 3), dtype=np.float64)
    for col in range(n_cols):
        pts[2 * col] = (col, 0.0, 0.0)
        pts[2 * col + 1] = (col, 1.0, 0.0)
    cells = np.hstack([
        [4, 2 * k, 2 * k + 2, 2 * k + 3, 2 * k + 1] for k in range(n_cells)
    ])
    types = np.full(n_cells, VTK_QUAD, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, pts)


def _column_values(grid: pv.UnstructuredGrid, offset: float = 0.0):
    """Per-POINT scalars: node ``n`` -> ``n // 2 + offset`` (its column)."""
    return (np.arange(grid.n_points) // 2).astype(np.float64) + float(offset)


def _mixed_mesh() -> pv.UnstructuredGrid:
    """tri(0,1,2) quad(2,3,4,5) line(5,6) tet(6,7,8,9) — runs of 3/4/2/4."""
    pts = np.zeros((10, 3), dtype=np.float64)
    pts[:, 0] = np.arange(10)
    cells = np.hstack([
        [3, 0, 1, 2], [4, 2, 3, 4, 5], [2, 5, 6], [4, 6, 7, 8, 9],
    ])
    types = np.array(
        [VTK_TRIANGLE, VTK_QUAD, VTK_LINE, VTK_TETRA], dtype=np.uint8,
    )
    return pv.UnstructuredGrid(cells, types, pts)


class _Scene:
    """The two attributes the threshold refresh reads off a scene."""

    def __init__(self, grid: pv.UnstructuredGrid) -> None:
        self.grid = grid
        self.element_visibility = ElementVisibility(grid)


def _hidden(scene: _Scene) -> set:
    """The EFFECTIVE hidden cell ids, read off the composed ghost array."""
    return set(np.flatnonzero(scene.element_visibility.hidden_mask()).tolist())


class _Reader:
    """Records reads; serves ``values_by_step`` (or raises / returns None)."""

    def __init__(self, values_by_step=None, *, raises=False, missing=False):
        self._by_step = values_by_step or {}
        self._raises = raises
        self._missing = missing
        self.calls: list[tuple] = []

    def __call__(self, component, step, *, stage_id=None, topology="nodes"):
        self.calls.append((component, int(step), stage_id, topology))
        if self._raises:
            raise IndexError("List time_slice contains out-of-range indices")
        if self._missing:
            return None
        return self._by_step.get(int(step))


def _controller(reader, **kw) -> tuple[ThresholdController, list]:
    """Controller with a recording ``on_failure`` (never the global sink)."""
    seen: list = []
    ctrl = ThresholdController(
        read_values=reader,
        on_failure=lambda action, exc, **p: seen.append((action, exc)),
        **kw,
    )
    return ctrl, seen


# =====================================================================
# The cell rule — ALL NODES IN RANGE
# =====================================================================


def test_a_cell_straddling_the_boundary_is_hidden():
    """The ratified rule: EVERY node must be inside ``[lo, hi]``.

    With ``[1, 3]`` on the strip, cell 0 carries ``{0, 1}`` and cell 3
    carries ``{3, 4}`` — each has exactly ONE node outside, and each
    must hide. Nothing is clipped; a partly-in cell is simply gone.
    """
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, _column_values(grid), ThresholdSettings("c", 1.0, 3.0),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 3}


def test_all_in_and_all_out_are_the_two_saturating_ends():
    grid = _quad_strip()
    vals = _column_values(grid)
    none_hidden = compute_hidden_mask(
        grid, vals, ThresholdSettings("c", 0.0, 4.0),
    )
    all_hidden = compute_hidden_mask(
        grid, vals, ThresholdSettings("c", 10.0, 20.0),
    )
    assert set(np.flatnonzero(none_hidden).tolist()) == set()
    assert set(np.flatnonzero(all_hidden).tolist()) == {0, 1, 2, 3}


def test_lo_equals_hi_keeps_only_cells_whose_nodes_all_equal_it():
    """A degenerate range is legal. No strip cell has both nodes on the
    same column, so an exact-equality range hides everything."""
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, _column_values(grid), ThresholdSettings("c", 2.0, 2.0),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 1, 2, 3}


def test_lo_greater_than_hi_is_an_empty_range_and_hides_everything():
    """Sliders can cross mid-drag — that must hide, not raise."""
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, _column_values(grid), ThresholdSettings("c", 3.0, 1.0),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 1, 2, 3}


def test_nan_is_out_of_range_and_hides_its_cells():
    """NaN must NOT silently count as in-range.

    Node 4 (column 2) is shared by cells 1 and 2, so NaN there hides
    exactly those two even though the range covers every real value.
    """
    grid = _quad_strip()
    vals = _column_values(grid)
    vals[4] = np.nan
    hide = compute_hidden_mask(
        grid, vals, ThresholdSettings("c", 0.0, 4.0),
    )
    assert set(np.flatnonzero(hide).tolist()) == {1, 2}


def test_empty_mesh_yields_an_empty_mask():
    grid = pv.UnstructuredGrid(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint8),
        np.zeros((0, 3)),
    )
    hide = compute_hidden_mask(
        grid, np.zeros(0), ThresholdSettings("c", 0.0, 1.0),
    )
    assert hide.shape == (0,)


# =====================================================================
# The vectorised reduction
# =====================================================================


def test_reduction_handles_mixed_cell_types():
    """Runs of 3 / 4 / 2 / 4 reduce independently — the offsets carry
    the cell sizes, so no branch per cell type is needed."""
    grid = _mixed_mesh()
    in_range = np.ones(10, dtype=bool)
    in_range[4] = False          # node 4 belongs to the QUAD only
    keep = cells_with_all_nodes_in_range(grid, in_range)
    assert keep.tolist() == [True, False, True, True]

    in_range = np.ones(10, dtype=bool)
    in_range[6] = False          # node 6 is shared by the LINE and the TET
    keep = cells_with_all_nodes_in_range(grid, in_range)
    assert keep.tolist() == [True, True, False, False]


def test_reduction_agrees_with_a_per_cell_python_oracle():
    """The vectorised reduceat must equal the obvious slow loop.

    Cross-checked on a mixed mesh over many random masks — this is the
    guard against an off-by-one in the offsets (the classic reduceat
    trap is the trailing offset, which would silently merge the last
    two cells)."""
    grid = _mixed_mesh()
    rng = np.random.default_rng(0)
    conn, offs = grid.cell_connectivity, grid.offset
    for _ in range(200):
        in_range = rng.random(grid.n_points) > 0.35
        oracle = [
            bool(in_range[conn[offs[k]:offs[k + 1]]].all())
            for k in range(grid.n_cells)
        ]
        assert cells_with_all_nodes_in_range(grid, in_range).tolist() == oracle


def test_reduction_is_correct_on_a_hex_mesh_of_realistic_size():
    """Sanity at scale, and a guard that 8-node runs reduce correctly."""
    grid = pv.ImageData(dimensions=(21, 21, 21)).cast_to_unstructured_grid()
    assert grid.n_cells == 8000
    # Keep only cells entirely below the mid-plane in z.
    z = grid.points[:, 2]
    keep = cells_with_all_nodes_in_range(grid, z <= 10.0)
    # A hex spanning z in [9,10] is in; one spanning [10,11] straddles
    # and is out. 10 layers of 20x20 cells survive.
    assert int(keep.sum()) == 10 * 20 * 20


# =====================================================================
# Cell (gauss) components
# =====================================================================


def test_cell_data_component_is_tested_directly_without_reduction():
    grid = _quad_strip()
    cell_vals = np.arange(grid.n_cells, dtype=np.float64)   # 0,1,2,3
    hide = compute_hidden_mask(
        grid, cell_vals,
        ThresholdSettings("s_xx", 1.0, 2.0, topology=TOPOLOGY_GAUSS),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 3}


def test_cell_topology_nan_hides_only_its_own_cell():
    grid = _quad_strip()
    cell_vals = np.array([0.0, np.nan, 2.0, 3.0])
    hide = compute_hidden_mask(
        grid, cell_vals,
        ThresholdSettings("s_xx", 0.0, 5.0, topology=TOPOLOGY_GAUSS),
    )
    assert set(np.flatnonzero(hide).tolist()) == {1}


# =====================================================================
# Controller — apply / clear against a real ElementVisibility
# =====================================================================


def test_refresh_applies_the_threshold_layer_to_the_scene():
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))
    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)
    assert _hidden(scene) == {0, 3}


def test_clearing_the_threshold_restores_exactly_its_own_cells():
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))
    geom = SimpleNamespace(id="g1")

    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    assert ctrl.settings_for("g1") == ThresholdSettings("u", 1.0, 3.0)
    ctrl.refresh(geom, scene, 0)
    assert _hidden(scene) == {0, 3}

    ctrl.clear_threshold("g1")
    assert ctrl.settings_for("g1") is None
    ctrl.refresh(geom, scene, 0)
    assert _hidden(scene) == set()


def test_a_disabled_geometry_is_untouched_by_another_geometrys_threshold():
    """State is keyed per geometry — one geometry's range must not
    reach another geometry's scene."""
    grid = _quad_strip()
    scene_a, scene_b = _Scene(grid.copy()), _Scene(grid.copy())
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))
    ctrl.set_threshold("gA", component="u", lo=1.0, hi=3.0)

    ctrl.refresh(SimpleNamespace(id="gA"), scene_a, 0)
    ctrl.refresh(SimpleNamespace(id="gB"), scene_b, 0)
    assert _hidden(scene_a) == {0, 3}
    assert _hidden(scene_b) == set()


def test_needs_refresh_is_false_until_a_threshold_exists_and_after_it_is_gone():
    """The pump's zero-cost gate, including the take-down pass: after the
    last clear it must stay True for exactly one more refresh."""
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))
    geom = SimpleNamespace(id="g1")
    assert ctrl.needs_refresh() is False

    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    assert ctrl.needs_refresh() is True
    ctrl.refresh(geom, scene, 0)

    ctrl.clear_threshold("g1")
    assert ctrl.needs_refresh() is True        # layer still applied
    ctrl.refresh(geom, scene, 0)
    assert ctrl.needs_refresh() is False
    assert _hidden(scene) == set()


# =====================================================================
# Failure paths (ADR 0084 D4 — loud, never a bare except)
# =====================================================================


def test_a_missing_component_is_loud_and_leaves_the_layer_down():
    """A component absent from the current stage must not crash the
    scrub, must be reported, and must leave the cells VISIBLE rather
    than frozen at a stale mask that lies about the current step."""
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, seen = _controller(_Reader(missing=True))
    ctrl.set_threshold("g1", component="ghost", lo=1.0, hi=3.0)
    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)

    assert [action for action, _exc in seen] == ["threshold"]
    assert _hidden(scene) == set()


def test_a_raising_slab_read_is_loud_and_leaves_the_layer_down():
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, seen = _controller(_Reader(raises=True))
    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)

    assert [action for action, _exc in seen] == ["threshold"]
    assert isinstance(seen[0][1], IndexError)
    assert _hidden(scene) == set()


def test_a_failing_read_takes_down_a_previously_applied_mask():
    """The stale-mask guard: a good step then a bad one must not leave
    the good step's hidden set on screen."""
    grid = _quad_strip()
    scene = _Scene(grid)
    reader = _Reader({0: _column_values(grid)})
    ctrl, seen = _controller(reader)
    geom = SimpleNamespace(id="g1")
    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    ctrl.refresh(geom, scene, 0)
    assert _hidden(scene) == {0, 3}

    reader._missing = True
    ctrl.refresh(geom, scene, 1)
    assert _hidden(scene) == set()
    assert [action for action, _exc in seen] == ["threshold"]


# =====================================================================
# Layer composition — the whole reason threshold is a LAYER
# =====================================================================


def test_threshold_composes_with_manual_and_dim_layers():
    """The effective hidden set is the OR of all three, and clearing the
    threshold reveals ONLY the cells the threshold hid.

    This is the exact clobbering the layered design exists to prevent:
    a single ghost writer would have made ``clear`` reveal the user's
    manual hide too.
    """
    grid = _quad_strip()
    scene = _Scene(grid)
    ev = scene.element_visibility
    geom = SimpleNamespace(id="g1")
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))

    ev.hide([1])                                     # manual: cell 1
    dim_mask = np.zeros(grid.n_cells, dtype=bool)
    dim_mask[2] = True                               # dim filter: cell 2
    ev.set_layer(LAYER_DIM, dim_mask)
    assert _hidden(scene) == {1, 2}

    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    ctrl.refresh(geom, scene, 0)                     # threshold: cells 0, 3
    assert _hidden(scene) == {0, 1, 2, 3}

    ctrl.clear_threshold("g1")
    ctrl.refresh(geom, scene, 0)
    assert _hidden(scene) == {1, 2}                  # manual + dim survive
    assert set(np.flatnonzero(ev._layers[LAYER_MANUAL]).tolist()) == {1}
    assert LAYER_THRESHOLD not in ev._layers


def test_show_all_does_not_reveal_thresholded_cells():
    """``show_all`` clears the MANUAL layer only — the threshold's cells
    stay hidden, mirroring the dim filter's guarantee."""
    grid = _quad_strip()
    scene = _Scene(grid)
    ctrl, _ = _controller(_Reader({0: _column_values(grid)}))
    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)
    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)

    scene.element_visibility.hide([1])
    assert _hidden(scene) == {0, 1, 3}
    scene.element_visibility.show_all()
    assert _hidden(scene) == {0, 3}


# =====================================================================
# LIVE — the mask follows the time step, through the STEP pump
# =====================================================================


def _director(geom_mgr, scenes, *, step_index=0, local_steps=None,
              active_step=None):
    """The director surface the STEP pump uses (see test_pump_set)."""
    local_steps = local_steps or {}
    return SimpleNamespace(
        geometries=geom_mgr,
        registry=SimpleNamespace(diagrams=lambda: []),
        step_index=step_index,
        local_step_for_stage=lambda sid: local_steps[sid],
        local_step_for_active_stage=lambda: (
            step_index if active_step is None else active_step
        ),
        scene_for=lambda geom: scenes.get(geom.id),
    )


def _pumps(director, thresholds, scene=None) -> PumpSet:
    return PumpSet(
        director=director,
        scene=scene,
        read_deform_field=lambda *a, **k: None,
        render_geometries=lambda: list(director.geometries.geometries),
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
        thresholds=thresholds,
    )


# Node scalars rise by one column per step, so the kept band slides
# right and the hidden set is different at every step.
_STEP_VALUES = {
    step: (np.arange(10) // 2).astype(np.float64) + step
    for step in range(3)
}
# Hand-checked against range [1, 3]: cell k carries {k+step, k+1+step}.
_EXPECTED_HIDDEN = {0: {0, 3}, 1: {2, 3}, 2: {1, 2, 3}}


def test_scrubbing_moves_the_thresholded_region():
    """LIVE: one STEP pump per tick, and the hidden set CHANGES to the
    hand-checked set for that step."""
    grid = _quad_strip()
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    scene = _Scene(grid)
    ctrl, _ = _controller(_Reader(_STEP_VALUES))
    ctrl.set_threshold(gid, component="u", lo=1.0, hi=3.0)

    for step, expected in _EXPECTED_HIDDEN.items():
        director = _director(geom_mgr, {gid: scene}, step_index=step)
        _pumps(director, ctrl).pump_step(None)
        assert _hidden(scene) == expected, f"step {step}"


def test_the_step_pump_skips_the_refresh_when_no_threshold_is_set():
    """Zero cost when off — the reader is never called."""
    grid = _quad_strip()
    geom_mgr = GeometryManager()
    scene = _Scene(grid)
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)

    director = _director(geom_mgr, {geom_mgr.active_id: scene}, step_index=1)
    _pumps(director, ctrl).pump_step(None)

    assert reader.calls == []
    assert _hidden(scene) == set()


def test_a_layer_scoped_step_pump_does_not_refresh_thresholds():
    """Geometry-level work belongs to the FULL pump; a single diagram's
    re-attach must not re-run every geometry's threshold."""
    grid = _quad_strip()
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    scene = _Scene(grid)
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)
    ctrl.set_threshold(gid, component="u", lo=1.0, hi=3.0)

    director = _director(geom_mgr, {gid: scene}, step_index=0)
    layer = SimpleNamespace(update_to_step=lambda s: None)
    _pumps(director, ctrl).pump_step(layer)

    assert reader.calls == []
    assert _hidden(scene) == set()


def test_unpinned_geometry_reads_the_translated_local_step():
    """Combined mode: the pump must pass ``local_step_for_active_stage``,
    NOT the raw global cursor — the defect ADR 0084 fixed for STEP and
    again for DEFORM."""
    grid = _quad_strip()
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    scene = _Scene(grid)
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)
    ctrl.set_threshold(gid, component="u", lo=1.0, hi=3.0)

    # Global cursor 17 sits at local step 2 of the real active stage.
    director = _director(
        geom_mgr, {gid: scene}, step_index=17, active_step=2,
    )
    _pumps(director, ctrl).pump_step(None)

    assert [(c[1], c[2]) for c in reader.calls] == [(2, None)]
    assert _hidden(scene) == _EXPECTED_HIDDEN[2]


def test_pinned_geometry_reads_its_own_stage_at_the_clamped_local_step():
    """ADR 0058 S3b: a stage-PINNED geometry thresholds through its
    pinned stage, clamped into that stage's range, and the read is
    scoped to that stage id."""
    grid = _quad_strip()
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    geom_mgr.set_stage_pin(gid, "stage-a")
    scene = _Scene(grid)
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)
    ctrl.set_threshold(gid, component="u", lo=1.0, hi=3.0)

    director = _director(
        geom_mgr, {gid: scene},
        step_index=99, active_step=2, local_steps={"stage-a": 1},
    )
    _pumps(director, ctrl).pump_step(None)

    # Pinned wins: local step 1 of stage-a, not the active stage's 2.
    assert [(c[1], c[2]) for c in reader.calls] == [(1, "stage-a")]
    assert _hidden(scene) == _EXPECTED_HIDDEN[1]


def test_each_geometry_thresholds_its_own_scene_at_its_own_step():
    """Per-geometry by construction: two geometries, two scenes, one
    pinned and one not — each gets its own hidden set."""
    geom_mgr = GeometryManager()
    gid_a = geom_mgr.active_id
    geom_b = geom_mgr.add(name="B")
    geom_mgr.set_stage_pin(geom_b.id, "stage-a")

    scene_a, scene_b = _Scene(_quad_strip()), _Scene(_quad_strip())
    ctrl, _ = _controller(_Reader(_STEP_VALUES))
    ctrl.set_threshold(gid_a, component="u", lo=1.0, hi=3.0)
    ctrl.set_threshold(geom_b.id, component="u", lo=1.0, hi=3.0)

    director = _director(
        geom_mgr, {gid_a: scene_a, geom_b.id: scene_b},
        step_index=0, active_step=0, local_steps={"stage-a": 2},
    )
    _pumps(director, ctrl).pump_step(None)

    assert _hidden(scene_a) == _EXPECTED_HIDDEN[0]
    assert _hidden(scene_b) == _EXPECTED_HIDDEN[2]


@pytest.mark.allow_pump_failures
def test_a_failing_threshold_is_loud_through_the_pump_and_keeps_scrubbing(
    pump_failures,
):
    """One geometry's unreadable component must not strand the rest of
    the scrub, and must surface through the ``_failures`` registry."""
    geom_mgr = GeometryManager()
    gid_a = geom_mgr.active_id
    geom_b = geom_mgr.add(name="B")
    scene_a, scene_b = _Scene(_quad_strip()), _Scene(_quad_strip())

    class _HalfBad(_Reader):
        def __call__(self, component, step, *, stage_id=None,
                     topology="nodes"):
            if component == "ghost":
                raise RuntimeError("no such component")
            return super().__call__(
                component, step, stage_id=stage_id, topology=topology,
            )

    ctrl = ThresholdController(read_values=_HalfBad(_STEP_VALUES))
    ctrl.set_threshold(gid_a, component="ghost", lo=1.0, hi=3.0)
    ctrl.set_threshold(geom_b.id, component="u", lo=1.0, hi=3.0)

    director = _director(
        geom_mgr, {gid_a: scene_a, geom_b.id: scene_b}, step_index=0,
    )
    _pumps(director, ctrl).pump_step(None)

    assert [name for name, _exc in pump_failures.failures] == [
        "pump.threshold",
    ]
    assert _hidden(scene_a) == set()                 # failed -> visible
    assert _hidden(scene_b) == _EXPECTED_HIDDEN[0]   # healthy -> applied
