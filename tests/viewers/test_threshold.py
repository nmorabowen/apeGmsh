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
    TOPOLOGY_NODES,
    SceneValueReader,
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


# =====================================================================
# SceneValueReader — ROW ALIGNMENT (the risky part)
# =====================================================================
#
# A slab's row order is NOT the scene's row order, and getting that
# wrong paints a plausible picture of the WRONG cells rather than
# raising. So the fixture below is built so a positional zip produces a
# DIFFERENT, checkable answer — a test that only asserted "some cells
# hid" would pass with any permutation and prove nothing.
#
# Scene rows carry deliberately scrambled, non-contiguous FEM ids:
#
#   scene point row r  ->  node id  1 + 3·((7·r) % 10)
#   scene cell  row c  ->  elem id  5 + 4·((3·c) % 4)
#
# and both slabs present their rows in SORTED id order, the way a
# reader normally hands them over. Node id 13 and element id 13 both
# exist on purpose: the two lookups must not reach each other's table.

_SCRAMBLED_NODE_IDS = np.array(
    [1 + 3 * ((7 * r) % 10) for r in range(10)], dtype=np.int64,
)
_SCRAMBLED_ELEMENT_IDS = np.array(
    [5 + 4 * ((3 * c) % 4) for c in range(4)], dtype=np.int64,
)


def test_the_alignment_fixture_is_actually_scrambled():
    """Guard the guard: if these ever became row-ordered or contiguous
    the alignment tests below would silently stop testing anything."""
    assert _SCRAMBLED_NODE_IDS.tolist() == [
        1, 22, 13, 4, 25, 16, 7, 28, 19, 10,
    ]
    assert _SCRAMBLED_ELEMENT_IDS.tolist() == [5, 17, 13, 9]
    for ids in (_SCRAMBLED_NODE_IDS, _SCRAMBLED_ELEMENT_IDS):
        assert ids.tolist() != sorted(ids.tolist())      # not row order
        assert int(np.diff(np.sort(ids)).max()) > 1      # not contiguous
    # …and the same integer names a node AND an element.
    assert set(_SCRAMBLED_NODE_IDS.tolist()) & set(
        _SCRAMBLED_ELEMENT_IDS.tolist()
    ) == {13}


def _scrambled_scene():
    """A real ``FEMSceneData`` over the 4-quad strip with scrambled ids."""
    from apeGmsh.viewers.scene.fem_scene import FEMSceneData

    grid = _quad_strip()
    node_ids = _SCRAMBLED_NODE_IDS
    eids = _SCRAMBLED_ELEMENT_IDS
    scene = FEMSceneData(
        grid=grid,
        node_ids=node_ids,
        node_id_to_idx={int(n): r for r, n in enumerate(node_ids)},
        cell_to_element_id=eids,
        element_id_to_cell={int(e): c for c, e in enumerate(eids)},
        model_diagonal=1.0,
        cell_dim=np.full(grid.n_cells, 2, dtype=np.int8),
        reference_points=np.asarray(grid.points).copy(),
    )
    scene.element_visibility = ElementVisibility(grid)
    return scene


def _sorted_node_slab_values() -> tuple:
    """``(ids, values)`` a nodal reader would hand over, sorted by id.

    The value each id carries is its scene row's COLUMN — the same
    ``row // 2`` scalar the rest of this file uses — so the expected
    hidden set is hand-checkable against the module docstring's strip.
    """
    row_of = {int(n): r for r, n in enumerate(_SCRAMBLED_NODE_IDS)}
    ids = np.sort(_SCRAMBLED_NODE_IDS)
    values = np.array([row_of[int(n)] // 2 for n in ids], dtype=np.float64)
    return ids, values


def _sorted_gauss_slab_values() -> tuple:
    """``(element_index, values)`` for TWO gauss points per element,
    sorted by element id. Each element's value is its CELL row."""
    cell_of = {int(e): c for c, e in enumerate(_SCRAMBLED_ELEMENT_IDS)}
    eids = np.repeat(np.sort(_SCRAMBLED_ELEMENT_IDS), 2)
    values = np.array([cell_of[int(e)] for e in eids], dtype=np.float64)
    return eids, values


def test_the_hand_checked_slab_rows_are_what_we_think_they_are():
    """The two arrays every alignment assertion below derives from,
    written out so the expected hidden sets can be checked by eye."""
    ids, values = _sorted_node_slab_values()
    assert ids.tolist() == [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    assert values.tolist() == [0, 1, 3, 4, 1, 2, 4, 0, 2, 3]

    eids, gvalues = _sorted_gauss_slab_values()
    assert eids.tolist() == [5, 5, 9, 9, 13, 13, 17, 17]
    assert gvalues.tolist() == [0, 0, 3, 3, 2, 2, 1, 1]


class _FakeNodes:
    """``results.nodes`` — serves the slab in SORTED id order."""

    def __init__(self, ids, values_by_step, component="u") -> None:
        self._ids = np.asarray(ids, dtype=np.int64)
        self._values = np.asarray(values_by_step, dtype=np.float64)
        self._component = component

    def available_components(self):
        return [self._component]

    def get(self, *, component, time=None, ids=None):
        from apeGmsh.results._slabs import NodeSlab
        if component != self._component:
            return NodeSlab(
                component, np.zeros((0, 0)),
                np.zeros(0, dtype=np.int64), np.zeros(0),
            )
        vals = (
            self._values if time is None
            else self._values[[int(t) for t in time]]
        )
        return NodeSlab(
            component, vals, self._ids,
            np.arange(vals.shape[0], dtype=np.float64),
        )


class _FakeGauss:
    """``results.elements.gauss`` — sorted by element id, 2 GP each."""

    def __init__(self, element_index, values_by_step,
                 component="s_xx") -> None:
        self._eidx = np.asarray(element_index, dtype=np.int64)
        self._values = np.asarray(values_by_step, dtype=np.float64)
        self._component = component

    def available_components(self):
        return [self._component]

    def get(self, *, component, time=None, ids=None):
        from apeGmsh.results._slabs import GaussSlab
        if component != self._component:
            return GaussSlab(
                component, np.zeros((0, 0)),
                np.zeros(0, dtype=np.int64), np.zeros((0, 2)), None,
                np.zeros(0),
            )
        vals = (
            self._values if time is None
            else self._values[[int(t) for t in time]]
        )
        return GaussSlab(
            component, vals, self._eidx,
            np.zeros((self._eidx.size, 2)), None,
            np.arange(vals.shape[0], dtype=np.float64),
        )


def _fake_director(*, nodes, gauss, with_store=False):
    """A director surface carrying the two composites the reader reads.

    ``with_store`` swaps in a REAL :class:`VisualDataStore`, the
    production hot path (a scrub slices a cached full-time row);
    ``False`` exercises the per-step fallback. Both must agree.
    """
    scoped = SimpleNamespace(
        nodes=nodes, elements=SimpleNamespace(gauss=gauss),
    )
    store = None
    if with_store:
        from apeGmsh.viewers.diagrams._visual_store import VisualDataStore
        store = VisualDataStore()
    return SimpleNamespace(
        results=SimpleNamespace(stage=lambda _sid: scoped),
        stage_id="stage-0",
        visual_store=store,
    )


@pytest.mark.parametrize("with_store", [False, True])
def test_nodal_values_land_on_the_scene_row_of_their_fem_id(with_store):
    """The nodal half of row alignment, asserted on EXACT hidden cells.

    Slab row order is sorted-by-id; scene row order is scrambled, so
    the reader must scatter by id. With ``[1, 3]`` on the strip that
    leaves exactly cells 0 and 3 hidden (each straddles the boundary) —
    a positional zip would hide all four, which the next test pins.
    Step 1 shifts every value by +10 so reading the wrong STEP is
    visible too.
    """
    _ids, values = _sorted_node_slab_values()
    scene = _scrambled_scene()
    director = _fake_director(
        nodes=_FakeNodes(
            np.sort(_SCRAMBLED_NODE_IDS),
            np.stack([values, values + 10.0]),
        ),
        gauss=_FakeGauss(_sorted_gauss_slab_values()[0], np.zeros((2, 8))),
        with_store=with_store,
    )
    ctrl, seen = _controller(SceneValueReader(director=director, scene=scene))
    geom = SimpleNamespace(id="g1")
    ctrl.set_threshold("g1", component="u", lo=1.0, hi=3.0)

    ctrl.refresh(geom, scene, 0)
    assert _hidden(scene) == {0, 3}
    ctrl.refresh(geom, scene, 1)
    assert _hidden(scene) == {0, 1, 2, 3}
    assert seen == []


def test_a_positional_zip_of_the_nodal_slab_would_hide_a_different_set():
    """Proof the alignment test discriminates.

    The bug the reader must not have: take the slab's values in SLAB
    order and drop them onto scene rows 0..N-1. On this fixture that
    hides every cell instead of two, so the assertion above cannot pass
    with a wrong permutation.
    """
    _ids, values = _sorted_node_slab_values()
    zipped = compute_hidden_mask(
        _quad_strip(), values, ThresholdSettings("u", 1.0, 3.0),
    )
    assert set(np.flatnonzero(zipped).tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize("with_store", [False, True])
def test_gauss_values_land_on_the_cell_row_of_their_element_id(with_store):
    """The cell half of row alignment, asserted on EXACT hidden cells.

    Each element's gauss points carry that element's CELL row as their
    value, so ``[1, 2]`` must keep cells 1 and 2 and hide 0 and 3. Read
    in slab order instead (elements 5, 9, 13, 17 → values 0, 3, 2, 1)
    the kept pair would be cells 2 and 3.
    """
    eidx, gvalues = _sorted_gauss_slab_values()
    scene = _scrambled_scene()
    director = _fake_director(
        nodes=_FakeNodes(np.sort(_SCRAMBLED_NODE_IDS), np.zeros((1, 10))),
        gauss=_FakeGauss(eidx, gvalues[None, :]),
        with_store=with_store,
    )
    ctrl, seen = _controller(SceneValueReader(director=director, scene=scene))
    ctrl.set_threshold(
        "g1", component="s_xx", lo=1.0, hi=2.0, topology=TOPOLOGY_GAUSS,
    )

    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)
    assert _hidden(scene) == {0, 3}
    assert seen == []


def test_a_cells_gauss_points_are_averaged():
    """Several slab rows land on one cell — their mean is its value.

    Element 5 (cell 0) gets 0 and 4 → mean 2, inside ``[1, 3]``, so
    cell 0 is KEPT even though one of its gauss points is outside.
    Every other element stays at 10 and is hidden.
    """
    scene = _scrambled_scene()
    eidx, _ = _sorted_gauss_slab_values()          # [5,5,9,9,13,13,17,17]
    values = np.full(eidx.size, 10.0)
    values[0], values[1] = 0.0, 4.0
    director = _fake_director(
        nodes=_FakeNodes(np.sort(_SCRAMBLED_NODE_IDS), np.zeros((1, 10))),
        gauss=_FakeGauss(eidx, values[None, :]),
    )
    ctrl, _ = _controller(SceneValueReader(director=director, scene=scene))
    ctrl.set_threshold(
        "g1", component="s_xx", lo=1.0, hi=3.0, topology=TOPOLOGY_GAUSS,
    )

    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)
    assert _hidden(scene) == {1, 2, 3}


def test_a_cell_no_slab_row_reaches_is_hidden_not_silently_kept():
    """An element the component was not recorded for has no value, and
    "no value" is not evidence that the cell belongs in the kept set."""
    scene = _scrambled_scene()
    eidx, gvalues = _sorted_gauss_slab_values()
    keep = eidx != _SCRAMBLED_ELEMENT_IDS[2]       # drop cell 2's rows
    director = _fake_director(
        nodes=_FakeNodes(np.sort(_SCRAMBLED_NODE_IDS), np.zeros((1, 10))),
        gauss=_FakeGauss(eidx[keep], gvalues[keep][None, :]),
    )
    ctrl, _ = _controller(SceneValueReader(director=director, scene=scene))
    ctrl.set_threshold(
        "g1", component="s_xx", lo=0.0, hi=9.0, topology=TOPOLOGY_GAUSS,
    )

    ctrl.refresh(SimpleNamespace(id="g1"), scene, 0)
    assert _hidden(scene) == {2}


def test_topology_is_honoured_never_guessed():
    """The same component name under BOTH tables must resolve by the
    caller's topology, not by whichever the reader tries first.

    ``both`` is flat 2.0 on nodes and flat 7.0 on gauss, so the value
    that comes back names the table it came from. There is no
    ``is_nodal`` flag anywhere to fall back on — this is why
    ``ThresholdSettings.topology`` exists and is persisted.
    """
    scene = _scrambled_scene()
    director = _fake_director(
        nodes=_FakeNodes(
            np.sort(_SCRAMBLED_NODE_IDS), np.full((1, 10), 2.0),
            component="both",
        ),
        gauss=_FakeGauss(
            _sorted_gauss_slab_values()[0], np.full((1, 8), 7.0),
            component="both",
        ),
    )
    reader = SceneValueReader(director=director, scene=scene)

    nodal = reader("both", 0, topology=TOPOLOGY_NODES)
    gauss = reader("both", 0, topology=TOPOLOGY_GAUSS)
    assert nodal.shape == (10,) and np.all(nodal == 2.0)
    assert gauss.shape == (4,) and np.all(gauss == 7.0)


def test_a_component_absent_from_the_stage_reads_as_none():
    """``None``, not an exception and not zeros — the controller turns
    it into the loud "not available" report and takes the layer down."""
    scene = _scrambled_scene()
    director = _fake_director(
        nodes=_FakeNodes(np.sort(_SCRAMBLED_NODE_IDS), np.zeros((1, 10))),
        gauss=_FakeGauss(_sorted_gauss_slab_values()[0], np.zeros((1, 8))),
    )
    reader = SceneValueReader(director=director, scene=scene)
    assert reader("ghost", 0) is None
    assert reader("ghost", 0, topology=TOPOLOGY_GAUSS) is None


def test_an_unpinned_read_falls_back_to_the_directors_active_stage():
    """``stage_id=None`` is an UNPINNED geometry: the pump already
    translated the cursor onto the ACTIVE stage, so the read scopes
    there. A pinned read scopes to the pin instead."""
    scene = _scrambled_scene()
    _ids, values = _sorted_node_slab_values()
    director = _fake_director(
        nodes=_FakeNodes(np.sort(_SCRAMBLED_NODE_IDS), values[None, :]),
        gauss=_FakeGauss(_sorted_gauss_slab_values()[0], np.zeros((1, 8))),
    )
    scoped = director.results.stage("x")
    asked: list = []
    director.results = SimpleNamespace(
        stage=lambda sid: (asked.append(sid), scoped)[1],
    )
    reader = SceneValueReader(director=director, scene=scene)

    assert reader("u", 0) is not None
    assert reader("u", 0, stage_id="stage-9") is not None
    assert asked == ["stage-0", "stage-9"]


# =====================================================================
# Seeding on scene materialization
# =====================================================================


def test_a_scene_materialized_mid_session_wears_the_threshold_at_once():
    """``_materialize_scene`` seeds the layer through this seam.

    A geometry whose scene appears mid-session (a new geometry, a
    visibility flip) must render filtered from its FIRST frame. Without
    the seed it shows every cell until the next STEP moves the cursor —
    the same gap ``LAYER_DIM`` and ``LAYER_STAGE`` are already seeded
    to close.
    """
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    ctrl, _ = _controller(_Reader(_STEP_VALUES))
    ctrl.set_threshold(gid, component="u", lo=1.0, hi=3.0)
    director = _director(geom_mgr, {}, step_index=1)
    pumps = _pumps(director, ctrl)

    fresh = _Scene(_quad_strip())            # freshly cloned, unfiltered
    assert _hidden(fresh) == set()

    pumps.refresh_threshold_for(geom_mgr.active, fresh)
    assert _hidden(fresh) == _EXPECTED_HIDDEN[1]


def test_materialization_costs_nothing_when_no_threshold_is_set():
    geom_mgr = GeometryManager()
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)
    director = _director(geom_mgr, {}, step_index=1)

    fresh = _Scene(_quad_strip())
    _pumps(director, ctrl).refresh_threshold_for(geom_mgr.active, fresh)

    assert reader.calls == []
    assert _hidden(fresh) == set()


def test_a_materialized_pinned_geometry_seeds_from_its_own_stage():
    """Same pin-aware step resolution the STEP path uses — a geometry
    materialized while pinned must not wear the active stage's mask."""
    geom_mgr = GeometryManager()
    geom = geom_mgr.add(name="Pinned", make_active=False)
    geom_mgr.set_stage_pin(geom.id, "stage-a")
    reader = _Reader(_STEP_VALUES)
    ctrl, _ = _controller(reader)
    ctrl.set_threshold(geom.id, component="u", lo=1.0, hi=3.0)
    director = _director(
        geom_mgr, {}, step_index=99, active_step=0,
        local_steps={"stage-a": 2},
    )

    fresh = _Scene(_quad_strip())
    _pumps(director, ctrl).refresh_threshold_for(geom, fresh)

    assert [(c[1], c[2]) for c in reader.calls] == [(2, "stage-a")]
    assert _hidden(fresh) == _EXPECTED_HIDDEN[2]


# =====================================================================
# Real window, real HDF5 (qt lane — run per file in a fresh process)
# =====================================================================


@pytest.fixture
def cube_results(g, tmp_path):
    """A meshed cube whose ``displacement_x`` IS the node id.

    That makes every expected hidden set derivable from the scene's own
    id→row maps, without assuming anything about gmsh's ordering.
    """
    from apeGmsh.results import Results
    from apeGmsh.results.writers import NativeWriter
    from tests.conftest import _open_model_from_h5

    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="cube")
    g.physical.add_volume("cube", name="Body")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)

    path = tmp_path / "threshold_cube.h5"
    with NativeWriter(path) as w:
        w.open(fem=fem)
        sid = w.begin_stage(
            name="grav", kind="static", time=np.array([0.0, 1.0]),
        )
        w.write_nodes(
            sid, "partition_0", node_ids=node_ids,
            components={
                "displacement_x": np.stack([
                    node_ids.astype(np.float64),
                    node_ids.astype(np.float64) + 1000.0,
                ]),
            },
        )
        w.end_stage()
    return Results.from_native(path, model=_open_model_from_h5(path))


def _all_nodes_in_range_oracle(scene, cut: float) -> set:
    """Cells with ANY node id above ``cut`` — the all-nodes rule, by
    hand, against the scene's real connectivity."""
    conn = np.asarray(scene.grid.cell_connectivity)
    offsets = np.asarray(scene.grid.offset)
    keep = np.asarray(scene.node_ids, dtype=np.float64) <= cut
    return {
        c for c in range(scene.grid.n_cells)
        if not bool(keep[conn[offsets[c]:offsets[c + 1]]].all())
    }


def _hidden_cells(scene) -> set:
    return set(
        np.flatnonzero(scene.element_visibility.hidden_mask()).tolist(),
    )


@pytest.mark.qt
def test_a_real_viewer_thresholds_real_hdf5_through_the_step_path(
    cube_results,
):
    """The wiring, in a real window: the shell's ``SceneValueReader``
    reads real HDF5 through the visual store, the STEP path applies the
    mask on the bound scene, and the hidden set matches the all-nodes
    rule computed by hand against the real connectivity.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._dispatch import STEP_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="threshold-real",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geom = director.geometries.active
            scene = director.scene_for(geom)
            cut = float(np.median(np.asarray(scene.node_ids)))
            director.set_step(0)

            director.thresholds.set_threshold(
                geom.id, component="displacement_x", lo=-1.0, hi=cut,
            )
            director.dispatcher.fire(STEP_CHANGED)
            seen["hidden"] = _hidden_cells(scene)
            seen["expected"] = _all_nodes_in_range_oracle(scene, cut)

            # Scrubbing to step 1 (+1000 on every node) drops the whole
            # mesh out of the band — LIVE, and read from real HDF5.
            director.set_step(1)
            seen["after_scrub"] = _hidden_cells(scene)
            seen["n_cells"] = int(scene.grid.n_cells)

            # Clearing restores exactly what it hid.
            director.thresholds.clear_threshold(geom.id)
            director.dispatcher.fire(STEP_CHANGED)
            seen["after_clear"] = _hidden_cells(scene)
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    # A cut that hid nothing (or everything) would make this vacuous.
    assert 0 < len(seen.get("expected") or ()) < seen["n_cells"]
    assert seen.get("hidden") == seen.get("expected")
    assert seen.get("after_scrub") == set(range(seen["n_cells"]))
    assert seen.get("after_clear") == set()


@pytest.mark.qt
def test_a_scene_materialized_by_an_eye_click_wears_its_threshold(
    cube_results,
):
    """The materialization seed, end to end and user-reachable.

    A restored geometry that was saved HIDDEN never has its scene
    materialized (the STEP and RENDER passes both walk visible
    geometries only). Clicking its eye runs DEFORM + GATE — no STEP —
    so ``_materialize_scene`` is the only thing that can apply its
    threshold. Without the seed the geometry appears unfiltered until
    the user happens to scrub.
    """
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    pytest.importorskip("qtpy.QtWidgets").QApplication.instance() \
        or pytest.importorskip("qtpy.QtWidgets").QApplication([])
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot, ThresholdSnapshot, ViewerSession,
    )
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="threshold-seed",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    class _Win:
        def set_status(self, *_a, **_k) -> None:
            pass

    def _drive_then_close():
        try:
            director = viewer._director
            bound = director.scene_for(director.geometries.active)
            cut = float(np.median(np.asarray(bound.node_ids)))
            director.set_step(0)

            viewer._apply_session(ViewerSession(
                schema_version=11, results_path="", fem_snapshot_id=None,
                saved_at="", diagrams=(),
                geometries=(
                    GeometrySnapshot(id="g0", name="Shown", visible=True),
                    GeometrySnapshot(
                        id="g1", name="Hidden", visible=False,
                        threshold=ThresholdSnapshot(
                            component="displacement_x", lo=-1.0, hi=cut,
                        ),
                    ),
                ),
            ), _Win())

            hidden_geom = next(
                geo for geo in director.geometries.geometries
                if geo.name == "Hidden"
            )
            # Never rendered, so never materialized.
            seen["unmaterialized"] = (
                hidden_geom.id not in director._scenes
            )
            # The eye: DEFORM + GATE, no STEP.
            director.geometries.set_visible(hidden_geom.id, True)
            scene = director.scene_for(hidden_geom)
            seen["hidden_cells"] = _hidden_cells(scene)
            seen["expected"] = _all_nodes_in_range_oracle(scene, cut)
            seen["n_cells"] = int(scene.grid.n_cells)
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("unmaterialized") is True
    assert 0 < len(seen.get("expected") or ()) < seen["n_cells"]
    assert seen.get("hidden_cells") == seen.get("expected")
