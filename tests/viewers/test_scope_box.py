"""Spatial SCOPE BOX — the catalog's second region tool.

A scope box hides every cell that has NO node inside an axis-aligned
world box. Like the threshold it is a layer on the existing
:class:`ElementVisibility` mechanism, so these tests assert on the two
things that define the feature — **which cell ids end up hidden**, and
**that the set does NOT follow the time cursor** — rather than on call
counts.

The rule diverges from the threshold's on purpose, and the divergence
gets its own test: threshold keeps a cell only when EVERY value is in
range, scope keeps it when ANY node is in the box. All-nodes would
erode the boundary (elements straddling a box face would vanish) and on
a coarse mesh a small box would hide everything. See
``test_a_cell_straddling_a_box_face_stays_visible``.

Everything here runs in the default headless lane: real
``ElementVisibility`` over real ``pyvista`` grids, the real
``ScopeController`` / ``GeometryManager`` / ``PumpSet``, and the real
session serializer. No Qt, no GL, no window.

The reference mesh is the threshold suite's strip of four quads sharing
edges, laid out along +x::

    1---3---5---7---9      cell k spans x in [k, k+1]
    |c0 |c1 |c2 |c3 |      node n sits at x = n // 2
    0---2---4---6---8

so every expected hidden set is checkable by hand against an x-range.
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
    LAYER_SCOPE,
    LAYER_THRESHOLD,
    ElementVisibility,
)
from apeGmsh.viewers.core.scope_controller import (
    ScopeController,
    cells_with_any_node_in_box,
    compute_hidden_mask,
    points_in_box,
    scope_points,
)
from apeGmsh.viewers.core.threshold_controller import (
    ThresholdController,
    ThresholdSettings,
)
from apeGmsh.viewers.diagrams._geometries import GeometryManager
from apeGmsh.viewers.scene_ir import BBox

# The qt-lane fixture, imported rather than re-typed: the meshed cube +
# NativeWriter stage the threshold suite already builds is exactly the
# shape the two real-window tests at the bottom need, and a second copy
# of a 30-line HDF5 writer would be a second thing to keep honest.
from tests.viewers.test_threshold import cube_results  # noqa: F401


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


def _mixed_mesh() -> pv.UnstructuredGrid:
    """tri(0,1,2) quad(2,3,4,5) line(5,6) tet(6,7,8,9) — runs of 3/4/2/4.

    Node ``n`` sits at ``x = n``, so an x-range names a node set.
    """
    pts = np.zeros((10, 3), dtype=np.float64)
    pts[:, 0] = np.arange(10)
    cells = np.hstack([
        [3, 0, 1, 2], [4, 2, 3, 4, 5], [2, 5, 6], [4, 6, 7, 8, 9],
    ])
    types = np.array(
        [VTK_TRIANGLE, VTK_QUAD, VTK_LINE, VTK_TETRA], dtype=np.uint8,
    )
    return pv.UnstructuredGrid(cells, types, pts)


def _ragged_mesh(rng) -> pv.UnstructuredGrid:
    """A random run of tri / quad / line / tet cells over 40 points.

    Deliberately RAGGED (runs of 2, 3, 4 in random order) — that is
    what makes the reduceat offsets load-bearing.
    """
    pts = rng.random((40, 3)) * 10.0
    kinds = [
        (VTK_LINE, 2), (VTK_TRIANGLE, 3), (VTK_QUAD, 4), (VTK_TETRA, 4),
    ]
    cells: list[int] = []
    types: list[int] = []
    for _ in range(int(rng.integers(6, 16))):
        vtk_type, n = kinds[int(rng.integers(0, len(kinds)))]
        nodes = rng.choice(40, size=n, replace=False)
        cells.extend([n, *(int(i) for i in nodes)])
        types.append(vtk_type)
    return pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.asarray(types, dtype=np.uint8),
        pts,
    )


class _Scene:
    """The three attributes the scope refresh reads off a scene."""

    def __init__(self, grid: pv.UnstructuredGrid) -> None:
        self.grid = grid
        self.reference_points = np.asarray(grid.points).copy()
        self.element_visibility = ElementVisibility(grid)


def _hidden(scene: _Scene) -> set:
    """The EFFECTIVE hidden cell ids, read off the composed ghost array."""
    return set(np.flatnonzero(scene.element_visibility.hidden_mask()).tolist())


def _box(lo, hi) -> BBox:
    return BBox(np.asarray(lo, dtype=np.float64),
                np.asarray(hi, dtype=np.float64))


def _geom(gid: str = "g1", offset=(0.0, 0.0, 0.0)):
    return SimpleNamespace(id=gid, offset=tuple(offset))


# =====================================================================
# The cell rule — ANY NODE IN THE BOX (the threshold divergence)
# =====================================================================


def test_a_cell_straddling_a_box_face_stays_visible():
    """THE ratified divergence from the threshold, stated as a test.

    The box covers x in [1.5, 2.5]. Cell 1 spans x in [1, 2] and cell 2
    spans [2, 3] — each has exactly ONE column of nodes inside (x = 2),
    the other outside. Under the threshold's all-values rule both would
    be hidden; under the scope's any-node rule both must STAY VISIBLE,
    because "show me this region" cannot mean "shrink the region by one
    element in every direction".

    Cells 0 (x in [0, 1]) and 3 (x in [3, 4]) have no node inside at
    all and are the only ones hidden.
    """
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, grid.points, _box((1.5, -1, -1), (2.5, 2, 1)),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 3}

    # …and the same geometry under the THRESHOLD's all-values rule on
    # the x coordinate hides the straddlers, which is the whole point.
    from apeGmsh.viewers.core.threshold_controller import (
        compute_hidden_mask as threshold_hidden_mask,
    )
    thr_hide = threshold_hidden_mask(
        grid, np.asarray(grid.points)[:, 0],
        ThresholdSettings("x", 1.5, 2.5),
    )
    assert set(np.flatnonzero(thr_hide).tolist()) == {0, 1, 2, 3}


def test_a_box_smaller_than_one_element_still_shows_that_element():
    """The coarse-mesh failure the any-node rule exists to prevent.

    A box entirely INSIDE cell 1 touches no node at all, so nothing is
    kept — the honest outcome for a box that contains no mesh. Grow it
    to reach a single node of cell 1 and that cell comes back on its
    own, without the box having to swallow the whole element.
    """
    grid = _quad_strip()
    empty = compute_hidden_mask(
        grid, grid.points, _box((1.4, 0.4, -0.1), (1.6, 0.6, 0.1)),
    )
    assert set(np.flatnonzero(empty).tolist()) == {0, 1, 2, 3}

    one_node = compute_hidden_mask(
        grid, grid.points, _box((0.9, -0.1, -0.1), (1.6, 0.6, 0.1)),
    )
    # x = 1 nodes are shared by cells 0 and 1; both are kept.
    assert set(np.flatnonzero(one_node).tolist()) == {2, 3}


def test_a_cell_entirely_outside_the_box_is_hidden():
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, grid.points, _box((-1, -1, -1), (1.5, 2, 1)),
    )
    # Cells 0 and 1 reach x <= 1.5; cells 2 and 3 do not.
    assert set(np.flatnonzero(hide).tolist()) == {2, 3}


def test_a_box_containing_everything_hides_nothing():
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, grid.points, _box((-10, -10, -10), (10, 10, 10)),
    )
    assert set(np.flatnonzero(hide).tolist()) == set()


def test_a_box_containing_nothing_hides_everything():
    grid = _quad_strip()
    hide = compute_hidden_mask(
        grid, grid.points, _box((100, 100, 100), (101, 101, 101)),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 1, 2, 3}


def test_membership_is_inclusive_on_the_face():
    """``BBox.contains`` is inclusive, and so is the vectorised twin —
    a node exactly on the box face is IN."""
    grid = _quad_strip()
    on_face = _box((3.0, -1, -1), (4.0, 2, 1))
    assert on_face.contains((3.0, 0.0, 0.0)) is True
    hide = compute_hidden_mask(grid, grid.points, on_face)
    # x = 3 nodes belong to cells 2 and 3.
    assert set(np.flatnonzero(hide).tolist()) == {0, 1}


def test_a_nan_coordinate_is_outside_the_box():
    """NaN compares False against both bounds, so it does not count as
    a node inside — a cell whose only in-range node is NaN hides."""
    grid = _quad_strip()
    pts = np.asarray(grid.points).copy()
    pts[4] = np.nan          # x = 2 column, bottom node
    pts[5] = np.nan          # x = 2 column, top node
    hide = compute_hidden_mask(
        grid, pts, _box((1.5, -1, -1), (2.5, 2, 1)),
    )
    assert set(np.flatnonzero(hide).tolist()) == {0, 1, 2, 3}


def test_empty_mesh_yields_an_empty_mask():
    grid = pv.UnstructuredGrid(
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint8),
        np.zeros((0, 3)),
    )
    hide = compute_hidden_mask(
        grid, np.zeros((0, 3)), _box((0, 0, 0), (1, 1, 1)),
    )
    assert hide.shape == (0,)


# =====================================================================
# The vectorised reduction
# =====================================================================


def test_the_reduction_handles_mixed_cell_types():
    """Runs of 3 / 4 / 2 / 4 reduce independently — the offsets carry
    the cell sizes, so no branch per cell type is needed."""
    grid = _mixed_mesh()
    inside = np.zeros(10, dtype=bool)
    inside[4] = True             # node 4 belongs to the QUAD only
    keep = cells_with_any_node_in_box(grid, inside)
    assert keep.tolist() == [False, True, False, False]

    inside = np.zeros(10, dtype=bool)
    inside[6] = True             # node 6 is shared by the LINE and TET
    keep = cells_with_any_node_in_box(grid, inside)
    assert keep.tolist() == [False, False, True, True]

    inside = np.zeros(10, dtype=bool)
    inside[0] = True             # node 0 belongs to the TRI only
    keep = cells_with_any_node_in_box(grid, inside)
    assert keep.tolist() == [True, False, False, False]


def test_the_reduction_agrees_with_a_per_cell_python_oracle():
    """The vectorised reduceat must equal the obvious slow loop.

    Cross-checked on RANDOMISED RAGGED meshes (runs of 2 / 3 / 4 in
    random order) over many random boxes — this is the guard against an
    off-by-one in the offsets. The classic reduceat trap is the
    trailing offset, which silently merges the last two cells; a ragged
    tail makes that visible.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        grid = _ragged_mesh(rng)
        conn, offs = grid.cell_connectivity, grid.offset
        lo = rng.random(3) * 10.0
        box = _box(lo, lo + rng.random(3) * 6.0)
        inside = points_in_box(box, grid.points)
        oracle = [
            bool(inside[conn[offs[k]:offs[k + 1]]].any())
            for k in range(grid.n_cells)
        ]
        assert cells_with_any_node_in_box(grid, inside).tolist() == oracle


def test_the_reduction_agrees_with_the_oracle_on_random_masks_too():
    """Same cross-check driven by raw random point masks rather than a
    box, so a degenerate box choice cannot make the oracle vacuous."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        grid = _ragged_mesh(rng)
        conn, offs = grid.cell_connectivity, grid.offset
        inside = rng.random(grid.n_points) > 0.7
        oracle = [
            bool(inside[conn[offs[k]:offs[k + 1]]].any())
            for k in range(grid.n_cells)
        ]
        assert cells_with_any_node_in_box(grid, inside).tolist() == oracle


def test_the_reduction_is_correct_on_a_hex_mesh_of_realistic_size():
    """Sanity at scale, and a guard that 8-node runs reduce correctly.

    Keep every hex with ANY node at z <= 10 on a 20x20x20 cell grid:
    that is the 10 layers fully below the plane PLUS the layer that
    straddles it (its bottom face sits at z = 10) — 11 in all. The
    all-nodes rule would give 10, so this number IS the divergence.
    """
    grid = pv.ImageData(dimensions=(21, 21, 21)).cast_to_unstructured_grid()
    assert grid.n_cells == 8000
    keep = cells_with_any_node_in_box(grid, grid.points[:, 2] <= 10.0)
    assert int(keep.sum()) == 11 * 20 * 20


# =====================================================================
# Controller — apply / clear against a real ElementVisibility
# =====================================================================


def test_refresh_applies_the_scope_layer_to_the_scene():
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    box = _box((1.5, -1, -1), (2.5, 2, 1))
    ctrl.set_scope("g1", box)
    assert ctrl.box_for("g1") is box

    ctrl.refresh(_geom(), scene)
    assert _hidden(scene) == {0, 3}


def test_clearing_the_scope_restores_exactly_its_own_cells():
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    geom = _geom()
    ctrl.set_scope("g1", _box((1.5, -1, -1), (2.5, 2, 1)))
    ctrl.refresh(geom, scene)
    assert _hidden(scene) == {0, 3}

    ctrl.clear_scope("g1")
    assert ctrl.box_for("g1") is None
    ctrl.refresh(geom, scene)
    assert _hidden(scene) == set()


def test_a_disabled_geometry_is_untouched_by_another_geometrys_scope():
    """State is keyed per geometry — one geometry's box must not reach
    another geometry's scene."""
    scene_a, scene_b = _Scene(_quad_strip()), _Scene(_quad_strip())
    ctrl = ScopeController()
    ctrl.set_scope("gA", _box((1.5, -1, -1), (2.5, 2, 1)))

    ctrl.refresh(_geom("gA"), scene_a)
    ctrl.refresh(_geom("gB"), scene_b)
    assert _hidden(scene_a) == {0, 3}
    assert _hidden(scene_b) == set()


def test_needs_refresh_is_false_until_a_scope_exists_and_after_it_is_gone():
    """The take-down pass: after the last clear it must stay True for
    exactly one more refresh."""
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    geom = _geom()
    assert ctrl.needs_refresh() is False

    ctrl.set_scope("g1", _box((1.5, -1, -1), (2.5, 2, 1)))
    assert ctrl.needs_refresh() is True
    ctrl.refresh(geom, scene)

    ctrl.clear_scope("g1")
    assert ctrl.needs_refresh() is True        # layer still applied
    ctrl.refresh(geom, scene)
    assert ctrl.needs_refresh() is False
    assert _hidden(scene) == set()


def test_a_scene_without_element_visibility_is_a_no_op():
    ctrl = ScopeController()
    ctrl.set_scope("g1", _box((0, 0, 0), (1, 1, 1)))
    ctrl.refresh(_geom(), SimpleNamespace(grid=None, element_visibility=None))


def test_set_scope_refuses_anything_that_is_not_a_bbox():
    """ADR 0045 INV-2 — there is exactly one bounding-box value type,
    and a 6-tuple silently accepted here would become a second one."""
    ctrl = ScopeController()
    with pytest.raises(TypeError):
        ctrl.set_scope("g1", (0.0, 0.0, 0.0, 1.0, 1.0, 1.0))


def test_an_inverted_box_cannot_be_built_at_all():
    """``BBox`` validates ``min <= max`` at construction, so the
    controller never has to. The UI catches this one."""
    with pytest.raises(ValueError):
        _box((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))


# =====================================================================
# REFERENCE geometry + the per-geometry offset
# =====================================================================


def test_the_mask_is_evaluated_against_reference_plus_offset():
    """ADR 0058 S3a: the box is world-frame and the offset moves the
    geometry through it, so an offset genuinely changes the mask.

    Same box (x in [-1, 1.5]), same mesh. At zero offset the node
    columns inside are x = 0 and 1, so cells 0 and 1 are kept and 2 / 3
    hide. Slide the geometry -3 in x and the columns inside become the
    ones that started at x = 2, 3, 4 — cells 1, 2 and 3 now touch the
    box and only cell 0 hides.
    """
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    box = _box((-1, -1, -1), (1.5, 2, 1))
    ctrl.set_scope("g1", box)

    ctrl.refresh(_geom(offset=(0.0, 0.0, 0.0)), scene)
    assert _hidden(scene) == {2, 3}

    ctrl.refresh(_geom(offset=(-3.0, 0.0, 0.0)), scene)
    assert _hidden(scene) == {0}


def test_scope_points_are_reference_not_the_deformed_grid():
    """The zero-per-step property, at its root: the points fed to the
    mask come from ``reference_points``, so warping ``grid.points``
    (what the DEFORM pump does every tick) changes nothing."""
    scene = _Scene(_quad_strip())
    before = scope_points(_geom(), scene).copy()
    scene.grid.points = np.asarray(scene.grid.points) + 100.0
    assert np.array_equal(scope_points(_geom(), scene), before)


def test_offset_change_recomputes_the_mask_through_the_controller():
    """The wired trigger, at controller level: a geometry object whose
    offset moved yields a different layer on the SAME scene."""
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    ctrl.set_scope("g1", _box((-1, -1, -1), (1.5, 2, 1)))
    geom = _geom()

    ctrl.refresh(geom, scene)
    assert _hidden(scene) == {2, 3}

    geom.offset = (-3.0, 0.0, 0.0)
    ctrl.refresh(geom, scene)
    assert _hidden(scene) == {0}


# =====================================================================
# Layer composition — the whole reason scope is a LAYER
# =====================================================================


def test_scope_composes_with_threshold_manual_and_dim_layers():
    """The effective hidden set is the OR of all four, and clearing the
    scope reveals ONLY the cells the scope hid.

    This is the exact clobbering the layered design exists to prevent:
    a single ghost writer would have made ``clear_scope`` reveal the
    user's manual hide too.
    """
    grid = _quad_strip(5)
    scene = _Scene(grid)
    ev = scene.element_visibility

    ev.hide([1])                                     # manual: cell 1
    dim_mask = np.zeros(grid.n_cells, dtype=bool)
    dim_mask[2] = True                               # dim filter: cell 2
    ev.set_layer(LAYER_DIM, dim_mask)
    thr_mask = np.zeros(grid.n_cells, dtype=bool)
    thr_mask[3] = True                               # threshold: cell 3
    ev.set_layer(LAYER_THRESHOLD, thr_mask)
    assert _hidden(scene) == {1, 2, 3}

    ctrl = ScopeController()
    ctrl.set_scope("g1", _box((2.5, -1, -1), (10, 2, 1)))
    ctrl.refresh(_geom(), scene)                     # scope hides 0, 1
    assert _hidden(scene) == {0, 1, 2, 3}

    ctrl.clear_scope("g1")
    ctrl.refresh(_geom(), scene)
    assert _hidden(scene) == {1, 2, 3}               # the other three survive
    assert set(np.flatnonzero(ev._layers[LAYER_MANUAL]).tolist()) == {1}
    assert LAYER_SCOPE not in ev._layers


def test_show_all_does_not_reveal_scoped_cells():
    """``show_all`` clears the MANUAL layer only — the scope's cells
    stay hidden, mirroring the dim filter's guarantee."""
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    ctrl.set_scope("g1", _box((1.5, -1, -1), (2.5, 2, 1)))
    ctrl.refresh(_geom(), scene)

    scene.element_visibility.hide([1])
    assert _hidden(scene) == {0, 1, 3}
    scene.element_visibility.show_all()
    assert _hidden(scene) == {0, 3}


def test_the_scope_layer_is_named_scope_and_is_its_own_layer():
    scene = _Scene(_quad_strip())
    ctrl = ScopeController()
    ctrl.set_scope("g1", _box((1.5, -1, -1), (2.5, 2, 1)))
    ctrl.refresh(_geom(), scene)
    layers = scene.element_visibility._layers
    assert LAYER_SCOPE == "scope"
    assert set(np.flatnonzero(layers[LAYER_SCOPE]).tolist()) == {0, 3}
    assert LAYER_THRESHOLD not in layers


# =====================================================================
# STATIC under the time cursor — the zero-per-step-cost property
# =====================================================================


def _director(geom_mgr, scenes, *, step_index=0):
    """The director surface the STEP pump uses (see test_pump_set)."""
    return SimpleNamespace(
        geometries=geom_mgr,
        registry=SimpleNamespace(diagrams=lambda: []),
        step_index=step_index,
        local_step_for_stage=lambda sid: step_index,
        local_step_for_active_stage=lambda: step_index,
        scene_for=lambda geom: scenes.get(geom.id),
    )


def _pumps(director, thresholds=None) -> PumpSet:
    return PumpSet(
        director=director,
        scene=None,
        read_deform_field=lambda *a, **k: None,
        render_geometries=lambda: list(director.geometries.geometries),
        sync_node_cloud=lambda _pts: None,
        sync_diagram_substrate_points=lambda *a, **k: None,
        thresholds=thresholds,
    )


class _CountingScopes(ScopeController):
    """A controller that records every :meth:`refresh`."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list = []

    def refresh(self, geometry, scene) -> None:
        self.calls.append(getattr(geometry, "id", geometry))
        super().refresh(geometry, scene)


def test_scrubbing_does_not_change_the_scope_mask_or_recompute_it():
    """The zero-per-step-cost property, pinned.

    The scope is off the STEP path by construction: the pump set has no
    handle on it. Ten scrub ticks must therefore leave the hidden set
    byte-identical AND never call ``refresh`` — this is a feature (a
    seismic scrub does not make cells pop in and out of the scope), so
    it is asserted rather than left to luck.
    """
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    scene = _Scene(_quad_strip())
    scopes = _CountingScopes()
    scopes.set_scope(gid, _box((1.5, -1, -1), (2.5, 2, 1)))
    scopes.refresh(geom_mgr.active, scene)
    baseline = _hidden(scene)
    assert baseline == {0, 3}
    scopes.calls.clear()

    for step in range(10):
        director = _director(geom_mgr, {gid: scene}, step_index=step)
        _pumps(director).pump_step(None)
        assert _hidden(scene) == baseline, f"step {step}"

    assert scopes.calls == []


def test_a_live_threshold_scrubbing_beside_a_static_scope_keeps_both():
    """The two region tools together: the threshold's mask MOVES with
    the cursor while the scope's stays put, and the composed hidden set
    is the OR of the two at every step."""
    geom_mgr = GeometryManager()
    gid = geom_mgr.active_id
    grid = _quad_strip()
    scene = _Scene(grid)

    # Node scalars rise by one column per step (the threshold suite's
    # fixture) so [1, 3] slides right; hand-checked hidden sets.
    values = {
        step: (np.arange(10) // 2).astype(np.float64) + step
        for step in range(3)
    }
    thr_expected = {0: {0, 3}, 1: {2, 3}, 2: {1, 2, 3}}

    thresholds = ThresholdController(
        read_values=lambda comp, step, **kw: values[int(step)],
        on_failure=lambda *a, **k: None,
    )
    thresholds.set_threshold(gid, component="u", lo=1.0, hi=3.0)

    scopes = ScopeController()
    scopes.set_scope(gid, _box((1.5, -1, -1), (2.5, 2, 1)))   # hides 0, 3
    scopes.refresh(geom_mgr.active, scene)

    for step, expected in thr_expected.items():
        director = _director(geom_mgr, {gid: scene}, step_index=step)
        _pumps(director, thresholds).pump_step(None)
        assert _hidden(scene) == expected | {0, 3}, f"step {step}"


def test_a_scene_materialized_mid_session_wears_the_scope_at_once():
    """``_materialize_scene`` seeds the layer through this seam.

    A geometry whose scene appears mid-session (a new geometry, an eye
    click on a restored-hidden one) must render scoped from its FIRST
    frame. There is no later STEP to rescue it — the scope is not on
    the step path — so the seed is the only chance.
    """
    geom_mgr = GeometryManager()
    scopes = ScopeController()
    scopes.set_scope(geom_mgr.active_id, _box((1.5, -1, -1), (2.5, 2, 1)))

    fresh = _Scene(_quad_strip())            # freshly cloned, unfiltered
    assert _hidden(fresh) == set()

    scopes.refresh(geom_mgr.active, fresh)
    assert _hidden(fresh) == {0, 3}


def test_materialization_costs_nothing_when_no_scope_is_set():
    geom_mgr = GeometryManager()
    scopes = ScopeController()
    fresh = _Scene(_quad_strip())
    assert scopes.needs_refresh() is False
    scopes.refresh(geom_mgr.active, fresh)
    assert _hidden(fresh) == set()


# =====================================================================
# The dispatcher trigger set
# =====================================================================


def test_the_scope_event_runs_no_primitive():
    """``GEOMETRY_SCOPE_CHANGED`` is render-only by design: the mask is
    time-invariant, so there is nothing for STEP or DEFORM to do, and a
    row with primitives would put scope work back on the scrub tick the
    ADR 0084 D1 freeze pins."""
    from apeGmsh.viewers.diagrams._dispatch import (
        _MATRIX, GEOMETRY_SCOPE_CHANGED,
    )
    assert _MATRIX[GEOMETRY_SCOPE_CHANGED] == frozenset()


def test_the_scope_event_is_not_a_granular_geometry_kind():
    """It is not fired by ``GeometryManager``, so it must NOT arm the
    omnibus-suppression flag — doing so would swallow the next real
    ``GEOMETRIES_CHANGED``."""
    from apeGmsh.viewers.diagrams._dispatch import (
        _GRANULAR_GEOMETRY_KINDS, GEOMETRY_SCOPE_CHANGED,
    )
    assert GEOMETRY_SCOPE_CHANGED not in _GRANULAR_GEOMETRY_KINDS


# =====================================================================
# Session persistence (schema v12)
# =====================================================================


def _session_dict(scope_payload, *, name="Geometry 1"):
    """A minimal v12 session envelope carrying one geometry."""
    return {
        "schema_version": 12,
        "results_path": "",
        "fem_snapshot_id": None,
        "saved_at": "",
        "diagrams": [],
        "geometries": [{
            "id": "stale-uuid",
            "name": name,
            "scope": scope_payload,
            "compositions": [],
        }],
    }


def test_the_schema_version_is_twelve():
    from apeGmsh.viewers.diagrams._session import SESSION_SCHEMA_VERSION
    assert SESSION_SCHEMA_VERSION == 12


def test_a_scope_round_trips_through_save_and_deserialize(tmp_path):
    """save -> deserialize -> apply: the box lands on the RIGHT
    geometry's controller and the mask it produces is the same one."""
    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot, ScopeSnapshot, deserialize_session, load_session,
        save_session,
    )

    box = _box((1.5, -1.0, -1.0), (2.5, 2.0, 1.0))
    path = save_session(
        specs=[],
        results_path=tmp_path / "r.h5",
        fem_snapshot_id=None,
        geometries=[
            GeometrySnapshot(id="a", name="Alpha"),
            GeometrySnapshot(
                id="b", name="Beta",
                scope=ScopeSnapshot(
                    min=tuple(box.min), max=tuple(box.max),
                ),
            ),
        ],
    )
    session = load_session(path)
    assert session.schema_version == 12
    assert session.geometries[0].scope is None
    snap = session.geometries[1].scope
    assert snap.min == (1.5, -1.0, -1.0)
    assert snap.max == (2.5, 2.0, 1.0)

    # Apply it the way ``_session_apply`` does — keyed by the LIVE id,
    # because the saved one is a stale UUID.
    geom_mgr = GeometryManager()
    live = geom_mgr.add(name="Beta", make_active=False)
    ctrl = ScopeController()
    ctrl.set_scope(live.id, BBox(snap.min, snap.max))
    scene = _Scene(_quad_strip())
    ctrl.refresh(live, scene)
    assert _hidden(scene) == {0, 3}

    # …and the deserializer is the same on the raw dict.
    raw = deserialize_session(_session_dict(
        {"min": list(box.min), "max": list(box.max)},
    ))
    assert raw.geometries[0].scope.max == (2.5, 2.0, 1.0)


def test_a_legacy_session_with_no_scope_field_reads_as_no_scope():
    """Every pre-v12 save must load clean, with the scope simply off."""
    from apeGmsh.viewers.diagrams._session import deserialize_session

    data = _session_dict(None)
    del data["geometries"][0]["scope"]
    data["schema_version"] = 11
    session = deserialize_session(data)
    assert len(session.geometries) == 1
    assert session.geometries[0].scope is None
    assert session.geometries[0].name == "Geometry 1"


@pytest.mark.parametrize("payload", [
    {"min": [0.0, 0.0], "max": [1.0, 1.0, 1.0]},     # short vector
    {"min": [0.0, 0.0, 0.0]},                        # no max
    {"min": "nope", "max": [1.0, 1.0, 1.0]},         # not numeric
    {"min": None, "max": None},
    "not-a-dict",
])
def test_a_malformed_scope_degrades_to_none_and_keeps_the_geometry(payload):
    """Raising here would DROP the geometry, and ``layer_indices`` are
    positional references into the diagram list — every later layer
    would slide into the wrong composition (the ADR 0084 D6 hazard)."""
    from apeGmsh.viewers.diagrams._session import deserialize_session

    session = deserialize_session(_session_dict(payload))
    assert len(session.geometries) == 1               # NOT dropped
    assert session.geometries[0].scope is None
    assert session.geometries[0].name == "Geometry 1"


def test_an_inverted_saved_box_does_not_abort_the_restore():
    """A hand-edited (or corrupt) session with min > max deserializes
    fine; ``BBox`` refuses it at apply time and the restore path drops
    just that scope."""
    from apeGmsh.viewers.diagrams._session import deserialize_session

    session = deserialize_session(_session_dict(
        {"min": [5.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
    ))
    snap = session.geometries[0].scope
    assert snap is not None
    with pytest.raises(ValueError):
        BBox(snap.min, snap.max)


# =====================================================================
# Real window, real HDF5 (qt lane — run per file in a fresh process)
# =====================================================================
#
# Everything above proves the RULE. These two prove the WIRING: that
# the four recompute triggers are actually subscribed in the shell, and
# that scrubbing a real viewer leaves the mask alone.


def _any_node_oracle(scene, box: BBox, offset=(0.0, 0.0, 0.0)) -> set:
    """Cells with NO node in ``box`` — the any-node rule by hand,
    against the scene's real connectivity and reference points."""
    pts = np.asarray(scene.reference_points) + np.asarray(offset)
    inside = np.all((pts >= box.min) & (pts <= box.max), axis=1)
    conn = np.asarray(scene.grid.cell_connectivity)
    offsets = np.asarray(scene.grid.offset)
    return {
        c for c in range(scene.grid.n_cells)
        if not bool(inside[conn[offsets[c]:offsets[c + 1]]].any())
    }


def _hidden_cells(scene) -> set:
    return set(
        np.flatnonzero(scene.element_visibility.hidden_mask()).tolist(),
    )


def _need_qt():
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    qtw = pytest.importorskip("qtpy.QtWidgets")
    qtw.QApplication.instance() or qtw.QApplication([])


@pytest.mark.qt
def test_a_real_viewer_scopes_real_geometry_and_scrubbing_leaves_it(
    cube_results,
):
    """The wiring, in a real window.

    ``GEOMETRY_SCOPE_CHANGED`` reaches the shell's RENDER-lane
    subscriber, the mask lands on the bound scene and matches the
    any-node rule computed by hand — and then a SCRUB leaves it byte
    identical, which is the whole point of evaluating against reference
    geometry. Clearing restores exactly what it hid.
    """
    _need_qt()
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._dispatch import GEOMETRY_SCOPE_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-real",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geom = director.geometries.active
            scene = director.scene_for(geom)
            director.set_step(0)

            # Half the cube in x: a box that must both keep and hide.
            pts = np.asarray(scene.reference_points)
            cut = float(np.median(pts[:, 0]))
            lo = pts.min(axis=0) - 1.0
            hi = pts.max(axis=0) + 1.0
            box = BBox(lo, (cut, hi[1], hi[2]))

            director.scopes.set_scope(geom.id, box)
            director.dispatcher.fire(GEOMETRY_SCOPE_CHANGED)
            seen["hidden"] = _hidden_cells(scene)
            seen["expected"] = _any_node_oracle(scene, box)
            seen["n_cells"] = int(scene.grid.n_cells)

            # A scrub must not move it — zero per-step cost, and no
            # cells popping in and out while the structure deforms.
            director.set_step(1)
            seen["after_scrub"] = _hidden_cells(scene)

            director.scopes.clear_scope(geom.id)
            director.dispatcher.fire(GEOMETRY_SCOPE_CHANGED)
            seen["after_clear"] = _hidden_cells(scene)
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    # A box that hid nothing (or everything) would make this vacuous.
    assert 0 < len(seen.get("expected") or ()) < seen["n_cells"]
    assert seen.get("hidden") == seen.get("expected")
    assert seen.get("after_scrub") == seen.get("expected")
    assert seen.get("after_clear") == set()


@pytest.mark.qt
def test_a_scene_materialized_by_an_eye_click_wears_its_scope(cube_results):
    """The materialization seed, end to end and user-reachable.

    A restored geometry saved HIDDEN never has its scene materialized.
    Clicking its eye runs DEFORM + GATE — and the scope is not even on
    the STEP path, so ``_materialize_scene`` is the ONLY thing that can
    apply its box. Without the seed it would appear unscoped and stay
    that way, because no later event recomputes a mask that does not
    follow the cursor.
    """
    _need_qt()
    from qtpy import QtCore

    from apeGmsh.viewers.diagrams._session import (
        GeometrySnapshot, ScopeSnapshot, ViewerSession,
    )
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-seed",
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
            pts = np.asarray(bound.reference_points)
            cut = float(np.median(pts[:, 0]))
            lo = pts.min(axis=0) - 1.0
            hi = pts.max(axis=0) + 1.0
            box = BBox(lo, (cut, hi[1], hi[2]))
            director.set_step(0)

            viewer._apply_session(ViewerSession(
                schema_version=12, results_path="", fem_snapshot_id=None,
                saved_at="", diagrams=(),
                geometries=(
                    GeometrySnapshot(id="g0", name="Shown", visible=True),
                    GeometrySnapshot(
                        id="g1", name="Hidden", visible=False,
                        scope=ScopeSnapshot(
                            min=tuple(box.min), max=tuple(box.max),
                        ),
                    ),
                ),
            ), _Win())

            hidden_geom = next(
                geo for geo in director.geometries.geometries
                if geo.name == "Hidden"
            )
            # Never rendered, so never materialized.
            seen["unmaterialized"] = hidden_geom.id not in director._scenes
            # The eye: DEFORM + GATE, no STEP.
            director.geometries.set_visible(hidden_geom.id, True)
            scene = director.scene_for(hidden_geom)
            seen["hidden_cells"] = _hidden_cells(scene)
            seen["expected"] = _any_node_oracle(scene, box)
            seen["n_cells"] = int(scene.grid.n_cells)
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(400, _drive_then_close)
    viewer.show()

    assert seen.get("unmaterialized") is True
    assert 0 < len(seen.get("expected") or ()) < seen["n_cells"]
    assert seen.get("hidden_cells") == seen.get("expected")
