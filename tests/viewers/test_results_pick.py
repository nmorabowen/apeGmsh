"""Results pick controller — geometric hit → FEM result resolution.

ADR 0047 R-D.2b-ii: the gesture machine (vtkCellPicker, observers,
rubber-band, projection) moved to ``PyVistaPickBackend`` — covered by
``test_pyvista_pick_backend.py``. Here we inject a stub backend and fire
``PickHit`` / ``BoxGesture`` to test the results-specific resolution:
mode routing (node / element / gp), the dim-pick gate, GP resolution via
the ``PickInventory``, element-mode GP-occlusion routing, ghost masking,
and the box projection path. Fully headless.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.core.element_visibility import HIDDENCELL
from apeGmsh.viewers.core.results_pick import (
    ResultsPickController,
    _inside_box,
    install_results_pick,
)
from apeGmsh.viewers.core.results_pick_engine import PickInventory
from apeGmsh.viewers.scene_ir import BoxGesture, PickHit, PickModifiers


# ── Stubs ───────────────────────────────────────────────────────────

class _StubBackend:
    """Stand-in PickBackend: records install callbacks, projects via the
    injected function, and lets a test fire pick / box gestures."""

    def __init__(self, project=None) -> None:
        self._project = project
        self.on_pick = None
        self.on_box = None
        self.uninstalled = 0

    def install(self, *, on_pick, on_hover=None, on_box=None) -> None:
        self.on_pick = on_pick
        self.on_box = on_box

    def project_points(self, pts):
        if self._project is not None:
            return self._project(pts)
        return np.asarray(pts, dtype=np.float64)[:, :2]

    def uninstall(self) -> None:
        self.uninstalled += 1

    # test drivers
    def fire_pick(self, hit, mods=None) -> None:
        self.on_pick(hit, mods or PickModifiers())

    def fire_box(self, box) -> None:
        self.on_box(BoxGesture(box=box, crossing=box[2] < box[0]))


class _Scene:
    """Minimal FEMSceneData stand-in."""

    def __init__(self, *, cell_to_element_id=None, node_ids=None, grid=None,
                 cell_dim=None, element_id_to_cell=None, inventory=None) -> None:
        self.cell_to_element_id = np.asarray(
            cell_to_element_id if cell_to_element_id is not None
            else [1001, 1002, 1003], dtype=np.int64,
        )
        self.node_ids = np.asarray(
            node_ids if node_ids is not None else [10, 20, 30], dtype=np.int64,
        )
        self.grid = grid
        self.cell_dim = np.asarray(
            cell_dim if cell_dim is not None else [], dtype=np.int8,
        )
        self.element_id_to_cell = element_id_to_cell or {}
        self.pick_engine = inventory


def _install(scene, *, on_pick=None, on_box_pick=None, gp_candidates=None,
             backend=None):
    seen, boxes = [], []
    cb = on_pick if on_pick is not None else (lambda r: seen.append(r))
    boxcb = on_box_pick if on_box_pick is not None else (lambda r: boxes.append(r))
    backend = backend or _StubBackend()
    ctrl = install_results_pick(
        None, cb, scene=scene, on_box_pick=boxcb,
        gp_candidates=gp_candidates, pick_backend=backend,
    )
    return ctrl, backend, seen, boxes


def _hit(prop_id=None, cell_id=0, world=(0.0, 0.0, 0.0)):
    return PickHit(world=world, cell_id=cell_id, prop_id=prop_id)


# ── click: node ─────────────────────────────────────────────────────

def test_default_mode_is_node():
    ctrl, backend, seen, _ = _install(_Scene())
    assert ctrl.mode == "node"
    backend.fire_pick(_hit(cell_id=2, world=(1.0, 2.0, 3.0)))
    assert len(seen) == 1
    assert seen[0].kind == "node"
    assert seen[0].world == (1.0, 2.0, 3.0)
    assert seen[0].element_id is None


def test_miss_hit_none_fires_nothing():
    ctrl, backend, seen, _ = _install(_Scene())
    backend.fire_pick(None)
    assert seen == []


def test_negative_cell_id_fires_nothing():
    ctrl, backend, seen, _ = _install(_Scene())
    ctrl.set_mode("element")
    backend.fire_pick(_hit(cell_id=-1))
    assert seen == []


# ── click: element (substrate) ──────────────────────────────────────

def test_element_substrate_resolves_cell_to_element_id():
    ctrl, backend, seen, _ = _install(
        _Scene(cell_to_element_id=[1001, 1002, 1003])
    )
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=999, cell_id=1))  # prop not in inventory
    assert seen[0].kind == "element"
    assert seen[0].element_id == 1002
    assert seen[0].cell_id == 1


def test_element_out_of_range_cell_drops():
    ctrl, backend, seen, _ = _install(_Scene(cell_to_element_id=[1001]))
    ctrl.set_mode("element")
    backend.fire_pick(_hit(prop_id=999, cell_id=5))
    assert seen == []


def test_element_dim_gate_rejects_inactive_dim():
    ctrl, backend, seen, _ = _install(
        _Scene(cell_to_element_id=[1001, 1002, 1003], cell_dim=[1, 2, 3])
    )
    ctrl.set_mode("element")
    ctrl.active_dims = frozenset({2})        # only dim-2 cells pick
    backend.fire_pick(_hit(prop_id=999, cell_id=0))  # cell 0 is dim 1
    assert seen == []
    backend.fire_pick(_hit(prop_id=999, cell_id=1))  # cell 1 is dim 2
    assert len(seen) == 1 and seen[0].element_id == 1002


# ── click: element via GP-marker occlusion routing ──────────────────

def test_element_mode_gp_marker_resolves_to_owning_element():
    inv = PickInventory()
    gp_actor = object()
    inv.register_actor(gp_actor, "gp", lambda c: (1002, 4, (9.0, 9.0, 9.0)))
    scene = _Scene(
        cell_to_element_id=[1001, 1002, 1003],
        element_id_to_cell={1002: 1},
        inventory=inv,
    )
    ctrl, backend, seen, _ = _install(scene)
    ctrl.set_mode("element")
    # prop_id is the GP actor → resolves to its element 1002, highlight via cell 1.
    backend.fire_pick(_hit(prop_id=id(gp_actor), cell_id=0))
    assert seen[0].kind == "element"
    assert seen[0].element_id == 1002
    assert seen[0].cell_id == 1


# ── click: gp ───────────────────────────────────────────────────────

def test_gp_mode_resolves_via_inventory():
    inv = PickInventory()
    gp_actor = object()
    inv.register_actor(gp_actor, "gp", lambda c: (1003, 7, (5.0, 6.0, 7.0)))
    ctrl, backend, seen, _ = _install(_Scene(inventory=inv))
    ctrl.set_mode("gp")
    backend.fire_pick(_hit(prop_id=id(gp_actor), cell_id=0))
    assert seen[0].kind == "gp"
    assert seen[0].element_id == 1003
    assert seen[0].gp_index == 7
    assert seen[0].world == (5.0, 6.0, 7.0)


def test_gp_mode_substrate_hit_drops():
    """A substrate hit (prop not registered) in GP mode resolves to nothing."""
    ctrl, backend, seen, _ = _install(_Scene(inventory=PickInventory()))
    ctrl.set_mode("gp")
    backend.fire_pick(_hit(prop_id=12345, cell_id=0))
    assert seen == []


def test_gp_mode_no_inventory_drops():
    ctrl, backend, seen, _ = _install(_Scene(inventory=None))
    ctrl.set_mode("gp")
    backend.fire_pick(_hit(prop_id=1, cell_id=0))
    assert seen == []


# ── callback safety + mode validation ───────────────────────────────

def test_on_pick_exception_swallowed(capsys):
    def _boom(_r):
        raise RuntimeError("kaboom")
    ctrl, backend, _, _ = _install(_Scene(), on_pick=_boom)
    backend.fire_pick(_hit(world=(0.0, 0.0, 0.0)))  # must not raise
    assert "kaboom" in capsys.readouterr().err


def test_set_mode_rejects_invalid():
    ctrl = ResultsPickController()
    with pytest.raises(ValueError, match="must be one of"):
        ctrl.set_mode("bogus")


# ── uninstall delegation ────────────────────────────────────────────

def test_uninstall_delegates_to_backend():
    ctrl, backend, _, _ = _install(_Scene())
    ctrl.uninstall()
    assert backend.uninstalled == 1


# ── box pick ────────────────────────────────────────────────────────

class _Grid:
    """Minimal grid stub exposing points + cell_centers + ghost data."""
    def __init__(self, points, centers, ghost=None):
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        self.cell_data = {} if ghost is None else {"vtkGhostType": np.asarray(ghost)}

    def cell_centers(self):
        return type("C", (), {"points": self._centers})()


def _proj_xy(pts):
    """Project = take (x, y) as display coords (identity)."""
    return np.asarray(pts, dtype=np.float64)[:, :2]


def test_box_node_selects_inside():
    grid = _Grid(points=[[0, 0, 0], [5, 5, 0], [100, 100, 0]], centers=[])
    scene = _Scene(node_ids=[10, 20, 30], grid=grid)
    ctrl, backend, _, boxes = _install(scene, backend=_StubBackend(project=_proj_xy))
    backend.fire_box((-1, -1, 10, 10))   # box covers nodes 0,1 not 2
    assert len(boxes) == 1
    r = boxes[0]
    assert r.kind == "node"
    assert sorted(r.ids.tolist()) == [10, 20]
    assert r.crossing is False


def test_box_element_selects_inside_with_ghost_mask():
    grid = _Grid(
        points=[], centers=[[0, 0, 0], [5, 5, 0], [100, 100, 0]],
        # The real HIDDENCELL byte, not a literal 1: this said `1` while
        # production read 0x01 too, so the pair agreed on the wrong bit
        # and the test proved nothing (see element_visibility.HIDDENCELL).
        ghost=[0, HIDDENCELL, 0],
    )
    scene = _Scene(cell_to_element_id=[1001, 1002, 1003], grid=grid)
    ctrl, backend, _, boxes = _install(scene, backend=_StubBackend(project=_proj_xy))
    ctrl.set_mode("element")
    backend.fire_box((-1, -1, 10, 10))   # covers cells 0,1; 1 ghost-hidden
    r = boxes[0]
    assert r.kind == "element"
    assert r.ids.tolist() == [1001]      # cell 1 excluded by ghost mask
    assert r.cell_ids.tolist() == [0]


def test_box_degenerate_rectangle_drops():
    grid = _Grid(points=[[0, 0, 0]], centers=[])
    scene = _Scene(node_ids=[10], grid=grid)
    ctrl, backend, _, boxes = _install(scene, backend=_StubBackend(project=_proj_xy))
    backend.fire_box((5, 5, 5, 50))      # x0 == x1 → degenerate
    assert boxes == []


def test_box_gp_via_candidates():
    centers = np.array([[0, 0, 0], [100, 100, 0]], dtype=np.float64)
    gp_eids = np.array([1001, 1002], dtype=np.int64)
    gp_idxs = np.array([0, 1], dtype=np.int64)
    scene = _Scene(grid=_Grid(points=[], centers=[]))
    backend = _StubBackend(project=_proj_xy)
    ctrl, backend, _, boxes = _install(
        scene, gp_candidates=lambda: (centers, gp_eids, gp_idxs), backend=backend,
    )
    ctrl.set_mode("gp")
    backend.fire_box((-1, -1, 10, 10))   # covers center 0 only
    r = boxes[0]
    assert r.kind == "gp"
    assert r.ids.tolist() == [1001]
    assert r.gp_indices.tolist() == [0]


# ── _inside_box helper (kept; imported by test_element_visibility too) ─

def test_inside_box_window():
    xy = np.array([[1.0, 1.0], [5.0, 5.0], [11.0, 11.0]])
    mask = _inside_box(xy, 0, 0, 10, 10)
    assert mask.tolist() == [True, True, False]


def test_inside_box_reversed_corners_normalized():
    xy = np.array([[5.0, 5.0]])
    assert _inside_box(xy, 10, 10, 0, 0).tolist() == [True]
