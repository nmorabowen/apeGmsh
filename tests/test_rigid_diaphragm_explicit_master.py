"""B5 (#1817, fixes #1804; #1838, fixes #1829): ``rigid_diaphragm`` with
an explicit ``master_point`` keeps the declared master.

Oracles, each naming the right answer independently of the code:

* **#1804.** In the reference frame (``model-scripts.md``,
  ``frame_modal.py``) the floor-2 point (6, 6, 6.4) coincides with joint
  ``J11_L2``.  Before the fix the nearest-node rule picked that joint
  (node 23) as master and demoted the declared, fixed and massed master
  ``floor_2_master`` (node 29) to a zero-offset slave.  The master must be
  the one node of the master label, by construction of the frame.
* **A fix on the declared master holds.**  With the master held in plane and
  a lateral X load on a floor corner, the floor's rigid-body ux (the
  displacement at the plan centre, where the master sits) is zero; with
  the master's ux free the same load moves it.
* **Multi-node master label (#1804 ties; #1829).**  The master is the
  master-label plane node nearest the point.  A slave-label node is never
  promoted, whether it is coincident with a master node or strictly
  nearer the point than every master node.
* **#1829, live.**  The frame with floor 2's master label widened to
  {``floor_2_master``, ``J22_L2``} and the point next to slave joint
  ``J00_L2``: the master stays ``floor_2_master``, and holding it in plane
  holds the floor.
* **Refusals.**  A one-node master off the plane through the point, and a
  multi-node master label with no node in that plane, raise naming the
  node and the plane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.defs.constraints import RigidDiaphragmDef
from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver

from .test_rigid_diaphragm_master_default import (
    _bridge, _diaphragms, _frame, _ids,
)

STOREY = 3.2


@pytest.fixture(scope="module")
def fem():
    return _frame(given=True)


# ---------------------------------------------------------------------------
# #1804 — frame_modal floor 2 with the explicit point
# ---------------------------------------------------------------------------

def test_1804_floor_2_master_is_the_declared_node(fem):
    for k, (perp, master, slaves) in zip((1, 2), _diaphragms(fem)):
        (declared,) = _ids(fem, pg=f"floor_{k}_master")
        joints = _ids(fem, pg=f"floor_{k}_joints")
        assert perp == 3
        # main (ae97bed3), floor 2: master 23 (J11_L2), and the declared
        # master 29 among the slaves.
        assert master == declared
        assert slaves == joints
        assert declared not in slaves


def test_1804_the_deck_ties_the_floor_to_the_fixed_master(fem, tmp_path: Path):
    deck = tmp_path / "given.tcl"
    _bridge(fem).tcl(str(deck))
    lines = deck.read_text().splitlines()
    rd = [ln.split() for ln in lines if ln.startswith("rigidDiaphragm")]
    fixed = {ln.split()[1] for ln in lines if ln.startswith("fix ")}
    for k, parts in zip((1, 2), rd):
        (declared,) = _ids(fem, pg=f"floor_{k}_master")
        assert parts[2] == str(declared)
        assert str(declared) not in parts[3:]
        assert str(declared) in fixed


#: The master held on every in-plane DOF (ux, uy, rz). A master held on
#: some in-plane DOFs only is refused under Transformation (#2427); held
#: fully, a demoted master (#1804) still lets the floor slide.
HELD = (1, 1, 1, 1, 1, 1)


def _floor_ux(fem, *, master_fix) -> tuple[float, float]:
    """Floor-2 ux at the declared master and at the coincident joint under
    a 100 kN X load on corner joint J00_L2."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    ops = _bridge(fem, master_fix=master_fix)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="J00_L2"),
               forces=(1.0e5, 0.0, 0.0, 0.0, 0.0, 0.0))
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    (master,) = _ids(fem, pg="floor_2_master")
    (joint,) = _ids(fem, label="J11_L2")
    return live.nodeDisp(master, 1), live.nodeDisp(joint, 1)


@pytest.mark.live
def test_1804_a_fix_on_the_declared_master_constrains_the_floor(fem):
    pytest.importorskip("openseespy.opensees")
    free_master, free_joint = _floor_ux(fem, master_fix=(0, 0, 1, 1, 1, 0))
    # Not vacuous: the load moves the floor when the master's ux is free.
    assert free_master > 1e-5
    assert free_joint == pytest.approx(free_master, rel=1e-9)
    held_master, held_joint = _floor_ux(fem, master_fix=HELD)
    # main: the declared master was a slave, Transformation dropped the
    # fix on its constrained ux, and the floor slid as if it were free.
    assert abs(held_master) < 1e-12
    assert abs(held_joint) < 1e-12


# ---------------------------------------------------------------------------
# #1829 — a multi-node master label with a strictly nearer slave
# ---------------------------------------------------------------------------

def _frame_wide_master():
    """frame_modal with floor 2's master label widened to two nodes,
    ``floor_2_master`` (plan centre) and corner joint ``J22_L2``, and the
    floor-2 point placed next to corner joint ``J00_L2``, a slave.  The
    ``floor_k_master`` groups that ``_bridge`` fixes and masses keep one
    node each."""
    bay, n = 6.0, 2
    with apeGmsh(model_name="b5j_wide_master") as g:
        joints: dict[int, list[str]] = {}
        for k in range(3):
            joints[k] = []
            for i in range(n + 1):
                for j in range(n + 1):
                    name = f"J{i}{j}_L{k}"
                    g.model.geometry.add_point(i * bay, j * bay, k * STOREY,
                                               label=name)
                    joints[k].append(name)
        for k in (1, 2):
            g.model.geometry.add_point(bay, bay, k * STOREY,
                                       label=f"floor_{k}_master")
        cols, beams = [], []
        for k in (1, 2):
            for i in range(n + 1):
                for j in range(n + 1):
                    cols.append(f"C{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k - 1}",
                                              f"J{i}{j}_L{k}", label=cols[-1])
            for i in range(n):
                for j in range(n + 1):
                    beams.append(f"BX{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k}",
                                              f"J{i + 1}{j}_L{k}", label=beams[-1])
            for i in range(n + 1):
                for j in range(n):
                    beams.append(f"BY{i}{j}_{k}")
                    g.model.geometry.add_line(f"J{i}{j}_L{k}",
                                              f"J{i}{j + 1}_L{k}", label=beams[-1])
        g.physical.add_curve(cols, name="columns")
        g.physical.add_curve(beams, name="beams")
        g.physical.add_point(joints[0], name="column_bases")
        for k in (1, 2):
            g.physical.add_point(joints[k], name=f"floor_{k}_joints")
            g.physical.add_point([f"floor_{k}_master"], name=f"floor_{k}_master")
        g.physical.add_point(["floor_2_master", "J22_L2"], name="floor_2_master_set")
        g.constraints.rigid_diaphragm(
            "floor_1_master", "floor_1_joints", master_point=(bay, bay, STOREY),
            plane_normal=(0, 0, 1), plane_tolerance=0.05)
        g.constraints.rigid_diaphragm(
            "floor_2_master_set", "floor_2_joints",
            master_point=(0.1, 0.1, 2 * STOREY),
            plane_normal=(0, 0, 1), plane_tolerance=0.05)
        g.mesh.structured.set_transfinite_curve("columns", n_nodes=2)
        g.mesh.structured.set_transfinite_curve("beams", n_nodes=2)
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


@pytest.fixture(scope="module")
def wide():
    return _frame_wide_master()


def test_1829_floor_2_master_stays_in_the_master_label(wide):
    (_, master, slaves) = _diaphragms(wide)[1]
    (declared,) = _ids(wide, pg="floor_2_master")
    (corner,) = _ids(wide, label="J00_L2")
    # main (0492b8ac): the master was J00_L2, a slave-label joint 0.14 m
    # from the point, and floor_2_master was one of its slaves.
    assert master == declared
    assert corner in slaves
    assert sorted(slaves) == [t for t in _ids(wide, pg="floor_2_joints")]


@pytest.mark.live
def test_1829_a_fix_on_the_declared_master_constrains_the_floor(wide):
    pytest.importorskip("openseespy.opensees")
    free_master, _ = _floor_ux(wide, master_fix=(0, 0, 1, 1, 1, 0))
    assert free_master > 1e-5
    held_master, held_joint = _floor_ux(wide, master_fix=HELD)
    # main: floor_2_master was a slave, Transformation dropped the fix on
    # its constrained ux, and the floor slid as if it were free.
    assert abs(held_master) < 1e-12
    assert abs(held_joint) < 1e-12


# ---------------------------------------------------------------------------
# Resolver unit oracles
# ---------------------------------------------------------------------------

def _resolver(coords: dict[int, tuple[float, float, float]]) -> ConstraintResolver:
    tags = np.array(sorted(coords), dtype=int)
    return ConstraintResolver(tags, np.array([coords[t] for t in tags]))


def _defn(point, **kw) -> RigidDiaphragmDef:
    return RigidDiaphragmDef(master_label="m", slave_label="s",
                             master_point=point, plane_tolerance=0.01, **kw)


# Node 1 is a slave joint coincident with the point; 7 and 8 are the
# master-label nodes, 7 coincident with joint 1 and 8 elsewhere.
TIE = {1: (2.0, 2.0, STOREY), 2: (0.0, 0.0, STOREY), 3: (4.0, 0.0, STOREY),
       7: (2.0, 2.0, STOREY), 8: (4.0, 4.0, STOREY)}


def test_tie_prefers_the_master_label_node():
    rec = _resolver(TIE).resolve_rigid_diaphragm(
        _defn((2.0, 2.0, STOREY)), {1, 2, 3, 7, 8}, master_nodes={7, 8})
    # Before: node 1 (a slave-label joint) at zero distance became master.
    assert rec.master_node == 7
    assert sorted(rec.slave_nodes) == [1, 2, 3, 8]


@pytest.mark.parametrize("point, slave", [
    ((0.1, 0.1, STOREY), 2),     # slave 2 at 0.14; master 7 at 2.69
    ((4.0, 0.5, STOREY), 3),     # slave 3 at 0.5; master 7 at 2.5
])
def test_1829_a_strictly_nearer_slave_is_never_the_master(point, slave):
    rec = _resolver(TIE).resolve_rigid_diaphragm(
        _defn(point), {1, 2, 3, 7, 8}, master_nodes={7, 8})
    # main (0492b8ac): the slave-label node, strictly nearest the point,
    # became master and demoted the declared master set.
    assert rec.master_node == 7
    assert slave in rec.slave_nodes
    assert sorted(rec.slave_nodes) == [1, 2, 3, 8]


def test_one_node_master_wins_over_a_nearer_slave():
    # The point sits on slave joint 1; the master label names node 8 alone.
    rec = _resolver(TIE).resolve_rigid_diaphragm(
        _defn((2.0, 2.0, STOREY)), {1, 2, 3, 8}, master_nodes={8})
    assert rec.master_node == 8
    np.testing.assert_allclose(
        rec.offsets[rec.slave_nodes.index(1)], [-2.0, -2.0, 0.0])


def test_off_plane_one_node_master_is_refused():
    coords = dict(TIE)
    coords[8] = (4.0, 4.0, 2 * STOREY)      # the master is a floor above
    with pytest.raises(ValueError,
                       match=r"master node 8 at \(4\.0, 4\.0, 6\.4\) lies "
                             r"3\.2 off the plane through master_point="
                             r"\(2\.0, 2\.0, 3\.2\)"):
        _resolver(coords).resolve_rigid_diaphragm(
            _defn((2.0, 2.0, STOREY)), {1, 2, 3, 8}, master_nodes={8})


def test_multi_node_master_with_no_node_in_the_plane_is_refused():
    coords = dict(TIE)
    coords[7] = (2.0, 2.0, 2 * STOREY)
    coords[8] = (4.0, 4.0, 2 * STOREY)
    with pytest.raises(ValueError,
                       match=r"none of the 2 node\(s\) of master 'm' lies in "
                             r"the plane through master_point"):
        _resolver(coords).resolve_rigid_diaphragm(
            _defn((2.0, 2.0, STOREY)), {1, 2, 3, 7, 8}, master_nodes={7, 8})


def test_frame_floor_with_a_point_in_the_wrong_storey_is_refused(fem):
    """The floor-2 master with the floor-1 point: the plane collects the
    floor-1 joints, and the declared master lies a storey above it."""
    from apeGmsh._kernel.resolvers._chain_phase_router import route_def_to_fem

    defn = RigidDiaphragmDef(master_label="floor_2_master",
                             slave_label="floor_1_joints",
                             master_point=(6.0, 6.0, STOREY),
                             plane_tolerance=0.05)
    (declared,) = _ids(fem, pg="floor_2_master")
    with pytest.raises(ValueError, match=rf"master node {declared} at "
                                         r"\(6\.0, 6\.0, 6\.4\)"):
        route_def_to_fem(fem, defn)
