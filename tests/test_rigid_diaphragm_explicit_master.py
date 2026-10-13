"""B5 (#1817, fixes #1804): ``rigid_diaphragm`` with an explicit
``master_point`` keeps the declared master.

Oracles, each naming the right answer independently of the code:

* **#1804.** In the reference frame (``model-scripts.md``,
  ``frame_modal.py``) the floor-2 point (6, 6, 6.4) coincides with joint
  ``J11_L2``.  Before the fix the nearest-node rule picked that joint
  (node 23) as master and demoted the declared, fixed and massed master
  ``floor_2_master`` (node 29) to a zero-offset slave.  The master must be
  the one node of the master label, by construction of the frame.
* **A fix on the declared master holds.**  With the master's ux fixed and
  a lateral X load on a floor corner, the floor's rigid-body ux (the
  displacement at the plan centre, where the master sits) is zero; with
  the master's ux free the same load moves it.
* **Tie-break.**  A multi-node master label whose node is coincident with
  a slave node at the point's nearest distance keeps the master-label
  node, whatever the tags.
* **Refusals.**  A one-node master off the plane through the point, and a
  multi-node master label with no node in that plane, raise naming the
  node and the plane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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
    held_master, held_joint = _floor_ux(fem, master_fix=(1, 0, 1, 1, 1, 0))
    # main: the declared master was a slave, Transformation dropped the
    # fix on its constrained ux, and the floor slid as if it were free.
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


def test_tie_preference_does_not_depend_on_tag_order():
    # Same geometry with the slave joint holding the higher tag.
    coords = {9: TIE[1], 2: TIE[2], 3: TIE[3], 5: TIE[7], 8: TIE[8]}
    rec = _resolver(coords).resolve_rigid_diaphragm(
        _defn((2.0, 2.0, STOREY)), {9, 2, 3, 5, 8}, master_nodes={5, 8})
    assert rec.master_node == 5


def test_multi_node_master_keeps_the_nearest_plane_node_without_a_tie():
    rec = _resolver(TIE).resolve_rigid_diaphragm(
        _defn((3.9, 3.9, STOREY)), {1, 2, 3, 7, 8}, master_nodes={7, 8})
    assert rec.master_node == 8


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
