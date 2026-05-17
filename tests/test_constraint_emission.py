"""
Emission-iterator routing contract (deep-review PR-B).

The NodeConstraintSet iterators are the documented OpenSees emission
surface.  Two silent-wrong defects are locked out here:

* ``rigid_link_groups()`` must NOT collapse ``rigid_diaphragm`` (→
  double-constrain, since ``rigid_diaphragms()`` also emits it) or
  ``kinematic_coupling`` (→ silent 6-DOF over-constraint, ignoring
  its DOF subset) into a full ``rigidLink``.
* ``rigid_diaphragms()`` must surface the resolved plane normal as
  ``perpDirn`` so a non-horizontal diaphragm is not silently emitted
  with a hardcoded ``3``.
"""
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh._record_set import NodeConstraintSet
from apeGmsh.mesh.records._constraints import NodeGroupRecord, NodePairRecord
from apeGmsh.mesh.records._kinds import ConstraintKind as K


# =====================================================================
# distributing_coupling / mortar are fail-loud at the API (PR-C):
# never silently emit a mechanically-wrong RBE3 / mortar record.
# =====================================================================

def test_distributing_coupling_factory_raises_not_implemented():
    with apeGmsh(model_name="pc_distrib", verbose=False) as g:
        with pytest.raises(NotImplementedError, match="RBE3|distributing"):
            g.constraints.distributing_coupling("A", "B")


def test_mortar_factory_raises_not_implemented():
    with apeGmsh(model_name="pc_mortar", verbose=False) as g:
        with pytest.raises(NotImplementedError, match="mortar|tied_contact"):
            g.constraints.mortar("A", "B")


def _diaphragm(master, slaves, normal):
    return NodeGroupRecord(
        kind=K.RIGID_DIAPHRAGM, master_node=master, slave_nodes=slaves,
        dofs=[1, 2, 3, 4, 5, 6], plane_normal=(None if normal is None
                                               else np.asarray(normal)),
    )


def _kinematic(master, slaves, dofs):
    return NodeGroupRecord(
        kind=K.KINEMATIC_COUPLING, master_node=master,
        slave_nodes=slaves, dofs=dofs,
    )


def _rigid_body(master, slaves):
    return NodeGroupRecord(
        kind=K.RIGID_BODY, master_node=master, slave_nodes=slaves,
        dofs=[1, 2, 3, 4, 5, 6],
    )


# =====================================================================
# rigid_link_groups() must exclude diaphragm + kinematic_coupling
# =====================================================================

def test_rigid_link_groups_excludes_diaphragm_and_kinematic():
    cs = NodeConstraintSet([
        _diaphragm(1, [2, 3], (0, 0, 1)),
        _kinematic(10, [11, 12], dofs=[1, 3, 5]),
        _rigid_body(20, [21]),
        NodePairRecord(kind=K.RIGID_BEAM, master_node=30, slave_node=31),
    ])
    masters = {m for m, _ in cs.rigid_link_groups()}
    # rigid_body + rigid_beam ARE rigid links; diaphragm (1) and
    # kinematic_coupling (10) are NOT.
    assert masters == {20, 30}
    assert 1 not in masters       # diaphragm not double-emitted
    assert 10 not in masters      # kinematic_coupling not 6-DOF-collapsed


def test_kinematic_coupling_still_reachable_via_pairs_with_dofs():
    cs = NodeConstraintSet([_kinematic(10, [11, 12], dofs=[1, 3, 5])])
    kp = [p for p in cs.pairs() if p.kind == K.KINEMATIC_COUPLING]
    assert len(kp) == 2
    assert all(p.master_node == 10 for p in kp)
    assert {p.slave_node for p in kp} == {11, 12}
    # The DOF subset is preserved (this is what the 6-DOF collapse
    # silently destroyed).
    assert all(p.dofs == [1, 3, 5] for p in kp)


# =====================================================================
# rigid_diaphragms() must surface perpDirn from the plane normal
# =====================================================================

def test_rigid_diaphragms_yields_perp_dirn_from_normal():
    cs = NodeConstraintSet([
        _diaphragm(1, [2, 3], (0, 0, 1)),     # XY plane  → perp 3
        _diaphragm(4, [5], (1, 0, 0)),        # YZ plane  → perp 1
        _diaphragm(6, [7], (0, 1, 0)),        # XZ plane  → perp 2
        _diaphragm(8, [9], None),             # missing   → 3 (legacy)
    ])
    got = {master: perp for perp, master, _ in cs.rigid_diaphragms()}
    assert got == {1: 3, 4: 1, 6: 2, 8: 3}


def test_diaphragm_not_double_emitted():
    """A rigid_diaphragm must appear in rigid_diaphragms() ONLY —
    never also in rigid_link_groups() (that double-constrains)."""
    cs = NodeConstraintSet([_diaphragm(1, [2, 3], (0, 0, 1))])
    diap_masters = {m for _, m, _ in cs.rigid_diaphragms()}
    link_masters = {m for m, _ in cs.rigid_link_groups()}
    assert diap_masters == {1}
    assert link_masters == set()          # NOT also here
