"""#1333 — a ``rigid_diaphragm`` master that no element touches has DOFs
nothing stiffens; :func:`validate_diaphragm_master_stiffness` warns.

Oracle: ``rigidDiaphragm`` with ``perpDirn=3`` ties exactly ``(ux, uy,
rz)`` of a 6-DOF node (``RigidDiaphragm.cpp:176-187``), so a master in
no element has ``uz, rx, ry`` free unless a ``fix`` / ``sp`` / another
constraint holds them.  Each case below names the DOF set the gate must
report, and the silent cases name what closes it.  Warn-as-contract:
the silent cases run with the category promoted to an error.
"""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import DetachedDiaphragmMasterWarning, apeSees
from apeGmsh.opensees._internal.build import (
    FixRecord,
    SupportRecord,
    validate_diaphragm_master_stiffness,
)
from apeGmsh.opensees.pattern.pattern import _SPRecord

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _frame_with_master(master: int = 5) -> FEMStub:
    """Two columns (1-2, 3-4) and a free-standing node 5 at the floor's
    centre; the diaphragm ties the column tops to ``master``."""
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5],
        coords=[
            (0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0), (1.0, 0.0, 1.0),
            (0.5, 0.0, 1.0),
        ],
        node_pgs={"Base": [1, 3], "Top": [2, 4], "Master": [5]},
    )
    elements = _ElementsStub(
        elem_pgs={
            "Cols": _ElementGroupView(ids=(1, 2), connectivity=((1, 2), (3, 4))),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.add_node_constraints([_diaphragm(master)])
    return fem


def _diaphragm(master: int, name: str | None = "floor") -> NodeGroupRecord:
    return NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM,
        master_node=master, slave_nodes=[2, 4], dofs=[1, 2, 6],
        plane_normal=np.array([0.0, 0.0, 1.0]), name=name,
    )


def _column_specs(fem: FEMStub) -> list:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    return [ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )]


def _run(fem: FEMStub, **kw) -> None:
    validate_diaphragm_master_stiffness(
        cast("object", fem), _column_specs(fem), 3, 6, {1: 6, 2: 6, 3: 6, 4: 6},
        **kw,
    )


def _assert_silent(fem: FEMStub, **kw) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetachedDiaphragmMasterWarning)
        _run(fem, **kw)


# ---------------------------------------------------------------------------
# The hazard
# ---------------------------------------------------------------------------


def test_detached_master_warns_naming_the_free_dofs_and_the_fix() -> None:
    with pytest.warns(DetachedDiaphragmMasterWarning) as caught:
        _run(_frame_with_master())
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "rigid_diaphragm 'floor': master node 5 is attached to no element" in msg
    assert "uz, rx, ry (3, 4, 5)" in msg
    assert "ops.fix(nodes=(5,), dofs=(0, 0, 1, 1, 1, 0))" in msg
    assert "#1333" in msg


def test_partial_fix_warns_on_the_remaining_dofs() -> None:
    with pytest.warns(DetachedDiaphragmMasterWarning, match=r"rx, ry \(4, 5\)"):
        _run(
            _frame_with_master(),
            fix_records=[FixRecord(pg=None, nodes=(5,), dofs=(0, 0, 1, 0, 0, 0))],
        )


def test_stage_claimed_diaphragm_is_covered() -> None:
    fem = _frame_with_master()
    rec = fem.nodes.constraints._records.pop()  # claimed: leaves the broker set
    with pytest.warns(DetachedDiaphragmMasterWarning, match="master node 5"):
        _run(fem, stage_constraint_records=[rec])


def test_two_floors_on_detached_masters_report_one_entry_each() -> None:
    fem = _frame_with_master()
    fem.nodes._pgs["Master"].append(6)
    fem.nodes._ids = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    fem.nodes._coords = np.vstack([fem.nodes._coords, [(0.5, 0.0, 2.0)]])
    fem.nodes._id_to_idx[6] = 5
    fem.add_node_constraints([_diaphragm(5, "floor1"), _diaphragm(6, "floor2")])
    with pytest.warns(DetachedDiaphragmMasterWarning) as caught:
        _run(fem)
    msg = str(caught[0].message)
    assert "'floor1': master node 5" in msg and "'floor2': master node 6" in msg


def test_vertical_wall_diaphragm_frees_the_in_wall_normal_and_rotations() -> None:
    # perpDirn 2 (normal y) ties (ux, uz, ry) -> free uy, rx, rz.
    fem = _frame_with_master()
    fem.add_node_constraints([NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM, master_node=5, slave_nodes=[2, 4],
        dofs=[1, 3, 5], plane_normal=np.array([0.0, 1.0, 0.0]), name="wall",
    )])
    with pytest.warns(DetachedDiaphragmMasterWarning, match=r"uy, rx, rz \(2, 4, 6\)"):
        _run(fem)


# ---------------------------------------------------------------------------
# What closes the hole — every one of these stays silent under -W error
# ---------------------------------------------------------------------------


def test_full_fix_on_the_free_dofs_is_silent() -> None:
    _assert_silent(
        _frame_with_master(),
        fix_records=[FixRecord(pg=None, nodes=(5,), dofs=(0, 0, 1, 1, 1, 0))],
    )


def test_fix_by_pg_is_silent() -> None:
    _assert_silent(
        _frame_with_master(),
        fix_records=[FixRecord(pg="Master", nodes=None, dofs=(0, 0, 1, 1, 1, 0))],
    )


def test_support_plus_sp_is_silent() -> None:
    _assert_silent(
        _frame_with_master(),
        fix_records=[SupportRecord(pg=None, nodes=(5,), dofs=(0, 0, 0, 1, 1, 0))],
        sp_records=[_SPRecord(target_kind="node", target="5", dof=3, value=0.0)],
    )


def test_attached_master_is_silent() -> None:
    # The master is a column top: the beam carries every DOF.
    _assert_silent(_frame_with_master(master=2))


def test_master_retained_by_a_rigid_link_is_silent() -> None:
    fem = _frame_with_master()
    fem.add_node_constraints([
        _diaphragm(5),
        NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM, master_node=5, slave_node=2,
            dofs=[1, 2, 3, 4, 5, 6], offset=np.array([-0.5, 0.0, 0.0]),
        ),
    ])
    _assert_silent(fem)


def test_master_held_by_equal_dof_on_the_free_dofs_is_silent() -> None:
    fem = _frame_with_master()
    fem.add_node_constraints([
        _diaphragm(5),
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, master_node=2, slave_node=5,
            dofs=[3, 4, 5],
        ),
    ])
    _assert_silent(fem)


def test_master_held_by_equal_dof_on_other_dofs_still_warns() -> None:
    fem = _frame_with_master()
    fem.add_node_constraints([
        _diaphragm(5),
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, master_node=2, slave_node=5,
            dofs=[3],
        ),
    ])
    with pytest.warns(DetachedDiaphragmMasterWarning, match=r"rx, ry \(4, 5\)"):
        _run(fem)


def test_master_in_a_surface_coupling_is_left_alone() -> None:
    # No per-DOF set the bridge can read -> assumed stiffened, silent.
    fem = _frame_with_master()
    fem.add_surface_constraints([InterpolationRecord(
        kind=ConstraintKind.TIE, slave_node=5, master_nodes=[2, 4],
        weights=np.array([0.5, 0.5]),
    )])
    _assert_silent(fem)


def test_no_diaphragm_is_a_no_op() -> None:
    fem = _frame_with_master()
    fem.add_node_constraints([])
    _assert_silent(fem)
