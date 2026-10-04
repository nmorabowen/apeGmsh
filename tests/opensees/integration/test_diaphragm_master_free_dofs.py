"""#1333 end to end — a detached ``rigid_diaphragm`` master warns at
build, and the fix it names restores the closed-form modes.

Two columns (``L = 1``, cantilever in both bending planes because the
diaphragm leaves the tops' ``rx, ry`` free), tops tied to a lone master
at the floor's centre with lumped mass ``M`` on ``ux, uy`` and ``I`` on
``rz``.  Closed form::

    k_col = 3 E I / L^3                      (cantilever, each column)
    ω²_x = ω²_y = 2 k_col / M
    ω²_θ = (2 k_col d² + 2 G J / L) / I      (d = 0.5, the lever arm)

The free-DOF run is asserted on the warning, never on ``analyze``'s
return code: with the master's ``uz, rx, ry`` free, a static analyze
returns 0 after "matrix singular" and eigen reports periods of ~1e4 s.
"""
from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import NodeGroupRecord
from apeGmsh._kernel.records._kinds import ConstraintKind
from apeGmsh.opensees import DetachedDiaphragmMasterWarning, apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

E, G, I_SEC, J, L, D = 200e9, 80e9, 1e-4, 1e-4, 1.0, 0.5
M, I_RZ = 1e3, 1e4
K_COL = 3 * E * I_SEC / L**3
OMEGA2_TRANS = 2 * K_COL / M
OMEGA2_TORSION = (2 * K_COL * D**2 + 2 * G * J / L) / I_RZ
FIX_MASK = (0, 0, 1, 1, 1, 0)

# The gate (``validate_diaphragm_master_stiffness``) is implemented and
# unit-tested, but its call from ``apeSees.build()`` is not wired yet
# (apesees.py was under a concurrent edit).  Strict: the marks come off
# the moment the call lands, or these XPASS and fail.
_AWAITING_BUILD_CALL = pytest.mark.xfail(
    strict=True,
    reason="#1333: validate_diaphragm_master_stiffness not yet called "
           "from apeSees.build()",
)


def _frame_with_master() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5],
        coords=[
            (0.0, 0.0, 0.0), (0.0, 0.0, L),
            (2 * D, 0.0, 0.0), (2 * D, 0.0, L),
            (D, 0.0, L),
        ],
        node_pgs={"Base": [1, 3], "Top": [2, 4], "Master": [5]},
    )
    elements = _ElementsStub(elem_pgs={
        "Cols": _ElementGroupView(ids=(1, 2), connectivity=((1, 2), (3, 4))),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.add_node_constraints([NodeGroupRecord(
        kind=ConstraintKind.RIGID_DIAPHRAGM,
        master_node=5, slave_nodes=[2, 4], dofs=[1, 2, 6],
        plane_normal=np.array([0.0, 0.0, 1.0]), name="floor",
    )])
    return fem


def _bridge(fem: FEMStub, *, fix_master: bool) -> apeSees:
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf, A=0.01, E=E, Iz=I_SEC, Iy=I_SEC, G=G, J=J,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    if fix_master:
        ops.fix(pg="Master", dofs=FIX_MASK)
    ops.mass(pg="Master", values=(M, M, 0.0, 0.0, 0.0, I_RZ))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    return ops


@_AWAITING_BUILD_CALL
def test_detached_master_warns_on_the_deck_route(tmp_path) -> None:
    ops = _bridge(_frame_with_master(), fix_master=False)
    with pytest.warns(DetachedDiaphragmMasterWarning, match=r"master node 5") as caught:
        ops.tcl(tmp_path / "free.tcl")
    assert f"ops.fix(nodes=(5,), dofs={FIX_MASK})" in str(caught[0].message)


def test_fixed_master_is_silent_on_the_deck_route(tmp_path) -> None:
    ops = _bridge(_frame_with_master(), fix_master=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetachedDiaphragmMasterWarning)
        ops.tcl(tmp_path / "fixed.tcl")


@pytest.mark.live
@_AWAITING_BUILD_CALL
def test_fix_named_by_the_warning_restores_the_closed_form_modes() -> None:
    pytest.importorskip("openseespy.opensees")
    expected = np.sort([OMEGA2_TORSION, OMEGA2_TRANS, OMEGA2_TRANS])

    # The hazard: the build warns.  (Not asserted on the return code —
    # analyze returns 0 on this singular K, with garbage displacements.)
    free = _bridge(_frame_with_master(), fix_master=False)
    with pytest.warns(DetachedDiaphragmMasterWarning, match=r"master node 5"):
        free.eigen(3, solver="-fullGenLapack")

    # The fix the warning names: silent, and the modes are the closed form.
    fixed = _bridge(_frame_with_master(), fix_master=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DetachedDiaphragmMasterWarning)
        result = fixed.eigen(3, solver="-fullGenLapack")
    got = np.sort(np.asarray(list(result.eigenvalues), dtype=float))
    assert got == pytest.approx(expected, rel=1e-6), (got, expected)
