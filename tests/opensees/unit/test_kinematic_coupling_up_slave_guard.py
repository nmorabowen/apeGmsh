"""A1 — ``kinematic_coupling`` refuses a u–p slave under ``dofs=None``.

The fork's ``LadrunoKinematicCoupling`` default ("every DOF the slave
has") gates on DOF *count* only (``LadrunoKinematicCoupling.cpp:260-275``):
for an ndf-4 u–p slave in 3D, component 4 is kept as a *rotation* row and
``buildB`` (``:335-350``) ties the node's pore pressure to the master's
θx — silently.  The bridge's G2 gate now fails loud at emit unless the
user passes ``dofs=`` explicitly.

End-to-end on a LadrunoUP tet box with a decoupled ndf-6 master:

* ``dofs=None`` → ``BridgeError`` naming the slave node and ``dofs=``;
* ``dofs=[1, 2, 3]`` → the deck line carries ``-dof 1 2 3``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError

_WORK_PT = (0.5, 0.5, 1.0)


def _up_box(*, dofs):
    """1×1×1 tet10 box of LadrunoUP (vertices ndf 4, mid-edge ndf 3),
    top face ``end_face`` slaved to a decoupled ndf-6 knot at ``_WORK_PT``."""
    g = apeGmsh(model_name="kc_up_guard", verbose=False)
    g.begin()
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="body")
    g.model.sync()
    g.physical.add_volume("body", name="Soil")
    g.model.select(dim=2).on_plane(
        (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), tol=1e-3,
    ).to_physical("end_face")
    h = g.decouple_node(coords=_WORK_PT, label="work_pt")
    g.constraints.kinematic_coupling(
        "work_pt", "end_face", dofs=dofs, name="footing",
    )
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.generation.set_order(2)     # tet10: the only tet u-p provider
    fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=4)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=2.0)
    ops.element.LadrunoUP(
        pg="Soil", material=mat,
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4, 1e-4, 1e-4),
    )
    ops.system.UmfPack()          # LadrunoUP refuses a symmetric system
    ops.ndf(h, ndf=6)
    return g, ops


def test_up_slave_with_default_dofs_fails_loud(tmp_path: Path) -> None:
    g, ops = _up_box(dofs=None)
    try:
        with pytest.raises(
            BridgeError, match=r"'footing'.*slave node \d+ has ndf 4.*dofs=",
        ):
            ops.tcl(str(tmp_path / "guard.tcl"))
    finally:
        g.end()


def test_up_slave_with_explicit_dofs_emits_dof_flag(tmp_path: Path) -> None:
    g, ops = _up_box(dofs=[1, 2, 3])
    try:
        out = tmp_path / "ok.tcl"
        ops.tcl(str(out))
        text = out.read_text(encoding="utf-8")
    finally:
        g.end()
    kc = [ln for ln in text.splitlines() if "LadrunoKinematicCoupling" in ln]
    assert len(kc) == 1, text
    assert kc[0].rstrip().endswith("-dof 1 2 3"), kc[0]
