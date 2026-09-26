"""Live check of ``PlateRebar`` / ``PlateFromPlaneStress`` shell layers.

One ``ASDShellQ4`` (1 x 1, in the XY plane, local x = global X) with a
four-layer ``LayeredShell`` section — concrete (via ``PlateFromPlaneStress``),
two ``PlateRebar`` layers of different thickness, concrete — pulled in
uniaxial membrane tension along X. With ``nu = 0`` the strain field is
uniform and the axial stiffness is closed-form::

    k = b / L * (E_c * h_c + E_s * sum(t_i * cos(angle_i)**4))

so a bar at 0 deg adds ``E_s * t`` and a bar at 90 deg adds nothing. Swapping
the two bar angles changes which thickness counts, which checks the emitted
argument order, the dependency order (uniaxial before ``PlateRebar``, nD
before ``PlateFromPlaneStress``) and the angle convention (degrees from the
shell local x axis) against a real build.

Gated by the ``live`` marker; needs ``openseespy`` (stock or the fork — both
materials are stock OpenSees).
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.section.plate import ShellLayer

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

E_C, H_C = 30e9, 0.2
E_S = 200e9
T_A, T_B = 0.002, 0.0005
P = 1.0e5


def _one_quad() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        node_pgs={"All": [1, 2, 3, 4]},
    )
    elements = _ElementsStub(
        elem_pgs={"Wall": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),))},
    )
    return FEMStub(nodes=nodes, elements=elements)


def _pull(angle_a: float, angle_b: float) -> float:
    ops = apeSees(cast("object", _one_quad()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.ElasticMaterial(E=E_S)
    conc = ops.nDMaterial.ElasticIsotropic(E=E_C, nu=0.0)
    conc_layer = ops.nDMaterial.PlateFromPlaneStress(material=conc, G_out=E_C / 2)
    bar_a = ops.nDMaterial.PlateRebar(material=steel, angle=angle_a)
    bar_b = ops.nDMaterial.PlateRebar(material=steel, angle=angle_b)
    # ``LayeredShell`` builds the C++ LayeredShellFiberSection; it is the
    # only keyword the Tcl and Python interpreters register for it (the
    # ``LayeredShellFiberSection`` primitive's own token is unknown to both),
    # and the C++ parser wants at least three layers.
    sec = ops.section.LayeredShell(layers=(
        ShellLayer(material=conc_layer, thickness=H_C / 2),
        ShellLayer(material=bar_a, thickness=T_A),
        ShellLayer(material=bar_b, thickness=T_B),
        ShellLayer(material=conc_layer, thickness=H_C / 2),
    ))
    ops.element.ASDShellQ4(pg="Wall", section=sec)

    # Membrane-only: out-of-plane translation and all rotations held.
    ops.fix(pg="All", dofs=(0, 0, 1, 1, 1, 1))
    ops.fix(nodes=[1], dofs=(1, 1, 0, 0, 0, 0))
    ops.fix(nodes=[4], dofs=(1, 0, 0, 0, 0, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        for n in (2, 3):
            p.load(node=n, forces=(P / 2, 0.0, 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    return float(emitter.ops.nodeDisp(2, 1))


@pytest.mark.live
@pytest.mark.parametrize(
    ("angle_a", "angle_b", "t_along_x"),
    [(0.0, 90.0, T_A), (90.0, 0.0, T_B)],
)
def test_plate_rebar_layers_add_axial_stiffness_along_their_angle(
    angle_a: float, angle_b: float, t_along_x: float,
) -> None:
    expected = P / (E_C * H_C + E_S * t_along_x)
    assert _pull(angle_a, angle_b) == pytest.approx(expected, rel=1e-6)
