"""Live check of ``PlateRebar`` / ``PlateFromPlaneStress`` shell layers.

One ``ASDShellQ4`` (1 x 1, in the XY plane, local x = global X) with a
four-layer layered section — concrete (via ``PlateFromPlaneStress``),
two ``PlateRebar`` layers of different thickness, concrete — pulled in
uniaxial membrane tension along X. With ``nu = 0`` the strain field is
uniform and the axial stiffness is closed-form::

    k = b / L * (E_c * h_c + E_s * sum(t_i * cos(angle_i)**4))

where ``angle_i`` is measured from global X. A bar along X adds
``E_s * t`` and a bar across it adds nothing. Swapping the two bar angles
changes which thickness counts. That checks the emitted argument order, the
dependency order (uniaxial before ``PlateRebar``, nD before
``PlateFromPlaneStress``) and that ``angle`` is in degrees.

**Which bar lies along X depends on the build.** ``PlateRebar``'s angle is
measured in the section frame, and ``ASDShellQ4`` rotates the strain from
its element frame into that section frame by an angle it computes in
``setDomain``. Without ``-local``, the section x axis should be the
mid-side vector from edge 1-4 to edge 2-3, which is +X for this square:

* **Fork, and upstream from PR #1606 (merged 2025-05-16):** the section x
  axis is that mid-side vector. 0 deg lies along X.
* **Older upstream, including PyPI openseespy 3.7.1.x (the CI
  ``live-stock`` job):** the default branch declares a second ``e1`` that
  hides the outer one. The outer ``e1`` stays zero, so the angle becomes
  ``acos(0) = +90`` deg. The section x axis is then the element local y
  axis, and 90 deg lies along X.

The build-independent assertions are therefore that the two stiffnesses
match the expected pair {with T_A along X, with T_B along X} in either
order, and that swapping the angles swaps them. Measured: 1.5625e-5 and
1.6393e-5 m, in opposite order on the two builds. The ``-local`` option
takes the other code path, which is correct on both builds.

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


def _pull(angle_a: float, angle_b: float, section_ns: str = "LayeredShell") -> float:
    ops = apeSees(cast("object", _one_quad()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)

    steel = ops.uniaxialMaterial.ElasticMaterial(E=E_S)
    conc = ops.nDMaterial.ElasticIsotropic(E=E_C, nu=0.0)
    conc_layer = ops.nDMaterial.PlateFromPlaneStress(material=conc, G_out=E_C / 2)
    bar_a = ops.nDMaterial.PlateRebar(material=steel, angle=angle_a)
    bar_b = ops.nDMaterial.PlateRebar(material=steel, angle=angle_b)
    # Both primitives emit ``section LayeredShell`` (the only keyword the
    # interpreters register for the C++ LayeredShellFiberSection); the C++
    # parser wants at least three layers.
    sec = getattr(ops.section, section_ns)(layers=(
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


def _expected(t_along_x: float) -> float:
    return P / (E_C * H_C + E_S * t_along_x)


@pytest.mark.live
def test_plate_rebar_layers_add_axial_stiffness_along_their_angle() -> None:
    u_0_90 = _pull(0.0, 90.0)
    u_90_0 = _pull(90.0, 0.0)
    # The two orientations give the expected pair of answers; which one
    # belongs to which depends on the build's ASDShellQ4 section frame
    # (see the module docstring).
    assert sorted([u_0_90, u_90_0]) == pytest.approx(
        sorted([_expected(T_A), _expected(T_B)]), rel=1e-6,
    )
    # And the angle matters: swapping the bars changes the stiffness.
    assert u_0_90 != pytest.approx(u_90_0, rel=1e-3)


@pytest.mark.live
def test_plate_rebar_angle_is_a_direction_not_a_sense() -> None:
    # Build-independent: 0 and 180 deg are the same bar, on any section frame.
    assert _pull(180.0, 90.0) == pytest.approx(_pull(0.0, 90.0), rel=1e-9)


@pytest.mark.live
def test_layered_shell_fiber_section_keyword_parses_live() -> None:
    """Regression: ``LayeredShellFiberSection`` used to emit a keyword that no
    interpreter registers (``section type LayeredShellFiberSection is
    unknown``). It now emits ``LayeredShell``, which builds, and gives the
    same answer as the ``LayeredShell`` primitive."""
    got = _pull(0.0, 90.0, section_ns="LayeredShellFiberSection")
    assert got == pytest.approx(_pull(0.0, 90.0, section_ns="LayeredShell"), rel=1e-12)
    # A real layered answer, on either build's section frame.
    assert min(abs(got / _expected(t) - 1.0) for t in (T_A, T_B)) < 1e-6
