"""Live static acceptance: a ``MultiLinear`` fuse reproduces its
trilinear force–deformation law, including last-slope extrapolation.

Gated by the ``live`` marker. ``LiveOpsEmitter`` locates the Ladruno
fork itself (set ``APEGMSH_OPENSEES_BIN``); there is no ``openseespy``
importorskip — that package is not in this venv and would skip the
module silently.

Physics: a characterised structural fuse (yielding splice) whose
specification *is* the trilinear law, carried on a ``zeroLength`` in
global dir 2. Units are kN and m. The positive-branch breakpoints
are ``(1.25 mm, 781.58 kN)``, ``(4.8 mm, 942.95 kN)``,
``(15 mm, 1079.80 kN)``; stiffnesses K1 / K2 / K3 follow as the
segment secants. The material is odd-symmetric (``F(-u) == -F(u)``).

Past the last breakpoint OpenSees ``MultiLinear::setTrialStrain``
clamps its branch search to ``numSlope - 1``, so the backbone keeps
K3 rather than capping or going flat. That is load-bearing for a
device with no bearing backstop at this rung, and it is exactly the
behaviour a rewrite silently loses.

The only free DOF is DisplacementControl's, so each increment is
kinematically determined: convergence is immediate and the measured
force is the material response at the prescribed deformation.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

from tests.opensees.fixtures.fem_stub import make_two_node_beam

_POINTS: tuple[tuple[float, float], ...] = (
    (0.0012499999999999998, 781.5821391014429),
    (0.004799999999999999, 942.946692139763),
    (0.015000000000000081, 1079.7965946244644),
)
_F_AT_15_MM = 1079.7965946244644
_N_STEPS = 320
_U_END = 16e-3
_DU = -_U_END / _N_STEPS  # compression; −5e-5 m per step


def _backbone(u: float) -> float:
    """Analytic trilinear force at deformation ``u``; odd-symmetric, and
    linear on the LAST slope past the last breakpoint (no cap)."""
    if u == 0.0:
        return 0.0
    sign = 1.0 if u > 0.0 else -1.0
    x = abs(u)
    u_prev, f_prev = 0.0, 0.0
    for ui, fi in _POINTS:
        if x <= ui:
            return sign * (
                f_prev + (fi - f_prev) * (x - u_prev) / (ui - u_prev)
            )
        u_prev, f_prev = ui, fi
    u_last, f_last = _POINTS[-1]
    u_prev, f_prev = _POINTS[-2]
    k_last = (f_last - f_prev) / (u_last - u_prev)
    return sign * (f_last + k_last * (x - u_last))


@pytest.mark.live
def test_multilinear_fuse_backbone_and_last_slope_extrapolation() -> None:
    fem = make_two_node_beam()  # node 1 @origin (Base), node 2; line "Cols"
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=3)

    mat = ops.uniaxialMaterial.MultiLinear(points=_POINTS)
    ops.element.ZeroLength(
        pg="Cols",
        mat_dirs=(ZeroLengthMatDir(material=mat, dof=2),),
    )

    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(nodes=(2,), dofs=(1, 0, 1))

    # DisplacementControl needs a non-zero reference load on the
    # controlled DOF (else domainChanged aborts with "zero reference
    # load"). Magnitude is irrelevant — it solves for the load factor —
    # so this unit load does not contaminate the measured force.
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(0.0, -1.0, 0.0))

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=25)
    ops.algorithm.Newton()
    ops.integrator.DisplacementControl(node=2, dof=2, dU=_DU)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)

    us: list[float] = []
    forces: list[float] = []
    for _ in range(_N_STEPS):
        assert emitter.analyze(steps=1) == 0
        u = float(emitter.ops.nodeDisp(2, 2))
        # Reaction at the grounded node is the device force by
        # equilibrium; it does not depend on a bridge-allocated ele tag
        # (ZeroLength lands on tag 2, not the stub's line eid 1).
        emitter.ops.reactions()
        f = -float(emitter.ops.nodeReaction(1, 2))
        us.append(u)
        forces.append(f)

    worst = max(abs(fm - _backbone(u)) for u, fm in zip(us, forces))
    assert worst <= 1e-9

    # Sample 300 is nominally 15 mm, but 320 accumulations of -5e-5
    # drift u to -0.014999999999999937, so |F| is 1.8e-12 kN off the
    # breakpoint value — gate with a tolerance, not exact equality.
    assert abs(forces[299]) == pytest.approx(_F_AT_15_MM, abs=1e-9)

    tail_u = us[300:]
    tail_f = forces[300:]
    assert len(tail_f) == 20
    mag_15 = abs(forces[299])
    for u, f in zip(tail_u, tail_f):
        assert f == pytest.approx(_backbone(u), abs=1e-9)
        assert abs(f) > mag_15
    assert all(
        abs(a) < abs(b) for a, b in zip(tail_f, tail_f[1:])
    ), "force magnitude must keep rising past 15 mm (no cap/plateau)"

    u1, f1 = _POINTS[0]
    assert forces[0] == pytest.approx((f1 / u1) * us[0], abs=1e-9)
