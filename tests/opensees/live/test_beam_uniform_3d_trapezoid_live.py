"""Live oracle: a 3-D trapezoidal ``-beamUniform`` keeps its tail (#2550).

``Plain.beam_uniform(wy=(a, b), wz=(a, b), wx=(a, b), span=(aL, bL))`` emits
``eleLoad -beamUniform wy wz wx aL bL wyb wzb wxb`` on a 3-D model. The fork
reads all eight values into a ``Beam3dPartialUniformLoad``; each element then
decides whether it honours the tail (``wyb wzb wxb``). On fork build 79e0623,
``elasticBeamColumn`` read only the start values, and ``ElasticTimoshenkoBeam``
and ``dispBeamColumn`` dropped the whole load with a "load type unknown"
warning while ``analyze`` still returned 0. #2470's review measured 32.98
against 48. At ``FORK_PIN`` every 3-D element the bridge emits reads the tail.

Oracle (closed form, statics only): a fixed-fixed beam along global X with
``vecxz = +Z`` has local axes equal to the global ones. Its support reactions
must balance the trapezoid, which has resultant ``0.5 (w_a + w_b) ell`` per
component, and a first moment about node i of ``a F + ell^2 (w_a + 2 w_b) / 6``.
Equilibrium holds whatever the element's stiffness, so the check is exact for
every element class, the force-based ones included.

Fork-only: stock openseespy has no 3-D partial/trapezoid ``-beamUniform``
(the live emitter gates it).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter import live
from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

_L = 4.0
_A, _E, _G, _J, _IY, _IZ = 0.01, 200e9, 80e9, 1e-4, 2e-4, 1e-4

# (wy, wz, wx) as (start, end) pairs, local = global here.
_W: dict[str, tuple[float, float]] = {
    "wy": (2.0, 10.0), "wz": (-4.0, -8.0), "wx": (1.0, 3.0),
}


def _beam_fem() -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2], coords=[(0.0, 0.0, 0.0), (_L, 0.0, 0.0)],
        node_pgs={"Ends": [1, 2]},
    )
    elements = _ElementsStub(
        elem_pgs={"Beams": _ElementGroupView(ids=(1,), connectivity=((1, 2),))},
    )
    return FEMStub(nodes=nodes, elements=elements)


def _elastic(ops: apeSees, transf: Any) -> None:
    ops.element.elasticBeamColumn(
        pg="Beams", transf=transf, A=_A, E=_E, Iz=_IZ, Iy=_IY, G=_G, J=_J,
    )


def _timoshenko(ops: apeSees, transf: Any) -> None:
    ops.element.ElasticTimoshenkoBeam(
        pg="Beams", transf=transf, E=_E, G=_G, A=_A, Iz=_IZ, Iy=_IY, J=_J,
        Avy=_A, Avz=_A,
    )


def _integration(ops: apeSees) -> Any:
    sec = ops.section.Elastic(E=_E, A=_A, Iz=_IZ, Iy=_IY, G=_G, J=_J)
    return ops.beamIntegration.Lobatto(section=sec, n_ip=5)


def _disp(ops: apeSees, transf: Any) -> None:
    ops.element.dispBeamColumn(
        pg="Beams", transf=transf, integration=_integration(ops))


def _force(ops: apeSees, transf: Any) -> None:
    ops.element.forceBeamColumn(
        pg="Beams", transf=transf, integration=_integration(ops))


def _ladruno_disp(ops: apeSees, transf: Any) -> None:
    ops.element.LadrunoDispBeamColumn(
        pg="Beams", transf=transf, integration=_integration(ops))


def _ladruno_force(ops: apeSees, transf: Any) -> None:
    ops.element.LadrunoForceBeamColumn(
        pg="Beams", transf=transf, integration=_integration(ops))


def _ladruno_imk(ops: apeSees, transf: Any) -> None:
    ops.element.LadrunoIMKBeam(
        pg="Beams", transf=transf, A=_A, E=_E, Iz=_IZ, G=_G, Jx=_J, Iy=_IY,
        ends="none",
    )


_ELEMENTS: dict[str, Callable[[apeSees, Any], None]] = {
    "elasticBeamColumn": _elastic,
    "ElasticTimoshenkoBeam": _timoshenko,
    "dispBeamColumn": _disp,
    "forceBeamColumn": _force,
    "LadrunoDispBeamColumn": _ladruno_disp,
    "LadrunoForceBeamColumn": _ladruno_force,
    "LadrunoIMKBeam": _ladruno_imk,
}

_SPANS = {"full": (0.0, 1.0), "partial": (0.25, 0.75)}


def _closed_form(a_l: float, b_l: float) -> tuple[np.ndarray, np.ndarray]:
    """Load resultant (Fx, Fy, Fz) and its moment about node i (Mx, My, Mz)."""
    a, ell = a_l * _L, (b_l - a_l) * _L

    def res(pair: tuple[float, float]) -> tuple[float, float]:
        wa, wb = pair
        f = 0.5 * (wa + wb) * ell
        return f, a * f + ell * ell * (wa + 2.0 * wb) / 6.0

    fx, _ = res(_W["wx"])
    fy, sy = res(_W["wy"])
    fz, sz = res(_W["wz"])
    # r x F with r = (s, 0, 0) at the load's first moment s: (0, -s Fz, s Fy).
    return np.array([fx, fy, fz]), np.array([0.0, -sz, sy])


def _reactions(kind: str, a_l: float, b_l: float) -> tuple[np.ndarray, np.ndarray]:
    ops = apeSees(cast(Any, _beam_fem()))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    _ELEMENTS[kind](ops, transf)
    ops.fix(pg="Ends", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.beam_uniform(pg="Beams", span=(a_l, b_l), **_W)
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = live.LiveOpsEmitter(wipe=True)
    try:
        ops.build().emit(emitter)
        assert emitter.analyze(steps=1) == 0
        o = emitter.ops
        o.reactions()
        r1 = np.asarray(o.nodeReaction(1), dtype=float)
        r2 = np.asarray(o.nodeReaction(2), dtype=float)
    finally:
        live.LiveOpsEmitter(wipe=True)
    force = r1[:3] + r2[:3]
    moment = r1[3:] + r2[3:] + np.cross([_L, 0.0, 0.0], r2[:3])
    return force, moment


@pytest.mark.parametrize("span", sorted(_SPANS))
@pytest.mark.parametrize("kind", sorted(_ELEMENTS))
def test_3d_trapezoid_reactions_balance_the_full_load(kind: str, span: str) -> None:
    a_l, b_l = _SPANS[span]
    f_load, m_load = _closed_form(a_l, b_l)
    force, moment = _reactions(kind, a_l, b_l)
    scale_f = float(np.max(np.abs(f_load)))
    scale_m = float(np.max(np.abs(m_load)))
    np.testing.assert_allclose(
        -force, f_load, rtol=0.0, atol=1e-9 * scale_f,
        err_msg=f"{kind} ({span}): reactions do not balance the trapezoid "
        "resultant; the element dropped the -beamUniform tail",
    )
    np.testing.assert_allclose(
        -moment, m_load, rtol=0.0, atol=1e-9 * scale_m,
        err_msg=f"{kind} ({span}): reaction moment about node i does not "
        "balance the trapezoid's first moment",
    )
