"""Live oracle: a 3-D trapezoidal ``-beamUniform`` keeps its tail (#2550).

``Plain.beam_uniform(wy=(a, b), wz=(a, b), wx=(a, b), span=(aL, bL))`` emits
``eleLoad -beamUniform wy wz wx aL bL wyb wzb wxb`` on a 3-D model. The fork
reads all eight values into a ``Beam3dPartialUniformLoad``; each element then
decides whether it honours the tail (``wyb wzb wxb``). On fork build 79e0623,
``elasticBeamColumn`` read only the start values, and ``ElasticTimoshenkoBeam``
and ``dispBeamColumn`` dropped the whole load with a "load type unknown"
warning while ``analyze`` still returned 0. #2470's review measured 32.98
against 48. At ``FORK_PIN`` every 3-D element the bridge emits reads the tail.

Oracle: the reactions at **each** node of a fixed-fixed beam, forces and
moments, against the Euler-Bernoulli fixed-end forces of the trapezoid. Sums
alone would not do: with both nodes fixed, the resultant and its moment come
from the element's statics (``p0``) only, so an element that dropped the tail
from its fixed-end moments (``q0``) or from its section forces would still
balance (Opus review of #2561). The oracle lives in
``tests/opensees/unit/test_beam_uniform_3d_trapezoid.py`` (pinned there
against textbook fixed-end forces) and is computed by quadrature of the
influence functions, independently of OpenSees:

* elastic, Timoshenko (with a shear area large enough that ``phi`` ~ 1e-11),
  displacement-based and IMK elements: the exact fixed-end forces;
* force-based elements: the same beam with the compatibility integrals taken
  at the element's own Lobatto points, which is what a force-based element
  solves. Over the full span the integrand is a polynomial Lobatto-5
  integrates exactly, so this equals the exact answer there.

Two geometries: a beam along global X with ``vecxz = +Z`` (local = global),
and a skew beam with ``vecxz = (0, -1, 0)``, whose local triad the oracle
builds on its own from the documented ``geomTransf`` rule.

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
from tests.opensees.unit.test_beam_uniform_3d_trapezoid import (
    LOBATTO_5,
    fixed_end_reactions_global,
    local_triad,
)

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

_A, _E, _G, _J, _IY, _IZ = 0.01, 200e9, 80e9, 1e-4, 2e-4, 1e-4
# ElasticTimoshenkoBeam: phi = 12 E I / (G Av L^2) ~ 1e-11 with this area.
_AV_RIGID = 1e9 * _A

# (wy, wz, wx) as (start, end) pairs in local axes.
_W: dict[str, tuple[float, float]] = {
    "wy": (2.0, 10.0), "wz": (-4.0, -8.0), "wx": (1.0, 3.0),
}

# name -> (node j coordinates, vecxz); node i is at the origin.
_GEOMETRY: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "x_axis": ((4.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "skew": ((2.0, 1.0, 2.0), (0.0, -1.0, 0.0)),
}

_SPANS = {"full": (0.0, 1.0), "partial": (0.25, 0.75)}


def _beam_fem(xj: tuple[float, float, float]) -> FEMStub:
    nodes = _NodesStub(
        ids=[1, 2], coords=[(0.0, 0.0, 0.0), xj], node_pgs={"Ends": [1, 2]},
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
        Avy=_AV_RIGID, Avz=_AV_RIGID,
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


# kind -> (builder, quadrature the oracle uses: None = exact integrals).
_ELEMENTS: dict[str, tuple[Callable[[apeSees, Any], None], Any]] = {
    "elasticBeamColumn": (_elastic, None),
    "ElasticTimoshenkoBeam": (_timoshenko, None),
    "dispBeamColumn": (_disp, None),
    "forceBeamColumn": (_force, LOBATTO_5),
    "LadrunoDispBeamColumn": (_ladruno_disp, None),
    "LadrunoForceBeamColumn": (_ladruno_force, LOBATTO_5),
    "LadrunoIMKBeam": (_ladruno_imk, None),
}


def _reactions(kind: str, geometry: str, a_l: float, b_l: float) -> np.ndarray:
    """Live reactions, rows ``(node i, node j)``, columns ``Fx Fy Fz Mx My Mz``."""
    xj, vecxz = _GEOMETRY[geometry]
    ops = apeSees(cast(Any, _beam_fem(xj)))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=vecxz)
    _ELEMENTS[kind][0](ops, transf)
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
        out = np.array([o.nodeReaction(1), o.nodeReaction(2)], dtype=float)
    finally:
        live.LiveOpsEmitter(wipe=True)
    return out


@pytest.mark.parametrize("geometry", sorted(_GEOMETRY))
@pytest.mark.parametrize("span", sorted(_SPANS))
@pytest.mark.parametrize("kind", sorted(_ELEMENTS))
def test_3d_trapezoid_reactions_match_fixed_end_forces(
    kind: str, span: str, geometry: str,
) -> None:
    a_l, b_l = _SPANS[span]
    xj, vecxz = _GEOMETRY[geometry]
    length = float(np.linalg.norm(xj))
    expected = fixed_end_reactions_global(
        length, local_triad(xj, vecxz), _W, a_l, b_l,
        quadrature=_ELEMENTS[kind][1],
    )
    got = _reactions(kind, geometry, a_l, b_l)
    for cols, what in ((slice(0, 3), "forces"), (slice(3, 6), "moments")):
        scale = float(np.max(np.abs(expected[:, cols])))
        np.testing.assert_allclose(
            got[:, cols], expected[:, cols], rtol=0.0, atol=1e-8 * scale,
            err_msg=f"{kind} ({span}, {geometry}): node i/j reaction {what} "
            "differ from the trapezoid's fixed-end forces; the element "
            "dropped or misread the -beamUniform tail",
        )
