"""Live, fork-gated run of ``LadrunoLoadControl -tangentPredictor``.

Gated by the ``live`` marker AND a fork build that carries the integrator
(2026-08-04; ``-tangentPredictor`` 2026-09-04). A fork build that predates
it accepts ``integrator LadrunoLoadControl`` with a warning and keeps the
previous integrator, so neither ``has_fork`` nor an ``integrator`` call can
be the probe. The ``ladrunoLoadControl`` runtime command shipped in the
same fork commit and is the skip condition here, as it is in
``LiveOpsEmitter.integrator``.

The model is the ADR-80 idiom at its smallest: one elastic column, a unit
``sp`` on the tip under a ``Linear`` series, ``constraints
Transformation``, and the load factor stepping the prescribed motion.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from tests.opensees.fixtures.fem_stub import make_two_node_beam


def _live_emitter_with_llc() -> LiveOpsEmitter:
    try:
        emitter = LiveOpsEmitter(wipe=True)
    except Exception as exc:  # no OpenSees backend importable
        pytest.skip(f"no OpenSees backend: {exc}")
    if not hasattr(emitter.ops, "ladrunoLoadControl"):
        pytest.skip(
            "OpenSees build lacks LadrunoLoadControl (needs a Ladruno fork "
            "build from 2026-08-04 on; -tangentPredictor from 2026-09-04)"
        )
    return emitter


@pytest.mark.live
def test_sp_protocol_under_tangent_predictor() -> None:
    emitter = _live_emitter_with_llc()
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(node=2, dof=1, value=1.0e-3)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LadrunoLoadControl(dlam=0.1)
    ops.analysis.Static()

    ops.build().emit(emitter)   # the gate confirms -tangentPredictor armed
    assert float(emitter.ops.ladrunoLoadControl("tangentPredictor")) == 1.0
    assert emitter.analyze(steps=10) == 0
    assert emitter.ops.nodeDisp(2, 1) == pytest.approx(1.0e-3, rel=1e-9)
