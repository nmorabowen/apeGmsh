"""Fork-only — a user ``equationConstraint`` row is enforced in-process.

Two parallel bars, each fixed at one end. Bar 2 is loaded at its free end;
``u_x(2) = u_x(4)`` is declared with ``ops.equation_constraint``. Enforced,
the bars share the load: ``u = P / (2 E A / L)``. Dropped (what
``Transformation`` would do), bar 2 alone carries it: twice that.

No handler is declared, so the bridge auto-emits ``Lagrange`` — the check
that the auto-emit took the EQ-capable handler, not ``Transformation``.

Fork-only because it runs in the test process: stock openseespy >= 3.8.0
enforces the row too, but its ``wipe()`` keeps EQ rows, so a stock process
that ran one refuses every later live model.
"""
from __future__ import annotations

import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

pytestmark = [pytest.mark.ladruno_fork, pytest.mark.live]

E, A, L, P = 200e9, 1e-3, 1.0, 1.0e4


def test_equation_constraint_ties_the_two_bars() -> None:
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[(0.0, 0.0, 0.0), (L, 0.0, 0.0),
                    (0.0, 1.0, 0.0), (L, 1.0, 0.0)],
            node_pgs={"Base": [1, 3], "Tips": [2, 4]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Bars": _ElementGroupView(
                ids=(1, 2), connectivity=((1, 2), (3, 4)),
            ),
        }),
    )
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    steel = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Bars", A=A, material=steel)
    ops.fix(pg="Base", dofs=(1, 1, 1))
    ops.fix(pg="Tips", dofs=(0, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=4, forces=(P, 0.0, 0.0))
    ops.equation_constraint(constrained=(4, 1), retained=[(2, 1, -1.0)])
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormUnbalance(tol=1e-6, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops.build().emit(emitter)
    assert any("Auto-emitting 'Lagrange'" in str(w.message) for w in caught)
    assert emitter.analyze(steps=1) == 0

    u2 = float(emitter.ops.nodeDisp(2, 1))
    u4 = float(emitter.ops.nodeDisp(4, 1))
    expected = P / (2.0 * E * A / L)
    assert u4 == pytest.approx(expected, rel=1e-9)
    assert u2 == pytest.approx(u4, rel=1e-9)
