"""Live: ``ASDConcrete3D(tangent="numerical")`` reaches the parser.

An unknown trailing flag would be the silent failure, so assert the effect.
One ``stdBrick`` pulled past its tensile peak by a prescribed displacement,
run twice: the stress path is identical (the tangent only steers Newton),
but the material tangent the element reports differs — the default secant
stays positive on the softening branch, the numerical ``-tangent`` goes
negative there.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)

CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _pull(tangent: str) -> tuple[float, float]:
    fem = FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 9)),
            coords=[tuple(float(c) for c in p) for p in CUBE],
            node_pgs={"Bottom": [1, 2, 3, 4], "Top": [5, 6, 7, 8],
                      "X0": [1, 4, 5, 8], "Y0": [1, 2, 5, 6]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Body": _ElementGroupView(ids=(1,), connectivity=(tuple(range(1, 9)),)),
        }),
    )
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.ASDConcrete3D(
        E=30e9, v=0.2, fc=30e6, ft=3e6, Gf=150.0, lch_ref=1.0, tangent=tangent,
    )
    ops.element.stdBrick(pg="Body", material=conc)
    ops.fix(pg="Bottom", dofs=(0, 0, 1))
    ops.fix(pg="X0", dofs=(1, 0, 0))
    ops.fix(pg="Y0", dofs=(0, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="Top", dof=3, value=3e-4)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / 30)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for _ in range(30):
        assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    (tag,) = live.getEleTags()
    c33 = float(live.eleResponse(tag, "material", "1", "tangent")[14])
    s33 = float(live.eleResponse(tag, "material", "1", "stress")[2])
    return c33, s33


@pytest.mark.live
def test_numerical_tangent_changes_the_reported_tangent_not_the_path() -> None:
    c_sec, s_sec = _pull("secant")
    c_num, s_num = _pull("numerical")
    assert s_num == pytest.approx(s_sec, rel=1e-6)   # same converged path
    assert s_sec > 0.0                                # still carrying tension
    assert c_sec > 0.0                                # secant: positive
    assert c_num < 0.0                                # numerical: softening
