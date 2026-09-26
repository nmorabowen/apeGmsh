"""Fork-only — ``LadrunoConcrete3D`` tension-law / Gc-reading flags reach the parser.

``-tensionLaw``, ``-epsFc`` and ``-gcLegacy`` shipped in fork commit
``1334d1e24`` (``LADRUNO_CONCRETE3D_TENSION_LAW_MIN_BUILD``). The parser
refuses an unknown option loudly, so a material that builds proves the
token parsed; the tension-law case also asserts the effect — one
``LadrunoBrick`` pulled well past its tensile peak carries a different
residual stress under the bilinear and the exponential law.

Skipped on a fork build that predates the flags (the 2026-06-25 installer
build does); ``-flowPotential`` (``916576661``) is newer still and is only
checked at the emit level (``tests/opensees/unit/primitives``).
"""
from __future__ import annotations

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

E, FT = 30e9, 2.0e6
U_MAX = 6.0 * FT / E
N_STEPS = 30
CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _require_flags() -> None:
    from apeGmsh.opensees.emitter.live import _get_ops

    live = _get_ops()
    live.wipe()
    live.model("basic", "-ndm", 3, "-ndf", 3)
    try:
        live.nDMaterial("LadrunoConcrete3D", 1, E, 0.2, 30e6, FT, 200.0,
                        30000.0, "-tensionLaw", "exp")
    except Exception:
        pytest.skip("fork build predates -tensionLaw (commit 1334d1e24)")
    finally:
        live.wipe()


def _pull(**flags: object) -> float:
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
    conc = ops.nDMaterial.LadrunoConcrete3D(
        E=E, nu=0.2, fc=30e6, ft=FT, Gf=200.0, Gc=30000.0, **flags,
    )
    ops.element.LadrunoBrick(pg="Body", material=conc)
    ops.fix(pg="Bottom", dofs=(0, 0, 1))
    ops.fix(pg="X0", dofs=(1, 0, 0))
    ops.fix(pg="Y0", dofs=(0, 1, 0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.sp(pg="Top", dof=3, value=U_MAX)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0 / N_STEPS)
    ops.analysis.Static()
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for _ in range(N_STEPS):
        assert emitter.analyze(steps=1) == 0
    live = emitter.ops
    (tag,) = live.getEleTags()
    return float(live.eleResponse(tag, "material", "1", "stress")[2])


def test_tension_law_changes_the_post_peak_stress() -> None:
    _require_flags()
    bilinear = _pull(tension_law="bilinear")
    exp = _pull(tension_law="exp")
    assert 0.0 < bilinear < FT and 0.0 < exp < FT      # both softened
    assert exp != pytest.approx(bilinear, rel=1e-3)


@pytest.mark.parametrize("flags", [{"eps_fc": 3e-3}, {"gc_legacy": True}])
def test_gc_reading_flags_are_accepted(flags: dict) -> None:
    # The parser refuses an unknown option, so building and running at all
    # is the check; a tension pull does not engage compression softening.
    _require_flags()
    assert 0.0 < _pull(**flags) < FT
