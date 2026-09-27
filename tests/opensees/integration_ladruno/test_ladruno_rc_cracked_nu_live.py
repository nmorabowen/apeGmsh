"""Fork-only — ``LadrunoRCConcrete`` / ``LadrunoRCFiniteStrain`` C2 flags reach the parser.

``-crackedNu``, ``-betaC`` and the ``vc`` tension-stiffening default change
(500 -> 200) shipped together in fork commit ``8cd8f43f5``
(``nmorabowen/OpenSees#873``, branch ``wp/phase-c-elements``). Unlike the
LadrunoConcrete3D fork flags (``test_ladruno_concrete3d_flags_live.py``),
this parser's option loop has no catch-all ``else`` on an unrecognized
flag token — an unknown ``-crackedNu`` is silently skipped (and its value
token misparsed downstream) rather than raising, so a bare "did the
constructor throw" probe does NOT detect a pre-C2 build (confirmed live:
the installed 2026-06 build accepts the call and returns identical
results for every ``cracked_nu`` value). The material response
``nuCracked`` (added in the same C2 commit) is the reliable probe: it
returns an empty vector on a build that doesn't know it.

Skipped unless the resolved build's ``nuCracked`` response comes back
non-empty. PV20 finding (documented on ``_LadrunoRC.cracked_nu``, not
asserted numerically here -- a plain uniaxial free-lateral-contraction
pull did not show a measurable difference for this flag, likely because
this smeared-crack law decouples the crack-normal softening from the
lateral elastic response for this loading path): keeping the elastic
Poisson ratio after cracking overstates shear strength by 8-10 % on the
PV20 panel; ``cracked_nu=0`` reproduces the MCFT hand solution. Run once
against the newer build at
``C:\\Users\\nmora\\Documents\\Github\\OpenSees\\dist\\bin`` via
``benchmarks/00_material_point/_bootstrap.py`` (``LCV_OPENSEES_BIN``,
``PMI_RANK=0``) in the ``ladruno-concrete-validation`` repo to confirm it
passes there.
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

E, FC = 30e9, 30e6
FT = 0.1 * FC          # matches _laws.default_ft(fc), from_fc's own default
EPS_CR = FT / E
U_MAX = 6.0 * EPS_CR
N_STEPS = 30
CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _cube_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=list(range(1, 9)),
            coords=[tuple(float(c) for c in p) for p in CUBE],
            node_pgs={"Bottom": [1, 2, 3, 4], "Top": [5, 6, 7, 8],
                      "X0": [1, 4, 5, 8], "Y0": [1, 2, 5, 6],
                      "Body": list(range(1, 9))},
        ),
        elements=_ElementsStub(elem_pgs={
            "Body": _ElementGroupView(ids=(1,), connectivity=(tuple(range(1, 9)),)),
        }),
    )


def _require_flags() -> None:
    # A material that builds is NOT proof the flag parsed (see module
    # docstring) -- this parser silently skips an unrecognized flag token
    # instead of erroring. Probe the "nuCracked" response instead (shipped
    # in the same C2 commit): empty on a build that doesn't know it.
    ops = apeSees(cast("object", _cube_fem()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.LadrunoRCConcrete(
        E=E, nu=0.2, fc=FC, ft=FT, cracked_nu=0.0)
    ops.element.LadrunoBrick(pg="Body", material=conc)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    live = emitter.ops
    (tag,) = live.getEleTags()
    resp = live.eleResponse(tag, "material", "1", "nuCracked")
    live.wipe()
    if not resp:
        pytest.skip(
            "fork build predates -crackedNu / the nuCracked response "
            "(commit 8cd8f43f5)"
        )


def _pull(**flags: object) -> float:
    """Build an 8-node cube of ``LadrunoBrick`` + ``LadrunoRCConcrete``,
    pull it in z past the cracking strain, and return the free-face
    lateral displacement (node 2, x=1 face; x=0 face is fixed).
    """
    ops = apeSees(cast("object", _cube_fem()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=3)
    conc = ops.nDMaterial.LadrunoRCConcrete(
        E=E, nu=0.2, fc=FC, ft=FT, **flags,
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
    return float(live.nodeDisp(2, 1))


@pytest.mark.parametrize(
    "flags",
    [
        {"cracked_nu": 0.0},
        {"beta_c": 189.0},
        {"tens_stiff": "vc", "tens_stiff_c": 500.0},
    ],
)
def test_c2_flags_are_accepted_and_run(flags: dict) -> None:
    # The parser silently skips an unrecognized flag rather than erroring
    # (see module docstring), so "does it run at all" is not by itself
    # proof of acceptance for any ONE of these three -- the ``nuCracked``
    # response probe in ``_require_flags`` is what actually gates the
    # module on the C2-or-newer build all three shipped with together.
    # This test then confirms the full mesh -> emit -> analyze loop
    # completes (converges) with each flag wired in, mirroring
    # ``test_gc_reading_flags_are_accepted`` in
    # ``test_ladruno_concrete3d_flags_live.py``.
    _require_flags()
    disp = _pull(**flags)
    assert disp < 0.0   # free face still contracts under the axial pull
