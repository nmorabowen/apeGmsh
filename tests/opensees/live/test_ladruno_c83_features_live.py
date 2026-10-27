"""Live oracles for the fork features merged into ``ladruno`` at ``c83ee812e``.

* ASD shells ``-pDelta`` (fork WP-205): a cantilever strip under a tip axial
  load ``P = 0.5 Pcr``, gravity then ``loadConst`` then ``eigen``, lengthens
  its first period by ``T/T0 = 1.408`` (the fork's measured value; the
  ``1/sqrt(1 - P/Pcr)`` estimate is 1.414). Without the flag the period does
  not move.
* ``ElasticTimoshenkoBeam`` (3-D) honours a ``PDelta`` transformation (fork
  WP-206): its tip lateral stiffness under axial compression matches
  ``elasticBeamColumn`` on the same transformation and mesh, and the exact
  beam-column value; ``basicForce`` answers ``(N, Mz1, Mz2, My1, My2, T)``.
* ``response_spectrum_analysis(scale=)`` multiplies ``Sa`` (the bridge scales
  the ``-Sa`` ordinates; WP-202's ``-scale`` is not emitted): ``scale=2``
  doubles the combined displacement.
* ``modalProperties`` after ``eigen -feast`` (fork WP-203), with and without
  ``-traceStop``: the band's participation factors equal the ARPACK ones.
  FEAST needs the MKL Extended Eigensolver, which only the Windows/oneAPI
  fork build links (fork ADR 43 P1). On a build that prints that refusal
  the test skips as ``ladruno_mkl`` (#2555); any other FEAST failure fails.

Every test probes the bound build for the c83 behaviour and skips on an
older fork build: the ASD shells refuse ``-pDelta -corotational`` only from
c83 on.
"""
from __future__ import annotations

import math
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
    make_two_node_beam,
)

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]


def _has_c83() -> bool:
    """The bound build refuses ``ASDShellQ4 -pDelta -corotational`` (c83+)."""
    ops = live.get_ops()
    live.LiveOpsEmitter(wipe=True)
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for i, (x, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1)), start=1):
        ops.node(i, float(x), float(y), 0.0)
    ops.section("ElasticMembranePlateSection", 1, 1.0e6, 0.2, 0.1, 0.0)
    try:
        ops.element("ASDShellQ4", 1, 1, 2, 3, 4, 1, "-pDelta", "-corotational")
        tags = ops.getEleTags()
        refused = 1 not in ([tags] if isinstance(tags, int) else (tags or []))
    except Exception:
        refused = True
    finally:
        live.LiveOpsEmitter(wipe=True)
    return refused


@pytest.fixture(autouse=True)
def _require_c83() -> None:
    if not _has_c83():
        pytest.skip("fork build predates c83ee812e (no ASD shell -pDelta)")


# ---------------------------------------------------------------------------
# ASDShellQ4 -pDelta: T/T0 of a compressed cantilever strip
# ---------------------------------------------------------------------------

_E, _NU, _T, _B, _H, _M = 25e6, 0.2, 0.3, 1.0, 30.0, 100.0
_NX, _NZ = 2, 20
_PCR = math.pi ** 2 * _E * (_B * _T ** 3 / 12.0) / (4.0 * _H ** 2)


def _strip_fem() -> FEMStub:
    def tag(i: int, k: int) -> int:
        return 1 + i + k * (_NX + 1)

    ids, coords = [], []
    for k in range(_NZ + 1):
        for i in range(_NX + 1):
            ids.append(tag(i, k))
            coords.append((i * _B / _NX, 0.0, k * _H / _NZ))
    quads = tuple(
        (tag(i, k), tag(i + 1, k), tag(i + 1, k + 1), tag(i, k + 1))
        for k in range(_NZ) for i in range(_NX)
    )
    nodes = _NodesStub(
        ids=ids, coords=coords,
        node_pgs={
            "Base": [tag(i, 0) for i in range(_NX + 1)],
            "Top": [tag(i, _NZ) for i in range(_NX + 1)],
        },
    )
    elements = _ElementsStub(elem_pgs={
        "Wall": _ElementGroupView(
            ids=tuple(range(1, len(quads) + 1)), connectivity=quads,
        ),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _strip_period(p_fac: float, *, p_delta: bool) -> float:
    ops = apeSees(cast(Any, _strip_fem()))
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=_E, nu=_NU, h=_T)
    ops.element.ASDShellQ4(pg="Wall", section=sec, p_delta=p_delta)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    n_top = _NX + 1
    ops.mass(pg="Top", values=(_M / n_top, _M / n_top, 0.0, 0.0, 0.0, 0.0))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(pg="Top", forces=(0.0, 0.0, -p_fac * _PCR / n_top, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-10, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    try:
        assert emitter.analyze(steps=1) == 0
        lv = emitter.ops
        lv.loadConst("-time", 0.0)
        lam = float(lv.eigen("-genBandArpack", 1)[0])
    finally:
        live.LiveOpsEmitter(wipe=True)
    return 2.0 * math.pi / math.sqrt(lam)


def test_asdshellq4_p_delta_lengthens_the_period_at_half_pcr() -> None:
    t0 = _strip_period(0.0, p_delta=True)
    t_pd = _strip_period(0.5, p_delta=True)
    t_lin = _strip_period(0.5, p_delta=False)
    assert t_pd / t0 == pytest.approx(1.408, rel=5e-3)
    # Without the flag the gravity state does not reach the tangent.
    assert t_lin / t0 == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ElasticTimoshenkoBeam 3-D on a PDelta transformation
# ---------------------------------------------------------------------------

_EB, _GB, _AB, _IB, _JB, _LB, _NE = 200e9, 80e9, 0.01, 1e-5, 2e-5, 4.0, 10


def _column_fem() -> FEMStub:
    ids = list(range(1, _NE + 2))
    coords = [(0.0, 0.0, k * _LB / _NE) for k in range(_NE + 1)]
    nodes = _NodesStub(
        ids=ids, coords=coords, node_pgs={"Base": [1], "Top": [_NE + 1]},
    )
    elements = _ElementsStub(elem_pgs={
        "Col": _ElementGroupView(
            ids=tuple(range(1, _NE + 1)),
            connectivity=tuple((k, k + 1) for k in range(1, _NE + 1)),
        ),
    })
    return FEMStub(nodes=nodes, elements=elements)


def _column_tip_drift(kind: str, transf: str, p: float, h: float) -> "tuple[float, list[float]]":
    ops = apeSees(cast(Any, _column_fem()))
    ops.model(ndm=3, ndf=6)
    tr = getattr(ops.geomTransf, transf)(vecxz=(1.0, 0.0, 0.0))
    if kind == "timoshenko":
        # A huge shear area: the Timoshenko beam reads as Euler-Bernoulli.
        ops.element.ElasticTimoshenkoBeam(
            pg="Col", transf=tr, E=_EB, G=_GB, A=_AB, Iz=_IB, Iy=_IB,
            J=_JB, Avy=1e6 * _AB, Avz=1e6 * _AB,
        )
    else:
        ops.element.elasticBeamColumn(
            pg="Col", transf=tr, A=_AB, E=_EB, Iz=_IB, Iy=_IB, G=_GB, J=_JB,
        )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Constant()) as pat:
        pat.load(pg="Top", forces=(h, 0.0, -p, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-12, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    emitter = live.LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    try:
        assert emitter.analyze(steps=1) == 0
        lv = emitter.ops
        drift = float(lv.nodeDisp(_NE + 1, 1))
        tags = lv.getEleTags()
        first = int(tags[0] if not isinstance(tags, int) else tags)
        basic = [float(v) for v in lv.eleResponse(first, "basicForce")]
    finally:
        live.LiveOpsEmitter(wipe=True)
    return drift, basic


def test_timoshenko_beam_honours_the_pdelta_transformation() -> None:
    pcr = math.pi ** 2 * _EB * _IB / (4.0 * _LB ** 2)
    p, h = 0.5 * pcr, 1.0e3
    d0, _ = _column_tip_drift("timoshenko", "Linear", p, h)
    d_pd, basic = _column_tip_drift("timoshenko", "PDelta", p, h)
    d_ebc, _ = _column_tip_drift("elastic", "PDelta", p, h)
    k = math.sqrt(p / (_EB * _IB))
    exact = h * (math.tan(k * _LB) - k * _LB) / (p * k)
    assert d0 == pytest.approx(h * _LB ** 3 / (3.0 * _EB * _IB), rel=1e-6)
    # PDelta softens: the drift grows by about 1/(1 - P/Pcr).
    assert d_pd / d0 > 1.8
    assert d_pd == pytest.approx(d_ebc, rel=1e-6)
    assert d_pd == pytest.approx(exact, rel=1e-2)
    # basicForce (N, Mz1, Mz2, My1, My2, T): the base element carries -P.
    assert len(basic) == 6
    assert basic[0] == pytest.approx(-p, rel=1e-9)


# ---------------------------------------------------------------------------
# response_spectrum_analysis(scale=) and FEAST + modalProperties
# ---------------------------------------------------------------------------


def _cantilever() -> apeSees:
    fem = make_two_node_beam()
    ops = apeSees(cast(Any, fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=2e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(pg="Top", values=(100.0, 100.0, 100.0, 1.0, 1.0, 1.0))
    return ops


def test_rsa_scale_two_doubles_the_combined_displacement() -> None:
    kw: dict[str, Any] = dict(
        periods=[0.01, 0.1, 1.0], accels=[3.0, 2.0, 1.0],
        combine="SRSS", num_modes=2, solver="-fullGenLapack",
    )
    u1 = _cantilever().response_spectrum_analysis(1, **kw).node_disp(2, 1)
    u2 = _cantilever().response_spectrum_analysis(1, scale=2.0, **kw).node_disp(2, 1)
    live.LiveOpsEmitter(wipe=True)
    assert u1 != 0.0
    assert u2 == pytest.approx(2.0 * u1, rel=1e-12)


#: The fork's refusal on a build without the MKL Extended Eigensolver (the
#: Linux release tarball): ``FeastEigenSolver::solve()``, fork ADR 43 P1.
FEAST_WITHOUT_MKL = "FEAST requires the MKL Extended Eigensolver"


@pytest.mark.ladruno_mkl
@pytest.mark.parametrize("trace_stop", [False, True])
def test_feast_then_modal_properties_matches_arpack(
    trace_stop: bool, capfd: pytest.CaptureFixture[str],
) -> None:
    ref = _cantilever().modal_properties(2, solver="-fullGenLapack")
    f = np.sqrt(ref.eigenvalues) / (2.0 * math.pi)
    capfd.readouterr()
    try:
        band = _cantilever().eigen_feast(
            0.5 * float(f[0]), 1.5 * float(f[1]), trace_stop=trace_stop,
        )
    except Exception:
        err = capfd.readouterr().err
        live.LiveOpsEmitter(wipe=True)
        if FEAST_WITHOUT_MKL not in err:
            raise
        pytest.skip(f"ladruno_mkl: FEAST unavailable ({FEAST_WITHOUT_MKL})")
    props = band.modal_properties()
    live.LiveOpsEmitter(wipe=True)
    assert len(band.eigenvalues) == 2
    np.testing.assert_allclose(
        np.sort(band.eigenvalues), np.sort(ref.eigenvalues), rtol=1e-8,
    )
    for key in ("partiMassRatiosMX", "partiMassRatiosMY"):
        np.testing.assert_allclose(
            sorted(props.properties[key]), sorted(ref.properties[key]),
            rtol=1e-6, atol=1e-9,
        )
    assert props.properties["totalMass"] == pytest.approx(
        ref.properties["totalMass"], rel=1e-12,
    )
