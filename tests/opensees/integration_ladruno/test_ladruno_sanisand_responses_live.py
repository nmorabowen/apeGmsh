"""Fork-only — ``LadrunoSANISAND``'s IMPL-EX/state responses reach the
Gauss level through the ``.ladruno`` recorder and reader (TIMs A12, fork
PR #805/#820).

Mirrors :mod:`test_ladruno_gauss_generic_columns`'s recorder-through-reader
shape (build a deck, record, ``Results.from_ladruno``, check
``available_components()``), but for the seven SANISAND-only buckets
registered in ``_MATERIAL_BUCKET_TOKENS`` / ``_MATERIAL_ONLY_ELEM_TOKENS``
instead of the plane-element generic-column case.

One hex, ``LadrunoBrick`` + ``LadrunoSANISAND``, loaded ISOTROPICALLY only
(no deviatoric leg, no stage flip) — the confine-first elastic leg of
``test_ladruno_sanisand_live.py``'s ``_triaxial_run``, without the
post-flip continuation. With the back-stress ratio ``alpha`` still at its
construction-time zero, the yield function collapses to
``f = sqrt(2/3)*||dev(alpha - r)|| - m`` evaluated at ``r = 0``, i.e.
``-sqrt(2/3)*m*p'`` with ``p'`` the (compression-positive) mean effective
stress — negative under any confining pressure, since ``m > 0``. Checked
against ``mean_stress`` (the reader's tensor-derived, TENSION-positive
scalar): ``p' = -mean_stress``.

Skips: ``material.psi`` et al. do not raise on a fork build that predates
them — the material / recorder answers with an EMPTY response vector, no
exception (the same silent-empty shape as ADR 0105's bare-token pitfall,
just on the other side of the ``material.`` prefix) — so the in-test probe
checks for a NON-EMPTY ``eleResponse(..., "material", 1, "psi")`` rather
than catching an exception. As of this slice the installed build
(``e95a1c74f7e15d7de8655eeeb004d7f34d81d512``) predates the responses —
``material.psi`` returns ``[]`` and the ``LadrunoRecorder`` itself warns
"unknown or unsupported token" — so the probe skips and this test could
only be written and gated, not run to green, in this session.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

pytestmark = pytest.mark.ladruno_fork

#: Gorini's calibrated set (ADR 86 §6) — copied verbatim from
#: ``test_ladruno_sanisand_live.py``'s ``_GORINI``, not invented.
_GORINI = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}

_HAS_RESPONSES: bool | None = None


def _single_hex_fem():
    """One 1 m³ 8-node hex, PG 'soil' — the material-driver mesh."""
    with apeGmsh(model_name="ls_resp_hex", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="soil")
        return g.mesh.queries.get_fem_data(dim=3)


def _plane(fem, axis: int, value: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[axis]) - value) < 1e-9
    ]


def _node_coords(fem) -> dict[int, tuple[float, float, float]]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return {
        int(n): (float(p[0]), float(p[1]), float(p[2]))
        for n, p in zip(ids, xyz)
    }


def _require_sanisand_responses() -> None:
    """Probe-skip on fork builds that predate the IMPL-EX/state responses.

    Builds a throwaway one-hex deck and reads ``material.psi`` straight
    off the live domain: a build that has the responses answers with a
    one-element vector, a build that does not answers with an EMPTY one
    (no exception either way) — same detection shape as
    :func:`test_ladruno_sanisand_live._require_ladruno_sanisand`, adapted
    for a silent-empty rather than a raise.
    """
    global _HAS_RESPONSES
    if _HAS_RESPONSES is None:
        fem = _single_hex_fem()
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=3)
        sand = ops.nDMaterial.LadrunoSANISAND(**_GORINI)
        ops.element.LadrunoBrick(pg="soil", material=sand)
        ops.fix(nodes=_plane(fem, 2, 0.0), dofs=(1, 1, 1))
        em = LiveOpsEmitter(wipe=True)
        ops.build().emit(em)
        tags = em.ops.getEleTags()
        etag = int(tags if isinstance(tags, int) else tags[0])
        try:
            resp = em.ops.eleResponse(etag, "material", 1, "psi")
        except Exception:
            resp = []
        _HAS_RESPONSES = bool(resp)
        em.ops.wipe()
    if not _HAS_RESPONSES:
        pytest.skip(
            "fork build predates LadrunoSANISAND's psi/yieldDistance/"
            "implexDetail responses (TIMs A12, fork PR #805/#820)"
        )


def test_sanisand_responses_reach_the_gauss_level_with_negative_yield_distance(
    tmp_path,
) -> None:
    _require_sanisand_responses()
    fem = _single_hex_fem()
    coords = _node_coords(fem)
    bottom = _plane(fem, 2, 0.0)
    origin = next(n for n, c in coords.items() if c == (0.0, 0.0, 0.0))
    on_x = next(n for n, c in coords.items() if c == (1.0, 0.0, 0.0))

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    sand = ops.nDMaterial.LadrunoSANISAND(**_GORINI)
    ops.element.LadrunoBrick(pg="soil", material=sand)
    ops.fix(nodes=[origin], dofs=(1, 1, 1))
    ops.fix(nodes=[on_x], dofs=(0, 1, 1))
    ops.fix(
        nodes=[n for n in bottom if n not in (origin, on_x)], dofs=(0, 0, 1),
    )

    # Isotropic confinement only — no deviatoric leg, no stage flip. Alpha
    # stays at its construction-time zero throughout.
    p_c = 100.0
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid, (x, y, z) in coords.items():
            fx = (p_c / 4.0) if x == 0.0 else (-p_c / 4.0)
            fy = (p_c / 4.0) if y == 0.0 else (-p_c / 4.0)
            fz = (-p_c / 4.0) if z == 1.0 else 0.0
            p.load(node=nid, forces=(fx, fy, fz))

    path = str(tmp_path / "sanisand_responses.ladruno")
    ops.recorder.Ladruno(
        file=path,
        elem_responses=("stress", "material.psi", "material.yieldDistance",
                         "material.implexDetail"),
    )
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormUnbalance(tol=1.0e-4 * p_c, max_iter=200)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    for _ in range(10):
        assert emitter.analyze(steps=1) == 0, "isotropic confinement diverged"
    emitter.ops.remove("recorders")   # flush the .ladruno

    r = Results.from_ladruno(path)
    available = r.elements.gauss.available_components()
    wanted = (
        "state_parameter", "yield_distance",
        "implex_detail_total", "implex_detail_dev", "implex_detail_vol",
        "implex_detail_clamp_fired", "implex_detail_clamp_count",
        "implex_detail_f",
    )
    missing = [c for c in wanted if c not in available]
    assert not missing, f"{missing} absent from {sorted(available)}"

    # alpha = 0 (never flipped, never loaded deviatorically): the yield
    # function collapses to -sqrt(2/3)*m*p', p' = -mean_stress (the
    # reader's mean_stress is tension-positive; SANISAND's p' is
    # compression-positive).
    mean_stress = r.elements.gauss.get(component="mean_stress").values[-1]
    yield_distance = r.elements.gauss.get(component="yield_distance").values[-1]
    p_prime = -mean_stress
    expected = -math.sqrt(2.0 / 3.0) * _GORINI["m"] * p_prime
    np.testing.assert_allclose(yield_distance, expected, rtol=1e-6)
    assert np.all(yield_distance < 0.0), (
        "yield_distance should stay negative (inside the surface) through "
        "a purely isotropic, never-flipped confinement leg"
    )
