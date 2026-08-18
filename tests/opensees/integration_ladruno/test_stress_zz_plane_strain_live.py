"""The recorded out-of-plane sigma_zz survives yielding — the estimate does not.

For a 2-D plane-strain element apeGmsh has always reconstructed
sigma_zz = nu*(sigma_xx + sigma_yy) at read time.  That identity is exact
for a linear-elastic material and simply false once the Gauss point yields:
the true out-of-plane stress is the material's internal sigma_33, which the
3-component ``stresses`` response does not carry.

The fork's ``stressesPlaneStrain`` response carries it.  This is the live
proof, on a partially-yielded LadrunoLST / LadrunoJ2 cantilever:

* a plane-strain deck that requests ``stress_zz`` emits the promoted token;
* the run's ``.ladruno`` exposes ``stress_zz``, finite everywhere;
* ``von_mises_stress`` agrees with a hand-computed von Mises over the four
  recorded components to machine precision — including at the Gauss points
  where the recorded sigma_zz departs from the nu-estimate by tens of
  percent, i.e. exactly where the old path was wrong;
* nothing raises ``OutOfPlaneRecoveryWarning``, because nothing was
  recovered.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._element_capabilities import (
    STRESS_PLANE_STRAIN_RESPONSE,
)
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.results import Results
from apeGmsh.results._plane_recovery import OutOfPlaneRecoveryWarning

pytestmark = pytest.mark.ladruno_fork

# LadrunoJ2 elastic constants -> nu = (3K - 2G) / (2(3K + G)).
K, G = 1.333e8, 8.0e7
SIG0, HISO = 1.2e5, 2.0e6
NU = (3.0 * K - 2.0 * G) / (2.0 * (3.0 * K + G))

LX, LY = 4.0, 1.0
TIP_LOAD = -3.0e2
N_STEPS = 40


def _mesh():
    with apeGmsh(model_name="szz_lst", verbose=False) as g:
        rect = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, LX, LY)
        g.model.sync()
        g.physical.add(2, [rect], name="Body")
        g.mesh.sizing.set_global_size(0.4)
        g.mesh.generation.generate(2)
        g.mesh.generation.set_order(2, bubble=False)
        return g.mesh.queries.get_fem_data(dim=2)


def _nodes_at_x(fem, x: float) -> list[int]:
    ids = np.asarray(fem.nodes.ids)
    xyz = np.asarray(fem.nodes.coords)
    return [
        int(n) for n, p in zip(ids, xyz) if abs(float(p[0]) - x) < 1e-9
    ]


def _bridge(fem, ladruno_path: str, file_root: str):
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.LadrunoJ2(K=K, G=G, sig0=SIG0, Hiso=HISO)
    ops.element.LadrunoLST(pg="Body", thickness=0.1, material=mat)

    ops.fix(nodes=_nodes_at_x(fem, 0.0), dofs=(1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        for nid in _nodes_at_x(fem, LX):
            p.load(node=nid, forces=(0.0, TIP_LOAD))

    # The declaration under test: a canonical ``stress_zz`` request.
    ops.recorder.declare(
        gauss=("stress_xx", "stress_yy", "stress_xy", "stress_zz"),
        pg="Body", file_root=file_root,
    )
    # The read channel: a self-describing .ladruno carrying the same
    # promoted response, so the assertions below read real recorded data
    # rather than the .out transcoder's catalog view.
    ops.recorder.Ladruno(
        file=ladruno_path,
        elem_responses=("stress", STRESS_PLANE_STRAIN_RESPONSE),
    )

    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=30)
    ops.algorithm.KrylovNewton()
    ops.integrator.LoadControl(dlam=1.0 / N_STEPS)
    ops.analysis.Static()
    return ops


def _von_mises(sxx, syy, sxy, szz):
    """von Mises over a plane-strain stress state (sigma_xz = sigma_yz = 0)."""
    return np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * sxy ** 2
    )


def test_plane_strain_stress_zz_is_recorded_not_estimated(tmp_path) -> None:
    fem = _mesh()
    group = list(fem.elements)[0]
    assert group.element_type.npe == 6, "set_order(2) did not give tri6"

    path = str(tmp_path / "lst.ladruno")
    ops = _bridge(fem, path, str(tmp_path))

    # (1) The deck a plane-strain stress_zz request produces.
    tcl = TclEmitter()
    ops.build().emit(tcl)
    deck = "\n".join(tcl.lines())
    promoted = [
        ln for ln in deck.splitlines()
        if ln.startswith("recorder Element")
        and ln.endswith(f" {STRESS_PLANE_STRAIN_RESPONSE}")
    ]
    assert len(promoted) == 1, (
        f"expected one promoted recorder line, got:\n{deck}"
    )

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=N_STEPS) == 0
    emitter.ops.remove("recorders")

    # (2) stress_zz reaches the result surface.
    with Results.from_ladruno(path, fem=fem) as r:
        assert "stress_zz" in r.elements.gauss.available_components()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sxx = r.elements.gauss.get(component="stress_xx").values[-1]
            syy = r.elements.gauss.get(component="stress_yy").values[-1]
            sxy = r.elements.gauss.get(component="stress_xy").values[-1]
            szz = r.elements.gauss.get(component="stress_zz").values[-1]
            vm = r.elements.gauss.get(component="von_mises_stress").values[-1]

        # (4) Nothing was recovered, so nothing warns about recovery.
        assert not [
            w for w in caught
            if issubclass(w.category, OutOfPlaneRecoveryWarning)
        ]

    assert np.isfinite(szz).all(), "recorded sigma_zz carries NaN"

    # Per-point departure from the elastic identity nu*(sxx+syy) that the
    # fallback assumes: ~0 while the point is elastic, large once it yields.
    estimate = NU * (sxx + syy)
    departure = np.abs(szz - estimate) / np.abs(szz)
    elastic = departure < 1e-10
    plastic = departure > 1e-3
    assert elastic.any(), "no elastic Gauss point to anchor the identity"
    assert plastic.any(), (
        "no Gauss point departs from the nu-estimate: the model stayed "
        "elastic, so this run cannot discriminate recorded from recovered"
    )
    assert departure.max() > 0.1, (
        f"largest departure from the nu-estimate is only "
        f"{departure.max():.3%}; expected tens of percent at plastic points"
    )
    # The elastic points still satisfy the identity to machine precision,
    # which is what says the recorded column really is sigma_33 and not
    # some other quantity that merely happens to differ.
    np.testing.assert_allclose(
        szz[elastic], estimate[elastic], rtol=1e-12, atol=0.0,
    )

    # (3) Machine-precision agreement, plastic points included.
    np.testing.assert_allclose(
        vm, _von_mises(sxx, syy, sxy, szz), rtol=1e-12, atol=1e-9,
    )
    np.testing.assert_allclose(
        vm[plastic], _von_mises(sxx, syy, sxy, szz)[plastic],
        rtol=1e-12, atol=1e-9,
    )
    # The estimate would NOT have agreed — that is the whole point.
    assert not np.allclose(
        vm[plastic], _von_mises(sxx, syy, sxy, estimate)[plastic],
        rtol=1e-6,
    )
