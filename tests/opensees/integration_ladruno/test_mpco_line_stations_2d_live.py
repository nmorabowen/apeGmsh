"""Fork lane: 2-D beam ``localForce`` line stations read back from MPCO (#2010).

The #1740 portal frame, solved through the bridge with sequential tags, so
FEM ids are not OpenSees tags, recording only ``localForce`` to MPCO. The
fork's MPCO recorder spells a 2-D bucket ``N_1,V_1,M_1,..`` (ElasticBeam2d,
ElasticTimoshenkoBeam2d, ForceBeamColumn2d) or ``N1,V1,M1,..``
(DispBeamColumn2d and the fork's disp beam). Before #2010 the MPCO reader
dropped ``V``/``M`` silently, so an elastic girder had ``axial_force``
stations only, and a disp beam had none.

The girder runs through every 2-D beam family that records ``localForce``
to MPCO; the columns are force beams. With no ``section.force`` recorded,
every element takes its two ``localForce`` stations. Each station is checked
at ``rtol=1e-10`` against the live domain: ``eleResponse(tag,
"localForce")``, end i as recorded and end j negated (the station
convention on main; #2009 revisits N and M). For the elastic girders the
same stations are also checked against a DomainCapture of the same steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

from tests.test_results_recorder_tag_parity import _portal_fem

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
ATOL = 1e-9          # moments near a zero crossing
COMPS = ("axial_force", "shear_y", "bending_moment_z")


def _sectioned(make: str) -> Callable[[apeSees, Any], Any]:
    def build(ops: apeSees, tr: Any) -> Any:
        sec = ops.section.Elastic(E=2.0e11, A=0.02, Iz=3.0e-4)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        return getattr(ops.element, make)(pg="Beam", transf=tr,
                                          integration=integ)
    return build


#: girder class -> (bridge builder, its DomainCapture stations are localForce)
FAMILIES: dict[str, tuple[Callable[[apeSees, Any], Any], bool]] = {
    "ElasticBeam2d": (
        lambda ops, tr: ops.element.elasticBeamColumn(
            pg="Beam", transf=tr, A=0.02, E=2.0e11, Iz=3.0e-4), True),
    "ElasticTimoshenkoBeam2d": (
        lambda ops, tr: ops.element.ElasticTimoshenkoBeam(
            pg="Beam", transf=tr, E=2.0e11, G=8.0e10, A=0.02, Iz=3.0e-4,
            Avy=0.015), True),
    "ForceBeamColumn2d": (_sectioned("forceBeamColumn"), False),
    "DispBeamColumn2d": (_sectioned("dispBeamColumn"), False),
    "LadrunoDispBeamColumn2d": (_sectioned("LadrunoDispBeamColumn"), False),
}


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def _solve(tmp: Path, family: str) -> dict[str, Any]:
    build_girder, captured = FAMILIES[family]
    fem = _portal_fem()
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Girder first: its sequential tags are taken before the columns'.
    build_girder(ops, tr)
    sec = ops.section.Elastic(E=2.0e11, A=0.03, Iz=5.0e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=4)
    ops.element.forceBeamColumn(pg="Cols", transf=tr, integration=integ)
    ops.fix(label="base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tl"), forces=(10e3, -30e3, 0.0))
        p.load(node=ops.nodes.one(label="tr"), forces=(0.0, -50e3, 4e3))
    mpco = str(tmp / "portal.mpco")
    ops.recorder.MPCO(file=mpco, nodal_responses=("displacement",),
                      elem_responses=("localForce",))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    fem_to_ops = _fem_to_ops(model_h5)
    cols = [int(e) for e in fem.elements.select(pg="Cols").ids]
    beam = [int(e) for e in fem.elements.select(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in cols + beam)   # not vacuous

    spec = DomainCaptureSpec(opensees=ops)
    spec.line_stations(pg="Beam", components=list(COMPS), name="girder")
    cap_path = str(tmp / "capture.h5")
    live: dict[int, list[np.ndarray]] = {e: [] for e in cols + beam}
    with ops.domain_capture(spec, path=cap_path, ops=emitter.ops) as cap:
        cap.begin_stage("load", kind="static")
        for _ in range(2):
            assert emitter.analyze(steps=1) == 0
            cap.step(t=emitter.ops.getTime())
            for e in live:         # (2, 3): end i, end j negated
                lf = np.asarray(
                    emitter.ops.eleResponse(fem_to_ops[e], "localForce"))
                assert lf.shape == (6,), (family, e, lf)
                live[e].append(np.stack([lf[0:3], -lf[3:6]]))
        cap.end_stage()
    emitter.ops.remove("recorders")

    capture = None
    if captured:
        om = OpenSeesModel.from_h5(cap_path, fem_root="/model")
        with Results.from_native(cap_path, model=om) as rc:
            capture = {
                c: rc.elements.line_stations.get(component=c, ids=beam)
                for c in COMPS
            }
    return {
        "results": Results.from_mpco(mpco, fem=fem, model_h5=model_h5),
        "cols": cols, "beam": beam, "capture": capture,
        "live": {e: np.stack(v) for e, v in live.items()},
    }


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def portal(request: pytest.FixtureRequest,
           tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = _solve(tmp_path_factory.mktemp(f"mpco_2d_{request.param}"),
                 request.param)
    out["family"] = request.param
    return out


def _rows(slab: Any, eid: int) -> list[int]:
    return [i for i, x in enumerate(np.asarray(slab.element_index))
            if int(x) == eid]


def test_available_components_list_all_three(portal: dict[str, Any]) -> None:
    comps = portal["results"].elements.line_stations.available_components()
    assert set(COMPS) <= set(comps), (portal["family"], comps)


@pytest.mark.parametrize("k, comp", list(enumerate(COMPS)))
def test_stations_are_the_local_force_ends(
    portal: dict[str, Any], k: int, comp: str,
) -> None:
    """Girder and columns: two stations at xi = -1, +1 per element."""
    elems = portal["beam"] + portal["cols"]
    slab = portal["results"].elements.line_stations.get(
        component=comp, ids=elems,
    )
    assert sorted({int(e) for e in slab.element_index}) == sorted(elems)
    values = np.asarray(slab.values)                   # (T, sum_S)
    assert values.shape[0] == 2
    xi = np.asarray(slab.station_natural_coord)
    for eid in elems:
        rows = _rows(slab, eid)
        np.testing.assert_array_equal(xi[rows], [-1.0, 1.0])
        np.testing.assert_allclose(
            values[:, rows], portal["live"][eid][:, :, k],
            rtol=RTOL, atol=ATOL,
            err_msg=f"{comp} of {portal['family']} element {eid}",
        )


@pytest.mark.parametrize("comp", COMPS)
def test_elastic_girder_matches_domain_capture(
    portal: dict[str, Any], comp: str,
) -> None:
    if portal["capture"] is None:
        pytest.skip(f"{portal['family']}: DomainCapture reads its sections, "
                    f"not localForce")
    mpco = portal["results"].elements.line_stations.get(
        component=comp, ids=portal["beam"],
    )
    cap = portal["capture"][comp]
    for eid in portal["beam"]:
        np.testing.assert_allclose(
            np.asarray(mpco.values)[:, _rows(mpco, eid)],
            np.asarray(cap.values)[:, _rows(cap, eid)],
            rtol=RTOL, atol=ATOL,
            err_msg=f"{comp} of {portal['family']} girder {eid}",
        )
