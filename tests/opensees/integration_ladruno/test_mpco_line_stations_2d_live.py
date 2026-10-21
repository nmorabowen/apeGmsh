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
"localForce")`` in the force-beam ``section force`` convention every route
shares (#2009): N and M are ``[-end_i, +end_j]``, V is ``[end_i, -end_j]``,
for elastic, force and disp beams alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.results import Results

from tests.test_results_recorder_tag_parity import _portal_fem

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
ATOL = 1e-9          # moments near a zero crossing
COMPS = ("axial_force", "shear_y", "bending_moment_z")
#: #2009: end-i sign of (N, V, M), written out (the oracle does not call the
#: table it checks); station 2 takes the opposite sign of end j.
SIGN = np.array([-1.0, 1.0, -1.0])


def _sectioned(make: str) -> Callable[[apeSees, Any], Any]:
    def build(ops: apeSees, tr: Any) -> Any:
        sec = ops.section.Elastic(E=2.0e11, A=0.02, Iz=3.0e-4)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        return getattr(ops.element, make)(pg="Beam", transf=tr,
                                          integration=integ)
    return build


#: girder class -> bridge builder
FAMILIES: dict[str, Callable[[apeSees, Any], Any]] = {
    "ElasticBeam2d": (
        lambda ops, tr: ops.element.elasticBeamColumn(
            pg="Beam", transf=tr, A=0.02, E=2.0e11, Iz=3.0e-4)),
    "ElasticTimoshenkoBeam2d": (
        lambda ops, tr: ops.element.ElasticTimoshenkoBeam(
            pg="Beam", transf=tr, E=2.0e11, G=8.0e10, A=0.02, Iz=3.0e-4,
            Avy=0.015)),
    "ForceBeamColumn2d": _sectioned("forceBeamColumn"),
    "DispBeamColumn2d": _sectioned("dispBeamColumn"),
    "LadrunoDispBeamColumn2d": _sectioned("LadrunoDispBeamColumn"),
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
    build_girder = FAMILIES[family]
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

    live: dict[int, list[np.ndarray]] = {e: [] for e in cols + beam}
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
        for e in live:             # (2, 3): force-beam section convention
            lf = np.asarray(
                emitter.ops.eleResponse(fem_to_ops[e], "localForce"))
            assert lf.shape == (6,), (family, e, lf)
            live[e].append(np.stack([SIGN * lf[0:3], -SIGN * lf[3:6]]))
    emitter.ops.remove("recorders")

    return {
        "results": Results.from_mpco(mpco, fem=fem, model_h5=model_h5),
        "cols": cols, "beam": beam,
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


# ---------------------------------------------------------------------------
# A refused bucket blocks only its own elements (#2020 review F1)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mixed_3d(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two 3-D cantilevers: ElasticBeam3d (tag 1), ForceBeamColumnCBDI3d (2).

    The CBDI3d tags its 12-value localForce with the six 2-D names, so
    MPCO writes a 6-column, zero-filled bucket that the reader refuses.
    """
    from apeGmsh.opensees.emitter.live import get_ops

    o = get_ops()
    o.wipe()
    o.model("basic", "-ndm", 3, "-ndf", 6)
    for n, (x, y) in enumerate([(0, 0), (3, 0), (0, 2), (3, 2)], start=1):
        o.node(n, float(x), float(y), 0.0)
    o.fix(1, 1, 1, 1, 1, 1, 1)
    o.fix(3, 1, 1, 1, 1, 1, 1)
    o.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    o.element("elasticBeamColumn", 1, 1, 2, 0.02, 2.0e11, 8.0e10,
              1.0e-4, 3.0e-4, 3.0e-4, 1)
    o.section("Elastic", 1, 2.0e11, 0.02, 3.0e-4, 3.0e-4, 8.0e10, 1.0e-4)
    o.beamIntegration("Lobatto", 1, 1, 4)
    o.element("forceBeamColumnCBDI", 2, 3, 4, 1, 1)
    o.timeSeries("Linear", 1)
    o.pattern("Plain", 1, 1)
    o.load(2, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0)
    o.load(4, 0.0, 1000.0, 0.0, 0.0, 0.0, 0.0)
    path = str(tmp_path_factory.mktemp("mpco_mixed_3d") / "mixed.mpco")
    o.recorder("mpco", path, "-N", "displacement", "-E", "localForce")
    o.constraints("Plain")
    o.numberer("RCM")
    o.system("BandGeneral")
    o.test("NormDispIncr", 1e-10, 10)
    o.algorithm("Newton")
    o.integrator("LoadControl", 1.0)
    o.analysis("Static")
    assert o.analyze(1) == 0
    lf = np.asarray(o.eleResponse(1, "localForce"))
    o.remove("recorders")
    o.wipe()
    return {"path": path, "beam_shear": [lf[1], -lf[7]]}


def test_a_refused_bucket_blocks_only_its_own_elements(
    mixed_3d: dict[str, Any],
) -> None:
    from apeGmsh.results.readers._mpco import MPCOReader
    from apeGmsh.results.readers._protocol import ResultLevel

    with MPCOReader(mixed_3d["path"]) as r:
        sid = r.stages()[0].id
        comps = r.available_components(sid, ResultLevel.LINE_STATIONS)
        assert {"shear_y", "shear_z", "bending_moment_y"} <= set(comps)
        slab = r.read_line_stations(sid, "shear_y", element_ids=np.array([1]))
        np.testing.assert_array_equal(slab.element_index, [1, 1])
        np.testing.assert_allclose(slab.values[0], mixed_3d["beam_shear"],
                                   rtol=RTOL)
        np.testing.assert_allclose(slab.values[0], [-1000.0, -1000.0],
                                   rtol=1e-9)
        for ids in (np.array([2]), None):
            with pytest.raises(ValueError,
                               match=r"ForceBeamColumnCBDI3d.*SPATIAL_DIM"):
                r.read_line_stations(sid, "shear_y", element_ids=ids)
