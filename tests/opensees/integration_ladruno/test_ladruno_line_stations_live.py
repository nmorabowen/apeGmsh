"""Fork lane: beam line stations read back from ``.ladruno`` (#1741, gap G3).

The #1740 portal frame, solved through the bridge with sequential tags, so
FEM ids are not OpenSees tags: two force-beam columns (``Cols``, Lobatto-4,
elastic section answering ``P, Mz``) and an elastic-beam girder (``Beam``,
no integration points). The fork's ``ladruno`` recorder writes
``section.force``, ``localForce`` and ``basicForce``.

Every station is checked against the live domain at ``rtol=1e-10``:

* a column's ``axial_force`` / ``bending_moment_z`` stations are its four
  ``eleResponse(tag, "section", ip, "force")`` values, and nothing else: the
  ``localForce`` and ``basicForce`` buckets the column also writes add no
  stations;
* the girder's two stations at xi = -1, +1 are ``localForce`` in the
  force-beam ``section force`` convention every route shares (#2009):
  N and M are ``[-end_i, +end_j]``, V is ``[end_i, -end_j]``. The 2-D bucket
  spells shear and moment ``V``/``M``; before #1741 both were dropped
  silently;
* with only ``basicForce`` recorded, no beam has shear or moment stations:
  the ``V``/``M`` alias stays off ``basicForce``'s basic-system end moments.
"""
from __future__ import annotations

from typing import Any

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
N_IP = 4
COMPS = ("axial_force", "shear_y", "bending_moment_z")
#: #2009: end-i sign of (N, V, M); station 2 takes the opposite sign of end j.
SIGN = np.array([-1.0, 1.0, -1.0])


def _fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


@pytest.fixture(scope="module")
def portal(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _solve(tmp_path_factory.mktemp("ladruno_line_stations"),
                  ("section.force", "localForce", "basicForce"))


@pytest.fixture(scope="module")
def basic_only(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _solve(tmp_path_factory.mktemp("ladruno_basic_force"),
                  ("basicForce",))


def _solve(tmp: Any, elem_responses: tuple[str, ...]) -> dict[str, Any]:
    """Solve the portal in two steps, recording ``elem_responses``."""
    fem = _portal_fem()
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    ops.model(ndm=2, ndf=3)
    tr = ops.geomTransf.Linear()
    # Girder first: its sequential tags are taken before the columns'.
    ops.element.elasticBeamColumn(pg="Beam", transf=tr,
                                  A=0.02, E=2.0e11, Iz=3.0e-4)
    sec = ops.section.Elastic(E=2.0e11, A=0.03, Iz=5.0e-4)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=N_IP)
    ops.element.forceBeamColumn(pg="Cols", transf=tr, integration=integ)
    ops.fix(label="base", dofs=(1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="tl"), forces=(10e3, -30e3, 0.0))
        p.load(node=ops.nodes.one(label="tr"), forces=(0.0, -50e3, 4e3))
    path = str(tmp / "portal.ladruno")
    ops.recorder.Ladruno(file=path, elem_responses=elem_responses)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
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

    col_live: dict[int, list[np.ndarray]] = {e: [] for e in cols}
    beam_live: dict[int, list[np.ndarray]] = {e: [] for e in beam}
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
        o = emitter.ops
        for e in cols:     # (N_IP, 2): P, Mz
            col_live[e].append(np.stack([
                np.asarray(o.eleResponse(fem_to_ops[e], "section", ip, "force"))
                for ip in range(1, N_IP + 1)
            ]))
        for e in beam:     # (2, 3): force-beam section convention
            lf = np.asarray(o.eleResponse(fem_to_ops[e], "localForce"))
            beam_live[e].append(np.stack([SIGN * lf[0:3], -SIGN * lf[3:6]]))
    emitter.ops.remove("recorders")
    return {
        "results": Results.from_ladruno(path, fem=fem, model_h5=model_h5),
        "cols": cols, "beam": beam,
        "col_live": {e: np.stack(v) for e, v in col_live.items()},
        "beam_live": {e: np.stack(v) for e, v in beam_live.items()},
    }


def _rows(slab: Any, eid: int) -> list[int]:
    return [i for i, x in enumerate(np.asarray(slab.element_index))
            if int(x) == eid]


@pytest.mark.parametrize("comp, k", [("axial_force", 0),
                                     ("bending_moment_z", 1)])
def test_column_stations_are_its_section_forces(
    portal: dict[str, Any], comp: str, k: int,
) -> None:
    slab = portal["results"].elements.line_stations.get(
        component=comp, ids=portal["cols"],
    )
    assert sorted({int(e) for e in slab.element_index}) == sorted(portal["cols"])
    values = np.asarray(slab.values)                   # (T, sum_S)
    for eid in portal["cols"]:
        rows = _rows(slab, eid)
        assert len(rows) == N_IP, (eid, rows)          # no end-force extras
        np.testing.assert_allclose(
            values[:, rows], portal["col_live"][eid][:, :, k],
            rtol=RTOL, atol=ATOL, err_msg=f"{comp} of column {eid}",
        )


@pytest.mark.parametrize("k, comp", list(enumerate(COMPS)))
def test_girder_stations_are_its_local_force_ends(
    portal: dict[str, Any], k: int, comp: str,
) -> None:
    slab = portal["results"].elements.line_stations.get(
        component=comp, ids=portal["beam"],
    )
    assert sorted({int(e) for e in slab.element_index}) == sorted(portal["beam"])
    values = np.asarray(slab.values)
    xi = np.asarray(slab.station_natural_coord)
    for eid in portal["beam"]:
        rows = _rows(slab, eid)
        np.testing.assert_array_equal(xi[rows], [-1.0, 1.0])
        np.testing.assert_allclose(
            values[:, rows], portal["beam_live"][eid][:, :, k],
            rtol=RTOL, atol=ATOL, err_msg=f"{comp} of girder {eid}",
        )


@pytest.mark.parametrize("comp", ["shear_y", "bending_moment_z"])
def test_basic_force_moments_are_not_line_stations(
    basic_only: dict[str, Any], comp: str,
) -> None:
    """The 2-D ``V``/``M`` alias is for ``localForce`` only.

    A 2-D ``basicForce`` bucket is ``N,M_1,M_2``: end moments in the basic
    system, not section forces, so with no ``section.force`` or
    ``localForce`` recorded no beam has a shear or moment station.
    Applying the alias to ``basicForce`` would surface ``M_1``/``M_2`` as
    ``bending_moment_z`` stations here.
    """
    ls = basic_only["results"].elements.line_stations
    assert comp not in ls.available_components()
    slab = ls.get(component=comp, ids=basic_only["beam"] + basic_only["cols"])
    assert np.asarray(slab.element_index).size == 0
