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

A 3-D ``basicForce`` bucket is ``N,Mz_1,Mz_2,My_1,My_2,T`` (#2011). Its end
moments read as stations in the same section convention as ``localForce``:
every fork class that names them reports ``localForce`` end moments equal
to them, so station 1 = ``-Mz_1`` and station 2 = ``+Mz_2`` (likewise
``My``). A 3-D cantilever per beam family that records ``basicForce``
(``elasticBeamColumn``, ``forceBeamColumn``) checks this against the live
``localForce`` and against statics.
"""
from __future__ import annotations

from typing import Any

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
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


# ---------------------------------------------------------------------------
# 3-D basicForce-only (#2011): one cantilever per beam family
# ---------------------------------------------------------------------------

#: Cantilever along +x, fixed at x = 0, tip load (Fx, Fy, Fz, Mx, My, Mz):
#: the card's repro, on two elements so an interior end is checked too.
L3 = 4.0
TIP = (1.0e3, 2.0e3, 3.0e3, 4.0e2, 5.0e2, 6.0e2)
#: localForce index of (end i, end j) per moment, and its #2009 end-i sign.
MOMENT_LF = {"bending_moment_y": (4, 10, -1.0),
             "bending_moment_z": (5, 11, -1.0)}
#: basicForce index of (station 1, station 2) per moment.
MOMENT_BF = {"bending_moment_y": (3, 4), "bending_moment_z": (1, 2)}


def _closed_3d(comp: str, x: float) -> float:
    """Statics on the free body ``x..L``, section convention."""
    fx, fy, fz, mx, my, mz = TIP
    return {
        "axial_force": fx, "torsion": mx,
        "bending_moment_y": my - fz * (L3 - x),
        "bending_moment_z": mz + fy * (L3 - x),
    }[comp]


def _cantilever_fem(name: str) -> Any:
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        pa = geo.add_point(0.0, 0.0, 0.0)
        pm = geo.add_point(L3 / 2.0, 0.0, 0.0)
        pb = geo.add_point(L3, 0.0, 0.0)
        curves = [geo.add_line(pa, pm), geo.add_line(pm, pb)]
        for c in curves:
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        g.physical.add(1, curves, name="Beam")
        g.labels.add(0, [pa], "a")
        g.labels.add(0, [pb], "b")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def _beam_3d(ops: apeSees, family: str, tr: Any) -> None:
    props = dict(E=2.0e11, A=0.03, Iz=5.0e-4, Iy=4.0e-4, G=8.0e10, J=3.0e-4)
    if family == "elasticBeamColumn":
        ops.element.elasticBeamColumn(pg="Beam", transf=tr, **props)
    else:
        sec = ops.section.Elastic(**props)
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=N_IP)
        ops.element.forceBeamColumn(pg="Beam", transf=tr, integration=integ)


@pytest.fixture(scope="module", params=["elasticBeamColumn", "forceBeamColumn"])
def basic_only_3d(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    family = request.param
    tmp = tmp_path_factory.mktemp(f"ladruno_basic_force_3d_{family}")
    fem = _cantilever_fem(f"basic3d_{family}")
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))  # local y = Y, z = Z
    _beam_3d(ops, family, tr)
    ops.fix(label="a", dofs=(1, 1, 1, 1, 1, 1))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"), forces=TIP)
    path = str(tmp / "cantilever.ladruno")
    ops.recorder.Ladruno(file=path, elem_responses=("basicForce",))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()

    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    fem_to_ops = _fem_to_ops(model_h5)
    eids = [int(e) for e in fem.elements.select(pg="Beam").ids]
    assert any(fem_to_ops[e] != e for e in eids)       # not vacuous
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    ends = {int(e): (float(coords[int(c[0])][0]), float(coords[int(c[-1])][0]))
            for grp in fem.elements
            for e, c in zip(grp.ids, grp.connectivity)}

    lf: dict[int, list[np.ndarray]] = {e: [] for e in eids}
    bf: dict[int, list[np.ndarray]] = {e: [] for e in eids}
    for _ in range(2):
        assert emitter.analyze(steps=1) == 0
        for e in eids:
            o = emitter.ops
            lf[e].append(np.asarray(o.eleResponse(fem_to_ops[e], "localForce")))
            bf[e].append(np.asarray(o.eleResponse(fem_to_ops[e], "basicForce")))
    emitter.ops.remove("recorders")
    return {
        "family": family, "path": path, "fem": fem, "model_h5": model_h5,
        "results": Results.from_ladruno(path, fem=fem, model_h5=model_h5),
        "eids": eids, "ends": ends,
        "lf": {e: np.stack(v) for e, v in lf.items()},     # (T, 12)
        "bf": {e: np.stack(v) for e, v in bf.items()},     # (T, 6)
    }


@pytest.mark.parametrize("comp", ["bending_moment_y", "bending_moment_z"])
def test_basic_force_3d_moments_read_in_section_convention(
    basic_only_3d: dict[str, Any], comp: str,
) -> None:
    """A ``basicForce``-only file's 3-D end moments read as the stations
    ``localForce`` gives the same element: ``[-M_1, +M_2]`` of the basic
    moments, equal to statics, at ``rtol=1e-10``."""
    sol = basic_only_3d
    ls = sol["results"].elements.line_stations
    assert comp in ls.available_components()
    slab = ls.get(component=comp, ids=sol["eids"])
    values = np.asarray(slab.values)
    xi = np.asarray(slab.station_natural_coord)
    i_lf, j_lf, s = MOMENT_LF[comp]
    i_bf, j_bf = MOMENT_BF[comp]
    for eid in sol["eids"]:
        rows = _rows(slab, eid)
        np.testing.assert_array_equal(xi[rows], [-1.0, 1.0])
        got = values[:, rows]
        lf, bf = sol["lf"][eid], sol["bf"][eid]
        from_lf = np.stack([s * lf[:, i_lf], -s * lf[:, j_lf]], axis=1)
        from_bf = np.stack([-bf[:, i_bf], bf[:, j_bf]], axis=1)
        xa, xb = sol["ends"][eid]
        closed = np.outer([0.5, 1.0], [_closed_3d(comp, xa),
                                       _closed_3d(comp, xb)])
        msg = f"{sol['family']} {comp} of element {eid}"
        for want, what in ((from_lf, "localForce"), (from_bf, "basicForce"),
                           (closed, "statics")):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL,
                                       err_msg=f"{msg} vs {what}")


@pytest.mark.parametrize("comp, k", [("axial_force", 0), ("torsion", 5)])
def test_basic_force_3d_axial_and_torsion_are_live_basic_force(
    basic_only_3d: dict[str, Any], comp: str, k: int,
) -> None:
    """``N`` and ``T`` stay single stations at xi = 0, equal to the live
    ``basicForce`` value and to statics."""
    sol = basic_only_3d
    slab = sol["results"].elements.line_stations.get(
        component=comp, ids=sol["eids"])
    values = np.asarray(slab.values)
    xi = np.asarray(slab.station_natural_coord)
    for eid in sol["eids"]:
        rows = _rows(slab, eid)
        assert len(rows) == 1 and xi[rows[0]] == 0.0, (eid, rows)
        np.testing.assert_allclose(
            values[:, rows[0]], sol["bf"][eid][:, k], rtol=RTOL, atol=ATOL,
            err_msg=f"{sol['family']} {comp} of element {eid}")
        np.testing.assert_allclose(
            values[:, rows[0]], np.array([0.5, 1.0]) * _closed_3d(comp, 0.0),
            rtol=RTOL, err_msg=f"{sol['family']} {comp} vs statics")


def test_basic_force_two_station_non_moment_is_refused_by_name(
    basic_only_3d: dict[str, Any], tmp_path: Any,
) -> None:
    """A ``basicForce`` two-station component other than an end moment has
    no basic-to-section identity: it lists and reads no stations, and both
    warn naming the bucket and the component. The real file's ``Mz_1,Mz_2``
    are relabelled ``Vy_1,Vy_2`` to make one."""
    sol = basic_only_3d
    path = str(tmp_path / "relabelled.ladruno")
    with open(sol["path"], "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    relabelled = 0
    with h5py.File(path, "r+") as f:
        def _relabel(name: str, obj: Any) -> None:
            nonlocal relabelled
            if name.endswith("COLUMN_MAP") and "/basicForce/" in name:
                raw = obj.attrs["COMP_NAMES"]
                text = raw.decode() if isinstance(raw, bytes) else str(raw)
                obj.attrs["COMP_NAMES"] = (text.replace("Mz_1", "Vy_1")
                                           .replace("Mz_2", "Vy_2"))
                relabelled += 1
        f.visititems(_relabel)
    assert relabelled
    match = r"basicForce bucket .*'shear_y' at stations \[1, 2\]"
    with pytest.warns(UserWarning, match=match):
        r = Results.from_ladruno(path, fem=sol["fem"],
                                 model_h5=sol["model_h5"])
        ls = r.elements.line_stations
        available = ls.available_components()
    assert "shear_y" not in available
    assert "bending_moment_y" in available       # the untouched pair reads
    with pytest.warns(UserWarning, match=match):
        slab = ls.get(component="shear_y", ids=sol["eids"])
    assert np.asarray(slab.element_index).size == 0
