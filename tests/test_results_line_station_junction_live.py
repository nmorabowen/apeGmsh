"""Live: elastic-beam line stations meet the force beams at a junction (#2009).

A straight bar along +x, x = 0..3, of three one-element lines: a force
beam [0, 1], an elastic beam [1, 2] and a force beam [2, 3], fixed at
x = 0 and loaded at the tip. The force beams integrate a shear-capable
``section Elastic`` (2-D ``P, Mz, Vy``; 3-D ``P, Mz, Vy, My, Vz, T``) at
three Lobatto points, so their end points sit on the shared nodes.

The elastic beam's two line stations are synthesised from ``localForce``.
Each must equal the adjacent force beam's live ``section force`` at the
shared node, at ``rtol=1e-10``, for every component and step: N, V and M
are then continuous across both junctions. Before #2009 every route read
N and M (and, in 3-D, Vz and T) with the opposite sign; only Vy agreed.

This file holds the ``.out`` route, on the backend the bridge resolves
(stock openseespy in CI); an in-process capture route sat beside it until
#2191 deleted that route. The MPCO and ``.ladruno`` routes of the same
model are in
``tests/opensees/integration_ladruno/test_line_station_junction_live.py``.

The oracle is checked twice: the live section force against the closed
form of the cantilever (``test_force_beam_sections_are_statics``), and
each read-back station against the live section force.

``ModElasticBeam2d`` is in the catalog but not here: the bridge has no
primitive for it, and its ``localForce`` response is all zeros on the
backends probed (fork 79e062367), so it has no station to pin.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import gmsh
import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh

pytestmark = pytest.mark.live

RTOL = 1e-10
LAMS = (0.5, 1.0)                       # LoadControl dlam = 0.5, two steps
LENGTH = 3.0
E_, G_, A_ = 2.0e11, 8.0e10, 0.03
IZ, IY, J_ = 5.0e-4, 4.0e-4, 3.0e-4
ALPHA = 0.8
#: tip load per dimension: 2-D (Fx, Fy, Mz), 3-D (Fx, Fy, Fz, Mx, My, Mz)
TIP = {2: (10.0e3, -3.0e3, 2.0e3),
       3: (10.0e3, -3.0e3, 7.0e3, 2.0e3, 1.0e3, -1.5e3)}
COMPS = {2: ("axial_force", "shear_y", "bending_moment_z"),
         3: ("axial_force", "shear_y", "shear_z", "torsion",
             "bending_moment_y", "bending_moment_z")}
#: the force beam's ``section force`` order (ElasticShear2d / 3d)
SECTION_ORDER = {2: ("axial_force", "bending_moment_z", "shear_y"),
                 3: ("axial_force", "bending_moment_z", "shear_y",
                     "bending_moment_y", "shear_z", "torsion")}


def _ndm(cls: str) -> int:
    return 3 if cls.endswith("3d") else 2


#: elastic class -> bridge builder over the ``Mid`` physical group
ELASTIC: dict[str, Callable[[Any, Any], Any]] = {
    "ElasticBeam2d": lambda ops, tr: ops.element.elasticBeamColumn(
        pg="Mid", transf=tr, A=A_, E=E_, Iz=IZ),
    "ElasticTimoshenkoBeam2d": lambda ops, tr: ops.element.ElasticTimoshenkoBeam(
        pg="Mid", transf=tr, E=E_, G=G_, A=A_, Iz=IZ, Avy=ALPHA * A_),
    "ElasticBeam3d": lambda ops, tr: ops.element.elasticBeamColumn(
        pg="Mid", transf=tr, A=A_, E=E_, Iz=IZ, Iy=IY, G=G_, J=J_),
    "ElasticTimoshenkoBeam3d": lambda ops, tr: ops.element.ElasticTimoshenkoBeam(
        pg="Mid", transf=tr, E=E_, G=G_, A=A_, Iz=IZ, Iy=IY, J=J_,
        Avy=ALPHA * A_, Avz=ALPHA * A_),
}


def closed_form(ndm: int, comp: str, x: float) -> float:
    """Force-beam section resultant of the tip-loaded cantilever at ``x``
    (the +x face; OpenSees reports Vy as ``dMz/dx``, i.e. ``-Fy``)."""
    if ndm == 2:
        fx, fy, mz = TIP[2]
        fz = mx = my = 0.0
    else:
        fx, fy, fz, mx, my, mz = TIP[3]
    arm = LENGTH - x
    return {
        "axial_force": fx, "shear_y": -fy, "shear_z": fz, "torsion": mx,
        "bending_moment_y": my - arm * fz, "bending_moment_z": mz + arm * fy,
    }[comp]


def _bar_fem(name: str) -> Any:
    """Three one-element lines along +x: ``Left`` [0, 1], ``Mid`` [1, 2],
    ``Right`` [2, 3]; labels ``a`` (x = 0) and ``b`` (x = 3)."""
    with apeGmsh(model_name=name, verbose=False) as g:
        geo = g.model.geometry
        pts = [geo.add_point(float(x), 0.0, 0.0) for x in range(4)]
        lines = [geo.add_line(pts[k], pts[k + 1]) for k in range(3)]
        for c in lines:
            gmsh.model.mesh.setTransfiniteCurve(c, 2)
        for c, pg in zip(lines, ("Left", "Mid", "Right")):
            g.physical.add(1, [c], name=pg)
        g.labels.add(0, [pts[0]], "a")
        g.labels.add(0, [pts[3]], "b")
        g.mesh.generation.generate(dim=1)
        return g.mesh.queries.get_fem_data(dim=1)


def fem_to_ops(model_h5: str) -> dict[int, int]:
    """``{fem_eid: ops_tag}`` from the bridge's own ``element_meta``."""
    with h5py.File(model_h5, "r") as f:
        meta = f["/opensees/element_meta"]
        return {
            int(e): int(t)
            for tok in meta
            for t, e in zip(meta[tok]["ids"][...], meta[tok]["fem_eids"][...])
        }


def build_junction(fem: Any, cls: str,
                   declare_recorders: Callable[[Any], None] | None = None,
                   build_mid: Callable[[Any, Any, Any], Any] | None = None,
                   ) -> Any:
    """The bridge model of the bar; ``declare_recorders(ops)`` adds the
    file recorders a fork route needs before the analysis chain.
    ``build_mid(ops, transf, integration)`` replaces the elastic ``Mid``
    beam with another family (a sectioned beam on the same section)."""
    from apeGmsh.opensees import apeSees

    ndm = _ndm(cls)
    ops = apeSees(fem, default_orientation=None)       # sequential tags
    if ndm == 2:
        ops.model(ndm=2, ndf=3)
        tr = ops.geomTransf.Linear()
        sec = ops.section.Elastic(E=E_, A=A_, Iz=IZ, G=G_, alphaY=ALPHA)
    else:
        ops.model(ndm=3, ndf=6)
        tr = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        sec = ops.section.Elastic(E=E_, A=A_, Iz=IZ, Iy=IY, G=G_, J=J_,
                                  alphaY=ALPHA, alphaZ=ALPHA)
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    # Mid first: its sequential tag is taken before the force beams'.
    if build_mid is None:
        ELASTIC[cls](ops, tr)
    else:
        build_mid(ops, tr, integ)
    for pg in ("Left", "Right"):
        ops.element.forceBeamColumn(pg=pg, transf=tr, integration=integ)
    ops.fix(label="a", dofs=(1,) * (3 if ndm == 2 else 6))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=ops.nodes.one(label="b"), forces=TIP[ndm])
    if declare_recorders is not None:
        declare_recorders(ops)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    # Newton, not Linear: under Linear a disp beam's ``localForce`` reads 0
    # (no state determination after the solve). One iteration converges.
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=LAMS[0])
    ops.analysis.Static()
    return ops


def ids(fem: Any) -> dict[str, int]:
    """FEM element id of ``Left``, ``Mid`` and ``Right``; each runs +x."""
    coords = {int(n): np.asarray(c, dtype=float)
              for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    out = {}
    for pg in ("Left", "Mid", "Right"):
        (eid,) = [int(e) for e in fem.elements.select(pg=pg).ids]
        out[pg] = eid
    for grp in fem.elements:
        for eid, conn in zip(grp.ids, grp.connectivity):
            assert coords[int(conn[0])][0] < coords[int(conn[-1])][0], eid
    return out


def junction_truth(o: Any, cls: str, tags: dict[str, int]) -> np.ndarray:
    """``(2, n_comp)``: the force beams' live ``section force`` at x = 1
    (Left, its last point) and x = 2 (Right, its first point), in
    :data:`COMPS` order."""
    ndm = _ndm(cls)
    order = SECTION_ORDER[ndm]
    out = np.empty((2, len(COMPS[ndm])))
    for row, (pg, ip) in enumerate((("Left", 3), ("Right", 1))):
        sec = np.asarray(o.eleResponse(tags[pg], "section", ip, "force"),
                         dtype=float)
        assert sec.shape == (len(order),), (cls, pg, sec)
        for k, comp in enumerate(order):
            out[row, COMPS[ndm].index(comp)] = sec[k]
    return out


def close(got: np.ndarray, want: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


def station_values(r: Any, comp: str, eid: int) -> np.ndarray:
    """``(T, 2)`` stations of ``eid`` read back, asserting xi = -1, +1."""
    slab = r.elements.line_stations.get(component=comp, ids=[eid])
    rows = [i for i, x in enumerate(np.asarray(slab.element_index))
            if int(x) == eid]
    assert len(rows) == 2, f"{comp} of element {eid}: rows {rows}"
    np.testing.assert_array_equal(
        np.asarray(slab.station_natural_coord)[rows], [-1.0, 1.0])
    return np.asarray(slab.values)[:, rows]


# ---------------------------------------------------------------------------
# .out
# ---------------------------------------------------------------------------


def _solve(tmp: Path, cls: str) -> dict[str, Any]:
    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter
    from apeGmsh.results import Results
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord, ResolvedRecorderSpec,
    )

    ndm = _ndm(cls)
    fem = _bar_fem(f"junction_{cls}")
    eid = ids(fem)
    ops = build_junction(fem, cls)
    model_h5 = str(tmp / "model.h5")
    ops.h5(model_h5)
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    to_ops = fem_to_ops(model_h5)
    assert any(to_ops[e] != e for e in eid.values())    # not vacuous
    tags = {pg: to_ops[e] for pg, e in eid.items()}
    assert emitter.ops.eleType(tags["Mid"]) == cls

    rspec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="line_stations", name="mid", components=COMPS[ndm],
            dt=None, n_steps=None, element_ids=np.asarray([eid["Mid"]]),
            element_class_name=cls,
        ),),
    )
    model = OpenSeesModel.from_h5(model_h5)
    out_dir = tmp / "out"
    truth = []
    with rspec.emit_recorders(str(out_dir), ops=emitter.ops,
                              model=model) as live:
        live.begin_stage("s", kind="static")
        for _ in LAMS:
            assert emitter.analyze(steps=1) == 0
            truth.append(junction_truth(emitter.ops, cls, tags))
        live.end_stage()
    emitter.ops.remove("recorders")

    return {
        "results": {
            "out": Results.from_recorders(
                rspec, out_dir, fem=fem, stage_id="s", model=model,
                cache_root=tmp / "cache"),
        },
        "mid": eid["Mid"],
        "truth": np.stack(truth),                       # (T, 2, n_comp)
    }


@pytest.fixture(scope="module", params=sorted(ELASTIC))
def solved(request: pytest.FixtureRequest,
           tmp_path_factory: pytest.TempPathFactory) -> Any:
    pytest.importorskip("openseespy.opensees")
    sol = _solve(tmp_path_factory.mktemp(f"junction_{request.param}"),
                 request.param)
    sol["cls"] = request.param
    yield sol
    sol["results"]["out"].close()


def test_force_beam_sections_are_statics(solved: dict[str, Any]) -> None:
    """The oracle: each live junction section force is the closed form."""
    ndm = _ndm(solved["cls"])
    for k, comp in enumerate(COMPS[ndm]):
        want = np.outer(LAMS, [closed_form(ndm, comp, 1.0),
                               closed_form(ndm, comp, 2.0)])
        close(solved["truth"][:, :, k], want, f"{solved['cls']} {comp}")


@pytest.mark.parametrize("route", ["out"])
def test_elastic_stations_meet_the_force_beams(
    solved: dict[str, Any], route: str,
) -> None:
    """Station 1 equals the left force beam at x = 1 and station 2 the
    right one at x = 2: every component is continuous at both junctions."""
    ndm = _ndm(solved["cls"])
    r = solved["results"][route]
    for k, comp in enumerate(COMPS[ndm]):
        close(station_values(r, comp, solved["mid"]),
              solved["truth"][:, :, k],
              f"{route} {solved['cls']} {comp} at the junctions")
