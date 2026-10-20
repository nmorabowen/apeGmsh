"""#2269 live oracles: a raw openseespy deck plus a FEMData archive records
ELEMENT results through ``spec.emit_default`` on the classic-recorder route.

A FEMData archive pairs no FEM element id with an OpenSees tag. The route
pairs each recorded FEM element with the live element of its own id only
when ``eleNodes`` is the FEM connectivity in order, and types it by
``eleType``; the read carries that pairing. The build is wrapped in
:class:`~tests.test_results_live_default_format.NoMPCO`, so the cases run
the classic-recorder route on pip openseespy (where it is the module
itself) and on a build that has MPCO.

- The ``column_to_surface`` case-study deck: every tet's Gauss stress
  equals the live ``eleResponse(e, "stresses")`` at rtol 1e-12.
- A planar portal: force-beam columns' line stations equal the live
  ``section force``, the elastic girder's two stations equal its signed
  ``localForce``, and its nodal forces equal ``globalForce``, at rtol
  1e-12.
- A deck whose element connectivity is not the FEM's refuses by id, and
  an element class with no bridge token refuses by name.
- A ``Brick`` of ``ASDConcrete3D`` pulled off a fixed base: its
  per-Gauss-point ``damage`` record reads through the live pairing and
  equals the live ``material <gp> damage`` at rtol 1e-12; the base points
  damage more than the top ones, which the catalog's point order places
  (#2290 A).
- A statically determinate truss: its Gauss ``axial_force`` record, which
  MPCO cannot write, takes the classic recorders on any stock build and
  equals the live ``axialForce`` at rtol 1e-12 and the joint equilibrium
  closed form (#2290 D).
- Two fibre-section cantilevers (a force beam, a displacement beam): each
  fibre's stress and strain equals the live ``fiberData2`` at rtol 1e-12,
  the section locations and fibre counts having been read from the live
  domain at ``__enter__``; each station's summed section forces are the
  closed form ``N = -PN``, ``|M| = PL (H - x)``, which checks the station
  order physically. A beam of elastic sections refuses by name (#2290 B).
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.opensees.live.test_femdata_archive_live import (
    _build_column,
    _column_fem,
)
from tests.test_results_live_default_format import NoMPCO

pytestmark = pytest.mark.live

_STEPS = 10
_RTOL = dict(rtol=1e-12, atol=0)
_TET_STRESS = ("stress_xx", "stress_yy", "stress_zz",
               "stress_xy", "stress_yz", "stress_xz")


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


def _record(category: str, name: str, comps: "tuple[str, ...]",
            ids: "np.ndarray", cls: "str | None") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(category=category, name=name,
                                  components=comps, dt=None, n_steps=None,
                                  element_ids=np.asarray(ids, dtype=np.int64),
                                  element_class_name=cls)


def _emit(spec: ResolvedRecorderSpec, tmp: Path, model_h5: Path, ops: Any,
          steps: int) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(tmp / "run", model_h5=model_h5,
                               ops=NoMPCO(ops)) as run:
            for _ in range(steps):
                assert ops.analyze(1) == 0
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return run.results(cache_root=tmp / "cache")


def _by_element(slab: Any, eid: int) -> "list[int]":
    index = [int(i) for i in np.asarray(slab.element_index)]
    return [i for i, x in enumerate(index) if x == eid]


# ---------------------------------------------------------------------------
# The column_to_surface case-study deck: tet Gauss stress
# ---------------------------------------------------------------------------


def test_column_deck_tet_gauss_stress_is_the_live_response(opspy, tmp_path) -> None:
    fem = _column_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_column(opspy, fem)
    tets = np.asarray(sorted(int(e) for g in fem.elements.select(
        element_type="tet4").groups() for e in g.ids), dtype=np.int64)
    assert tets.size > 1000                       # the case study's 1770
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "sigma", _TET_STRESS, tets,
                         "FourNodeTetrahedron"),))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        truth = np.array([opspy.eleResponse(int(e), "stresses") for e in tets])
        assert truth.shape == (tets.size, 6) and np.abs(truth).max() > 0
        for k, comp in enumerate(_TET_STRESS):
            slab = res.elements.gauss.get(component=comp, ids=tets)
            index = np.asarray(slab.element_index, dtype=np.int64)
            assert sorted(index.tolist()) == tets.tolist()   # one GP each
            order = np.argsort(index)
            np.testing.assert_allclose(
                np.asarray(slab.values)[-1][order], truth[:, k], **_RTOL,
                err_msg=comp)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# A planar portal: line stations and nodal forces
# ---------------------------------------------------------------------------

E, A, IZ, H, W = 2.0e11, 1.0e-2, 8.0e-6, 3.0, 5.0
N_IP = 4


def _portal_fem() -> FEMData:
    """Columns 1 (nodes 1-2) and 2 (nodes 4-3), girder 3 (nodes 2-3)."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, H, 0.0], [W, H, 0.0],
                       [W, 0.0, 0.0]])
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=3)
    group = ElementGroup(
        element_type=line, ids=np.array([1, 2, 3], dtype=np.int64),
        connectivity=np.array([[1, 2], [4, 3], [2, 3]], dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 5, dtype=np.int64),
                            node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=4, n_elems=3, bandwidth=3, types=[line]),
        model_name="femdata_elements_portal",
    )


def _build_portal(ops: Any, fem: FEMData, *, girder: "tuple[int, int]" = (2, 3),
                  girder_type: str = "elasticBeamColumn") -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]))
    ops.fix(1, 1, 1, 1)
    ops.fix(4, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    ops.section("Elastic", 1, E, A, IZ)
    ops.beamIntegration("Lobatto", 1, 1, N_IP)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.element("forceBeamColumn", 2, 4, 3, 1, 1)
    if girder_type == "elasticBeamColumn":
        ops.element("elasticBeamColumn", 3, *girder, A, E, IZ, 1)
    else:
        ops.element(girder_type, 3, *girder, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0e4, -2.0e4, 0.0)
    ops.load(3, 0.0, -3.0e4, 5.0e3)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / _STEPS)
    ops.analysis("Static")


def test_portal_line_stations_and_nodal_forces_are_the_live_response(
    opspy, tmp_path,
) -> None:
    fem = _portal_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_portal(opspy, fem)
    cols, beam = np.array([1, 2]), np.array([3])
    comps = ("axial_force", "shear_y", "bending_moment_z")
    nodal = ("nodal_resisting_force_x", "nodal_resisting_force_y",
             "nodal_resisting_moment_z")
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            _record("line_stations", "cols", comps, cols, "ForceBeamColumn2d"),
            _record("line_stations", "beam", comps, beam, "ElasticBeam2d"),
            _record("elements", "beam_nodal", nodal, beam, "ElasticBeam2d"),
        ))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        ls = res.elements.line_stations
        # The elastic section's (P, Mz) at each Lobatto point.
        for comp, k in (("axial_force", 0), ("bending_moment_z", 1)):
            slab = ls.get(component=comp, ids=cols)
            vals = np.asarray(slab.values)[-1]
            for e in cols.tolist():
                truth = [opspy.eleResponse(e, "section", ip, "force")[k]
                         for ip in range(1, N_IP + 1)]
                rows = _by_element(slab, e)
                assert len(rows) == N_IP
                np.testing.assert_allclose(vals[rows], truth, **_RTOL,
                                           err_msg=f"{comp} of column {e}")
        # The girder's two stations, in the force-beam section convention
        # every route signs localForce with (#2009).
        lf = np.asarray(opspy.eleResponse(3, "localForce"))
        sign = np.array([-1.0, 1.0, -1.0])
        girder = np.stack([sign * lf[0:3], -sign * lf[3:6]])
        assert np.abs(girder).max() > 0
        for k, comp in enumerate(comps):
            slab = ls.get(component=comp, ids=beam)
            rows = _by_element(slab, 3)
            np.testing.assert_array_equal(
                np.asarray(slab.station_natural_coord)[rows], [-1.0, 1.0])
            np.testing.assert_allclose(
                np.asarray(slab.values)[-1][rows], girder[:, k], **_RTOL,
                err_msg=f"{comp} of the girder")
        gf = np.asarray(opspy.eleResponse(3, "globalForce"))
        for k, comp in enumerate(nodal):
            slab = res.elements.get(component=comp, ids=beam)
            assert [int(i) for i in np.asarray(slab.element_ids)] == [3]
            np.testing.assert_allclose(
                np.asarray(slab.values)[-1][0], gf[[k, k + 3]], **_RTOL,
                err_msg=comp)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Refusals, by id and by name
# ---------------------------------------------------------------------------


def _refuses(spec: ResolvedRecorderSpec, tmp: Path, model_h5: Path, ops: Any,
             match: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError, match=match):
            with spec.emit_default(tmp / "run", model_h5=model_h5,
                                   ops=NoMPCO(ops)):
                pass


def test_a_deck_whose_connectivity_is_not_the_fem_refuses_by_id(
    opspy, tmp_path,
) -> None:
    """The girder built 3-2 instead of the FEM's 2-3: its end forces would
    read swapped."""
    fem = _portal_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_portal(opspy, fem, girder=(3, 2))
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("elements", "beam", ("nodal_resisting_force_y",),
                         np.array([1, 3]), None),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"FEM element 3 has nodes \[2, 3\], the live element 3 has "
             r"\[3, 2\]")


def _quad_fem(ids: "list[int]") -> FEMData:
    """Quads ``ids``, all on nodes 1-4 (a stacked pair when two)."""
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0],
                       [0.0, 1.0, 0.0]])
    quad = make_type_info(code=3, gmsh_name="Quadrilateral 4", dim=2,
                          order=1, npe=4, count=len(ids))
    group = ElementGroup(element_type=quad, ids=np.array(ids, dtype=np.int64),
                         connectivity=np.array([[1, 2, 3, 4]] * len(ids),
                                               dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 5, dtype=np.int64),
                            node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={3: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=4, n_elems=len(ids), bandwidth=4, types=[quad]),
        model_name=f"femdata_elements_quads_{len(ids)}",
    )


def _build_quads(ops: Any, mats: "dict[int, int]") -> None:
    """Quads tagged by ``mats`` keys on nodes 1-4, each with its material:
    a soft (1) and a stiff (2) one."""
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for n, xy in zip((1, 2, 3, 4), ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
                                    (0.0, 1.0))):
        ops.node(n, *xy)
    ops.fix(1, 1, 1)
    ops.fix(4, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 2.0e5, 0.25)
    ops.nDMaterial("ElasticIsotropic", 2, 2.0e7, 0.25)
    for eid, mat in mats.items():
        ops.element("quad", eid, 1, 2, 3, 4, 1.0, "PlaneStress", mat)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 1.0e3, 0.0)
    ops.load(3, 1.0e3, 0.0)
    for cmd, args in (("constraints", ("Plain",)), ("numberer", ("RCM",)),
                      ("system", ("BandGeneral",)),
                      ("test", ("NormDispIncr", 1.0e-12, 10)),
                      ("algorithm", ("Linear",)),
                      ("integrator", ("LoadControl", 1.0 / _STEPS)),
                      ("analysis", ("Static",))):
        getattr(ops, cmd)(*args)


_QUAD_STRESS = ("stress_xx", "stress_yy", "stress_xy")


def test_stacked_quads_on_the_same_nodes_refuse_naming_both(opspy, tmp_path) -> None:
    """The review's swap deck: quads 5 and 6 on nodes [1, 2, 3, 4] with
    their materials swapped. eleNodes matches both ways, so the pairing
    would attach each one's stress to the other's id."""
    fem = _quad_fem([5, 6])
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_quads(opspy, {5: 2, 6: 1})
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "s", _QUAD_STRESS, np.array([5, 6]),
                         "FourNodeQuad"),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"FEM element 5 shares nodes \[1, 2, 3, 4\] with \[6\].*"
             r"eleNodes cannot tell those elements apart")


def test_a_single_quad_records_its_gauss_stress(opspy, tmp_path) -> None:
    fem = _quad_fem([5])
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_quads(opspy, {5: 2})
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "s", _QUAD_STRESS, np.array([5]),
                         "FourNodeQuad"),))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        truth = np.asarray(opspy.eleResponse(5, "stresses"))   # 4 GPs x 3
        assert np.abs(truth).max() > 0
        for k, comp in enumerate(_QUAD_STRESS):
            slab = res.elements.gauss.get(component=comp, ids=np.array([5]))
            assert [int(i) for i in np.asarray(slab.element_index)] == [5] * 4
            np.testing.assert_allclose(np.asarray(slab.values)[-1],
                                       truth[k::3], **_RTOL, err_msg=comp)
    finally:
        res.close()


def test_a_zero_length_nodal_force_record_refuses_before_the_run(
    opspy, tmp_path,
) -> None:
    """A ZeroLength has no global_force nodal-force layout: refused at
    entry, not when the run is read."""
    fem = _portal_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    opspy.model("basic", "-ndm", 2, "-ndf", 3)
    opspy.node(2, 0.0, H)
    opspy.node(3, W, H)
    opspy.uniaxialMaterial("Elastic", 9, 1.0e3)
    opspy.element("zeroLength", 3, 2, 3, "-mat", 9, "-dir", 1)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("elements", "nf", ("nodal_resisting_force_x",),
                         np.array([3]), "ZeroLength"),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"no nodal-force layout \[\('ZeroLength', 'global_force'\)\]")


@pytest.mark.parametrize("cls", ["ElasticBeam2d", None],
                         ids=["named", "class-less"])
def test_an_elastic_girder_axial_force_refuses_by_name_before_the_run(
    opspy, tmp_path, cls,
) -> None:
    """Only the Truss family's Gauss ``axial_force`` leaves MPCO (#2290 D);
    the portal's elastic girder keeps the MPCO refusal at ``__enter__``,
    named or found by its live ``eleType``."""
    fem = _portal_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_portal(opspy, fem)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "axial", ("axial_force",), np.array([3]),
                         cls),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"component 'axial_force' \(the Truss axial force\) cannot go "
             r"through MPCO")


def test_an_unmapped_live_class_refuses_by_name(opspy, tmp_path) -> None:
    """``ModElasticBeam2d`` is an elastic beam the live-class table does not
    hold: its type token is unknown, so it refuses by its class name."""
    fem = _portal_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    opspy.model("basic", "-ndm", 2, "-ndf", 3)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        opspy.node(int(nid), float(xyz[0]), float(xyz[1]))
    opspy.geomTransf("Linear", 1)
    opspy.element("ModElasticBeam2d", 3, 2, 3, A, E, IZ, 1.0, 1.0, 1.0, 1)
    assert opspy.eleType(3) == "ModElasticBeam2d"
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("line_stations", "beam",
                         ("axial_force", "shear_y", "bending_moment_z"),
                         np.array([3]), "ElasticBeam2d"),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"'ModElasticBeam2d': FEM elements \[3\]")


# ---------------------------------------------------------------------------
# A determinate truss: the Gauss axial force MPCO cannot write (#2290 D)
# ---------------------------------------------------------------------------

PX, PY = 1.0e4, -2.0e4


class _StockView(NoMPCO):
    """The live module with no fork signal, its MPCO recorder kept where the
    build has one; each recorder type asked is logged in ``asked``."""

    def __init__(self, ops: Any) -> None:
        super().__init__(ops)
        self.asked: "list[str]" = []

    def recorder(self, *args: Any) -> Any:
        self.asked.append(str(args[0]))
        return self._ops.recorder(*args)


def test_truss_gauss_axial_force_skips_mpco_and_is_the_live_response(
    opspy, tmp_path,
) -> None:
    """Pinned at 1 (0, 0), a roller at 2 (4, 0), the apex 3 (2, 3) loaded
    (PX, PY). MPCO stays visible (``_StockView``, not NoMPCO):
    ``mpco_ops_args`` refuses the record on every stock build, so the route
    takes the classic recorders without probing MPCO."""
    coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    conn = {1: (1, 2), 2: (2, 3), 3: (1, 3)}
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=3)
    group = ElementGroup(element_type=line, ids=np.array([1, 2, 3], dtype=np.int64),
                         connectivity=np.array(list(conn.values()), dtype=np.int64))
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 4, dtype=np.int64),
                            node_coords=coords, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=3, n_elems=3, bandwidth=3, types=[line]),
        model_name="femdata_elements_truss")
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    opspy.model("basic", "-ndm", 2, "-ndf", 2)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        opspy.node(int(nid), float(xyz[0]), float(xyz[1]))
    opspy.fix(1, 1, 1)
    opspy.fix(2, 0, 1)
    opspy.uniaxialMaterial("Elastic", 1, E)
    for e, (i, j) in conn.items():
        opspy.element("Truss", e, i, j, A * e, 1)
    opspy.timeSeries("Linear", 1)
    opspy.pattern("Plain", 1, 1)
    opspy.load(3, PX, PY)
    for cmd, args in (("constraints", ("Plain",)), ("numberer", ("RCM",)),
                      ("system", ("BandGeneral",)),
                      ("test", ("NormDispIncr", 1.0e-12, 10)),
                      ("algorithm", ("Linear",)),
                      ("integrator", ("LoadControl", 1.0 / _STEPS)),
                      ("analysis", ("Static",))):
        getattr(opspy, cmd)(*args)
    ids = np.array([3, 1, 2], dtype=np.int64)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "axial", ("axial_force",), ids, "Truss"),))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        view = _StockView(opspy)
        with spec.emit_default(tmp_path / "run", model_h5=model_h5,
                               ops=view) as run:
            for _ in range(_STEPS):
                assert opspy.analyze(1) == 0
    assert run.route == "recorders" and "mpco" not in view.asked
    (w,) = caught
    assert w.category is DefaultRouteWarning
    assert ("MPCO cannot write the Truss axial force (records {'axial': "
            "['axial_force']}") in str(w.message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        slab = res.elements.gauss.get(component="axial_force", ids=ids)
        index = np.asarray(slab.element_index, dtype=np.int64)
        assert sorted(index.tolist()) == [1, 2, 3]       # one Gauss point each
        # Line_GL_1: the one point at mid-length, where a truss's constant
        # axial force is sampled.
        np.testing.assert_array_equal(np.asarray(slab.natural_coords),
                                      [[0.0]] * 3)
        order = np.argsort(index)
        got = np.asarray(slab.values)[:, order]
        truth = np.array([opspy.eleResponse(e, "axialForce")[0]
                          for e in (1, 2, 3)])
        np.testing.assert_allclose(got[-1], truth, **_RTOL)
        # Joint 3, members 2 (toward 2) and 3 (toward 1) in tension:
        # N2 - N3 = -PX sqrt(13) / 2, N2 + N3 = PY sqrt(13) / 3; joint 2
        # (free in x): N1 = -2 N2 / sqrt(13).
        r13 = np.sqrt(13.0)
        n2 = (-PX * r13 / 2 + PY * r13 / 3) / 2
        n3 = (PX * r13 / 2 + PY * r13 / 3) / 2
        np.testing.assert_allclose(got[-1], [-2 * n2 / r13, n2, n3], rtol=1e-9)
        np.testing.assert_allclose(got[0], got[-1] / _STEPS, rtol=1e-9)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# A Brick pulled off a fixed base: per-Gauss-point damage (#2290 A)
# ---------------------------------------------------------------------------

_HEX = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
_DAMAGE = ("damage_tension", "damage_compression")


def test_brick_per_material_damage_is_the_live_response(opspy, tmp_path) -> None:
    """The base face (nodes 1-4) is fixed and the top face is pulled up
    uniformly. The field is symmetric about the element's two vertical
    mid-planes, so a point's damage depends on its height alone. The base
    points are held against lateral contraction, so they carry the larger
    tensile stress and damage more. The catalog's natural coordinates must
    put that split on the third coordinate."""
    coords = np.array(_HEX, dtype=float)
    hexa = make_type_info(code=5, gmsh_name="Hexahedron 8", dim=3, order=1,
                          npe=8, count=1)
    group = ElementGroup(element_type=hexa, ids=np.array([7], dtype=np.int64),
                         connectivity=np.arange(1, 9, dtype=np.int64)[None, :])
    pg = PhysicalGroupSet({})
    fem = FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 9, dtype=np.int64),
                            node_coords=coords, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={5: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=8, n_elems=1, bandwidth=8, types=[hexa]),
        model_name="femdata_elements_brick")
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    opspy.model("basic", "-ndm", 3, "-ndf", 3)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        opspy.node(int(nid), *(float(v) for v in xyz))
    for n in (1, 2, 3, 4):
        opspy.fix(n, 1, 1, 1)
    opspy.nDMaterial("ASDConcrete3D", 1, 30.0e9, 0.2,
                     "-Te", 0.0, 1e-4, 2e-4, 1e-2, "-Ts", 0.0, 3e6, 1e6, 1e5,
                     "-Td", 0.0, 0.0, 0.5, 0.9,
                     "-Ce", 0.0, 1e-3, 2e-3, 1e-2, "-Cs", 0.0, 30e6, 20e6, 5e6,
                     "-Cd", 0.0, 0.0, 0.3, 0.8)
    opspy.element("stdBrick", 7, *range(1, 9), 1)
    assert opspy.eleType(7) == "Brick"
    opspy.timeSeries("Linear", 1)
    opspy.pattern("Plain", 1, 1)
    for n in (5, 6, 7, 8):
        opspy.sp(n, 3, 3.0e-4)
    for cmd, args in (("constraints", ("Transformation",)),
                      ("numberer", ("RCM",)), ("system", ("FullGeneral",)),
                      ("test", ("NormDispIncr", 1.0e-10, 50)),
                      ("algorithm", ("Newton",)),
                      ("integrator", ("LoadControl", 1.0 / _STEPS)),
                      ("analysis", ("Static",))):
        getattr(opspy, cmd)(*args)
    ids = np.array([7], dtype=np.int64)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("gauss", "dmg", _DAMAGE, ids, "Brick"),))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        truth = np.array([opspy.eleResponse(7, "material", gp, "damage")
                          for gp in range(1, 9)])          # (8 GP, 2)
        assert truth[:, 0].max() > 0.1                     # tension damage
        for k, comp in enumerate(_DAMAGE):
            slab = res.elements.gauss.get(component=comp, ids=ids)
            assert [int(i) for i in np.asarray(slab.element_index)] == [7] * 8
            np.testing.assert_allclose(np.asarray(slab.values)[-1],
                                       truth[:, k], **_RTOL, err_msg=comp)
        slab = res.elements.gauss.get(component="damage_tension", ids=ids)
        zeta = np.asarray(slab.natural_coords)[:, 2]
        d = np.asarray(slab.values)[-1]
        base, top = d[zeta < 0], d[zeta > 0]
        assert base.size == top.size == 4
        np.testing.assert_allclose(base, base[0], rtol=1e-9)
        np.testing.assert_allclose(top, top[0], rtol=1e-9)
        assert base[0] > top[0]
    finally:
        res.close()


# ---------------------------------------------------------------------------
# Fibres through the live section facts (#2290 B)
# ---------------------------------------------------------------------------

#: Two vertical cantilevers of height H, loaded at the tip by a lateral
#: PL and an axial -PN: a force beam (Lobatto 5, 4 fibres) on nodes 1-2
#: and a displacement beam (Legendre 3, 6 fibres) on nodes 3-4.
PL, PN = 2.0e3, 5.0e4
_FIBRE_STRESS = ("fiber_stress", "fiber_strain")
_LOBATTO_5 = (-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0)
_LEGENDRE_3 = (-np.sqrt(0.6), 0.0, np.sqrt(0.6))


def _cantilevers_fem() -> FEMData:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, H, 0.0], [W, 0.0, 0.0],
                       [W, H, 0.0]])
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    group = ElementGroup(
        element_type=line, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 5, dtype=np.int64),
                            node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=2, types=[line]),
        model_name="femdata_elements_fibre_cantilevers",
    )


def _build_cantilevers(ops: Any, fem: FEMData, *, elastic: bool = False) -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]))
    ops.fix(1, 1, 1, 1)
    ops.fix(3, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", 1, E)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 4, 1, -0.1, -0.05, 0.1, 0.05)
    ops.section("Fiber", 2)
    ops.patch("rect", 1, 6, 1, -0.15, -0.05, 0.15, 0.05)
    ops.section("Elastic", 3, E, A, IZ)
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 3 if elastic else 1, 5)
    ops.beamIntegration("Legendre", 2, 2, 3)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.element("dispBeamColumn", 2, 3, 4, 1, 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for tip in (2, 4):
        ops.load(tip, PL, -PN, 0.0)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / _STEPS)
    ops.analysis("Static")


def test_fibres_over_femdata_are_the_live_fiber_data(opspy, tmp_path) -> None:
    """Each recorded fibre's stress and strain is the live ``fiberData2``
    at rtol 1e-12, station by station. The order is checked physically:
    each station's section forces, summed from the recorded fibres, are
    the cantilever's closed form ``N = -PN``, ``|M| = PL (H - x)``, with
    ``x`` placed by the station's natural coordinate, so the base station
    carries the largest moment and the tip none."""
    fem = _cantilevers_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_cantilevers(opspy, fem)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(
            _record("fibers", "fib_force", _FIBRE_STRESS, np.array([1]),
                    "ForceBeamColumn2d"),
            _record("fibers", "fib_disp", _FIBRE_STRESS, np.array([2]),
                    "DispBeamColumn2d"),
        ))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        for e, n_fib, stations in ((1, 4, _LOBATTO_5), (2, 6, _LEGENDRE_3)):
            n_ip = len(stations)
            truth = np.array([opspy.eleResponse(e, "section", k, "fiberData2")
                              for k in range(1, n_ip + 1)]).reshape(n_ip, n_fib, 6)
            for c, comp in ((4, "fiber_stress"), (5, "fiber_strain")):
                slab = res.elements.fibers.get(component=comp, ids=np.array([e]))
                rows = _by_element(slab, e)
                assert len(rows) == n_ip * n_fib
                np.testing.assert_array_equal(
                    np.asarray(slab.gp_index)[rows],
                    np.repeat(np.arange(n_ip), n_fib))
                np.testing.assert_allclose(
                    np.asarray(slab.station_natural_coord)[rows],
                    np.repeat(stations, n_fib), rtol=0, atol=1e-12)
                np.testing.assert_allclose(
                    np.asarray(slab.y)[rows], truth[:, :, 0].ravel(), **_RTOL)
                np.testing.assert_allclose(
                    np.asarray(slab.values)[-1][rows], truth[:, :, c].ravel(),
                    **_RTOL, err_msg=f"{comp} of element {e}")
            slab = res.elements.fibers.get(component="fiber_stress",
                                           ids=np.array([e]))
            rows = _by_element(slab, e)
            sigma = np.asarray(slab.values)[-1][rows].reshape(n_ip, n_fib)
            y = np.asarray(slab.y)[rows].reshape(n_ip, n_fib)
            area = np.asarray(slab.area)[rows].reshape(n_ip, n_fib)
            xi = np.asarray(slab.station_natural_coord)[rows].reshape(
                n_ip, n_fib)[:, 0]
            x = 0.5 * (xi + 1.0) * H
            np.testing.assert_allclose((sigma * area).sum(axis=1), -PN,
                                       rtol=1e-9)
            moment = np.abs((sigma * area * y).sum(axis=1))
            np.testing.assert_allclose(moment, PL * (H - x), rtol=1e-8,
                                       atol=1e-8 * PL * H)
            assert np.argmax(moment) == 0
    finally:
        res.close()


def test_a_fibres_record_over_an_elastic_section_refuses_by_name(
    opspy, tmp_path,
) -> None:
    """A force beam whose sections are elastic answers no ``fiberData2``:
    its fibres could not be placed, so the live facts refuse it by FEM id
    at ``__enter__``, before the run."""
    fem = _cantilevers_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_cantilevers(opspy, fem, elastic=True)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("fibers", "fib", _FIBRE_STRESS, np.array([1]),
                         "ForceBeamColumn2d"),))
    _refuses(spec, tmp_path, model_h5, opspy,
             r"fibres record 'fib': FEM element 1 answers fiberData2 of 0 "
             r"values at section 1, not whole fibres of 6")


# ---------------------------------------------------------------------------
# Line stations sliced by the live section counts (#2328)
# ---------------------------------------------------------------------------

#: Two vertical displacement beams of elastic sections: element 1 on nodes
#: 1-2 (L = 1, Legendre 2), element 2 on nodes 3-4 (L = 13, Legendre 4).
#: Their 6 stations split 3 + 3 by an equal slice, and both blocks increase
#: within [-1, 1] once normalised by their own block's length, so before
#: #2328 the record read element 2's base station as element 1's third,
#: with no warning.
_MIXED_L = (1.0, 13.0)
_LEGENDRE = {2: np.polynomial.legendre.leggauss(2)[0],
             4: np.polynomial.legendre.leggauss(4)[0]}
_LS_COMPS = ("axial_force", "bending_moment_z")


def _mixed_fem() -> FEMData:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, _MIXED_L[0], 0.0],
                       [W, 0.0, 0.0], [W, _MIXED_L[1], 0.0]])
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=2)
    group = ElementGroup(
        element_type=line, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64))
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=np.arange(1, 5, dtype=np.int64),
                            node_coords=coords, physical=pg,
                            labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=2, types=[line]),
        model_name="femdata_elements_mixed_stations",
    )


def _build_mixed(ops: Any, fem: FEMData) -> None:
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        ops.node(int(nid), float(xyz[0]), float(xyz[1]))
    ops.fix(1, 1, 1, 1)
    ops.fix(3, 1, 1, 1)
    ops.geomTransf("Linear", 1)
    ops.section("Elastic", 1, E, A, IZ)
    ops.beamIntegration("Legendre", 1, 1, 2)
    ops.beamIntegration("Legendre", 2, 1, 4)
    ops.element("dispBeamColumn", 1, 1, 2, 1, 1)
    ops.element("dispBeamColumn", 2, 3, 4, 1, 2)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for tip in (2, 4):
        ops.load(tip, PL, -PN, 0.0)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / _STEPS)
    ops.analysis("Static")


def _stations_rec(name: str, ids: "list[int]") -> ResolvedRecorderRecord:
    return _record("line_stations", name, _LS_COMPS, np.array(ids),
                   "DispBeamColumn2d")


def test_mixed_line_stations_refuse_by_their_live_counts(opspy, tmp_path) -> None:
    """The stock reproducer of #2328: the record is refused by its live
    section counts, naming each element, before the columns are divided."""
    fem = _mixed_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_mixed(opspy, fem)
    spec = ResolvedRecorderSpec(fem_snapshot_id=fem.snapshot_id,
                                records=(_stations_rec("ls", [1, 2]),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with spec.emit_default(tmp_path / "run", model_h5=model_h5,
                               ops=NoMPCO(opspy)) as run:
            for _ in range(_STEPS):
                assert opspy.analyze(1) == 0
    with pytest.raises(ValueError,
                       match=r"different integration-point counts in the live "
                             r"domain \(2 points: FEM elements \[1\]; 4 points: "
                             r"FEM elements \[2\]\)"):
        run.results(cache_root=tmp_path / "cache")


def test_mixed_line_stations_split_by_count_are_the_live_section_force(
    opspy, tmp_path,
) -> None:
    """Split by count, each record reads its element's own stations: the
    Legendre points, and each station's ``(P, Mz)`` equal to the live
    ``section force`` at rtol 1e-12. The base station carries the largest
    moment ``PL (L - x)`` (rtol 1e-9), which checks the order physically."""
    fem = _mixed_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_mixed(opspy, fem)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_stations_rec("short", [1]), _stations_rec("long", [2])))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        ls = res.elements.line_stations
        for e, n_ip, length in ((1, 2, _MIXED_L[0]), (2, 4, _MIXED_L[1])):
            ids = np.array([e])
            for k, comp in enumerate(_LS_COMPS):
                slab = ls.get(component=comp, ids=ids)
                rows = _by_element(slab, e)
                assert len(rows) == n_ip
                xi = np.asarray(slab.station_natural_coord)[rows]
                np.testing.assert_allclose(xi, _LEGENDRE[n_ip], rtol=0,
                                           atol=1e-12)
                truth = [opspy.eleResponse(e, "section", ip, "force")[k]
                         for ip in range(1, n_ip + 1)]
                vals = np.asarray(slab.values)[-1][rows]
                np.testing.assert_allclose(vals, truth, **_RTOL,
                                           err_msg=f"{comp} of element {e}")
                if comp == "bending_moment_z":
                    x = 0.5 * (xi + 1.0) * length
                    np.testing.assert_allclose(np.abs(vals), PL * (length - x),
                                               rtol=1e-9)
                    assert np.argmax(np.abs(vals)) == 0
    finally:
        res.close()
