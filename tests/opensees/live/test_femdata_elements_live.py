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
