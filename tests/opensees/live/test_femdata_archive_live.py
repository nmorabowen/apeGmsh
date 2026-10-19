"""#2251 live oracles: a raw openseespy deck plus a FEMData archive records
and reads through ``spec.emit_default`` on the classic-recorder route.

The FEMData archive (``fem.to_h5``) declares no ndm and archives no
per-node ndf or element classes; the route reads them from the live
domain (``getNDM``, ``getNDF``, ``eleType``). The build is wrapped in
:class:`~tests.test_results_live_default_format.NoMPCO`, so the same cases
run the classic-recorder route on pip openseespy (where it is the module
itself) and on a build that has MPCO.

- The ``column_to_surface`` case-study deck: a tet column (ndf 3) between
  two 6-DOF reference nodes tied by rigid links, with a zero-length base
  spring the FEMData does not hold. Every node's displacement and
  reaction, and the reference nodes' rotations, equal the live
  ``nodeDisp`` / ``nodeReaction`` at rtol 1e-12, with exactly one
  :class:`DefaultRouteWarning` (the route's own) and no other.
- A mixed ndf 3/6 deck: a tet beside a 3-D cantilever. The tip reads
  ``PL^3/3EI`` and ``PL^2/2EI``, one displacement record serves both ndf,
  and a rotation over a tet node refuses by name.
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
from tests.test_results_live_default_format import NoMPCO

pytestmark = pytest.mark.live

_AXES = ("x", "y", "z")
_STEPS = 10


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


def _record(name: str, comps: "tuple[str, ...]", ids: "np.ndarray") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(category="nodes", name=name, components=comps,
                                  dt=None, n_steps=None, node_ids=ids)


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


def _last(res: Any, ids: "np.ndarray", comps: "tuple[str, ...]") -> "np.ndarray":
    slabs = [res.nodes.get(ids=ids, component=c) for c in comps]
    assert [int(n) for n in slabs[0].node_ids] == [int(n) for n in ids]
    return np.column_stack([np.asarray(s.values)[-1] for s in slabs])


# ---------------------------------------------------------------------------
# The column_to_surface case-study deck
# ---------------------------------------------------------------------------


def _column_fem() -> FEMData:
    from apeGmsh import apeGmsh

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the unit-system advisory
        m = apeGmsh(model_name="beam_column", verbose=False)
    m.begin()
    try:
        column = m.sections.W_solid(bf=150, tf=20, h=300, tw=10, length=2000,
                                    label="column")
        m.model.select(None, dim=3).to_physical(name="pg_column")
        m.model.geometry.add_point(x=0, y=0, z=0, lc=100, label="base")
        m.model.geometry.add_point(x=0, y=0, z=2000, lc=100, label="top")
        m.constraints.node_to_surface("base", column.labels.start_face)
        m.constraints.node_to_surface("top", column.labels.end_face)
        m.loads.point.force(target="top", force=[0, 1000, 0])
        m.mesh.sizing.set_global_size(100)
        m.mesh.generation.generate(dim=3)
        return m.mesh.queries.get_fem_data(remove_orphans=True)
    finally:
        m.end()


def _build_column(ops: Any, fem: FEMData) -> "np.ndarray":
    """The case study's deck (cell 4), tagged by FEM id; returns the
    reference nodes."""
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.timeSeries("Linear", 1)
    for nid, xyz in fem.nodes.select(target="pg_column"):
        ops.node(nid, *xyz, "-ndf", 3)
    ref = fem.nodes.select(target=["base", "top"])
    for nid, xyz in ref:
        ops.node(nid, *xyz)
    for nid, xyz in fem.nodes.constraints.phantom_nodes():
        ops.node(nid, *xyz)
    base_id = int(next(iter(fem.nodes.select(target="base").ids)))
    base_xyz = next(xyz for nid, xyz in ref if nid == base_id)
    ops.node(10_000, *base_xyz)
    ops.fix(10_000, 1, 1, 1, 1, 1, 1)
    for i in range(6):
        ops.uniaxialMaterial("Elastic", 100 + i, 1e14)
    ops.element("zeroLength", 20_000, 10_000, base_id, "-mat",
                100, 101, 102, 103, 104, 105, "-dir", 1, 2, 3, 4, 5, 6)
    ops.nDMaterial("ElasticIsotropic", 1, 200e3, 0.3)
    for group in fem.elements.select(element_type="tet4").groups():
        for eid, conn in group:
            ops.element("FourNodeTetrahedron", int(eid),
                        *[int(n) for n in conn], 1)
    for master, slaves in fem.nodes.constraints.rigid_link_groups():
        for slave in slaves:
            ops.rigidLink("beam", int(master), int(slave))
    for pair in fem.nodes.constraints.equal_dofs():
        ops.equalDOF(int(pair.master_node), int(pair.slave_node), *pair.dofs)
    ops.pattern("Plain", 1, 1)
    for load in fem.nodes.loads.by_kind("nodal"):
        fx, fy, fz = load.force_xyz or (0.0, 0.0, 0.0)
        mx, my, mz = load.moment_xyz or (0.0, 0.0, 0.0)
        ops.load(int(load.node_id), fx, fy, fz, mx, my, mz)
    ops.constraints("Penalty", 1e15, 1e15)
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormDispIncr", 1.0e-6, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    return np.asarray(sorted(int(n) for n, _ in ref), dtype=np.int64)


class _Rounded:
    """The live module, whose ``node`` command rounds each coordinate as a
    hand-written deck does."""

    def __init__(self, ops: Any, rnd: Any) -> None:
        self._ops, self._rnd = ops, rnd

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ops, name)

    def node(self, tag: int, *args: Any) -> Any:
        return self._ops.node(tag, *[self._rnd(float(v)) for v in args[:3]],
                              *args[3:])


_ROUNDINGS = {
    "exact": lambda v: v,
    "6-digit": lambda v: float(f"{v:.6g}"),
    "float32": lambda v: float(np.float32(v)),
    "mm-3-decimals": lambda v: round(v, 3),
}


@pytest.mark.parametrize("rounding", list(_ROUNDINGS))
def test_column_to_surface_deck_reads_the_live_domain(
    opspy, tmp_path, rounding,
) -> None:
    fem = _column_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    ref = _build_column(_Rounded(opspy, _ROUNDINGS[rounding]), fem)
    built = {int(n) for n in opspy.getNodeTags()}
    ids = np.asarray([int(n) for n in fem.nodes.ids if int(n) in built],
                     dtype=np.int64)
    disp = tuple(f"displacement_{a}" for a in _AXES)
    react = tuple(f"reaction_force_{a}" for a in _AXES)
    rot = tuple(f"rotation_{a}" for a in _AXES)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("disp", disp, ids), _record("react", react, ids),
                 _record("rot", rot, ref)))
    res = _emit(spec, tmp_path, model_h5, opspy, _STEPS)
    try:
        opspy.reactions()
        live_u = np.array([opspy.nodeDisp(int(n))[:3] for n in ids])
        live_r = np.array([opspy.nodeReaction(int(n))[:3] for n in ids])
        live_rot = np.array([opspy.nodeDisp(int(n))[3:6] for n in ref])
        assert np.abs(live_u).max() > 0 and np.abs(live_r).max() > 0
        np.testing.assert_allclose(_last(res, ids, disp), live_u, rtol=1e-12, atol=0)
        np.testing.assert_allclose(_last(res, ids, react), live_r, rtol=1e-12, atol=0)
        np.testing.assert_allclose(_last(res, ref, rot), live_rot, rtol=1e-12, atol=0)
    finally:
        res.close()


# ---------------------------------------------------------------------------
# A mixed ndf 3/6 deck, against the cantilever closed form
# ---------------------------------------------------------------------------

E, L, IZ, P = 2.0e11, 2.0, 8.0e-6, 1.0e3
UY, RZ = P * L ** 3 / (3.0 * E * IZ), P * L ** 2 / (2.0 * E * IZ)


def _mixed_fem() -> FEMData:
    """Tet nodes 1-4 (element 1) beside the cantilever's nodes 5-6
    (element 2)."""
    node_ids = np.arange(1, 7, dtype=np.int64)
    coords = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0],
                       [0.0, 0.0, 6.0], [0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    tet = make_type_info(code=4, gmsh_name="Tetrahedron 4", dim=3, order=1,
                         npe=4, count=1)
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    groups = {
        4: ElementGroup(element_type=tet, ids=np.array([1], dtype=np.int64),
                        connectivity=np.array([[1, 2, 3, 4]], dtype=np.int64)),
        1: ElementGroup(element_type=line, ids=np.array([2], dtype=np.int64),
                        connectivity=np.array([[5, 6]], dtype=np.int64)),
    }
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups=groups, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=6, n_elems=2, bandwidth=4, types=[tet, line]),
        model_name="femdata_archive_mixed",
    )


def _build_mixed(ops: Any, fem: FEMData) -> None:
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for nid, xyz in zip(fem.nodes.ids, fem.nodes.coords):
        extra = ("-ndf", 3) if int(nid) <= 4 else ()
        ops.node(int(nid), *[float(c) for c in xyz], *extra)
    for n in (1, 2, 3):
        ops.fix(n, 1, 1, 1)
    ops.fix(5, 1, 1, 1, 1, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 2.0e10, 0.25)
    ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)
    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.element("elasticBeamColumn", 2, 5, 6, 1.0e-2, E, 8.0e10, 1.0e-5,
                1.0e-5, IZ, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(6, 0.0, P, 0.0, 0.0, 0.0, 0.0)
    ops.load(4, 1.0e3, 0.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def test_a_mixed_ndf_deck_reads_each_node_at_its_own_dofs(opspy, tmp_path) -> None:
    fem = _mixed_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_mixed(opspy, fem)
    ids, beam = np.arange(1, 7, dtype=np.int64), np.array([5, 6], dtype=np.int64)
    disp = tuple(f"displacement_{a}" for a in _AXES)
    rot = tuple(f"rotation_{a}" for a in _AXES)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("disp", disp, ids), _record("rot", rot, beam)))
    res = _emit(spec, tmp_path, model_h5, opspy, 1)
    try:
        u = _last(res, ids, disp)
        np.testing.assert_allclose(
            u, [opspy.nodeDisp(int(n))[:3] for n in ids], rtol=1e-12, atol=0)
        np.testing.assert_allclose(u[5], [0.0, UY, 0.0], rtol=1e-9, atol=1e-15)
        r = _last(res, beam, rot)
        np.testing.assert_allclose(r[1], [0.0, 0.0, RZ], rtol=1e-9, atol=1e-15)
        assert abs(u[3, 0]) > 0              # the tet node moved too
    finally:
        res.close()


def _bar_fem(tip: "tuple[float, float, float]") -> FEMData:
    """Nodes 1 at the origin and 2 at ``tip``, one line element."""
    line = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    pg = PhysicalGroupSet({})
    group = ElementGroup(element_type=line, ids=np.array([7], dtype=np.int64),
                         connectivity=np.array([[1, 2]], dtype=np.int64))
    return FEMData(
        nodes=NodeComposite(node_ids=np.array([1, 2], dtype=np.int64),
                            node_coords=np.array([[0.0, 0.0, 0.0], tip]),
                            physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={1: group}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=2, types=[line]),
        model_name="femdata_archive_bar",
    )


@pytest.mark.parametrize("fem_tip, deck_tip, match", [
    # A FEM column along global Z, its planar deck mapping (x, z) to (x, y):
    # the deck's displacement_y is the FEM's z motion.
    ((0.0, 0.0, L), (0.0, L),
     r"node 2: live \[0\.0, 2\.0\], FEM \[0\.0, 0\.0\]"),
    # A deck built from another FEMData that shares node tags 1 and 2.
    ((L, 0.0, 0.0), (2.0 * L, 0.0),
     r"node 2: live \[4\.0, 0\.0\], FEM \[2\.0, 0\.0\]"),
], ids=["axis-swap", "foreign-femdata"])
def test_a_deck_node_that_is_not_the_fem_node_refuses_by_id(
    opspy, tmp_path, fem_tip, deck_tip, match,
) -> None:
    fem = _bar_fem(fem_tip)
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    opspy.model("basic", "-ndm", 2, "-ndf", 3)
    opspy.node(1, 0.0, 0.0)
    opspy.node(2, *deck_tip)
    opspy.geomTransf("Linear", 1)
    opspy.element("elasticBeamColumn", 7, 1, 2, 1.0e-2, E, IZ, 1)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("tip", ("displacement_x", "displacement_y"),
                         np.array([1, 2])),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError, match=match):
            with spec.emit_default(tmp_path / "run", model_h5=model_h5,
                                   ops=NoMPCO(opspy)):
                pass


def test_a_rotation_at_a_tet_node_refuses_by_name(opspy, tmp_path) -> None:
    fem = _mixed_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_mixed(opspy, fem)
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(_record("rot", ("rotation_z",), np.array([4, 6])),))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError, match=r"nodes \[4\] carry no 'rz' DOF "
                                             r"in the live domain"):
            with spec.emit_default(tmp_path / "run", model_h5=model_h5,
                                   ops=NoMPCO(opspy)):
                pass
