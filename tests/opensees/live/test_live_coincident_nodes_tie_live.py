"""#2342 (gap b) live oracles: coincident nodes of a tie or contact deck
are pinned by the elements the deck built, not by every FEM element.

``get_fem_data()`` holds the point, edge and face elements of every
meshed entity beside the beams or solids a hand-built deck builds. The
node identity check (``LiveDomainFacts._check_shared_positions``, #2251)
once required every FEM element touching a coincident node to be live,
so the curriculum decks 12 (``12_interface_tie``) and 16
(``16_tied_contact_nonmatching``) were refused on pip with "FEM element
3 has nodes [3], no live element has that tag", although their live
elements pin those nodes. Each case is rebuilt from its notebook through
an apeGmsh session, so the FEMData is the real one.

Oracles, on the classic-recorder route (wrapped in
:class:`~tests.test_results_live_default_format.NoMPCO`, so a build with
MPCO runs it too):

* **notebook 12**: every node's displacement equals the live
  ``nodeDisp`` at rtol 1e-12, and the tip deflection is the cantilever's
  ``-P L^3 / 3 E I`` (the elastic beam is exact at its nodes);
* **notebook 16**: every node's displacement equals the live
  ``nodeDisp`` at rtol 1e-12;
* **a swap**: notebook 12's deck with the two junction tags swapped in
  its beam elements has the same coordinates, so only its elements tell
  it apart. It refuses, naming each node and element.

Each case first proves its precondition: a recorded node shares its
position with another, and a FEM element touching it is not live.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.results.live import DefaultRouteWarning
from apeGmsh.results.spec._resolved import (
    ResolvedRecorderRecord,
    ResolvedRecorderSpec,
)
from tests.test_results_live_default_format import NoMPCO

pytestmark = pytest.mark.live

_DISP = ("displacement_x", "displacement_y", "displacement_z")


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


def _spec(fem: FEMData) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="u_all", components=_DISP, dt=None,
            n_steps=None, node_ids=fem.nodes.ids),))


def _read(spec: ResolvedRecorderSpec, tmp: Path, model_h5: Path,
          ops: Any) -> Any:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with spec.emit_default(tmp / "run", model_h5=model_h5,
                               ops=NoMPCO(ops)) as run:
            assert ops.analyze(1) == 0
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return run.results(cache_root=tmp / "cache")


def _assert_equals_live(res: Any, fem: FEMData, ops: Any) -> None:
    ids = np.sort(np.asarray(fem.nodes.ids, dtype=np.int64))
    got = []
    for comp in _DISP:
        slab = res.nodes.get(ids=ids, component=comp, time=-1)
        np.testing.assert_array_equal(np.asarray(slab.node_ids), ids)
        got.append(np.asarray(slab.values)[-1])
    live = np.array([ops.nodeDisp(int(n))[:3] for n in ids])
    assert np.abs(live).max() > 0
    np.testing.assert_allclose(np.column_stack(got), live, rtol=1e-12, atol=0)


def _assert_unbuilt_touch_coincident(fem: FEMData, ops: Any) -> None:
    """The precondition: some coincident node is touched by a FEM element
    the deck did not build."""
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    _, inv, counts = np.unique(np.round(coords, 12), axis=0,
                               return_inverse=True, return_counts=True)
    coincident = set(ids[counts[inv.ravel()] > 1].tolist())
    live = {int(t) for t in ops.getEleTags()}
    unbuilt_touching = [
        int(e) for group in fem.elements for e, conn in group
        if int(e) not in live and coincident & {int(n) for n in conn}]
    assert coincident and unbuilt_touching


# ---------------------------------------------------------------------------
# Notebook 12: two beams tied by equalDOF at a junction
# ---------------------------------------------------------------------------

L_A, L_B, P = 1.5, 1.5, 10_000.0
E, NU = 2.1e11, 0.3
A, IY, IZ, J = 1.0e-3, 1.0e-5, 1.0e-5, 2.0e-5


def _tie_fem() -> FEMData:
    """Notebook 12's model: two collinear beams with distinct junction
    points, tied by ``equal_dof`` over all six DOFs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the unit-system advisory
        m = apeGmsh(model_name="coincident_tie_live", verbose=False)
    m.begin()
    g = m
    try:
        lc = (L_A + L_B) / 20.0
        geo = g.model.geometry
        pa0, pa1 = geo.add_point(0.0, 0.0, 0.0, lc=lc), geo.add_point(L_A, 0.0, 0.0, lc=lc)
        pb0 = geo.add_point(L_A, 0.0, 0.0, lc=lc)
        pb1 = geo.add_point(L_A + L_B, 0.0, 0.0, lc=lc)
        ln_a, ln_b = geo.add_line(pa0, pa1), geo.add_line(pb0, pb1)
        g.model.sync()
        g.parts.register("beamA", [(1, ln_a), (0, pa0), (0, pa1)])
        g.parts.register("beamB", [(1, ln_b), (0, pb0), (0, pb1)])
        g.physical.add(0, [pa0], name="base")
        g.physical.add(0, [pb1], name="tip")
        g.constraints.equal_dof(
            master_label="beamA", slave_label="beamB",
            master_entities=[(0, pa1)], slave_entities=[(0, pb0)],
            tolerance=1e-9, dofs=[1, 2, 3, 4, 5, 6])
        g.mesh.generation.generate(1)
        return g.mesh.queries.get_fem_data()
    finally:
        m.end()


def _build_tie(ops: Any, fem: FEMData, swap: "dict[int, int]") -> None:
    """Notebook 12's deck, tagged by FEM id; ``swap`` renames nodes in the
    beam elements only."""
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for nid, xyz in fem.nodes.select():
        ops.node(int(nid), *[float(c) for c in xyz])
    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)
    g_mod = E / (2.0 * (1.0 + NU))
    for group in fem.elements:
        if group.dim != 1:
            continue
        for eid, (ni, nj) in group:
            ops.element("elasticBeamColumn", int(eid),
                        swap.get(int(ni), int(ni)), swap.get(int(nj), int(nj)),
                        A, E, g_mod, J, IY, IZ, 1)
    for rec in fem.nodes.constraints.equal_dofs():
        ops.equalDOF(int(rec.master_node), int(rec.slave_node), *rec.dofs)
    for n in fem.nodes.select(target="base").ids:
        ops.fix(int(n), 1, 1, 1, 1, 1, 1)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for n in fem.nodes.select(target="tip").ids:
        ops.load(int(n), 0.0, 0.0, -P, 0.0, 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormUnbalance", 1e-10, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def test_notebook_12_interface_tie_reads_the_live_domain(opspy, tmp_path) -> None:
    fem = _tie_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_tie(opspy, fem, swap={})
    _assert_unbuilt_touch_coincident(fem, opspy)
    res = _read(_spec(fem), tmp_path, model_h5, opspy)
    try:
        _assert_equals_live(res, fem, opspy)
        tip = int(next(iter(fem.nodes.select(target="tip").ids)))
        u_tip = float(res.nodes.get(ids=[tip], component="displacement_z",
                                    time=-1).values[0, 0])
        np.testing.assert_allclose(
            u_tip, -P * (L_A + L_B) ** 3 / (3.0 * E * IZ), rtol=1e-9)
    finally:
        res.close()


def test_a_coincident_node_tag_swap_still_refuses_by_name(opspy, tmp_path) -> None:
    fem = _tie_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    (pair,) = fem.nodes.constraints.equal_dofs()
    m, s = int(pair.master_node), int(pair.slave_node)
    _build_tie(opspy, fem, swap={m: s, s: m})
    _assert_unbuilt_touch_coincident(fem, opspy)
    np.testing.assert_array_equal(opspy.nodeCoord(m), opspy.nodeCoord(s))
    beams = [(int(e), [int(n) for n in conn]) for group in fem.elements
             if group.dim == 1 for e, conn in group]
    (e_m, conn_m), = [(e, c) for e, c in beams if m in c]
    (e_s, conn_s), = [(e, c) for e, c in beams if s in c]

    def named(n: int, e: int, conn: "list[int]", other: int) -> str:
        live = [other if v == n else v for v in conn]
        return re.escape(f"node {n}: FEM element {e} has nodes {conn}, the "
                         f"live element {e} has {live}")

    with pytest.raises(ValueError, match="not pinned by their elements: ") as info:
        _read(_spec(fem), tmp_path, model_h5, opspy)
    assert re.search(named(m, e_m, conn_m, s), str(info.value))
    assert re.search(named(s, e_s, conn_s, m), str(info.value))


# ---------------------------------------------------------------------------
# Notebook 16: two tet blocks tied across a non-matching interface
# ---------------------------------------------------------------------------


def _contact_fem() -> FEMData:
    """Notebook 16's model, coarser: two boxes meshed at different sizes,
    block B's interface face tied into block A's."""
    import gmsh

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the unit-system advisory
        m = apeGmsh(model_name="coincident_contact_live", verbose=False)
    m.begin()
    g = m
    try:
        box_a = g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 0.3, 0.3)
        box_b = g.model.geometry.add_box(1.0, 0.0, 0.0, 1.0, 0.3, 0.3)
        g.model.sync()
        g.parts.register("blockA", [(3, box_a)])
        g.parts.register("blockB", [(3, box_b)])
        faces: "dict[str, int]" = {}
        for dim, tag in gmsh.model.getEntities(dim=2):
            x0, _, _, x1, _, _ = gmsh.model.getBoundingBox(dim, tag)
            if abs(x1 - x0) > 1e-6:
                continue
            if abs(x0) < 1e-6:
                faces["left"] = tag
            elif abs(x0 - 2.0) < 1e-6:
                faces["right"] = tag
            else:
                up, _ = gmsh.model.getAdjacencies(2, tag)
                faces["iface_A" if up[0] == box_a else "iface_B"] = tag
        for name, tag in faces.items():
            g.physical.add(2, [tag], name=name)
        g.mesh.sizing.set_size([(3, box_a)], 0.3)
        g.mesh.sizing.set_size([(3, box_b)], 0.15)
        g.constraints.tie(
            master_label="blockA", slave_label="blockB",
            master_entities=[(2, faces["iface_A"])],
            slave_entities=[(2, faces["iface_B"])],
            tolerance=0.01, dofs=[1, 2, 3])
        g.mesh.generation.generate(3)
        return g.mesh.queries.get_fem_data()
    finally:
        m.end()


def _build_contact(ops: Any, fem: FEMData) -> None:
    """Notebook 16's deck: tets by FEM id, one ``ASDEmbeddedNodeElement``
    per interpolation record (tags after the last tet), left face fixed,
    the right face pulled."""
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for nid, xyz in fem.nodes.select():
        ops.node(int(nid), *[float(c) for c in xyz])
    ops.nDMaterial("ElasticIsotropic", 1, E, NU)
    next_eid = 1
    for group in fem.elements.select(element_type="tet4").result():
        for eid, nodes in zip(group.ids, group.connectivity):
            ops.element("FourNodeTetrahedron", int(eid),
                        *[int(n) for n in nodes], 1)
            next_eid = max(next_eid, int(eid) + 1)
    for rec in fem.elements.constraints.interpolations():
        ops.element("ASDEmbeddedNodeElement", next_eid, int(rec.slave_node),
                    *[int(n) for n in rec.master_nodes])
        next_eid += 1
    for n in fem.nodes.select(target="left").ids:
        ops.fix(int(n), 1, 1, 1)
    right = list(fem.nodes.select(target="right").ids)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    for n in right:
        ops.load(int(n), 1.0e6 * 0.09 / len(right), 0.0, 0.0)
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-8, 20)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def test_notebook_16_tied_contact_reads_the_live_domain(opspy, tmp_path) -> None:
    fem = _contact_fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build_contact(opspy, fem)
    _assert_unbuilt_touch_coincident(fem, opspy)
    res = _read(_spec(fem), tmp_path, model_h5, opspy)
    try:
        _assert_equals_live(res, fem, opspy)
    finally:
        res.close()
