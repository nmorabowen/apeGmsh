"""#2352 live oracles: the coincident-node check compares a built element's
``eleNodes`` in its live class's node order.

``LiveDomainFacts._check_shared_positions`` pins a node that shares its
position with another by the live elements touching it: each must have
``eleNodes`` equal to its FEM connectivity. A ``Twenty_Node_Brick`` holds
its nodes in upstream's serendipity order, not Gmsh's
(``_NODE_ORDER_FOR_LIVE_CLASS``, #2313), so the check once compared it in
Gmsh order and refused a correctly tagged brick on a coincident node.

The model is one Gmsh hex20 (FEM element 11) with a zero-length spring
(FEM element 12) from its top corner, FEM node 7, to FEM node 21 at the
same position. The deck builds the brick in its own node order (read off
the edges here, not from the bridge's table), since a brick built in
Gmsh order exits the process at ``setDomain`` (#2313).

Oracles, on the classic-recorder route (wrapped in
:class:`~tests.test_results_live_default_format.NoMPCO`, so a build with
MPCO runs it too):

* **reads**: every node's displacement equals the live ``nodeDisp`` at
  rtol 1e-12;
* **a swap**: the same deck with FEM nodes 7 and 21 swapped in both
  elements has the same coordinates, so only its elements tell it apart.
  It refuses, naming each node and element.

Each case first proves its precondition: nodes 7 and 21 share their
position, and the live brick's ``eleNodes`` is not its Gmsh connectivity.
"""
from __future__ import annotations

import re
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

_DISP = ("displacement_x", "displacement_y", "displacement_z")
E, NU, K = 200e9, 0.3, 1.0e9
BRICK, SPRING = 11, 12
CORNER, TWIN = 7, 21

_CUBE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                  [1.1, 1.0, 1.2], [0.0, 1.0, 1.0]])
#: Gmsh's hex20 edge order (slots 8-19).
_GMSH_EDGES = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6),
               (3, 7), (4, 5), (4, 7), (5, 6), (6, 7)]
#: Twenty_Node_Brick's edge order (upstream ``shap3dv``): lower ring,
#: upper ring, verticals.
_TNB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]
#: FEM node ``k + 1`` at row ``k``: the hex20 in Gmsh order, then the twin.
_XYZ = np.vstack([_CUBE, [0.5 * (_CUBE[a] + _CUBE[b]) for a, b in _GMSH_EDGES],
                  _CUBE[CORNER - 1]])
_HEX = list(range(1, 21))


def _brick_order() -> "list[int]":
    """The FEM node of each Twenty_Node_Brick slot, read off the edges."""
    mids = [_GMSH_EDGES.index(tuple(sorted(e))) + 9 for e in _TNB_EDGES]
    return list(range(1, 9)) + mids


@pytest.fixture
def opspy() -> Any:
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    ops.wipe()
    yield ops
    ops.wipe()


def _group(code: int, name: str, dim: int, order: int, ids: "list[int]",
           conn: "list[list[int]]") -> ElementGroup:
    info = make_type_info(code=code, gmsh_name=name, dim=dim, order=order,
                          npe=len(conn[0]), count=len(conn))
    return ElementGroup(element_type=info,
                        ids=np.asarray(ids, dtype=np.int64),
                        connectivity=np.asarray(conn, dtype=np.int64))


def _fem() -> FEMData:
    hex20 = _group(17, "Hexahedron 20", 3, 2, [BRICK], [_HEX])
    line = _group(1, "Line 2", 1, 1, [SPRING], [[CORNER, TWIN]])
    pg = PhysicalGroupSet({})
    return FEMData(
        nodes=NodeComposite(
            node_ids=np.arange(1, len(_XYZ) + 1, dtype=np.int64),
            node_coords=_XYZ, physical=pg, labels=LabelSet({})),
        elements=ElementComposite(groups={17: hex20, 1: line}, physical=pg,
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=len(_XYZ), n_elems=2, bandwidth=len(_XYZ),
                      types=[hex20.element_type, line.element_type]),
        model_name="tnb_coincident",
    )


def _build(ops: Any, swap: "dict[int, int]") -> None:
    """The brick in its own node order and the spring, tagged by FEM id;
    ``swap`` renames nodes in both elements. The base is fixed, the twin
    is pulled sideways and down through the spring."""
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, E, NU)
    ops.uniaxialMaterial("Elastic", 2, K)
    for k, xyz in enumerate(_XYZ, 1):
        ops.node(k, *(float(v) for v in xyz))
    for k in np.flatnonzero(_XYZ[:, 2] < 1e-9) + 1:
        ops.fix(int(k), 1, 1, 1)
    ops.element("20NodeBrick", BRICK,
                *(swap.get(n, n) for n in _brick_order()), 1, 0.0, 0.0, 0.0)
    ops.element("zeroLength", SPRING, swap.get(CORNER, CORNER),
                swap.get(TWIN, TWIN), "-mat", 2, 2, 2, "-dir", 1, 2, 3)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(TWIN, 2.0e5, -1.0e5, -1.0e6)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1.0e-12, 10)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")


def _precondition(ops: Any) -> None:
    np.testing.assert_array_equal(ops.nodeCoord(CORNER), ops.nodeCoord(TWIN))
    assert ops.eleType(BRICK) == "Twenty_Node_Brick"
    assert [int(n) for n in ops.eleNodes(BRICK)] != _HEX


def _spec(fem: FEMData) -> ResolvedRecorderSpec:
    return ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="nodes", name="u_all", components=_DISP, dt=None,
            n_steps=None, node_ids=fem.nodes.ids),))


def test_a_twenty_node_brick_on_a_coincident_node_reads_the_live_domain(
        opspy, tmp_path) -> None:
    fem = _fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    _build(opspy, swap={})
    _precondition(opspy)
    assert [int(n) for n in opspy.eleNodes(BRICK)] == _brick_order()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _spec(fem).emit_default(tmp_path / "run", model_h5=model_h5,
                                     ops=NoMPCO(opspy)) as run:
            assert opspy.analyze(1) == 0
    assert run.route == "recorders"
    assert [w.category for w in caught] == [DefaultRouteWarning], [
        str(w.message) for w in caught]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = run.results(cache_root=tmp_path / "cache")
    try:
        ids = np.arange(1, len(_XYZ) + 1, dtype=np.int64)
        got = []
        for comp in _DISP:
            slab = res.nodes.get(ids=ids, component=comp, time=-1)
            np.testing.assert_array_equal(np.asarray(slab.node_ids), ids)
            got.append(np.asarray(slab.values)[-1])
        live = np.array([opspy.nodeDisp(int(n)) for n in ids])
        assert np.abs(live).max() > 0
        assert not np.allclose(live[CORNER - 1], live[TWIN - 1])
        np.testing.assert_allclose(np.column_stack(got), live,
                                   rtol=1e-12, atol=0)
    finally:
        res.close()


def test_a_coincident_node_swap_on_a_twenty_node_brick_refuses_by_name(
        opspy, tmp_path) -> None:
    fem = _fem()
    model_h5 = tmp_path / "model.h5"
    fem.to_h5(str(model_h5))
    swap = {CORNER: TWIN, TWIN: CORNER}
    _build(opspy, swap=swap)
    _precondition(opspy)
    order = _brick_order()
    live_brick = [swap.get(n, n) for n in order]
    assert [int(n) for n in opspy.eleNodes(BRICK)] == live_brick
    brick = re.escape(
        f"node {CORNER}: FEM element {BRICK} has nodes {_HEX} "
        f"(Twenty_Node_Brick order {order}), the live element {BRICK} has "
        f"{live_brick}")
    spring = re.escape(
        f"node {TWIN}: FEM element {SPRING} has nodes {[CORNER, TWIN]}, the "
        f"live element {SPRING} has {[TWIN, CORNER]}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DefaultRouteWarning)
        with pytest.raises(ValueError,
                           match="not pinned by their elements: ") as info:
            with _spec(fem).emit_default(tmp_path / "run", model_h5=model_h5,
                                         ops=NoMPCO(opspy)):
                pass
    assert re.search(brick, str(info.value))
    assert re.search(spring, str(info.value))
