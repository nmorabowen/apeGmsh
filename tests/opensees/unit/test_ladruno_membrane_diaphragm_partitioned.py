"""``LadrunoMembrane(diaphragm_nodes=)`` past the emit call: stages, ranks, ndf.

The fixture is the two-panel grid of
``test_ladruno_membrane_frames_orphans``.  ``D`` is PanelA's interior mesh
node (2, 2): only the panel's own gmsh cells touch it, so it is in the
model only because ``diaphragm_nodes=`` names it.  Each test pins one
build site that must count it (#2549):

* stage ownership (``ownership.compute_stage_ownership``): a stage-claimed
  membrane's diaphragm node is emitted inside that stage, not in the
  global block where no element would hold it during the earlier stages;
* the partition rank (``apeSees`` partitioned emit, ``macro_element_rank``):
  the membrane goes to the rank that holds ``D`` as well as its loop;
* ndf inference counts no membrane node at all, the diaphragm nodes
  included: the class is ndf-adaptive, which is why ``infer_node_ndf``
  needs no named-node block.

It also holds the empty refusals: ``()`` and an empty ``NodeSet`` are
refused alike for ``support_nodes=`` and ``diaphragm_nodes=``.
"""
from __future__ import annotations

from typing import Any, cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.element.shell import LadrunoMembrane
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.node import Node, NodeSet
from tests.opensees.unit.test_ladruno_membrane_frames_orphans import (
    COL_BASE,
    INTERIOR,
    _chain,
    _fem,
    _nid,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::apeGmsh.opensees._internal.build.pg_expand.MembraneFramesFilteredWarning")

D = INTERIOR[0]                     # PanelA's interior node (2, 2)
STIFF = (30e6, 0.2, 0.2)
PANEL_A_LOOP_SUPPORTS = (_nid(0, 0), _nid(2, 0), _nid(2, 2), _nid(0, 2))


def _empty_set() -> NodeSet:
    return NodeSet(nodes=(), bridge=cast("Any", None))


def _calls(rec: RecordingEmitter) -> list[tuple[str, tuple[Any, ...]]]:
    return [(name, tuple(args)) for name, args, _kw in rec.calls]


def _is_node(call: tuple[str, tuple[Any, ...]], tag: int) -> bool:
    return call[0] == "node" and int(call[1][0]) == tag


def _is_membrane(call: tuple[str, tuple[Any, ...]]) -> bool:
    return call[0] == "element" and call[1][0] == "LadrunoMembrane"


# ---------------------------------------------------------------------------
# Empty refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["support_nodes", "diaphragm_nodes"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param((), id="empty-tuple"),
        pytest.param("empty-set", id="bare-empty-NodeSet"),
        pytest.param(("empty-set",), id="empty-NodeSet-in-a-tuple"),
        pytest.param(("empty-set", 5), id="empty-NodeSet-beside-a-tag"),
    ],
)
def test_empty_node_lists_are_refused_alike(field: str, value: Any) -> None:
    if value == "empty-set":
        value = _empty_set()
    elif isinstance(value, tuple):
        value = tuple(_empty_set() if v == "empty-set" else v for v in value)
    with pytest.raises(ValueError, match=rf"{field}= must not be empty"):
        LadrunoMembrane(nodes=(1, 2, 3), stiff=STIFF, **{field: value})


@pytest.mark.parametrize("field", ["support_nodes", "diaphragm_nodes"])
def test_a_non_empty_node_set_still_expands_to_its_tags(field: str) -> None:
    bridge = cast("Any", None)
    nodes = tuple(Node(tag=t, coords=(0.0, 0.0, 0.0), bridge=bridge) for t in (7, 9))
    m = LadrunoMembrane(
        nodes=(1, 2, 3), stiff=STIFF,
        **{field: (NodeSet(nodes=nodes, bridge=bridge), 4)})
    assert getattr(m, field) == (7, 9, 4)


# ---------------------------------------------------------------------------
# Stage ownership
# ---------------------------------------------------------------------------


def _staged_deck() -> list[tuple[str, tuple[Any, ...]]]:
    fem = _fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
    ops.element.elasticBeamColumn(
        pg="Beams", transf=t, A=0.1, E=30e6, Iz=1e-3, Iy=1e-3, G=12e6, J=1e-4)
    ops.element.LadrunoMembrane(
        pg="PanelA", frames="Beams", stiff=STIFF, diaphragm_nodes=(D,))
    ops.fix(pg="Corners", dofs=(1, 1, 1, 1, 1, 1))
    with ops.stage(name="slab") as s:
        s.activate(pgs=("PanelA",))
        s.analysis(**_chain(ops))
        s.run(n_increments=1, dt=1.0)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    return _calls(rec)


def test_a_staged_membranes_diaphragm_node_is_emitted_in_its_stage() -> None:
    calls = _staged_deck()
    opens = [i for i, c in enumerate(calls) if c[0] == "stage_open"]
    node_d = [i for i, c in enumerate(calls) if _is_node(c, D)]
    membrane = [i for i, c in enumerate(calls) if _is_membrane(c)]
    assert len(opens) == 1 and len(node_d) == 1 and len(membrane) == 1
    # Inside the stage, before the element that names it.  In the global
    # block it would sit in the model, held by nothing, before the stage.
    assert opens[0] < node_d[0] < membrane[0], (opens, node_d, membrane)


# ---------------------------------------------------------------------------
# Partition rank
# ---------------------------------------------------------------------------


def _rank_of_each_call(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> list[int | None]:
    rank: int | None = None
    out: list[int | None] = []
    for name, args in calls:
        if name == "partition_open":
            rank = int(args[0])
        out.append(rank)
        if name == "partition_close":
            rank = None
    return out


def test_the_membrane_goes_to_the_rank_that_holds_its_diaphragm_node() -> None:
    """PanelA's loop lies on both ranks; ``D`` lies on rank 1 only.

    The loop alone would put the membrane on rank 0 (the lowest common
    rank), where node ``D`` does not exist.  Counting ``D`` leaves rank 1.
    """
    fem = _fem()
    grid = sorted({_nid(ix, iy) for ix in range(5) for iy in range(3)})
    panel_a = sorted({_nid(ix, iy) for ix in range(3) for iy in range(3)})
    fem.set_partitions([
        (0, [n for n in grid if n != D] + [COL_BASE], [5, 6, 7, 8]),
        (1, panel_a, [1, 2, 3, 4]),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(
        pg="PanelA", support_nodes=PANEL_A_LOOP_SUPPORTS, stiff=STIFF,
        diaphragm_nodes=(D,))
    rec = RecordingEmitter()
    ops.build().emit(rec)
    calls = _calls(rec)
    ranks = _rank_of_each_call(calls)
    membrane_ranks = {ranks[i] for i, c in enumerate(calls) if _is_membrane(c)}
    node_d_ranks = {ranks[i] for i, c in enumerate(calls) if _is_node(c, D)}
    assert membrane_ranks == {1}
    assert node_d_ranks == {1}


# ---------------------------------------------------------------------------
# ndf inference
# ---------------------------------------------------------------------------


def test_ndf_inference_takes_no_opinion_from_a_membrane() -> None:
    """``infer_node_ndf`` drops every ndf-adaptive class, LadrunoMembrane
    among them, so naming a node in ``diaphragm_nodes=`` (or a hole, or
    ``support_nodes=``) cannot change its inferred ndf: it takes the
    other elements' ndf, or the ``ops.model`` envelope.  If this fails,
    the membrane gained an ndf floor, and ``infer_node_ndf`` must count
    ``pg_expand.ladruno_membrane_named_nodes`` as well as the loop.
    """
    from apeGmsh.opensees._element_capabilities import element_class_ndf_ok
    from apeGmsh.opensees._internal.build.ndf import (
        _ADAPTIVE_NDF_OK,
        infer_node_ndf,
    )
    from apeGmsh.opensees._internal.types import Element

    assert element_class_ndf_ok("LadrunoMembrane") == _ADAPTIVE_NDF_OK

    fem = _fem()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    ops.element.LadrunoMembrane(
        pg="PanelA", support_nodes=PANEL_A_LOOP_SUPPORTS, stiff=STIFF,
        diaphragm_nodes=(D,))
    elements = [p for p in ops._primitives if isinstance(p, Element)]
    assert infer_node_ndf(fem, elements, ndm=3) == {}
