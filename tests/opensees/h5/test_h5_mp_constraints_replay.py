"""Deck replay re-emits every MP constraint the bridge emitted (#2405).

``OpenSeesModel.from_h5(path).build("tcl")`` replayed an ``ops.h5(path)``
archive without one ``equalDOF`` line, although the bridge's own deck of the
same model had them: the replayed model ran untied. The writer stores each
global MP line as a row of ``/opensees/constraints/{equalDOF, rigidLink,
rigidDiaphragm, embeddedNode}`` (schema 2.7.0); the reader read those rows
only inside stage blocks, so the replay had nothing to emit.

The oracle is the bridge deck of the same model: for every MP kind the
bridge fans a FEMData constraint out to, the replayed deck carries the same
MP lines, in the same order, on a flat archive and on a partitioned one
(whose ranks replicate a cross-rank constraint, captured once by ADR 0027
INV-1, and replayed once). An archive that lacks the rows replays with a
:class:`ReplaySkippedStreamWarning` that names the stream, never silently.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
)
from apeGmsh.mesh._element_types import ElementGroup, make_type_info
from apeGmsh.mesh._group_set import LabelSet, PhysicalGroupSet
from apeGmsh.mesh.FEMData import (
    ElementComposite,
    FEMData,
    MeshInfo,
    NodeComposite,
)
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.compose import ReplaySkippedStreamWarning


#: The MP tokens a deck line can start with; ``element`` lines are checked
#: by type (an ``embeddedNode`` goes out as an ASDEmbeddedNodeElement).
_MP_TOKENS = frozenset({"equalDOF", "rigidLink", "rigidDiaphragm"})
_MP_ELEMENTS = frozenset({"ASDEmbeddedNodeElement"})

#: Two ranks: column A (nodes 1-2, element 1) and column B (3-4, element 2).
#: Rank 1 also lists node 3, so the embedded host (1, 2, 3) has one owner.
_PARTITIONS: dict[int, dict[str, list[int]]] = {
    1: {"node_ids": [1, 2, 3], "element_ids": [1]},
    2: {"node_ids": [3, 4], "element_ids": [2]},
}


def _fem(*, partitioned: bool, node_constraints: "list[Any]" = (),
         surface_constraints: "list[Any]" = ()) -> FEMData:
    """Two columns at x = 0 and 1 rising along z (PG ``"Cols"``), fixed at
    ``"Base"`` (1, 3), with the given constraint streams. A partitioned
    model puts each column on its own rank, so a constraint between the two
    tops (2 and 4) crosses ranks."""
    ids = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    line = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    lines = ElementGroup(element_type=line, ids=ids([1, 2]),
                         connectivity=ids([[1, 2], [3, 4]]))

    def pg(name: str, nodes: "list[int]", elements: "list[int]") -> dict:
        return {"name": name, "node_ids": ids(nodes),
                "node_coords": node_coords[[n - 1 for n in nodes]],
                "element_ids": ids(elements)}

    groups = PhysicalGroupSet({
        (1, 100): pg("Cols", [1, 2, 3, 4], [1, 2]),
        (0, 101): pg("Base", [1, 3], []),
    })
    parts = ({pid: {k: ids(v) for k, v in p.items()}
              for pid, p in _PARTITIONS.items()} if partitioned else None)
    nodes = NodeComposite(
        node_ids=ids([1, 2, 3, 4]), node_coords=node_coords, physical=groups,
        labels=LabelSet({}), constraints=list(node_constraints),
        **({"partitions": parts} if parts else {}))
    elements = ElementComposite(
        groups={1: lines}, physical=groups, labels=LabelSet({}),
        constraints=list(surface_constraints),
        **({"partitions": parts} if parts else {}))
    return FEMData(
        nodes=nodes, elements=elements,
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=1, types=[line]),
        model_name="mp_replay")


def _archive(fem: FEMData, tmp_path: Path) -> "tuple[str, Path]":
    """The bridge's flat Tcl deck and the ``model.h5`` of the same model."""
    with warnings.catch_warnings():
        # The auto-emitted handler and partition chain advisories.
        warnings.simplefilter("ignore")
        ops = apeSees(fem)
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf,
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
        ops.fix(pg="Base", dofs=(1,) * 6)
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def _replay(path: Path, target: str = "tcl") -> str:
    """The replayed flat deck; a skipped-stream warning fails it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", ReplaySkippedStreamWarning)
        out = OpenSeesModel.from_h5(str(path)).build(target, flat=True)
    assert isinstance(out, str)
    return out


def _mp_lines(deck: str) -> "list[str]":
    """The MP-constraint lines of a Tcl deck, in deck order."""
    out = []
    for ln in deck.splitlines():
        tok = ln.split()
        if tok[:1] and tok[0] in _MP_TOKENS:
            out.append(" ".join(tok))
        elif tok[:1] == ["element"] and len(tok) > 1 and tok[1] in _MP_ELEMENTS:
            out.append(" ".join(tok))
    return out


#: One FEMData-side constraint per MP kind the bridge fans out to a deck
#: line, between the two column tops (nodes 2 and 4).
_KINDS: "dict[str, dict[str, list[Any]]]" = {
    "equal_dof": {"node_constraints": [NodePairRecord(
        kind="equal_dof", master_node=2, slave_node=4, dofs=[1, 2, 3])]},
    "rigid_beam": {"node_constraints": [NodePairRecord(
        kind="rigid_beam", master_node=2, slave_node=4)]},
    "rigid_rod": {"node_constraints": [NodePairRecord(
        kind="rigid_rod", master_node=2, slave_node=4)]},
    "rigid_diaphragm": {"node_constraints": [NodeGroupRecord(
        kind="rigid_diaphragm", master_node=2, slave_nodes=[4],
        plane_normal=np.array([0.0, 0.0, 1.0]))]},
    "rigid_body": {"node_constraints": [NodeGroupRecord(
        kind="rigid_body", master_node=2, slave_nodes=[4])]},
    "embedded": {"surface_constraints": [InterpolationRecord(
        kind="embedded", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.0, 1.0, 0.0]), dofs=[1, 2, 3])]},
    "named_equal_dof": {"node_constraints": [NodePairRecord(
        kind="equal_dof", name="tops", master_node=2, slave_node=4,
        dofs=[1, 2])]},
}


@pytest.mark.parametrize("partitioned", [False, True],
                         ids=["flat", "partitioned"])
@pytest.mark.parametrize("kind", sorted(_KINDS))
def test_replay_carries_every_bridge_mp_line(kind, partitioned, tmp_path):
    fwd, path = _archive(_fem(partitioned=partitioned, **_KINDS[kind]),
                         tmp_path)
    expected = _mp_lines(fwd)
    # The oracle has something to compare: the bridge tied the tops.
    assert expected, f"the bridge deck carries no MP line for {kind}"
    assert _mp_lines(_replay(path)) == expected


def test_named_constraint_keeps_its_comment(tmp_path):
    fwd, path = _archive(_fem(partitioned=False, **_KINDS["named_equal_dof"]),
                         tmp_path)
    replay = _replay(path)
    i = fwd.splitlines().index(next(
        ln for ln in fwd.splitlines() if ln.startswith("equalDOF")))
    j = replay.splitlines().index(next(
        ln for ln in replay.splitlines() if ln.startswith("equalDOF")))
    assert replay.splitlines()[j - 1] == fwd.splitlines()[i - 1]


@pytest.mark.parametrize("partitioned", [False, True],
                         ids=["flat", "partitioned"])
def test_replay_deck_equals_bridge_deck(partitioned, tmp_path):
    # Every kind at once: the replay is the bridge's deck, line for line.
    streams: "dict[str, list[Any]]" = {
        "node_constraints": [], "surface_constraints": []}
    for k in ("equal_dof", "rigid_beam", "rigid_diaphragm", "embedded"):
        for stream, recs in _KINDS[k].items():
            streams[stream] += recs
    fwd, path = _archive(_fem(partitioned=partitioned, **streams), tmp_path)
    assert _mp_lines(_replay(path)) == _mp_lines(fwd)
    assert len(_mp_lines(fwd)) == 4


def test_py_target_carries_the_mp_calls(tmp_path):
    _fwd, path = _archive(_fem(partitioned=False, **_KINDS["equal_dof"]),
                          tmp_path)
    py = _replay(path, "py")
    assert py.count("ops.equalDOF(") == 1


def test_archive_without_the_rows_warns_by_name(tmp_path):
    # A file whose /opensees/constraints rows are gone (an archive the
    # rows never reached) must not replay untied in silence.
    _fwd, path = _archive(_fem(partitioned=False, **_KINDS["equal_dof"]),
                          tmp_path)
    with h5py.File(path, "r+") as f:
        del f["opensees/constraints/equalDOF"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", ReplaySkippedStreamWarning)
        with pytest.warns(ReplaySkippedStreamWarning,
                          match=r"fem\.nodes\.constraints \(equal_dof: 1\)"):
            deck = OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)
    assert _mp_lines(deck) == []
