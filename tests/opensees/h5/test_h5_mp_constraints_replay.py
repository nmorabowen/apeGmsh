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
_MP_TOKENS = frozenset({"equalDOF", "rigidLink", "rigidDiaphragm",
                        "equationConstraint"})
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


def _bridge(fem: FEMData) -> apeSees:
    """The elastic column pair on ``fem``, fixed at ``"Base"``."""
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1,) * 6)
    return ops


def _archive(fem: FEMData, tmp_path: Path) -> "tuple[str, Path]":
    """The bridge's flat Tcl deck and the ``model.h5`` of the same model."""
    with warnings.catch_warnings():
        # The auto-emitted handler and partition chain advisories.
        warnings.simplefilter("ignore")
        ops = _bridge(fem)
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
    # #2416: the equation route writes one equationConstraint line per tied
    # DOF and no element. Three weighted masters, so each line carries three
    # retained terms; the tie is named, so its comment line rides along.
    "tie_equation": {"surface_constraints": [InterpolationRecord(
        kind="tie", name="tops_eq", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.25, 0.5, 0.25]), dofs=[1, 2, 3],
        enforce="equation")]},
    "distributing_equation": {"surface_constraints": [InterpolationRecord(
        kind="distributing", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.25, 0.5, 0.25]), dofs=[1, 2],
        enforce="equation")]},
    "embedded_equation": {"surface_constraints": [InterpolationRecord(
        kind="embedded", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.25, 0.5, 0.25]), dofs=[1, 2, 3],
        enforce="equation")]},
}


def _streams(kinds: str) -> "dict[str, list[Any]]":
    """The :data:`_KINDS` streams of the comma-separated ``kinds``."""
    out: "dict[str, list[Any]]" = {}
    for k in kinds.split(","):
        for stream, recs in _KINDS[k].items():
            out.setdefault(stream, []).extend(recs)
    return out


def _mp_bridge(kinds: str = "equal_dof", partitioned: bool = False) -> apeSees:
    """The bridge on :func:`_fem` carrying the comma-separated ``kinds``:
    the R1b round-trip oracle's MP-constraint corpus rows (#2405)."""
    return _bridge(_fem(partitioned=partitioned, **_streams(kinds)))


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
    streams = _streams("equal_dof,rigid_beam,rigid_diaphragm,embedded")
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


def test_to_h5_rewrite_keeps_the_rows(tmp_path):
    # The rewrite writes the global rows back, so its own replay is tied.
    fwd, path = _archive(_fem(partitioned=False, **_KINDS["rigid_body"]),
                         tmp_path)
    rewrite = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OpenSeesModel.from_h5(str(path)).to_h5(str(rewrite))
    with h5py.File(path, "r") as a, h5py.File(rewrite, "r") as b:
        assert (a["opensees/constraints/rigidLink"][()].tolist()
                == b["opensees/constraints/rigidLink"][()].tolist())
    assert _mp_lines(_replay(rewrite)) == _mp_lines(fwd)


# ---------------------------------------------------------------------------
# Streams the archive cannot replay refuse, by name: phantom-node ties
# (#2411) and equation-route ties (#2416)
# ---------------------------------------------------------------------------


def _node_to_surface(kind: str) -> Any:
    from apeGmsh._kernel.records._constraints import NodeToSurfaceRecord

    return NodeToSurfaceRecord(
        kind=kind, master_node=2, slave_nodes=[4], phantom_nodes=[200],
        phantom_coords=np.array([[1.0, 0.0, 1.0]]),
        rigid_link_records=[NodePairRecord(
            kind="rigid_beam", master_node=2, slave_node=200)],
        equal_dof_records=[NodePairRecord(
            kind="equal_dof", master_node=200, slave_node=4, dofs=[1, 2, 3])],
        dofs=[1, 2, 3])


def _mixed_ndf_interface_archive(tmp_path: Path) -> "tuple[str, Path]":
    """A 2D model whose interface joins a 2-DOF truss node (4) to the
    coincident 3-DOF beam node (2) through phantom node 200, the only
    place a mixed-ndf interface mints a phantom (ADR 0093 D4)."""
    from apeGmsh._kernel.records._constraints import (
        InterfaceRecord, NormalLaw, TangentialLaw,
    )

    ids = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    line = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)

    def pg(name: str, nodes: "list[int]", elements: "list[int]") -> dict:
        return {"name": name, "node_ids": ids(nodes),
                "node_coords": xyz[[n - 1 for n in nodes]],
                "element_ids": ids(elements)}

    groups = PhysicalGroupSet({
        (1, 100): pg("Beam", [1, 2], [1]), (1, 101): pg("Bar", [3, 4], [2]),
        (0, 102): pg("Base", [1], []), (0, 103): pg("BarBase", [3], []),
    })
    iface = InterfaceRecord(
        kind="interface", master_node=4, slave_node=2, backing_element=2,
        orient=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0), a_trib=0.5,
        normal_law=NormalLaw(kind="ent", k_per_area=1.0e6),
        tangential_law=TangentialLaw(kind="elastic", k_per_area=1.0e6),
        phantom_node=200, phantom_coords=np.array([0.0, 1.0, 0.0]),
        phantom_ndf=2, equal_dof_records=[NodePairRecord(
            kind="equal_dof", master_node=2, slave_node=200, dofs=[1, 2])])
    fem = FEMData(
        nodes=NodeComposite(node_ids=ids([1, 2, 3, 4]), node_coords=xyz,
                            physical=groups, labels=LabelSet({})),
        elements=ElementComposite(
            groups={1: ElementGroup(element_type=line, ids=ids([1, 2]),
                                    connectivity=ids([[1, 2], [3, 4]]))},
            physical=groups, labels=LabelSet({}), interfaces=[iface]),
        info=MeshInfo(n_nodes=4, n_elems=2, bandwidth=1, types=[line]),
        model_name="mixed_ndf_interface")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = apeSees(fem)
        ops.model(ndm=2, ndf=3)
        ops.element.elasticBeamColumn(
            pg="Beam", transf=ops.geomTransf.Linear(), A=0.01, E=200e9,
            Iz=1e-4)
        ops.element.Truss(pg="Bar", A=0.01,
                          material=ops.uniaxialMaterial.ElasticMaterial(E=200e9))
        ops.fix(pg="Base", dofs=(1, 1, 1))
        ops.fix(pg="BarBase", dofs=(1, 1))
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def _equation_tie(kind: str) -> Any:
    return InterpolationRecord(
        kind=kind, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.0, 1.0, 0.0]), dofs=[1, 2, 3], enforce="equation")


#: ``(streams, the line the bridge writes, the name the refusal gives,
#: the issue it cites)`` for every stream the replay refuses. ``None``
#: streams: the case builds its own archive (the 2D mixed-ndf interface).
_REFUSED: "dict[str, tuple[dict[str, list[Any]] | None, str, str, str]]" = {
    "node_to_surface": (
        {"node_constraints": [_node_to_surface("node_to_surface")]},
        "equalDOF", "node_to_surface: 1", "#2411"),
    "node_to_surface_spring": (
        {"node_constraints": [_node_to_surface("node_to_surface_spring")]},
        "equalDOF", "node_to_surface_spring: 1", "#2411"),
    "mixed_ndf_interface": (
        None, "equalDOF", "interface (mixed-ndf): 1", "#2411"),
    **{f"{k}_equation": (
        {"surface_constraints": [_equation_tie(k)]},
        "equationConstraint", f"{k} (enforce='equation'): 1", "#2416")
       for k in ("tie", "distributing", "embedded")},
}


def _refused_archive(case: str, tmp_path: Path) -> Path:
    """The archive of a refused case, after checking the bridge deck
    really carries the lines the replay would leave out."""
    streams, token, _named, _issue = _REFUSED[case]
    fwd, path = (_mixed_ndf_interface_archive(tmp_path) if streams is None
                 else _archive(_fem(partitioned=False, **streams), tmp_path))
    assert any(ln.split()[:1] == [token] for ln in fwd.splitlines())
    return path


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_replay_refuses_by_name(case, tmp_path):
    from apeGmsh.opensees._internal.build.errors import BridgeError

    _streams_, _token, named, issue = _REFUSED[case]
    path = _refused_archive(case, tmp_path)
    for target in ("tcl", "py"):
        with pytest.raises(BridgeError) as caught:
            OpenSeesModel.from_h5(str(path)).build(target, flat=True)
        assert named in str(caught.value)
        assert issue in str(caught.value)


def test_refusal_does_not_rest_on_the_skipped_stream_census(
        tmp_path, monkeypatch):
    # Stubbing out the warning must not stub out the refusal.
    from apeGmsh.opensees._internal import compose
    from apeGmsh.opensees._internal.build.errors import BridgeError

    path = _refused_archive("mixed_ndf_interface", tmp_path)
    monkeypatch.setattr(compose, "_warn_skipped_replay_streams",
                        lambda *_a, **_k: None)
    with pytest.raises(BridgeError, match=r"#2411"):
        OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)


@pytest.mark.parametrize("case", ["mixed_ndf_interface", "tie_equation"])
def test_live_refusal_comes_before_the_emitter(case, tmp_path, monkeypatch):
    # LiveOpsEmitter(wipe=True) wipes the domain, so a refusal after it
    # would destroy a model the caller still holds.
    from apeGmsh.opensees._internal.build.errors import BridgeError
    from apeGmsh.opensees.emitter import live

    class _NoEmitter:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise AssertionError("the live emitter was made before refusing")

    monkeypatch.setattr(live, "LiveOpsEmitter", _NoEmitter)
    path = _refused_archive(case, tmp_path)
    with pytest.raises(BridgeError, match=_REFUSED[case][3]):
        OpenSeesModel.from_h5(str(path)).build("live")


@pytest.mark.live
def test_refused_live_build_leaves_the_domain_untouched(tmp_path):
    from apeGmsh.opensees._internal.build.errors import BridgeError
    from apeGmsh.opensees.emitter.live import get_ops

    path = _refused_archive("node_to_surface", tmp_path)
    ops = get_ops()
    ops.wipe()
    try:
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(901, 0.0, 0.0, 0.0)
        with pytest.raises(BridgeError, match=r"#2411"):
            OpenSeesModel.from_h5(str(path)).build("live")
        assert [int(t) for t in ops.getNodeTags()] == [901]
    finally:
        ops.wipe()


# ---------------------------------------------------------------------------
# The match is a multiset on the written host: one row excuses one record
# ---------------------------------------------------------------------------


def _staged_and_global_embedded(tmp_path: Path) -> "tuple[str, Path]":
    """Two ``embedded`` records on constrained node 4, hosts (1, 2, 3) and
    (3, 1, 2); a stage claims the second (``s.embedded(name="st")``)."""
    from tests.opensees.h5.test_h5_stages_writer import _chain

    glob = InterpolationRecord(
        kind="embedded", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.2, 0.5, 0.3]), dofs=[1, 2, 3])
    staged = InterpolationRecord(
        kind="embedded", name="st", slave_node=4, master_nodes=[3, 1, 2],
        weights=np.array([0.3, 0.2, 0.5]), dofs=[1, 2, 3])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = _bridge(_fem(partitioned=False,
                           surface_constraints=[glob, staged]))
        with ops.stage(name="s0") as st:
            st.embedded(name="st")
            st.analysis(**_chain(ops))
            st.run(n_increments=1, dt=1.0)
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def _embedded_lines(deck: str) -> "list[str]":
    return [ln for ln in deck.splitlines()
            if ln.split()[:2] == ["element", "ASDEmbeddedNodeElement"]]


def test_stage_row_does_not_excuse_a_missing_global_row(tmp_path):
    fwd, path = _staged_and_global_embedded(tmp_path)
    assert len(_embedded_lines(fwd)) == 2
    # With both rows, the replay writes both lines and warns of nothing.
    assert sorted(_embedded_lines(_replay(path))) == sorted(
        _embedded_lines(fwd))
    # Without the global row (an archive the rows never reached), the
    # stage's row on the same constrained node must not cover for it.
    with h5py.File(path, "r+") as f:
        del f["opensees/constraints/embeddedNode"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("always", ReplaySkippedStreamWarning)
        with pytest.warns(ReplaySkippedStreamWarning,
                          match=r"fem\.elements\.constraints \(embedded: 1\)"):
            deck = OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)
    assert len(_embedded_lines(deck)) == 1


def test_one_row_excuses_one_of_two_equal_records():
    # Two equal equalDOF records need two archived lines (a multiset).
    from collections import Counter

    from apeGmsh.opensees._internal.compose import _skipped_replay_streams

    rec = _KINDS["equal_dof"]["node_constraints"][0]
    fem = _fem(partitioned=False, node_constraints=[rec, rec])
    row = ("equal_dof", 2, 4, (1, 2, 3))
    assert _skipped_replay_streams(fem, stage_mp_keys=Counter({row: 2})) == {}
    assert _skipped_replay_streams(fem, stage_mp_keys=Counter({row: 1})) == {
        "fem.nodes.constraints": Counter({"equal_dof": 1})}
