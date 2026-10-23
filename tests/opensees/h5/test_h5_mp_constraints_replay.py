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
# (#2411), and equation-route ties in an archive below opensees 2.35.0
# (#2416), which stores none of their lines
# ---------------------------------------------------------------------------


def _as_pre_2_35(path: Path) -> None:
    """Make the 2.35 archive ``path`` read as one below opensees 2.35.0, as
    its writer would have left it: no ``equationConstraint`` store, the
    program's ``equationConstraint`` calls linked to no store (a ledger
    verb then), and the zone stamped 2.34.0."""
    from tests.fixtures.schema import OPENSEES_BEFORE_EQUATION_CONSTRAINT

    store = "{scope}/constraints/equationConstraint"
    with h5py.File(path, "r+") as f:
        cons = f["opensees/constraints"]
        if "equationConstraint" in cons:
            del cons["equationConstraint"]
        if len(cons.keys()) == 0:
            del f["opensees/constraints"]
        program = f["opensees/program"]
        methods = [m.decode() if isinstance(m, bytes) else str(m)
                   for m in program.attrs["methods"]]
        stores = [m.decode() if isinstance(m, bytes) else str(m)
                  for m in program.attrs["stores"]]
        if store in stores:
            gone = stores.index(store)
            rows = program[()]
            for r in rows:
                if methods[int(r["method"])] == "equationConstraint":
                    r["store"], r["row"] = -1, -1
                elif int(r["store"]) > gone:
                    r["store"] = int(r["store"]) - 1
            program[...] = rows
            del stores[gone]
            program.attrs["stores"] = np.array(stores, dtype=object)
        f["meta"].attrs["opensees_schema_version"] = (
            OPENSEES_BEFORE_EQUATION_CONSTRAINT)


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
    if case.endswith("_equation"):
        # A 2.35 archive stores the tie's lines and replays them
        # (test_replay_carries_every_bridge_mp_line); one below it does not.
        _as_pre_2_35(path)
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


# ---------------------------------------------------------------------------
# Equation-route ties replay from their stored lines (#2416, opensees 2.35.0)
# ---------------------------------------------------------------------------

#: The schema corpus' assembly archive: real bytes from a writer below
#: opensees 2.35.0 (its zone is 2.22.0), joined by nine equation ties.
_PRE_FIX_CORPUS = (Path(__file__).resolve().parents[2]
                   / "fixtures" / "schema_corpus" / "assembly_1.0.h5")


def test_a_pre_fix_corpus_archive_still_refuses_its_equation_ties():
    from apeGmsh.opensees._internal.build.errors import BridgeError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = OpenSeesModel.from_h5(str(_PRE_FIX_CORPUS))
        with pytest.raises(BridgeError) as caught:
            model.build("tcl", flat=True)
    assert "tie (enforce='equation'): 9" in str(caught.value)
    assert "#2416" in str(caught.value)


def _mixed_equation_bridge() -> apeSees:
    """An equalDOF, a named equation tie, a penalty embedded node, an
    equation distributing coupling and a user ``equation_constraint`` row:
    the bridge writes them in that order, so the replay must interleave
    the stored lines with the other MP rows by emit index."""
    eq = lambda kind, name, w, dofs: InterpolationRecord(  # noqa: E731
        kind=kind, name=name, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array(w), dofs=dofs, enforce="equation")
    fem = _fem(
        partitioned=False,
        node_constraints=[NodePairRecord(
            kind="equal_dof", master_node=2, slave_node=4, dofs=[1])],
        surface_constraints=[
            eq("tie", "eq_tie", [0.25, 0.5, 0.25], [1, 2]),
            InterpolationRecord(
                kind="embedded", slave_node=4, master_nodes=[1, 2, 3],
                weights=np.array([0.0, 1.0, 0.0]), dofs=[1, 2, 3]),
            eq("distributing", "", [0.0, 1.0, 0.0], [1, 2, 3]),
        ])
    ops = _bridge(fem)
    ops.equation_constraint(constrained=(4, 3), retained=[(2, 3, -1.0)])
    return ops


def _mixed_archive(tmp_path: Path) -> "tuple[str, Path]":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = _mixed_equation_bridge()
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def test_stored_lines_interleave_with_the_other_mp_rows(tmp_path):
    fwd, path = _mixed_archive(tmp_path)
    eq = [ln for ln in fwd.splitlines() if ln.startswith("equationConstraint")]
    assert len(eq) == 2 + 3 + 1  # the tie's, the coupling's, the user row
    # The bridge put the embedded element between the two ties' lines.
    kinds = [ln.split()[0] if ln.startswith(("equalDOF", "equation"))
             else ln.split()[1] for ln in _mp_lines(fwd)]
    assert kinds == ["equalDOF", *["equationConstraint"] * 2,
                     "ASDEmbeddedNodeElement",
                     *["equationConstraint"] * 4]
    assert _replay(path) == fwd
    assert _replay(path, "py").count("ops.equationConstraint(") == 6


def test_the_rows_are_the_lines_and_the_comment(tmp_path):
    _fwd, path = _mixed_archive(tmp_path)
    rows = OpenSeesModel.from_h5(str(path))
    from apeGmsh.opensees.emitter.h5_reader import open as open_h5

    with open_h5(str(path)) as model:
        stored = model.equation_constraints()
    assert [r.name for r in stored] == ["eq_tie", "", "", "", "", ""]
    assert stored[0].retained == ((1, 1, -0.25), (2, 1, -0.5), (3, 1, -0.25))
    # Zero weights are filtered, as the bridge does.
    assert stored[2].retained == ((2, 1, -1.0),)
    assert stored[-1].retained == ((2, 3, -1.0),)
    assert rows._equation_tie_rows() == stored[:-1]


def test_an_archive_that_lost_its_lines_refuses(tmp_path):
    from apeGmsh.opensees._internal.build.errors import BridgeError

    _fwd, path = _archive(_fem(partitioned=False, **_KINDS["tie_equation"]),
                          tmp_path)
    with h5py.File(path, "r+") as f:
        cons = f["opensees/constraints/equationConstraint"]
        ret_offsets = cons["ret_offsets"][()]
        # Keep the group well formed but drop the last line.
        for key in ("cnode", "cdof", "ccoef", "name"):
            vals = cons[key][()][:-1]
            del cons[key]
            cons.create_dataset(key, data=vals)
        last = int(ret_offsets[-2])
        for key in ("ret_node", "ret_dof", "ret_coef"):
            vals = cons[key][()][:last]
            del cons[key]
            cons.create_dataset(key, data=vals)
        del cons["ret_offsets"]
        cons.create_dataset("ret_offsets", data=ret_offsets[:-1])
    model = OpenSeesModel.from_h5(str(path))
    with pytest.raises(BridgeError, match=r"tie \(enforce='equation'\): 1"):
        model.build("tcl", flat=True)


@pytest.mark.parametrize("damage", ["below_2_35", "offsets", "zero_coef",
                                    "missing_column", "empty"])
def test_a_malformed_store_raises(damage, tmp_path):
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from tests.fixtures.schema import OPENSEES_BEFORE_EQUATION_CONSTRAINT

    _fwd, path = _archive(_fem(partitioned=False, **_KINDS["tie_equation"]),
                          tmp_path)
    with h5py.File(path, "r+") as f:
        cons = f["opensees/constraints/equationConstraint"]
        if damage == "below_2_35":
            f["meta"].attrs["opensees_schema_version"] = (
                OPENSEES_BEFORE_EQUATION_CONSTRAINT)
        elif damage == "offsets":
            off = cons["ret_offsets"][()]
            off[-1] += 1
            cons["ret_offsets"][...] = off
        elif damage == "zero_coef":
            coef = cons["ret_coef"][()]
            coef[0] = 0.0
            cons["ret_coef"][...] = coef
        elif damage == "empty":
            # No writer makes an empty group: one line at least, or none.
            for key in list(cons.keys()):
                dtype = cons[key].dtype
                del cons[key]
                cons.create_dataset(
                    key, data=np.zeros(1 if key == "ret_offsets" else 0,
                                       dtype=dtype))
        else:
            del cons["ret_dof"]
    with pytest.raises(MalformedH5Error, match="equationConstraint"):
        OpenSeesModel.from_h5(str(path))


def test_partition_replicas_are_stored_once(tmp_path):
    (tmp_path / "f").mkdir()
    (tmp_path / "p").mkdir()
    _flat_fwd, flat = _archive(
        _fem(partitioned=False, **_KINDS["tie_equation"]), tmp_path / "f")
    part_fwd, part = _archive(
        _fem(partitioned=True, **_KINDS["tie_equation"]), tmp_path / "p")
    from apeGmsh.opensees.emitter.h5_reader import open as open_h5

    with open_h5(str(flat)) as a, open_h5(str(part)) as b:
        assert len(a.equation_constraints()) == 3
        assert b.equation_constraints() == a.equation_constraints()
    assert _mp_lines(_replay(part)) == _mp_lines(part_fwd)


def test_to_h5_rewrite_keeps_the_lines(tmp_path):
    fwd, path = _mixed_archive(tmp_path)
    rewrite = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OpenSeesModel.from_h5(str(path)).to_h5(str(rewrite))
    from apeGmsh.opensees.emitter.h5_reader import open as open_h5

    with open_h5(str(path)) as a, open_h5(str(rewrite)) as b:
        assert b.equation_constraints() == a.equation_constraints()
    assert _replay(rewrite) == fwd


def _staged_equation_tie_archive(tmp_path: Path) -> Path:
    """A global penalty embedded node and an equation-route embedded tie a
    stage claims (``s.embedded(name="st")``)."""
    from tests.opensees.h5.test_h5_stages_writer import _chain

    staged = InterpolationRecord(
        kind="embedded", name="st", slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.25, 0.5, 0.25]), dofs=[1, 2, 3],
        enforce="equation")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = _bridge(_fem(partitioned=False, surface_constraints=[staged]))
        with ops.stage(name="s0") as st:
            st.embedded(name="st")
            st.analysis(**{**_chain(ops),
                           "constraints": ops.constraints.Lagrange()})
            st.run(n_increments=1, dt=1.0)
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True)
        assert "equationConstraint" in deck.read_text(encoding="utf-8")
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return path


def test_a_stage_claimed_equation_tie_replays_its_lines(tmp_path):
    # Opensees 2.37.0 (#2448): the stage block stores the tie's lines, so
    # the replay writes them inside the stage, where the bridge did.
    path = _staged_equation_tie_archive(tmp_path)
    with h5py.File(path, "r") as f:
        assert "equationConstraint" not in f["opensees"].get(
            "constraints", {})
        cons = f["opensees/stages/stage_000/constraints"]
        assert len(cons["equationConstraint/cnode"]) == 3
        assert len(cons["equationConstraint_emit_index"]) == 3
    deck = _replay(path)
    assert sum(ln.startswith("equationConstraint ")
               for ln in deck.splitlines()) == 3


# ---------------------------------------------------------------------------
# Stage-claimed equation-route ties (#2448, opensees 2.37.0): each stage
# block stores its equationConstraint lines with their emit index, and the
# replay writes them in the stage's own position
# ---------------------------------------------------------------------------


#: A stage claim of each kind: ``(claim method, stream, named record)``.
#: The mixed corpus row ties each DOF once (node 4: the tie on 1-3; node
#: 2: the equalDOF on 4-5 and the distributing tie on 1-2), so its decks
#: solve. Every equation tie crosses the ranks (the equalDOF stays on rank
#: 1: Lagrange refuses a replicated one), and none ghosts the fixed base
#: node 1, which only rank 1 holds: a partitioned stage that ghosts a fixed
#: node re-fixes it inside the stage, and the flat replay of that archive
#: writes the fix twice, for every MP kind (a stage equalDOF from node 1
#: to node 4 does it on main too; #2530). For the equation rows the replay
#: refuses that archive by name (``tie_fixed_master``).
#:
#: ``pen_embedded`` is a penalty embedded node (an ASDEmbeddedNodeElement
#: line): in a stage after ``tie_equation`` it pins the emit-index merge,
#: since the bridge writes the tie's lines before the element.
_STAGE_CLAIMS: "dict[str, tuple[str, str, Any]]" = {
    "tie_equation": ("tie", "surface_constraints", InterpolationRecord(
        kind="tie", name="st_tie", slave_node=4, master_nodes=[2, 3],
        weights=np.array([0.75, 0.25]), dofs=[1, 2, 3],
        enforce="equation")),
    "distributing_equation": ("distributing", "surface_constraints",
                              InterpolationRecord(
        kind="distributing", name="st_dist", slave_node=4,
        master_nodes=[2, 3], weights=np.array([0.75, 0.25]),
        dofs=[1, 2], enforce="equation")),
    "embedded_equation": ("embedded", "surface_constraints",
                          InterpolationRecord(
        kind="embedded", name="st_emb", slave_node=4,
        master_nodes=[2, 3], weights=np.array([0.75, 0.25]),
        dofs=[1, 2, 3], enforce="equation")),
    "rot_equal_dof": ("equal_dof", "node_constraints", NodePairRecord(
        kind="equal_dof", name="st_rot", master_node=1, slave_node=2,
        dofs=[4, 5])),
    "top_a_distributing": ("distributing", "surface_constraints",
                           InterpolationRecord(
        kind="distributing", name="st_top_a", slave_node=2,
        master_nodes=[3, 4], weights=np.array([0.5, 0.5]),
        dofs=[1, 2], enforce="equation")),
    "pen_embedded": ("embedded", "surface_constraints", InterpolationRecord(
        kind="embedded", name="st_pen", slave_node=2,
        master_nodes=[1, 3, 4], weights=np.array([0.2, 0.4, 0.4]),
        dofs=[1, 2, 3])),
    "tie_fixed_master": ("tie", "surface_constraints", InterpolationRecord(
        kind="tie", name="st_tg", slave_node=4, master_nodes=[1, 2],
        weights=np.array([0.5, 0.5]), dofs=[1], enforce="equation")),
}


def _staged_mp_bridge(stages: str = "tie_equation",
                      partitioned: bool = False) -> apeSees:
    """The bridge on :func:`_fem` whose stages claim the
    :data:`_STAGE_CLAIMS` kinds named by ``stages``: stages split on
    ``;``, the kinds of one stage on ``,``. Each stage declares the
    EQ-capable Lagrange handler. The R1b oracle's stage corpus rows
    (#2448)."""
    from tests.opensees.h5.test_h5_stages_writer import _chain

    plan = [s.split(",") for s in stages.split(";")]
    streams: "dict[str, list[Any]]" = {}
    for kinds in plan:
        for k in kinds:
            _claim, stream, rec = _STAGE_CLAIMS[k]
            streams.setdefault(stream, []).append(rec)
    ops = _bridge(_fem(partitioned=partitioned, **streams))
    for i, kinds in enumerate(plan):
        with ops.stage(name=f"s{i}") as st:
            for k in kinds:
                claim, _stream, rec = _STAGE_CLAIMS[k]
                getattr(st, claim)(name=rec.name)
            st.analysis(**{**_chain(ops),
                           "constraints": ops.constraints.Lagrange()})
            st.run(n_increments=1, dt=1.0)
    return ops


_STAGED_ROWS = ("tie_equation", "distributing_equation", "embedded_equation",
                "tie_equation,rot_equal_dof;top_a_distributing")


def _staged_archive(stages: str, partitioned: bool,
                    tmp_path: Path) -> "tuple[str, Path]":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = _staged_mp_bridge(stages, partitioned)
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True, progress=False)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def _stage_rows(path: Path) -> "list[int]":
    """The number of stored stage equationConstraint rows, per stage."""
    with h5py.File(path, "r") as f:
        stages = f["opensees/stages"]
        out = []
        for name in sorted(stages):
            cons = stages[name].get("constraints", {})
            n = (len(cons["equationConstraint/cnode"])
                 if "equationConstraint" in cons else 0)
            seq = (len(cons["equationConstraint_emit_index"])
                   if "equationConstraint_emit_index" in cons else 0)
            assert n == seq, f"{name}: {n} rows, {seq} emit indices"
            out.append(n)
    return out


@pytest.mark.parametrize("partitioned", [False, True],
                         ids=["flat", "partitioned"])
@pytest.mark.parametrize("stages", _STAGED_ROWS)
def test_staged_replay_is_the_bridge_deck(stages, partitioned, tmp_path):
    fwd, path = _staged_archive(stages, partitioned, tmp_path)
    eq = [ln for ln in _mp_lines(fwd) if ln.startswith("equationConstraint")]
    assert eq, "the bridge deck carries no equationConstraint line"
    # Every line is a stage row: none is global, none is lost.
    with h5py.File(path, "r") as f:
        assert "equationConstraint" not in f["opensees"].get(
            "constraints", {})
    assert sum(_stage_rows(path)) == len(eq)
    assert _replay(path) == fwd


def test_a_stage_stores_its_lines_once_per_partition_replica(tmp_path):
    # A partitioned emit writes the cross-rank tie on both ranks (ADR 0068
    # Open item 2); the stage block captures each line once (ADR 0027
    # INV-1), as the flat emit does.
    from apeGmsh.opensees.emitter.h5_reader import open as open_h5

    (tmp_path / "f").mkdir()
    (tmp_path / "p").mkdir()
    _ff, flat = _staged_archive("tie_equation", False, tmp_path / "f")
    _pf, part = _staged_archive("tie_equation", True, tmp_path / "p")
    with open_h5(str(flat)) as a, open_h5(str(part)) as b:
        sa, sb = a.stages(), b.stages()
        assert len(sa[0].equation_constraints) == 3
        assert sb[0].equation_constraints == sa[0].equation_constraints


def test_the_stage_merge_keeps_the_bridge_order(tmp_path):
    # Stage s0 claims an equation tie and an equalDOF: the replay merges
    # the five MP kinds by the stage's emit index, so the lines come back
    # in the bridge's order, not in a fixed kind order.
    fwd, path = _staged_archive(
        "tie_equation,rot_equal_dof;top_a_distributing", False, tmp_path)
    assert _stage_rows(path) == [3, 2]
    assert _mp_lines(_replay(path)) == _mp_lines(fwd)


def test_the_stage_lines_replay_before_a_later_embedded_element(tmp_path):
    # The bridge writes the stage's equation lines, then the penalty
    # embedded node's element: the replay keeps that order only by the
    # stage emit index (a merge that put the equation rows last fails).
    fwd, path = _staged_archive("tie_equation,pen_embedded", False, tmp_path)
    lines = _mp_lines(fwd)
    eq = [i for i, ln in enumerate(lines) if ln.startswith("equationConstraint")]
    el = [i for i, ln in enumerate(lines) if ln.startswith("element")]
    assert eq and el and max(eq) < min(el), lines
    assert _replay(path) == fwd


def test_a_partitioned_stage_tie_that_ghosts_a_fixed_node_refuses(tmp_path):
    # #2530: a stage tie to the fixed base node 1 ghosts it on rank 2,
    # whose stage re-fixes it; the stage block stores that fix, so a flat
    # replay would fix node 1 twice. The replay refuses by name; the flat
    # archive of the same model replays exactly.
    from apeGmsh.opensees._internal.build.errors import BridgeError

    (tmp_path / "f").mkdir()
    (tmp_path / "p").mkdir()
    fwd, flat = _staged_archive("tie_fixed_master", False, tmp_path / "f")
    assert _replay(flat) == fwd
    pfwd, part = _staged_archive("tie_fixed_master", True, tmp_path / "p")
    assert sum(ln.startswith("fix 1 ") for ln in pfwd.splitlines()) == 1
    with pytest.raises(BridgeError, match=r"node\(s\) \[1\].*#2530"):
        OpenSeesModel.from_h5(str(part)).build("tcl", flat=True)


def _staged_plan_archive(fem: FEMData, plan: "list[Any]",
                         tmp_path: Path) -> "tuple[str, Path]":
    """The bridge on ``fem`` with one stage per callable of ``plan`` (each
    is handed the stage), each under Lagrange; its flat deck and archive."""
    from tests.opensees.h5.test_h5_stages_writer import _chain

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = _bridge(fem)
        for i, fn in enumerate(plan):
            with ops.stage(name=f"s{i}") as st:
                fn(st)
                st.analysis(**{**_chain(ops),
                               "constraints": ops.constraints.Lagrange()})
                st.run(n_increments=1, dt=1.0)
        deck = tmp_path / "bridge.tcl"
        ops.tcl(str(deck), flat=True, progress=False)
        path = tmp_path / "model.h5"
        ops.h5(str(path))
    return deck.read_text(encoding="utf-8"), path


def test_a_zero_weight_fixed_master_still_refuses(tmp_path):
    # The lines leave the zero-weight master 1 out, but the rank plan still
    # ghosts it on rank 2, whose stage re-fixes it (#2530).
    from apeGmsh.opensees._internal.build.errors import BridgeError

    tie = InterpolationRecord(
        kind="tie", name="zt", slave_node=4, master_nodes=[1, 2],
        weights=np.array([0.0, 1.0]), dofs=[1], enforce="equation")
    fwd, path = _staged_plan_archive(
        _fem(partitioned=True, surface_constraints=[tie]),
        [lambda st: st.tie(name="zt")], tmp_path)
    assert sum(ln.startswith("fix 1 ") for ln in fwd.splitlines()) == 1
    # equationConstraint cnode cdof ccoef (rnode rdof rcoef)*: master 1 is
    # in no line.
    retained = {tok for ln in fwd.splitlines()
                if ln.startswith("equationConstraint ")
                for tok in ln.split()[4::3]}
    assert retained == {"2"}
    with pytest.raises(BridgeError, match=r"node\(s\) \[1\].*#2530"):
        OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)


def test_a_re_fix_after_a_full_release_replays(tmp_path):
    # s0 releases node 3 on every DOF; s1 fixes it again and claims a tie
    # on it. The bridge's deck holds both fixes of node 3 too, so the
    # replay is the bridge's deck, not a refusal.
    tie = InterpolationRecord(
        kind="tie", name="t4", slave_node=4, master_nodes=[2, 3],
        weights=np.array([0.5, 0.5]), dofs=[1], enforce="equation")
    fwd, path = _staged_plan_archive(
        _fem(partitioned=True, surface_constraints=[tie]),
        [lambda st: st.remove_sp(nodes=[3], dofs=[1, 2, 3, 4, 5, 6]),
         lambda st: (st.fix(nodes=[3], dofs=(1,) * 6), st.tie(name="t4"))],
        tmp_path)
    assert sum(ln.startswith("fix 3 ") for ln in fwd.splitlines()) == 2
    assert _replay(path) == fwd


def test_to_h5_rewrite_keeps_the_stage_lines(tmp_path):
    from apeGmsh.opensees.emitter.h5_reader import open as open_h5

    fwd, path = _staged_archive(
        "tie_equation,rot_equal_dof;top_a_distributing", False, tmp_path)
    rewrite = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OpenSeesModel.from_h5(str(path)).to_h5(str(rewrite))
    with open_h5(str(path)) as a, open_h5(str(rewrite)) as b:
        assert b.stages() == a.stages()
    assert _replay(rewrite) == fwd


def _as_pre_2_37(path: Path) -> None:
    """Make the 2.37 archive ``path`` read as one below opensees 2.37.0, as
    its writer would have left it: no stage ``equationConstraint`` store,
    the program's stage ``equationConstraint`` calls linked to no store
    (the verb ledgered a stage call then), and the zone stamped 2.36.0."""
    from tests.fixtures.schema import (
        OPENSEES_BEFORE_STAGE_EQUATION_CONSTRAINT,
    )

    with h5py.File(path, "r+") as f:
        for st in f["opensees/stages"].values():
            if "constraints" not in st:
                continue
            cons = st["constraints"]
            for key in ("equationConstraint", "equationConstraint_emit_index"):
                if key in cons:
                    del cons[key]
            if len(cons.keys()) == 0:
                del st["constraints"]
        program = f["opensees/program"]
        methods = [m.decode() if isinstance(m, bytes) else str(m)
                   for m in program.attrs["methods"]]
        stores = [m.decode() if isinstance(m, bytes) else str(m)
                  for m in program.attrs["stores"]]
        rows = program[()]
        for r in rows:
            if (methods[int(r["method"])] == "equationConstraint"
                    and int(r["stage"]) >= 0):
                r["store"], r["row"] = -1, -1
        used = {int(r["store"]) for r in rows}
        store = "{scope}/constraints/equationConstraint"
        if store in stores and stores.index(store) not in used:
            gone = stores.index(store)
            for r in rows:
                if int(r["store"]) > gone:
                    r["store"] = int(r["store"]) - 1
            del stores[gone]
            program.attrs["stores"] = np.array(stores, dtype=object)
        program[...] = rows
        f["meta"].attrs["opensees_schema_version"] = (
            OPENSEES_BEFORE_STAGE_EQUATION_CONSTRAINT)


@pytest.mark.parametrize("stages", _STAGED_ROWS[:3])
def test_a_pre_2_37_stage_claimed_tie_still_refuses(stages, tmp_path):
    # A 2.35 / 2.36 writer ledgered a stage's lines: the archive holds
    # nothing to write them from, so the replay refuses by name rather
    # than run the stage untied.
    from apeGmsh.opensees._internal.build.errors import BridgeError

    _fwd, path = _staged_archive(stages, False, tmp_path)
    _as_pre_2_37(path)
    kind = _STAGE_CLAIMS[stages][2].kind
    with pytest.raises(BridgeError) as caught:
        OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)
    msg = str(caught.value)
    assert f"{kind} (enforce='equation'): 1" in msg
    assert "below opensees 2.37.0" in msg


def test_a_rewrite_of_a_pre_2_37_stage_tie_archive_still_refuses(tmp_path):
    # The to_h5 rewrite is stamped 2.37.0 but holds no line the source did
    # not store, so its replay refuses too, and the reason names a rewrite.
    from apeGmsh.opensees._internal.build.errors import BridgeError

    _fwd, path = _staged_archive("tie_equation", False, tmp_path)
    _as_pre_2_37(path)
    rewrite = tmp_path / "rewrite.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        OpenSeesModel.from_h5(str(path)).to_h5(str(rewrite))
    with pytest.raises(BridgeError) as caught:
        OpenSeesModel.from_h5(str(rewrite)).build("tcl", flat=True)
    msg = str(caught.value)
    assert "tie (enforce='equation'): 1" in msg
    assert "below opensees 2.37.0, or a rewrite of one" in msg


@pytest.mark.parametrize("damage", ["below_2_37", "seq_length",
                                    "seq_missing", "rows_missing",
                                    "zero_coef"])
def test_a_malformed_stage_store_raises(damage, tmp_path):
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
    from tests.fixtures.schema import (
        OPENSEES_BEFORE_STAGE_EQUATION_CONSTRAINT,
    )

    _fwd, path = _staged_archive("tie_equation", False, tmp_path)
    with h5py.File(path, "r+") as f:
        cons = f["opensees/stages/stage_000/constraints"]
        if damage == "below_2_37":
            f["meta"].attrs["opensees_schema_version"] = (
                OPENSEES_BEFORE_STAGE_EQUATION_CONSTRAINT)
        elif damage == "seq_length":
            seq = cons["equationConstraint_emit_index"][()][:-1]
            del cons["equationConstraint_emit_index"]
            cons.create_dataset("equationConstraint_emit_index", data=seq)
        elif damage == "seq_missing":
            del cons["equationConstraint_emit_index"]
        elif damage == "rows_missing":
            del cons["equationConstraint"]
        else:
            coef = cons["equationConstraint/ret_coef"][()]
            coef[0] = 0.0
            cons["equationConstraint/ret_coef"][...] = coef
    with pytest.raises(MalformedH5Error, match="equationConstraint"):
        OpenSeesModel.from_h5(str(path))
