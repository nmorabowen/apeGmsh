"""The deck replay writes its units in the bridge's emit order (#2193).

``OpenSeesModel.build`` merges every replay unit (a declaration record, a
``/opensees/commands`` row, an analysis-chain component, a block) by the
1-based emit index ``/opensees/program`` records for it, when the archive
records its declaration joins (``/opensees/decls@program_decl`` is
``"complete"``). The oracle is the bridge itself: the deck ``apeSees``
writes for the same model.

``_frame_bridge(analysis=True)`` is the closed case: the user declares
``system`` and ``analysis`` before anything else, so the bridge writes them
above the ``geomTransf`` (the analysis chain survives in ``pre_survives``),
where the fixed category order put them last.

A partitioned archive (#2277) records the ADR 0027 emit, so its replay
sorts by ``(_emit_flat section, emit index)``. The oracle is
``ops.tcl(flat=True)`` for the #2195 twin of ``_broad_partitioned`` (no
``OpenSeesTarget`` pinned): the series above the ``geomTransf``, and the
MPCO ``-R`` region just above its recorder. Every unpartitioned CORPUS
model and ``schema_corpus`` file replays every deck target as main did.

Seams: an archive whose ``program_decl`` is ``"absent"`` (a ``to_h5``
rewrite) keeps the category order; an empty program (only declarations,
no calls) keeps it too; a chain component the program never wrote, and an
order that would put a builder-scoped declaration above the gated-element
bracket (ADR 0099 INV-1), raise. For a partitioned order, a unit kind no
section places, a recorder region whose recorder no call wrote, and
partition blocks only inside stages raise.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import h5py
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
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.compose import (
    _ReplayOrder,
    _replay_into,
    _run_units,
)
from apeGmsh.opensees._internal.typed_records import (
    MaterialRecord,
    SectionSimpleRecord,
    TimeSeriesRecord,
    TransformRecord,
)
from apeGmsh.opensees.emitter.h5_reader import (
    PROGRAM_DECL_ABSENT,
    PROGRAM_DECL_COMPLETE,
    DeclarationRO,
    DeclarationTable,
    ProgramRun,
    records_program_decl,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from tests.opensees.golden.test_roundtrip_oracle import CORPUS
from tests.opensees.h5.test_h5_partitions_roundtrip import _build_bridge
from tests.opensees.h5.test_h5_solve_stamp import _frame_bridge


def _lines(text: str) -> list[str]:
    """The deck's command lines: comments (the stamp) and blanks dropped."""
    return [ln for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def _archive(ops: Any, tmp_path: Path, name: str = "model.h5") -> Path:
    path = tmp_path / name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # advisory emit warnings
        ops.h5(str(path))
    return path


def _replay(path: Path, kind: str = "tcl") -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = OpenSeesModel.from_h5(str(path)).build(kind)
    assert isinstance(deck, str)
    return _lines(deck)


@pytest.mark.parametrize("kind", ["tcl", "py"])
def test_replay_writes_the_chain_where_the_bridge_did(
        kind: str, tmp_path: Path) -> None:
    ops = _frame_bridge(analysis=True)
    bridge = tmp_path / f"bridge.{kind}"
    getattr(ops, kind)(str(bridge))
    want = _lines(bridge.read_text(encoding="utf-8"))
    got = _replay(_archive(ops, tmp_path), kind)
    assert got == want
    # The closed form: ``system`` and ``analysis`` are above the
    # ``geomTransf``, the first declaration, in both decks.
    transf = next(i for i, ln in enumerate(got) if "geomTransf" in ln)
    assert next(i for i, ln in enumerate(got) if "UmfPack" in ln) < transf
    assert next(i for i, ln in enumerate(got) if "Static" in ln) < transf


def test_without_the_order_the_chain_falls_last(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: the fixed category order writes the chain after the
    ``fix``, which the bridge deck does not."""
    path = _archive(_frame_bridge(analysis=True), tmp_path)
    monkeypatch.setattr(OpenSeesModel, "_replay_order", lambda self: None)
    got = _replay(path)
    assert got[-2:] == ["system UmfPack", "analysis Static"]
    assert got[-3].startswith("fix ")


def test_an_archive_with_absent_joins_keeps_the_category_order(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reload seam: an archive whose ``program_decl`` is ``"absent"`` (a
    file below opensees 2.31.0, or a ``to_h5`` rewrite of one; made here by
    re-marking a copy, the group is hash-excluded) replays in today's
    category order, refusing nothing new."""
    src =_archive(_frame_bridge(analysis=True), tmp_path)
    dst = tmp_path / "absent.h5"
    shutil.copyfile(src, dst)
    with h5py.File(dst, "r+") as f:
        f["opensees/decls"].attrs["program_decl"] = PROGRAM_DECL_ABSENT
    rewrite = OpenSeesModel.from_h5(str(dst))
    assert rewrite.declarations is not None
    assert rewrite.declarations.program_decl == PROGRAM_DECL_ABSENT
    assert rewrite._replay_order() is None
    replayed = _replay(dst)
    # The same deck the source replays with the merge off.
    monkeypatch.setattr(OpenSeesModel, "_replay_order", lambda self: None)
    assert replayed == _replay(src)
    assert replayed[-2:] == ["system UmfPack", "analysis Static"]


def test_a_partitioned_archive_sorts_by_flat_section(
        tmp_path: Path) -> None:
    """Its program records the partitioned emit (materials above the
    partition blocks), not the single-domain deck the replay writes, so
    its order sorts by ``_emit_flat`` section first (#2277); a
    non-partitioned archive's does not."""
    om = OpenSeesModel.from_h5(str(_archive(_build_bridge(), tmp_path)))
    assert om.declarations is not None
    assert om.declarations.program_decl == PROGRAM_DECL_COMPLETE
    assert any(r.method == "partition_open" for r in om.program())
    order = om._replay_order()
    assert order is not None and order.partitioned
    assert order.section("nodes", 9) < order.section("uniaxialMaterial", 1)
    flat = OpenSeesModel.from_h5(str(_archive(
        _frame_bridge(analysis=True), tmp_path, "flat.h5")))._replay_order()
    assert flat is not None and not flat.partitioned
    assert flat.section("nodes", 9) == flat.section("uniaxialMaterial", 1) == 0


def test_build_names_an_unknown_target_before_flat(tmp_path: Path) -> None:
    """#2199 review: the target is checked first, so ``flat=True`` on a
    misspelt target reports the target, not the flat rule."""
    om = OpenSeesModel.from_h5(
        str(_archive(_frame_bridge(analysis=False), tmp_path)))
    with pytest.raises(ValueError, match="unknown target 'tlc'"):
        om.build("tlc", flat=True)
    with pytest.raises(ValueError, match="flat applies"):
        om.build("live", flat=True)


# ---------------------------------------------------------------------------
# The merge on synthetic inputs
# ---------------------------------------------------------------------------


def _table(program_decl: str = PROGRAM_DECL_COMPLETE) -> DeclarationTable:
    return DeclarationTable(
        decls=(), tags=(), rows={}, program_decl=program_decl)


def _declarations_only(order: "_ReplayOrder | None") -> list[Any]:
    emitter = RecordingEmitter()
    _replay_into(
        emitter, ndm=3, ndf=6,
        uniaxial_materials=(MaterialRecord("Elastic", 1, (2.0e11,)),),
        simple_sections=(SectionSimpleRecord("Elastic", 1, (1.0, 2.0)),),
        transforms=(TransformRecord("Linear", 1, (1.0, 0.0, 0.0)),),
        time_series=(TimeSeriesRecord("Linear", 1, ()),),
        order=order,
    )
    return emitter.calls


def test_an_empty_program_keeps_the_category_order() -> None:
    """Degenerate seam: a model with declarations and no program rows has
    no key to merge by, so every unit keeps its category place."""
    got = _declarations_only(_ReplayOrder((), _table()))
    assert got == _declarations_only(None)
    assert [c[0] for c in got] == [
        "model", "uniaxialMaterial", "section", "geomTransf", "timeSeries"]


def test_the_order_refuses_an_archive_without_joins() -> None:
    with pytest.raises(ValueError, match="does not record its program"):
        _ReplayOrder((), _table(PROGRAM_DECL_ABSENT))
    with pytest.raises(ValueError, match="does not record its program"):
        _ReplayOrder((), None)


def test_a_chain_component_with_no_call_raises() -> None:
    """Fail closed: the archive stores a ``system`` no program call wrote."""
    program = (ProgramRun(1, 1, "model", "/meta@ndm", 0, -1, -1),)
    with pytest.raises(LookupError, match="'system' call"):
        _replay_into(
            RecordingEmitter(), ndm=3, ndf=6,
            analysis_attrs={"system": "UmfPack"},
            order=_ReplayOrder(program, _table()),
        )


def test_the_order_needs_deck_ordering() -> None:
    with pytest.raises(ValueError, match="needs deck_ordering=True"):
        _replay_into(
            RecordingEmitter(), ndm=3, ndf=6, deck_ordering=False,
            order=_ReplayOrder((), _table()),
        )


def test_a_scoped_declaration_above_the_gated_bracket_raises() -> None:
    """ADR 0099 INV-1, fail closed: a program that puts a ``geomTransf``
    above the gated elements' ``model BasicBuilder`` bracket is refused,
    and nothing runs."""
    ran: list[str] = []
    units: list[tuple[int | None, str, Any]] = [
        (5, "gated_elements", lambda: ran.append("gated")),
        (3, "geomTransf", lambda: ran.append("transf")),
    ]
    with pytest.raises(ValueError, match=r"\['geomTransf'\] above"):
        _run_units(units, _ReplayOrder((), _table()))
    assert ran == []
    # The category order (no program to merge by) runs them as listed.
    _run_units(units, None)
    assert ran == ["gated", "transf"]


def test_a_keyless_unit_stays_after_the_unit_before_it() -> None:
    ran: list[str] = []
    units: list[tuple[int | None, str, Any]] = [
        (7, "a", lambda: ran.append("a")),
        (None, "b", lambda: ran.append("b")),
        (3, "c", lambda: ran.append("c")),
        (8, "d", lambda: ran.append("d")),
    ]
    _run_units(units, _ReplayOrder((), _table()))
    assert ran == ["c", "a", "b", "d"]


# ---------------------------------------------------------------------------
# A partitioned archive replays in the bridge's flat-section order (#2277)
# ---------------------------------------------------------------------------


def _two_column_frame_fem() -> FEMData:
    """``fem_stub.make_two_column_frame_partitioned`` as a real
    :class:`FEMData` (the #2195 twin, ``k-2195-broad-partitioned-twin`` @
    28a95c91): columns A (nodes 1-2, element 1) and B (nodes 3-4, element
    2) at x = 0 and 1, rising along z, in the PG ``"Cols"``, with the node
    sets ``"Base"`` (1, 3) and ``"Top"`` (2, 4); column A on partition 1,
    column B on partition 2. Its archive carries a neutral zone, so it
    replays."""
    node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    node_coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=np.float64)
    line_info = make_type_info(
        code=1, gmsh_name="Line 2", dim=1, order=1, npe=2, count=2)
    lines = ElementGroup(
        element_type=line_info, ids=np.array([1, 2], dtype=np.int64),
        connectivity=np.array([[1, 2], [3, 4]], dtype=np.int64))

    def ids(values: list[int]) -> np.ndarray:
        return np.array(values, dtype=np.int64)

    def pg(name: str, nodes: list[int], elements: list[int]) -> dict[str, Any]:
        return {"name": name, "node_ids": ids(nodes),
                "node_coords": node_coords[[n - 1 for n in nodes]],
                "element_ids": ids(elements)}

    groups = PhysicalGroupSet({
        (1, 100): pg("Cols", [1, 2, 3, 4], [1, 2]),
        (0, 101): pg("Base", [1, 3], []),
        (0, 102): pg("Top", [2, 4], []),
    })
    parts = {
        1: {"node_ids": ids([1, 2]), "element_ids": ids([1])},
        2: {"node_ids": ids([3, 4]), "element_ids": ids([2])},
    }
    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords, physical=groups,
        labels=LabelSet({}), partitions=parts)
    elements = ElementComposite(
        groups={1: lines}, physical=groups, labels=LabelSet({}),
        partitions=parts)
    info = MeshInfo(n_nodes=4, n_elems=2, bandwidth=1, types=[line_info])
    return FEMData(nodes=nodes, elements=elements, info=info,
                   model_name="two_column_frame_partitioned")


def _recorders(ops: apeSees) -> None:
    ops.recorder.Node(
        file="out/node_disp.out", response="disp", pg="Top", dofs=(1, 2))
    ops.recorder.Element(
        file="out/ele_force.out", response=("globalForce",), pg="Cols")
    ops.recorder.MPCO(
        file="out/run.mpco", nodal_responses=("displacement",),
        elem_responses=("force",), nodes_pg="Top", elements_pg="Cols")


def _frame(ops: apeSees) -> None:
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ops.fix(pg="Base", dofs=(1,) * 6)


def _two_column_frame_partitioned() -> apeSees:
    """The golden corpus's ``build_model("two_column_frame_partitioned",
    "staged_partitioned", "recording")`` recipe on the real
    :func:`_two_column_frame_fem`, with no ``OpenSeesTarget`` pinned: the
    archive does not record the target mode, so a pinned one is a header
    the replay cannot write. An elastic column pair fixed at ``"Base"``,
    the node, element and MPCO recorders, and two stages, each with its
    own series, pattern and the parallel static chain."""
    ops = apeSees(_two_column_frame_fem())
    _frame(ops)
    _recorders(ops)
    for name, force, n in (("gravity", 10.0, 2), ("push", -5.0, 1)):
        with ops.stage(name=name) as s:
            series = ops.timeSeries.Linear()
            with s.pattern(series=series) as p:
                p.load(pg="Top", forces=(force, 0.0, 0.0, 0.0, 0.0, 0.0))
            s.analysis(
                test=ops.test.NormDispIncr(tol=1e-6, max_iter=10),
                algorithm=ops.algorithm.Newton(),
                integrator=ops.integrator.LoadControl(dlam=0.5),
                constraints=ops.constraints.Plain(),
                numberer=ops.numberer.ParallelRCM(),
                system=ops.system.Mumps(),
                analysis=ops.analysis.Static())
            s.run(n_increments=n)
    return ops


def _flat_bridge(ops: apeSees, tmp_path: Path) -> list[str]:
    path = tmp_path / "bridge.tcl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path), progress=False, flat=True)
    return _lines(path.read_text(encoding="utf-8"))


def _replay_flat(path: Path) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = OpenSeesModel.from_h5(str(path)).build("tcl", flat=True)
    assert isinstance(deck, str)
    return _lines(deck)


def test_a_partitioned_archive_replays_the_flat_bridge_deck(
        tmp_path: Path) -> None:
    """The oracle is ``ops.tcl(flat=True)`` for the same model, line for
    line (the stamp comments aside). The closed form: both series sit above
    the ``geomTransf`` (``_emit_flat`` step 5b above 6), and the MPCO
    ``-R`` region below the Node and Element recorders, above its own."""
    ops = _two_column_frame_partitioned()
    want = _flat_bridge(ops, tmp_path)
    got = _replay_flat(_archive(ops, tmp_path))
    assert got == want
    transf = next(i for i, ln in enumerate(got) if "geomTransf" in ln)
    series = [i for i, ln in enumerate(got) if ln.startswith("timeSeries")]
    assert len(series) == 2 and max(series) < transf
    region = next(i for i, ln in enumerate(got) if ln.startswith("region"))
    recorders = [i for i, ln in enumerate(got) if ln.startswith("recorder")]
    assert len(recorders) == 3
    assert recorders[1] < region < recorders[2]


def _unstaged_partitioned() -> apeSees:
    """The same frame unstaged, with a user analysis chain and one global
    series and pattern. No recorders: ``_emit_flat`` writes a global
    pattern after the recorders (the Kahn order of ``topological_order``,
    a pattern waits on its series), while the partitioned program writes
    it in the rank blocks, above them; no emit index recovers that, and
    the replay keeps main's pattern-then-recorder order."""
    ops = apeSees(_two_column_frame_fem())
    _frame(ops)
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-6, max_iter=10)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.5)
    ops.analysis.Static()
    series = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=series) as p:
        p.load(pg="Top", forces=(10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    return ops


def test_an_unstaged_partitioned_archive_replays_the_flat_bridge_deck(
        tmp_path: Path) -> None:
    """The user chain is a 4a declaration: the partitioned program writes
    it above the partition blocks, as it writes the series, yet the flat
    deck writes it below the nodes and the series below the chain."""
    ops = _unstaged_partitioned()
    want = _flat_bridge(ops, tmp_path)
    got = _replay_flat(_archive(ops, tmp_path))
    assert got == want
    last_node = max(i for i, ln in enumerate(got) if ln.startswith("node "))
    chain = next(i for i, ln in enumerate(got) if ln.startswith("constraints"))
    series = next(i for i, ln in enumerate(got) if ln.startswith("timeSeries"))
    assert last_node < chain < series
    assert got[-4].startswith("pattern Plain")


def _main_replay_order(self: OpenSeesModel) -> "_ReplayOrder | None":
    """``OpenSeesModel._replay_order`` as origin/main had it before #2277:
    no order (the fixed category order) for a partitioned archive."""
    if (self._opensees_version is None
            or not records_program_decl(self.declarations)
            or any(r.method == "partition_open" for r in self.program())):
        return None
    return _ReplayOrder(self.program(), self.declarations)


@pytest.mark.parametrize("build", [
    _two_column_frame_partitioned, _unstaged_partitioned])
def test_with_the_partition_gate_restored_the_twin_fails(
        build: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Revert proof: the category order main kept for a partitioned
    archive writes the ``geomTransf`` above the series, which the bridge's
    flat deck does not."""
    ops = build()
    want = _flat_bridge(ops, tmp_path)
    path = _archive(ops, tmp_path)
    monkeypatch.setattr(OpenSeesModel, "_replay_order", _main_replay_order)
    got = _replay_flat(path)
    assert got != want
    transf = next(i for i, ln in enumerate(got) if "geomTransf" in ln)
    assert transf < next(
        i for i, ln in enumerate(got) if ln.startswith("timeSeries"))


# ---------------------------------------------------------------------------
# The flat sections on synthetic programs: every seam fails closed
# ---------------------------------------------------------------------------


def _run(first: int, method: str, *, row: int = 0, stage: int = -1,
         decl: int = -1, count: int = 1) -> ProgramRun:
    return ProgramRun(first, count, method, "/store", row, stage, decl)


def _partitioned_order(
    *extra: ProgramRun,
    decls: tuple[DeclarationRO, ...] = (),
    tags: tuple[tuple[str, int, int, int], ...] = (),
) -> _ReplayOrder:
    program = (
        _run(1, "model"), _run(2, "constraints"), _run(3, "partition_open"),
        _run(4, "node", count=2), _run(6, "partition_close"),
        _run(7, "constraints"), *extra)
    return _ReplayOrder(program, DeclarationTable(
        decls=decls, tags=tags, rows={}, program_decl=PROGRAM_DECL_COMPLETE))


def test_a_chain_unit_is_4a_above_the_partition_blocks_else_7c() -> None:
    """A user chain declaration is written above the first partition
    block; the auto-emitted handler, after the blocks (``_emit_flat``
    7c: below the ties, above the initial stress)."""
    order = _partitioned_order()
    user, auto = order.section("chain", 2), order.section("chain", 7)
    assert order.section("nodes", 4) < user < order.section("geomTransf", 1)
    assert order.section("rebar_cells", 1) < auto
    assert auto < order.section("initial_stress", 1)
    with pytest.raises(ValueError, match="no emit index"):
        order.section("chain", None)


def test_an_unknown_unit_kind_raises_for_a_partitioned_order() -> None:
    with pytest.raises(ValueError, match="no _emit_flat section places"):
        _partitioned_order().section("contact", 1)
    # Unpartitioned, every kind is section 0: the emit index alone.
    assert _ReplayOrder((), _table()).section("contact", 1) == 0


def test_partition_blocks_only_inside_stages_raise() -> None:
    program = (_run(1, "model"), _run(2, "partition_open", stage=0))
    with pytest.raises(ValueError, match="only inside stages"):
        _ReplayOrder(program, _table())


_MPCO = DeclarationRO("opensees/recorder/#1", "recorder", "", False)


def test_a_recorder_region_takes_its_recorders_emit_index() -> None:
    """The MPCO filter region (declaration family ``recorder``), written
    once per rank, sorts with its recorder: section 8, the recorder's
    index; ``_run_units`` keeps it above the recorder by list order."""
    order = _partitioned_order(
        _run(8, "partition_open", row=1), _run(9, "region", row=0, decl=0),
        _run(10, "partition_close", row=1), _run(11, "recorder", decl=0),
        decls=(_MPCO,), tags=(("recorder", 1, 1, 0), ("region", 1, 1, 0)))
    assert order.tagged("region", frozenset({"region"})) == {1: 11}
    assert order.section("region", 11) == order.section("recorder", 11)
    assert order.section("region", 9) < order.section("recorder", 11)


def test_a_recorder_region_without_its_recorder_raises() -> None:
    order = _partitioned_order(
        _run(8, "region", decl=0), decls=(_MPCO,),
        tags=(("region", 1, 1, 0),))
    with pytest.raises(LookupError, match="no 'recorder' call"):
        order.tagged("region", frozenset({"region"}))


def test_a_stage_claimed_recorders_region_stays_with_the_regions() -> None:
    """A stage claims the recorder (its call is inside a stage block), so
    the global region keeps its own index in the regions' section, the
    place main's category order gave it."""
    order = _partitioned_order(
        _run(8, "region", decl=0), _run(9, "recorder", stage=0, decl=0),
        decls=(_MPCO,), tags=(("region", 1, 1, 0),))
    assert order.tagged("region", frozenset({"region"})) == {1: 8}
    assert order.section("region", 8) == order.section("command", 8)


# ---------------------------------------------------------------------------
# Unpartitioned replay is byte-identical to main's
# ---------------------------------------------------------------------------


_SCHEMA_CORPUS = sorted(
    (Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus")
    .glob("*.h5"))


def _decks(path: Path) -> dict[str, str]:
    """Every deck target of ``path``'s replay, or the error it raises."""
    out: dict[str, str] = {}
    for kind, flat in (("tcl", False), ("tcl", True), ("py", False)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                deck = OpenSeesModel.from_h5(str(path)).build(kind, flat=flat)
            except Exception as exc:  # noqa: BLE001 - the outcome is compared
                deck = f"{type(exc).__name__}: {exc}"
        out[f"{kind}{'-flat' if flat else ''}"] = str(deck)
    return out


def _assert_main_replay(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            program = OpenSeesModel.from_h5(str(path)).program()
        except Exception:  # noqa: BLE001 - _decks compares the load error
            program = ()
    if any(r.method == "partition_open" for r in program):
        pytest.skip("partitioned: test_roundtrip_oracle pins it to the bridge")
    got = _decks(path)
    with monkeypatch.context() as m:
        m.setattr(OpenSeesModel, "_replay_order", _main_replay_order)
        # main's merge: the emit index alone, no section.
        m.setattr(_ReplayOrder, "section", lambda self, kind, key: 0)
        want = _decks(path)
    assert got == want


@pytest.mark.parametrize("model", CORPUS, ids=[m.id for m in CORPUS])
def test_an_unpartitioned_corpus_model_replays_as_on_main(
        model: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#2277 changes only a partitioned archive's replay: every other
    CORPUS model replays every deck target exactly as main's gate and
    merge did."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops = model.build()
    _assert_main_replay(_archive(ops, tmp_path), monkeypatch)


@pytest.mark.parametrize("path", _SCHEMA_CORPUS, ids=[p.name for p in _SCHEMA_CORPUS])
def test_a_schema_corpus_file_replays_as_on_main(
        path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The same over every ``schema_corpus`` era (a copy: a replay never
    writes, but the fixture is frozen)."""
    copy = tmp_path / path.name
    shutil.copyfile(path, copy)
    _assert_main_replay(copy, monkeypatch)
