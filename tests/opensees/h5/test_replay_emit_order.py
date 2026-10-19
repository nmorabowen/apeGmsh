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

Seams: an archive whose ``program_decl`` is ``"absent"`` (a ``to_h5``
rewrite) and a partitioned archive keep the category order; an empty
program (only declarations, no calls) keeps it too; a chain component the
program never wrote, and an order that would put a builder-scoped
declaration above the gated-element bracket (ADR 0099 INV-1), raise.
"""
from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel
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
    DeclarationTable,
    ProgramRun,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
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


def test_a_partitioned_archive_keeps_the_category_order(
        tmp_path: Path) -> None:
    """Its program records the partitioned emit (materials above the
    partition blocks), not the single-domain deck the replay writes."""
    om = OpenSeesModel.from_h5(str(_archive(_build_bridge(), tmp_path)))
    assert om.declarations is not None
    assert om.declarations.program_decl == PROGRAM_DECL_COMPLETE
    assert any(r.method == "partition_open" for r in om.program())
    assert om._replay_order() is None


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
