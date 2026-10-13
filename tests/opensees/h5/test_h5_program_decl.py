"""``/opensees/program``'s ``decl`` column (A7, #1839, opensees 2.31.0).

Every emitted call carries the ``/opensees/decls`` row of the declaration
that made it when the emit proves it, and an explicit ``-1`` otherwise.

Oracles, each naming the right answer:

* **The declaration table, not the writer.** A spy records every Protocol
  call that reaches the archive, in emit order, so emit index ``i`` is a
  known ``(verb, args)``. A tagged call's expected decl is the
  ``/opensees/decls/tags`` run (read back from the file) that holds its
  ``(kind, tag)``; a tagless call's is the ``/opensees/decls/rows`` entry
  of the row its run points at; every other call's is ``-1``. Driven
  over flat, staged and partitioned models, and per stage.
* **Never a guessed id.** A call no declaration proves (an undeclared tag,
  the analysis chain, a ``/opensees/commands`` row or a ``fix`` with none
  open)
  carries ``-1``, even right after a declaration was open.
* **Fail closed.** A reader fed an out-of-range ``decl`` raises.
* **Reload.** ``from_h5 -> to_h5`` keeps the column.
"""
from __future__ import annotations

import functools
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator

import h5py
import pytest

from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees.emitter import h5_reader
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error
from apeGmsh.opensees.emitter.verbs import VERBS
from tests.opensees.golden.builder import build_model
from tests.opensees.h5.test_h5_decls import (
    _all_verb_models,
    _flat_frame,
    _staged,
)

#: Tagged verbs: ``(allocator kind, index of the tag argument)``. Test-side
#: and independent of the writer's own table.
TAGGED: dict[str, tuple[str, int]] = {
    "uniaxialMaterial": ("uniaxialMaterial", 1),
    "nDMaterial": ("nDMaterial", 1),
    "section": ("section", 1), "section_open": ("section", 1),
    "geomTransf": ("geomTransf", 1),
    "beamIntegration": ("beamIntegration", 1),
    "timeSeries": ("timeSeries", 1), "pattern_open": ("pattern", 1),
    "damping": ("damping", 1), "region": ("region", 0),
    "element": ("element", 1),
}

#: Tagless verbs whose row's ``/opensees/decls/rows`` entry is the
#: declaration open on the call: ``store template -> rows leaf``.
ROWS_LEAF: dict[str, str] = {
    "{scope}/bcs/fix": "bcs/fix",
    "{scope}/bcs/mass": "bcs/mass",
    "{scope}/recorders/{kind}_{k}": "recorders",
    "/opensees/stages/stage_{k:03d}/remove_sp": "remove_sp",
    "/opensees/stages/stage_{k:03d}/remove_element": "remove_element",
    "/opensees/stages/stage_{k:03d}/update_material_stage":
        "update_material_stage",
    "/opensees/stages/stage_{k:03d}/rayleigh": "rayleigh",
    "/opensees/stages/stage_{k:03d}/patterns/{type}_{tag}/sp_holds":
        "sp_holds",
    "/opensees/commands": "commands",
}


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[str, tuple[Any, ...]]]]:
    """Every Protocol call the archive counts, as ``(verb, args)`` in emit
    order: a top-level call (the tape's depth is 0) that returned."""
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def wrap(verb: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def spied(self: H5Emitter, *args: Any, **kwargs: Any) -> Any:
            top = not self._program.depth
            out = fn(self, *args, **kwargs)
            if top:
                calls.append((verb, args))
            return out
        return spied

    for verb, row in VERBS.items():
        if row.via == "protocol":
            monkeypatch.setattr(
                H5Emitter, verb, wrap(verb, getattr(H5Emitter, verb)))
    yield calls


def _run_of(table: h5_reader.DeclarationTable, kind: str, tag: int) -> int:
    for k, first, count, decl in table.tags:
        if k == kind and first <= tag < first + count:
            return int(decl)
    return -1


def _expected(
    path: Path, calls: list[tuple[str, tuple[Any, ...]]],
) -> tuple[list[int], list[int], list[tuple[str, int]]]:
    """``(got, want, (method, stage))`` per emit index, ``want`` read from
    the file's own declaration table."""
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
        runs = m.program()
    assert table is not None
    with h5py.File(str(path), "r") as f:
        ops = f["opensees"]
        cmd_methods = ([m.decode() for m in ops["commands"]["method"][()]]
                       if "commands" in ops else [])
        cmd_stages = (ops["commands"]["stage"][()].tolist()
                      if "commands" in ops else [])
        holds: dict[int, list[str]] = {}
        for name in (ops["stages"] if "stages" in ops else ()):
            pats = ops["stages"][name].get("patterns", {})
            holds[int(name.rsplit("_", 1)[1])] = sorted(
                f"stages/{name}/patterns/{p}/sp_holds" for p in pats
                if "sp_holds" in pats[p])
    assert sum(r.count for r in runs) == len(calls)
    got: list[int] = []
    want: list[int] = []
    where: list[tuple[str, int]] = []
    for run in runs:
        leaf = ROWS_LEAF.get(run.store)
        for i in range(run.count):
            verb, args = calls[run.first - 1 + i]
            assert verb == run.method, (run, verb)
            got.append(run.decl)
            where.append((run.method, run.stage))
            if verb in TAGGED:
                kind, pos = TAGGED[verb]
                want.append(_run_of(table, kind, int(args[pos])))
            elif leaf is not None and run.row >= 0:
                row = run.row + i
                prefix = "" if run.stage < 0 else f"stages/stage_{run.stage:03d}/"
                if leaf == "commands":
                    ix = [j for j, (m, s) in enumerate(zip(cmd_methods, cmd_stages))
                          if m == verb and s == run.stage][row]
                    want.append(table.rows["commands"][ix])
                elif leaf == "sp_holds":
                    assert len(holds[run.stage]) == 1, holds
                    want.append(table.rows[holds[run.stage][0]][row])
                else:
                    want.append(table.rows[prefix + leaf][row])
            else:
                want.append(-1)
    return got, want, where


def _models() -> Iterator[tuple[str, apeSees]]:
    yield "staged", _staged(named=True)
    yield "flat_frame", _flat_frame(named=False)
    for i, ops in enumerate(_all_verb_models()):
        yield f"verbs_{i}", ops
    for mode in ("flat", "partitioned", "staged", "staged_partitioned"):
        yield (f"golden_{mode}",
               build_model("two_column_frame_partitioned", mode, "recording"))


def test_every_call_joins_the_declaration_table(
    tmp_path: Path, spy: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """Over every model: each emit index's decl is the one the file's own
    declaration table gives it, ``-1`` where it gives none."""
    declared = 0
    for name, ops in _models():
        spy.clear()
        p = tmp_path / f"{name}.h5"
        ops.h5(str(p))
        got, want, _where = _expected(p, spy)
        assert got == want, name
        declared += sum(d != -1 for d in got)
    assert declared > 0


def test_each_stage_joins_its_declarations(
    tmp_path: Path, spy: list[tuple[str, tuple[Any, ...]]],
) -> None:
    """A staged model: every stage's declared calls (its element, fix,
    mass and recorder) join to the stage's own declarations."""
    p = tmp_path / "staged.h5"
    _staged(named=True).h5(str(p))
    got, want, where = _expected(p, spy)
    assert got == want
    table = h5_reader.open(str(p)).declarations()
    assert table is not None
    stage_keys = {table.decls[d].key for d, (_m, s) in zip(got, where)
                  if s == 0 and d != -1}
    assert stage_keys == {
        "opensees/element/#2", "opensees/fix/fill_top",
        "opensees/mass/fill_mass", "opensees/recorder/fill_disp"}
    # The stage's chain, bracket and domain calls declare nothing.
    assert {m for d, (m, s) in zip(got, where) if s == 0 and d == -1} == {
        "stage_open", "stage_close", "node", "domain_change", "constraints",
        "numberer", "system", "test", "algorithm", "integrator", "analysis",
        "analyze"}


def _emitter() -> H5Emitter:
    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.set_declarations(
        [("opensees/uniaxialMaterial/#1", "uniaxialMaterial", "", False),
         ("opensees/fix/#1", "fix", "", False),
         ("opensees/fix/#2", "fix", "", False)],
        [("uniaxialMaterial", 7, 1, 0)],
    )
    return e


def _decls_by_method(e: H5Emitter) -> dict[str, list[int]]:
    tape = e._program
    out: dict[str, list[int]] = {}
    for first, count, m, _st, _row, _stage, decl in tape.runs:
        out.setdefault(tape.methods[m], []).extend([decl] * count)
    return out


def test_unproved_calls_carry_minus_one() -> None:
    """An undeclared tag, the analysis chain and a fix or commands row
    with no declaration open carry ``-1``, even right after one was open.
    (A raw ``command`` token has no allow-list row yet, ADR 0114 Q7, so it
    cannot reach the archive at all.)"""
    e = _emitter()
    e.uniaxialMaterial("Elastic", 7, 1.0)       # declared: run 0
    e.uniaxialMaterial("Elastic", 8, 1.0)       # no run holds tag 8
    e.set_declaration(1)
    e.fix(1, 1, 1)                              # proved: decl 1
    e.numberer("RCM")                           # open, but not proved
    e.node(9, 0.0, 0.0)
    e.set_declaration(-1)
    e.fix(2, 1, 1)                              # none open
    e.modal_damping(0.05)                       # a commands row, none open
    got = _decls_by_method(e)
    assert got["uniaxialMaterial"] == [0, -1]
    assert got["fix"] == [1, -1]
    assert got["numberer"] == [-1]
    assert got["node"] == [-1]
    assert got["modal_damping"] == [-1]


def test_runs_break_where_the_declaration_changes() -> None:
    """Two consecutive fixes of two declarations are two runs, though
    their rows are consecutive."""
    e = _emitter()
    e.set_declaration(1)
    e.fix(1, 1, 1)
    e.set_declaration(2)
    e.fix(2, 1, 1)
    e.set_declaration(-1)
    fix = e._program.methods.index("fix")
    runs = [r for r in e._program.runs if r[2] == fix]
    assert [(r[1], r[4], r[6]) for r in runs] == [(1, 0, 1), (1, 1, 2)]


@pytest.mark.parametrize("bad", [-2, 99])
def test_reader_refuses_an_out_of_range_decl(tmp_path: Path, bad: int) -> None:
    p = tmp_path / "m.h5"
    _flat_frame(named=False).h5(str(p))
    q = tmp_path / "tampered.h5"
    shutil.copy(p, q)
    with h5py.File(str(q), "a") as f:
        ds = f["opensees"]["program"]
        rows = ds[()]
        rows["decl"][0] = bad
        ds[...] = rows
    with h5_reader.open(str(p)) as m:
        assert m.program()
    with h5_reader.open(str(q)) as m, pytest.raises(
            MalformedH5Error, match="names declaration"):
        m.program()


def test_writer_refuses_an_out_of_range_decl(tmp_path: Path) -> None:
    e = _emitter()
    e.set_declaration(1)
    e.fix(1, 1, 1)
    e._program.runs[-1][6] = 3                  # past the 3 declarations
    with pytest.raises(RuntimeError, match="names declaration 3"):
        e.write(str(tmp_path / "x.h5"))


@pytest.mark.parametrize("model", ["flat", "staged", "partitioned"])
def test_rewrite_keeps_the_decl_column(tmp_path: Path, model: str) -> None:
    """``from_h5 -> to_h5`` echoes the column (the K1-8 round-1 class)."""
    from tests.opensees.h5.test_h5_commands_program import (
        _staged as _staged_real,
    )

    p, q = tmp_path / "src.h5", tmp_path / "out.h5"
    ops = (_flat_frame(named=True) if model == "flat" else _staged_real(
        partitioned=model == "partitioned", rayleigh=True, profile=True))
    ops.h5(str(p))
    OpenSeesModel.from_h5(str(p)).to_h5(str(q))
    with h5_reader.open(str(p)) as a, h5_reader.open(str(q)) as b:
        src, out = a.program(), b.program()
    assert any(r.decl != -1 for r in src)
    assert [r.decl for r in out] == [r.decl for r in src]


# ---------------------------------------------------------------------------
# The completeness marker (/opensees/decls@program_decl)
# ---------------------------------------------------------------------------

CORPUS = Path(__file__).resolve().parents[2] / "fixtures" / "schema_corpus"


def _hash(path: Path) -> tuple[str, str]:
    """``(stored, recomputed)`` ``model_hash``."""
    from apeGmsh.opensees._internal.lineage import compute_model_hash

    with h5py.File(str(path), "r") as f:
        lin = f["meta"]["lineage"].attrs
        return (str(lin["model_hash"]),
                compute_model_hash(str(lin["fem_hash"]), f["opensees"]))


def _marker(path: Path) -> str | None:
    with h5py.File(str(path), "r") as f:
        value = f["opensees"]["decls"].attrs.get("program_decl")
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def test_a_legacy_rewrite_is_marked_absent_and_keeps_its_hash(
    tmp_path: Path,
) -> None:
    """The prior-minor corpus file (no marker: its decl column was never
    recorded) rewritten through ``to_h5`` is stamped current, marked
    ``"absent"``, refuses the join query, still reads, and keeps its
    ``model_hash`` (the marker sits on the hash-excluded ``decls``)."""
    from tests.fixtures.schema import OPENSEES_CURRENT, OPENSEES_PRIOR_MINOR

    legacy = tmp_path / "legacy.h5"
    minor = OPENSEES_PRIOR_MINOR.rsplit(".", 1)[0]
    shutil.copy(CORPUS / f"opensees_{minor}.h5", legacy)
    assert _marker(legacy) is None
    with h5_reader.open(str(legacy)) as m:
        assert not h5_reader.records_program_decl(m.declarations())
    out, again = tmp_path / "rw.h5", tmp_path / "rw2.h5"
    OpenSeesModel.from_h5(str(legacy)).to_h5(str(out))
    OpenSeesModel.from_h5(str(out)).to_h5(str(again))
    with h5py.File(str(out), "r") as f:
        stamp = f["meta"].attrs["opensees_schema_version"]
    assert (stamp.decode() if isinstance(stamp, bytes) else str(stamp)) == (
        OPENSEES_CURRENT)
    for path in (out, again):
        assert _marker(path) == "absent"
        with h5_reader.open(str(path)) as m:
            assert not h5_reader.records_program_decl(m.declarations())
            assert {r.decl for r in m.program()} == {-1}
            with pytest.raises(h5_reader.ProgramDeclAbsentError,
                               match="not recorded"):
                m.program_declarations()
        stored, recomputed = _hash(path)
        assert stored == recomputed == _hash(legacy)[0]


def test_a_current_rewrite_is_marked_complete_and_keeps_its_hash(
    tmp_path: Path,
) -> None:
    """A current emit is marked ``"complete"``; its rewrite echoes the
    marker and the column, and keeps ``model_hash``."""
    from tests.opensees.h5.test_h5_commands_program import (
        _staged as _staged_real,
    )

    src, out = tmp_path / "src.h5", tmp_path / "rw.h5"
    _staged_real(partitioned=False, rayleigh=True, profile=True).h5(str(src))
    OpenSeesModel.from_h5(str(src)).to_h5(str(out))
    for path in (src, out):
        assert _marker(path) == "complete"
        with h5_reader.open(str(path)) as m:
            assert h5_reader.records_program_decl(m.declarations())
            joins = m.program_declarations()
        assert any(d is not None for _r, d in joins)
        assert all((d is None) == (r.decl == -1) for r, d in joins)
    assert _hash(out) == _hash(src)
    assert _hash(src)[0] == _hash(src)[1]


def test_reader_refuses_an_unknown_marker(tmp_path: Path) -> None:
    p = tmp_path / "m.h5"
    _flat_frame(named=False).h5(str(p))
    with h5py.File(str(p), "a") as f:
        f["opensees"]["decls"].attrs["program_decl"] = "partial"
    with h5_reader.open(str(p)) as m, pytest.raises(
            MalformedH5Error, match="program_decl"):
        m.declarations()
