"""V3a (#1640): the ``/sequence`` readable-stages zone (ADR 0112 D2b).

Oracles:

* **Closed-form rows.** The staged golden (``two_column_frame`` x
  ``staged``) is written by a script whose stages are known: ``gravity``
  (2 increments) then ``push`` (1), each opening one ``Plain`` pattern,
  plus one global ``fix``.  The vanilla golden gives one implicit stage
  from ``/opensees/analysis``, ``@staged`` 0.
* **Single source.** An independent h5py walk of ``/opensees`` (it does
  not import the deriver) lists every record a change row must point at;
  each row's ``source`` resolves in the file and carries the stage its
  path names.  The mutation tests break the deriver (drop a row, swap the
  stage order) and show this oracle fails.
* **Locks.** The deriver imports no record or primitive module and reads
  ``/opensees`` only through the reader's public methods (AST); the
  ``(op, what)`` vocabulary equals ``emitter/verbs.py::VERBS``'s ``seq``
  pairs plus ``("activate", "group")`` (the folded V3b lock).
"""
from __future__ import annotations

import ast
import math
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees import OpenSeesTarget, apeSees
from apeGmsh.opensees._internal import sequence_h5
from apeGmsh.opensees._internal.lineage import (
    compute_fem_hash,
    compute_model_hash,
    read_stored_lineage,
)
from apeGmsh.opensees._internal.schema_version import (
    _ZONE_KEY,
    SEQUENCE,
    SEQUENCE_KEY,
    SchemaVersion,
    read_zone_version,
    reader_floor,
    reader_version,
)
from apeGmsh.opensees._internal.sequence_h5 import (
    CHAIN_ARGS,
    CHAIN_TOKENS,
    CHANGE_KINDS,
    SequenceError,
    SequenceOverflowError,
    write_sequence_into,
)
from apeGmsh.opensees.emitter.verbs import SEQ_OPS, SEQ_WHATS, VERBS
from tests.opensees.golden.builder import (
    FIXTURES,
    _chain,
    _declare_elements,
    _fem_for,
    build_model,
)

MODULE = Path(sequence_h5.__file__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(v: Any) -> Any:
    return v.decode("utf-8") if isinstance(v, bytes) else v


def _nan_safe(read: Any) -> Any:
    """``_read`` output with NaN replaced by a token, so equality holds."""
    def cell(v: Any) -> Any:
        return "NaN" if isinstance(v, float) and math.isnan(v) else v
    attrs, stages, changes = read
    return (attrs, [{k: cell(v) for k, v in r.items()} for r in stages],
            [{k: cell(v) for k, v in r.items()} for r in changes])


def _table(g: h5py.Group) -> list[dict[str, Any]]:
    cols = {c: [_text(v) for v in g[c][()]] for c in g}
    n = {len(v) for v in cols.values()}
    assert len(n) <= 1, f"{g.name}: columns disagree in length {n}"
    rows = n.pop() if n else 0
    return [{c: cols[c][i] for c in cols} for i in range(rows)]


def _read(path: Path) -> tuple[dict[str, Any], list[dict], list[dict]]:
    with h5py.File(path, "r") as f:
        g = f["sequence"]
        attrs = {k: _text(v) for k, v in g.attrs.items()}
        return attrs, _table(g["stages"]), _table(g["changes"])


def _write(ops: apeSees, tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    ops.h5(str(p))
    return p


def _stage_of(source: str) -> int | None:
    m = re.match(r"/opensees/stages/stage_(\d{3})", source)
    return int(m.group(1)) if m else None


def _expected_sources(path: Path) -> set[str]:
    """Every record a change row must name, walked straight off the file."""
    want: set[str] = set()
    with h5py.File(path, "r") as f:
        ops = f["opensees"]
        if "patterns" in ops:
            want |= {f"/opensees/patterns/{n}" for n in ops["patterns"]}
        if "recorders" in ops:
            want |= {f"/opensees/recorders[{i}]"
                     for i in range(len(ops["recorders"]))}
        if "stages" in ops:
            for sname, sg in ops["stages"].items():
                pre = f"/opensees/stages/{sname}"
                if "patterns" in sg:
                    want |= {f"{pre}/patterns/{n}" for n in sg["patterns"]}
                if "recorders" in sg:
                    want |= {f"{pre}/recorders[{i}]"
                             for i in range(len(sg["recorders"]))}
                if "bcs" in sg:
                    want |= {f"{pre}/bcs/{n}" for n in sg["bcs"]}
    return want


def _check_single_source(path: Path) -> None:
    """The mutation oracle: raises AssertionError on any disagreement."""
    attrs, stages, changes = _read(path)
    with h5py.File(path, "r") as f:
        n_stage_groups = (len(f["opensees/stages"])
                          if "opensees/stages" in f else 0)
        for i, row in enumerate(stages):
            assert row["order"] == i, f"stage row {i}: order {row['order']}"
            src = row["source"]
            assert src in f, f"stage row {i}: source {src!r} not in file"
            if attrs["staged"]:
                assert _stage_of(src) == i, f"stage row {i} reads {src}"
                assert _text(f[src].attrs["name"]) == row["name"]
                assert int(f[src].attrs["analyze_steps"]) == row["steps"]
        if attrs["staged"]:
            assert len(stages) == n_stage_groups
        else:
            assert n_stage_groups == 0 and len(stages) <= 1
        for row in changes:
            base = row["source"].split("[", 1)[0]
            assert base in f, f"change source {row['source']!r} not in file"
            k = _stage_of(row["source"])
            assert row["stage"] == (0 if k is None else k), row
            assert (row["op"], row["what"]) in CHANGE_KINDS, row
    got = {r["source"] for r in changes}
    missing = _expected_sources(path) - got
    assert not missing, f"no change row for {sorted(missing)}"


def _rich_staged(tmp_path: Path) -> Path:
    """A staged model with named declarations, a stage-bound mass and
    recorder, and a stage with no change at all."""
    spec = FIXTURES["two_column_frame"]
    ops = apeSees(_fem_for("two_column_frame", "staged"),  # type: ignore[arg-type]
                  opensees=OpenSeesTarget(mode="stock"))
    ops.model(ndm=spec.ndm, ndf=spec.ndf)
    _declare_elements(ops, spec)
    ops.fix(pg=spec.fixed_pg, dofs=(1,) * spec.ndf, name="base")
    rec = ops.recorder.Node(
        file="out/tip.out", response="disp", pg=spec.loaded_pg, dofs=(1, 2),
        name="tip",
    )
    with ops.stage(name="gravity") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts, name="dead") as p:
            p.load(pg=spec.loaded_pg, forces=(10.0,) + (0.0,) * 5)
        s.mass(pg=spec.loaded_pg, values=(1.0,) * spec.ndf, name="top_mass")
        s.recorder(rec)
        s.analysis(**_chain(ops, False))
        s.run(n_increments=3)
    with ops.stage(name="hold") as s:
        s.set_time(5.0)
        s.analysis(**_chain(ops, False))
        s.run(n_increments=1)
    return _write(ops, tmp_path, "rich.h5")


# ---------------------------------------------------------------------------
# Closed-form oracles
# ---------------------------------------------------------------------------


def test_staged_golden_rows(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")
    attrs, stages, changes = _read(path)
    assert attrs["staged"] == 1 and attrs["derived_from"] == "/opensees"
    assert [(r["order"], r["name"], r["steps"]) for r in stages] == [
        (0, "gravity", 2), (1, "push", 1)]
    for r in stages:
        assert (r["analysis"], r["integrator"], r["algorithm"], r["test"],
                r["system"], r["numberer"], r["handler"]) == (
            "Static", "LoadControl", "Newton", "NormDispIncr", "UmfPack",
            "RCM", "Plain")
        assert r["integrator_args"] == "[0.5]"
        assert r["test_args"] == "[1e-06, 10.0, 0.0, 2.0]"
        assert r["algorithm_args"] == "[]"
        assert math.isnan(r["dt"]) and math.isnan(r["set_time"])
        assert r["domain_change"] == 1
    assert [(r["stage"], r["op"], r["what"], r["name"], r["count"],
             r["source"]) for r in changes] == [
        (0, "add", "fix", "", 2, "/opensees/bcs/fix"),
        (0, "add", "pattern", "", 1,
         "/opensees/stages/stage_000/patterns/Plain_1"),
        (1, "add", "pattern", "", 1,
         "/opensees/stages/stage_001/patterns/Plain_2"),
    ]
    _check_single_source(path)
    with h5py.File(path, "r") as f:
        assert _text(f["meta"].attrs[SEQUENCE_KEY]) == "1.0.0"
        assert f["sequence/stages/order"].dtype == np.int32
        assert f["sequence/stages/steps"].dtype == np.int32
        assert f["sequence/changes/count"].dtype == np.int32
        assert f["sequence/changes/stage"].dtype == np.int32
        assert attrs["model_hash"] == read_stored_lineage(f["meta"])[1]


def test_vanilla_deck_is_one_implicit_stage(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "flat", "h5"),
                  tmp_path, "flat.h5")
    attrs, stages, changes = _read(path)
    assert attrs["staged"] == 0
    assert len(stages) == 1
    row = stages[0]
    assert (row["order"], row["name"], row["source"]) == (
        0, "", "/opensees/analysis")
    assert row["integrator"] == "LoadControl"
    with h5py.File(path, "r") as f:
        a = f["opensees/analysis"].attrs
        assert row["steps"] == (int(a["analyze_steps"])
                                if "analyze_steps" in a else -1)
    assert {(r["op"], r["what"], r["source"]) for r in changes} == {
        ("add", "fix", "/opensees/bcs/fix"),
        ("add", "pattern", "/opensees/patterns/Plain_1"),
    }
    assert all(r["stage"] == 0 for r in changes)
    _check_single_source(path)


def test_model_only_file_has_empty_zone(tmp_path: Path) -> None:
    spec = FIXTURES["two_column_frame"]
    ops = apeSees(_fem_for("two_column_frame", "flat"),  # type: ignore[arg-type]
                  opensees=OpenSeesTarget(mode="stock"))
    ops.model(ndm=spec.ndm, ndf=spec.ndf)
    _declare_elements(ops, spec)
    ops.fix(pg=spec.fixed_pg, dofs=(1,) * spec.ndf)
    path = _write(ops, tmp_path, "model_only.h5")
    attrs, stages, changes = _read(path)
    assert attrs["staged"] == 0 and stages == [] and changes == []


def test_names_stage_bound_records_and_an_empty_stage(tmp_path: Path) -> None:
    path = _rich_staged(tmp_path)
    attrs, stages, changes = _read(path)
    assert [(r["name"], r["steps"]) for r in stages] == [
        ("gravity", 3), ("hold", 1)]
    assert stages[1]["set_time"] == 5.0
    assert math.isnan(stages[0]["set_time"])
    rows = [(r["stage"], r["op"], r["what"], r["name"]) for r in changes]
    assert (0, "add", "fix", "base") in rows
    assert (0, "add", "pattern", "dead") in rows
    assert (0, "add", "mass", "top_mass") in rows
    assert (0, "add", "recorder", "tip") in rows
    # The second stage only re-binds the chain: no change row.
    assert not [r for r in changes if r["stage"] == 1]
    _check_single_source(path)


# ---------------------------------------------------------------------------
# Mutation oracle: a broken deriver fails the single-source check
# ---------------------------------------------------------------------------


def test_mutation_dropped_change_row_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = sequence_h5._stage_changes

    def drop_last(k: int, st: Any, names: Any) -> list:
        return real(k, st, names)[:-1]

    monkeypatch.setattr(sequence_h5, "_stage_changes", drop_last)
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "mut1.h5")
    with pytest.raises(AssertionError, match="no change row"):
        _check_single_source(path)


def test_mutation_swapped_stage_order_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = sequence_h5.derive_sequence

    def swapped(model: Any) -> Any:
        seq = real(model)
        a, b = seq.stages
        return sequence_h5.Sequence(
            staged=seq.staged, stages=(b, a), changes=seq.changes)

    monkeypatch.setattr(sequence_h5, "derive_sequence", swapped)
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "mut2.h5")
    with pytest.raises(AssertionError, match="order"):
        _check_single_source(path)


# ---------------------------------------------------------------------------
# Seams: reload, partial writes, overflow, unknown attrs, hashes
# ---------------------------------------------------------------------------


def test_rewrite_replaces_never_appends(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")
    before = _nan_safe(_read(path))
    with h5py.File(path, "r+") as f:
        write_sequence_into(f)
        write_sequence_into(f)
        assert "_sequence_partial" not in f
    assert _nan_safe(_read(path)) == before


def test_opensees_model_rewrite_rederives(tmp_path: Path) -> None:
    from apeGmsh.opensees.opensees_model import OpenSeesModel
    from tests.opensees.h5._opensees_model_fixtures import (
        build_simple_frame_fem,
    )

    # A real FEMData (the rewrite reads the neutral zone back).
    ops = apeSees(build_simple_frame_fem())
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Cols", dofs=(1, 1, 1, 0, 0, 0), name="pins")
    with ops.stage(name="gravity") as s:
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts, name="dead") as p:
            p.load(pg="Cols", forces=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0))
        s.analysis(**_chain(ops, False))
        s.run(n_increments=2)
    path = _write(ops, tmp_path, "staged.h5")
    out = tmp_path / "rewritten.h5"
    OpenSeesModel.from_h5(str(path)).to_h5(str(out))
    _a1, s1, c1 = _nan_safe(_read(path))
    a2, s2, c2 = _nan_safe(_read(out))
    assert (s1, c1) == (s2, c2)
    assert [(r["what"], r["name"]) for r in c2] == [
        ("fix", "pins"), ("pattern", "dead")]
    with h5py.File(out, "r") as f:
        assert a2["model_hash"] == read_stored_lineage(f["meta"])[1]


def test_failed_write_leaves_no_zone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")

    def boom(g: Any, seq: Any) -> None:
        g.create_group("stages")
        raise RuntimeError("disk full")

    monkeypatch.setattr(sequence_h5, "_write_tables", boom)
    with h5py.File(path, "r+") as f:
        with pytest.raises(RuntimeError, match="disk full"):
            write_sequence_into(f)
        assert "sequence" not in f and "_sequence_partial" not in f
        assert SEQUENCE_KEY not in f["meta"].attrs
        assert "opensees/stages/stage_001" in f


def test_unknown_chain_attr_refuses_before_writing(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")
    with h5py.File(path, "r+") as f:
        f["opensees/stages/stage_001/analysis"].attrs["strategy"] = "x"
        with pytest.raises(SequenceError, match="strategy"):
            write_sequence_into(f)
        assert "sequence" not in f and "_sequence_partial" not in f
        assert SEQUENCE_KEY not in f["meta"].attrs


def test_int32_overflow_refuses(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")
    with h5py.File(path, "r+") as f:
        f["opensees/stages/stage_000"].attrs["analyze_steps"] = np.int64(
            2**31)
        with pytest.raises(SequenceOverflowError, match="int32"):
            write_sequence_into(f)
        assert "sequence" not in f
    with pytest.raises(OverflowError):
        sequence_h5._int32(-(2**31) - 1, "x")
    assert sequence_h5._int32(2**31 - 1, "x") == 2**31 - 1


def test_zone_changes_no_hash(tmp_path: Path) -> None:
    from tests.opensees.h5._opensees_model_fixtures import (
        build_simple_frame_h5,
    )

    path, _fem = build_simple_frame_h5(tmp_path)
    stripped = tmp_path / "stripped.h5"
    shutil.copy(path, stripped)
    with h5py.File(stripped, "r+") as f:
        assert "sequence" in f
        del f["sequence"]
    hashes = []
    for p in (path, stripped):
        with h5py.File(p, "r") as f:
            fem_hash = compute_fem_hash(f)
            hashes.append((fem_hash, compute_model_hash(fem_hash, f["opensees"])))
            stored = read_stored_lineage(f["meta"])
            assert (stored[0], stored[1]) == hashes[-1]
    assert hashes[0] == hashes[1]


def test_zone_key_registered() -> None:
    assert _ZONE_KEY[SEQUENCE] == SEQUENCE_KEY == "sequence_schema_version"
    assert reader_version(SEQUENCE) == reader_floor(SEQUENCE) == (
        SchemaVersion(1, 0, 0))
    # It postdates the envelope: the legacy key never stands in for it.
    assert read_zone_version({"schema_version": "2.6.0"}, SEQUENCE) is None


# ---------------------------------------------------------------------------
# Locks (V3a AST lock + the folded V3b vocabulary lock)
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = {
    "__future__", "json", "math", "dataclasses", "typing", "numpy", "h5py",
    "..emitter.h5_reader", ".schema_version", ".lineage",
}
#: The reader methods the deriver may call.
_READER_API = {
    "stages", "analysis", "declarations", "regions", "commands",
    "patterns", "recorders", "initial_stress",
}
#: The functions that see the H5Model (the deriver proper).
_DERIVER_FUNCS = {"derive_sequence", "_global_changes", "_stage_changes"}


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text(encoding="utf-8"))


def test_deriver_imports_no_record_or_primitive_module() -> None:
    seen = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            seen |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            seen.add("." * node.level + (node.module or ""))
    assert seen <= _ALLOWED_IMPORTS, sorted(seen - _ALLOWED_IMPORTS)
    for name in seen:
        assert not re.search(r"typed_records|primitives|records|build", name)


def test_deriver_reads_only_through_reader_methods() -> None:
    tree = _tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # No reach past the reader into its raw handle.
            assert node.attr not in {"handle", "_f", "_neutral"}, node.attr
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for fname in _DERIVER_FUNCS:
        fn = funcs[fname]
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "model"):
                assert node.func.attr in _READER_API, (fname, node.func.attr)
            # The deriver never indexes an h5py object.
            if isinstance(node, ast.Subscript) and isinstance(
                    node.value, ast.Name):
                assert node.value.id not in {"f", "model"}, fname


def test_change_vocabulary_is_the_verbs_seq_pairs() -> None:
    verb_pairs = {(v.seq_op, v.seq_what) for v in VERBS.values()
                  if (v.seq_op, v.seq_what) != ("", "")}
    assert CHANGE_KINDS == verb_pairs | {("activate", "group")}
    assert {op for op, _ in CHANGE_KINDS} <= SEQ_OPS - {""}
    assert {w for _, w in CHANGE_KINDS} == SEQ_WHATS - {""}


def test_stage_columns_are_the_ratified_set(tmp_path: Path) -> None:
    path = _write(build_model("two_column_frame", "staged", "h5"),
                  tmp_path, "staged.h5")
    with h5py.File(path, "r") as f:
        assert set(f["sequence/stages"]) == {
            "order", "name", *CHAIN_TOKENS, *CHAIN_ARGS, "steps", "dt",
            "set_time", "domain_change", "source"}
        assert set(f["sequence/changes"]) == {
            "stage", "op", "what", "name", "count", "source"}
        assert set(f["sequence"].attrs) == {
            "derived_from", "model_hash", "staged"}


def test_partitioned_runtime_fallback_shows_the_primary(tmp_path: Path) -> None:
    """A partitioned deck archives a runtime fallback beside the numberer
    and system (``H5Emitter.parallel_runtime_fallback_*``, ADR 0027
    INV-5); the zone shows the primary token and does not refuse.  (The
    suite's ranked-assembly archive hit this before the attrs were
    allowed.)"""
    path = _write(build_model("two_column_frame", "flat", "h5"),
                  tmp_path, "flat.h5")
    with h5py.File(path, "r+") as f:
        a = f["opensees/analysis"].attrs
        a["numberer"] = "ParallelPlain"
        a["numberer_runtime_fallback"] = "RCM"
        a["system"] = "Mumps"
        a["system_runtime_fallback"] = "UmfPack"
        write_sequence_into(f)
    _attrs, stages, _changes = _read(path)
    assert (stages[0]["numberer"], stages[0]["system"]) == (
        "ParallelPlain", "Mumps")
    _check_single_source(path)
