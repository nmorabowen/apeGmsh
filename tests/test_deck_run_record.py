"""#2210 — the deck run record, writer side (route (c), the ratified spec).

A deck ``ops.py`` / ``ops.tcl`` emits with ``analyze_steps`` writes
``<recorder file>.run.json`` beside each MPCO recorder's output, as its last
action, once its analyze loop has returned. The reader side (the five checks
in ``convert_mpco``) is ``tests/test_results_mpco_to_ladruno.py``; the
out-of-process runs on a real engine are
``tests/opensees/integration_ladruno/test_deck_run_record_live.py``.

Oracles, none of which needs an OpenSees engine:

* **the emitted code itself runs**: the record block of each deck is cut
  out of the file and executed, the Python one with ``exec`` against a
  stand-in ``ops`` and the Tcl one in Python's own Tcl interpreter
  (``tkinter.Tcl``), and the JSON each writes is read back;
* **cross-check**: the record's ``model_hash`` / ``fem_hash`` equal
  ``/meta/lineage`` of an ``ops.h5()`` written separately, and the Python
  and Tcl records agree field for field;
* **the spec's 12 fields** in order, with the engine stamped by the G10
  rule (a 40-hex ``ladrunoBuild`` is the fork, anything else stock with an
  ``unknown`` build);
* **write rule**: no record lines without ``analyze_steps`` or an MPCO
  recorder, the lines follow the loop and end the deck, rank 0 writes after
  a barrier, no ``.tmp`` is left, and ``model.h5`` is byte-identical across
  an export (F1/F3 of #2224's review).
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
import time
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
from apeGmsh.opensees.emitter.py import (
    DECK_RUN_SCHEMA,
    DECK_RUN_SUFFIX,
    deck_run_body,
)

#: The ratified spec's fields, in order.
RUN_FIELDS = ("schema", "model_hash", "fem_hash", "recorder_file",
              "recorder_size", "recorder_mtime", "analysis", "analyze_steps",
              "steps_completed", "dt", "opensees_backend",
              "opensees_build_sha", "writer", "completed_iso")
MARK = "# apeGmsh deck run record: "
SHA = "0123456789abcdef0123456789abcdef01234567"

# A two-node truss under a static ramp; tip displacement P L / (E A).
L, E, A, P = 2.0, 2.0e11, 1.0e-4, 1.0e4
STEPS = 4
TIP = P * L / (E * A)


def truss_fem() -> FEMData:
    node_ids = np.array([1, 2], dtype=np.int64)
    coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0]])
    info = make_type_info(code=1, gmsh_name="Line 2", dim=1, order=1, npe=2,
                          count=1)
    group = ElementGroup(element_type=info, ids=np.array([1], dtype=np.int64),
                         connectivity=np.array([[1, 2]], dtype=np.int64))
    pg = {
        (1, 1): {"name": "Bar", "node_ids": node_ids, "node_coords": coords,
                 "element_ids": group.ids},
        (0, 2): {"name": "Base", "node_ids": node_ids[:1],
                 "node_coords": coords[:1]},
        (0, 3): {"name": "Tip", "node_ids": node_ids[1:],
                 "node_coords": coords[1:]},
    }
    return FEMData(
        nodes=NodeComposite(node_ids=node_ids, node_coords=coords,
                            physical=PhysicalGroupSet(pg), labels=LabelSet({})),
        elements=ElementComposite(groups={1: group},
                                  physical=PhysicalGroupSet(pg),
                                  labels=LabelSet({})),
        info=MeshInfo(n_nodes=2, n_elems=1, bandwidth=2, types=[info]),
        model_name="deck_run_record_truss",
    )


def truss_bridge(mpco: "Path | None", *, analysis: str = "Static",
                 artifacts: bool = True) -> Any:
    """The truss on a bridge, unstaged, with one ``analysis`` declaration
    and (with ``mpco``) a bridge MPCO recorder on the nodal displacements."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(truss_fem(), default_orientation=None, _artifacts=artifacts)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=E)
    ops.element.Truss(pg="Bar", A=A, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    ops.fix(pg="Tip", dofs=(0, 1))
    if mpco is not None:
        ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
        pat.load(pg="Tip", forces=(P, 0.0))
    ops.test.NormDispIncr(tol=1e-12, max_iter=10)
    ops.algorithm.Linear()
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    if analysis == "Static":
        ops.integrator.LoadControl(dlam=1.0 / STEPS)
        ops.analysis.Static()
    else:
        ops.integrator.Newmark(gamma=0.5, beta=0.25)
        ops.analysis.Transient()
    return ops


def lineage(model: Path) -> "tuple[str, str]":
    with h5py.File(model, "r") as f:
        lin = f["meta/lineage"].attrs
        return str(lin["model_hash"]), str(lin["fem_hash"])


def record_block(deck: Path) -> str:
    """The deck's record lines: from the marker to the end of the file."""
    text = deck.read_text(encoding="utf-8")
    assert text.count(MARK) == 1, text[-2000:]
    return text[text.index(MARK):]


#: Bytes a stand-in recorder adds to its file when it closes.
FLUSH = 100


def open_files(paths: "list[Path]") -> None:
    """The ``.mpco`` files as an open recorder leaves them, before its
    close flushes the last ``FLUSH`` bytes."""
    for p in paths:
        p.write_bytes(b"\x89HDF" + b"\0" * 2000)


class FakeOps:
    """The ``ops`` a record block calls: ``ladrunoBuild`` answers ``build``
    (absent when ``None``); ``barrier`` counts its calls; ``remove`` closes
    the recorders, adding ``FLUSH`` bytes to each of ``files``; ``getNP``
    answers ``np``."""

    def __init__(self, build: "str | None", files: "list[Path]",
                 np: int = 1) -> None:
        self.barriers = 0
        self.closed = 0
        self._files, self._np = files, np
        if build is not None:
            self.ladrunoBuild = lambda: build

    def barrier(self) -> None:
        self.barriers += 1

    def getNP(self) -> int:
        return self._np

    def remove(self, what: str) -> None:
        assert what == "recorders"
        self.closed += 1
        for p in self._files:
            with open(p, "ab") as f:
                f.write(b"x" * FLUSH)


def run_py(block: str, *, build: "str | None", done: int, pid: int = 0,
           files: "list[Path] | None" = None, np: int = 1) -> FakeOps:
    files = files or []
    open_files(files)
    ops = FakeOps(build, files, np)
    exec(compile(block, "<record>", "exec"),
         {"ops": ops, "_apesees_i": done - 1, "getPID": lambda: pid})
    return ops


def tcl() -> Any:
    tkinter = pytest.importorskip("tkinter")
    error: "Exception | None" = None
    for _ in range(5):  # Windows sometimes fails one read of init.tcl
        try:
            interp = tkinter.Tcl()
        except tkinter.TclError as e:
            error = e
            time.sleep(0.2)
            continue
        # The console has no stdout under pytest on Windows: a message the
        # deck prints is collected in ``::_apesees_out`` instead; ``puts``
        # to a channel (the record file) is the real one.
        interp.eval("rename puts _apesees_puts; set ::_apesees_out {}")
        interp.eval(
            "proc puts {args} { if {[llength $args] == 1 || ([llength $args]"
            " == 2 && [lindex $args 0] eq {-nonewline})} { lappend "
            "::_apesees_out [lindex $args end] } else { _apesees_puts {*}$args"
            " } }")
        return interp
    pytest.skip(f"tkinter.Tcl() unavailable: {error}")


def run_tcl(block: str, *, build: "str | None", done: int, pid: int = 0,
            files: "list[Path] | None" = None, np: int = 1) -> Any:
    files = files or []
    open_files(files)
    interp = tcl()
    interp.eval(f"set _apesees_i {done}")
    interp.eval("set _apesees_barriers 0")
    interp.eval("set _apesees_closed 0")
    interp.call("set", "::_apesees_files", [str(p) for p in files])
    interp.eval(f"proc getPID {{}} {{ return {pid} }}")
    interp.eval(f"proc getNP {{}} {{ return {np} }}")
    interp.eval("proc barrier {} { incr ::_apesees_barriers }")
    interp.eval(
        "proc remove {what} { incr ::_apesees_closed; "
        "foreach p $::_apesees_files { set f [open $p a]; "
        f"fconfigure $f -translation binary; puts -nonewline $f "
        f"[string repeat x {FLUSH}]; close $f }} }}")
    if build is not None:
        interp.eval(f"proc ladrunoBuild {{}} {{ return {build} }}")
    interp.eval(block)
    return interp


def read_record(mpco: Path, parts: "list[Path] | None" = None,
                ) -> "dict[str, Any]":
    """The record beside ``mpco``, holding the spec's fields and the files
    as they are now, after their close (``parts`` for a set)."""
    path = mpco.with_name(mpco.name + DECK_RUN_SUFFIX)
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(rec) == RUN_FIELDS
    stats = [p.stat() for p in (parts or [mpco])]
    sizes, mtimes = rec["recorder_size"], rec["recorder_mtime"]
    if parts is None:
        sizes, mtimes = [sizes], [mtimes]
    assert sizes == [st.st_size for st in stats]
    assert all(st.st_size == 2004 + FLUSH for st in stats)  # after the close
    for m, st in zip(mtimes, stats):
        assert m == (st.st_mtime_ns // 10**9 if isinstance(m, int)
                     else st.st_mtime)
    return rec


@pytest.fixture
def exported(tmp_path: Path) -> "dict[str, Path]":
    """``model.h5``, then both decks of the same bridge, ``analyze_steps``."""
    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    bridge.py(str(tmp_path / "deck.py"), analyze_steps=STEPS, progress=False)
    bridge.tcl(str(tmp_path / "deck.tcl"), analyze_steps=STEPS, progress=False)
    return {"mpco": mpco, "model": model, "py": tmp_path / "deck.py",
            "tcl": tmp_path / "deck.tcl"}


# ---------------------------------------------------------------------------
# What the record says
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build, backend, sha", [
    (None, "stock", "unknown"), (SHA, "ladruno", SHA),
    ("not-a-sha", "stock", "unknown"), (SHA.upper(), "stock", "unknown"),
], ids=["stock", "fork", "garbage", "uppercase"])
@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_the_record_holds_the_spec_fields(
    exported, dialect, build, backend, sha,
) -> None:
    block = record_block(exported[dialect])
    before = time.time()
    if dialect == "py":
        run_py(block, build=build, done=STEPS, files=[exported["mpco"]])
    else:
        run_tcl(block, build=build, done=STEPS, files=[exported["mpco"]])
    rec = read_record(exported["mpco"])
    model_hash, fem_hash = lineage(exported["model"])
    assert rec["schema"] == DECK_RUN_SCHEMA == "apegmsh.deck_run/2"
    assert (rec["model_hash"], rec["fem_hash"]) == (model_hash, fem_hash)
    assert rec["recorder_file"] == "deck.mpco"
    assert rec["analysis"] == "Static"
    assert rec["analyze_steps"] == rec["steps_completed"] == STEPS
    assert rec["dt"] is None
    assert (rec["opensees_backend"], rec["opensees_build_sha"]) == (backend, sha)
    import apeGmsh

    assert rec["writer"] == f"apeGmsh deck {apeGmsh.__version__}"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
                        rec["completed_iso"])
    stamped = calendar.timegm(time.strptime(rec["completed_iso"],
                                            "%Y-%m-%dT%H:%M:%SZ"))
    assert before - 2 <= stamped <= time.time() + 2


def test_the_two_decks_write_the_same_record(tmp_path) -> None:
    """A transient deck with a step size: the Python and the Tcl record are
    equal field for field (``completed_iso`` aside)."""
    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco, analysis="Transient")
    records = []
    for dialect in ("py", "tcl"):
        deck = tmp_path / f"deck.{dialect}"
        getattr(bridge, dialect)(str(deck), analyze_steps=3, analyze_dt=0.01,
                                 progress=False)
        block = record_block(deck)
        if dialect == "py":
            run_py(block, build=SHA, done=3, files=[mpco])
        else:
            run_tcl(block, build=SHA, done=3, files=[mpco])
        records.append(read_record(mpco))
    # Each runtime's own mtime: Python's float, Tcl's whole seconds.
    assert isinstance(records[0]["recorder_mtime"], float)
    assert isinstance(records[1]["recorder_mtime"], int)
    py, tc = ({k: v for k, v in r.items()
               if k not in ("completed_iso", "recorder_mtime")}
              for r in records)
    assert py == tc
    assert (py["analysis"], py["dt"], py["steps_completed"]) == ("Transient", 0.01, 3)


def test_the_steps_completed_are_the_loops_own_count(exported) -> None:
    """``steps_completed`` is read from the loop's counter when the record is
    written, not copied from ``analyze_steps``."""
    run_py(record_block(exported["py"]), build=None, done=STEPS - 1,
           files=[exported["mpco"]])
    assert read_record(exported["mpco"])["steps_completed"] == STEPS - 1
    run_tcl(record_block(exported["tcl"]), build=None, done=STEPS - 2,
            files=[exported["mpco"]])
    assert read_record(exported["mpco"])["steps_completed"] == STEPS - 2


def test_a_tcl_path_tcl_would_substitute_is_written_verbatim(tmp_path) -> None:
    """Every character Tcl substitutes or counts survives the quoting."""
    where = tmp_path / "we{ir}d $x [y]"
    where.mkdir()
    mpco = where / "deck.mpco"
    deck = tmp_path / "deck.tcl"
    truss_bridge(mpco).tcl(str(deck), analyze_steps=2, progress=False)
    run_tcl(record_block(deck), build=None, done=2, files=[mpco])
    assert read_record(mpco)["recorder_file"] == "deck.mpco"


# ---------------------------------------------------------------------------
# When it is written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_the_record_follows_the_loop_and_ends_the_deck(exported, dialect) -> None:
    text = exported[dialect].read_text(encoding="utf-8")
    loop = ("for _apesees_i in range(4):" if dialect == "py"
            else "for {set _apesees_i 0} {$_apesees_i < 4} {incr _apesees_i} {")
    assert text.index(loop) < text.index(MARK)
    block = record_block(exported[dialect]).rstrip().splitlines()
    close = "ops.remove('recorders')" if dialect == "py" else "remove recorders"
    assert block[1] == close  # the recorders close before anything is stated
    rename = "_apesees_os.replace(" if dialect == "py" else "file rename -force "
    assert any(rename in ln for ln in block[-8:])  # the rename ends the deck


def test_a_torn_write_leaves_no_record(exported, monkeypatch, capsys) -> None:
    """Atomic: the record is written to ``<name>.tmp`` and renamed, so a
    write that fails half way leaves no ``.run.json`` for a reader to see."""
    def torn(obj: Any, f: Any, *args: Any, **kw: Any) -> None:
        f.write('{"schema": "apeg')
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", torn)
    run_py(record_block(exported["py"]), build=None, done=STEPS,
           files=[exported["mpco"]])  # the run is not failed by it
    assert "record was not written: disk full" in capsys.readouterr().out
    final = exported["mpco"].with_name("deck.mpco" + DECK_RUN_SUFFIX)
    assert not final.exists()
    assert final.with_name(final.name + ".tmp").exists()
    # The Tcl block takes the same two steps.
    lines = record_block(exported["tcl"]).splitlines()
    (opened,) = [ln for ln in lines
                 if ln.strip().startswith("set _apesees_f [open ")]
    assert opened.endswith('.run.json.tmp" w]')
    (renamed,) = [ln for ln in lines if "file rename -force " in ln]
    assert '.run.json.tmp" "' in renamed
    assert renamed.endswith('.run.json"} _apesees_rerr]} { break }')


#: Runs a py deck in a fresh interpreter with every OpenSees call a no-op
#: that succeeds and every rename refused, then prints the run's marker.
_REFUSING_RUNNER = """
import os, sys, types
def refuse(*a, **k):
    raise PermissionError("held by another process")
os.replace = refuse
ops = types.ModuleType("openseespy.opensees")
ops.__getattr__ = lambda name: (lambda *a, **k: 0)
pkg = types.ModuleType("openseespy")
pkg.opensees = ops
sys.modules["openseespy"], sys.modules["openseespy.opensees"] = pkg, ops
deck = sys.argv[1]
exec(compile(open(deck, encoding="utf-8").read(), deck, "exec"),
     {"__name__": "__main__"})
print("DECK-DONE")
"""


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_a_refused_rename_never_fails_a_finished_run(tmp_path, dialect) -> None:
    """Fix A: the whole deck runs, its loop succeeds, and the record's
    rename is refused every time (a handle Defender or the indexer holds).
    The rename is retried, then the deck prints that the record was not
    written and still ends normally: exit 0 and its marker (py, in a fresh
    process), or no Tcl error and the marker (Tcl). No record is left."""
    import subprocess
    import sys

    mpco = tmp_path / "deck.mpco"
    deck = tmp_path / f"deck.{dialect}"
    getattr(truss_bridge(mpco), dialect)(str(deck), analyze_steps=STEPS,
                                         progress=False)
    open_files([mpco])  # what the recorder wrote
    if dialect == "py":
        runner = tmp_path / "runner.py"
        runner.write_text(_REFUSING_RUNNER, encoding="utf-8")
        proc = subprocess.run([sys.executable, str(runner), str(deck)],
                              cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr[-2000:]
        out = proc.stdout
    else:
        interp = tcl()
        interp.eval("proc unknown {args} { return 0 }")
        interp.eval("rename file _apesees_real_file; set ::_apesees_tries 0")
        interp.eval("proc file {cmd args} { if {$cmd eq {rename}} "
                    "{ incr ::_apesees_tries; error {permission denied} }; "
                    "uplevel 1 [list _apesees_real_file $cmd {*}$args] }")
        interp.eval(deck.read_text(encoding="utf-8") + "\nputs DECK-DONE\n")
        assert int(interp.eval("set ::_apesees_tries")) == 10
        out = "\n".join(interp.splitlist(interp.eval("set ::_apesees_out")))
    assert "the deck run record was not written" in out
    assert out.rstrip().endswith("DECK-DONE")
    assert not mpco.with_name("deck.mpco" + DECK_RUN_SUFFIX).exists()


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_a_record_writing_deck_removes_an_old_record_before_its_recorder(
    tmp_path, dialect,
) -> None:
    """Ruling (a): the unlink comes before the recorder opens, and only in
    a deck that writes a record (other decks are unchanged)."""
    mpco = tmp_path / "deck.mpco"
    deck, plain = tmp_path / f"deck.{dialect}", tmp_path / f"plain.{dialect}"
    bridge = truss_bridge(mpco)
    getattr(bridge, dialect)(str(deck), analyze_steps=STEPS, progress=False)
    getattr(bridge, dialect)(str(plain), progress=False)
    text = deck.read_text(encoding="utf-8")
    unlink = ("__import__('os').remove(" if dialect == "py"
              else "file delete -force -- ")
    opens = "ops.recorder('mpco'" if dialect == "py" else "recorder mpco "
    assert text.count(unlink) == 1
    assert text.index(unlink) < text.index(opens) < text.index(MARK)
    assert unlink not in plain.read_text(encoding="utf-8")


class FailingOps:
    """Every OpenSees call succeeds, except ``analyze``, which fails."""

    def __getattr__(self, name: str) -> Any:
        return (lambda *a, **k: -3) if name == "analyze" else (
            lambda *a, **k: 0)


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_a_rerun_whose_loop_fails_leaves_no_record(tmp_path, dialect) -> None:
    """The deck ran once and recorded; a re-run whose loop fails removes
    that record before its recorder opens and writes none: a reader then
    refuses on check 1."""
    mpco = tmp_path / "deck.mpco"
    deck = tmp_path / f"deck.{dialect}"
    getattr(truss_bridge(mpco), dialect)(
        str(deck), analyze_steps=STEPS, progress=False)
    old = mpco.with_name("deck.mpco" + DECK_RUN_SUFFIX)
    old.write_text("{}", encoding="utf-8")  # the earlier run's record
    text = deck.read_text(encoding="utf-8")
    if dialect == "py":
        src = text.replace("import openseespy.opensees as ops", "")
        with pytest.raises(SystemExit, match="analyze FAILED"):
            exec(compile(src, str(deck), "exec"), {"ops": FailingOps()})
    else:
        interp = tcl()
        interp.eval("proc unknown {args} { return 0 }")
        interp.eval("proc analyze {args} { return -3 }")
        import tkinter

        with pytest.raises(tkinter.TclError, match="analyze FAILED"):
            interp.eval(text)
    assert not old.exists()
    assert not old.with_name(old.name + ".tmp").exists()


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_no_record_without_analyze_steps_or_an_mpco_recorder(
    tmp_path, dialect,
) -> None:
    """No loop of its own, or nothing a record would describe: no lines."""
    for name, bridge, steps in (
        ("no_steps", truss_bridge(tmp_path / "a.mpco"), None),
        ("no_mpco", truss_bridge(None), STEPS),
    ):
        deck = tmp_path / f"{name}.{dialect}"
        getattr(bridge, dialect)(str(deck), analyze_steps=steps, progress=False)
        text = deck.read_text(encoding="utf-8")
        assert MARK not in text and DECK_RUN_SUFFIX not in text


def test_an_export_writes_no_record_and_leaves_model_h5_alone(tmp_path) -> None:
    """F1/F3 of #2224's review: the export writes the deck only. No record
    exists until the deck runs, ``model.h5`` is byte-identical, and a fresh
    archive has the same ``model_hash``."""
    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    bridge.py(str(tmp_path / "deck.py"), analyze_steps=STEPS, progress=False)
    bridge.tcl(str(tmp_path / "deck.tcl"), analyze_steps=STEPS, progress=False,
               stream=True)
    assert hashlib.sha256(model.read_bytes()).hexdigest() == sha
    assert list(tmp_path.glob("*.run.json*")) == []
    bridge.h5(str(tmp_path / "again.h5"))
    assert lineage(tmp_path / "again.h5") == lineage(model)
    assert MARK in (tmp_path / "deck.tcl").read_text(encoding="utf-8")


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_a_partitioned_deck_writes_on_rank_0_after_a_barrier(
    tmp_path, dialect,
) -> None:
    """With a lineage already cached for these declarations (here from a
    serial export of the same bridge), a partitioned deck's record lines
    close every rank's recorders, then a barrier, then rank 0 writes."""
    mpco = tmp_path / "run.mpco"
    parts = [tmp_path / f"run.part-{k}.mpco" for k in (0, 1)]
    bridge = truss_bridge(mpco)
    bridge.py(str(tmp_path / "serial.py"), analyze_steps=STEPS, progress=False)
    body = "\n".join(deck_run_body(
        dialect, bridge.h5, [str(mpco)], "Static", STEPS, None,
        partitioned=True, bridge=bridge)) + "\n"
    for pid in (1, 0):
        if dialect == "py":
            barriers = run_py(body, build=None, done=STEPS, pid=pid,
                              files=parts, np=2).barriers
        else:
            interp = run_tcl(body, build=None, done=STEPS, pid=pid,
                             files=parts, np=2)
            barriers = int(interp.eval("set _apesees_barriers"))
        assert barriers == 1
        written = mpco.with_name("run.mpco" + DECK_RUN_SUFFIX).exists()
        # Windows only: another process (Defender, the indexer) can
        # hold the fresh .tmp; the deck then prints that it wrote no
        # record (fix A) instead of failing the run.
        assert written == (pid == 0), (
            "" if dialect == "py" else interp.eval("set ::_apesees_out"))
    assert read_record(mpco, parts)["recorder_file"] == "run.mpco"


def _count_archives(monkeypatch: Any) -> "list[Any]":
    """Every path ``apeSees._write_h5`` is asked to write, in order."""
    from apeGmsh.opensees.apesees import apeSees

    seen: "list[Any]" = []
    real = apeSees._write_h5

    def spy(self: Any, path: Any, bm: Any, **kw: Any) -> None:
        seen.append(path)
        real(self, path, bm, **kw)

    monkeypatch.setattr(apeSees, "_write_h5", spy)
    return seen


def test_the_record_reuses_the_automatic_archive(tmp_path, monkeypatch) -> None:
    """With the automatic ``model.h5`` on, an export composes one archive,
    the automatic one, and the record reads its lineage. A second deck of
    the same declarations composes none; a changed declaration is a new
    digest, and the record follows it."""
    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco)
    seen = _count_archives(monkeypatch)
    bridge.py(str(tmp_path / "a.py"), analyze_steps=STEPS, progress=False)
    assert len(seen) == 1 and isinstance(seen[0], str)  # the automatic write
    bridge.tcl(str(tmp_path / "a.tcl"), analyze_steps=STEPS, progress=False)
    assert len(seen) == 1
    bridge.uniaxialMaterial.ElasticMaterial(E=1.0)  # a new declaration
    bridge.py(str(tmp_path / "b.py"), analyze_steps=STEPS, progress=False)
    assert len(seen) == 2
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    for deck in ("a.py", "a.tcl", "b.py"):
        if deck.endswith(".py"):
            run_py(record_block(tmp_path / deck), build=None, done=STEPS,
                   files=[mpco])
        else:
            run_tcl(record_block(tmp_path / deck), build=None, done=STEPS,
                    files=[mpco])
        rec = read_record(mpco)
        if deck == "b.py":
            assert (rec["model_hash"], rec["fem_hash"]) == lineage(model)
        else:
            assert rec["model_hash"] != lineage(model)[0]


def test_without_the_automatic_archive_it_is_composed_in_memory(
    tmp_path, monkeypatch,
) -> None:
    """``_artifacts=False``: the archive is composed once, into memory (no
    file), and its lineage is the user's ``ops.h5()`` lineage; a second
    export reuses it."""
    import io

    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco, artifacts=False)
    seen = _count_archives(monkeypatch)
    bridge.py(str(tmp_path / "d.py"), analyze_steps=STEPS, progress=False)
    bridge.tcl(str(tmp_path / "d.tcl"), analyze_steps=STEPS, progress=False)
    assert len(seen) == 1 and isinstance(seen[0], io.BytesIO)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    run_py(record_block(tmp_path / "d.py"), build=None, done=STEPS, files=[mpco])
    assert (read_record(mpco)["model_hash"],
            read_record(mpco)["fem_hash"]) == lineage(model)


def _partitioned_bridge(mpco: Path) -> Any:
    """A bridge on a two-rank snapshot (the emit-cost bench's builder), so
    its decks carry ``getPID`` blocks."""
    from typing import cast

    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees.material.nd import ElasticIsotropic
    from apeGmsh.opensees.material.uniaxial import ElasticMaterial
    from tests.benchmarks.test_cross_rank_constraint_cost import (
        _build_embedded_fem,
    )

    fem, host_pg, rebar_pg = _build_embedded_fem(8, 2, "hex_host_line_embed")
    ops = apeSees(cast("object", fem), _artifacts=False)
    ops.model(ndm=3, ndf=3)
    nd, ux = ElasticIsotropic(E=2.0e10, nu=0.2), ElasticMaterial(E=2.0e11)
    ops.register(nd)
    ops.register(ux)
    ops.element.stdBrick(pg=host_pg, material=nd)
    ops.element.Truss(pg=rebar_pg, A=1e-4, material=ux)
    ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _emit_partitioned(tmp_path: Path, dialect: str, monkeypatch: Any,
                      ) -> "tuple[Path, list[str]]":
    """Emit the two-rank bridge deck with ``analyze_steps``; composing an
    archive fails the test, and the run-record warnings are returned."""
    import warnings

    from apeGmsh.opensees.emitter import py as pym

    def no_compose(self: Any, *a: Any, **k: Any) -> None:
        raise AssertionError("a partitioned export composed an archive")

    monkeypatch.setattr(pym._MemoryArchive, "__init__", no_compose)
    mpco = tmp_path / "run.mpco"
    deck = tmp_path / f"deck.{dialect}"
    bridge = _partitioned_bridge(mpco)
    seen = _count_archives(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(bridge, dialect)(str(deck), analyze_steps=STEPS,
                                 progress=False)
    assert seen == []  # nor through the bridge
    ours = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "run record" in str(w.message)]
    return deck, ours


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_a_partitioned_bridge_deck_skips_its_record_without_composing(
    tmp_path, dialect, monkeypatch,
) -> None:
    """T's option (2): a partitioned model with no cached lineage composes
    no archive and emits no record lines; it warns once, naming why. The
    old record is still removed before the recorder opens."""
    deck, ours = _emit_partitioned(tmp_path, dialect, monkeypatch)
    assert len(ours) == 1, ours
    assert "partitioned" in ours[0] and "no lineage" in ours[0]
    assert "Results.from_mpco" in ours[0]
    text = deck.read_text(encoding="utf-8")
    assert "getPID" in text  # a partitioned deck
    assert MARK not in text
    unlink = ("__import__('os').remove(" if dialect == "py"
              else "file delete -force -- ")
    assert unlink in text


def test_a_skipped_partitioned_record_reads_with_from_mpco(
    tmp_path, monkeypatch,
) -> None:
    """The run of such a deck leaves its parts and no record: read_deck_mpco
    finds none (check 1), warns once and reads with ``from_mpco``."""
    import shutil
    import warnings

    from apeGmsh.results.live import DefaultRouteWarning, read_deck_mpco
    from tests.test_results_mpco_to_ladruno import (
        SPRINGS,
        _bridge_copy,
        _unstaged_springs_model,
    )

    deck, _ = _emit_partitioned(tmp_path, "tcl", monkeypatch)
    model = _unstaged_springs_model(tmp_path / "model.h5", ["static"])
    one = _bridge_copy(SPRINGS, model, tmp_path / "one.mpco")
    for k in (0, 1):  # the parts the deck's MPCO recorder writes
        shutil.copy(one, tmp_path / f"run.part-{k}.mpco")
    assert not (tmp_path / ("run.mpco" + DECK_RUN_SUFFIX)).exists()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = read_deck_mpco(tmp_path / "run.part-0.mpco", model_h5=model)
    res.close()
    msgs = [str(w.message) for w in caught if w.category is DefaultRouteWarning]
    assert len(msgs) == 1 and "check 1 (the record)" in msgs[0]
    assert sorted(p.name for p in tmp_path.glob("*.ladruno")) == []


@pytest.mark.filterwarnings("ignore:model.h5 not written")  # the auto-write's
def test_an_archive_that_cannot_be_composed_warns_and_writes_no_record(
    tmp_path, monkeypatch,
) -> None:
    """Fail loud at the seam: the deck still emits, without a record, and
    the reason is named once."""
    from apeGmsh.opensees.apesees import apeSees

    def boom(self: Any, path: str, bm: Any, **kw: Any) -> None:
        raise OSError("disk full")

    bridge = truss_bridge(tmp_path / "deck.mpco")
    monkeypatch.setattr(apeSees, "_write_h5", boom)
    with pytest.warns(RuntimeWarning, match=r"no run record.*disk full"):
        bridge.py(str(tmp_path / "deck.py"), analyze_steps=STEPS,
                  progress=False)
    text = (tmp_path / "deck.py").read_text(encoding="utf-8")
    assert "range(4)" in text and MARK not in text


def test_the_in_process_emitter_has_no_record() -> None:
    """``run()`` / ``emit_default`` / raw ``opspy.analyze`` drive the live
    emitter, which cannot write one."""
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    assert not hasattr(LiveOpsEmitter, "_deck_run_record")


def test_the_reader_reads_the_format_the_writer_writes() -> None:
    from apeGmsh.opensees.emitter.py import deck_run_fields
    from apeGmsh.results.readers import _ladruno_from_mpco as conv

    assert conv._RUN_SCHEMA == DECK_RUN_SCHEMA
    assert conv._RUN_SUFFIX == DECK_RUN_SUFFIX
    fields = deck_run_fields(("a", "b"), "x.mpco", "Static", 1, None)
    assert tuple(fields) == tuple(conv._RUN_FIELDS) == RUN_FIELDS
