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
  (``tkinter.Tcl``, in a fresh process), and the JSON each writes is
  read back;
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


#: Runs one Tcl case in a fresh Python's own Tcl interpreter (never in the
#: pytest process: interpreters shared there lose the thread's ``stderr``
#: standard channel, and the next file channel's OS handle then survives its
#: ``close``, which no OpenSees.exe deck does). ``argv``: the prelude (stubs),
#: the body, the JSON result file.
_TCL_RUNNER = """
import json, sys, time, tkinter
interp, error = None, None
for _ in range(5):  # Windows sometimes fails one read of init.tcl
    try:
        interp = tkinter.Tcl()
        break
    except tkinter.TclError as e:
        error = str(e)
        time.sleep(0.2)
if interp is None:
    json.dump({"unavailable": error}, open(sys.argv[3], "w"))
    sys.exit(0)
interp.eval(open(sys.argv[1], encoding="utf-8").read())
failed = None
try:
    interp.eval(open(sys.argv[2], encoding="utf-8").read())
except tkinter.TclError as e:
    failed = str(e)
def get(name):
    if interp.eval("info exists " + name) != "1":
        return None
    return interp.eval("set " + name)
json.dump({
    "error": failed,
    "out": list(interp.splitlist(interp.eval("set ::_apesees_out"))),
    "barriers": get("::_apesees_barriers"),
    "tries": get("::_apesees_tries"),
    "waits": list(interp.splitlist(get("::_apesees_waits") or "")),
}, open(sys.argv[3], "w"))
"""

#: Every case's prelude: a message the deck prints is collected in
#: ``::_apesees_out``; ``puts`` to a channel (the record file) is the real one.
_TCL_PRELUDE = (
    "rename puts _apesees_puts; set ::_apesees_out {}\n"
    "proc puts {args} { if {[llength $args] == 1 || ([llength $args] == 2 "
    "&& [lindex $args 0] eq {-nonewline})} { lappend ::_apesees_out "
    "[lindex $args end] } else { _apesees_puts {*}$args } }\n"
)


def run_tcl_case(body: str, prelude: str = "") -> "dict[str, Any]":
    """Run ``body`` after ``prelude`` (and the ``puts`` capture) in a fresh
    process; its result: ``error`` (a Tcl error, or ``None``), ``out`` (the
    printed messages), and the ``barriers`` / ``tries`` / ``waits``
    counters the stubs keep."""
    import subprocess
    import sys
    import tempfile

    pytest.importorskip("tkinter")
    with tempfile.TemporaryDirectory(prefix="apegmsh-tcl-case-") as d:
        files = [Path(d) / n for n in ("runner.py", "prelude.tcl", "body.tcl",
                                       "result.json")]
        files[0].write_text(_TCL_RUNNER, encoding="utf-8")
        files[1].write_text(_TCL_PRELUDE + prelude, encoding="utf-8")
        files[2].write_text(body, encoding="utf-8")
        proc = subprocess.run([sys.executable, *map(str, files)], input="",
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-2000:]
        result = json.loads(files[3].read_text(encoding="utf-8"))
    if "unavailable" in result:
        pytest.skip(f"tkinter.Tcl() unavailable: {result['unavailable']}")
    return result


def run_tcl(block: str, *, build: "str | None", done: int, pid: int = 0,
            files: "list[Path] | None" = None, np: int = 1,
            ) -> "dict[str, Any]":
    """A record block, run with the deck's OpenSees commands stubbed: the
    loop counter at ``done``, ``getPID`` / ``getNP``, a counting
    ``barrier``, ``remove`` that closes ``files`` (adding ``FLUSH`` bytes)
    and ``ladrunoBuild`` answering ``build``. A Tcl error fails the test."""
    files = files or []
    open_files(files)
    names = " ".join("{" + str(f) + "}" for f in files)
    prelude = (
        f"set _apesees_i {done}\n"
        "set ::_apesees_barriers 0\n"
        f"set ::_apesees_files [list {names}]\n"
        f"proc getPID {{}} {{ return {pid} }}\n"
        f"proc getNP {{}} {{ return {np} }}\n"
        "proc barrier {} { incr ::_apesees_barriers }\n"
        "proc remove {what} { foreach p $::_apesees_files { set f [open $p a]; "
        "fconfigure $f -translation binary; puts -nonewline $f "
        f"[string repeat x {FLUSH}]; close $f }} }}\n"
    )
    if build is not None:
        prelude += f"proc ladrunoBuild {{}} {{ return {build} }}\n"
    result = run_tcl_case(block, prelude)
    assert result["error"] is None, result["error"]
    return result


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
    assert any(rename in ln for ln in block[-14:])  # the rename ends the deck


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
    assert renamed.endswith('.run.json"} _apesees_rerr]} {')


#: Runs a py deck in a fresh interpreter with every OpenSees call a no-op
#: that succeeds and every rename refused, then prints the run's marker.
_REFUSING_RUNNER = """
import os, sys, time, types
def refuse(*a, **k):
    raise PermissionError("held by another process")
os.replace = refuse
waits = []
time.sleep = waits.append  # the backoff's waits, not slept
ops = types.ModuleType("openseespy.opensees")
ops.__getattr__ = lambda name: (lambda *a, **k: 0)
pkg = types.ModuleType("openseespy")
pkg.opensees = ops
sys.modules["openseespy"], sys.modules["openseespy.opensees"] = pkg, ops
deck = sys.argv[1]
exec(compile(open(deck, encoding="utf-8").read(), deck, "exec"),
     {"__name__": "__main__"})
print("WAITS=" + ",".join(repr(w) for w in waits))
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
        (line,) = [ln for ln in out.splitlines() if ln.startswith("WAITS=")]
        waits = [1000 * float(w) for w in line[len("WAITS="):].split(",")]
    else:
        result = run_tcl_case(
            deck.read_text(encoding="utf-8") + "\nputs DECK-DONE\n",
            "proc unknown {args} { return 0 }\n"
            "rename file _apesees_real_file; set ::_apesees_tries 0\n"
            "set ::_apesees_waits {}\n"
            "proc after {ms} { lappend ::_apesees_waits $ms }\n"
            "proc file {cmd args} { if {$cmd eq {rename}} "
            "{ incr ::_apesees_tries; error {permission denied} }; "
            "uplevel 1 [list _apesees_real_file $cmd {*}$args] }\n")
        assert result["error"] is None, result["error"]
        waits = [float(w) for w in result["waits"]]
        assert int(result["tries"]) == len(waits) + 1
        out = "\n".join(result["out"])
    assert "the deck run record was not written" in out
    assert out.rstrip().endswith("DECK-DONE")
    assert not mpco.with_name("deck.mpco" + DECK_RUN_SUFFIX).exists()
    # Backoff: 50 ms doubling, capped at 1 s, until 30 s of waits are spent.
    assert waits[:6] == [50, 100, 200, 400, 800, 1000]
    assert max(waits) == 1000 and waits == sorted(waits)
    assert 30000 <= sum(waits) < 31000


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
        result = run_tcl_case(text, "proc unknown {args} { return 0 }\n"
                                    "proc analyze {args} { return -3 }\n")
        assert "analyze FAILED" in (result["error"] or "")
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
            result = run_tcl(body, build=None, done=STEPS, pid=pid,
                             files=parts, np=2)
            barriers = int(result["barriers"])
        assert barriers == 1
        written = mpco.with_name("run.mpco" + DECK_RUN_SUFFIX).exists()
        assert written == (pid == 0), (
            "" if dialect == "py" else result["out"])
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
    ours = [w for w in caught if issubclass(w.category, RuntimeWarning)
            and "run record" in str(w.message)]
    # It points at the user's export call, here this file.
    assert all(Path(w.filename).name == "test_deck_run_record.py" for w in ours)
    return deck, [str(w.message) for w in ours]


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
    assert str(deck) in ours[0]  # names the deck
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


@pytest.fixture
def memo(monkeypatch: Any) -> "dict[str, tuple[str, str]]":
    """An empty process-wide lineage memo for this test, restored after it
    (#2263: ``ops.h5()`` and every deck record fill it)."""
    from apeGmsh.opensees._internal import artifact_write as aw

    fresh: "dict[str, tuple[str, str]]" = {}
    monkeypatch.setattr(aw, "_LINEAGE_BY_DIGEST", fresh)
    return fresh


@pytest.fixture(scope="module")
def parted_fem() -> FEMData:
    """A real two-partition FEMData (a gmsh-partitioned tet box), so the
    bridge can digest its declarations: the bench stub cannot be."""
    from apeGmsh import apeGmsh

    with apeGmsh(verbose=False, _artifacts=False, model_name="parted") as g:
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b")
        g.physical.add_volume("b", name="B")
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(dim=3)
        g.mesh.partitioning.partition(2)
        fem = g.mesh.queries.get_fem_data(dim=3)
    assert len(fem.partitions) == 2
    return fem


def _parted_bridge(fem: FEMData, mpco: Path) -> Any:
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, _artifacts=False)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
    ops.element.FourNodeTetrahedron(pg="B", material=mat)
    ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    return ops


def _export_no_compose(bridge: Any, deck: Path, monkeypatch: Any,
                       ) -> "list[str]":
    """Export ``deck`` (its dialect by suffix) with ``analyze_steps``; any
    archive composed during the export fails the test. The run-record
    warnings are returned."""
    import warnings

    from apeGmsh.opensees.apesees import apeSees
    from apeGmsh.opensees.emitter import py as pym

    def no_compose(*a: Any, **k: Any) -> None:
        raise AssertionError("the export composed an archive")

    with monkeypatch.context() as m:
        m.setattr(pym._MemoryArchive, "__init__", no_compose)
        m.setattr(apeSees, "_write_h5", no_compose)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            getattr(bridge, deck.suffix[1:])(str(deck), analyze_steps=STEPS,
                                             progress=False)
    return [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "run record" in str(w.message)]


def _run_parted(deck: Path, mpco: Path) -> "dict[str, Any]":
    """Run ``deck``'s record block as rank 0 of two; the record it writes."""
    parts = [mpco.with_name(f"{mpco.stem}.part-{k}.mpco") for k in (0, 1)]
    run = run_py if deck.suffix == ".py" else run_tcl
    run(record_block(deck), build=None, done=STEPS, pid=0, files=parts, np=2)
    return read_record(mpco, parts)


@pytest.mark.parametrize("dialect", ["py", "tcl"])
def test_ops_h5_lets_a_partitioned_deck_record_without_composing(
    tmp_path, dialect, monkeypatch, memo, parted_fem,
) -> None:
    """#2263: ``ops.h5()`` memoises the lineage it wrote, by declaration
    digest. A partitioned deck exported afterwards stamps its record with
    it, composing no archive, and the record equals that ``model.h5``."""
    mpco = tmp_path / "run.mpco"
    bridge = _parted_bridge(parted_fem, mpco)
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    assert list(memo.values()) == [lineage(model)]
    deck = tmp_path / f"deck.{dialect}"
    assert _export_no_compose(bridge, deck, monkeypatch) == []
    text = deck.read_text(encoding="utf-8")
    assert "getPID" in text and MARK in text
    rec = _run_parted(deck, mpco)
    assert (rec["model_hash"], rec["fem_hash"]) == lineage(model)


def test_the_memo_follows_the_declarations_in_every_order(
    tmp_path, monkeypatch, memo, parted_fem,
) -> None:
    """Export before ``ops.h5()``: no record, with the remedy named. After
    it: the record is that archive's. A changed declaration makes the memo
    stale for the new digest (no record again, never the old lineage) until
    the next ``ops.h5()``, whose lineage the next record then carries."""
    mpco = tmp_path / "run.mpco"
    bridge = _parted_bridge(parted_fem, mpco)
    first = _export_no_compose(bridge, tmp_path / "d0.py", monkeypatch)
    assert len(first) == 1 and "call ops.h5(path) before" in first[0]
    assert MARK not in (tmp_path / "d0.py").read_text(encoding="utf-8")
    m1 = tmp_path / "m1.h5"
    bridge.h5(str(m1))
    assert _export_no_compose(bridge, tmp_path / "d1.py", monkeypatch) == []
    rec1 = _run_parted(tmp_path / "d1.py", mpco)
    assert (rec1["model_hash"], rec1["fem_hash"]) == lineage(m1)
    bridge.uniaxialMaterial.ElasticMaterial(E=1.0)  # a new declaration
    stale = _export_no_compose(bridge, tmp_path / "d2.tcl", monkeypatch)
    assert len(stale) == 1
    assert MARK not in (tmp_path / "d2.tcl").read_text(encoding="utf-8")
    m2 = tmp_path / "m2.h5"
    bridge.h5(str(m2))
    assert lineage(m2)[0] != lineage(m1)[0]
    assert lineage(m2)[1] == lineage(m1)[1]  # the same mesh
    assert _export_no_compose(bridge, tmp_path / "d3.tcl", monkeypatch) == []
    rec2 = _run_parted(tmp_path / "d3.tcl", mpco)
    assert (rec2["model_hash"], rec2["fem_hash"]) == lineage(m2)


def test_an_explicit_h5_spares_the_in_memory_compose(
    tmp_path, monkeypatch, memo,
) -> None:
    """A serial model with no automatic archive (``_artifacts=False``)
    composes nothing either once ``ops.h5()`` ran: the lineage is the
    declarations', whatever name or directory the archive was written
    under, and the record equals another ``ops.h5()`` of the bridge."""
    mpco = tmp_path / "deck.mpco"
    bridge = truss_bridge(mpco, artifacts=False)
    (tmp_path / "elsewhere").mkdir()
    bridge.h5(str(tmp_path / "elsewhere" / "x.h5"), model_name="other")
    assert _export_no_compose(bridge, tmp_path / "d.py", monkeypatch) == []
    model = tmp_path / "model.h5"
    bridge.h5(str(model))
    assert lineage(model) == lineage(tmp_path / "elsewhere" / "x.h5")
    run_py(record_block(tmp_path / "d.py"), build=None, done=STEPS,
           files=[mpco])
    rec = read_record(mpco)
    assert (rec["model_hash"], rec["fem_hash"]) == lineage(model)


def test_a_kept_automatic_file_never_lends_its_lineage(
    tmp_path, monkeypatch, memo,
) -> None:
    """Bridge A's export writes the automatic ``model.h5``; a ``/geometry``
    zone added to it makes the next automatic write a "keep" (this run's
    fuller file, left as it is). Bridge B on the same snapshot with another
    E then exports: its record carries B's lineage, never the kept file's
    (A's), though P5 now records B's digest at that path."""
    import h5py

    from apeGmsh.opensees.apesees import apeSees

    monkeypatch.setenv("APEGMSH_ARTIFACT_DIR", str(tmp_path))
    fem = truss_fem()
    mpco = tmp_path / "run.mpco"

    def bridge(e: float) -> Any:
        ops = apeSees(fem, default_orientation=None)
        ops.model(ndm=2, ndf=2)
        mat = ops.uniaxialMaterial.ElasticMaterial(E=e)
        ops.element.Truss(pg="Bar", A=A, material=mat)
        ops.fix(pg="Base", dofs=(1, 1))
        ops.fix(pg="Tip", dofs=(0, 1))
        ops.recorder.MPCO(file=str(mpco), nodal_responses=("displacement",))
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as pat:
            pat.load(pg="Tip", forces=(P, 0.0))
        ops.test.NormDispIncr(tol=1e-12, max_iter=10)
        ops.algorithm.Linear()
        ops.constraints.Plain()
        ops.numberer.RCM()
        ops.system.BandGeneral()
        ops.integrator.LoadControl(dlam=1.0 / STEPS)
        ops.analysis.Static()
        return ops

    bridge(E).py(str(tmp_path / "a.py"), analyze_steps=STEPS, progress=False)
    target = tmp_path / "deck_run_record_truss.h5"
    with h5py.File(target, "a") as f:
        f.create_group("geometry")  # a zone the bridge's write would drop
    kept = lineage(target)
    b = bridge(E / 2)
    seen = _count_archives(monkeypatch)
    b.py(str(tmp_path / "b.py"), analyze_steps=STEPS, progress=False)
    assert lineage(target) == kept  # "keep": the file was not rewritten
    assert len(seen) == 1 and not isinstance(seen[0], str)  # in memory
    model = tmp_path / "b.h5"
    b.h5(str(model))
    assert lineage(model)[0] != kept[0]
    run_py(record_block(tmp_path / "b.py"), build=None, done=STEPS,
           files=[mpco])
    rec = read_record(mpco)
    assert (rec["model_hash"], rec["fem_hash"]) == lineage(model)


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
