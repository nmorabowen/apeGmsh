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
RUN_FIELDS = ("schema", "model_hash", "fem_hash", "recorder_file", "analysis",
              "analyze_steps", "steps_completed", "dt", "opensees_backend",
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


def truss_bridge(mpco: "Path | None", *, analysis: str = "Static") -> Any:
    """The truss on a bridge, unstaged, with one ``analysis`` declaration
    and (with ``mpco``) a bridge MPCO recorder on the nodal displacements."""
    from apeGmsh.opensees import apeSees

    ops = apeSees(truss_fem(), default_orientation=None)
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


class FakeOps:
    """The ``ops`` a record block calls: ``ladrunoBuild`` answers ``build``
    (absent when ``None``); ``barrier`` counts its calls."""

    def __init__(self, build: "str | None") -> None:
        self.barriers = 0
        if build is not None:
            self.ladrunoBuild = lambda: build

    def barrier(self) -> None:
        self.barriers += 1


def run_py(block: str, *, build: "str | None", done: int,
           pid: int = 0) -> FakeOps:
    ops = FakeOps(build)
    exec(compile(block, "<record>", "exec"),
         {"ops": ops, "_apesees_i": done - 1, "getPID": lambda: pid})
    return ops


def tcl() -> Any:
    tkinter = pytest.importorskip("tkinter")
    try:
        return tkinter.Tcl()
    except tkinter.TclError as e:  # no Tcl runtime beside this Python
        pytest.skip(f"tkinter.Tcl() unavailable: {e}")


def run_tcl(block: str, *, build: "str | None", done: int, pid: int = 0,
            ) -> Any:
    interp = tcl()
    interp.eval(f"set _apesees_i {done}")
    interp.eval("set _apesees_barriers 0")
    interp.eval(f"proc getPID {{}} {{ return {pid} }}")
    interp.eval("proc barrier {} { incr ::_apesees_barriers }")
    if build is not None:
        interp.eval(f"proc ladrunoBuild {{}} {{ return {build} }}")
    interp.eval(block)
    return interp


def read_record(mpco: Path) -> "dict[str, Any]":
    path = mpco.with_name(mpco.name + DECK_RUN_SUFFIX)
    assert path.is_file()
    assert not path.with_name(path.name + ".tmp").exists()
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert tuple(rec) == RUN_FIELDS
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
        run_py(block, build=build, done=STEPS)
    else:
        run_tcl(block, build=build, done=STEPS)
    rec = read_record(exported["mpco"])
    model_hash, fem_hash = lineage(exported["model"])
    assert rec["schema"] == DECK_RUN_SCHEMA == "apegmsh.deck_run/1"
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
            run_py(block, build=SHA, done=3)
        else:
            run_tcl(block, build=SHA, done=3)
        records.append(read_record(mpco))
    py, tc = ({k: v for k, v in r.items() if k != "completed_iso"}
              for r in records)
    assert py == tc
    assert (py["analysis"], py["dt"], py["steps_completed"]) == ("Transient", 0.01, 3)


def test_the_steps_completed_are_the_loops_own_count(exported) -> None:
    """``steps_completed`` is read from the loop's counter when the record is
    written, not copied from ``analyze_steps``."""
    run_py(record_block(exported["py"]), build=None, done=STEPS - 1)
    assert read_record(exported["mpco"])["steps_completed"] == STEPS - 1
    run_tcl(record_block(exported["tcl"]), build=None, done=STEPS - 2)
    assert read_record(exported["mpco"])["steps_completed"] == STEPS - 2


def test_a_tcl_path_tcl_would_substitute_is_written_verbatim(tmp_path) -> None:
    """Every character Tcl substitutes or counts survives the quoting."""
    where = tmp_path / "we{ir}d $x [y]"
    where.mkdir()
    mpco = where / "deck.mpco"
    deck = tmp_path / "deck.tcl"
    truss_bridge(mpco).tcl(str(deck), analyze_steps=2, progress=False)
    run_tcl(record_block(deck), build=None, done=2)
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
    block = record_block(exported[dialect])
    rename = "_apesees_os.replace(" if dialect == "py" else "file rename -force "
    assert block.rstrip().splitlines()[-1].startswith(rename)


def test_a_torn_write_leaves_no_record(exported, monkeypatch) -> None:
    """Atomic: the record is written to ``<name>.tmp`` and renamed, so a
    write that fails half way leaves no ``.run.json`` for a reader to see."""
    def torn(obj: Any, f: Any, *args: Any, **kw: Any) -> None:
        f.write('{"schema": "apeg')
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", torn)
    with pytest.raises(OSError, match="disk full"):
        run_py(record_block(exported["py"]), build=None, done=STEPS)
    final = exported["mpco"].with_name("deck.mpco" + DECK_RUN_SUFFIX)
    assert not final.exists()
    assert final.with_name(final.name + ".tmp").exists()
    # The Tcl block takes the same two steps.
    lines = record_block(exported["tcl"]).splitlines()
    (opened,) = [ln for ln in lines if ln.startswith("set _apesees_f [open ")]
    assert opened.endswith('.run.json.tmp" w]')
    assert lines[-1].startswith("file rename -force ")
    assert '.run.json.tmp" "' in lines[-1]
    assert lines[-1].endswith('.run.json"')


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
    mpco = tmp_path / "run.mpco"
    bridge = truss_bridge(mpco)
    body = "\n".join(deck_run_body(
        dialect, bridge.h5, [str(mpco)], "Static", STEPS, None,
        partitioned=True)) + "\n"
    for pid in (1, 0):
        if dialect == "py":
            barriers = run_py(body, build=None, done=STEPS, pid=pid).barriers
        else:
            interp = run_tcl(body, build=None, done=STEPS, pid=pid)
            barriers = int(interp.eval("set _apesees_barriers"))
        assert barriers == 1
        written = mpco.with_name("run.mpco" + DECK_RUN_SUFFIX).exists()
        assert written == (pid == 0)
    assert read_record(mpco)["recorder_file"] == "run.mpco"


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
