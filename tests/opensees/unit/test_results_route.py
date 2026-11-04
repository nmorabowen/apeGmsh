"""ADR 0115 D3 (R2-c, #1704) without a solver: ``ops.results(path)``.

``stream_run`` is replaced by a fake that writes the files a real run
would write (values from the 2-D cantilever's closed form), or nothing,
or a partial file, or raises as a non-zero exit does. That shows:

- the spec synthesis: a typed ``recorder.Node`` whose DOF columns are
  out of the reader's order, on two nodes, reads back each value under
  its own component (a transcription that kept the file's column order
  would swap them);
- every refusal, each named, each before any file is written (the
  directory listing is the same before and after the raise).

The real-run oracles are in ``tests/opensees/live/test_results_route_live.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import numpy as np
import pytest

from apeGmsh.opensees import apesees as apesees_mod
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees._results_route import RUN_COMPLETE_MARKER, _history_files
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.live.test_readback_live import (
    E,
    I_BEAM,
    L_BEAM,
    P_TIP,
    build_cantilever_fem,
    cantilever_bridge,
)
from tests.opensees.unit.test_label_selector_fix_merge import (
    build_footing_fem,
    chain,
    footing,
)
from tests.opensees.unit.test_readback import (
    FakeOps,
    up_bridge,
    with_session,
)

UY = -P_TIP * L_BEAM ** 3 / (3.0 * E * I_BEAM)     # tip deflection
RZ = -P_TIP * L_BEAM ** 2 / (2.0 * E * I_BEAM)     # tip rotation


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


@pytest.fixture
def run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run").mkdir()
    return tmp_path


def fake_run(
    monkeypatch: pytest.MonkeyPatch,
    write: Callable[[], None] | None = None,
    *, fail: bool = False, complete: bool = True,
) -> None:
    """Replace the subprocess: ``write`` stands for what the run records;
    the log holds the Tcl driver's completion line iff ``complete``."""
    def _stream_run(argv: list[str], **kw: Any) -> None:
        if write is not None:
            write()
        with open(kw["log_path"], "w", encoding="utf-8") as log:
            log.write("OpenSees -- Open System For Earthquake Engineering\n")
            if complete:
                log.write(RUN_COMPLETE_MARKER + "\n")
            else:
                log.write("apeGmsh: analyze FAILED at increment 2/2\n")
        if fail:
            raise RuntimeError("OpenSees exited with code 1")
        return None

    monkeypatch.setattr(apesees_mod, "stream_run", _stream_run)
    monkeypatch.setattr(
        apesees_mod, "_resolve_python_binary", lambda *a, **k: "x")
    monkeypatch.setattr(
        apesees_mod, "_resolve_opensees_binary", lambda *a, **k: "x")


def write_rows(path: str, *rows: list[float]) -> Callable[[], None]:
    def _write() -> None:
        np.savetxt(path, np.asarray(rows, dtype=np.float64), fmt="%.17g")
    return _write


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def refuses(ops: apeSees, path: str, root: Path, match: str) -> None:
    """``ops.results(path)`` raises ``match`` and writes nothing."""
    before = listing(root)
    with pytest.raises(BridgeError, match=match):
        ops.results(path)
    assert listing(root) == before


def tip_and_base(cantilever: Any) -> tuple[int, int]:
    ops = cantilever_bridge(cantilever)
    return ops.nodes.one(label="tip").tag, ops.nodes.one(label="base").tag


# ---------------------------------------------------------------------------
# The spec synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["py", "tcl"])
def test_typed_node_columns_read_back_under_their_components(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    verb: str,
) -> None:
    """DOFs declared (3, 1, 2) over (tip, base): the file holds
    ``t, rz, ux, uy`` per node; the reader's layout is ``ux, uy, rz``."""
    tip, base = tip_and_base(cantilever)
    ops = cantilever_bridge(cantilever)
    ops.recorder.Node(file="run/two.out", nodes=(tip, base), dofs=(3, 1, 2),
                      response="disp", time_format="dt")
    fake_run(monkeypatch, write_rows(
        "run/two.out", [1.0, RZ, 0.0, UY, 0.0, 0.0, 0.0]))
    getattr(ops, verb)(f"run/deck.{verb}", run=True)
    r = ops.results(f"run/deck.{verb}")
    get = r.nodes.get
    assert get(label="tip", component="displacement_y").item() == UY
    assert get(label="tip", component="rotation_z").item() == RZ
    assert get(label="tip", component="displacement_x").item() == 0.0
    assert get(label="base", component="displacement_y").item() == 0.0
    assert list(get(ids=[tip], component="rotation_z").time) == [1.0]
    # What a read writes sits beside the deck.
    assert (run_dir / "run" / f"deck.{verb}.results" / "model.h5").is_file()


def test_declared_nodes_read_back(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(nodes=("displacement_y", "displacement_x"),
                         label="tip", file_root="run", name="tipd")
    # The build's own naming: <file_root>/<decl>__<record>__<token>.out,
    # columns in the declaration's DOF order.
    fake_run(monkeypatch, write_rows(
        "run/tipd__default__disp.out", [1.0, UY, 0.0]))
    ops.py("run/deck.py", run=True)
    r = ops.results("run/deck.py")
    assert r.nodes.get(label="tip", component="displacement_y").item() == UY
    assert r.nodes.get(label="tip", component="displacement_x").item() == 0.0


# ---------------------------------------------------------------------------
# Route refusals
# ---------------------------------------------------------------------------


def typed_tip(ops: apeSees, **kw: Any) -> None:
    tip = ops.nodes.one(label="tip").tag
    ops.recorder.Node(file="run/tip.out", nodes=(tip,), dofs=(2,),
                      response="disp", **{"time_format": "dt", **kw})


def test_no_deck_run_refuses(cantilever: Any, run_dir: Path) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    refuses(ops, "run/deck.py", run_dir, r"no deck run happened")


def test_the_live_route_refuses_naming_read(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    fake_run(monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    ops.py("run/deck.py", run=True)
    with_session(ops, FakeOps(ndf=3))    # a live analyze since the run
    refuses(ops, "run/deck.py", run_dir, r"live ops\.analyze.*\.read.*Q4")


def test_another_path_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    fake_run(monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    ops.py("run/deck.py", run=True)
    refuses(ops, "run/other.py", run_dir, r"is not that deck")


def test_a_failed_run_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    fake_run(monkeypatch, write_rows("run/tip.out", [1.0, UY]), fail=True)
    with pytest.raises(RuntimeError):
        ops.py("run/deck.py", run=True)
    refuses(ops, "run/deck.py", run_dir, r"did not finish")


def test_declarations_changed_after_the_run_refuse(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    fake_run(monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    ops.py("run/deck.py", run=True)
    ops.mass(label="tip", values=(1.0, 1.0, 0.0))
    refuses(ops, "run/deck.py", run_dir, r"declarations changed after")


# ---------------------------------------------------------------------------
# Record refusals
# ---------------------------------------------------------------------------


def ran(ops: apeSees, monkeypatch: pytest.MonkeyPatch,
        write: Callable[[], None] | None = None) -> None:
    fake_run(monkeypatch, write)
    ops.py("run/deck.py", run=True)


def test_no_recorder_refuses_naming_the_alternatives(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir,
            r"declared no recorder.*recorder\.Ladruno\(\.\.\.\) on the "
            r"Ladruno fork build, with ops\.recorder\.MPCO\(\.\.\.\) under "
            r"declared ops\.stage\(\.\.\.\) blocks on stock.*"
            r"recorder\.declare.*np\.loadtxt")


@pytest.mark.parametrize("with_node", [False, True])
def test_typed_element_records_refuse_never_dropped(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    with_node: bool,
) -> None:
    """Element-only typed records, and a readable Node record beside an
    Element one: both refuse, naming the Element record (fail closed)."""
    ops = cantilever_bridge(cantilever)
    if with_node:
        typed_tip(ops)
    ops.recorder.Element(file="run/forces.out", pg="Beam",
                         response=("localForce",))
    ran(ops, monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    refuses(ops, "run/deck.py", run_dir,
            r"partial Results: typed recorder\.Element"
            r"\(file='run/forces\.out'\)")


@pytest.mark.parametrize("kind", ["raw", "line_stations"])
def test_declared_records_the_route_cannot_read_refuse(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(nodes=("displacement_y",), label="tip",
                         file_root="run", name="n")
    if kind == "raw":
        ops.recorder.declare(raw_nodes=("pressure",), label="tip",
                             file_root="run", name="raw")
        match = r"raw_nodes=\['pressure'\]"
    else:
        ops.recorder.declare(line_stations=("axial_force",), pg="Beam",
                             file_root="run", name="ls")
        match = r"name='ls'\) line_stations= record.*recorder\.MPCO"
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir, match)


def test_step_format_typed_record_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops, time_format="step")
    ran(ops, monkeypatch, write_rows("run/tip.out", [UY]))
    refuses(ops, "run/deck.py", run_dir, r"writes no time column")


def test_missing_output_file_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir, r"does not exist")


def test_partial_output_file_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    tip, base = tip_and_base(cantilever)
    ops.recorder.Node(file="run/two.out", nodes=(tip, base), dofs=(2,),
                      response="disp", time_format="dt")
    ran(ops, monkeypatch, write_rows("run/two.out", [1.0, UY]))
    refuses(ops, "run/deck.py", run_dir, r"has 2 columns.*writes 3")


def test_mismatched_time_columns_refuse(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    ops.recorder.declare(nodes=("displacement_x",), label="tip",
                         file_root="run", name="d")

    def write() -> None:
        write_rows("run/tip.out", [1.0, UY], [2.0, UY])()
        write_rows("run/d__default__disp.out", [0.5, 0.0], [2.0, 0.0])()

    ran(ops, monkeypatch, write)
    refuses(ops, "run/deck.py", run_dir, r"other times")


def test_declared_rotation_on_an_ndf3_beam_node_reads_back(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B5-b (#1723): ``declare`` maps a planar ``rotation_z`` to DOF 3 of
    the beam node, and the read names that column ``rotation_z``."""
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(nodes=("displacement_y", "rotation_z"), label="tip",
                         file_root="run", name="r")
    ran(ops, monkeypatch,
        write_rows("run/r__default__disp.out", [1.0, UY, RZ]))
    assert "'-dof', 2, 3, 'disp'" in (run_dir / "run" / "deck.py").read_text()
    r = ops.results("run/deck.py")
    assert r.nodes.get(label="tip", component="rotation_z").item() == RZ
    assert r.nodes.get(label="tip", component="displacement_y").item() == UY


def test_declared_rotation_on_a_up_node_refuses_on_read(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deck whose declare recorded DOF 3 of a LadrunoUP node (pore
    pressure) is refused by the read too, by the shared rotation-role
    table, even when the emit is stood in for by one that let it through
    (a deck from a build before the emit refused)."""
    from apeGmsh.opensees._internal.build import recorders as rec_mod

    ops = up_bridge(build_footing_fem())
    corner = ops.nodes.one(label="corner").tag
    ops.recorder.declare(nodes=("rotation_z",), ids=(corner,),
                         file_root="run", name="p")
    permissive = rec_mod._node_record_layout

    def _no_role_check(record: Any, decl: Any, node_ids: Any,
                       element_plan: Any, **kw: Any) -> tuple[int, ...]:
        return permissive(record, decl, node_ids, None)

    with monkeypatch.context() as m:
        m.setattr(rec_mod, "_node_record_layout", _no_role_check)
        ran(ops, m, write_rows("run/p__default__disp.out", [1.0, 0.5]))
    refuses(ops, "run/deck.py", run_dir,
            r"'rotation_z' has no established DOF at node .*LadrunoUP")


def test_a_rotational_slot_no_beam_establishes_refuses(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DOF 3 of a u-p node at ndm=2, ndf=3 is pore pressure."""
    ops = up_bridge(build_footing_fem())
    corner = ops.nodes.one(label="corner").tag
    ops.recorder.Node(file="run/p.out", nodes=(corner,), dofs=(3,),
                      response="disp", time_format="dt")
    ran(ops, monkeypatch, write_rows("run/p.out", [1.0, 0.0]))
    refuses(ops, "run/deck.py", run_dir,
            r"DOF 3 of node .*LadrunoUP touches it")


def test_a_staged_deck_text_record_refuses(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = footing(build_footing_fem())
    corner = ops.nodes.one(label="corner").tag
    ops.recorder.Node(file="run/c.out", nodes=(corner,), dofs=(1,),
                      response="disp", time_format="dt")
    with ops.stage(name="s1") as s:
        s.fix(label="left", dofs=(1, 1))
        s.analysis(**chain(ops))
        s.run(n_increments=1)
    ran(ops, monkeypatch, write_rows("run/c.out", [1.0, 0.0]))
    # It recommends routes that read a staged run (ADR 0131 D11).
    refuses(ops, "run/deck.py", run_dir,
            r"is staged.*ops\.recorder\.Ladruno on the Ladruno fork build, "
            r"or with ops\.recorder\.MPCO on stock OpenSees \(its stages "
            r"are declared, so its \.mpco converts\)")


def test_mpco_beside_text_records_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    ran(ops, monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    refuses(ops, "run/deck.py", run_dir, r"an MPCO recorder and 1 text")


def test_a_missing_mpco_file_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir, r"MPCO.*does not exist or is empty")


# ---------------------------------------------------------------------------
# Completion proof on the Tcl route; Invariant 2; MPCO partition count
# ---------------------------------------------------------------------------


def test_a_tcl_deck_that_stopped_before_its_end_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenSees.exe exits 0 after a Tcl ``error``: an exit code of 0 with
    no completion line in the log is not a finished run."""
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    fake_run(monkeypatch, write_rows("run/tip.out", [1.0, UY]),
             complete=False)
    # The fake replaces stream_run, whose completion check raises on a
    # real run (tests/opensees/subprocess/test_run_report.py); the
    # bridge's own check still leaves the run unfinished.
    ops.tcl("run/deck.tcl", run=True)
    refuses(ops, "run/deck.tcl", run_dir,
            rf"did not finish.*no {RUN_COMPLETE_MARKER} line")


def test_the_tcl_run_sources_the_deck_unchanged_through_a_driver(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    seen: list[list[str]] = []

    def _stream_run(argv: list[str], **kw: Any) -> None:
        seen.append(argv)
        Path(kw["log_path"]).write_text(RUN_COMPLETE_MARKER + "\n")

    fake_run(monkeypatch)
    monkeypatch.setattr(apesees_mod, "stream_run", _stream_run)
    ops.tcl("run/deck.tcl", run=False)
    deck_bytes = (run_dir / "run" / "deck.tcl").read_bytes()
    ops.tcl("run/deck.tcl", run=True)
    assert (run_dir / "run" / "deck.tcl").read_bytes() == deck_bytes
    [argv] = seen
    driver = Path(argv[1])
    assert driver.name == "deck.tcl.run.tcl"
    text = driver.read_text(encoding="utf-8")
    deck = (run_dir / "run" / "deck.tcl").resolve().as_posix()
    assert f"source {{{deck}}}" in text
    assert text.rstrip().endswith(f'puts "{RUN_COMPLETE_MARKER}"')


def test_a_read_builds_nothing_and_allocates_nothing(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant 2: the archive is written from the frozen BuiltModel the
    deck ran; ops.results makes no build() and no allocate* call."""
    ops = cantilever_bridge(cantilever)
    typed_tip(ops)
    ran(ops, monkeypatch, write_rows("run/tip.out", [1.0, UY]))
    calls: list[str] = []
    real_build = apeSees.build

    def spy_build(self: apeSees) -> Any:
        calls.append("build")
        return real_build(self)

    monkeypatch.setattr(apeSees, "build", spy_build)
    for name in dir(TagAllocator):
        if name.startswith("allocate"):
            real = getattr(TagAllocator, name)

            def spy(*a: Any, _n: str = name, _r: Any = real,
                    **k: Any) -> Any:
                calls.append(_n)
                return _r(*a, **k)

            monkeypatch.setattr(TagAllocator, name, spy)
    r = ops.results("run/deck.py")
    assert r.nodes.get(label="tip", component="displacement_y").item() == UY
    assert calls == []


# ---------------------------------------------------------------------------
# History recorders (ADR 0131 D11): MPCO converts, Ladruno reads directly
# ---------------------------------------------------------------------------
#
# The committed ``zl_springs.mpco`` (one stage, five steps, nodes 1-4)
# stands for the run's output, its element tags renumbered to the
# bridge's own (``_bridge_copy``). The oracle is the file itself: every
# displacement read back equals the ``.mpco``'s own dataset, read with
# h5py, step for step.


def springs_ops(*, staged: bool) -> apeSees:
    """The springs model the fixture records: one ``ops.stage`` block
    (static), or unstaged with one global static ``analysis``."""
    from apeGmsh.opensees.element.zero_length import ZeroLengthMatDir

    from tests.test_results_mpco_to_ladruno import (
        SPRINGS,
        _fem_from_mpco,
        _stages,
        _unstaged_springs_bridge,
    )

    if not staged:
        return cast(apeSees, _unstaged_springs_bridge(["static"]))
    ops = apeSees(_fem_from_mpco(SPRINGS, {"One": [100], "Three": [200]}))
    ops.model(ndm=3, ndf=3)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1000.0)
    ops.element.ZeroLength(
        pg="One", mat_dirs=(ZeroLengthMatDir(material=mat, dof=1),))
    ops.element.ZeroLength(pg="Three", mat_dirs=tuple(
        ZeroLengthMatDir(material=mat, dof=d) for d in (1, 2, 3)))
    _stages(ops, [("load", "static")])
    return ops


def run_output(ops: apeSees, where: Path) -> Path:
    """The fixture ``.mpco`` as this bridge's run would write it, at
    ``where`` (outside the run directory, so listings stay the run's)."""
    from tests.test_results_mpco_to_ladruno import SPRINGS, _bridge_copy

    where.mkdir(parents=True, exist_ok=True)
    tags = where / "tags.h5"
    ops.h5(str(tags))
    return cast(Path, _bridge_copy(SPRINGS, tags, where / "r.mpco"))


def copy_to(src: Path, dst: str) -> Callable[[], None]:
    import shutil

    def _write() -> None:
        shutil.copy(src, dst)
    return _write


def mpco_displacements(mpco: Path) -> dict[int, np.ndarray]:
    """``{node: (T, 3)}`` displacement history, straight from the file."""
    import h5py

    with h5py.File(mpco, "r") as f:
        g = f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT"]
        ids = np.asarray(g["ID"][...]).ravel()
        steps = sorted(g["DATA"], key=lambda k: int(k.split("_")[1]))
        data = np.stack([np.asarray(g["DATA"][k][...]) for k in steps])
    return {int(n): data[:, i, :] for i, n in enumerate(ids)}


def assert_reads_the_file(r: Any, mpco: Path) -> None:
    expected = mpco_displacements(mpco)
    for node, rows in expected.items():
        for j, axis in enumerate("xyz"):
            got = r.nodes.get(ids=[node], component=f"displacement_{axis}")
            np.testing.assert_array_equal(
                np.asarray(got.values).ravel(), rows[:, j])
    assert len(expected) == 4 and expected[1].shape[0] == 5


def ladrunos(root: Path) -> list[str]:
    return sorted(p.name for p in root.rglob("*.ladruno"))


def test_a_staged_mpco_converts_and_reads_the_ladruno(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The program declares its stage, so the kind is a fact: the
    ``.mpco`` converts beside itself and the read is the ``.ladruno``."""
    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    r = ops.results("run/deck.py")
    try:
        assert type(r._reader).__name__ == "LadrunoReader"
        assert_reads_the_file(r, src)
    finally:
        r.close()
    assert ladrunos(run_dir) == ["r.ladruno"]
    # A second read replaces the converted file an earlier read left.
    r = ops.results("run/deck.py")
    try:
        assert_reads_the_file(r, src)
    finally:
        r.close()


def test_a_second_read_while_the_first_is_open_reuses_its_conversion(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Review of #2796 (finding 1): the first Results holds the converted
    ``.ladruno`` open. A second read of the same run reuses that
    conversion (it still pairs: same ``.mpco``, same file, same lineage)
    instead of replacing a file Windows will not let it replace."""
    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    r1 = ops.results("run/deck.py")
    try:
        target = run_dir / "run" / "r.ladruno"
        stamp = target.stat().st_mtime_ns
        r2 = ops.results("run/deck.py")
        try:
            assert_reads_the_file(r2, src)
        finally:
            r2.close()
        assert target.stat().st_mtime_ns == stamp    # not converted again
        assert_reads_the_file(r1, src)
    finally:
        r1.close()


def test_a_rewritten_mpco_under_an_open_read_refuses_by_name(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The run wrote the ``.mpco`` again, so the earlier conversion no
    longer pairs and must be replaced. With the earlier Results still open
    the replace fails; the read refuses by name, never with an OSError."""
    import os

    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    r1 = ops.results("run/deck.py")
    try:
        mpco = run_dir / "run" / "r.mpco"
        st = mpco.stat()
        os.utime(mpco, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        try:
            r2 = ops.results("run/deck.py")
        except BridgeError as e:
            assert "close it (results.close()) and read again" in str(e).lower()
            # The reason is the true one (#2796 review 2, finding 3).
            assert "changed on disk since this deck run's earlier read" in (
                str(e))
            assert isinstance(e.__cause__, OSError)
        else:
            # Windows never lets an open file be replaced; a platform
            # that does converts again.
            assert sys.platform != "win32"
            try:
                assert_reads_the_file(r2, src)
            finally:
                r2.close()
    finally:
        r1.close()
    # Closed, the next read converts again and reads the file.
    r3 = ops.results("run/deck.py")
    try:
        assert_reads_the_file(r3, src)
    finally:
        r3.close()


def test_a_conversion_os_error_on_a_first_read_quotes_the_plain_reason(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """#2796 review 3: a first read with no ``.ladruno`` on disk replaces
    nothing, so a disk-full ``OSError`` from the converter is quoted by
    name and never blamed on an open Results."""
    from apeGmsh.results.readers import _ladruno_from_mpco as conv

    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    assert ladrunos(run_dir) == []

    def full(*a: Any, **k: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(conv, "_convert_part", full)
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    msg = str(info.value)
    assert "converting its .mpco to a .ladruno failed (OSError: [Errno 28] " \
        "No space left on device)." in msg
    assert "open" not in msg.split("failed", 1)[1].lower()
    assert "close it" not in msg.lower()
    assert ladrunos(run_dir) == []


def doubled(src: Path, where: Path) -> Path:
    """``src`` with every displacement doubled: the same bytes count."""
    import shutil

    import h5py

    dst = where / "doubled.mpco"
    shutil.copy(src, dst)
    with h5py.File(dst, "r+") as f:
        data = f["MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA"]
        for k in data:
            data[k][...] = 2.0 * data[k][...]
    assert dst.stat().st_size == src.stat().st_size
    return dst


@pytest.mark.parametrize("keep_open", [False, True])
def test_a_same_size_rerun_with_a_reset_mtime_never_serves_the_old_read(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory, keep_open: bool,
) -> None:
    """#2796 review 2, finding 1: a second deck run rewrites ``r.mpco``
    with other values at the same size, and its mtime is reset to the
    first run's (a 1-second-mtime filesystem does the same). The read of
    the new run is the new values: an earlier read's conversion is reused
    only by the deck run that made it. With the first Results still open,
    the replace refuses by name, giving the true reason."""
    import os

    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    where = tmp_path_factory.mktemp("springs")
    src = run_output(ops, where)
    twice = doubled(src, where)
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    mpco = run_dir / "run" / "r.mpco"
    first = mpco.stat()
    r1 = ops.results("run/deck.py")
    try:
        assert_reads_the_file(r1, src)

        def rewrite() -> None:
            copy_to(twice, "run/r.mpco")()
            os.utime(mpco, ns=(first.st_atime_ns, first.st_mtime_ns))

        ran(ops, monkeypatch, rewrite)
        assert (mpco.stat().st_size, mpco.stat().st_mtime_ns) == (
            first.st_size, first.st_mtime_ns)
        if keep_open:
            try:
                r2 = ops.results("run/deck.py")
            except BridgeError as e:
                msg = str(e)
                assert "no earlier read of this deck run converted it" in msg
                assert "close it (results.close()) and read again" in msg.lower()
                assert isinstance(e.__cause__, OSError)
            else:
                assert sys.platform != "win32"
                try:
                    assert_reads_the_file(r2, twice)
                finally:
                    r2.close()
            return
    finally:
        r1.close()
    r2 = ops.results("run/deck.py")
    try:
        assert_reads_the_file(r2, twice)
    finally:
        r2.close()


def test_an_unstaged_mpco_deck_that_wrote_nothing_names_the_remedies(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review of #2796 (finding 2): an unstaged deck emitted without
    ``analyze_steps`` drives no analysis, so no ``.mpco`` is written; the
    missing-file refusal names what makes an MPCO read."""
    ops = springs_ops(staged=False)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir,
            r"does not exist or is empty; the run wrote no history there\. "
            r"The deck run declares no ops\.stage block and left no run "
            r"record .*r\.mpco\.run\.json'\. Declare the stages with "
            r"ops\.stage\(\.\.\.\), or emit and run the deck with "
            r"analyze_steps=")


def test_a_staged_mpco_that_wrote_nothing_names_no_stage_remedy(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    before = listing(run_dir)
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    assert str(info.value).endswith("the run wrote no history there.")
    assert listing(run_dir) == before


def test_an_unstaged_mpco_without_a_run_record_refuses_by_name(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """No ``ops.stage`` block and no run record (a deck emitted without
    ``analyze_steps``): the kind is not a fact, so the read refuses before
    any write, naming the two remedies and never ``convert_mpco``."""
    ops = springs_ops(staged=False)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    before = listing(run_dir)
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    msg = str(info.value)
    assert "recorder.MPCO(file='run/r.mpco')" in msg
    assert "left no run record" in msg and "r.mpco.run.json" in msg
    assert "ops.stage(" in msg and "analyze_steps=" in msg
    assert "recorder.Ladruno" in msg
    assert "convert_mpco" not in msg and "from_mpco" not in msg
    assert listing(run_dir) == before
    assert ladrunos(run_dir) == []


def test_an_unstaged_mpco_whose_record_fails_a_check_refuses_by_name(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A run record that fails the converter's checks: the read refuses,
    quoting the failed check, and never reads the ``.mpco`` with
    ``Results.from_mpco``; no ``.ladruno`` is left beside it."""
    ops = springs_ops(staged=False)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))

    def write() -> None:
        copy_to(src, "run/r.mpco")()
        Path("run/r.mpco.run.json").write_text("{}", encoding="utf-8")

    ran(ops, monkeypatch, write)
    from apeGmsh.results import Results

    def no_from_mpco(*a: Any, **k: Any) -> Any:
        raise AssertionError("ops.results read the .mpco with from_mpco")

    monkeypatch.setattr(Results, "from_mpco", classmethod(no_from_mpco))
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    msg = str(info.value)
    assert "the stage kinds of the deck run" in msg
    assert "are not facts, so its .mpco is not converted" in msg
    assert "check 1 (the record)" in msg
    assert "ops.stage(" in msg and "analyze_steps=" in msg
    assert "convert_mpco" not in msg
    assert ladrunos(run_dir) == []


def test_a_stage_kind_refusal_names_each_remedy_once(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """#2805 item 6: the read quotes the converter's refusal and names
    the bridge's remedies once. ``read_deck_mpco``'s own remedies (and its
    name, which the user never called) are not repeated after them."""
    ops = springs_ops(staged=False)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))

    def write() -> None:
        copy_to(src, "run/r.mpco")()
        Path("run/r.mpco.run.json").write_text("{}", encoding="utf-8")

    ran(ops, monkeypatch, write)
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    msg = str(info.value)
    assert "check 1 (the record)" in msg
    for remedy in ("ops.stage(", "emit and run the deck with analyze_steps=",
                   "recorder.Ladruno("):
        assert msg.count(remedy) == 1, (remedy, msg)
    # read_deck_mpco's own wording of the deck remedy is not repeated.
    assert "run the program as a deck emitted with analyze_steps" not in msg
    assert "read_deck_mpco" not in msg


def _replace_refuses(monkeypatch: pytest.MonkeyPatch, winerror: int) -> None:
    """The converter's first replace (moving the old ``.ladruno`` aside)
    raises a ``PermissionError`` carrying ``winerror``; later calls (the
    rollback) run as usual."""
    from apeGmsh import _atomic_io

    real = _atomic_io.replace_with_retry
    calls = []

    def refuse_first(src: Any, dest: Any) -> None:
        calls.append(src)
        if len(calls) == 1:
            e = PermissionError(13, "denied", str(src))
            e.winerror = winerror  # type: ignore[attr-defined]
            raise e
        real(src, dest)

    monkeypatch.setattr(_atomic_io, "replace_with_retry", refuse_first)


@pytest.mark.parametrize("winerror, blames_open", [
    (32, True),     # ERROR_SHARING_VIOLATION: an open Results holds it
    (5, False),     # ERROR_ACCESS_DENIED: a directory ACL, say
], ids=["sharing-violation", "access-denied"])
def test_only_a_sharing_violation_blames_an_open_results(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory, winerror: int,
    blames_open: bool,
) -> None:
    """#2805 amendment 1: a replace of the earlier ``.ladruno`` that fails
    for another reason than a sharing violation quotes the plain failure
    and never tells the user to close a Results."""
    import os

    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))
    ran(ops, monkeypatch, copy_to(src, "run/r.mpco"))
    ops.results("run/deck.py").close()
    mpco = run_dir / "run" / "r.mpco"
    st = mpco.stat()
    os.utime(mpco, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    _replace_refuses(monkeypatch, winerror)
    with pytest.raises(BridgeError) as info:
        ops.results("run/deck.py")
    msg = str(info.value)
    assert "converting its .mpco to a .ladruno failed (PermissionError: " \
        in msg
    assert isinstance(info.value.__cause__, PermissionError)
    assert ("close it (results.close())" in msg.lower()) is blames_open
    assert ("blocks the replace" in msg) is blames_open
    assert (run_dir / "run" / "r.ladruno").exists()   # rolled back


def test_a_target_the_converter_did_not_write_is_never_replaced(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    ops = springs_ops(staged=True)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    src = run_output(ops, tmp_path_factory.mktemp("springs"))

    def write() -> None:
        copy_to(src, "run/r.mpco")()
        Path("run/r.ladruno").write_bytes(b"the fork's own file")

    ran(ops, monkeypatch, write)
    refuses(ops, "run/deck.py", run_dir,
            r"converts to \[.*r\.ladruno'\], which already exist and were "
            r"not written by the converter")
    assert (run_dir / "run" / "r.ladruno").read_bytes() == (
        b"the fork's own file")


def test_a_ladruno_history_reads_back(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``recorder.Ladruno``: the fork's file is read with
    ``Results.from_ladruno`` (on ``main`` before D11 it was refused)."""
    from apeGmsh.results import convert_mpco

    ops = springs_ops(staged=True)
    ops.recorder.Ladruno(file="run/r.ladruno",
                         nodal_responses=("displacement",))
    where = tmp_path_factory.mktemp("springs")
    src = run_output(ops, where)
    [fork] = convert_mpco(src, where / "fork.ladruno",
                          model_h5=where / "tags.h5")
    ran(ops, monkeypatch, copy_to(fork, "run/r.ladruno"))
    r = ops.results("run/deck.py")
    try:
        assert type(r._reader).__name__ == "LadrunoReader"
        assert_reads_the_file(r, src)
    finally:
        r.close()


def test_a_ladruno_with_no_file_refuses_by_name(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.Ladruno(file="", nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir,
            r"recorder\.Ladruno\(file=''\) names no file")


def test_two_history_recorders_refuse(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.MPCO(file="run/r.mpco", nodal_responses=("displacement",))
    ops.recorder.Ladruno(file="run/r.ladruno",
                         nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir,
            r"2 history recorders \(recorder\.MPCO\(file='run/r\.mpco'\), "
            r"recorder\.Ladruno\(file='run/r\.ladruno'\)\)")


def test_a_missing_ladruno_file_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = cantilever_bridge(cantilever)
    ops.recorder.Ladruno(file="run/r.ladruno",
                         nodal_responses=("displacement",))
    ran(ops, monkeypatch)
    refuses(ops, "run/deck.py", run_dir,
            r"recorder\.Ladruno\(file='run/r\.ladruno'\)'s output .* does "
            r"not exist or is empty")


def _partitioned_run(root: Path, n_ranks: int) -> Any:
    fem = SimpleNamespace(partitions=list(range(n_ranks)))
    return cast(Any, SimpleNamespace(
        cwd=root, partitioned=True, built=SimpleNamespace(fem=fem)))


@pytest.mark.parametrize("files, match", [
    (["r.part-0.mpco"], r"r\.part-1\.mpco' does not exist"),
    (["r.part-0.mpco", "r.part-1.mpco", "r.part-2.mpco"],
     r"2 rank\(s\).*\['r\.part-2\.mpco'\]"),
])
def test_partitioned_mpco_reads_exactly_the_runs_ranks(
    tmp_path: Path, files: list[str], match: str,
) -> None:
    for name in files:
        (tmp_path / name).write_bytes(b"x")
    rec = SimpleNamespace(file="r.mpco")
    with pytest.raises(BridgeError, match=match):
        _history_files(_partitioned_run(tmp_path, 2), cast(Any, rec))


def test_partitioned_mpco_returns_one_file_per_rank(tmp_path: Path) -> None:
    for k in range(2):
        (tmp_path / f"r.part-{k}.mpco").write_bytes(b"x")
    rec = SimpleNamespace(file="r.mpco")
    got = _history_files(_partitioned_run(tmp_path, 2), cast(Any, rec))
    assert [p.name for p in got] == ["r.part-0.mpco", "r.part-1.mpco"]
