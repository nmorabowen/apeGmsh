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

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh.opensees import apesees as apesees_mod
from apeGmsh.opensees._internal.build import BridgeError
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
    *, fail: bool = False,
) -> None:
    """Replace the subprocess: ``write`` stands for what the run records."""
    def _stream_run(argv: list[str], **kw: Any) -> None:
        if write is not None:
            write()
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
            r"declared no recorder.*recorder\.MPCO.*recorder\.declare"
            r".*np\.loadtxt")


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


def test_declared_rotation_on_an_ndf3_node_refuses(
    cantilever: Any, run_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``declare`` sends ``rotation_z`` to DOF 6, which an ndf=3 node
    lacks: the column would not be the rotation."""
    ops = cantilever_bridge(cantilever)
    ops.recorder.declare(nodes=("rotation_z",), label="tip",
                         file_root="run", name="r")
    ran(ops, monkeypatch, write_rows("run/r__default__disp.out", [1.0, RZ]))
    refuses(ops, "run/deck.py", run_dir, r"'rotation_z' as DOF 6.*ndf=3")


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
    refuses(ops, "run/deck.py", run_dir, r"is staged")


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
