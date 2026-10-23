"""#2504: a deck that needs a fork build floor checks the running build first.

The Ladruno continuum solids hand ``materialState`` to their materials only
on a fork build at or after ``FORK_PIN`` 100eb9ae (fork PR #859, measured by
#2457).  On an older build the per-carrier sweep of a plastic
ManzariDafalias flip initialises the first element only, and the run gives
the wrong answer without a warning (#2445).  So a Tcl or openseespy deck
whose sweep, or whose ``materialState`` update, reaches one of
:data:`MATERIAL_STATE_FORK_FLOOR`'s elements checks the build before its
first command, and stops by name when the build lacks ``ladrunoSchema``
(fork PR #972; its first commit descends from fork PR #859).  Stock
OpenSees lacks it too.

**The join with #2457.**  On ``main`` the stage close still refuses a
multi-element flip on these classes, and their tokens are not in
``MATERIAL_STATE_ELEMENT_TOKENS``; #2457 adds both.  The :func:`joined`
fixture makes that change for one test, so the guard is exercised as it
will run once #2457 lands; after it lands the fixture changes nothing.

The live cases run the deck in a child process on the bound build: on stock
(``live-stock``) and on a fork build older than the pin it stops by name,
and at ``FORK_PIN`` (``live-fork``) it runs.  The Tcl deck needs an
OpenSees Tcl binary, which no CI lane has; :func:`test_the_tcl_guard`
evaluates the guard in ``tkinter.Tcl`` on every lane.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import h5py
import pytest

import apeGmsh
from apeGmsh.opensees import OpenSeesModel, apeSees
from apeGmsh.opensees._internal.build import stage_emit
from apeGmsh.opensees._internal.build.stage_emit import (
    LADRUNO_SOLID_MATERIAL_STATE_FLOOR,
    MATERIAL_STATE_FORK_FLOOR,
    fork_floor_message,
)
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.stage import _builder
from tests.opensees.h5._opensees_model_fixtures import build_two_quad_fem

_ROOT = Path(__file__).resolve().parents[3]

_MD = dict(
    G0=125.0, nu=0.05, e_init=0.8, Mc=1.25, c=0.712, lambda_c=0.019,
    e0=0.934, ksi=0.7, P_atm=100.0, m=0.01, h0=7.05, Ch=0.968, nb=1.1,
    A0=0.704, nd=3.5, z_max=4.0, cz=600.0, rho=1.42,
)
_MD_ARGS = tuple(_MD.values())

#: The argv index of the nDMaterial tag of each floor class (#2457's
#: ``MATERIAL_STATE_ELEMENT_TOKENS`` rows): right after the node tags.
_TOKEN_TAG_AT = {"LadrunoQuad": 4, "LadrunoCST": 3, "LadrunoLST": 6,
                 "LadrunoBrick": 8, "LadrunoBrick20": 20}

_FLOOR = LADRUNO_SOLID_MATERIAL_STATE_FLOOR
_MESSAGE = fork_floor_message(_FLOOR)
_TCL_GUARD = [
    'if {[info commands ladrunoSchema] eq ""} {',
    f'    error "{_MESSAGE}"',
    "}",
]
_PY_GUARD = [
    "if not hasattr(ops, 'ladrunoSchema'):",
    f"    raise SystemExit({_MESSAGE!r})",
]

#: The fork build #2457 measured to drop the sweep on these classes.
_STALE_FORK = "79e0623670c61c768357a01dff1477a4a0ef7e06"


def _pin_sha() -> str:
    for line in (_ROOT / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


@pytest.fixture
def joined(monkeypatch: pytest.MonkeyPatch) -> None:
    """#2457's change: the floor classes carry the sweep and pass the
    stage close."""
    assert set(_TOKEN_TAG_AT) == set(MATERIAL_STATE_FORK_FLOOR)
    for token, at in _TOKEN_TAG_AT.items():
        monkeypatch.setitem(stage_emit.MATERIAL_STATE_ELEMENT_TOKENS, token, at)
    monkeypatch.setattr(
        _builder, "MATERIAL_STATE_ELEMENTS",
        _builder.MATERIAL_STATE_ELEMENTS | set(MATERIAL_STATE_FORK_FLOOR))


def _bridge(*, route: str = "stage",
            pgs: "tuple[str, ...]" = ("Rock", "Fill")) -> apeSees:
    """Two LadrunoQuad elements (Rock, Fill) on one ManzariDafalias, flipped
    to plastic in stage ``push`` by ``route``."""
    ops = apeSees(build_two_quad_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mat = ops.nDMaterial.ManzariDafalias(name="sand", **_MD)
    for pg in pgs:
        ops.element.LadrunoQuad(pg=pg, thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="push") as s:
        if route == "stage":
            s.update_material_stage(materials=[mat], stage=1)
        else:
            for pg in pgs:
                s.update_parameter("materialState", 1.0, pg=pg, material=mat)
        s.analysis(
            test=ops.test.NormUnbalance(tol=1e-6, max_iter=50),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=0.1),
            constraints=ops.constraints.Plain(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    return ops


def _tcl(ops: apeSees) -> list[str]:
    em = TclEmitter()
    ops.build().emit(em)
    return em.lines()


def _py(ops: apeSees) -> list[str]:
    em = PyEmitter()
    ops.build().emit(em)
    return em.lines()


# ---------------------------------------------------------------------------
# Where the guard goes, and when
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_the_tcl_deck_checks_before_its_first_command(
    joined: None, route: str,
) -> None:
    lines = _tcl(_bridge(route=route))
    assert lines[0].startswith("#")
    assert lines[1:4] == _TCL_GUARD
    assert lines[4].startswith("model ")
    assert lines.count(_TCL_GUARD[0]) == 1
    flip = ("setParameter -val 1.0 -ele" if route == "stage"
            else "addToParameter")
    assert any(ln.startswith(flip) for ln in lines)


@pytest.mark.parametrize("route", ["stage", "materialState"])
def test_the_py_deck_checks_after_its_import(joined: None, route: str) -> None:
    lines = _py(_bridge(route=route))
    at = lines.index("import openseespy.opensees as ops")
    assert lines[at + 1:at + 3] == _PY_GUARD
    assert lines[at + 3] == "ops.wipe()"
    assert lines.count(_PY_GUARD[0]) == 1


def test_one_carrier_needs_no_check(joined: None) -> None:
    """On one element the bare updateMaterialStage reaches it on any
    build, so the deck is not guarded."""
    for lines in (_tcl(_bridge(pgs=("Rock",))), _py(_bridge(pgs=("Rock",)))):
        assert not any("ladrunoSchema" in ln for ln in lines)
        assert not any("setParameter" in ln for ln in lines)


def _without(lines: "list[str]", guard: "list[str]") -> "list[str]":
    at = lines.index(guard[0])
    assert lines[at:at + len(guard)] == guard
    return lines[:at] + lines[at + len(guard):]


def test_the_guard_is_the_only_difference(
    joined: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the floor table empty the deck is today's deck: the guard adds
    its lines and nothing else."""
    tcl, py = _tcl(_bridge()), _py(_bridge())
    monkeypatch.setattr(stage_emit, "MATERIAL_STATE_FORK_FLOOR", {})
    bare_tcl, bare_py = _tcl(_bridge()), _py(_bridge())
    assert _without(tcl, _TCL_GUARD) == bare_tcl
    assert _without(py, _PY_GUARD) == bare_py
    assert not any("ladrunoSchema" in ln for ln in bare_tcl + bare_py)


def test_model_hash_and_replay(
    joined: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard is derived from the emit calls: ``model_hash`` does not
    see it, and the deck replayed from the archive carries it as the
    bridge's deck does (R1b)."""
    ops = _bridge()
    ops.h5(str(tmp_path / "a.h5"))
    deck = tmp_path / "a.tcl"
    ops.tcl(str(deck), flat=True, progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        replay = OpenSeesModel.from_h5(str(tmp_path / "a.h5")).build(
            "tcl", flat=True)
    assert isinstance(replay, str)
    assert replay.splitlines() == deck.read_text(encoding="utf-8").splitlines()
    assert _TCL_GUARD[0] in replay.splitlines()

    monkeypatch.setattr(stage_emit, "MATERIAL_STATE_FORK_FLOOR", {})
    _bridge().h5(str(tmp_path / "b.h5"))

    def model_hash(name: str) -> str:
        with h5py.File(str(tmp_path / name), "r") as f:
            return str(f["meta"]["lineage"].attrs["model_hash"])

    assert model_hash("a.h5") == model_hash("b.h5")


def _two_floor_elements(em: Any, *, rank: "int | None" = None) -> None:
    em.model(ndm=2, ndf=2)
    em.nDMaterial("ManzariDafalias", 1, *_MD_ARGS)
    if rank is not None:
        em.partition_open(rank)
    em.element("LadrunoQuad", 3, 1, 2, 3, 4, 1, "-thick", 1.0)
    em.element("LadrunoQuad", 4, 4, 3, 5, 6, 1, "-thick", 1.0)
    if rank is not None:
        em.partition_close()


def test_the_partition_spans_move_with_the_guard(joined: None) -> None:
    """A per-rank Tcl deck is sliced by its recorded spans, so the spans
    move down with the guard: closed ones, and the one open at the flip."""
    em = TclEmitter()
    _two_floor_elements(em, rank=0)
    em.partition_open(0)
    em.update_material_stage(1, 1)
    em.partition_close()
    lines = em.lines()
    assert lines[1:4] == _TCL_GUARD
    spans = em.partition_spans()
    assert [s.rank for s in spans] == [0, 0]
    for span in spans:
        assert lines[span.header] == f"if {{[getPID] == {span.rank}}} {{"
        assert lines[span.end - 2] == "}"
    assert any(ln.startswith("    element LadrunoQuad 3")
               for ln in lines[spans[0].body_start:spans[0].body_end])
    assert any(ln.startswith("setParameter -val 1.0 -ele 3 4")
               for ln in lines[spans[1].body_start:spans[1].body_end])


def test_each_floor_is_checked_once(joined: None) -> None:
    for em in (TclEmitter(), PyEmitter()):
        _two_floor_elements(em)
        em.update_material_stage(1, 1)
        em.update_material_stage(1, 1)
        em.update_parameter(9, (3, 4), ("materialState", 1), 1.0)
        lines = em.lines()
        assert lines.count(_TCL_GUARD[0]) + lines.count(_PY_GUARD[0]) == 1


def test_an_elastic_flip_and_other_names_need_no_check(joined: None) -> None:
    for em in (TclEmitter(), PyEmitter()):
        _two_floor_elements(em)
        em.update_material_stage(1, 0)
        em.update_parameter(9, (3, 4), ("poissonRatio", 1), 0.3)
        assert not any("ladrunoSchema" in ln for ln in em.lines())


def test_a_removed_element_needs_no_check(joined: None) -> None:
    em = TclEmitter()
    _two_floor_elements(em)
    em.remove_element(3)
    em.remove_element(4)
    em.update_parameter(9, (5,), ("materialState", 1), 1.0)
    assert not any("ladrunoSchema" in ln for ln in em.lines())


def test_a_streamed_deck_checks_before_the_flip(
    joined: None, tmp_path: Path,
) -> None:
    """A streamed deck's start is on disk already, so it checks right
    before the line that needs the floor."""
    em = TclEmitter()
    path = tmp_path / "s.tcl"
    em.stream_to(str(path))
    _two_floor_elements(em)
    em.update_material_stage(1, 1)
    em.stream_finish()
    lines = path.read_text(encoding="utf-8").splitlines()
    at = lines.index(_TCL_GUARD[0])
    assert lines[at:at + 3] == _TCL_GUARD
    assert lines[at + 3].startswith("setParameter -val 1.0 -ele 3 4")


# ---------------------------------------------------------------------------
# What the guard does
# ---------------------------------------------------------------------------


def test_the_message_names_the_floor() -> None:
    for floor in set(MATERIAL_STATE_FORK_FLOOR.values()):
        msg = fork_floor_message(floor)
        assert floor.command in msg and "100eb9ae" in msg
        assert all(cls in msg for cls in MATERIAL_STATE_FORK_FLOOR)
    assert _pin_sha().startswith("100eb9ae"), (
        "FORK_PIN moved: re-read the floor (a later pin still forwards)")


def test_the_tcl_guard() -> None:
    """Stock-like Tcl (no ladrunoSchema) stops with the message; a build
    that registers the command runs on."""
    tkinter = pytest.importorskip("tkinter")
    tcl = tkinter.Tcl()
    with pytest.raises(tkinter.TclError) as err:
        tcl.eval("\n".join(_TCL_GUARD))
    assert str(err.value) == _MESSAGE
    tcl.eval("proc ladrunoSchema {} { return {} }")
    tcl.eval("\n".join(_TCL_GUARD))


def test_the_py_guard() -> None:
    code = compile("\n".join(_PY_GUARD), "<guard>", "exec")
    with pytest.raises(SystemExit) as err:
        exec(code, {"ops": types.SimpleNamespace()})
    assert err.value.code == _MESSAGE
    exec(code, {"ops": types.SimpleNamespace(ladrunoSchema=lambda: "{}")})


# ---------------------------------------------------------------------------
# Live: the deck on the bound build, in a child process
# ---------------------------------------------------------------------------

_PROBE = """\
import json, sys
{import_line}
print("FLOOR_PROBE " + json.dumps({{
    "build": ops.ladrunoBuild() if hasattr(ops, "ladrunoBuild") else None,
    "schema": hasattr(ops, "ladrunoSchema")}}), flush=True)
"""

_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()")
_STOCK_IMPORT = "import openseespy.opensees as ops"


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _run_py_deck(tmp_path: Path, *, fork_module: bool) -> None:
    """Run the guarded py deck as emitted (on the fork, through the
    bridge's resolver, since its module is not ``openseespy``) and check
    the outcome against the build the child bound."""
    import json

    deck = tmp_path / "deck.py"
    _bridge().py(str(deck))
    text = deck.read_text(encoding="utf-8")
    assert _STOCK_IMPORT in text and _PY_GUARD[0] in text
    import_line = _FORK_IMPORT if fork_module else _STOCK_IMPORT
    probe = _PROBE.format(import_line=import_line)
    body = text.replace(_STOCK_IMPORT, import_line)
    deck.write_text(probe + body + '\nprint("DECK_RAN", flush=True)\n',
                    encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=tmp_path,
                          env=_env(), capture_output=True, text=True,
                          check=False)
    out = proc.stdout + proc.stderr
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("FLOOR_PROBE ")), None)
    assert line is not None, out[-2000:]
    probe_ = json.loads(line.split(" ", 1)[1])
    build, schema = probe_["build"], probe_["schema"]
    # Anchors that do not trust the probe: stock and the stale fork lack
    # the command, the pin has it.
    if build is None or build == _STALE_FORK:
        assert not schema, probe_
    if build == _pin_sha():
        assert schema, probe_
    if schema:
        assert proc.returncode == 0 and "DECK_RAN" in proc.stdout, out[-2000:]
        assert _MESSAGE not in out
    else:
        assert proc.returncode != 0, out[-2000:]
        assert _MESSAGE in proc.stderr, out[-2000:]
        assert "DECK_RAN" not in proc.stdout
        assert "ops.model" not in proc.stderr


@pytest.mark.live
def test_the_py_deck_on_the_bound_openseespy(
    joined: None, tmp_path: Path,
) -> None:
    """``live-stock``: stock openseespy stops the deck by name."""
    _run_py_deck(tmp_path, fork_module=False)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_the_py_deck_on_the_fork(joined: None, tmp_path: Path) -> None:
    """``live-fork``: at FORK_PIN the deck runs; on an older fork build
    (the local 79e062367) it stops by name."""
    _run_py_deck(tmp_path, fork_module=True)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_the_tcl_deck_on_opensees_exe(joined: None, tmp_path: Path) -> None:
    """The Tcl deck on an OpenSees Tcl binary, which no CI lane has.
    ``OpenSees.exe`` exits 0 on a Tcl error, so the marker after the deck
    is the proof it ran."""
    binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
    if not binary:
        pytest.skip("no OpenSees Tcl binary ($OPENSEES_BIN or OpenSees on "
                    "PATH); test_the_tcl_guard runs the guard in tkinter.Tcl")
    deck = tmp_path / "deck.tcl"
    _bridge().tcl(str(deck))
    with deck.open("a", encoding="utf-8") as f:
        f.write('puts "DECK_RAN"\n')
    probe = tmp_path / "probe.tcl"
    probe.write_text(
        'puts "SCHEMA [llength [info commands ladrunoSchema]]"\n'
        'puts "BUILD [ladrunoBuild]"\n', encoding="utf-8")
    seen = subprocess.run([binary, str(probe)], cwd=tmp_path, env=_env(),
                          capture_output=True, text=True, check=False)
    schema = "SCHEMA 1" in seen.stdout
    if f"BUILD {_STALE_FORK}" in seen.stdout:
        assert not schema, seen.stdout
    proc = subprocess.run([binary, str(deck)], cwd=tmp_path, env=_env(),
                          capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    if schema:
        assert "DECK_RAN" in proc.stdout, out[-2000:]
    else:
        assert _MESSAGE[:60] in out, out[-2000:]
        assert "DECK_RAN" not in proc.stdout


def test_the_streamed_bridge_deck(joined: None, tmp_path: Path) -> None:
    """``ops.tcl(stream=True)`` keeps the check, before the flip."""
    path = tmp_path / "s.tcl"
    _bridge().tcl(str(path), stream=True)
    lines = path.read_text(encoding="utf-8").splitlines()
    at = lines.index(_TCL_GUARD[0])
    assert lines[at + 3].startswith("setParameter -val 1.0 -ele")
    buf = io.StringIO()
    buf.write("\n".join(_tcl(_bridge())))
    assert _TCL_GUARD[0] in buf.getvalue()
