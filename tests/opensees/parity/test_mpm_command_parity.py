"""Chain M link M3: every ``mpm`` member on all five emitters (ADR 0134 D8).

Each member of :mod:`apeGmsh.opensees.mpm` reaches every emitter through
``command("mpm", ...)`` (ADR 0114 D2), so it states one fork command line
(fork guide ``232_ladruno_mpm_guide.md`` "Commands", parser
``OPS_LadrunoMPM.cpp``, both at fork ``f0b9a1df6``): Tcl and py as text,
recording as the channel's call, H5 as the ``/opensees/commands`` row, and
live as the refusal (MPM is deck-only, ADR 0134 D1).

Two members render specially on the deck emitters, by design: ``grid`` is
followed by the bound particle set's anchored ``mpm particles -file`` line,
and a ``body`` row is a comment (the fork's own ``mpm body`` seeds boxes,
discs and balls; apeGmsh's bodies are in the particle set).

The lines above are hard-coded, so they cannot see a subcommand the fork
gains. The ``test_fork_surface_*`` tests below read the fork's parser
dispatch at ``FORK_PIN`` (the snapshot ``scripts/refresh_mpm_fork_surface.py``
writes, see ``mpm_fork_surface.py``): every subcommand there is a member's or
a reasoned ``NOT_OFFERED`` row, by name.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from apeGmsh.opensees import mpm
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic
from apeGmsh.opensees.time_series.time_series import Linear
from tests.opensees.parity import mpm_fork_surface as surface

#: The ``mpm`` family's contract list (``tests/families.py``).
ALL_MPM = [
    mpm.Grid, mpm.Body, mpm.Fix, mpm.Wall, mpm.Gravity, mpm.TimeSeries,
    mpm.Damping, mpm.Flip, mpm.MassCutoff, mpm.Safety, mpm.Units,
    mpm.Output, mpm.Analyze,
]
assert ALL_MPM == list(mpm.ALL_MPM)

MAT = ElasticIsotropic(E=1.0e7, nu=0.3)
SERIES = Linear()
#: The tags the resolver hands the members' dependencies.
TAGS = {id(MAT): 4, id(SERIES): 9}
GRID_TAG = 7
SET = "case1.particles.h5"
TCL_SET_LINE = f"mpm particles -file [file join [file dirname [info script]] {{{SET}}}]"
PY_SET_LINE = ("ops.mpm('particles', '-file', os.path.join(os.path.dirname("
               f"os.path.abspath(__file__)), \"{SET}\"))")

#: (member, tag passed to _emit, fork Tcl line, the arguments after ``mpm``)
CASES: list[tuple[Any, int, str, tuple[Any, ...]]] = [
    (mpm.Grid(origin=(0.0, 0.0), h=0.1, cells=(50, 10)), GRID_TAG,
     "mpm grid 7 -origin 0.0 0.0 -h 0.1 -cells 50 10",
     ("grid", 7, "-origin", 0.0, 0.0, "-h", 0.1, "-cells", 50, 10)),
    (mpm.Grid(origin=(0.0, -1.0, 2.0), h=0.25, cells=(4, 8, 2), basis="linear"),
     GRID_TAG,
     "mpm grid 7 -origin 0.0 -1.0 2.0 -h 0.25 -cells 4 8 2 -basis linear",
     ("grid", 7, "-origin", 0.0, -1.0, 2.0, "-h", 0.25, "-cells", 4, 8, 2,
      "-basis", "linear")),
    (mpm.Body(name="soil", pg="Soil", material=MAT, density=1800.0, ppc=2), 3,
     "mpm body soil -pg Soil -mat 4 -rho 1800.0 -ppc 2",
     ("body", "soil", "-pg", "Soil", "-mat", 4, "-rho", 1800.0, "-ppc", 2)),
    (mpm.Fix(face="xmin", components="x"), 3, "mpm fix xmin x", ("fix", "xmin", "x")),
    (mpm.Fix(face="zmax", components="xyz"), 3, "mpm fix zmax xyz",
     ("fix", "zmax", "xyz")),
    (mpm.Wall(face="ymin", mu=0.3), 3, "mpm wall ymin 0.3", ("wall", "ymin", 0.3)),
    (mpm.Gravity(g=(5.0, -8.66)), 3, "mpm gravity 5.0 -8.66", ("gravity", 5.0, -8.66)),
    (mpm.TimeSeries(series=SERIES), 3, "mpm timeSeries 9", ("timeSeries", 9)),
    (mpm.Damping(alpha=0.05), 3, "mpm damping local 0.05",
     ("damping", "local", 0.05)),
    (mpm.Flip(alpha=0.95), 3, "mpm flip 0.95", ("flip", 0.95)),
    (mpm.MassCutoff(c=1e-8), 3, "mpm massCutoff 1e-08", ("massCutoff", 1e-8)),
    (mpm.Safety(s=0.4), 3, "mpm safety 0.4", ("safety", 0.4)),
    (mpm.Units(length="m", force="kN", time="s"), 3, "mpm units m kN s",
     ("units", "m", "kN", "s")),
    (mpm.Units(length="m", force="kN", time="s", system="kN_m_s"), 3,
     "mpm units m kN s -system kN_m_s",
     ("units", "m", "kN", "s", "-system", "kN_m_s")),
    (mpm.Output(path="run.ladruno"), 3, "mpm output {run.ladruno}",
     ("output", "run.ladruno")),
    (mpm.Output(path="out dir/run.ladruno", every=10,
                fields=("VELOCITY", "STRESS"), compress=1, flush=2.0), 3,
     "mpm output {out dir/run.ladruno} -every 10 -fields VELOCITY STRESS "
     "-compress 1 -flush 2.0",
     ("output", "out dir/run.ladruno", "-every", 10, "-fields", "VELOCITY",
      "STRESS", "-compress", 1, "-flush", 2.0)),
    (mpm.Output(path="r.ladruno", dt=0.01), 3, "mpm output {r.ladruno} -dt 0.01",
     ("output", "r.ladruno", "-dt", 0.01)),
    (mpm.Analyze(steps=5000, dt=2.0e-4), 3, "mpm analyze 5000 0.0002",
     ("analyze", 5000, 0.0002)),
]
IDS = [line for _, _, line, _ in CASES]
assert {type(p) for p, _, _, _ in CASES} == set(ALL_MPM)


def _resolved(e: Any) -> Any:
    set_tag_resolver(e, lambda p: TAGS[id(p)])
    return e


def _tcl() -> TclEmitter:
    e = _resolved(TclEmitter())
    e.bind_particle_set(SET)
    return e


def _py() -> PyEmitter:
    e = _resolved(PyEmitter())
    e.bind_particle_set(SET)
    return e


def _new_lines(e: Any, prim: Any, tag: int) -> list[str]:
    before = len(e.lines())
    prim._emit(e, tag)
    return e.lines()[before:]


@pytest.mark.parametrize(("prim", "tag", "line", "args"), CASES, ids=IDS)
def test_tcl_writes_the_fork_line(prim: Any, tag: int, line: str, args: tuple) -> None:
    got = _new_lines(_tcl(), prim, tag)
    if isinstance(prim, mpm.Grid):
        assert got == [line, TCL_SET_LINE]
    elif isinstance(prim, mpm.Body):
        assert got == [f"# {line}  (seeded into the particle set)"]
    else:
        assert got == [line]


@pytest.mark.parametrize(("prim", "tag", "line", "args"), CASES, ids=IDS)
def test_py_writes_the_fork_call(prim: Any, tag: int, line: str, args: tuple) -> None:
    call = f"ops.mpm({', '.join(map(repr, args))})"
    got = _new_lines(_py(), prim, tag)
    if isinstance(prim, mpm.Grid):
        assert got == [call, "import os", PY_SET_LINE]
    elif isinstance(prim, mpm.Body):
        assert got == [f"# {call}  (seeded into the particle set)"]
    else:
        assert got == [call]


@pytest.mark.parametrize(("prim", "tag", "line", "args"), CASES, ids=IDS)
def test_recording_captures_the_channel_call(
    prim: Any, tag: int, line: str, args: tuple,
) -> None:
    e = _resolved(RecordingEmitter())
    prim._emit(e, tag)
    assert e.calls == [("command", ("mpm", *args), {})]


@pytest.mark.parametrize(("prim", "tag", "line", "args"), CASES, ids=IDS)
def test_h5_records_a_commands_row(prim: Any, tag: int, line: str, args: tuple) -> None:
    e = _resolved(H5Emitter(model_name="m"))
    prim._emit(e, tag)
    rec = e._commands[-1]
    assert (rec.method, rec.token, rec.args) == ("command", "mpm", args)


@pytest.mark.parametrize(("prim", "tag", "line", "args"), CASES, ids=IDS)
def test_live_refuses_every_member(prim: Any, tag: int, line: str, args: tuple) -> None:
    class _Ops:
        def mpm(self, *a: Any) -> None:  # a fork binding: still refused
            raise AssertionError("the live route reached ops.mpm")

    e = _resolved(LiveOpsEmitter.__new__(LiveOpsEmitter))
    e._ops = _Ops()  # type: ignore[assignment]
    e._in_partition = False
    with pytest.raises(RuntimeError, match="deck-only.*fork or stock"):
        prim._emit(e, tag)


@pytest.mark.parametrize("make", [TclEmitter, PyEmitter], ids=["tcl", "py"])
def test_an_unbound_deck_refuses_the_grid(make: Any) -> None:
    """A deck emitter with no particle set bound (``modal_deck``, a bare
    emitter) refuses the grid by name instead of writing a deck whose
    particles are missing."""
    with pytest.raises(BridgeError, match="particle set"):
        CASES[0][0]._emit(make(), GRID_TAG)


@pytest.mark.parametrize("make", [_tcl, _py], ids=["tcl", "py"])
def test_deck_emitters_refuse_a_particles_row_and_a_second_grid(make: Any) -> None:
    e = make()
    with pytest.raises(BridgeError, match="never passed as a row"):
        e.command("mpm", "particles", "-file", SET)
    CASES[0][0]._emit(e, GRID_TAG)
    with pytest.raises(BridgeError, match="second mpm grid"):
        CASES[0][0]._emit(e, GRID_TAG)


@pytest.mark.parametrize("make", [_tcl, _py], ids=["tcl", "py"])
def test_bind_refuses_a_name_the_deck_line_cannot_quote(make: Any) -> None:
    from apeGmsh.opensees._mpm import ParticleSetError

    with pytest.raises(ParticleSetError, match="cannot quote"):
        make().bind_particle_set("we{ird}.particles.h5")


# -- the fork's command surface at FORK_PIN (#3063) ---------------------------

def _member_subcommands() -> set[str]:
    """The subcommand each member emits, read off the channel call (not
    typed here), so a member that changes its subcommand moves this set."""
    subs = set()
    for prim, tag, _, _ in CASES:
        e = _resolved(RecordingEmitter())
        prim._emit(e, tag)
        ((method, (token, sub, *_), _kw),) = e.calls
        assert (method, token) == ("command", "mpm")
        subs.add(sub)
    return subs


def _refresh_script() -> Any:
    path = surface.REPO / "scripts" / "refresh_mpm_fork_surface.py"
    spec = importlib.util.spec_from_file_location("_refresh_mpm_fork_surface", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SNAP = surface.load_snapshot()
FORK_SUBS = SNAP["subcommands"]
MEMBER_SUBS = _member_subcommands()


def test_fork_surface_snapshot_is_read_at_fork_pin() -> None:
    assert surface.pin_problems(SNAP, surface.read_pin()) == []


def test_fork_surface_every_fork_subcommand_is_a_member_or_a_row() -> None:
    assert surface.surface_problems(FORK_SUBS, MEMBER_SUBS, surface.NOT_OFFERED) == []


def test_fork_surface_members_are_the_thirteen_and_rows_the_rest() -> None:
    """The partition at ladruno-v20261118b, spelled out so a review sees it."""
    assert MEMBER_SUBS == {
        "analyze", "body", "damping", "fix", "flip", "gravity", "grid",
        "massCutoff", "output", "safety", "timeSeries", "units", "wall"}
    assert set(surface.NOT_OFFERED) == set(FORK_SUBS) - MEMBER_SUBS


# Damage cases (slice card #3063): each check fails, by name, on a bad input.

def test_fork_surface_a_new_fork_subcommand_fails_by_name() -> None:
    got = surface.surface_problems([*FORK_SUBS, "newThing"], MEMBER_SUBS,
                                   surface.NOT_OFFERED)
    assert len(got) == 1 and "'newThing' is neither" in got[0]


@pytest.mark.parametrize("dropped", ["stage", "pforce", "particles"])
def test_fork_surface_a_dropped_row_fails_by_name(dropped: str) -> None:
    rows = {k: v for k, v in surface.NOT_OFFERED.items() if k != dropped}
    got = surface.surface_problems(FORK_SUBS, MEMBER_SUBS, rows)
    assert len(got) == 1 and f"{dropped!r} is neither" in got[0]


def test_fork_surface_a_row_the_fork_lost_fails_as_stale() -> None:
    fork = [s for s in FORK_SUBS if s != "gridvel"]
    got = surface.surface_problems(fork, MEMBER_SUBS, surface.NOT_OFFERED)
    assert got == ["NOT_OFFERED row 'gridvel' is stale: the fork's parser at "
                   "the snapshot has no such subcommand"]


def test_fork_surface_a_member_the_fork_lost_fails_by_name() -> None:
    fork = [s for s in FORK_SUBS if s != "wall"]
    got = surface.surface_problems(fork, MEMBER_SUBS, surface.NOT_OFFERED)
    assert len(got) == 1 and "emits subcommand 'wall'" in got[0]


def test_fork_surface_a_member_and_a_row_at_once_fails() -> None:
    rows = {**surface.NOT_OFFERED, "fix": "a reason"}
    got = surface.surface_problems(FORK_SUBS, MEMBER_SUBS, rows)
    assert got == ["'fix' is both a member's subcommand and a NOT_OFFERED row"]


def test_fork_surface_a_row_without_a_reason_fails() -> None:
    rows = {**surface.NOT_OFFERED, "debug": "  "}
    assert surface.surface_problems(FORK_SUBS, MEMBER_SUBS, rows) == [
        "NOT_OFFERED row 'debug' gives no reason"]


@pytest.mark.parametrize("key", ["sha", "tag"])
def test_fork_surface_a_snapshot_off_the_pin_says_refresh(key: str) -> None:
    pin = {**surface.read_pin(), key: "something-else"}
    (got,) = surface.pin_problems(SNAP, pin)
    assert "something-else" in got and surface.REFRESH in got


@pytest.mark.parametrize("sub", ["pforce", "condForce"])
def test_fork_surface_mode_a_only_rows_say_so(sub: str) -> None:
    assert "mode A only, refused in mode F" in surface.NOT_OFFERED[sub]


def test_fork_surface_a_malformed_snapshot_is_refused(tmp_path: Path) -> None:
    bad = dict(SNAP, subcommands=[*FORK_SUBS, "aaa"])        # not sorted
    (tmp_path / "s.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted list"):
        surface.load_snapshot(tmp_path / "s.json")
    (tmp_path / "s.json").write_text(json.dumps({**SNAP, "x": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="keys"):
        surface.load_snapshot(tmp_path / "s.json")


# The refresh script's reading of the parser, offline: a synthetic source in
# the shape of the fork's ``OPS_LadrunoMPM.cpp`` at cfe478254 (a forward
# declaration, a helper that compares ``sub`` too, a usage line split over two
# C string literals).

_PARSER = '''\
int OPS_LadrunoMPM(void);
int helper(void) { if (sub == "notInTheDispatch") return 0; }
int OPS_LadrunoMPM(void)
{
  if (dom == 0) return fail("-- no Domain");
  if (OPS_GetNumRemainingInputArgs() < 1)
    return fail("<grid|body|"
                "stage> ...");
  const std::string sub = OPS_GetString();
  if (sub == "grid") return cmdGrid(dom);
  if (sub == "body") return cmdBody(h);
  if (sub == "stage") {%s}
  return fail("-- unknown sub-command '" + sub + "'");
}
'''


def test_refresh_reads_the_dispatch_of_the_definition() -> None:
    r = _refresh_script()
    assert r.extract_subcommands(_PARSER % "") == ["body", "grid", "stage"]


def test_refresh_refuses_a_dispatch_the_usage_line_does_not_name() -> None:
    r = _refresh_script()
    with pytest.raises(r.SurfaceError, match=r"only dispatched: \['affine'\]"):
        r.extract_subcommands(_PARSER % 'if (sub == "affine") return 0;')


@pytest.mark.parametrize("odd", ['sub=="affine"', '"affine" == sub',
                                 'sub.compare("affine")'])
def test_refresh_refuses_a_dispatch_spelled_another_way(odd: str) -> None:
    """A comparison the strict pattern cannot read is refused, even when the
    usage line omits the name too (review of #3065, finding 2)."""
    r = _refresh_script()
    with pytest.raises(r.SurfaceError, match=r"compares `sub` with \['affine'\]"):
        r.extract_subcommands(_PARSER % f"if ({odd}) return 0;")


def test_refresh_refuses_a_source_without_the_definition() -> None:
    r = _refresh_script()
    with pytest.raises(r.SurfaceError, match="found 0"):
        r.extract_subcommands(_PARSER.replace("int OPS_LadrunoMPM(void)\n{", "x"))
