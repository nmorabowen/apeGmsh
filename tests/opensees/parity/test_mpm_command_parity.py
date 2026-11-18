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
"""
from __future__ import annotations

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
