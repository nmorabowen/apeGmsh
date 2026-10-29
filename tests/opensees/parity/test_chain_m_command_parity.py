"""Chain A link A3a: ``residualModes`` / ``ritzVectors`` on all five emitters.

:class:`~apeGmsh.opensees.analysis.modal.ResidualModes` and
:class:`~apeGmsh.opensees.analysis.modal.RitzVectors` reach every emitter
through ``command()`` (ADR 0114 D2), so each one must state the same fork
command line (WP-213, WP-215): Tcl and py as text, recording and H5 as the
channel's call and ``/opensees/commands`` row, live as the binding call.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees.analysis.modal import ResidualModes, RitzVectors
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

#: (primitive, fork Tcl line, the arguments after the command name)
CASES: list[tuple[Any, str, tuple[Any, ...]]] = [
    (ResidualModes(dirs=(1,)), "residualModes -dir 1", ("-dir", 1)),
    (ResidualModes(dirs=(1, 2), period=0.05, replace=True, tol=1e-5),
     "residualModes -dir 1 2 -period 0.05 -replace -tol 1e-05",
     ("-dir", 1, 2, "-period", 0.05, "-replace", "-tol", 1e-5)),
    (ResidualModes(dirs=(3,), period="rayleigh"),
     "residualModes -dir 3", ("-dir", 3)),
    (RitzVectors(n=4, dirs=(1,)), "ritzVectors 4 -dir 1", (4, "-dir", 1)),
    (RitzVectors(n=6, dirs=(1, 2), patterns=(7,), cycles=(2, 0, 3), tol=1e-9),
     "ritzVectors 6 -dir 1 2 -pattern 7 -cycles 2 0 3 -tol 1e-09",
     (6, "-dir", 1, 2, "-pattern", 7, "-cycles", 2, 0, 3, "-tol", 1e-9)),
    (RitzVectors(n=3, patterns=(5, 9)), "ritzVectors 3 -pattern 5 9",
     (3, "-pattern", 5, 9)),
]
IDS = [line for _, line, _ in CASES]


class _Ops:
    """A stand-in fork module that records the two commands."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def residualModes(self, *args: Any) -> None:
        self.calls.append(("residualModes", args))

    def ritzVectors(self, *args: Any) -> None:
        self.calls.append(("ritzVectors", args))


def _verb(line: str) -> str:
    return line.split(" ", 1)[0]


@pytest.mark.parametrize(("prim", "line", "args"), CASES, ids=IDS)
def test_tcl_writes_the_fork_line(prim: Any, line: str, args: tuple) -> None:
    e = TclEmitter()
    prim._emit(e, 0)
    assert e.lines()[-1] == line


@pytest.mark.parametrize(("prim", "line", "args"), CASES, ids=IDS)
def test_py_writes_the_fork_call(prim: Any, line: str, args: tuple) -> None:
    e = PyEmitter()
    prim._emit(e, 0)
    assert e.lines()[-1] == f"ops.{_verb(line)}({', '.join(map(repr, args))})"


@pytest.mark.parametrize(("prim", "line", "args"), CASES, ids=IDS)
def test_recording_captures_the_channel_call(prim: Any, line: str, args: tuple) -> None:
    e = RecordingEmitter()
    prim._emit(e, 0)
    assert e.calls == [("command", (_verb(line), *args), {})]


@pytest.mark.parametrize(("prim", "line", "args"), CASES, ids=IDS)
def test_h5_records_a_commands_row(prim: Any, line: str, args: tuple) -> None:
    e = H5Emitter(model_name="m")
    prim._emit(e, 0)
    rec = e._commands[-1]
    assert (rec.method, rec.token, rec.args) == ("command", _verb(line), args)


@pytest.mark.parametrize(("prim", "line", "args"), CASES, ids=IDS)
def test_live_calls_the_binding(prim: Any, line: str, args: tuple) -> None:
    ops = _Ops()
    e = LiveOpsEmitter.__new__(LiveOpsEmitter)
    e._ops = ops  # type: ignore[assignment]
    e._in_partition = False
    prim._emit(e, 0)
    assert ops.calls == [(_verb(line), args)]


def test_live_names_the_fork_when_the_binding_is_missing() -> None:
    e = LiveOpsEmitter.__new__(LiveOpsEmitter)
    e._ops = object()  # type: ignore[assignment]
    e._in_partition = False
    with pytest.raises(RuntimeError, match="requires fork"):
        ResidualModes(dirs=(1,))._emit(e, 0)
