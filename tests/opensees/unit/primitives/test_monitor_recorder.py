"""Unit tests for the ``Monitor`` recorder primitive (fork-only live sink).

``recorder Monitor`` streams selected nodal scalars to a SWMR-HDF5 sidecar
for live tailing. This covers construction, validation, the ``_emit`` deck
shape, the ``pg=`` materialize/guard, and the ``ops.recorder.Monitor(...)``
namespace — all fork-free (no openseespy / gmsh). The fork-build *run* gate
and the round-trip read live in the results-side tests.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.recorder import Monitor


class TestMonitorConstruction:
    def test_minimal(self) -> None:
        r = Monitor(sink="live.h5", nodes=(2,), dofs=(1,))
        assert r.sink == "live.h5"
        assert r.nodes == (2,)
        assert r.dofs == (1,)
        assert r.resp == "disp"
        assert r.every is None and r.hz is None

    def test_explicit(self) -> None:
        r = Monitor(
            sink="live.h5", nodes=(2, 3), dofs=(1, 2), resp="accel",
            every=5, hz=30.0,
        )
        assert r.resp == "accel"
        assert r.every == 5 and r.hz == 30.0

    def test_dependencies_empty(self) -> None:
        assert Monitor(sink="m.h5", nodes=(1,), dofs=(1,)).dependencies() == ()


class TestMonitorValidation:
    def test_both_nodes_and_pg_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of nodes= or pg="):
            Monitor(sink="m.h5", nodes=(1,), pg="roof", dofs=(1,))

    def test_neither_nodes_nor_pg_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of nodes= or pg="):
            Monitor(sink="m.h5", dofs=(1,))

    def test_no_dofs_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one dof"):
            Monitor(sink="m.h5", nodes=(1,), dofs=())

    def test_bad_resp_raises(self) -> None:
        with pytest.raises(ValueError, match="resp must be one of"):
            Monitor(sink="m.h5", nodes=(1,), dofs=(1,), resp="stress")

    def test_every_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="every must be >= 1"):
            Monitor(sink="m.h5", nodes=(1,), dofs=(1,), every=0)

    def test_nonpositive_hz_raises(self) -> None:
        with pytest.raises(ValueError, match="hz must be > 0"):
            Monitor(sink="m.h5", nodes=(1,), dofs=(1,), hz=0.0)


class TestMonitorEmit:
    def test_nodes_minimal(self) -> None:
        r = Monitor(sink="live.h5", nodes=(2,), dofs=(1,))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                ("Monitor", "-node", 2, "-dof", 1, "-resp", "disp",
                 "-sink", "live.h5"),
                {},
            )
        ]

    def test_multi_node_dof_with_gates(self) -> None:
        r = Monitor(
            sink="live.h5", nodes=(2, 3), dofs=(1, 2), resp="vel",
            every=5, hz=30.0,
        )
        e = RecordingEmitter()
        r._emit(e, tag=1)
        assert e.calls == [
            (
                "recorder",
                ("Monitor", "-node", 2, 3, "-dof", 1, 2, "-resp", "vel",
                 "-sink", "live.h5", "-every", 5, "-hz", 30.0),
                {},
            )
        ]

    def test_no_gates_omits_every_and_hz(self) -> None:
        r = Monitor(sink="live.h5", nodes=(1,), dofs=(1,))
        e = RecordingEmitter()
        r._emit(e, tag=1)
        args = e.calls[0][1]
        assert "-every" not in args and "-hz" not in args

    def test_pg_emit_without_materialize_raises(self) -> None:
        r = Monitor(sink="live.h5", pg="roof", dofs=(1,))
        e = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="pg= must be resolved"):
            r._emit(e, tag=1)

    def test_materialize_explicit_nodes_is_identity(self) -> None:
        r = Monitor(sink="live.h5", nodes=(2, 3), dofs=(1,))
        out = r.materialize(RecordingEmitter(), cast("object", None), None)
        assert out is r


class TestMonitorDeckEmission:
    def test_tcl_line(self) -> None:
        r = Monitor(sink="live.h5", nodes=(2,), dofs=(1,), every=5)
        e = TclEmitter()
        r._emit(e, tag=1)
        assert (
            "recorder Monitor -node 2 -dof 1 -resp disp -sink live.h5 -every 5"
            in e.lines()
        )

    def test_py_line(self) -> None:
        r = Monitor(sink="live.h5", nodes=(2,), dofs=(1,))
        e = PyEmitter()
        r._emit(e, tag=1)
        assert (
            "ops.recorder('Monitor', '-node', 2, '-dof', 1, "
            "'-resp', 'disp', '-sink', 'live.h5')"
            in e.lines()
        )


class TestMonitorNamespace:
    def test_constructs_and_registers(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        r = ops.recorder.Monitor(sink="live.h5", nodes=(2,), dofs=(1,))
        assert isinstance(r, Monitor)
        assert r in ops._primitives

    def test_namespace_validation_propagates(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        with pytest.raises(ValueError, match="at least one dof"):
            ops.recorder.Monitor(sink="live.h5", nodes=(2,), dofs=())
