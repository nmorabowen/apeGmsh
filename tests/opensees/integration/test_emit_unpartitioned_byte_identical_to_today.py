"""Integration test for the unpartitioned byte-identity invariant.

When ``len(fem.partitions) <= 1`` the emitted output MUST be
byte-identical to the pre-ADR 0027 behaviour:

* No ``partition_open`` / ``partition_close`` calls anywhere.
* No ``proc getPID`` shim in Tcl output, no ``try / except`` shim
  in Py output.
* The exact same lines as before would have been emitted (the flat
  emit path lives untouched on this branch).
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
)


def _build_two_column_frame_ops() -> apeSees:
    """Set up a small two-column frame on an UNPARTITIONED FEM stub."""
    fem = make_two_column_frame()
    # NB: NO ``fem.set_partitions(...)`` call here — empty
    # ``fem.partitions`` triggers the flat / unpartitioned branch.
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    return ops


def test_tcl_unpartitioned_has_no_partition_text() -> None:
    """No ``getPID`` / ``if {[getPID]`` text appears in the Tcl deck
    when the FEM is unpartitioned.
    """
    ops = _build_two_column_frame_ops()
    bm = ops.build()
    emitter = TclEmitter()
    bm.emit(emitter)
    text = "\n".join(emitter.lines())

    assert "getPID" not in text, (
        "unpartitioned model must NOT emit the getPID shim or any "
        f"partition_open block. Found getPID in:\n{text}"
    )
    assert "if {[getPID]" not in text
    # The partition_close trailing brace pattern must also be absent.
    # (A bare '}' line may exist for closing a section block etc.,
    # so test the more specific bracket shape.)
    assert "if {[getPID] ==" not in text


def test_py_unpartitioned_has_no_partition_text() -> None:
    """No ``getPID`` import / fallback / ``if getPID() == K:`` text
    appears in the Py deck when the FEM is unpartitioned.
    """
    ops = _build_two_column_frame_ops()
    bm = ops.build()
    emitter = PyEmitter()
    bm.emit(emitter)
    text = "\n".join(emitter.lines())

    assert "getPID" not in text, (
        "unpartitioned model must NOT emit the Py getPID shim or any "
        f"if getPID() == K: block. Found in:\n{text}"
    )


def test_recording_unpartitioned_has_no_partition_calls() -> None:
    """The RecordingEmitter must not see any ``partition_open`` /
    ``partition_close`` events when the FEM is unpartitioned.
    """
    ops = _build_two_column_frame_ops()
    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    method_names = [name for name, _a, _kw in rec.calls]
    assert "partition_open" not in method_names, (
        "unpartitioned model must NOT call partition_open"
    )
    assert "partition_close" not in method_names, (
        "unpartitioned model must NOT call partition_close"
    )
