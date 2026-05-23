"""Integration test for ADR 0027 per-rank element fan-out (Tcl emitter).

Verifies that under a partitioned FEM (``len(fem.partitions) > 1``) the
Tcl deck contains one ``if {[getPID] == K} { ... }`` block per partition,
each carrying only its own subset of elements (per-rank fan-out per
ADR 0027 §"Decision").  Mirrored on the Py emitter and the
RecordingEmitter so the test asserts both the text shape and the
RecordingEmitter event sequence used by the unit-level tests.

The unpartitioned path is byte-identical to today — see
``test_emit_unpartitioned_byte_identical_to_today.py`` for the
no-shim / no-bracket assertion.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def test_tcl_deck_has_two_partition_blocks() -> None:
    """A 2-partition frame emits two ``if {[getPID] == K} { ... }`` blocks."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    emitter = TclEmitter()
    bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # Two rank brackets — one per partition.
    assert text.count("if {[getPID] == 0}") == 1
    assert text.count("if {[getPID] == 1}") == 1
    # Shim emitted exactly once at the top of the deck.
    assert text.count("proc getPID") == 1


def test_recording_emitter_captures_partition_blocks() -> None:
    """The bridge calls ``partition_open(K)`` / ``partition_close()``
    around each rank's emit, with element calls scoped per-rank."""
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    rec = RecordingEmitter()
    bm.emit(rec)

    ranks_seen: list[int] = []
    elements_per_rank: dict[int, list[int]] = {}
    cur_rank: int | None = None
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            cur_rank = int(args[0])
            ranks_seen.append(cur_rank)
            elements_per_rank.setdefault(cur_rank, [])
        elif name == "partition_close":
            cur_rank = None
        elif name == "element" and cur_rank is not None:
            # args = (ele_type, ele_tag, *connectivity_and_params)
            elements_per_rank[cur_rank].append(int(args[1]))

    assert ranks_seen == [0, 1], (
        f"expected two ranks 0, 1; got {ranks_seen}"
    )
    # Rank 0 owns element 1; rank 1 owns element 2 — exclusive
    # ownership per :func:`make_two_column_frame_partitioned`. Per ADR
    # 0027 §"Tag determinism" element tags come from the single
    # canonical TagAllocator, so the rank-K block uses the SAME element
    # tag as the owning rank's block. The two emitted element tags are
    # whatever the allocator assigned to elements 1 and 2 — they must
    # be distinct and the same as the flat path would have used.
    assert len(elements_per_rank[0]) == 1
    assert len(elements_per_rank[1]) == 1
    assert elements_per_rank[0] != elements_per_rank[1], (
        "rank-0 and rank-1 must emit distinct element tags "
        "(tag-allocator must give each FEM element its own tag)."
    )


def test_py_deck_has_two_partition_blocks() -> None:
    """The Py emitter mirrors Tcl — ``if getPID() == K:`` per rank."""
    from apeGmsh.opensees.emitter.py import PyEmitter

    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )

    bm = ops.build()
    emitter = PyEmitter()
    bm.emit(emitter)
    text = "\n".join(emitter.lines())

    assert "if getPID() == 0:" in text
    assert "if getPID() == 1:" in text
    # getPID shim emitted exactly once.
    assert text.count("def getPID() -> int:") == 1
