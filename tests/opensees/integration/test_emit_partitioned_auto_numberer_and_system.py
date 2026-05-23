"""Integration test for ADR 0027 INV-5 — auto-numberer / auto-system.

When ``len(fem.partitions) > 1`` and the user has NOT explicitly set
a numberer / system, the bridge auto-emits:

* ``numberer ParallelPlain`` (OpenSeesMP-compatible parallel numberer)
* ``system Mumps`` (OpenSeesMP-compatible parallel direct solver)

A single ``UserWarning`` fires for each (one auto-emit per category).
"""
from __future__ import annotations

import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def test_auto_emits_parallel_plain_and_mumps_with_warning() -> None:
    """No user numberer / system + ``len(fem.partitions) == 2`` →
    one ``numberer ParallelPlain`` line, one ``system Mumps`` line,
    one ``UserWarning`` per auto-emit.
    """
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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # Each appears exactly once.
    assert text.count("numberer ParallelPlain") == 1
    assert text.count("system Mumps") == 1

    # UserWarning fired for both auto-emits.
    messages = [str(w.message) for w in caught if w.category is UserWarning]
    assert any("auto-emitting 'numberer ParallelPlain'" in m for m in messages), (
        f"expected numberer auto-emit warning; got {messages}"
    )
    assert any("auto-emitting 'system Mumps'" in m for m in messages), (
        f"expected system auto-emit warning; got {messages}"
    )


def test_analysis_commands_emitted_globally_not_inside_partition_block() -> None:
    """Per ADR 0027 §"Constraint handler interaction" the analysis
    chain is GLOBAL — numberer / system / constraints / etc. must NOT
    be wrapped in any partition_open block.
    """
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

    # Walk the call stream; numberer / system / constraints must
    # appear OUTSIDE any partition_open / partition_close bracket.
    in_block = False
    seen_outside = {"numberer": False, "system": False}
    for name, args, kwargs in rec.calls:
        if name == "partition_open":
            in_block = True
        elif name == "partition_close":
            in_block = False
        elif name in seen_outside:
            assert not in_block, (
                f"{name!r} must be emitted OUTSIDE any partition_open block"
            )
            seen_outside[name] = True
    assert all(seen_outside.values()), (
        f"expected numberer + system emitted globally; got {seen_outside}"
    )
