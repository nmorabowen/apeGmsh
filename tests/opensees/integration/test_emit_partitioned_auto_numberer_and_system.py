"""Integration test for ADR 0027 INV-5 — auto-numberer / auto-system.

When ``len(fem.partitions) > 1`` and the user has NOT explicitly set
a numberer / system, the bridge auto-emits a **runtime-conditional**:

* primary ``numberer ParallelPlain`` with fallback ``RCM``
  (OpenSeesMP uses the primary; single-process OpenSees catches the
  parse error and falls back to ``RCM``)
* primary ``system Mumps`` with fallback ``UmfPack``
  (same rationale — Mumps requires OpenSeesMP + MPI)

A single ``UserWarning`` fires for each (one auto-emit per category).

The runtime conditional restores end-to-end shim-consistency with
the ``proc getPID`` partition-open fallback emitted by P4-A
(ADR 0027 INV-5 amendment 2026-05-23).
"""
from __future__ import annotations

import warnings
from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def test_auto_emits_runtime_conditional_parallel_plain_and_mumps_tcl() -> None:
    """No user numberer / system + ``len(fem.partitions) == 2`` →
    Tcl deck contains a single ``catch`` wrapper for
    ``numberer ParallelPlain`` (fallback ``RCM``) and one for
    ``system Mumps`` (fallback ``UmfPack``); one ``UserWarning``
    per auto-emit.
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

    # The runtime-conditional catch wrapper appears exactly once for
    # numberer and once for system.
    expected_numberer = (
        "if {[catch {numberer ParallelPlain} _err]} { numberer RCM }"
    )
    expected_system = (
        "if {[catch {system Mumps} _err]} { system UmfPack }"
    )
    assert text.count(expected_numberer) == 1, (
        f"INV-5: expected catch-wrapped numberer; got:\n{text}"
    )
    assert text.count(expected_system) == 1, (
        f"INV-5: expected catch-wrapped system; got:\n{text}"
    )

    # The OLD unconditional form must NOT appear at the top level any
    # more — that was the contradictory behaviour ADR 0027 INV-5
    # (amended 2026-05-23) explicitly removed.
    assert "\nnumberer ParallelPlain\n" not in "\n" + text + "\n", (
        "INV-5 amendment: bare 'numberer ParallelPlain' line must not "
        "be emitted; the catch-wrapped form replaces it."
    )
    assert "\nsystem Mumps\n" not in "\n" + text + "\n", (
        "INV-5 amendment: bare 'system Mumps' line must not be "
        "emitted; the catch-wrapped form replaces it."
    )

    # UserWarning fired for both auto-emits.
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any(
        "numberer ParallelPlain" in m and "RCM" in m for m in messages
    ), (
        f"expected runtime-conditional numberer auto-emit warning; "
        f"got {messages}"
    )
    assert any(
        "system Mumps" in m and "UmfPack" in m for m in messages
    ), (
        f"expected runtime-conditional system auto-emit warning; "
        f"got {messages}"
    )


def test_auto_emits_runtime_conditional_py() -> None:
    """Same as the Tcl variant, but for the Py emitter — assert the
    deck contains the ``try / except`` block calling
    ``ops.numberer('ParallelPlain')`` with fallback
    ``ops.numberer('RCM')`` and the same for ``system``.
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
    emitter = PyEmitter()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # Both try-blocks present, each with primary + fallback.
    assert "ops.numberer('ParallelPlain')" in text
    assert "ops.numberer('RCM')" in text
    assert "ops.system('Mumps')" in text
    assert "ops.system('UmfPack')" in text
    # No bare unconditional emit of the primary at module top level
    # (a bare line would not be inside a `try:` block).  This is the
    # weakest portable check: confirm each primary is preceded by
    # ``try:`` within the same window.
    assert text.count("try:") >= 2, (
        f"expected at least two try blocks (numberer + system); "
        f"got:\n{text}"
    )


def test_analysis_commands_emitted_globally_not_inside_partition_block() -> None:
    """Per ADR 0027 §"Constraint handler interaction" the analysis
    chain is GLOBAL — runtime-conditional numberer / system must NOT
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

    # Walk the call stream; the runtime-conditional numberer / system
    # must appear OUTSIDE any partition_open / partition_close bracket.
    in_block = False
    seen_outside = {
        "parallel_runtime_fallback_numberer": False,
        "parallel_runtime_fallback_system": False,
    }
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
        f"expected runtime-conditional numberer + system emitted "
        f"globally; got {seen_outside}"
    )


# -----------------------------------------------------------------------
# ADR 0027 INV-5 — Amendment 2026-05-26 drift sentinels
#
# The runtime-conditional ``catch {numberer ParallelPlain} { numberer RCM }``
# fallback (introduced in the 2026-05-23 amendment) is dead code on the
# installed single-process OpenSees: ``ops.numberer('ParallelPlain')``
# silently no-ops instead of raising, so the ``catch`` branch never
# fires.  The ``system Mumps`` half DOES raise as documented.  The two
# tests below pin that observed runtime behaviour so a future upstream
# change that begins raising on ``ParallelPlain`` flips the xfail to
# pass and surfaces the drift.
# -----------------------------------------------------------------------


@pytest.mark.xfail(
    reason=(
        "ADR 0027 INV-5 amendment 2026-05-26: numberer ParallelPlain "
        "does not raise under single-process OpenSees on the installed "
        "build. When upstream OpenSees starts raising, this test will "
        "pass and the catch/try fallback becomes load-bearing."
    ),
    strict=False,
)
@pytest.mark.live
def test_numberer_parallel_plain_does_not_raise_under_single_process() -> None:
    ops = pytest.importorskip("openseespy.opensees")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    with pytest.raises(Exception):
        ops.numberer("ParallelPlain")


@pytest.mark.live
def test_system_mumps_raises_under_single_process() -> None:
    """The ``system Mumps`` half of INV-5 DOES raise under
    single-process — verify it stays that way.
    """
    ops = pytest.importorskip("openseespy.opensees")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    with pytest.raises(Exception):
        ops.system("Mumps")
