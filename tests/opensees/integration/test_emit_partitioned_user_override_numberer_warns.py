"""Integration test for ADR 0027 INV-5 — user-set MP-incompatible system warns.

When the user has explicitly set a numberer or system that is
detectably MP-incompatible (e.g. ``BandSPD`` system under
partitioning), the bridge emits a ``UserWarning`` at build time but
does NOT override the user's choice — the deck still emits ``system
BandSPD``.
"""
from __future__ import annotations

import warnings
from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame_partitioned,
)


def test_user_band_spd_under_partitioning_warns_but_preserves_choice() -> None:
    """User picks BandSPD under partitions > 1 → ``UserWarning`` at
    build time, but the deck still emits ``system BandSPD`` (no
    override).
    """
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.system.BandSPD()

    bm = ops.build()
    emitter = TclEmitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # User's choice preserved verbatim.
    assert text.count("system BandSPD") == 1, (
        f"INV-5: user-declared 'BandSPD' must be emitted unchanged; got:\n{text}"
    )
    # Mumps is NOT auto-emitted (user choice respected).
    assert "system Mumps" not in text, (
        "INV-5: user-set system must not be silently overridden"
    )
    # UserWarning fired flagging the MP-incompatibility.
    messages = [str(w.message) for w in caught if w.category is UserWarning]
    assert any(
        "BandSPD" in m and "OpenSeesMP" in m
        for m in messages
    ), (
        f"INV-5: expected MP-incompatibility warning for 'BandSPD'; "
        f"got {messages}"
    )


def test_user_plain_numberer_under_partitioning_warns_but_preserves_choice() -> None:
    """User picks serial ``Plain`` numberer under partitions > 1 →
    ``UserWarning`` at build time; deck still emits ``numberer Plain``.
    """
    fem = make_two_column_frame_partitioned()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.numberer.Plain()

    bm = ops.build()
    emitter = TclEmitter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bm.emit(emitter)
    text = "\n".join(emitter.lines())

    # User's choice preserved.
    assert text.count("numberer Plain") == 1
    assert "numberer ParallelPlain" not in text, (
        "INV-5: user-set numberer must not be silently overridden"
    )
    messages = [str(w.message) for w in caught if w.category is UserWarning]
    assert any(
        "Plain" in m and "OpenSeesMP" in m
        for m in messages
    ), (
        f"INV-5: expected MP-incompatibility warning for 'Plain' numberer; "
        f"got {messages}"
    )
