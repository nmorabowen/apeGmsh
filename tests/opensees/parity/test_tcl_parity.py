"""Parity tests: TclEmitter vs RecordingEmitter on shared fixtures.

Drives the same model through both emitters and verifies the call
counts agree (per-line in Tcl, per-call in Recording, modulo block
braces).
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.fixtures.fem_stub import (
    make_two_column_frame,
    make_two_node_beam,
)


def _build_minimal_force_beam() -> apeSees:
    fem = make_two_node_beam()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(100.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    return ops


def _payload_lines(tcl_emitter: TclEmitter) -> list[str]:
    """Return non-comment Tcl payload lines (skip banner / blank)."""
    return [
        ln for ln in tcl_emitter.lines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def test_tcl_emitter_payload_matches_recording_call_count() -> None:
    """Each non-block emitter call produces exactly one Tcl payload line.

    Section / pattern open + close pairs add one ``\\{`` / ``\\}`` line
    each on the Tcl side; we expect ``recording_calls + brace_pairs``
    Tcl payload lines.
    """
    rec_ops = _build_minimal_force_beam()
    rec = RecordingEmitter()
    rec_ops.build().emit(rec)

    tcl_ops = _build_minimal_force_beam()
    tcl = TclEmitter()
    tcl_ops.build().emit(tcl)

    # Count Recording method calls and the *_open events that produce
    # a block in Tcl.
    n_rec = len(rec.calls)
    # Each section_open/section_close pair contributes 0 extra Tcl
    # lines beyond the 2 in Recording (Tcl renders the open as
    # 'section ... \\{' on one line, the close as '\\}' on another).
    # Same for block patterns. So Tcl payload lines == Recording calls.
    n_tcl = len(_payload_lines(tcl))
    assert n_tcl == n_rec, (
        f"Tcl payload lines ({n_tcl}) != Recording calls ({n_rec}).\n"
        f"Tcl: {_payload_lines(tcl)}\n"
        f"Rec: {[c[0] for c in rec.calls]}"
    )


def test_tcl_section_open_appears_with_brace_after_material() -> None:
    """``uniaxialMaterial`` line appears before ``section ... \\{``."""
    ops = _build_minimal_force_beam()
    tcl = TclEmitter()
    ops.build().emit(tcl)

    payload = _payload_lines(tcl)
    idx_mat = next(i for i, ln in enumerate(payload)
                   if ln.startswith("uniaxialMaterial Steel02"))
    idx_sec = next(i for i, ln in enumerate(payload)
                   if ln.startswith("section Fiber"))
    assert idx_mat < idx_sec
    assert payload[idx_sec].endswith("{")
    # Closing brace later on.
    idx_close = next(i for i, ln in enumerate(payload[idx_sec + 1:], idx_sec + 1)
                     if ln.strip() == "}")
    assert idx_close > idx_sec


def test_tcl_pattern_block_with_load_indented_inside() -> None:
    """``pattern Plain ... \\{`` opens, ``load`` is indented inside,
    ``\\}`` closes."""
    ops = _build_minimal_force_beam()
    tcl = TclEmitter()
    ops.build().emit(tcl)

    payload = _payload_lines(tcl)
    idx_pat_open = next(i for i, ln in enumerate(payload)
                        if ln.startswith("pattern Plain"))
    assert payload[idx_pat_open].endswith("{")
    # Inner load lines are indented.
    inner_loads = [ln for ln in payload[idx_pat_open + 1:]
                   if ln.startswith("    load ")]
    assert len(inner_loads) >= 1


def test_tcl_two_element_fan_out_emits_two_element_lines() -> None:
    """A 2-element PG produces 2 ``element`` lines in the Tcl deck."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    tcl = TclEmitter()
    ops.build().emit(tcl)

    element_lines = [ln for ln in _payload_lines(tcl)
                     if ln.startswith("element ")]
    assert len(element_lines) == 2
