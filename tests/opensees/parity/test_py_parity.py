"""Parity tests: PyEmitter vs RecordingEmitter on shared fixtures.

The Py emitter renders ``ops.X(...)`` lines one per Protocol call.
section_close / pattern_close are no-ops in Python (the Py dialect
uses openseespy's stateful current-X), so the Py payload has FEWER
lines than Recording — exactly two fewer per section/pattern block.
"""
from __future__ import annotations

from typing import cast

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
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
        p.load(node=2, forces=(100.0, 0.0, 0.0))
    return ops


def _payload_lines(py_emitter: PyEmitter) -> list[str]:
    """Return non-comment, non-import-line, non-wipe payload."""
    skip_prefixes = ("#", "import ", "ops.wipe(")
    return [
        ln for ln in py_emitter.lines()
        if ln.strip() and not any(ln.startswith(p) for p in skip_prefixes)
    ]


def test_py_emitter_payload_count_vs_recording() -> None:
    """PyEmitter emits one line per Protocol call MINUS section_close +
    pattern_close (those are no-ops in py).

    The fixture has one Fiber section block + one Plain pattern block,
    so PyEmitter has 2 fewer payload lines than the Recording call count.
    """
    rec_ops = _build_minimal_force_beam()
    rec = RecordingEmitter()
    rec_ops.build().emit(rec)

    py_ops = _build_minimal_force_beam()
    py = PyEmitter()
    py_ops.build().emit(py)

    n_rec = len(rec.calls)
    n_close = sum(1 for c in rec.calls
                  if c[0] in ("section_close", "pattern_close"))
    n_py = len(_payload_lines(py))
    assert n_py == n_rec - n_close


def test_py_section_starts_with_ops_section_call() -> None:
    """The Fiber section emits ``ops.section('Fiber', ...)`` BEFORE the
    fiber lines, mirroring openseespy's stateful current-section."""
    ops = _build_minimal_force_beam()
    py = PyEmitter()
    ops.build().emit(py)
    payload = _payload_lines(py)
    idx_section = next(i for i, ln in enumerate(payload)
                       if ln.startswith("ops.section('Fiber'"))
    idx_fiber = next(i for i, ln in enumerate(payload)
                     if ln.startswith("ops.fiber("))
    assert idx_section < idx_fiber


def test_py_pattern_emits_pattern_call_then_load() -> None:
    """``ops.pattern(...)`` precedes ``ops.load(...)`` in the Py output."""
    ops = _build_minimal_force_beam()
    py = PyEmitter()
    ops.build().emit(py)
    payload = _payload_lines(py)
    idx_pat = next(i for i, ln in enumerate(payload)
                   if ln.startswith("ops.pattern('Plain'"))
    idx_load = next(i for i, ln in enumerate(payload)
                    if ln.startswith("ops.load("))
    assert idx_pat < idx_load


def test_py_two_element_fan_out_emits_two_element_calls() -> None:
    """A 2-element PG produces two ``ops.element(...)`` calls."""
    fem = make_two_column_frame()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols",
        transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    py = PyEmitter()
    ops.build().emit(py)
    element_lines = [ln for ln in _payload_lines(py)
                     if ln.startswith("ops.element(")]
    assert len(element_lines) == 2


def test_py_output_is_valid_python_syntactically() -> None:
    """The generated Python must compile to a syntax-valid module."""
    ops = _build_minimal_force_beam()
    py = PyEmitter()
    ops.build().emit(py)
    src = "\n".join(py.lines())
    # ``compile`` with mode='exec' raises SyntaxError if the source
    # is malformed.
    compile(src, "<py-emitter-output>", "exec")
