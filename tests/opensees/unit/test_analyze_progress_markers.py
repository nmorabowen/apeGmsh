"""Progress-marker injection in the emitted analyze loop.

``apeSees.tcl(progress=)`` / ``apeSees.py(progress=)`` inject a
throttled ``APEGMSH_PROGRESS`` line (~20 samples over the run) into the
per-increment analyze loop so the ``run=True`` streamer can render a
live step counter. The marker is:

* **off by default** on a bare emitter (decks stay clean),
* emitted ~every ``n // 20`` increments plus the final one,
* present in BOTH the plain and strategy-ladder loop branches,
* carries the increment, total, and pseudo-time, and flushes stdout so
  it streams live through the subprocess pipe.
"""
from __future__ import annotations

from apeGmsh.opensees.emitter.base import StrategySpec
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


def _tcl(steps: int, *, progress: bool, strategy: StrategySpec | None = None) -> str:
    e = TclEmitter()
    e._emit_progress = progress
    e.analyze(steps=steps, dt=0.01, strategy=strategy)
    return "\n".join(e.lines())


def _py(steps: int, *, progress: bool, strategy: StrategySpec | None = None) -> str:
    e = PyEmitter()
    e._emit_progress = progress
    e.analyze(steps=steps, dt=0.01, strategy=strategy)
    return "\n".join(e.lines())


# --- default: no markers --------------------------------------------------

def test_tcl_no_marker_by_default() -> None:
    assert "APEGMSH_PROGRESS" not in _tcl(100, progress=False)


def test_py_no_marker_by_default() -> None:
    assert "APEGMSH_PROGRESS" not in _py(100, progress=False)


# --- marker shape + cadence ----------------------------------------------

def test_tcl_marker_emitted_with_cadence_and_flush() -> None:
    deck = _tcl(100, progress=True)
    # 100 // 20 == 5
    assert "% 5}] == 0" in deck
    assert "$_apesees_i + 1}] == 100" in deck
    assert 'puts "APEGMSH_PROGRESS i=[expr {$_apesees_i + 1}] n=100 t=[getTime]"' in deck
    assert "flush stdout" in deck


def test_py_marker_emitted_with_cadence_and_flush() -> None:
    deck = _py(60, progress=True)
    # 60 // 20 == 3
    assert "(_apesees_i + 1) % 3 == 0 or _apesees_i + 1 == 60:" in deck
    assert 'print("APEGMSH_PROGRESS i=%d n=60 t=%g"' in deck
    assert "flush=True" in deck


def test_cadence_floor_is_one_for_short_runs() -> None:
    # n < 20 -> every = max(1, n // 20) == 1 (a marker every increment)
    assert "% 1}] == 0" in _tcl(5, progress=True)
    assert "(_apesees_i + 1) % 1 == 0" in _py(5, progress=True)


# --- strategy-ladder branch also carries the marker -----------------------

def _ladder() -> StrategySpec:
    return StrategySpec(name="lad", rungs=[("Newton",), ("ModifiedNewton",)])


def test_tcl_marker_in_strategy_branch() -> None:
    assert "APEGMSH_PROGRESS" in _tcl(40, progress=True, strategy=_ladder())


def test_py_marker_in_strategy_branch() -> None:
    assert "APEGMSH_PROGRESS" in _py(40, progress=True, strategy=_ladder())
