"""F9 — the held-load augmentation sweep (``LiveOpsEmitter.augment``).

Fork PR #839 §3.2: ``enforce="al"`` advances its Uzawa recursion once per
*committed* step, so a single push leaves the penalty gap standing. The
within-step route that works is the ADR-41 held-load sweep —
``ladrunoBeginAugment`` / ``integrator LoadControl 0.0`` / repeated
``analyze(1)`` polling ``constraintViolation`` / ``ladrunoEndAugment``.

These tests are fork-free: they drive the context manager against a fake
``ops`` that records the call sequence. They lock the three things the fork
measured and that get the sweep wrong when missed:

* the exact call order, with ``LoadControl 0.0`` for the held passes
  whatever drove the real step (a zero-increment ``DisplacementControl``
  is degenerate and often still returns ``ok = 0``),
* ``ladrunoEndAugment`` in ``finally`` — a missed ``End`` does not fail, it
  silently voids every later recorder sample,
* the caller's integrator restored afterwards, and nesting refused.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.emitter.live import LiveOpsEmitter


class _FakeOps:
    """Fork-shaped fake: records every call, serves a decaying gap."""

    def __init__(self, gaps: list[float] | None = None,
                 rcs: list[int] | None = None) -> None:
        self.calls: list[tuple] = []
        self._gaps = list(gaps if gaps is not None else [1e-2, 1e-5, 1e-11])
        self._rcs = list(rcs or [])

    # the fork-build probe ``_stock_build_gate`` reads
    def criticalTimeStep(self) -> float:          # noqa: N802
        return 1.0

    def ladrunoBeginAugment(self) -> None:        # noqa: N802
        self.calls.append(("begin",))

    def ladrunoEndAugment(self) -> None:          # noqa: N802
        self.calls.append(("end",))

    def integrator(self, i_type: str, *args: object) -> None:
        self.calls.append(("integrator", i_type, *args))

    def analyze(self, steps: int) -> int:
        self.calls.append(("analyze", steps))
        return self._rcs.pop(0) if self._rcs else 0

    def eleResponse(self, tag: int, kind: str) -> list[float]:  # noqa: N802
        self.calls.append(("eleResponse", tag, kind))
        return [self._gaps.pop(0) if self._gaps else 0.0]


def _emitter(ops: object) -> LiveOpsEmitter:
    """A ``LiveOpsEmitter`` on a fake ops, bypassing ``__init__`` (no
    openseespy import / wipe). Only what :meth:`augment` reads is set."""
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[attr-defined]
    le._in_partition = False
    le._in_augment = False
    le._last_integrator = None
    return le


def test_sweep_call_sequence_and_integrator_restore() -> None:
    ops = _FakeOps(gaps=[1e-2, 1e-5, 1e-11])
    le = _emitter(ops)
    le.integrator("DisplacementControl", 3, 1, 0.01)   # what drove the step
    with le.augment(element=7, tol=1e-8, max_passes=10) as gaps:
        pass
    assert gaps == [1e-2, 1e-5, 1e-11]
    assert ops.calls == [
        ("integrator", "DisplacementControl", 3, 1, 0.01),
        ("begin",),
        # the held passes hold the LOAD, not the control DOF
        ("integrator", "LoadControl", 0.0),
        ("analyze", 1), ("eleResponse", 7, "constraintViolation"),
        ("analyze", 1), ("eleResponse", 7, "constraintViolation"),
        ("analyze", 1), ("eleResponse", 7, "constraintViolation"),
        ("end",),
        ("integrator", "DisplacementControl", 3, 1, 0.01),
    ]


def test_sweep_stops_at_tol_before_max_passes() -> None:
    ops = _FakeOps(gaps=[1e-9, 1e-12, 1e-14])
    le = _emitter(ops)
    with le.augment(element=1, tol=1e-8, max_passes=10) as gaps:
        pass
    assert gaps == [1e-9]
    assert sum(1 for c in ops.calls if c[0] == "analyze") == 1


def test_sweep_respects_max_passes() -> None:
    ops = _FakeOps(gaps=[1e-2] * 10)
    le = _emitter(ops)
    with le.augment(element=1, tol=1e-8, max_passes=3) as gaps:
        pass
    assert len(gaps) == 3


def test_end_augment_runs_even_when_the_body_raises() -> None:
    # The whole reason for try/finally: while the flag is on
    # Domain::commit() is recorder-silent, so a missed End does not fail —
    # it silently voids every later recorder sample.
    ops = _FakeOps(gaps=[1e-11])
    le = _emitter(ops)
    le.integrator("LoadControl", 1.0)
    with pytest.raises(ZeroDivisionError):
        with le.augment(element=1):
            raise ZeroDivisionError
    assert ("end",) in ops.calls
    assert ops.calls[-1] == ("integrator", "LoadControl", 1.0)
    assert le._in_augment is False


def test_failed_held_pass_fails_loud_and_still_ends() -> None:
    ops = _FakeOps(gaps=[1e-2], rcs=[-3])
    le = _emitter(ops)
    with pytest.raises(RuntimeError, match="held-load pass 1"):
        with le.augment(element=5):
            pass
    assert ("end",) in ops.calls


def test_nesting_is_refused_without_clearing_the_outer_flag() -> None:
    ops = _FakeOps(gaps=[1e-11, 1e-11])
    le = _emitter(ops)
    with le.augment(element=1):
        with pytest.raises(RuntimeError, match="already open"):
            with le.augment(element=1):
                pass
        # the refused inner block must NOT have ended the outer sweep
        assert ("end",) not in ops.calls
    assert ops.calls.count(("end",)) == 1


def test_no_integrator_seen_leaves_loadcontrol_zero_in_place() -> None:
    # Nothing to restore when the caller never went through this emitter's
    # ``integrator`` — better than guessing at a default.
    ops = _FakeOps(gaps=[1e-11])
    le = _emitter(ops)
    with le.augment(element=1):
        pass
    assert ops.calls[-1] == ("end",)


def test_stock_build_is_refused() -> None:
    class _StockOps:
        pass                       # no ``criticalTimeStep``

    le = _emitter(_StockOps())
    with pytest.raises(RuntimeError, match="requires the Ladruno fork build"):
        with le.augment(element=1):
            pass
