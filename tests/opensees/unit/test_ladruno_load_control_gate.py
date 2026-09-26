"""Live-build gate for ``integrator LadrunoLoadControl`` (fork ADR-80).

Fork-free: drives :meth:`LiveOpsEmitter.integrator` against fake ``ops``
modules shaped like the four builds that matter. The first three are the
builds that would otherwise run the wrong integrator without an error:

* **stock openseespy** — refused by the shared fork-integrator gate;
* **a fork build older than 2026-08-04** — has ``criticalTimeStep`` but not
  the integrator. Its ``integrator`` command warns and KEEPS THE PREVIOUS
  integrator (probed on the 2026-06-25 build), so it must be refused
  before the call;
* **an S1-only fork build (2026-08-04 .. 2026-09-04)** — knows the
  integrator but ignores ``-tangentPredictor`` with a warning and runs
  stock LoadControl; its ``ladrunoLoadControl`` command rejects the
  ``tangentPredictor`` query;
* **a current fork build** — passes, and reports the predictor armed.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees.emitter.live import LiveOpsEmitter


class _StockOps:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def integrator(self, i_type: str, *args: object) -> None:
        self.calls.append(("integrator", i_type, *args))


class _OldForkOps(_StockOps):
    def criticalTimeStep(self) -> float:          # noqa: N802
        return 1.0


class _S1ForkOps(_OldForkOps):
    """Has the integrator and runtime command, not the P3 query."""

    def ladrunoLoadControl(self, *args: object) -> float:  # noqa: N802
        if args and args[0] == "tangentPredictor":
            raise RuntimeError("See stderr output")   # unknown subcommand
        return 0.1


class _CurrentForkOps(_OldForkOps):
    def __init__(self) -> None:
        super().__init__()
        self._armed = 0.0

    def integrator(self, i_type: str, *args: object) -> None:
        super().integrator(i_type, *args)
        self._armed = 1.0 if "-tangentPredictor" in args else 0.0

    def ladrunoLoadControl(self, *args: object) -> float:  # noqa: N802
        if args and args[0] == "tangentPredictor":
            return self._armed
        return 0.1


def _emitter(ops: object, *, in_partition: bool = False) -> LiveOpsEmitter:
    """A ``LiveOpsEmitter`` on a fake ops, bypassing ``__init__``."""
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = in_partition
    le._last_integrator = None
    return le


def test_stock_build_refused() -> None:
    ops = _StockOps()
    with pytest.raises(RuntimeError, match="requires the Ladruno fork"):
        _emitter(ops).integrator("LadrunoLoadControl", 0.1, "-tangentPredictor")
    assert ops.calls == []


def test_fork_without_the_integrator_refused_before_the_call() -> None:
    ops = _OldForkOps()
    with pytest.raises(RuntimeError, match="fork without it"):
        _emitter(ops).integrator("LadrunoLoadControl", 0.1, "-tangentPredictor")
    assert ops.calls == []   # the silently-ignored call was never made


def test_s1_only_build_refused_when_the_predictor_is_asked_for() -> None:
    ops = _S1ForkOps()
    with pytest.raises(RuntimeError, match="was not honoured"):
        _emitter(ops).integrator("LadrunoLoadControl", 0.1, "-tangentPredictor")


def test_s1_only_build_runs_without_the_predictor() -> None:
    ops = _S1ForkOps()
    _emitter(ops).integrator("LadrunoLoadControl", 0.1)
    assert ops.calls == [("integrator", "LadrunoLoadControl", 0.1)]


def test_current_build_passes_and_records_the_integrator() -> None:
    ops = _CurrentForkOps()
    le = _emitter(ops)
    le.integrator("LadrunoLoadControl", 0.1, "-tangentPredictor")
    assert ops.calls == [
        ("integrator", "LadrunoLoadControl", 0.1, "-tangentPredictor"),
    ]
    assert le._last_integrator == (
        "LadrunoLoadControl", 0.1, "-tangentPredictor",
    )


def test_partition_block_skips_the_probe() -> None:
    # Inside a non-zero partition block ``_ops`` is the no-op stand-in,
    # which drives no domain; the gate must not interrogate it.
    ops = _OldForkOps()
    _emitter(ops, in_partition=True).integrator(
        "LadrunoLoadControl", 0.1, "-tangentPredictor",
    )
    assert ops.calls == [
        ("integrator", "LadrunoLoadControl", 0.1, "-tangentPredictor"),
    ]
