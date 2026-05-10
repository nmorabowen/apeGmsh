"""
Typed ``timeSeries`` primitives.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``timeSeries <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.time_series._TimeSeriesNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

Phase 1D-extra ships the OpenSees core 5:

  * :class:`Linear`   — ramp        (``timeSeries Linear``)
  * :class:`Constant` — step        (``timeSeries Constant``)
  * :class:`Path`     — time history (``timeSeries Path``)
  * :class:`Trig`     — sinusoid    (``timeSeries Trig``)
  * :class:`Pulse`    — square wave (``timeSeries Pulse``)

The cyclic-loading-protocol classes (``ASCE41Protocol``, ``FEMA461Protocol``,
``ATC24Protocol``) are deferred to a follow-up slice. They are
conceptually peers of TimeSeries — each emits a ``timeSeries Path`` with
a programmatically built ``time`` and ``values`` array — so they slot
into the same module without churn when added.

See ADR 0007 for why ``time_series/`` is separated from ``pattern/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._internal.types import Primitive, TimeSeries

if TYPE_CHECKING:
    from ..emitter.base import Emitter


__all__ = [
    "Linear",
    "Constant",
    "Path",
    "Trig",
    "Pulse",
]


# ---------------------------------------------------------------------------
# Linear — ramp
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Linear(TimeSeries):
    """``timeSeries Linear tag [-factor f]`` — linear ramp ``factor * t``.

    The simplest TimeSeries: returns ``factor * t`` at time ``t``.
    Common use: pseudo-static load control where the pseudo-time is the
    load factor.
    """

    factor: float = 1.0

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.factor == 1.0:
            emitter.timeSeries("Linear", tag)
        else:
            emitter.timeSeries("Linear", tag, "-factor", self.factor)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Constant — step
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Constant(TimeSeries):
    """``timeSeries Constant tag [-factor f]`` — constant value ``factor``.

    Returns ``factor`` for all time. Useful as the time-series of a
    sustained gravity load that does not vary with the analysis step.
    """

    factor: float = 1.0

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        if self.factor == 1.0:
            emitter.timeSeries("Constant", tag)
        else:
            emitter.timeSeries("Constant", tag, "-factor", self.factor)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Path — explicit time history (the workhorse for ground motions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Path(TimeSeries):
    """``timeSeries Path tag (-filePath s | -values v...) [-time t...]
    [-dt dt] [-factor f] [-startTime t0] [-prependZero]``.

    Reads a time-history of values, with the time axis given either by
    a uniform ``dt`` or by an explicit ``time`` sequence.

    Exactly one of ``file`` or ``values`` must be supplied. When
    ``values`` is supplied, exactly one of ``dt`` or ``time`` is also
    required. When ``file`` is supplied, ``dt`` and ``time`` are both
    optional (the file may already encode time / be uniformly sampled
    with a separate ``dt``).
    """

    file: str | None = None
    values: tuple[float, ...] | None = None
    time: tuple[float, ...] | None = None
    dt: float | None = None
    factor: float = 1.0
    start_time: float = 0.0
    prepend_zero: bool = False

    def __post_init__(self) -> None:
        # Exactly one of (file, values) must be set.
        if (self.file is None) == (self.values is None):
            raise ValueError(
                "Path: supply exactly one of file= or values= "
                f"(got file={self.file!r}, values={self.values!r})."
            )
        # Either dt OR explicit time, not both.
        if self.dt is not None and self.time is not None:
            raise ValueError("Path: supply dt= or time=, not both.")
        # When supplying values=, time information is mandatory.
        if (
            self.values is not None
            and self.dt is None
            and self.time is None
        ):
            raise ValueError(
                "Path: when supplying values=, also supply dt= or time=."
            )
        if self.factor <= 0:
            raise ValueError(
                f"Path: factor must be > 0, got {self.factor!r}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[float | str] = []
        if self.file is not None:
            args += ["-filePath", self.file]
        else:
            assert self.values is not None  # __post_init__ guarantee
            args += ["-values", *self.values]
        if self.dt is not None:
            args += ["-dt", self.dt]
        elif self.time is not None:
            args += ["-time", *self.time]
        if self.factor != 1.0:
            args += ["-factor", self.factor]
        if self.start_time != 0.0:
            args += ["-startTime", self.start_time]
        if self.prepend_zero:
            args += ["-prependZero"]
        emitter.timeSeries("Path", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Trig — sinusoid
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Trig(TimeSeries):
    """``timeSeries Trig tag tStart tEnd period [-factor f]
    [-shift phi] [-zeroShift v]``.

    Sinusoidal time series, active for ``t_start <= t <= t_end`` and
    zero outside that window. The default phase ``shift`` is ``0`` and
    the default zero-baseline ``zero_shift`` is ``0``.
    """

    t_start: float
    t_end: float
    period: float
    factor: float = 1.0
    shift: float = 0.0
    zero_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(
                f"Trig: period must be > 0, got {self.period!r}"
            )
        if self.t_end <= self.t_start:
            raise ValueError(
                "Trig: t_end must be > t_start, got "
                f"t_start={self.t_start!r}, t_end={self.t_end!r}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[float | str] = [self.t_start, self.t_end, self.period]
        if self.factor != 1.0:
            args += ["-factor", self.factor]
        if self.shift != 0.0:
            args += ["-shift", self.shift]
        if self.zero_shift != 0.0:
            args += ["-zeroShift", self.zero_shift]
        emitter.timeSeries("Trig", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Pulse — square wave
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Pulse(TimeSeries):
    """``timeSeries Pulse tag tStart tEnd period width [-factor f]
    [-shift phi] [-zeroShift v]``.

    Square-wave pulse train, active for ``t_start <= t <= t_end``.
    ``width`` is the duty fraction in ``(0, 1)`` — fraction of each
    period during which the pulse is "on".
    """

    t_start: float
    t_end: float
    period: float
    width: float
    factor: float = 1.0
    shift: float = 0.0
    zero_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError(
                f"Pulse: period must be > 0, got {self.period!r}"
            )
        if not 0.0 < self.width < 1.0:
            raise ValueError(
                "Pulse: width must be in (0, 1), got "
                f"{self.width!r}"
            )
        if self.t_end <= self.t_start:
            raise ValueError(
                "Pulse: t_end must be > t_start, got "
                f"t_start={self.t_start!r}, t_end={self.t_end!r}"
            )

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        args: list[float | str] = [
            self.t_start, self.t_end, self.period, self.width,
        ]
        if self.factor != 1.0:
            args += ["-factor", self.factor]
        if self.shift != 0.0:
            args += ["-shift", self.shift]
        if self.zero_shift != 0.0:
            args += ["-zeroShift", self.zero_shift]
        emitter.timeSeries("Pulse", tag, *args)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
