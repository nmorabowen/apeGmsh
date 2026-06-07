"""
Typed ``timeSeries`` primitives.

Each class is a ``@dataclass(frozen=True, kw_only=True, slots=True)``
mirroring the OpenSees Tcl ``timeSeries <Type> ...`` command. The
matching :class:`apeGmsh.opensees._internal.ns.time_series._TimeSeriesNS`
methods take the same kwargs and call ``self._bridge._register(Cls(...))``.

The OpenSees core 5:

  * :class:`Linear`   — ramp        (``timeSeries Linear``)
  * :class:`Constant` — step        (``timeSeries Constant``)
  * :class:`Path`     — time history (``timeSeries Path``)
  * :class:`Trig`     — sinusoid    (``timeSeries Trig``)
  * :class:`Pulse`    — square wave (``timeSeries Pulse``)

Plus apeGmsh-native generators that have no dedicated OpenSees command —
each stores its knobs and expands to a ``timeSeries Path`` (a
programmatically built ``time`` / ``values`` array) at emit:

  * :class:`Ricker`                — Gaussian-derivative wavelet
  * :class:`ASCE41Protocol`        — ASCE 41 cyclic displacement protocol
  * :class:`ModifiedATC24Protocol` — modified ATC-24 cyclic protocol
  * :class:`FEMA461Protocol`       — FEMA 461 cyclic protocol

See ADR 0007 for why ``time_series/`` is separated from ``pattern/``.
"""
from __future__ import annotations

import math
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
    "Ricker",
    "ASCE41Protocol",
    "ModifiedATC24Protocol",
    "FEMA461Protocol",
]


# ---------------------------------------------------------------------------
# Shared builder for cyclic displacement protocols
# ---------------------------------------------------------------------------

def _cyclic_displacement_path(
    amplitudes: tuple[float, ...],
    reps: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Build a normalized cyclic displacement protocol.

    Returns ``(time, values)`` for a sequence of fully-reversed cycles:
    each amplitude ``a`` contributes ``reps`` ``[+a, -a]`` pairs, framed
    by a leading and trailing zero. ``values`` are normalized so the
    largest peak is ``±1`` (the caller's ``factor`` carries the physical
    amplitude); ``time`` is a constant-slope pseudo-time on ``[0, 1]``
    built from the cumulative absolute increment, so every ramp segment
    advances at the same rate.
    """
    peak = max(amplitudes)
    values = [0.0]
    for a, n in zip(amplitudes, reps):
        amp = a / peak
        for _ in range(n):
            values += [amp, -amp]
    values.append(0.0)

    times = [0.0]
    acc = 0.0
    for i in range(1, len(values)):
        acc += abs(values[i] - values[i - 1])
        times.append(acc)
    t_max = times[-1]
    if t_max > 1e-10:
        times = [t / t_max for t in times]
    return tuple(times), tuple(values)


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


# ---------------------------------------------------------------------------
# Ricker — Gaussian-derivative wavelet, emitted as a Path
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class Ricker(TimeSeries):
    """A Ricker (Gaussian-derivative) wavelet, emitted as ``timeSeries Path``.

    OpenSees has no native wavelet command, so this primitive is a thin
    *declarative façade*: it stores the wavelet knobs, samples the closed
    form onto a uniform ``dt`` grid at emit time, and delegates to
    :class:`Path` (``timeSeries Path tag -values ... -dt dt``). Nothing
    new reaches OpenSees — the deck is an ordinary tabulated path.

    The wavelet rides a Gaussian envelope ``exp(-π² fₙ² τ²)`` with
    ``τ = t - t_center``; ``kind`` selects which kinematic order:

      * ``"acceleration"`` — the Ricker / Mexican-hat (2nd derivative of
        the Gaussian): ``(1 - 2 π² fₙ² τ²) · exp(-π² fₙ² τ²)``.
      * ``"velocity"`` — the 1st-derivative form: ``τ · exp(-π² fₙ² τ²)``.

    Because the form is analytic, the peak is placed by evaluating on a
    shifted axis (``t_center``) — no sample-rolling / wrap-around. Choose
    ``t_center`` a few wavelengths from ``0`` and ``t_total`` so the
    Gaussian tails decay to ~0 inside the window.

    Parameters
    ----------
    f_n
        Central frequency (Hz).
    t_total
        History duration (s); the number of samples is ``round(t_total/dt)``.
    dt
        Sample step (s) — becomes the Path ``-dt``.
    t_center
        Time of the wavelet peak (s). Default ``0.0``.
    kind
        ``"acceleration"`` (Ricker) or ``"velocity"`` (1st-derivative).
    factor
        Amplitude scale, passed through to Path ``-factor`` (OpenSees
        scales the tabulated ordinates at runtime). Default ``1.0``.
    """

    f_n: float
    t_total: float
    dt: float
    t_center: float = 0.0
    kind: str = "acceleration"
    factor: float = 1.0

    _KINDS = ("acceleration", "velocity")

    def __post_init__(self) -> None:
        if self.f_n <= 0:
            raise ValueError(f"Ricker: f_n must be > 0, got {self.f_n!r}")
        if self.t_total <= 0:
            raise ValueError(
                f"Ricker: t_total must be > 0, got {self.t_total!r}"
            )
        if self.dt <= 0:
            raise ValueError(f"Ricker: dt must be > 0, got {self.dt!r}")
        if self.kind not in self._KINDS:
            raise ValueError(
                f"Ricker: kind must be one of {self._KINDS}, got {self.kind!r}"
            )
        if self.factor <= 0:
            raise ValueError(
                f"Ricker: factor must be > 0, got {self.factor!r}"
            )
        if round(self.t_total / self.dt) < 2:
            raise ValueError(
                "Ricker: need at least 2 samples — t_total/dt = "
                f"{self.t_total / self.dt!r} (lower dt or raise t_total)."
            )

    def samples(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(times, values)`` — the raw wavelet ordinates.

        These are the exact numbers emitted as Path ``-values`` (the
        ``factor`` is *not* applied here; OpenSees applies it at runtime
        via ``-factor``). Handy for plotting the pulse before it goes
        into a pattern.
        """
        n = round(self.t_total / self.dt)
        c = math.pi ** 2 * self.f_n ** 2          # π² fₙ²
        times: list[float] = []
        values: list[float] = []
        for i in range(n):
            t = i * self.dt
            tau = t - self.t_center
            env = math.exp(-c * tau * tau)
            if self.kind == "acceleration":
                v = (1.0 - 2.0 * c * tau * tau) * env
            else:  # "velocity"
                v = tau * env
            times.append(t)
            values.append(v)
        return tuple(times), tuple(values)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        _, values = self.samples()
        # Delegate to Path: this IS a tabulated path time series.
        Path(values=values, dt=self.dt, factor=self.factor)._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


# ---------------------------------------------------------------------------
# Cyclic loading protocols — quasi-static, emitted as a Path
# ---------------------------------------------------------------------------
#
# Each protocol is a displacement-controlled cyclic test history (ASCE 41,
# ATC-24, FEMA 461). Like Ricker, they store knobs and expand to a
# ``timeSeries Path`` at emit. The shape is normalized to a ±1 peak; the
# physical peak amplitude (max displacement / strain) rides in ``factor``,
# so one protocol object is reusable at any amplitude.


@dataclass(frozen=True, kw_only=True, slots=True)
class ASCE41Protocol(TimeSeries):
    """ASCE 41 cyclic displacement protocol (emitted as ``timeSeries Path``).

    A fixed amplitude ladder (0.25 % → 6 % of peak) with decreasing
    repetitions, framed by leading/trailing zeros. The shape is
    normalized to ±1; ``factor`` is the peak displacement / strain.
    """

    factor: float = 1.0

    _AMPLITUDES = (
        0.0025, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040, 0.060,
    )
    _REPS = (3, 3, 3, 3, 3, 3, 2, 2, 2)

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError(
                f"ASCE41Protocol: factor must be > 0, got {self.factor!r}"
            )

    def samples(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(time, values)`` — the normalized ±1 protocol history."""
        return _cyclic_displacement_path(self._AMPLITUDES, self._REPS)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        time, values = self.samples()
        Path(values=values, time=time, factor=self.factor)._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class ModifiedATC24Protocol(TimeSeries):
    """Modified ATC-24 cyclic displacement protocol (emitted as ``Path``).

    Amplitude levels ``[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`` of peak with
    repetitions ``[3, 3, 3, 2, 2, 1]``. Normalized to ±1; ``factor`` is
    the peak displacement / strain.
    """

    factor: float = 1.0

    _AMPLITUDES = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
    _REPS = (3, 3, 3, 2, 2, 1)

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError(
                f"ModifiedATC24Protocol: factor must be > 0, "
                f"got {self.factor!r}"
            )

    def samples(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(time, values)`` — the normalized ±1 protocol history."""
        return _cyclic_displacement_path(self._AMPLITUDES, self._REPS)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        time, values = self.samples()
        Path(values=values, time=time, factor=self.factor)._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class FEMA461Protocol(TimeSeries):
    """FEMA 461 cyclic displacement protocol (emitted as ``timeSeries Path``).

    Amplitudes grow geometrically by ``(1 + alpha)`` from ``start_fraction``
    of the peak up to the peak, with **two cycles at each amplitude** per
    FEMA 461 §2.2. The default ``alpha = 0.4`` gives the standard's ~1.4×
    target increment.

    .. note::
       This reconciles the ported apeSees version, which did a single
       cycle per amplitude with ``alpha = 0.62``. The shape is normalized
       to ±1; ``factor`` is the peak displacement / strain.
    """

    factor: float = 1.0
    alpha: float = 0.4
    start_fraction: float = 0.01

    _CYCLES_PER_STEP = 2  # FEMA 461 §2.2

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError(
                f"FEMA461Protocol: factor must be > 0, got {self.factor!r}"
            )
        if self.alpha <= 0:
            raise ValueError(
                f"FEMA461Protocol: alpha must be > 0, got {self.alpha!r}"
            )
        if not 0.0 < self.start_fraction < 1.0:
            raise ValueError(
                "FEMA461Protocol: start_fraction must be in (0, 1), got "
                f"{self.start_fraction!r}"
            )

    def _ladder(self) -> tuple[tuple[float, ...], tuple[int, ...]]:
        amps: list[float] = []
        a = self.start_fraction
        while a < 1.0:
            amps.append(a)
            a *= 1.0 + self.alpha
        amps.append(1.0)  # land the peak exactly
        reps = tuple([self._CYCLES_PER_STEP] * len(amps))
        return tuple(amps), reps

    def samples(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return ``(time, values)`` — the normalized ±1 protocol history."""
        amps, reps = self._ladder()
        return _cyclic_displacement_path(amps, reps)

    def _emit(self, emitter: "Emitter", tag: int) -> None:
        time, values = self.samples()
        Path(values=values, time=time, factor=self.factor)._emit(emitter, tag)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()
