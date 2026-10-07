"""Warn when a stage pattern's ``Path`` series is zero at every increment,
or ends inside the stage and unloads there.

Every stage closes with ``loadConst -time 0.0``, so the next stage's
pseudo-time restarts at 0 unless ``s.set_time(t)`` moves it.  A ``Path``
series written on a global clock across stages (stage 2 meant to run
over ``t`` in ``[1, 2]``) then reads 0 over the whole of stage 2: the
deck exits 0 having applied none of that pattern's loads (#1334).

The bridge knows both the stage's clock and the series at declaration,
so :func:`warn_stage_series_outside_window` evaluates the series at the
pseudo-time of every increment when a stage closes (it is called from
``StageRecord.__post_init__`` in ``build.py``) and raises
:class:`SeriesOutsideStageWindowWarning` when every one of those factors
is zero.

Otherwise it raises :class:`SeriesEndsMidStageWarning` when some
increment lies past the series' last sample while that sample is
non-zero and the series has no ``use_last``: OpenSees reads the series
as 0 there, so the pattern drops its load mid-stage (#1363).  The two
warnings are exclusive, at most one per pattern.

The increments' pseudo-times are reproduced the way OpenSees produces
them: ``setTime`` (``s.set_time``, else the 0 the previous
``loadConst -time 0.0`` left; ``s.reset()`` before the analyze loop puts
it back at 0), then ``t += h`` once per increment, with ``h`` the stage's
``dt`` under ``Transient`` or the integrator's ``dlam`` under ``Static``
with ``LoadControl`` / ``LadrunoLoadControl``.  An analysis that solves
for its advance (any other static integrator, an adaptive load control
with ``min_lam``/``max_lam``, ``VariableTransient``) has no pseudo-times
before the run, and its stage is not checked.

The series is evaluated as ``PathSeries.cpp`` / ``PathTimeSeries.cpp`` do:

* ``time=`` (``PathTimeSeries``): linear between the points, 0 before
  ``time[0]`` and after ``time[-1]``.  ``Path`` refuses ``start_time``,
  ``prepend_zero`` and ``use_last`` on this route (OpenSees drops them).
* ``dt=`` (``PathSeries``): sample ``k`` sits at ``start_time + k*dt``,
  after a leading 0 when ``prepend_zero``; the factor is 0 before
  ``start_time`` and from the last sample's time on (at that time too:
  ``incr2 >= size``), unless ``use_last`` holds the last value there.
* ``file=``: the samples are in a file the deck reads at run time, so the
  series is not checked.
"""
from __future__ import annotations

import inspect
import os
import warnings
from types import FrameType
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.analysis import Static, Transient, VariableTransient
from ..analysis.integrator import LadrunoLoadControl, LoadControl
from ..time_series.time_series import Path

if TYPE_CHECKING:
    from .build import StageRecord


__all__ = [
    "SeriesOutsideStageWindowWarning",
    "SeriesEndsMidStageWarning",
    "stage_increment_times",
    "path_factor",
    "warn_stage_series_outside_window",
]


class SeriesOutsideStageWindowWarning(UserWarning):
    """A stage pattern's ``Path`` series is zero at every increment of the
    stage, so that pattern applies no load (#1334).

    Each stage starts at pseudo-time 0 (the previous stage closes with
    ``loadConst -time 0.0``) unless ``s.set_time(t)`` moves it.  Fix the
    stage clock with ``s.set_time(t)``, or re-base the series' time axis
    onto the stage's window.
    """


class SeriesEndsMidStageWarning(UserWarning):
    """A stage pattern's ``Path`` series ends inside the stage on a
    non-zero value and has no ``use_last``, so the pattern drops its load
    to 0 from the first increment past the last sample (#1363).

    Not a subclass of :class:`SeriesOutsideStageWindowWarning`, whose
    contract is "warns iff the pattern applies no load at all": here the
    pattern does load, then unloads.  Fix: ``use_last=True`` (``dt=``
    series) to hold the last value, a series that reaches the end of the
    stage, or a trailing 0 sample when the drop is intended (a record
    followed by free vibration).
    """


def _stage_t0(stage: "StageRecord") -> float:
    """Pseudo-time the stage's analyze loop starts from."""
    if stage.set_time is not None and not stage.pre_analyze_reset:
        return float(stage.set_time)
    return 0.0


def stage_increment_times(stage: "StageRecord") -> "np.ndarray | None":
    """Pseudo-time of each of the stage's increments, as OpenSees reaches it.

    ``None`` when the analysis solves for its advance, so the times do not
    exist before the run: ``VariableTransient``, a ``Transient`` stage
    without ``dt``, an adaptive load control (``min_lam``/``max_lam``),
    and ``Static`` under any other integrator.  Any other analysis object
    raises: this check has to learn how a new analysis class advances
    pseudo-time.
    """
    t0 = _stage_t0(stage)
    analysis = stage.analysis
    if isinstance(analysis, Transient):
        if stage.dt is None:
            return None
        h = float(stage.dt)
    elif isinstance(analysis, Static):
        integ = stage.integrator
        if not isinstance(integ, (LoadControl, LadrunoLoadControl)):
            return None
        if integ.min_lam is not None:
            return None
        h = float(integ.dlam)
    elif isinstance(analysis, VariableTransient):
        return None
    else:
        raise TypeError(
            f"Stage {stage.name!r}: cannot tell how analysis "
            f"{type(analysis).__name__} advances pseudo-time; teach "
            "apeGmsh.opensees._internal.stage_window.stage_increment_times "
            "about it."
        )
    # Sequential float adds, the order the integrators use (``t += h``),
    # so a time that lands on a support boundary lands where OpenSees'
    # does.
    steps = np.full(int(stage.n_increments) + 1, h, dtype=float)
    steps[0] = t0
    return np.cumsum(steps)[1:]


def path_factor(series: Path, t: np.ndarray) -> np.ndarray:
    """The series' load factor at pseudo-times ``t``, as OpenSees
    evaluates it (module docstring).  Caller guarantees ``values=``."""
    assert series.values is not None
    values = np.asarray(series.values, dtype=float)
    # Past the last sample: 0, or the last value under ``-useLast``.
    tail = float(values[-1]) if series.use_last else 0.0
    if series.time is not None:
        times = np.asarray(series.time, dtype=float)
        out = np.interp(t, times, values, left=0.0, right=tail)
    else:
        assert series.dt is not None  # Path.__post_init__ guarantee
        if series.prepend_zero:
            values = np.concatenate(([0.0], values))
        incr = (t - float(series.start_time)) / float(series.dt)
        i1 = np.floor(incr)
        live = (t >= float(series.start_time)) & (i1 + 1 < values.size)
        j = np.where(live, i1, 0).astype(np.int64)
        v1 = values[j]
        v2 = values[np.minimum(j + 1, values.size - 1)]
        out = np.where(live, v1 + (v2 - v1) * (incr - i1),
                       np.where(_past_end(series, t), tail, 0.0))
    factors: np.ndarray = float(series.factor) * np.asarray(out, dtype=float)
    return factors


def _past_end(series: Path, t: np.ndarray) -> np.ndarray:
    """Which pseudo-times ``t`` lie past the series' last sample, where
    OpenSees returns 0 (or the last value under ``-useLast``).

    ``time=``: after ``time[-1]``.  ``dt=``: at or after the last sample's
    time, ``incr2 >= size`` in ``PathSeries::getFactor``.
    """
    assert series.values is not None
    if series.time is not None:
        past: np.ndarray = t > float(series.time[-1])
        return past
    assert series.dt is not None
    n = len(series.values) + (1 if series.prepend_zero else 0)
    incr = (t - float(series.start_time)) / float(series.dt)
    past = (t >= float(series.start_time)) & (np.floor(incr) + 1 >= n)
    return past


def _support(series: Path) -> str:
    assert series.values is not None
    if series.time is not None:
        t = series.time
        return f"time=, support [{t[0]:g}, {t[-1]:g}]"
    assert series.dt is not None
    n = len(series.values) + (1 if series.prepend_zero else 0)
    end = float(series.start_time) + (n - 1) * float(series.dt)
    return f"dt={series.dt:g}, support [{series.start_time:g}, {end:g})"


def _onset(series: Path) -> "float | None":
    """Pseudo-time where the series leaves zero, ``None`` if it never does."""
    assert series.values is not None
    values = np.asarray(series.values, dtype=float)
    nz = np.flatnonzero(values)
    if not nz.size:
        return None
    k = max(int(nz[0]) - 1, 0)
    if series.time is not None:
        return float(series.time[k])
    assert series.dt is not None
    if series.prepend_zero:
        k = int(nz[0])  # the prepended 0 shifts every sample by one dt
    return float(series.start_time) + k * float(series.dt)


def _stacklevel() -> int:
    """``stacklevel`` naming the first frame outside ``apeGmsh``, counted
    from the ``warnings.warn`` call in this module.  A dataclass's
    generated ``__init__`` (filename ``<string>``) that the package
    called counts as the package's own."""
    pkg = os.path.normcase(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))) + os.sep

    def inside(f: FrameType) -> bool:
        return os.path.normcase(
            os.path.abspath(f.f_code.co_filename)).startswith(pkg)

    frame = inspect.currentframe()
    level = 1
    try:
        f = frame.f_back if frame is not None else None  # the warn site
        while f is not None and (
            inside(f)
            or (f.f_code.co_filename.startswith("<")
                and f.f_back is not None and inside(f.f_back))
        ):
            f = f.f_back
            level += 1
        return level
    finally:
        del frame


def _warn_ends_mid_stage(
    stage: "StageRecord", k: int, series: Path, t: np.ndarray,
    lo: float, hi: float,
) -> None:
    """Warn when an increment lies past the series' last sample while that
    sample is non-zero and ``use_last`` is off (module docstring)."""
    assert series.values is not None
    last = float(series.factor) * float(series.values[-1])
    if series.use_last or last == 0.0:
        return
    past = _past_end(series, t)
    if not np.any(past):
        return
    t_drop = float(t[past][0])
    if series.time is not None:
        fix = (f"extend the series to the end of the window with its last "
               f"value repeated, time=(*time, {hi:g}), "
               f"values=(*values, values[-1])")
    else:
        fix = "pass use_last=True to hold the last value"
    warnings.warn(
        f"Stage {stage.name!r}: pattern #{k}'s Path series "
        f"({_support(series)}) ends inside the stage's pseudo-time window "
        f"[{lo:g}, {hi:g}] on the non-zero factor {last:g}, so OpenSees "
        f"reads it as 0 from the increment at t={t_drop:g} on "
        f"({int(past.sum())} of {t.size} increment(s)) and this pattern "
        f"drops its load mid-stage. Fix: {fix}; or, if the drop is "
        f"intended (a record followed by free vibration), end the series "
        f"with a 0 sample.",
        SeriesEndsMidStageWarning,
        stacklevel=_stacklevel(),
    )


def warn_stage_series_outside_window(stage: "StageRecord") -> None:
    """Warn once per stage pattern whose ``Path`` series is zero at every
    increment of the stage, or else ends inside the stage on a non-zero
    value without ``use_last`` (module docstring)."""
    paths = [
        (k, p.series) for k, p in enumerate(stage.pattern_specs, start=1)
        if isinstance(p.series, Path) and p.series.values is not None
    ]
    if not paths:
        return
    t = stage_increment_times(stage)
    if t is None:
        return
    t0 = _stage_t0(stage)
    lo, hi = min(t0, float(t[-1])), max(t0, float(t[-1]))
    for k, series in paths:
        if np.any(path_factor(series, t) != 0.0):
            _warn_ends_mid_stage(stage, k, series, t, lo, hi)
            continue
        onset = _onset(series)
        if onset is None:
            fix = "every value of the series is zero; check its values"
        elif stage.pre_analyze_reset:
            fix = (f"s.reset() puts this stage's clock back at 0 right "
                   f"before the analyze loop, so s.set_time cannot move "
                   f"it; drop s.reset() and call s.set_time({onset:g}), or "
                   f"re-base the series' time axis onto [{lo:g}, {hi:g}]")
        else:
            fix = (f"call s.set_time({onset:g}) in this stage so its clock "
                   f"starts where the series' load does, or re-base the "
                   f"series' time axis onto [{lo:g}, {hi:g}]")
        warnings.warn(
            f"Stage {stage.name!r}: pattern #{k}'s Path series "
            f"({_support(series)}) is zero at all {t.size} increment(s) "
            f"of the stage's pseudo-time window [{lo:g}, {hi:g}], so this "
            f"pattern applies no load. Each stage starts at pseudo-time 0 "
            f"(the previous stage closes with loadConst -time 0.0) unless "
            f"s.set_time(t) moves it. Fix: {fix}.",
            SeriesOutsideStageWindowWarning,
            stacklevel=_stacklevel(),
        )
