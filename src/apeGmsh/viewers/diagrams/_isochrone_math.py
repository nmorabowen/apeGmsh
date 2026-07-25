"""Pure-numpy primitives shared by the three isochrone diagrams.

An *isochrone* view answers a question about **time** rather than about
a response value, so all three kinds need the same two things: reduce a
whole ``(T, N)`` history to one time per node, and pick a handful of
representative instants out of a step range. Both live here as free
functions over plain arrays — no Results, no Qt, no VTK — so the
semantics that matter (what counts as an arrival, which steps get
drawn) are unit-testable without building a scene.

Used by:

* ``_isochrone_map`` — :func:`arrival_times` (+ :func:`resolve_threshold`)
* ``_isochrone_profile`` — :func:`pick_step_indices`, :func:`dominant_axis`
* ``_isochrone_strobe`` — :func:`pick_step_indices`
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


#: ``mode`` values accepted by :func:`arrival_times`.
MODE_FIRST_CROSSING = "first_crossing"
MODE_TIME_TO_PEAK = "time_to_peak"
ARRIVAL_MODES = (MODE_FIRST_CROSSING, MODE_TIME_TO_PEAK)

#: ``path_axis`` / axis-name vocabulary shared by the profile diagram.
AXIS_NAMES = ("x", "y", "z")


def tracked_values(values: ndarray, *, use_abs: bool) -> ndarray:
    """The signal an arrival criterion is evaluated against.

    ``use_abs`` tracks ``|value|`` — the right choice for a wavefront,
    where a first arrival is a departure from zero in either direction.
    Non-finite entries become ``-inf`` so they can never win an argmax
    or satisfy a crossing test: a NaN in the middle of a history should
    not read as "the wave arrived here".
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.abs(arr) if use_abs else arr.copy()
    out[~np.isfinite(out)] = -np.inf
    return out


def resolve_threshold(
    tracked: ndarray, *, threshold: "float | None", fraction: float,
) -> float:
    """Crossing level for :func:`arrival_times`.

    An explicit ``threshold`` passes through. ``None`` derives
    ``fraction × max(tracked)`` so the caller doesn't have to know the
    field's units — the point of the fraction default. Raises when the
    tracked signal has no finite positive maximum to scale (an all-zero
    or all-NaN history has no meaningful arrival level).
    """
    if threshold is not None:
        return float(threshold)
    finite = tracked[np.isfinite(tracked)]
    peak = float(finite.max()) if finite.size else 0.0
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(
            f"cannot derive an arrival threshold as a fraction of the "
            f"peak: the tracked signal has no finite positive maximum "
            f"(peak={peak!r}). Either the field is flat/unrecorded over "
            f"this stage, or it is entirely negative and is being "
            f"tracked signed (use_abs=False) — a fraction of its peak "
            f"is then meaningless. Set an explicit threshold (negative "
            f"is fine), or enable use_abs to track magnitude."
        )
    return float(fraction) * peak


def arrival_times(
    values: ndarray,
    time: ndarray,
    *,
    mode: str = MODE_FIRST_CROSSING,
    threshold: "float | None" = None,
    threshold_fraction: float = 0.1,
    use_abs: bool = True,
    interpolate: bool = True,
) -> tuple[ndarray, float]:
    """Reduce a ``(T, N)`` history to one arrival time per column.

    Parameters
    ----------
    values
        ``(T, N)`` history — rows are steps, columns are nodes.
    time
        ``(T,)`` time vector matching ``values``' rows.
    mode
        ``"first_crossing"`` or ``"time_to_peak"`` (see
        :class:`~._styles.IsochroneMapStyle`).
    threshold, threshold_fraction, use_abs, interpolate
        Criterion parameters; see the style docstring.

    Returns
    -------
    ``(arrival, threshold_used)`` where ``arrival`` is ``(N,)`` float64.
    Columns that never satisfy the criterion are ``nan`` — only
    possible in ``"first_crossing"`` mode. ``threshold_used`` is the
    resolved level (``nan`` in ``"time_to_peak"`` mode, which has no
    threshold) so a caller can report what it actually applied.

    Raises
    ------
    ValueError
        Unknown ``mode``, mismatched shapes, or an underivable
        threshold.
    """
    arr = np.asarray(values, dtype=np.float64)
    t = np.asarray(time, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"arrival_times expects (T, N) values; got shape {arr.shape}."
        )
    if t.ndim != 1 or t.size != arr.shape[0]:
        raise ValueError(
            f"time must be (T,) matching values' {arr.shape[0]} rows; "
            f"got shape {t.shape}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return (np.zeros(arr.shape[1], dtype=np.float64), float("nan"))
    if mode not in ARRIVAL_MODES:
        raise ValueError(
            f"arrival_times mode must be one of {ARRIVAL_MODES}; "
            f"got {mode!r}."
        )

    tracked = tracked_values(arr, use_abs=use_abs)

    if mode == MODE_TIME_TO_PEAK:
        idx = np.argmax(tracked, axis=0)
        return (t[idx].astype(np.float64), float("nan"))

    level = resolve_threshold(
        tracked, threshold=threshold, fraction=threshold_fraction,
    )
    reached = tracked >= level
    ever = reached.any(axis=0)
    # argmax on a boolean gives the first True (0 for all-False rows,
    # which ``ever`` then masks out).
    first = np.argmax(reached, axis=0)

    out = np.full(arr.shape[1], np.nan, dtype=np.float64)
    out[ever] = t[first[ever]]

    if interpolate:
        # Interpolate only where there IS a bracketing step below the
        # level (first == 0 means the signal was already there at the
        # start of the stage — nothing to interpolate against).
        braket = ever & (first > 0)
        if braket.any():
            cols = np.where(braket)[0]
            k = first[cols]
            v1 = tracked[k, cols]
            v0 = tracked[k - 1, cols]
            span = v1 - v0
            # v0 < level <= v1 by construction, so span > 0 — EXCEPT
            # when v0 came from a non-finite sample, which
            # ``tracked_values`` sank to -inf: the span is then +inf and
            # ``(level - v0) / span`` is inf/inf = NaN, which would
            # propagate into the arrival time and silently drop a node
            # that demonstrably DID cross (and warn from numpy on the
            # way). Interpolation needs two usable samples; where the
            # earlier one is unusable, fall back to the un-interpolated
            # answer (frac = 0 -> the later step's own time), so
            # interpolate=True never reports *less* than
            # interpolate=False. Same fallback covers the degenerate
            # equal-endpoints case.
            frac = np.zeros(cols.size, dtype=np.float64)
            good = np.isfinite(span) & (span > 0.0) & np.isfinite(v0)
            frac[good] = (level - v0[good]) / span[good]
            np.clip(frac, 0.0, 1.0, out=frac)
            # frac == 0 must land on t[k], not t[k-1]: a zero fraction
            # here means "no usable bracket", not "crossed exactly at
            # the earlier step".
            lower = np.where(good, t[k - 1], t[k])
            out[cols] = lower + frac * (t[k] - lower)

    return (out, level)


def pick_step_indices(n_steps: int, n_wanted: int) -> ndarray:
    """``n_wanted`` step indices spread evenly over ``range(n_steps)``.

    The endpoints are always present (a strobe or curve family that
    omits the final state is misleading), duplicates are collapsed, and
    asking for more instants than there are steps simply returns every
    step. Returns an empty array when ``n_steps <= 0``.
    """
    n_steps = int(n_steps)
    if n_steps <= 0:
        return np.zeros(0, dtype=np.int64)
    n_wanted = int(n_wanted)
    if n_wanted <= 1:
        return np.asarray([n_steps - 1], dtype=np.int64)
    if n_wanted >= n_steps:
        return np.arange(n_steps, dtype=np.int64)
    raw = np.linspace(0.0, float(n_steps - 1), n_wanted)
    return np.unique(np.rint(raw).astype(np.int64))


def dominant_axis(coords: ndarray) -> int:
    """Index (0/1/2) of the axis with the largest extent in ``coords``.

    The profile diagram's ``path_axis="auto"``: order nodes along
    whichever direction the selection actually spans. Ties break toward
    the later axis (``np.argmax`` on the extents would pick the first);
    we prefer z for a vertical column, so scan in reverse. Degenerate
    input (fewer than 2 points, or zero extent everywhere) returns
    ``2`` — z, the most common profile direction.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        return 2
    extent = pts.max(axis=0) - pts.min(axis=0)
    if not np.any(extent > 0.0):
        return 2
    # Reverse-scan argmax => ties favour z, then y, then x.
    return int(2 - np.argmax(extent[::-1]))


__all__ = [
    "ARRIVAL_MODES",
    "AXIS_NAMES",
    "MODE_FIRST_CROSSING",
    "MODE_TIME_TO_PEAK",
    "arrival_times",
    "dominant_axis",
    "pick_step_indices",
    "resolve_threshold",
    "tracked_values",
]
