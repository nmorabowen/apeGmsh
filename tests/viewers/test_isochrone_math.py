"""Pure-numpy isochrone primitives.

The semantics that make or break an isochrone view live here, so they
are pinned without building a scene: what counts as an arrival, how a
crossing time is interpolated, which instants get drawn, and which axis
orders a path.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.diagrams._isochrone_math import (
    MODE_FIRST_CROSSING,
    MODE_TIME_TO_PEAK,
    arrival_times,
    dominant_axis,
    pick_step_indices,
    resolve_threshold,
    tracked_values,
)


# =====================================================================
# tracked_values
# =====================================================================

def test_tracked_values_takes_absolute_when_asked() -> None:
    v = np.array([[-3.0, 2.0]])
    np.testing.assert_allclose(
        tracked_values(v, use_abs=True), [[3.0, 2.0]],
    )
    np.testing.assert_allclose(
        tracked_values(v, use_abs=False), [[-3.0, 2.0]],
    )


def test_tracked_values_sinks_non_finite_to_minus_inf() -> None:
    """A NaN mid-history must never read as 'the wave arrived here'."""
    v = np.array([[np.nan, np.inf, 1.0]])
    out = tracked_values(v, use_abs=True)
    assert out[0, 0] == -np.inf
    assert out[0, 1] == -np.inf
    assert out[0, 2] == 1.0


# =====================================================================
# resolve_threshold
# =====================================================================

def test_resolve_threshold_passes_explicit_value_through() -> None:
    tracked = np.array([[0.0, 10.0]])
    assert resolve_threshold(tracked, threshold=2.5, fraction=0.5) == 2.5


def test_resolve_threshold_derives_fraction_of_peak() -> None:
    tracked = np.array([[0.0, 10.0], [4.0, 8.0]])
    assert resolve_threshold(
        tracked, threshold=None, fraction=0.1,
    ) == pytest.approx(1.0)


def test_resolve_threshold_raises_on_flat_zero_history() -> None:
    """No positive peak => no meaningful level; fail loud, not silently 0."""
    tracked = np.zeros((3, 4))
    with pytest.raises(ValueError, match="no finite positive maximum"):
        resolve_threshold(tracked, threshold=None, fraction=0.1)


# =====================================================================
# arrival_times — first crossing
# =====================================================================

def _ramp_history():
    """Two nodes, one arriving early and one late, over t = 0..4.

    Node 0 ramps 0,1,2,3,4; node 1 stays at 0 until the last step.
    """
    values = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 4.0],
    ])
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return values, time


def test_first_crossing_snaps_to_step_when_interpolation_off() -> None:
    values, time = _ramp_history()
    arrival, level = arrival_times(
        values, time, mode=MODE_FIRST_CROSSING,
        threshold=2.0, interpolate=False,
    )
    assert level == 2.0
    # Node 0 first reaches 2.0 at t=2; node 1 only at t=4.
    np.testing.assert_allclose(arrival, [2.0, 4.0])


def test_first_crossing_interpolates_between_bracketing_steps() -> None:
    values, time = _ramp_history()
    arrival, _ = arrival_times(
        values, time, mode=MODE_FIRST_CROSSING,
        threshold=1.5, interpolate=True,
    )
    # Node 0 crosses 1.5 halfway between t=1 (v=1) and t=2 (v=2).
    assert arrival[0] == pytest.approx(1.5)


def test_first_crossing_marks_never_arrived_as_nan() -> None:
    values = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    time = np.array([0.0, 1.0, 2.0])
    arrival, _ = arrival_times(
        values, time, mode=MODE_FIRST_CROSSING, threshold=1.5,
        interpolate=False,
    )
    assert arrival[0] == pytest.approx(2.0)
    assert np.isnan(arrival[1])


def test_first_crossing_at_step_zero_is_not_interpolated() -> None:
    """Already above the level at t0 — there is nothing to bracket."""
    values = np.array([[5.0], [6.0]])
    time = np.array([10.0, 11.0])
    arrival, _ = arrival_times(
        values, time, mode=MODE_FIRST_CROSSING, threshold=1.0,
    )
    assert arrival[0] == pytest.approx(10.0)


def test_first_crossing_uses_abs_by_default() -> None:
    """A negative excursion is an arrival for a wavefront."""
    values = np.array([[0.0], [-3.0]])
    time = np.array([0.0, 1.0])
    with_abs, _ = arrival_times(
        values, time, threshold=2.0, use_abs=True, interpolate=False,
    )
    assert with_abs[0] == pytest.approx(1.0)
    signed, _ = arrival_times(
        values, time, threshold=2.0, use_abs=False, interpolate=False,
    )
    assert np.isnan(signed[0])


def test_derived_threshold_is_reported_back() -> None:
    """The caller must be able to show what level it actually applied."""
    values, time = _ramp_history()
    _, level = arrival_times(
        values, time, threshold=None, threshold_fraction=0.25,
    )
    assert level == pytest.approx(1.0)     # 0.25 x peak(4.0)


# =====================================================================
# arrival_times — time to peak
# =====================================================================

def test_time_to_peak_is_always_defined() -> None:
    values = np.array([
        [0.0, 5.0],
        [9.0, 1.0],
        [1.0, 0.0],
    ])
    time = np.array([0.0, 0.5, 1.0])
    arrival, level = arrival_times(values, time, mode=MODE_TIME_TO_PEAK)
    np.testing.assert_allclose(arrival, [0.5, 0.0])
    assert np.isnan(level)          # no threshold in this mode
    assert np.isfinite(arrival).all()


# =====================================================================
# arrival_times — argument validation
# =====================================================================

def test_arrival_times_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        arrival_times(np.zeros((2, 2)), np.zeros(2), mode="whenever")


def test_arrival_times_rejects_1d_values() -> None:
    with pytest.raises(ValueError, match=r"\(T, N\) values"):
        arrival_times(np.zeros(4), np.zeros(4))


def test_arrival_times_rejects_time_length_mismatch() -> None:
    with pytest.raises(ValueError, match="matching values"):
        arrival_times(np.zeros((3, 2)), np.zeros(5))


# =====================================================================
# pick_step_indices
# =====================================================================

def test_pick_step_indices_always_includes_both_endpoints() -> None:
    idx = pick_step_indices(100, 5)
    assert idx[0] == 0
    assert idx[-1] == 99


def test_pick_step_indices_spreads_evenly() -> None:
    np.testing.assert_array_equal(
        pick_step_indices(9, 5), [0, 2, 4, 6, 8],
    )


def test_pick_step_indices_collapses_duplicates() -> None:
    """Asking for more instants than steps yields each step once."""
    np.testing.assert_array_equal(pick_step_indices(3, 10), [0, 1, 2])


def test_pick_step_indices_single_frame_takes_the_last_step() -> None:
    np.testing.assert_array_equal(pick_step_indices(7, 1), [6])


def test_pick_step_indices_empty_for_no_steps() -> None:
    assert pick_step_indices(0, 5).size == 0


# =====================================================================
# dominant_axis
# =====================================================================

def test_dominant_axis_picks_the_widest_spread() -> None:
    column = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 9.0]])
    assert dominant_axis(column) == 2
    beam = np.array([[0.0, 0.0, 0.0], [9.0, 0.1, 0.0]])
    assert dominant_axis(beam) == 0
    span = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 1.0]])
    assert dominant_axis(span) == 1


def test_dominant_axis_defaults_to_z_when_degenerate() -> None:
    """A single point, or a coincident cloud, has no spread to read."""
    assert dominant_axis(np.zeros((1, 3))) == 2
    assert dominant_axis(np.zeros((5, 3))) == 2


def test_dominant_axis_breaks_ties_toward_z() -> None:
    """Equal extents: prefer z, the usual profile direction."""
    cube = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert dominant_axis(cube) == 2


# =====================================================================
# Regressions found by adversarial review
# =====================================================================

def test_interpolation_survives_a_non_finite_bracketing_sample() -> None:
    """A NaN just before the crossing must not erase the arrival.

    ``tracked_values`` sinks non-finite samples to ``-inf``; the
    interpolation's ``(level - v0) / (v1 - v0)`` was then ``inf/inf`` =
    NaN, which propagated into the arrival time and silently dropped a
    node that demonstrably DID cross (the map excludes non-finite
    arrivals from its submesh) — while also warning from numpy.
    """
    values = np.array([[0.0], [np.nan], [10.0], [10.0]])
    time = np.array([0.0, 1.0, 2.0, 3.0])
    interpolated, _ = arrival_times(
        values, time, threshold=5.0, interpolate=True,
    )
    snapped, _ = arrival_times(
        values, time, threshold=5.0, interpolate=False,
    )
    assert np.isfinite(interpolated[0])
    # With no usable earlier sample, interpolation falls back to the
    # un-interpolated answer rather than inventing one.
    assert interpolated[0] == pytest.approx(snapped[0])
    assert interpolated[0] == pytest.approx(2.0)


def test_interpolation_never_reports_earlier_than_snapping() -> None:
    """interpolate=True lands in (t[k-1], t[k]] — never outside it."""
    rng = np.random.default_rng(3)
    time = np.arange(12, dtype=np.float64) * 0.25
    for _ in range(40):
        values = rng.standard_normal((12, 6)) * 3.0
        # Sprinkle non-finite samples through the history.
        values[rng.integers(0, 12, 5), rng.integers(0, 6, 5)] = np.nan
        interp, level = arrival_times(
            values, time, threshold=1.0, interpolate=True,
        )
        snap, _ = arrival_times(
            values, time, threshold=1.0, interpolate=False,
        )
        both = np.isfinite(interp) & np.isfinite(snap)
        # Same set of arrivals either way, and interpolation only ever
        # moves the answer EARLIER than the snapped step, never later,
        # and never before the preceding step.
        np.testing.assert_array_equal(
            np.isfinite(interp), np.isfinite(snap),
        )
        assert np.all(interp[both] <= snap[both] + 1e-12)
        assert np.all(interp[both] > snap[both] - 0.25 - 1e-12)


def test_no_numpy_warning_from_interpolation() -> None:
    """The inf/inf divide used to emit 'invalid value encountered'."""
    values = np.array([[0.0], [np.nan], [10.0]])
    time = np.array([0.0, 1.0, 2.0])
    with np.errstate(all="raise"):
        arrival_times(values, time, threshold=5.0, interpolate=True)
