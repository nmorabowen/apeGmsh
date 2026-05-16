"""Unit tests for the :class:`GroundMotion` dataclass.

Pure dataclass behaviour — construction, validation, derived
properties, scale-factor via the adapter. No parsers, no apeSees.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.ground_motions import GroundMotion


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_minimum_valid_inputs(self) -> None:
        gm = GroundMotion(accel=np.array([0.0, 1.0]), dt=0.01)
        assert gm.npts == 2
        assert gm.dt == 0.01
        assert gm.source == ""
        assert gm.metadata == {}

    def test_accel_must_be_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            GroundMotion(accel=np.zeros((2, 3)), dt=0.01)

    def test_accel_too_short(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            GroundMotion(accel=np.array([0.0]), dt=0.01)

    def test_dt_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            GroundMotion(accel=np.array([0.0, 1.0]), dt=0.0)
        with pytest.raises(ValueError, match="positive"):
            GroundMotion(accel=np.array([0.0, 1.0]), dt=-0.01)

    def test_accel_cast_to_float64(self) -> None:
        gm = GroundMotion(
            accel=np.array([1, 2, 3], dtype=np.int32),
            dt=0.01,
        )
        assert gm.accel.dtype == np.float64

    def test_is_frozen(self) -> None:
        gm = GroundMotion(accel=np.array([0.0, 1.0]), dt=0.01)
        with pytest.raises((AttributeError, Exception)):
            gm.dt = 0.02  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_npts(self) -> None:
        gm = GroundMotion(accel=np.zeros(100), dt=0.01)
        assert gm.npts == 100

    def test_duration(self) -> None:
        gm = GroundMotion(accel=np.zeros(101), dt=0.01)
        assert gm.duration == pytest.approx(1.0)

    def test_time_vector_uniform(self) -> None:
        gm = GroundMotion(accel=np.zeros(5), dt=0.5)
        # Uniform records don't store the time field — it's None.
        assert gm.time is None
        # ``time_vector`` synthesises on demand.
        np.testing.assert_array_equal(
            gm.time_vector, [0.0, 0.5, 1.0, 1.5, 2.0]
        )
        assert gm.is_uniform is True

    def test_pga(self) -> None:
        gm = GroundMotion(
            accel=np.array([-0.3, 0.1, 0.5, -0.4]),
            dt=0.01,
        )
        assert gm.pga == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Non-uniform sampling
# ---------------------------------------------------------------------------

class TestNonUniform:
    def test_construction_with_time(self) -> None:
        t = np.array([0.0, 0.1, 0.25, 0.5])
        gm = GroundMotion(
            accel=np.array([0.0, 0.1, 0.2, 0.3]),
            time=t,
        )
        assert gm.is_uniform is False
        np.testing.assert_array_equal(gm.time, t)
        np.testing.assert_array_equal(gm.time_vector, t)
        assert gm.dt == pytest.approx(0.5 / 3.0)

    def test_explicit_dt_with_time_is_kept(self) -> None:
        t = np.array([0.0, 0.1, 0.25])
        gm = GroundMotion(
            accel=np.array([1.0, 2.0, 3.0]),
            dt=0.125,
            time=t,
        )
        assert gm.dt == pytest.approx(0.125)
        np.testing.assert_array_equal(gm.time, t)

    def test_duration_uses_time_endpoints(self) -> None:
        gm = GroundMotion(
            accel=np.array([0.0, 0.1, 0.2]),
            time=np.array([0.0, 0.4, 1.0]),
        )
        assert gm.duration == pytest.approx(1.0)

    def test_time_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            GroundMotion(
                accel=np.array([0.0, 0.1, 0.2]),
                time=np.array([0.0, 0.5]),
            )

    def test_non_monotonic_time_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            GroundMotion(
                accel=np.array([0.0, 0.1, 0.2]),
                time=np.array([0.0, 0.5, 0.3]),
            )

    def test_neither_dt_nor_time_raises(self) -> None:
        with pytest.raises(ValueError, match="dt is required"):
            GroundMotion(accel=np.array([0.0, 0.1, 0.2]))


# ---------------------------------------------------------------------------
# to_time_series adapter (apeSees handoff)
# ---------------------------------------------------------------------------

class _FakeTimeSeriesNS:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def Path(self, **kwargs):  # noqa: N802 - mirrors apeSees API
        self.calls.append(kwargs)
        return ("Path", kwargs)


class _FakeOps:
    def __init__(self) -> None:
        self.timeSeries = _FakeTimeSeriesNS()


class TestToTimeSeries:
    def test_default_call_passes_values_and_dt(self) -> None:
        gm = GroundMotion(
            accel=np.array([0.1, 0.2, 0.3]),
            dt=0.005,
        )
        ops = _FakeOps()
        gm.to_time_series(ops)

        call = ops.timeSeries.calls[0]
        assert call["dt"] == pytest.approx(0.005)
        assert call["values"] == (0.1, 0.2, 0.3)
        assert call["prepend_zero"] is False
        # No factor when default (would force a tag in the emitter).
        assert "factor" not in call

    def test_explicit_factor_forwarded(self) -> None:
        gm = GroundMotion(
            accel=np.array([1.0, 2.0]),
            dt=0.005,
        )
        ops = _FakeOps()
        gm.to_time_series(ops, factor=9.81)

        call = ops.timeSeries.calls[0]
        assert call["factor"] == pytest.approx(9.81)
        # Values are stored as-is; the multiplier is OpenSees's job.
        assert call["values"] == (1.0, 2.0)

    def test_prepend_zero_forwarded(self) -> None:
        gm = GroundMotion(
            accel=np.array([0.1, 0.2]),
            dt=0.005,
        )
        ops = _FakeOps()
        gm.to_time_series(ops, prepend_zero=True)
        assert ops.timeSeries.calls[0]["prepend_zero"] is True

    def test_uniform_passes_dt_not_time(self) -> None:
        gm = GroundMotion(
            accel=np.array([0.1, 0.2, 0.3]),
            dt=0.005,
        )
        ops = _FakeOps()
        gm.to_time_series(ops)

        call = ops.timeSeries.calls[0]
        assert "dt" in call
        assert "time" not in call

    def test_non_uniform_passes_time_not_dt(self) -> None:
        gm = GroundMotion(
            accel=np.array([0.1, 0.2, 0.3]),
            time=np.array([0.0, 0.1, 0.25]),
        )
        ops = _FakeOps()
        gm.to_time_series(ops)

        call = ops.timeSeries.calls[0]
        assert "time" in call
        assert "dt" not in call
        assert call["time"] == (0.0, 0.1, 0.25)
        assert call["values"] == (0.1, 0.2, 0.3)


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

def test_repr_contains_key_fields() -> None:
    gm = GroundMotion(
        accel=np.array([0.1, -0.2, 0.3]),
        dt=0.005,
        source="record.txt",
    )
    r = repr(gm)
    assert "npts=3" in r
    assert "0.005" in r
    assert "uniform" in r
    assert "record.txt" in r
