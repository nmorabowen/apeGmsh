"""Unit tests for ``apeGmsh.opensees.time_series.time_series``.

Phase 1D-extra ships the OpenSees core 5: Linear, Constant, Path,
Trig, Pulse. Each class gets:

  * construction (defaults, explicit values)
  * validation (per-class invariants)
  * ``_emit`` records the right call into a ``RecordingEmitter``
  * ``dependencies()`` returns ``()`` (all five are leaves)
  * ``__repr__`` includes the class name

Tests use ``RecordingEmitter`` only — no openseespy, no gmsh, no
subprocess. Run with::

    pytest tests/opensees/unit/primitives/test_time_series.py -v
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.time_series.time_series import (
    ASCE41Protocol,
    Constant,
    FEMA461Protocol,
    Linear,
    ModifiedATC24Protocol,
    Path,
    Pulse,
    Ricker,
    Trig,
)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class TestLinear:
    def test_construction_default_factor(self) -> None:
        ts = Linear()
        assert ts.factor == 1.0

    def test_construction_explicit_factor(self) -> None:
        ts = Linear(factor=2.5)
        assert ts.factor == 2.5

    def test_emit_default_factor_omits_factor_flag(self) -> None:
        ts = Linear()
        e = RecordingEmitter()
        ts._emit(e, tag=7)
        assert e.calls == [("timeSeries", ("Linear", 7), {})]

    def test_emit_non_default_factor_records_factor_flag(self) -> None:
        ts = Linear(factor=9.81)
        e = RecordingEmitter()
        ts._emit(e, tag=3)
        assert e.calls == [
            ("timeSeries", ("Linear", 3, "-factor", 9.81), {})
        ]

    def test_dependencies_is_empty(self) -> None:
        assert Linear().dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "Linear" in repr(Linear())

    def test_is_frozen(self) -> None:
        ts = Linear(factor=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            ts.factor = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------

class TestConstant:
    def test_construction_default_factor(self) -> None:
        ts = Constant()
        assert ts.factor == 1.0

    def test_construction_explicit_factor(self) -> None:
        ts = Constant(factor=4.2)
        assert ts.factor == 4.2

    def test_emit_default_factor_omits_factor_flag(self) -> None:
        ts = Constant()
        e = RecordingEmitter()
        ts._emit(e, tag=11)
        assert e.calls == [("timeSeries", ("Constant", 11), {})]

    def test_emit_non_default_factor_records_factor_flag(self) -> None:
        ts = Constant(factor=2.0)
        e = RecordingEmitter()
        ts._emit(e, tag=5)
        assert e.calls == [
            ("timeSeries", ("Constant", 5, "-factor", 2.0), {})
        ]

    def test_dependencies_is_empty(self) -> None:
        assert Constant().dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "Constant" in repr(Constant())


# ---------------------------------------------------------------------------
# Path
# ---------------------------------------------------------------------------

class TestPath:
    # -- Construction & validation ---------------------------------------

    def test_file_with_dt_constructs(self) -> None:
        ts = Path(file="elcentro.txt", dt=0.01)
        assert ts.file == "elcentro.txt"
        assert ts.dt == 0.01
        assert ts.values is None

    def test_values_with_dt_constructs(self) -> None:
        ts = Path(values=(0.0, 1.0, 2.0), dt=0.5)
        assert ts.values == (0.0, 1.0, 2.0)
        assert ts.dt == 0.5

    def test_values_with_time_constructs(self) -> None:
        ts = Path(values=(0.0, 1.0, 2.0), time=(0.0, 0.5, 1.0))
        assert ts.values == (0.0, 1.0, 2.0)
        assert ts.time == (0.0, 0.5, 1.0)

    def test_file_only_constructs(self) -> None:
        # File-only is allowed (the file may already encode time).
        ts = Path(file="motion.txt")
        assert ts.file == "motion.txt"

    def test_neither_file_nor_values_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of file= or values="):
            Path(dt=0.01)

    def test_both_file_and_values_raises(self) -> None:
        with pytest.raises(ValueError, match="exactly one of file= or values="):
            Path(file="x.txt", values=(0.0, 1.0), dt=0.01)

    def test_both_dt_and_time_raises(self) -> None:
        with pytest.raises(ValueError, match="dt= or time=, not both"):
            Path(values=(0.0, 1.0), dt=0.01, time=(0.0, 0.5))

    def test_values_without_dt_or_time_raises(self) -> None:
        with pytest.raises(ValueError, match="when supplying values="):
            Path(values=(0.0, 1.0, 2.0))

    def test_negative_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="factor must be > 0"):
            Path(file="x.txt", factor=-1.0)

    def test_zero_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="factor must be > 0"):
            Path(file="x.txt", factor=0.0)

    # -- Emit shapes -----------------------------------------------------

    def test_emit_file_with_dt(self) -> None:
        ts = Path(file="elcentro.txt", dt=0.01)
        e = RecordingEmitter()
        ts._emit(e, tag=1)
        assert e.calls == [
            (
                "timeSeries",
                ("Path", 1, "-filePath", "elcentro.txt", "-dt", 0.01),
                {},
            )
        ]

    def test_emit_values_with_dt(self) -> None:
        ts = Path(values=(0.0, 1.0, 2.0, 1.0), dt=0.5)
        e = RecordingEmitter()
        ts._emit(e, tag=4)
        assert e.calls == [
            (
                "timeSeries",
                ("Path", 4, "-values", 0.0, 1.0, 2.0, 1.0, "-dt", 0.5),
                {},
            )
        ]

    def test_emit_values_with_time(self) -> None:
        ts = Path(values=(0.0, 1.0), time=(0.0, 1.0))
        e = RecordingEmitter()
        ts._emit(e, tag=2)
        assert e.calls == [
            (
                "timeSeries",
                ("Path", 2, "-values", 0.0, 1.0, "-time", 0.0, 1.0),
                {},
            )
        ]

    def test_emit_factor_appears_when_not_one(self) -> None:
        ts = Path(file="m.txt", dt=0.02, factor=9.81)
        e = RecordingEmitter()
        ts._emit(e, tag=8)
        assert e.calls == [
            (
                "timeSeries",
                (
                    "Path", 8,
                    "-filePath", "m.txt",
                    "-dt", 0.02,
                    "-factor", 9.81,
                ),
                {},
            )
        ]

    def test_emit_start_time_appears_when_nonzero(self) -> None:
        ts = Path(file="m.txt", dt=0.02, start_time=5.0)
        e = RecordingEmitter()
        ts._emit(e, tag=9)
        # -startTime appears after -dt
        call = e.calls[0]
        assert "-startTime" in call[1]
        assert call[1][call[1].index("-startTime") + 1] == 5.0

    def test_emit_prepend_zero_appears_when_set(self) -> None:
        ts = Path(file="m.txt", dt=0.02, prepend_zero=True)
        e = RecordingEmitter()
        ts._emit(e, tag=10)
        assert "-prependZero" in e.calls[0][1]

    def test_emit_prepend_zero_omitted_when_unset(self) -> None:
        ts = Path(file="m.txt", dt=0.02)
        e = RecordingEmitter()
        ts._emit(e, tag=10)
        assert "-prependZero" not in e.calls[0][1]

    def test_emit_full_optional_block(self) -> None:
        ts = Path(
            values=(0.0, 1.0, 0.5),
            dt=0.1,
            factor=2.0,
            start_time=1.5,
            prepend_zero=True,
        )
        e = RecordingEmitter()
        ts._emit(e, tag=12)
        assert e.calls == [
            (
                "timeSeries",
                (
                    "Path", 12,
                    "-values", 0.0, 1.0, 0.5,
                    "-dt", 0.1,
                    "-factor", 2.0,
                    "-startTime", 1.5,
                    "-prependZero",
                ),
                {},
            )
        ]

    def test_dependencies_is_empty(self) -> None:
        ts = Path(file="x.txt")
        assert ts.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "Path" in repr(Path(file="x.txt"))


# ---------------------------------------------------------------------------
# Trig
# ---------------------------------------------------------------------------

class TestTrig:
    def test_construction_required_args(self) -> None:
        ts = Trig(t_start=0.0, t_end=10.0, period=1.0)
        assert ts.t_start == 0.0
        assert ts.t_end == 10.0
        assert ts.period == 1.0
        assert ts.factor == 1.0
        assert ts.shift == 0.0
        assert ts.zero_shift == 0.0

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be > 0"):
            Trig(t_start=0.0, t_end=1.0, period=-1.0)

    def test_zero_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be > 0"):
            Trig(t_start=0.0, t_end=1.0, period=0.0)

    def test_t_end_le_t_start_raises(self) -> None:
        with pytest.raises(ValueError, match="t_end must be > t_start"):
            Trig(t_start=5.0, t_end=5.0, period=1.0)

    def test_emit_default_optionals_omits_them(self) -> None:
        ts = Trig(t_start=0.0, t_end=10.0, period=2.0)
        e = RecordingEmitter()
        ts._emit(e, tag=1)
        assert e.calls == [
            ("timeSeries", ("Trig", 1, 0.0, 10.0, 2.0), {})
        ]

    def test_emit_factor_only(self) -> None:
        ts = Trig(t_start=0.0, t_end=10.0, period=2.0, factor=3.0)
        e = RecordingEmitter()
        ts._emit(e, tag=2)
        assert e.calls == [
            (
                "timeSeries",
                ("Trig", 2, 0.0, 10.0, 2.0, "-factor", 3.0),
                {},
            )
        ]

    def test_emit_shift_only(self) -> None:
        ts = Trig(t_start=0.0, t_end=10.0, period=2.0, shift=0.5)
        e = RecordingEmitter()
        ts._emit(e, tag=3)
        assert e.calls == [
            (
                "timeSeries",
                ("Trig", 3, 0.0, 10.0, 2.0, "-shift", 0.5),
                {},
            )
        ]

    def test_emit_full_optional_block(self) -> None:
        ts = Trig(
            t_start=0.0, t_end=10.0, period=2.0,
            factor=1.5, shift=0.25, zero_shift=0.1,
        )
        e = RecordingEmitter()
        ts._emit(e, tag=4)
        assert e.calls == [
            (
                "timeSeries",
                (
                    "Trig", 4, 0.0, 10.0, 2.0,
                    "-factor", 1.5,
                    "-shift", 0.25,
                    "-zeroShift", 0.1,
                ),
                {},
            )
        ]

    def test_dependencies_is_empty(self) -> None:
        ts = Trig(t_start=0.0, t_end=1.0, period=0.5)
        assert ts.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "Trig" in repr(Trig(t_start=0.0, t_end=1.0, period=0.5))


# ---------------------------------------------------------------------------
# Pulse
# ---------------------------------------------------------------------------

class TestPulse:
    def test_construction_required_args(self) -> None:
        ts = Pulse(t_start=0.0, t_end=10.0, period=1.0, width=0.5)
        assert ts.t_start == 0.0
        assert ts.t_end == 10.0
        assert ts.period == 1.0
        assert ts.width == 0.5

    def test_negative_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period must be > 0"):
            Pulse(t_start=0.0, t_end=1.0, period=-1.0, width=0.5)

    def test_width_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="width must be in"):
            Pulse(t_start=0.0, t_end=1.0, period=0.5, width=0.0)

    def test_width_one_raises(self) -> None:
        with pytest.raises(ValueError, match="width must be in"):
            Pulse(t_start=0.0, t_end=1.0, period=0.5, width=1.0)

    def test_width_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="width must be in"):
            Pulse(t_start=0.0, t_end=1.0, period=0.5, width=1.5)

    def test_t_end_le_t_start_raises(self) -> None:
        with pytest.raises(ValueError, match="t_end must be > t_start"):
            Pulse(t_start=5.0, t_end=5.0, period=1.0, width=0.5)

    def test_emit_default_optionals_omits_them(self) -> None:
        ts = Pulse(t_start=0.0, t_end=10.0, period=2.0, width=0.4)
        e = RecordingEmitter()
        ts._emit(e, tag=1)
        assert e.calls == [
            ("timeSeries", ("Pulse", 1, 0.0, 10.0, 2.0, 0.4), {})
        ]

    def test_emit_full_optional_block(self) -> None:
        ts = Pulse(
            t_start=0.0, t_end=10.0, period=2.0, width=0.4,
            factor=2.0, shift=0.1, zero_shift=0.05,
        )
        e = RecordingEmitter()
        ts._emit(e, tag=2)
        assert e.calls == [
            (
                "timeSeries",
                (
                    "Pulse", 2, 0.0, 10.0, 2.0, 0.4,
                    "-factor", 2.0,
                    "-shift", 0.1,
                    "-zeroShift", 0.05,
                ),
                {},
            )
        ]

    def test_dependencies_is_empty(self) -> None:
        ts = Pulse(t_start=0.0, t_end=1.0, period=0.5, width=0.5)
        assert ts.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        ts = Pulse(t_start=0.0, t_end=1.0, period=0.5, width=0.5)
        assert "Pulse" in repr(ts)


# ---------------------------------------------------------------------------
# Ricker — Gaussian-derivative wavelet, emitted as a Path
# ---------------------------------------------------------------------------

class TestRicker:
    def test_construction_defaults(self) -> None:
        ts = Ricker(f_n=0.5, t_total=10.0, dt=0.05)
        assert ts.f_n == 0.5
        assert ts.t_total == 10.0
        assert ts.dt == 0.05
        assert ts.t_center == 0.0
        assert ts.kind == "acceleration"
        assert ts.factor == 1.0

    # -- Validation ------------------------------------------------------

    def test_non_positive_f_n_raises(self) -> None:
        with pytest.raises(ValueError, match="f_n must be > 0"):
            Ricker(f_n=0.0, t_total=10.0, dt=0.05)

    def test_non_positive_t_total_raises(self) -> None:
        with pytest.raises(ValueError, match="t_total must be > 0"):
            Ricker(f_n=0.5, t_total=0.0, dt=0.05)

    def test_non_positive_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="dt must be > 0"):
            Ricker(f_n=0.5, t_total=10.0, dt=0.0)

    def test_bad_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="kind must be one of"):
            Ricker(f_n=0.5, t_total=10.0, dt=0.05, kind="displacement")

    def test_non_positive_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="factor must be > 0"):
            Ricker(f_n=0.5, t_total=10.0, dt=0.05, factor=0.0)

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            Ricker(f_n=0.5, t_total=0.05, dt=0.05)

    # -- Wavelet math ----------------------------------------------------

    def test_samples_length_and_grid(self) -> None:
        ts = Ricker(f_n=0.5, t_total=10.0, dt=0.05)
        times, values = ts.samples()
        assert len(times) == len(values) == 200      # round(10/0.05)
        assert times[0] == 0.0
        assert times[1] == pytest.approx(0.05)

    def test_acceleration_peaks_at_center(self) -> None:
        # The Ricker accel form equals +1 at tau=0 (its peak).
        ts = Ricker(f_n=0.5, t_total=10.0, dt=0.05, t_center=5.0)
        times, values = ts.samples()
        peak_i = times.index(5.0)
        assert values[peak_i] == pytest.approx(1.0)
        # Symmetric (even function about t_center).
        assert values[peak_i - 3] == pytest.approx(values[peak_i + 3])

    def test_velocity_is_odd_about_center(self) -> None:
        ts = Ricker(f_n=0.5, t_total=10.0, dt=0.05, t_center=5.0,
                    kind="velocity")
        times, values = ts.samples()
        peak_i = times.index(5.0)
        assert values[peak_i] == pytest.approx(0.0)          # tau=0
        assert values[peak_i + 4] == pytest.approx(-values[peak_i - 4])

    # -- Emit delegates to Path -----------------------------------------

    def test_emit_is_a_path_command(self) -> None:
        ts = Ricker(f_n=0.5, t_total=0.2, dt=0.05)   # 4 samples, small
        _, values = ts.samples()
        e = RecordingEmitter()
        ts._emit(e, tag=6)
        assert e.calls == [
            ("timeSeries", ("Path", 6, "-values", *values, "-dt", 0.05), {})
        ]

    def test_emit_passes_factor_through_to_path(self) -> None:
        ts = Ricker(f_n=0.5, t_total=0.2, dt=0.05, factor=9.81)
        e = RecordingEmitter()
        ts._emit(e, tag=6)
        assert "-factor" in e.calls[0][1]
        assert e.calls[0][1][e.calls[0][1].index("-factor") + 1] == 9.81

    def test_dependencies_is_empty(self) -> None:
        assert Ricker(f_n=0.5, t_total=10.0, dt=0.05).dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "Ricker" in repr(Ricker(f_n=0.5, t_total=10.0, dt=0.05))


# ---------------------------------------------------------------------------
# Cyclic loading protocols — normalized ±1, emitted as a Path
# ---------------------------------------------------------------------------

class TestCyclicProtocols:
    @pytest.mark.parametrize(
        "cls", [ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol]
    )
    def test_normalized_to_unit_peak(self, cls: type) -> None:
        _, values = cls().samples()
        assert max(values) == pytest.approx(1.0)
        assert min(values) == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "cls", [ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol]
    )
    def test_framed_by_zeros(self, cls: type) -> None:
        _, values = cls().samples()
        assert values[0] == 0.0
        assert values[-1] == 0.0

    @pytest.mark.parametrize(
        "cls", [ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol]
    )
    def test_time_is_normalized_monotonic(self, cls: type) -> None:
        time, _ = cls().samples()
        assert time[0] == 0.0
        assert time[-1] == pytest.approx(1.0)
        assert all(b >= a for a, b in zip(time, time[1:]))

    @pytest.mark.parametrize(
        "cls", [ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol]
    )
    def test_non_positive_factor_raises(self, cls: type) -> None:
        with pytest.raises(ValueError, match="factor must be > 0"):
            cls(factor=0.0)

    def test_asce41_cycle_structure(self) -> None:
        # 3*6 + 2*3 = 24 cycles -> 48 reversals + 2 framing zeros = 50 pts.
        _, values = ASCE41Protocol().samples()
        assert len(values) == 50

    def test_atc24_cycle_structure(self) -> None:
        # reps 3+3+3+2+2+1 = 14 cycles -> 28 reversals + 2 zeros = 30 pts.
        _, values = ModifiedATC24Protocol().samples()
        assert len(values) == 30

    def test_fema461_two_cycles_per_amplitude(self) -> None:
        # Each amplitude appears with both signs exactly twice (2 cycles).
        _, values = FEMA461Protocol().samples()
        peaks = [v for v in values if v > 0]
        # Every positive peak value occurs an even number of times (==2).
        from collections import Counter
        counts = Counter(round(v, 9) for v in peaks)
        assert all(c == 2 for c in counts.values())

    def test_fema461_alpha_controls_step_count(self) -> None:
        # Larger alpha -> fewer amplitude steps -> shorter history.
        coarse = FEMA461Protocol(alpha=0.8).samples()[1]
        fine = FEMA461Protocol(alpha=0.2).samples()[1]
        assert len(fine) > len(coarse)

    def test_fema461_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            FEMA461Protocol(alpha=0.0)

    def test_fema461_bad_start_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="start_fraction must be in"):
            FEMA461Protocol(start_fraction=1.0)

    def test_emit_is_a_path_with_time_axis(self) -> None:
        ts = ASCE41Protocol(factor=0.02)
        time, values = ts.samples()
        e = RecordingEmitter()
        ts._emit(e, tag=7)
        assert e.calls == [
            (
                "timeSeries",
                ("Path", 7, "-values", *values, "-time", *time,
                 "-factor", 0.02),
                {},
            )
        ]

    @pytest.mark.parametrize(
        "cls", [ASCE41Protocol, ModifiedATC24Protocol, FEMA461Protocol]
    )
    def test_dependencies_is_empty(self, cls: type) -> None:
        assert cls().dependencies() == ()


# ---------------------------------------------------------------------------
# Namespace integration — namespace methods register with the bridge
# ---------------------------------------------------------------------------

def _make_ops() -> "apeSees":
    """Construct an apeSees with a stub FEMData (namespaces ignore it)."""
    from typing import cast
    from unittest.mock import MagicMock

    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


class TestTimeSeriesNamespace:
    def test_linear_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Linear(factor=2.0)
        assert isinstance(ts, Linear)
        assert ts.factor == 2.0
        assert ops.tag_for(ts) == 1

    def test_constant_namespace_default(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Constant()
        assert isinstance(ts, Constant)
        assert ts.factor == 1.0

    def test_path_namespace_full_kwargs(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Path(file="motion.txt", dt=0.01, factor=9.81)
        assert isinstance(ts, Path)
        assert ts.file == "motion.txt"
        assert ts.dt == 0.01
        assert ts.factor == 9.81

    def test_trig_namespace(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Trig(t_start=0.0, t_end=5.0, period=1.0)
        assert isinstance(ts, Trig)

    def test_pulse_namespace(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Pulse(
            t_start=0.0, t_end=5.0, period=1.0, width=0.5,
        )
        assert isinstance(ts, Pulse)

    def test_ricker_namespace_constructs_and_registers(self) -> None:
        ops = _make_ops()
        ts = ops.timeSeries.Ricker(f_n=0.5, t_total=10.0, dt=0.05,
                                   t_center=3.0, kind="velocity")
        assert isinstance(ts, Ricker)
        assert ts.kind == "velocity"
        assert ts.t_center == 3.0
        assert ops.tag_for(ts) == 1

    def test_protocol_namespaces_construct_and_register(self) -> None:
        ops = _make_ops()
        a = ops.timeSeries.ASCE41Protocol(factor=0.02)
        b = ops.timeSeries.ModifiedATC24Protocol(factor=0.03)
        c = ops.timeSeries.FEMA461Protocol(factor=0.03, alpha=0.5)
        assert isinstance(a, ASCE41Protocol)
        assert isinstance(b, ModifiedATC24Protocol)
        assert isinstance(c, FEMA461Protocol)
        assert c.alpha == 0.5
        assert (ops.tag_for(a), ops.tag_for(b), ops.tag_for(c)) == (1, 2, 3)

    def test_distinct_time_series_get_distinct_tags(self) -> None:
        ops = _make_ops()
        a = ops.timeSeries.Linear()
        b = ops.timeSeries.Constant()
        assert ops.tag_for(a) == 1
        assert ops.tag_for(b) == 2
