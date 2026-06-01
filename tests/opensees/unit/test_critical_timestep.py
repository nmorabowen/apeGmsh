"""Unit tests for the explicit ``dt_cr`` pure helpers.

The live ``apeSees.critical_time_step`` / ``analyze_explicit`` methods
compose two pure functions — the sentinel handling and the sub-step
sizing — which are tested here without openseespy. The live behaviour is
covered (fork-gated) in ``tests/opensees/live/test_critical_timestep_live.py``.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.opensees.apesees import (
    _dtcr_or_raise,
    _explicit_substep_count,
)


class TestDtcrOrRaise:
    def test_positive_passes_through(self) -> None:
        assert _dtcr_or_raise(2.5e-4) == 2.5e-4

    def test_zero_raises_not_computed(self) -> None:
        with pytest.raises(ValueError, match="not computed"):
            _dtcr_or_raise(0.0)

    def test_negative_raises_not_applicable(self) -> None:
        with pytest.raises(ValueError, match="not applicable"):
            _dtcr_or_raise(-1.0)

    def test_negative_message_names_element_mass(self) -> None:
        # The actionable hint — element mass, not nodal mass — must be present.
        with pytest.raises(ValueError, match="element mass"):
            _dtcr_or_raise(-1.0)


class TestExplicitSubstepCount:
    def test_tiles_duration_exactly(self) -> None:
        n, dt = _explicit_substep_count(
            1.0, 1e-3, safety=0.9, dt_max=None,
        )
        # dt_stable = 0.9e-3 -> n = ceil(1/0.9e-3) = 1112
        assert n == math.ceil(1.0 / 0.9e-3)
        assert dt * n == pytest.approx(1.0)
        assert dt <= 0.9e-3 + 1e-15

    def test_dt_max_caps_the_step(self) -> None:
        # dt_stable would be 0.9e-3, but dt_max forces a finer step.
        n, dt = _explicit_substep_count(
            1.0, 1e-3, safety=0.9, dt_max=1e-4,
        )
        assert dt <= 1e-4 + 1e-15
        assert n == math.ceil(1.0 / 1e-4)
        assert dt * n == pytest.approx(1.0)

    def test_dt_max_above_stable_is_ignored(self) -> None:
        # dt_max looser than stability -> stability still governs.
        n_capped, dt_capped = _explicit_substep_count(
            1.0, 1e-3, safety=0.9, dt_max=1.0,
        )
        n_plain, dt_plain = _explicit_substep_count(
            1.0, 1e-3, safety=0.9, dt_max=None,
        )
        assert (n_capped, dt_capped) == (n_plain, dt_plain)

    def test_short_duration_floors_at_one_step(self) -> None:
        n, dt = _explicit_substep_count(
            1e-6, 1e-3, safety=0.9, dt_max=None,
        )
        assert n == 1
        assert dt == pytest.approx(1e-6)

    def test_bad_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration must be > 0"):
            _explicit_substep_count(0.0, 1e-3, safety=0.9, dt_max=None)

    def test_bad_safety_raises(self) -> None:
        with pytest.raises(ValueError, match="safety must be in"):
            _explicit_substep_count(1.0, 1e-3, safety=0.0, dt_max=None)
        with pytest.raises(ValueError, match="safety must be in"):
            _explicit_substep_count(1.0, 1e-3, safety=1.5, dt_max=None)

    def test_bad_dt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="dt_max must be > 0"):
            _explicit_substep_count(1.0, 1e-3, safety=0.9, dt_max=-1e-4)
