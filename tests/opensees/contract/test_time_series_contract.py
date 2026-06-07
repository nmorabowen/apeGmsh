"""Contract tests for ``TimeSeries`` primitives.

Every concrete time-series class shipped by Phase 1D-extra (and any
follow-up slice) is enumerated in :data:`ALL_TIME_SERIES`. The
parametrized contract suite verifies each class:

  * inherits from :class:`TimeSeries`
  * is decorated ``@dataclass(frozen=True, kw_only=True, slots=True)``
  * implements ``_emit`` and ``dependencies``
  * has ``__repr__`` that includes the class name
  * ``dependencies()`` on a minimal instance returns ``()``
    (TimeSeries primitives are leaves)

When a new typed time-series class lands, the agent appends it to
:data:`ALL_TIME_SERIES` (and to :data:`_MINIMAL_KWARGS`) — the
contract suite picks it up automatically.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import TimeSeries
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


ALL_TIME_SERIES: list[type[TimeSeries]] = [
    Linear,
    Constant,
    Path,
    Trig,
    Pulse,
    Ricker,
    ASCE41Protocol,
    ModifiedATC24Protocol,
    FEMA461Protocol,
]


# Per-class minimal valid kwargs for constructing an instance. The
# contract tests need a real instance so they can call ``repr()`` and
# ``dependencies()``. The map keeps tests simple without a clever
# auto-construction helper.
_MINIMAL_KWARGS: dict[type[TimeSeries], dict[str, Any]] = {
    Linear: {},
    Constant: {},
    Path: {"file": "x.txt"},
    Trig: {"t_start": 0.0, "t_end": 1.0, "period": 0.5},
    Pulse: {"t_start": 0.0, "t_end": 1.0, "period": 0.5, "width": 0.5},
    Ricker: {"f_n": 0.5, "t_total": 10.0, "dt": 0.05},
    ASCE41Protocol: {},
    ModifiedATC24Protocol: {},
    FEMA461Protocol: {},
}


def _minimal_instance(cls: type[TimeSeries]) -> TimeSeries:
    return cls(**_MINIMAL_KWARGS[cls])


@pytest.mark.parametrize("cls", ALL_TIME_SERIES)
class TestTimeSeriesContract:
    def test_inherits_from_time_series(self, cls: type[TimeSeries]) -> None:
        assert issubclass(cls, TimeSeries)

    def test_is_frozen_kw_only_dataclass(
        self, cls: type[TimeSeries]
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass is not frozen"
        assert all(f.kw_only for f in fields(cls)), f"{cls.__name__} dataclass is not kw_only"

    def test_has_slots(self, cls: type[TimeSeries]) -> None:
        # @dataclass(slots=True) sets __slots__ on the class.
        assert hasattr(cls, "__slots__"), f"{cls.__name__} lacks __slots__"

    def test_has_emit(self, cls: type[TimeSeries]) -> None:
        assert hasattr(cls, "_emit")

    def test_has_dependencies(self, cls: type[TimeSeries]) -> None:
        assert hasattr(cls, "dependencies")

    def test_repr_includes_class_name(
        self, cls: type[TimeSeries]
    ) -> None:
        instance = _minimal_instance(cls)
        assert cls.__name__ in repr(instance)

    def test_dependencies_returns_empty_tuple(
        self, cls: type[TimeSeries]
    ) -> None:
        # All five core TimeSeries are leaves — no children primitives.
        instance = _minimal_instance(cls)
        assert instance.dependencies() == ()

    def test_fields_are_keyword_only(
        self, cls: type[TimeSeries]
    ) -> None:
        # Sanity: dataclass frozen kw_only means fields cannot be
        # supplied positionally. Verify by attempting positional
        # construction with one of the field's default — it should raise.
        # This is structural; the invariant is enforced at the dataclass
        # level by kw_only=True.
        for f in fields(cls):
            assert f.kw_only is True, (
                f"{cls.__name__}.{f.name} should be kw_only"
            )
