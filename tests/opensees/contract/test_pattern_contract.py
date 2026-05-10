"""Contract tests for ``Pattern`` primitives.

Every concrete pattern class shipped by Phase 3A (and any follow-up
slice) is enumerated in :data:`ALL_PATTERNS`. The parametrized contract
suite verifies each class:

  * inherits from :class:`Pattern` (and transitively :class:`Primitive`).
  * is decorated ``@dataclass(frozen=True, kw_only=True, slots=True)``.
  * implements ``_emit`` and ``dependencies``.
  * has ``__repr__`` that includes the class name.
  * implements the context-manager protocol (``__enter__`` / ``__exit__``).
  * ``dependencies()`` on a minimal instance returns ``(series,)`` —
    every shipping pattern composes exactly one TimeSeries.

When a new typed pattern class lands, the agent appends it to
:data:`ALL_PATTERNS` (and to :data:`_MINIMAL_KWARGS`) — the contract
suite picks it up automatically.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import Pattern, Primitive
from apeGmsh.opensees.pattern.pattern import Plain, UniformExcitation
from apeGmsh.opensees.time_series.time_series import Linear


ALL_PATTERNS: list[type[Pattern]] = [
    Plain,
    UniformExcitation,
]


# Per-class minimal valid kwargs for constructing an instance. The
# contract tests need a real instance so they can call ``repr()`` and
# ``dependencies()``. The map keeps tests simple without a clever
# auto-construction helper.
_MINIMAL_KWARGS: dict[type[Pattern], dict[str, Any]] = {
    Plain: {"series": Linear()},
    UniformExcitation: {"direction": 1, "series": Linear()},
}


def _minimal_instance(cls: type[Pattern]) -> Pattern:
    return cls(**_MINIMAL_KWARGS[cls])


@pytest.mark.parametrize("cls", ALL_PATTERNS)
class TestPatternContract:
    def test_inherits_from_pattern(self, cls: type[Pattern]) -> None:
        assert issubclass(cls, Pattern)
        assert issubclass(cls, Primitive)

    def test_is_frozen_kw_only_dataclass(
        self, cls: type[Pattern]
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass is not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass is not kw_only"

    def test_has_slots(self, cls: type[Pattern]) -> None:
        assert hasattr(cls, "__slots__"), f"{cls.__name__} lacks __slots__"

    def test_has_emit(self, cls: type[Pattern]) -> None:
        assert callable(getattr(cls, "_emit", None))

    def test_has_dependencies(self, cls: type[Pattern]) -> None:
        assert callable(getattr(cls, "dependencies", None))

    def test_is_context_manager(self, cls: type[Pattern]) -> None:
        # Patterns are explicit context managers per ADR 0005.
        assert callable(getattr(cls, "__enter__", None))
        assert callable(getattr(cls, "__exit__", None))

    def test_repr_includes_class_name(
        self, cls: type[Pattern]
    ) -> None:
        instance = _minimal_instance(cls)
        assert cls.__name__ in repr(instance)

    def test_dependencies_returns_series(
        self, cls: type[Pattern]
    ) -> None:
        # Every shipping pattern in Phase 3A composes exactly one
        # TimeSeries (the ``series=`` kwarg). dependencies() returns
        # that single primitive in a one-tuple.
        instance = _minimal_instance(cls)
        deps = instance.dependencies()
        assert isinstance(deps, tuple)
        assert len(deps) == 1
        assert deps[0] is _MINIMAL_KWARGS[cls]["series"]

    def test_fields_are_keyword_only(
        self, cls: type[Pattern]
    ) -> None:
        # Every field that participates in ``__init__`` must be
        # keyword-only (frozen kw_only dataclass invariant). Skip the
        # ``init=False`` private accumulators on Plain.
        for f in fields(cls):
            if not f.init:
                continue
            assert f.kw_only is True, (
                f"{cls.__name__}.{f.name} should be kw_only"
            )
