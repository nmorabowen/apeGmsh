"""Contract tests for shell element primitives (Phase 2γ).

Every concrete shell-:class:`Element` shipped by Phase 2γ is added to
:data:`ALL_SHELL_ELEMENTS` and verified against the family contract:

  * inherits from :class:`Element` (and transitively :class:`Primitive`)
  * is a frozen, kw-only, slotted dataclass
  * implements ``_emit`` and ``dependencies``
  * the type token (the class name) appears in ``repr``
  * ``dependencies()`` on a minimal instance returns a 1-tuple of the
    composed Section.

The contract list is **append-only within a phase** per the
parallel-execution conflict-avoidance rules. Other Phase 2 element
families (beam_column, truss, zero_length, solid, joint) maintain
their own ``ALL_*_ELEMENTS`` contract lists.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import Element, Primitive
from apeGmsh.opensees.element.shell import (
    ASDShellQ4,
    ASDShellT3,
    ShellDKGQ,
    ShellMITC3,
    ShellMITC4,
)
from apeGmsh.opensees.section.plate import ElasticMembranePlateSection


ALL_SHELL_ELEMENTS: list[type[Element]] = [
    ShellMITC4,
    ShellMITC3,
    ShellDKGQ,
    ASDShellQ4,
    ASDShellT3,
]


# ---------------------------------------------------------------------------
# Minimal-instance factory — every shell class composes a Section, so the
# factory produces a fresh section per call to keep instances independent.
# ---------------------------------------------------------------------------

def _make_section() -> ElasticMembranePlateSection:
    return ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)


def _make_minimal(cls: type[Element]) -> Element:
    return cls(pg="ShellPG", section=_make_section())


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_SHELL_ELEMENTS)
class TestShellElementContract:
    def test_inherits_from_element(self, cls: type[Element]) -> None:
        assert issubclass(cls, Element)
        assert issubclass(cls, Primitive)

    def test_has_emit(self, cls: type[Element]) -> None:
        assert callable(getattr(cls, "_emit", None))

    def test_has_dependencies(self, cls: type[Element]) -> None:
        assert callable(getattr(cls, "dependencies", None))

    def test_is_frozen_kw_only_slotted_dataclass(
        self, cls: type[Element]
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params: Any = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass not kw_only"
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} not slotted (no __slots__)"
        )

    def test_dataclass_has_at_least_one_field(
        self, cls: type[Element]
    ) -> None:
        assert len(fields(cls)) > 0

    def test_repr_contains_class_name(
        self, cls: type[Element]
    ) -> None:
        instance = _make_minimal(cls)
        assert cls.__name__ in repr(instance)

    def test_dependencies_returns_section(
        self, cls: type[Element]
    ) -> None:
        s = _make_section()
        ele = cls(pg="ShellPG", section=s)  # type: ignore[call-arg]
        assert ele.dependencies() == (s,)

    def test_fields_are_keyword_only(self, cls: type[Element]) -> None:
        for f in fields(cls):
            assert f.kw_only is True, (
                f"{cls.__name__}.{f.name} should be kw_only"
            )
