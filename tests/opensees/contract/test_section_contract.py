"""Contract tests for the ``Section`` family.

Every concrete :class:`Section` shipped by Phase 1C is added to
``ALL_SECTIONS`` and verified against the family contract:

  * inherits from :class:`Section`
  * is a frozen, kw-only, slotted dataclass
  * implements ``_emit`` and ``dependencies``
  * the type token (the class name) appears in ``repr``

The contract list is **append-only within a phase** per the
parallel-execution conflict-avoidance rules.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees._internal.types import (
    NDMaterial,
    Primitive,
    Section,
    UniaxialMaterial,
)
from apeGmsh.opensees.section.beam import ElasticSection
from apeGmsh.opensees.section.fiber import Fiber, RectPatch
from apeGmsh.opensees.section.plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
)


ALL_SECTIONS: list[type[Section]] = [
    ElasticSection,
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
    Fiber,
]


# ---------------------------------------------------------------------------
# Minimal-instance factories — the contract test exercises each class
# but each class needs different valid kwargs. A factory lookup keeps
# the parametrize loop simple.
# ---------------------------------------------------------------------------

def _make_minimal(cls: type[Section]) -> Section:
    if cls is ElasticSection:
        return ElasticSection(E=2e11, A=0.01, Iz=1e-4)
    if cls is ElasticMembranePlateSection:
        return ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
    if cls is LayeredShell:
        m = _fake_nd()
        return LayeredShell(
            layers=(ShellLayer(material=m, thickness=0.1),)
        )
    if cls is LayeredShellFiberSection:
        m = _fake_nd()
        return LayeredShellFiberSection(
            layers=(ShellLayer(material=m, thickness=0.1),)
        )
    if cls is Fiber:
        m = _fake_uniaxial()
        return Fiber(
            patches=(
                RectPatch(
                    material=m, ny=1, nz=1,
                    yI=0, zI=0, yJ=1, zJ=1,
                ),
            )
        )
    raise NotImplementedError(
        f"Contract test needs a minimal-instance factory for {cls!r}."
    )


def _fake_uniaxial() -> UniaxialMaterial:
    fake = MagicMock(spec=UniaxialMaterial)
    return fake  # type: ignore[no-any-return]


def _fake_nd() -> NDMaterial:
    fake = MagicMock(spec=NDMaterial)
    return fake  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_SECTIONS)
class TestSectionContract:
    def test_inherits_from_section(self, cls: type[Section]) -> None:
        assert issubclass(cls, Section)
        assert issubclass(cls, Primitive)

    def test_has_emit(self, cls: type[Section]) -> None:
        assert callable(getattr(cls, "_emit", None))

    def test_has_dependencies(self, cls: type[Section]) -> None:
        assert callable(getattr(cls, "dependencies", None))

    def test_is_frozen_kw_only_slotted_dataclass(
        self, cls: type[Section]
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params: Any = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass not kw_only"
        # Slots are visible via __slots__ on the class itself.
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} not slotted (no __slots__)"
        )

    def test_dataclass_has_at_least_one_field(
        self, cls: type[Section]
    ) -> None:
        # Every Section we ship has parameters; an empty dataclass
        # would suggest the class never made it past the stub stage.
        assert len(fields(cls)) > 0

    def test_repr_contains_class_name(self, cls: type[Section]) -> None:
        instance = _make_minimal(cls)
        assert cls.__name__ in repr(instance)
