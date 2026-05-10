"""Contract tests for the ``GeomTransf`` family.

Every concrete ``geomTransf`` primitive must satisfy:

  * inherits from :class:`GeomTransf`
  * is a frozen, kw-only, slotted dataclass (per P12 / api-design.md)
  * defines ``_emit`` and ``dependencies``
  * has the standard ``csys`` / ``vecxz`` / ``roll_deg`` parameter
    surface (Phase 1D contract from ADR 0010)
  * ``__repr__`` includes the class name

When Phase 1+ slices add new transforms they append to
``ALL_GEOM_TRANSF`` and these tests parameterize over the new entries
automatically.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, cast

import pytest

from apeGmsh.opensees._internal.types import GeomTransf, Primitive
from apeGmsh.opensees.transform import Cartesian, Corotational, Linear, PDelta


# The contract suite parametrizes over concrete transforms. We type the
# list as ``type[GeomTransf]`` so adding new transforms keeps the same
# signature; the construction-call sites use ``cast(Any, cls)`` because
# the abstract base does not advertise the concrete fields.
ALL_GEOM_TRANSF: list[type[GeomTransf]] = [
    Linear,
    PDelta,
    Corotational,
]


@pytest.mark.parametrize("cls", ALL_GEOM_TRANSF)
class TestGeomTransfContract:
    def test_inherits_from_geom_transf(
        self, cls: type[GeomTransf]
    ) -> None:
        assert issubclass(cls, GeomTransf)

    def test_inherits_from_primitive(
        self, cls: type[GeomTransf]
    ) -> None:
        assert issubclass(cls, Primitive)

    def test_is_frozen_dataclass(self, cls: type[GeomTransf]) -> None:
        assert is_dataclass(cls)
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} is not frozen"
        assert params.kw_only, f"{cls.__name__} is not kw_only"

    def test_has_slots(self, cls: type[GeomTransf]) -> None:
        # slots=True on a dataclass installs __slots__ on the class.
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} missing __slots__ (slots=True required)"
        )

    def test_has_emit(self, cls: type[GeomTransf]) -> None:
        assert hasattr(cls, "_emit") and callable(cls._emit)

    def test_has_dependencies(self, cls: type[GeomTransf]) -> None:
        assert hasattr(cls, "dependencies") and callable(cls.dependencies)

    def test_field_surface_csys_vecxz_roll_deg(
        self, cls: type[GeomTransf]
    ) -> None:
        names = {f.name for f in fields(cls)}
        # Phase 1D contract: every transform exposes the same three
        # construction parameters. New transforms (e.g. PDeltaJoint
        # offsets) may add fields but cannot drop these.
        assert {"csys", "vecxz", "roll_deg"} <= names

    def test_dependencies_returns_empty_tuple_for_leaves(
        self, cls: type[GeomTransf]
    ) -> None:
        # All Phase 1D transforms are leaves (no composed primitives).
        instance = cast(Any, cls)(vecxz=(0.0, 0.0, 1.0))
        assert instance.dependencies() == ()

    def test_repr_includes_class_name(
        self, cls: type[GeomTransf]
    ) -> None:
        instance = cast(Any, cls)(vecxz=(0.0, 0.0, 1.0))
        assert cls.__name__ in repr(instance)

    def test_construction_with_both_csys_and_vecxz_raises(
        self, cls: type[GeomTransf]
    ) -> None:
        with pytest.raises(ValueError, match="supply either"):
            cast(Any, cls)(csys=Cartesian(), vecxz=(0.0, 0.0, 1.0))
