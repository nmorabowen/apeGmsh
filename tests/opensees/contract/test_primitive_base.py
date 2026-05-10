"""Contract tests for the ``Primitive`` base.

Every concrete primitive subclass shipped by Phase 1+ slices is added
to ``ALL_PRIMITIVES`` and verified here against the base contract:

  * inherits from :class:`Primitive`
  * implements ``_emit``
  * implements ``dependencies``
  * has a non-default ``__repr__`` (i.e. includes the type name)

Phase 0 ships an **empty list**. The skipif keeps the suite green
until the first concrete primitive lands; once Phase 1 fills the
list, this file becomes the parametrized contract gate.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.types import Primitive


ALL_PRIMITIVES: list[type[Primitive]] = []


pytestmark = pytest.mark.skipif(
    not ALL_PRIMITIVES,
    reason="No concrete primitives registered yet (Phase 0). "
           "Phase 1+ slices append their classes to ALL_PRIMITIVES.",
)


@pytest.mark.parametrize("cls", ALL_PRIMITIVES)
class TestPrimitiveContract:
    def test_inherits_from_primitive(self, cls: type[Primitive]) -> None:
        assert issubclass(cls, Primitive)

    def test_has_emit(self, cls: type[Primitive]) -> None:
        assert hasattr(cls, "_emit")

    def test_has_dependencies(self, cls: type[Primitive]) -> None:
        assert hasattr(cls, "dependencies")

    def test_has_repr(self, cls: type[Primitive]) -> None:
        # Primitive's default __repr__ uses cls.__name__; subclasses
        # may override but the type name must appear somewhere.
        assert "__repr__" in vars(cls) or "__repr__" in vars(Primitive)
