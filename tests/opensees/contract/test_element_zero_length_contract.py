"""Family contract gate for typed zero-length elements (Phase 2β).

Every concrete subclass of :class:`Element` shipped by the
zero-length sub-slice is listed in ``ALL_ZERO_LENGTH_ELEMENTS`` and
verified against the family contract:

  * inherits from :class:`Element` (and therefore :class:`Primitive`),
  * is a frozen, kw-only, slotted dataclass,
  * implements ``_emit`` and ``dependencies``,
  * ``__repr__`` includes the class name,
  * a minimal-valid instance emits a single ``("element", ...)`` call
    when given an element-nodes context and a tag resolver.

Future zero-length-family slices append their classes to
``ALL_ZERO_LENGTH_ELEMENTS`` and supply factories in
``_MINIMAL_FACTORIES`` so the smoke-emit covers them automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import (
    Element,
    Primitive,
    Section,
    UniaxialMaterial,
)
from apeGmsh.opensees.element.zero_length import (
    ZeroLength,
    ZeroLengthMatDir,
    ZeroLengthSection,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter


ALL_ZERO_LENGTH_ELEMENTS: list[type[Element]] = [
    ZeroLength,
    ZeroLengthSection,
]


# ---------------------------------------------------------------------------
# Test-local UniaxialMaterial / Section
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMat(UniaxialMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeSection(Section):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.section("FakeSection", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


_FAKE_MAT = _FakeMat(name="contract")
_FAKE_SEC = _FakeSection(name="contract")


_MINIMAL_FACTORIES: dict[type[Element], Callable[[], Element]] = {
    ZeroLength: lambda: ZeroLength(
        pg="c",
        mat_dirs=(ZeroLengthMatDir(material=_FAKE_MAT, dof=1),),
    ),
    ZeroLengthSection: lambda: ZeroLengthSection(
        pg="c", section=_FAKE_SEC,
    ),
}


def _minimal(cls: type[Element]) -> Element:
    """Construct a minimal-valid instance for smoke tests."""
    return _MINIMAL_FACTORIES[cls]()


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_ZERO_LENGTH_ELEMENTS)
class TestZeroLengthElementContract:
    def test_inherits_from_element_base(
        self, cls: type[Element],
    ) -> None:
        assert issubclass(cls, Element)
        assert issubclass(cls, Primitive)

    def test_is_frozen_kw_only_slotted_dataclass(
        self, cls: type[Element],
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params: Any = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass not kw_only"
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} not slotted"
        )

    def test_has_emit_and_dependencies(
        self, cls: type[Element],
    ) -> None:
        assert "_emit" in cls.__dict__
        assert "dependencies" in cls.__dict__

    def test_dataclass_has_pg_field(self, cls: type[Element]) -> None:
        names = {f.name for f in fields(cls)}
        assert "pg" in names

    def test_repr_includes_class_name(
        self, cls: type[Element],
    ) -> None:
        assert cls.__name__ in repr(_minimal(cls))

    def test_emit_records_single_element_call(
        self, cls: type[Element],
    ) -> None:
        e = RecordingEmitter()
        # Both zero-length classes compose a primitive (material or
        # section) whose tag must be resolvable.
        set_tag_resolver(e, lambda p: 1)
        set_element_nodes(e, (10, 20))
        _minimal(cls)._emit(e, tag=99)
        assert len(e.calls) == 1
        assert e.calls[0][0] == "element"
