"""Family contract gate for typed truss elements (Phase 2β).

Every concrete subclass of :class:`Element` shipped by the truss
sub-slice is listed in ``ALL_TRUSS_ELEMENTS`` and verified against
the family contract:

  * inherits from :class:`Element` (and therefore :class:`Primitive`),
  * is a frozen, kw-only, slotted dataclass,
  * implements ``_emit`` and ``dependencies``,
  * ``__repr__`` includes the class name,
  * a minimal-valid instance emits a single ``("element", ...)`` call
    when given an element-nodes context (and a tag resolver for
    elements that depend on a material).

Future truss-family slices append their classes to
``ALL_TRUSS_ELEMENTS`` and add an entry to ``_MINIMAL_PARAMS`` /
``_NEEDS_RESOLVER`` so the smoke-emit covers them automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import (
    Element,
    Primitive,
    UniaxialMaterial,
)
from apeGmsh.opensees.element.truss import (
    CorotTruss,
    InertiaTruss,
    Truss,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter


ALL_TRUSS_ELEMENTS: list[type[Element]] = [
    Truss,
    CorotTruss,
    InertiaTruss,
]


# ---------------------------------------------------------------------------
# Test-local UniaxialMaterial — used by Truss / CorotTruss minimal
# instances. InertiaTruss has no material dep.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMat(UniaxialMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


_FAKE_MAT = _FakeMat(name="contract")


_MINIMAL_PARAMS: dict[type[Element], dict[str, Any]] = {
    Truss: {"pg": "c", "A": 0.01, "material": _FAKE_MAT},
    CorotTruss: {"pg": "c", "A": 0.01, "material": _FAKE_MAT},
    InertiaTruss: {"pg": "c", "mass": 100.0},
}

#: Truss / CorotTruss compose a UniaxialMaterial — emit needs a tag
#: resolver. InertiaTruss is a leaf — no resolver needed.
_NEEDS_RESOLVER: set[type[Element]] = {Truss, CorotTruss}


def _minimal(cls: type[Element]) -> Element:
    """Construct a minimal-valid instance for smoke tests."""
    return cls(**_MINIMAL_PARAMS[cls])


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_TRUSS_ELEMENTS)
class TestTrussElementContract:
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
        # All Phase 2 element specs carry a physical-group label.
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
        if cls in _NEEDS_RESOLVER:
            set_tag_resolver(e, lambda p: 1)
        set_element_nodes(e, (10, 20))
        _minimal(cls)._emit(e, tag=99)
        assert len(e.calls) == 1
        assert e.calls[0][0] == "element"
