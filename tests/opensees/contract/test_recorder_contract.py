"""Contract tests for ``Recorder`` primitives.

Every concrete recorder class shipped by Phase 3B (and any follow-up
slice) is enumerated in :data:`ALL_RECORDERS`. The parametrized
contract suite verifies each class:

  * inherits from :class:`Recorder`
  * is decorated ``@dataclass(frozen=True, kw_only=True, slots=True)``
  * implements ``_emit`` and ``dependencies``
  * has ``__repr__`` that includes the class name
  * ``dependencies()`` on a minimal instance returns ``()``
    (recorders are leaves)

When a new typed recorder class lands, the agent appends it to
:data:`ALL_RECORDERS` (and to :data:`_MINIMAL_KWARGS`) — the
contract suite picks it up automatically.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import Primitive, Recorder
from apeGmsh.opensees.recorder import MPCO, Element, Node


ALL_RECORDERS: list[type[Recorder]] = [
    Node,
    Element,
    MPCO,
]


# Per-class minimal valid kwargs for constructing an instance. The
# contract tests need a real instance so they can call ``repr()`` and
# ``dependencies()``.
_MINIMAL_KWARGS: dict[type[Recorder], dict[str, Any]] = {
    Node: {
        "file": "x.out",
        "response": "disp",
        "nodes": (1,),
        "dofs": (1,),
    },
    Element: {
        "file": "x.out",
        "response": ("globalForce",),
        "elements": (1,),
    },
    MPCO: {
        "file": "run.mpco",
        "nodal_responses": ("displacement",),
    },
}


def _minimal_instance(cls: type[Recorder]) -> Recorder:
    return cls(**_MINIMAL_KWARGS[cls])


@pytest.mark.parametrize("cls", ALL_RECORDERS)
class TestRecorderContract:
    def test_inherits_from_recorder(self, cls: type[Recorder]) -> None:
        assert issubclass(cls, Recorder)
        assert issubclass(cls, Primitive)

    def test_is_frozen_kw_only_dataclass(
        self, cls: type[Recorder]
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params: Any = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass is not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass is not kw_only"

    def test_has_slots(self, cls: type[Recorder]) -> None:
        # @dataclass(slots=True) sets __slots__ on the class.
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} lacks __slots__"
        )

    def test_has_emit(self, cls: type[Recorder]) -> None:
        assert callable(getattr(cls, "_emit", None))

    def test_has_dependencies(self, cls: type[Recorder]) -> None:
        assert callable(getattr(cls, "dependencies", None))

    def test_repr_includes_class_name(
        self, cls: type[Recorder]
    ) -> None:
        instance = _minimal_instance(cls)
        assert cls.__name__ in repr(instance)

    def test_dependencies_returns_empty_tuple(
        self, cls: type[Recorder]
    ) -> None:
        # All Phase 3B recorders are leaves — no children primitives.
        instance = _minimal_instance(cls)
        assert instance.dependencies() == ()

    def test_fields_are_keyword_only(
        self, cls: type[Recorder]
    ) -> None:
        for f in fields(cls):
            assert f.kw_only is True, (
                f"{cls.__name__}.{f.name} should be kw_only"
            )

    def test_dataclass_has_at_least_one_field(
        self, cls: type[Recorder]
    ) -> None:
        # Every Recorder we ship has at least the ``file`` field; an
        # empty dataclass would suggest the class never made it past
        # the stub stage.
        assert len(fields(cls)) > 0
