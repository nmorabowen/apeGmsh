"""Contract tests for the Phase 2δ ``Element`` solid family.

Every concrete :class:`Element` shipped by Phase 2δ is added to
``ALL_SOLID_ELEMENTS`` and verified against the family contract:

  * inherits from :class:`Element` (and transitively :class:`Primitive`).
  * is a frozen, kw-only, slotted dataclass.
  * implements ``_emit`` and ``dependencies``.
  * the Python class name appears in ``repr``.
  * ``_emit`` records exactly one ``Emitter.element`` call when the
    bridge has installed the tag resolver and per-element node tags.
  * the type-token used in the recorded ``element`` call is the one
    documented for that class (which **differs from the Python class
    name** for the 2D quad and tri).

The contract list is **append-only within a phase** per the parallel-
execution conflict-avoidance rules.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Element, Primitive
from apeGmsh.opensees.element.solid import (
    FourNodeQuad,
    FourNodeTetrahedron,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


ALL_SOLID_ELEMENTS: list[type[Element]] = [
    FourNodeTetrahedron,
    TenNodeTetrahedron,
    stdBrick,
    FourNodeQuad,
    Tri31,
]


# ---------------------------------------------------------------------------
# Per-class metadata: minimal-instance kwargs, expected node count, and
# the OpenSees Tcl type-token (which differs from the Python class name
# for the 2D quad and tri elements).
# ---------------------------------------------------------------------------

_NODE_COUNT: dict[type[Element], int] = {
    FourNodeTetrahedron: 4,
    TenNodeTetrahedron: 10,
    stdBrick: 8,
    FourNodeQuad: 4,
    Tri31: 3,
}


_TYPE_TOKEN: dict[type[Element], str] = {
    FourNodeTetrahedron: "FourNodeTetrahedron",
    TenNodeTetrahedron:  "TenNodeTetrahedron",
    stdBrick:            "stdBrick",
    # The 2D plane elements emit lowercase Tcl tokens that differ from
    # their Python class names. See ``element/solid.py`` for the
    # rationale.
    FourNodeQuad:        "quad",
    Tri31:               "tri31",
}


def _make_minimal(cls: type[Element]) -> Element:
    """Build a minimal valid instance of ``cls``."""
    m = ElasticIsotropic(E=30e9, nu=0.2)
    if cls in (FourNodeTetrahedron, TenNodeTetrahedron, stdBrick):
        return cls(pg="Body", material=m)  # type: ignore[call-arg]
    if cls in (FourNodeQuad, Tri31):
        return cls(pg="Plate", thickness=0.1, material=m)  # type: ignore[call-arg]
    raise NotImplementedError(
        f"Contract test needs a minimal-instance factory for {cls!r}."
    )


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_SOLID_ELEMENTS)
class TestElementSolidContract:
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

    def test_repr_contains_class_name(self, cls: type[Element]) -> None:
        instance = _make_minimal(cls)
        assert cls.__name__ in repr(instance)

    def test_dependencies_returns_tuple(self, cls: type[Element]) -> None:
        instance = _make_minimal(cls)
        deps = instance.dependencies()
        assert isinstance(deps, tuple)

    def test_emit_records_one_element_call(
        self, cls: type[Element]
    ) -> None:
        """Every solid element emits exactly one ``Emitter.element``
        call; the type-token matches the documented OpenSees Tcl
        token (which may differ from the Python class name)."""
        instance = _make_minimal(cls)
        # The minimal instance carries ElasticIsotropic as its
        # ``material`` attribute. We resolve that material to tag 99
        # and feed the right number of node tags into the per-
        # element-nodes context.
        material = getattr(instance, "material")
        emitter = RecordingEmitter()
        set_tag_resolver(
            emitter,
            lambda prim: 99 if id(prim) == id(material) else 0,
        )
        node_count = _NODE_COUNT[cls]
        set_element_nodes(
            emitter, tuple(range(1, node_count + 1)),
        )
        instance._emit(emitter, tag=42)

        assert len(emitter.calls) == 1
        method, args, _kwargs = emitter.calls[0]
        assert method == "element"
        assert args[0] == _TYPE_TOKEN[cls]
        assert args[1] == 42
        # The next ``node_count`` positional args are the element node
        # tags in the order set on the emitter.
        assert args[2 : 2 + node_count] == tuple(range(1, node_count + 1))
