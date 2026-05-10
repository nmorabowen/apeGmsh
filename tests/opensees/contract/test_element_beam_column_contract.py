"""Family contract gate for typed beam-column elements (Phase 2α).

Mirrors the shape of the truss / shell / solid family contract tests:
every concrete subclass of :class:`Element` shipped by the beam-column
sub-slice is listed in ``ALL_BEAM_COLUMN_ELEMENTS`` and verified against
the family contract (frozen kw-only slotted dataclass, has ``_emit`` /
``dependencies``, ``pg`` field, ``__repr__`` includes class name, smoke
emit produces a single ``("element", ...)`` call).
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
    GeomTransf,
    Primitive,
    Section,
)
from apeGmsh.opensees.element.beam_column import (
    ElasticTimoshenkoBeam,
    dispBeamColumn,
    elasticBeamColumn,
    forceBeamColumn,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.integration import Lobatto


ALL_BEAM_COLUMN_ELEMENTS: list[type[Element]] = [
    elasticBeamColumn,
    forceBeamColumn,
    dispBeamColumn,
    ElasticTimoshenkoBeam,
]


# ---------------------------------------------------------------------------
# Test-local stubs — minimal Section + GeomTransf the typed elements compose.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeSection(Section):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.section("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeTransf(GeomTransf):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.geomTransf("Fake", tag, 1.0, 0.0, 0.0)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


_FAKE_SEC = _FakeSection(name="contract")
_FAKE_TRANSF = _FakeTransf(name="contract")
_FAKE_INTEG = Lobatto(section=_FAKE_SEC, n_ip=5)


_MINIMAL_PARAMS: dict[type[Element], dict[str, Any]] = {
    elasticBeamColumn: {
        "pg": "c", "transf": _FAKE_TRANSF,
        "A": 0.01, "E": 200e9, "Iz": 1e-4,
    },
    forceBeamColumn: {
        "pg": "c", "transf": _FAKE_TRANSF, "integration": _FAKE_INTEG,
    },
    dispBeamColumn: {
        "pg": "c", "transf": _FAKE_TRANSF, "integration": _FAKE_INTEG,
    },
    ElasticTimoshenkoBeam: {
        "pg": "c", "transf": _FAKE_TRANSF,
        "E": 200e9, "G": 80e9, "A": 0.01, "Iz": 1e-4, "Avy": 0.005,
    },
}


def _minimal(cls: type[Element]) -> Element:
    return cls(**_MINIMAL_PARAMS[cls])


# ---------------------------------------------------------------------------
# Contract assertions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_BEAM_COLUMN_ELEMENTS)
class TestBeamColumnElementContract:
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
        assert params.frozen, f"{cls.__name__} not frozen"
        assert params.kw_only, f"{cls.__name__} not kw_only"
        assert hasattr(cls, "__slots__"), f"{cls.__name__} not slotted"

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
        # Beam-column elements all compose at least a transform; force /
        # disp also compose a section. Use a single resolver that maps
        # any registered Primitive to tag 1.
        set_tag_resolver(e, lambda p: 1)
        set_element_nodes(e, (10, 20))
        _minimal(cls)._emit(e, tag=99)
        assert len(e.calls) == 1
        assert e.calls[0][0] == "element"
