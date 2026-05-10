"""Unit tests for the typed truss-family element primitives (Phase 2β).

For each class shipped by ``apeGmsh.opensees.element.truss``:

  * construction with valid parameters,
  * validation rejects bad inputs,
  * ``_emit`` records the correct OpenSees command on a
    :class:`RecordingEmitter` (with element-nodes and tag-resolver
    context installed),
  * ``_emit`` without the element-nodes context raises ``RuntimeError``,
  * ``dependencies`` returns the right materials,
  * ``__repr__`` mentions the class name.

The contract gate ``test_element_truss_contract.py`` parametrizes the
family-wide checks; this file exercises per-class behavior.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive, UniaxialMaterial
from apeGmsh.opensees.element.truss import (
    CorotTruss,
    InertiaTruss,
    Truss,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# ---------------------------------------------------------------------------
# Test-local UniaxialMaterial — same shape as Phase 1C tests.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMat(UniaxialMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _resolver_from(tags: dict[int, int]) -> object:
    """Return a callable that maps Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


# ===========================================================================
# Truss
# ===========================================================================

class TestTrussConstruction:
    def test_minimal(self) -> None:
        m = _FakeMat(name="steel")
        t = Truss(pg="braces", A=0.01, material=m)
        assert t.pg == "braces"
        assert t.A == 0.01
        assert t.material is m
        assert t.rho is None
        assert t.c_mass is False
        assert t.do_rayleigh is False

    def test_with_optional_flags(self) -> None:
        m = _FakeMat(name="steel")
        t = Truss(
            pg="braces", A=0.01, material=m,
            rho=7850.0, c_mass=True, do_rayleigh=True,
        )
        assert t.rho == 7850.0
        assert t.c_mass is True
        assert t.do_rayleigh is True

    def test_zero_area_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="A must be > 0"):
            Truss(pg="x", A=0.0, material=m)

    def test_negative_area_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="A must be > 0"):
            Truss(pg="x", A=-1.0, material=m)

    def test_negative_rho_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="rho must be >= 0"):
            Truss(pg="x", A=0.01, material=m, rho=-1.0)

    def test_zero_rho_accepted(self) -> None:
        # rho=0 is a valid pass-through value (massless truss).
        m = _FakeMat(name="x")
        t = Truss(pg="x", A=0.01, material=m, rho=0.0)
        assert t.rho == 0.0


class TestTrussEmit:
    def test_minimal(self) -> None:
        m = _FakeMat(name="steel")
        t = Truss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 5}))
        set_element_nodes(e, (10, 20))
        t._emit(e, tag=42)
        assert e.calls == [
            ("element", ("Truss", 42, 10, 20, 0.01, 5), {}),
        ]

    def test_with_rho(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m, rho=7850.0)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4))
        t._emit(e, tag=1)
        assert e.calls[0] == (
            "element",
            ("Truss", 1, 3, 4, 0.01, 1, "-rho", 7850.0),
            {},
        )

    def test_with_c_mass(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m, c_mass=True)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4))
        t._emit(e, tag=1)
        assert e.calls[0] == (
            "element",
            ("Truss", 1, 3, 4, 0.01, 1, "-cMass", 1),
            {},
        )

    def test_with_do_rayleigh(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m, do_rayleigh=True)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4))
        t._emit(e, tag=1)
        assert e.calls[0] == (
            "element",
            ("Truss", 1, 3, 4, 0.01, 1, "-doRayleigh", 1),
            {},
        )

    def test_with_all_flags(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(
            pg="b", A=0.01, material=m,
            rho=10.0, c_mass=True, do_rayleigh=True,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4))
        t._emit(e, tag=1)
        assert e.calls[0] == (
            "element",
            (
                "Truss", 1, 3, 4, 0.01, 1,
                "-rho", 10.0, "-cMass", 1, "-doRayleigh", 1,
            ),
            {},
        )

    def test_emit_without_nodes_raises(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        with pytest.raises(RuntimeError, match="element-nodes"):
            t._emit(e, tag=1)

    def test_emit_without_resolver_raises(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_element_nodes(e, (3, 4))
        with pytest.raises(RuntimeError, match="tag resolver"):
            t._emit(e, tag=1)

    def test_emit_with_wrong_node_count_raises(self) -> None:
        m = _FakeMat(name="x")
        t = Truss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4, 5))
        with pytest.raises(ValueError, match="expected 2 node tags"):
            t._emit(e, tag=1)


class TestTrussMisc:
    def test_dependencies_returns_material(self) -> None:
        m = _FakeMat(name="x")
        assert Truss(pg="b", A=0.01, material=m).dependencies() == (m,)

    def test_repr_includes_class_name(self) -> None:
        m = _FakeMat(name="x")
        assert "Truss" in repr(Truss(pg="b", A=0.01, material=m))


# ===========================================================================
# CorotTruss — same shape as Truss, different type token
# ===========================================================================

class TestCorotTruss:
    def test_construction(self) -> None:
        m = _FakeMat(name="x")
        t = CorotTruss(pg="b", A=0.01, material=m)
        assert t.A == 0.01

    def test_emit_uses_corot_token(self) -> None:
        m = _FakeMat(name="x")
        t = CorotTruss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 5}))
        set_element_nodes(e, (10, 20))
        t._emit(e, tag=42)
        assert e.calls == [
            ("element", ("CorotTruss", 42, 10, 20, 0.01, 5), {}),
        ]

    def test_with_all_flags(self) -> None:
        m = _FakeMat(name="x")
        t = CorotTruss(
            pg="b", A=0.01, material=m,
            rho=10.0, c_mass=True, do_rayleigh=True,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        set_element_nodes(e, (3, 4))
        t._emit(e, tag=1)
        assert e.calls[0] == (
            "element",
            (
                "CorotTruss", 1, 3, 4, 0.01, 1,
                "-rho", 10.0, "-cMass", 1, "-doRayleigh", 1,
            ),
            {},
        )

    def test_zero_area_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="A must be > 0"):
            CorotTruss(pg="x", A=0.0, material=m)

    def test_negative_rho_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="rho must be >= 0"):
            CorotTruss(pg="x", A=0.01, material=m, rho=-1.0)

    def test_dependencies_returns_material(self) -> None:
        m = _FakeMat(name="x")
        assert CorotTruss(pg="b", A=0.01, material=m).dependencies() == (m,)

    def test_repr_includes_class_name(self) -> None:
        m = _FakeMat(name="x")
        assert "CorotTruss" in repr(
            CorotTruss(pg="b", A=0.01, material=m)
        )

    def test_emit_without_nodes_raises(self) -> None:
        m = _FakeMat(name="x")
        t = CorotTruss(pg="b", A=0.01, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(m): 1}))
        with pytest.raises(RuntimeError, match="element-nodes"):
            t._emit(e, tag=1)


# ===========================================================================
# InertiaTruss — mass-only
# ===========================================================================

class TestInertiaTruss:
    def test_construction(self) -> None:
        t = InertiaTruss(pg="b", mass=100.0)
        assert t.pg == "b"
        assert t.mass == 100.0

    def test_emit(self) -> None:
        t = InertiaTruss(pg="b", mass=100.0)
        e = RecordingEmitter()
        # InertiaTruss has no material — no resolver needed, but
        # element_nodes must be set.
        set_element_nodes(e, (1, 2))
        t._emit(e, tag=7)
        assert e.calls == [
            ("element", ("InertiaTruss", 7, 1, 2, 100.0), {}),
        ]

    def test_zero_mass_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be > 0"):
            InertiaTruss(pg="x", mass=0.0)

    def test_negative_mass_rejected(self) -> None:
        with pytest.raises(ValueError, match="mass must be > 0"):
            InertiaTruss(pg="x", mass=-1.0)

    def test_dependencies_is_empty(self) -> None:
        # InertiaTruss has no material — leaf element.
        assert InertiaTruss(pg="x", mass=1.0).dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        assert "InertiaTruss" in repr(InertiaTruss(pg="x", mass=1.0))

    def test_emit_without_nodes_raises(self) -> None:
        t = InertiaTruss(pg="x", mass=1.0)
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="element-nodes"):
            t._emit(e, tag=1)
