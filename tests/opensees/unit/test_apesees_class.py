"""Unit tests for the ``apeSees`` bridge class skeleton (Phase 0).

These tests verify only the foundation shape — construction, namespace
presence, registration, build. Concrete primitive type methods land
in Phase 1+ and are tested in ``unit/primitives/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.types import Primitive, UniaxialMaterial
from apeGmsh.opensees.emitter.base import Emitter


# ---------------------------------------------------------------------------
# Test-local primitive: a fake UniaxialMaterial just for register/tag tests.
# Lives here (not under fixtures/) because it is private to this test file.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeMaterial(UniaxialMaterial):
    fy: float

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.uniaxialMaterial("FakeMaterial", tag, self.fy)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _stub_fem() -> object:
    """Return a MagicMock posing as a FEMData snapshot."""
    return MagicMock(name="FEMData")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_apesees_constructs_with_fem() -> None:
    fem = _stub_fem()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    assert ops is not None


def test_apesees_holds_fem_reference() -> None:
    fem = _stub_fem()
    ops = apeSees(cast("object", fem))  # type: ignore[arg-type]
    assert ops.fem is fem


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

NAMESPACE_NAMES = (
    "uniaxialMaterial",
    "nDMaterial",
    "section",
    "geomTransf",
    "timeSeries",
    "pattern",
    "element",
    "recorder",
    "constraints",
    "numberer",
    "system",
    "test",
    "algorithm",
    "integrator",
    "analysis",
)


@pytest.mark.parametrize("name", NAMESPACE_NAMES)
def test_apesees_namespace_is_present(name: str) -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    assert hasattr(ops, name), f"missing namespace: {name}"


# ---------------------------------------------------------------------------
# Model dimensionality
# ---------------------------------------------------------------------------

def test_apesees_model_sets_ndm_ndf() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    assert ops._ndm == 3
    assert ops._ndf == 6


def test_apesees_build_requires_model_first() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="apeSees.model"):
        ops.build()


# ---------------------------------------------------------------------------
# Registration / tag allocation
# ---------------------------------------------------------------------------

def test_apesees_register_returns_same_instance() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    m = _FakeMaterial(fy=420.0)
    out = ops.register(m)
    assert out is m


def test_apesees_register_allocates_tag() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    m = _FakeMaterial(fy=420.0)
    ops.register(m)
    assert ops.tag_for(m) == 1


def test_apesees_register_is_idempotent_on_same_instance() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    m = _FakeMaterial(fy=420.0)
    ops.register(m)
    ops.register(m)
    assert ops.tag_for(m) == 1  # second registration must not bump


def test_apesees_register_distinct_primitives_get_distinct_tags() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    a = _FakeMaterial(fy=420.0)
    b = _FakeMaterial(fy=520.0)
    ops.register(a)
    ops.register(b)
    assert ops.tag_for(a) == 1
    assert ops.tag_for(b) == 2


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def test_apesees_build_returns_built_model() -> None:
    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    m = _FakeMaterial(fy=420.0)
    ops.register(m)

    bm = ops.build()
    assert bm.ndm == 3
    assert bm.ndf == 6
    assert bm.primitives == (m,)
    assert bm.tag_for[id(m)] == 1


def test_apesees_unknown_primitive_kind_raises() -> None:
    """A primitive that doesn't inherit a known family base is rejected."""

    @dataclass(frozen=True, kw_only=True, slots=True)
    class _OrphanPrimitive(Primitive):
        x: float

        def _emit(self, emitter: Emitter, tag: int) -> None:
            emitter.uniaxialMaterial("X", tag, self.x)

        def dependencies(self) -> tuple[Primitive, ...]:
            return ()

    ops = apeSees(cast("object", _stub_fem()))  # type: ignore[arg-type]
    orphan = _OrphanPrimitive(x=1.0)
    with pytest.raises(TypeError, match="family base"):
        ops.register(orphan)
