"""Unit tests for the ``geomTransf`` primitives (Phase 1D).

Tests construction, validation, and ``_emit`` behaviour for
:class:`Linear`, :class:`PDelta`, :class:`Corotational`. Uses
:class:`RecordingEmitter` exclusively — never boots openseespy.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.transform import (
    Cartesian,
    Corotational,
    Cylindrical,
    Linear,
    PDelta,
    Spherical,
)


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class TestLinear:
    def test_construction_with_vecxz(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        assert t.vecxz == (0.0, 0.0, 1.0)
        assert t.csys is None
        assert t.roll_deg == 0.0

    def test_construction_with_csys_cartesian(self) -> None:
        cs = Cartesian()
        t = Linear(csys=cs)
        assert t.csys is cs
        assert t.vecxz is None

    def test_construction_with_csys_cylindrical(self) -> None:
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        t = Linear(csys=cs)
        assert t.csys is cs

    def test_construction_with_csys_spherical(self) -> None:
        cs = Spherical(origin=(0, 0, 0))
        t = Linear(csys=cs)
        assert t.csys is cs

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        # 2D-only path: bridge will raise at build time if used in 3D.
        t = Linear()
        assert t.csys is None
        assert t.vecxz is None

    def test_construction_with_both_csys_and_vecxz_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Linear: supply either"):
            Linear(csys=Cartesian(), vecxz=(0.0, 0.0, 1.0))

    def test_construction_with_roll_deg(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0), roll_deg=15.0)
        assert t.roll_deg == 15.0

    def test_emit_with_explicit_vecxz_records_correct_call(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=42)
        assert emitter.calls == [
            ("geomTransf", ("Linear", 42, 0.0, 0.0, 1.0), {}),
        ]

    def test_emit_with_arbitrary_vecxz(self) -> None:
        t = Linear(vecxz=(0.5, -0.3, 0.7))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=7)
        assert emitter.calls == [
            ("geomTransf", ("Linear", 7, 0.5, -0.3, 0.7), {}),
        ]

    def test_emit_with_csys_set_raises_not_implemented(self) -> None:
        t = Linear(csys=Cartesian())
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="csys-derived"):
            t._emit(emitter, tag=1)
        assert emitter.calls == []

    def test_emit_with_neither_raises_not_implemented(self) -> None:
        # Same path: vecxz is None, so the deferred-fan-out error fires.
        t = Linear()
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="csys-derived"):
            t._emit(emitter, tag=1)

    def test_dependencies_is_empty(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        assert t.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        assert "Linear" in repr(t)

    def test_is_frozen(self) -> None:
        t = Linear(vecxz=(0.0, 0.0, 1.0))
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            t.roll_deg = 90.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PDelta
# ---------------------------------------------------------------------------

class TestPDelta:
    def test_construction_with_vecxz(self) -> None:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        assert t.vecxz == (0.0, 0.0, 1.0)
        assert t.csys is None

    def test_construction_with_csys(self) -> None:
        cs = Cartesian()
        t = PDelta(csys=cs)
        assert t.csys is cs

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        t = PDelta()
        assert t.csys is None
        assert t.vecxz is None

    def test_construction_with_both_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="PDelta: supply either"):
            PDelta(csys=Cartesian(), vecxz=(0.0, 0.0, 1.0))

    def test_emit_with_explicit_vecxz(self) -> None:
        t = PDelta(vecxz=(1.0, 0.0, 0.0))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=3)
        assert emitter.calls == [
            ("geomTransf", ("PDelta", 3, 1.0, 0.0, 0.0), {}),
        ]

    def test_emit_with_csys_raises_not_implemented(self) -> None:
        t = PDelta(csys=Cartesian())
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="PDelta._emit"):
            t._emit(emitter, tag=1)

    def test_dependencies_is_empty(self) -> None:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        assert t.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        t = PDelta(vecxz=(0.0, 0.0, 1.0))
        assert "PDelta" in repr(t)


# ---------------------------------------------------------------------------
# Corotational
# ---------------------------------------------------------------------------

class TestCorotational:
    def test_construction_with_vecxz(self) -> None:
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        assert t.vecxz == (0.0, 0.0, 1.0)
        assert t.csys is None

    def test_construction_with_csys(self) -> None:
        cs = Cartesian()
        t = Corotational(csys=cs)
        assert t.csys is cs

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        t = Corotational()
        assert t.csys is None
        assert t.vecxz is None

    def test_construction_with_both_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Corotational: supply either"):
            Corotational(csys=Cartesian(), vecxz=(0.0, 0.0, 1.0))

    def test_emit_with_explicit_vecxz(self) -> None:
        t = Corotational(vecxz=(0.0, 1.0, 0.0))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=99)
        assert emitter.calls == [
            ("geomTransf", ("Corotational", 99, 0.0, 1.0, 0.0), {}),
        ]

    def test_emit_with_csys_raises_not_implemented(self) -> None:
        t = Corotational(csys=Cartesian())
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="Corotational._emit"):
            t._emit(emitter, tag=1)

    def test_dependencies_is_empty(self) -> None:
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        assert t.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        t = Corotational(vecxz=(0.0, 0.0, 1.0))
        assert "Corotational" in repr(t)


# ---------------------------------------------------------------------------
# Namespace integration — Phase 1D wires _GeomTransfNS
# ---------------------------------------------------------------------------

def _stub_bridge() -> apeSees:
    """Build an ``apeSees`` over a stub FEM. Used only for namespace tests."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


class TestGeomTransfNamespace:
    def test_namespace_Linear_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        t = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        assert isinstance(t, Linear)
        assert t.vecxz == (0.0, 0.0, 1.0)
        assert ops.tag_for(t) == 1

    def test_namespace_PDelta_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        t = ops.geomTransf.PDelta(csys=Cartesian())
        assert isinstance(t, PDelta)
        assert isinstance(t.csys, Cartesian)
        assert ops.tag_for(t) == 1

    def test_namespace_Corotational_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        t = ops.geomTransf.Corotational(vecxz=(1.0, 0.0, 0.0), roll_deg=30.0)
        assert isinstance(t, Corotational)
        assert t.roll_deg == 30.0

    def test_namespace_allocates_sequential_tags_within_geomtransf_kind(self) -> None:
        ops = _stub_bridge()
        t1 = ops.geomTransf.Linear(vecxz=(0.0, 0.0, 1.0))
        t2 = ops.geomTransf.PDelta(vecxz=(0.0, 0.0, 1.0))
        t3 = ops.geomTransf.Corotational(vecxz=(0.0, 0.0, 1.0))
        assert ops.tag_for(t1) == 1
        assert ops.tag_for(t2) == 2
        assert ops.tag_for(t3) == 3
