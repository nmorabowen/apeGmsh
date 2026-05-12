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
        assert t.orientation is None
        assert t.roll_deg == 0.0

    def test_construction_with_orientation_cartesian(self) -> None:
        o = Cartesian()
        t = Linear(orientation=o)
        assert t.orientation is o
        assert t.vecxz is None

    def test_construction_with_orientation_cylindrical(self) -> None:
        o = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        t = Linear(orientation=o)
        assert t.orientation is o

    def test_construction_with_orientation_spherical(self) -> None:
        o = Spherical(origin=(0, 0, 0))
        t = Linear(orientation=o)
        assert t.orientation is o

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        # 2D-only path: bridge will raise at build time if used in 3D.
        t = Linear()
        assert t.orientation is None
        assert t.vecxz is None

    def test_construction_with_both_orientation_and_vecxz_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Linear: supply either"):
            Linear(orientation=Cartesian(), vecxz=(0.0, 0.0, 1.0))

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

    def test_emit_with_orientation_set_raises_not_implemented(self) -> None:
        t = Linear(orientation=Cartesian())
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="orientation-derived"):
            t._emit(emitter, tag=1)
        assert emitter.calls == []

    def test_emit_with_neither_raises_not_implemented(self) -> None:
        # Same path: vecxz is None, so the deferred-fan-out error fires.
        t = Linear()
        emitter = RecordingEmitter()
        with pytest.raises(NotImplementedError, match="orientation-derived"):
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
        assert t.orientation is None

    def test_construction_with_orientation(self) -> None:
        o = Cartesian()
        t = PDelta(orientation=o)
        assert t.orientation is o

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        t = PDelta()
        assert t.orientation is None
        assert t.vecxz is None

    def test_construction_with_both_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="PDelta: supply either"):
            PDelta(orientation=Cartesian(), vecxz=(0.0, 0.0, 1.0))

    def test_emit_with_explicit_vecxz(self) -> None:
        t = PDelta(vecxz=(1.0, 0.0, 0.0))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=3)
        assert emitter.calls == [
            ("geomTransf", ("PDelta", 3, 1.0, 0.0, 0.0), {}),
        ]

    def test_emit_with_orientation_raises_not_implemented(self) -> None:
        t = PDelta(orientation=Cartesian())
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
        assert t.orientation is None

    def test_construction_with_orientation(self) -> None:
        o = Cartesian()
        t = Corotational(orientation=o)
        assert t.orientation is o

    def test_construction_with_neither_is_allowed_for_2d_models(self) -> None:
        t = Corotational()
        assert t.orientation is None
        assert t.vecxz is None

    def test_construction_with_both_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Corotational: supply either"):
            Corotational(orientation=Cartesian(), vecxz=(0.0, 0.0, 1.0))

    def test_emit_with_explicit_vecxz(self) -> None:
        t = Corotational(vecxz=(0.0, 1.0, 0.0))
        emitter = RecordingEmitter()
        t._emit(emitter, tag=99)
        assert emitter.calls == [
            ("geomTransf", ("Corotational", 99, 0.0, 1.0, 0.0), {}),
        ]

    def test_emit_with_orientation_raises_not_implemented(self) -> None:
        t = Corotational(orientation=Cartesian())
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
        t = ops.geomTransf.PDelta(orientation=Cartesian())
        assert isinstance(t, PDelta)
        assert isinstance(t.orientation, Cartesian)
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


# ---------------------------------------------------------------------------
# Default-orientation substitution — _GeomTransfNS._resolve_orientation
# ---------------------------------------------------------------------------

class TestDefaultOrientationSubstitution:
    """The namespace substitutes the bridge's default_orientation when
    the user supplied neither ``orientation=`` nor ``vecxz=``, but only
    for 3D models where vecxz is meaningful."""

    def test_3d_neither_passed_inherits_bridge_default(self) -> None:
        """3D model, no kwargs: transform inherits Cartesian() Z-up."""
        ops = _stub_bridge()
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.PDelta()
        assert isinstance(t.orientation, Cartesian)
        assert t.vecxz is None

    def test_3d_user_orientation_overrides_default(self) -> None:
        """3D model, user passes orientation: user's value wins."""
        ops = _stub_bridge()
        ops.model(ndm=3, ndf=6)
        custom = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
        t = ops.geomTransf.PDelta(orientation=custom)
        assert t.orientation is custom

    def test_3d_user_vecxz_skips_substitution(self) -> None:
        """3D model, user passes vecxz: orientation stays None."""
        ops = _stub_bridge()
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.PDelta(vecxz=(0.0, 0.0, 1.0))
        assert t.orientation is None
        assert t.vecxz == (0.0, 0.0, 1.0)

    def test_2d_no_substitution(self) -> None:
        """2D model: no orientation substitution (vecxz omitted at emit)."""
        ops = _stub_bridge()
        ops.model(ndm=2, ndf=3)
        t = ops.geomTransf.PDelta()
        assert t.orientation is None
        assert t.vecxz is None

    def test_ndm_not_yet_set_no_substitution(self) -> None:
        """Transform created before model(): no substitution.

        Legacy test paths construct transforms before ``ndm`` is set.
        We keep the existing behavior (neither field populated) rather
        than guessing 3D and silently injecting a Z-up default.
        """
        ops = _stub_bridge()  # ndm not set
        t = ops.geomTransf.PDelta()
        assert t.orientation is None
        assert t.vecxz is None

    def test_explicit_none_default_disables_auto_substitution(self) -> None:
        """`default_orientation=None` at ctor turns off auto-substitution."""
        ops = apeSees(
            cast("object", MagicMock(name="FEMData")),  # type: ignore[arg-type]
            default_orientation=None,
        )
        ops.model(ndm=3, ndf=6)
        t = ops.geomTransf.PDelta()
        assert t.orientation is None
        assert t.vecxz is None

    def test_custom_default_orientation_inherited(self) -> None:
        """Custom orientation set at ctor flows to every untouched transform."""
        custom = Cartesian(reference_axis=(0, 1, 0))  # Y-up
        ops = apeSees(
            cast("object", MagicMock(name="FEMData")),  # type: ignore[arg-type]
            default_orientation=custom,
        )
        ops.model(ndm=3, ndf=6)
        t1 = ops.geomTransf.Linear()
        t2 = ops.geomTransf.PDelta()
        t3 = ops.geomTransf.Corotational()
        assert t1.orientation is custom
        assert t2.orientation is custom
        assert t3.orientation is custom
