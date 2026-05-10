"""Unit tests for the ``apeGmsh.opensees.element.solid`` primitives.

Each Phase 2δ class exercises:

  * Construction with a typed :class:`NDMaterial`.
  * Validation (where applicable — thickness, plane_type, rho, etc.).
  * ``_emit`` records the correct ``Emitter.element`` call when the
    bridge has installed both the tag resolver and the per-element
    node tags via :func:`set_tag_resolver` / :func:`set_element_nodes`.
  * ``_emit`` without an element-nodes context raises ``RuntimeError``.
  * Wrong-cardinality node tags raise ``ValueError``.
  * ``dependencies()`` returns the material reference.
  * ``__repr__`` includes the Python class name.
  * Bridge-namespace integration: ``ops.element.<Type>(...)`` returns
    the typed instance and the bridge allocates a tag.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.tag_resolution import (
    set_element_nodes,
    set_tag_resolver,
)
from apeGmsh.opensees._internal.types import Primitive
from apeGmsh.opensees.element.solid import (
    FourNodeQuad,
    FourNodeTetrahedron,
    TenNodeTetrahedron,
    Tri31,
    stdBrick,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_material() -> ElasticIsotropic:
    """Return a known-good :class:`ElasticIsotropic` instance."""
    return ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)


def _resolver_for(material: object, tag: int) -> object:
    """Build a tag resolver mapping ``id(material) -> tag``."""
    def _resolve(prim: Primitive) -> int:
        if id(prim) == id(material):
            return tag
        raise KeyError(f"unexpected primitive {prim!r}")
    return _resolve


def _emit_with(
    elem: Primitive,
    *,
    tag: int,
    nodes: tuple[int, ...],
    mat_tag: int,
    material: object,
) -> RecordingEmitter:
    """Run ``elem._emit`` against a fresh RecordingEmitter with both
    the tag resolver and the element-node context installed."""
    e = RecordingEmitter()
    set_tag_resolver(e, _resolver_for(material, mat_tag))
    set_element_nodes(e, nodes)
    elem._emit(e, tag=tag)  # type: ignore[attr-defined]
    return e


def _stub_bridge() -> apeSees:
    """Build an ``apeSees`` over a stub FEM. Used by namespace tests."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


# ===========================================================================
# FourNodeTetrahedron
# ===========================================================================

class TestFourNodeTetrahedron:
    def test_construction(self) -> None:
        m = _make_material()
        e = FourNodeTetrahedron(pg="Body", material=m)
        assert e.pg == "Body"
        assert e.material is m
        assert e.body_force is None

    def test_construction_with_body_force(self) -> None:
        m = _make_material()
        e = FourNodeTetrahedron(
            pg="Body", material=m, body_force=(0.0, 0.0, -9.81),
        )
        assert e.body_force == (0.0, 0.0, -9.81)

    def test_dependencies_returns_material(self) -> None:
        m = _make_material()
        e = FourNodeTetrahedron(pg="Body", material=m)
        assert e.dependencies() == (m,)

    def test_repr_includes_type_token(self) -> None:
        m = _make_material()
        e = FourNodeTetrahedron(pg="Body", material=m)
        assert "FourNodeTetrahedron" in repr(e)

    def test_emit_without_body_force(self) -> None:
        m = _make_material()
        elem = FourNodeTetrahedron(pg="Body", material=m)
        rec = _emit_with(
            elem, tag=7, nodes=(101, 102, 103, 104),
            mat_tag=2, material=m,
        )
        assert rec.calls == [
            (
                "element",
                ("FourNodeTetrahedron", 7, 101, 102, 103, 104, 2),
                {},
            )
        ]

    def test_emit_with_body_force(self) -> None:
        m = _make_material()
        elem = FourNodeTetrahedron(
            pg="Body", material=m, body_force=(0.0, 0.0, -9.81),
        )
        rec = _emit_with(
            elem, tag=11, nodes=(1, 2, 3, 4),
            mat_tag=5, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "FourNodeTetrahedron", 11,
                    1, 2, 3, 4,
                    5, 0.0, 0.0, -9.81,
                ),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        m = _make_material()
        elem = FourNodeTetrahedron(pg="Body", material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        with pytest.raises(RuntimeError, match="element-nodes"):
            elem._emit(e, tag=1)

    @pytest.mark.parametrize("bad_count", [0, 1, 3, 5, 8, 10])
    def test_emit_with_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = FourNodeTetrahedron(pg="Body", material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="expected 4 node tags"):
            elem._emit(e, tag=1)

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        m = _make_material()
        e = FourNodeTetrahedron(pg="Body", material=m)
        with pytest.raises(FrozenInstanceError):
            e.pg = "Other"  # type: ignore[misc]


# ===========================================================================
# TenNodeTetrahedron
# ===========================================================================

class TestTenNodeTetrahedron:
    def test_construction(self) -> None:
        m = _make_material()
        e = TenNodeTetrahedron(pg="Body", material=m)
        assert e.pg == "Body"
        assert e.material is m

    def test_dependencies_returns_material(self) -> None:
        m = _make_material()
        e = TenNodeTetrahedron(pg="Body", material=m)
        assert e.dependencies() == (m,)

    def test_repr_includes_type_token(self) -> None:
        m = _make_material()
        e = TenNodeTetrahedron(pg="Body", material=m)
        assert "TenNodeTetrahedron" in repr(e)

    def test_emit_without_body_force(self) -> None:
        m = _make_material()
        elem = TenNodeTetrahedron(pg="Body", material=m)
        nodes = tuple(range(101, 111))  # 10 tags
        rec = _emit_with(elem, tag=2, nodes=nodes, mat_tag=4, material=m)
        assert rec.calls == [
            ("element", ("TenNodeTetrahedron", 2, *nodes, 4), {})
        ]

    def test_emit_with_body_force(self) -> None:
        m = _make_material()
        elem = TenNodeTetrahedron(
            pg="Body", material=m, body_force=(1.0, 2.0, 3.0),
        )
        nodes = tuple(range(1, 11))
        rec = _emit_with(elem, tag=9, nodes=nodes, mat_tag=8, material=m)
        assert rec.calls == [
            (
                "element",
                ("TenNodeTetrahedron", 9, *nodes, 8, 1.0, 2.0, 3.0),
                {},
            )
        ]

    @pytest.mark.parametrize("bad_count", [0, 4, 8, 9, 11])
    def test_emit_with_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = TenNodeTetrahedron(pg="Body", material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="expected 10 node tags"):
            elem._emit(e, tag=1)


# ===========================================================================
# stdBrick
# ===========================================================================

class TestStdBrick:
    def test_construction(self) -> None:
        m = _make_material()
        e = stdBrick(pg="Body", material=m)
        assert e.pg == "Body"
        assert e.material is m

    def test_dependencies_returns_material(self) -> None:
        m = _make_material()
        e = stdBrick(pg="Body", material=m)
        assert e.dependencies() == (m,)

    def test_repr_includes_type_token(self) -> None:
        m = _make_material()
        e = stdBrick(pg="Body", material=m)
        assert "stdBrick" in repr(e)

    def test_emit_without_body_force(self) -> None:
        m = _make_material()
        elem = stdBrick(pg="Body", material=m)
        nodes = (1, 2, 3, 4, 5, 6, 7, 8)
        rec = _emit_with(elem, tag=3, nodes=nodes, mat_tag=2, material=m)
        assert rec.calls == [
            ("element", ("stdBrick", 3, *nodes, 2), {})
        ]

    def test_emit_with_body_force(self) -> None:
        m = _make_material()
        elem = stdBrick(
            pg="Body", material=m, body_force=(0.0, 0.0, -9.81),
        )
        nodes = (101, 102, 103, 104, 105, 106, 107, 108)
        rec = _emit_with(elem, tag=4, nodes=nodes, mat_tag=6, material=m)
        assert rec.calls == [
            (
                "element",
                ("stdBrick", 4, *nodes, 6, 0.0, 0.0, -9.81),
                {},
            )
        ]

    @pytest.mark.parametrize("bad_count", [0, 4, 7, 9, 10])
    def test_emit_with_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = stdBrick(pg="Body", material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="expected 8 node tags"):
            elem._emit(e, tag=1)


# ===========================================================================
# FourNodeQuad
# ===========================================================================

class TestFourNodeQuad:
    def test_construction_minimal(self) -> None:
        m = _make_material()
        e = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        assert e.pg == "Plate"
        assert e.thickness == 0.1
        assert e.material is m
        assert e.plane_type == "PlaneStrain"
        assert e.pressure is None
        assert e.rho is None
        assert e.body_force is None

    def test_construction_with_optional_tail(self) -> None:
        m = _make_material()
        e = FourNodeQuad(
            pg="Plate", thickness=0.2, material=m,
            plane_type="PlaneStress",
            pressure=1.5e3, rho=2200.0, body_force=(0.0, -9.81),
        )
        assert e.plane_type == "PlaneStress"
        assert e.pressure == 1.5e3
        assert e.rho == 2200.0
        assert e.body_force == (0.0, -9.81)

    def test_validation_rejects_non_positive_thickness(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="thickness must be > 0"):
            FourNodeQuad(pg="Plate", thickness=0.0, material=m)

    def test_validation_rejects_negative_thickness(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="thickness must be > 0"):
            FourNodeQuad(pg="Plate", thickness=-0.1, material=m)

    def test_validation_rejects_invalid_plane_type(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="plane_type must be one of"):
            FourNodeQuad(
                pg="Plate", thickness=0.1, material=m,
                plane_type="Plane3D",
            )

    def test_validation_rejects_negative_rho(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="rho must be >= 0"):
            FourNodeQuad(
                pg="Plate", thickness=0.1, material=m, rho=-1.0,
            )

    def test_dependencies_returns_material(self) -> None:
        m = _make_material()
        e = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        assert e.dependencies() == (m,)

    def test_repr_includes_type_token(self) -> None:
        m = _make_material()
        e = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        assert "FourNodeQuad" in repr(e)

    def test_emit_minimal_uses_quad_token(self) -> None:
        """Type token is the lowercase ``"quad"`` (not ``"FourNodeQuad"``)."""
        m = _make_material()
        elem = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        nodes = (1, 2, 3, 4)
        rec = _emit_with(elem, tag=5, nodes=nodes, mat_tag=2, material=m)
        assert rec.calls == [
            (
                "element",
                ("quad", 5, 1, 2, 3, 4, 0.1, "PlaneStrain", 2),
                {},
            )
        ]

    def test_emit_with_full_optional_tail(self) -> None:
        m = _make_material()
        elem = FourNodeQuad(
            pg="Plate", thickness=0.2, material=m,
            plane_type="PlaneStress",
            pressure=1.5e3, rho=2200.0, body_force=(0.0, -9.81),
        )
        nodes = (1, 2, 3, 4)
        rec = _emit_with(elem, tag=10, nodes=nodes, mat_tag=3, material=m)
        assert rec.calls == [
            (
                "element",
                (
                    "quad", 10, 1, 2, 3, 4,
                    0.2, "PlaneStress", 3,
                    1.5e3, 2200.0, 0.0, -9.81,
                ),
                {},
            )
        ]

    def test_emit_with_pressure_only(self) -> None:
        m = _make_material()
        elem = FourNodeQuad(
            pg="Plate", thickness=0.1, material=m, pressure=500.0,
        )
        rec = _emit_with(
            elem, tag=1, nodes=(1, 2, 3, 4), mat_tag=2, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "quad", 1, 1, 2, 3, 4,
                    0.1, "PlaneStrain", 2,
                    500.0,
                ),
                {},
            )
        ]

    def test_emit_with_pressure_and_rho(self) -> None:
        m = _make_material()
        elem = FourNodeQuad(
            pg="Plate", thickness=0.1, material=m,
            pressure=500.0, rho=2200.0,
        )
        rec = _emit_with(
            elem, tag=1, nodes=(1, 2, 3, 4), mat_tag=2, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "quad", 1, 1, 2, 3, 4,
                    0.1, "PlaneStrain", 2,
                    500.0, 2200.0,
                ),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        m = _make_material()
        elem = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        with pytest.raises(RuntimeError, match="element-nodes"):
            elem._emit(e, tag=1)

    @pytest.mark.parametrize("bad_count", [0, 1, 3, 5, 8])
    def test_emit_with_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = FourNodeQuad(pg="Plate", thickness=0.1, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="expected 4 node tags"):
            elem._emit(e, tag=1)


# ===========================================================================
# Tri31
# ===========================================================================

class TestTri31:
    def test_construction_minimal(self) -> None:
        m = _make_material()
        e = Tri31(pg="Plate", thickness=0.05, material=m)
        assert e.pg == "Plate"
        assert e.thickness == 0.05
        assert e.material is m
        assert e.plane_type == "PlaneStrain"

    def test_validation_rejects_non_positive_thickness(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="thickness must be > 0"):
            Tri31(pg="Plate", thickness=0.0, material=m)

    def test_validation_rejects_invalid_plane_type(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="plane_type must be one of"):
            Tri31(
                pg="Plate", thickness=0.1, material=m, plane_type="bogus",
            )

    def test_validation_rejects_negative_rho(self) -> None:
        m = _make_material()
        with pytest.raises(ValueError, match="rho must be >= 0"):
            Tri31(pg="Plate", thickness=0.1, material=m, rho=-2.0)

    def test_dependencies_returns_material(self) -> None:
        m = _make_material()
        e = Tri31(pg="Plate", thickness=0.05, material=m)
        assert e.dependencies() == (m,)

    def test_repr_includes_type_token(self) -> None:
        m = _make_material()
        e = Tri31(pg="Plate", thickness=0.05, material=m)
        assert "Tri31" in repr(e)

    def test_emit_minimal_uses_tri31_token(self) -> None:
        """Type token is lowercase ``"tri31"``."""
        m = _make_material()
        elem = Tri31(pg="Plate", thickness=0.05, material=m)
        rec = _emit_with(
            elem, tag=2, nodes=(11, 12, 13), mat_tag=4, material=m,
        )
        assert rec.calls == [
            (
                "element",
                ("tri31", 2, 11, 12, 13, 0.05, "PlaneStrain", 4),
                {},
            )
        ]

    def test_emit_with_full_optional_tail(self) -> None:
        m = _make_material()
        elem = Tri31(
            pg="Plate", thickness=0.05, material=m,
            plane_type="PlaneStress",
            pressure=200.0, rho=2200.0, body_force=(0.0, -9.81),
        )
        rec = _emit_with(
            elem, tag=20, nodes=(1, 2, 3), mat_tag=3, material=m,
        )
        assert rec.calls == [
            (
                "element",
                (
                    "tri31", 20, 1, 2, 3,
                    0.05, "PlaneStress", 3,
                    200.0, 2200.0, 0.0, -9.81,
                ),
                {},
            )
        ]

    def test_emit_without_element_nodes_raises(self) -> None:
        m = _make_material()
        elem = Tri31(pg="Plate", thickness=0.05, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        with pytest.raises(RuntimeError, match="element-nodes"):
            elem._emit(e, tag=1)

    @pytest.mark.parametrize("bad_count", [0, 1, 2, 4, 5])
    def test_emit_with_wrong_node_count_raises(self, bad_count: int) -> None:
        m = _make_material()
        elem = Tri31(pg="Plate", thickness=0.05, material=m)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_for(m, 1))
        set_element_nodes(e, tuple(range(1, bad_count + 1)))
        with pytest.raises(ValueError, match="expected 3 node tags"):
            elem._emit(e, tag=1)


# ===========================================================================
# Cross-cutting: namespace integration
# ===========================================================================

class TestSolidElementNamespace:
    """Verify ``ops.element.<Type>(...)`` registers and tags correctly."""

    def test_FourNodeTetrahedron_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
        e = ops.element.FourNodeTetrahedron(pg="Body", material=m)
        assert isinstance(e, FourNodeTetrahedron)
        assert ops.tag_for(e) == 1

    def test_TenNodeTetrahedron_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        e = ops.element.TenNodeTetrahedron(pg="Body", material=m)
        assert isinstance(e, TenNodeTetrahedron)
        assert ops.tag_for(e) == 1

    def test_stdBrick_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        e = ops.element.stdBrick(
            pg="Body", material=m, body_force=(0.0, 0.0, -9.81),
        )
        assert isinstance(e, stdBrick)
        assert ops.tag_for(e) == 1

    def test_FourNodeQuad_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        e = ops.element.FourNodeQuad(
            pg="Plate", thickness=0.1, material=m,
            plane_type="PlaneStress",
        )
        assert isinstance(e, FourNodeQuad)
        assert e.plane_type == "PlaneStress"
        assert ops.tag_for(e) == 1

    def test_Tri31_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        e = ops.element.Tri31(pg="Plate", thickness=0.05, material=m)
        assert isinstance(e, Tri31)
        assert ops.tag_for(e) == 1

    def test_distinct_elements_get_distinct_tags(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        e1 = ops.element.FourNodeTetrahedron(pg="A", material=m)
        e2 = ops.element.FourNodeTetrahedron(pg="B", material=m)
        e3 = ops.element.stdBrick(pg="C", material=m)
        # Element tags are independent of nDMaterial tags (separate
        # TagAllocator kinds).
        assert ops.tag_for(e1) == 1
        assert ops.tag_for(e2) == 2
        assert ops.tag_for(e3) == 3
