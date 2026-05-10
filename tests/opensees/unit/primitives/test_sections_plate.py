"""Unit tests for plate / shell sections.

Covers :class:`ElasticMembranePlateSection`, :class:`LayeredShell`,
and :class:`LayeredShellFiberSection`. The layered sections compose
nDMaterials and so reach for the
:func:`~apeGmsh.opensees.section._tag_resolver.set_tag_resolver`
contract during ``_emit`` (the open coordinator question — see
:mod:`section._tag_resolver`).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.types import NDMaterial, Primitive
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.section._tag_resolver import set_tag_resolver
from apeGmsh.opensees.section.plate import (
    ElasticMembranePlateSection,
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
)


# ---------------------------------------------------------------------------
# Test-local nDMaterial — mirrors the pattern in test_apesees_class.py.
# A concrete subclass of NDMaterial whose _emit / dependencies are
# trivial; we only need it as a typed reference to attach to layers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeND(NDMaterial):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:
        emitter.nDMaterial("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _resolver_from(tags: dict[int, int]) -> object:
    """Return a callable that maps Primitive -> tag via id-keyed map."""
    def _resolve(prim: Primitive) -> int:
        return tags[id(prim)]
    return _resolve


# ===========================================================================
# ElasticMembranePlateSection
# ===========================================================================

class TestElasticMembranePlateSectionConstruction:
    def test_construct_minimum(self) -> None:
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
        assert s.E == 30e9
        assert s.nu == 0.2
        assert s.h == 0.2
        assert s.rho == 0.0  # default

    def test_construct_with_rho(self) -> None:
        s = ElasticMembranePlateSection(
            E=30e9, nu=0.2, h=0.2, rho=2400.0,
        )
        assert s.rho == 2400.0


class TestElasticMembranePlateSectionValidation:
    @pytest.mark.parametrize("E", [0.0, -1.0])
    def test_E_positive(self, E: float) -> None:
        with pytest.raises(ValueError, match="E must be > 0"):
            ElasticMembranePlateSection(E=E, nu=0.2, h=0.2)

    @pytest.mark.parametrize("nu", [-0.1, 0.5, 0.6])
    def test_nu_in_range(self, nu: float) -> None:
        with pytest.raises(ValueError, match=r"nu must be in \[0, 0\.5\)"):
            ElasticMembranePlateSection(E=30e9, nu=nu, h=0.2)

    @pytest.mark.parametrize("h", [0.0, -0.1])
    def test_h_positive(self, h: float) -> None:
        with pytest.raises(ValueError, match="h must be > 0"):
            ElasticMembranePlateSection(E=30e9, nu=0.2, h=h)

    def test_rho_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="rho must be >= 0"):
            ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=-1.0)


class TestElasticMembranePlateSectionEmit:
    def test_emit_records_correct_call(self) -> None:
        s = ElasticMembranePlateSection(
            E=30e9, nu=0.2, h=0.2, rho=2400.0,
        )
        e = RecordingEmitter()
        s._emit(e, tag=5)
        assert e.calls == [
            (
                "section",
                ("ElasticMembranePlateSection", 5, 30e9, 0.2, 0.2, 2400.0),
                {},
            )
        ]

    def test_emit_default_rho_is_zero(self) -> None:
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
        e = RecordingEmitter()
        s._emit(e, tag=1)
        assert e.calls[0][1][-1] == 0.0


class TestElasticMembranePlateSectionMisc:
    def test_dependencies_empty(self) -> None:
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
        assert s.dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2)
        assert "ElasticMembranePlateSection" in repr(s)


# ===========================================================================
# ShellLayer (value object)
# ===========================================================================

class TestShellLayer:
    def test_construct(self) -> None:
        m = _FakeND(name="layerA")
        layer = ShellLayer(material=m, thickness=0.05)
        assert layer.material is m
        assert layer.thickness == 0.05

    @pytest.mark.parametrize("t", [0.0, -1.0])
    def test_thickness_positive(self, t: float) -> None:
        m = _FakeND(name="x")
        with pytest.raises(ValueError, match="thickness must be > 0"):
            ShellLayer(material=m, thickness=t)


# ===========================================================================
# LayeredShell
# ===========================================================================

class TestLayeredShellConstruction:
    def test_construct_with_one_layer(self) -> None:
        m = _FakeND(name="A")
        s = LayeredShell(layers=(ShellLayer(material=m, thickness=0.1),))
        assert len(s.layers) == 1

    def test_construct_with_multiple_layers(self) -> None:
        a, b = _FakeND(name="A"), _FakeND(name="B")
        s = LayeredShell(
            layers=(
                ShellLayer(material=a, thickness=0.05),
                ShellLayer(material=b, thickness=0.10),
                ShellLayer(material=a, thickness=0.05),
            )
        )
        assert len(s.layers) == 3

    def test_no_layers_rejected(self) -> None:
        with pytest.raises(
            ValueError, match="at least one ShellLayer is required"
        ):
            LayeredShell(layers=())


class TestLayeredShellDependencies:
    def test_dependencies_dedupes_in_order(self) -> None:
        a, b = _FakeND(name="A"), _FakeND(name="B")
        s = LayeredShell(
            layers=(
                ShellLayer(material=a, thickness=0.05),
                ShellLayer(material=b, thickness=0.10),
                ShellLayer(material=a, thickness=0.05),  # duplicate
            )
        )
        assert s.dependencies() == (a, b)


class TestLayeredShellEmit:
    def test_emit_records_correct_call(self) -> None:
        a, b = _FakeND(name="A"), _FakeND(name="B")
        s = LayeredShell(
            layers=(
                ShellLayer(material=a, thickness=0.05),
                ShellLayer(material=b, thickness=0.10),
            )
        )
        e = RecordingEmitter()
        # Composite section needs a tag resolver attached.
        set_tag_resolver(e, _resolver_from({id(a): 11, id(b): 22}))
        s._emit(e, tag=7)
        assert e.calls == [
            (
                "section",
                ("LayeredShell", 7,
                 2,                 # nLayers
                 11, 0.05,          # layer 1: matTag, thickness
                 22, 0.10),         # layer 2
                {},
            )
        ]

    def test_emit_without_resolver_raises(self) -> None:
        m = _FakeND(name="A")
        s = LayeredShell(
            layers=(ShellLayer(material=m, thickness=0.1),)
        )
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            s._emit(e, tag=1)


# ===========================================================================
# LayeredShellFiberSection
# ===========================================================================

class TestLayeredShellFiberSectionEmit:
    def test_emit_uses_correct_type_token(self) -> None:
        a = _FakeND(name="A")
        s = LayeredShellFiberSection(
            layers=(ShellLayer(material=a, thickness=0.1),)
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(a): 99}))
        s._emit(e, tag=4)
        assert e.calls == [
            (
                "section",
                ("LayeredShellFiberSection", 4, 1, 99, 0.1),
                {},
            )
        ]

    def test_dependencies_returns_materials(self) -> None:
        a, b = _FakeND(name="A"), _FakeND(name="B")
        s = LayeredShellFiberSection(
            layers=(
                ShellLayer(material=a, thickness=0.1),
                ShellLayer(material=b, thickness=0.2),
            )
        )
        assert s.dependencies() == (a, b)

    def test_empty_layers_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="at least one ShellLayer is required",
        ):
            LayeredShellFiberSection(layers=())
