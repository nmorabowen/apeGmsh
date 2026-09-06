"""Unit tests for plate / shell sections.

Covers :class:`ElasticMembranePlateSection`, :class:`LayeredShell`,
:class:`LayeredShellFiberSection`, and the fork decorator
:class:`LadrunoShellModifier`. The layered sections compose
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
    SHELL_MODIFIER_FLAGS,
    ElasticMembranePlateSection,
    LadrunoShellModifier,
    LayeredShell,
    LayeredShellFiberSection,
    ShellLayer,
    ShellModifierNonlinearInnerWarning,
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

    def test_default_Ep_mod_is_not_emitted(self) -> None:
        # Existing decks must stay byte-identical.
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, rho=2400.0)
        e = RecordingEmitter()
        s._emit(e, tag=1)
        assert s.Ep_mod == 1.0
        assert e.calls[0][1] == (
            "ElasticMembranePlateSection", 1, 30e9, 0.2, 0.2, 2400.0,
        )

    def test_non_default_Ep_mod_emits_as_the_fifth_double(self) -> None:
        s = ElasticMembranePlateSection(
            E=30e9, nu=0.2, h=0.2, rho=2400.0, Ep_mod=0.25,
        )
        e = RecordingEmitter()
        s._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ElasticMembranePlateSection", 1, 30e9, 0.2, 0.2, 2400.0, 0.25,
        )

    def test_Ep_mod_emits_after_a_default_rho(self) -> None:
        # Ep_mod is positional, so rho must still occupy slot 4.
        s = ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.2, Ep_mod=0.5)
        e = RecordingEmitter()
        s._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ElasticMembranePlateSection", 1, 30e9, 0.2, 0.2, 0.0, 0.5,
        )

    def test_negative_Ep_mod_raises(self) -> None:
        with pytest.raises(ValueError, match="Ep_mod must be >= 0"):
            ElasticMembranePlateSection(
                E=30e9, nu=0.2, h=0.2, Ep_mod=-0.1,
            )


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


# ===========================================================================
# LadrunoShellModifier (fork decorator — Ladruno ADR 91)
# ===========================================================================

def _plate() -> ElasticMembranePlateSection:
    return ElasticMembranePlateSection(E=25e9, nu=0.2, h=0.3)


class TestLadrunoShellModifierConstruction:
    def test_all_flags_default_to_one(self) -> None:
        s = LadrunoShellModifier(inner=_plate())
        for name in SHELL_MODIFIER_FLAGS:
            assert getattr(s, name) == 1.0

    def test_flag_order_matches_the_fork_parse_order(self) -> None:
        # The fork reads the ETABS OAPI array order; the tenth entry
        # (weight) is deliberately absent (ADR 91 §5).
        assert SHELL_MODIFIER_FLAGS == (
            "f11", "f22", "f12", "m11", "m22", "m12", "v13", "v23", "mass",
        )

    def test_cracked_wall_recipe(self) -> None:
        s = LadrunoShellModifier(
            inner=_plate(), f11=0.35, f22=0.35, f12=0.35,
        )
        assert (s.f11, s.f22, s.f12) == (0.35, 0.35, 0.35)
        assert s.m11 == 1.0


class TestLadrunoShellModifierValidation:
    @pytest.mark.parametrize("name", SHELL_MODIFIER_FLAGS)
    def test_negative_modifier_raises(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"{name} must be >= 0.0"):
            LadrunoShellModifier(**{"inner": _plate(), name: -0.1})

    @pytest.mark.parametrize("name", SHELL_MODIFIER_FLAGS)
    def test_zero_modifier_is_accepted(self, name: str) -> None:
        # 0.0 is ETABS-legal; the fork accepts it and warns once per
        # `section` command that the response mode is singular.
        s = LadrunoShellModifier(**{"inner": _plate(), name: 0.0})
        assert getattr(s, name) == 0.0

    def test_non_section_inner_raises(self) -> None:
        with pytest.raises(TypeError, match="inner must be a Section"):
            LadrunoShellModifier(inner=_FakeND(name="A"))  # type: ignore[arg-type]


class TestLadrunoShellModifierDependencies:
    def test_dependencies_is_the_inner_section(self) -> None:
        inner = _plate()
        s = LadrunoShellModifier(inner=inner, f11=0.35)
        assert s.dependencies() == (inner,)


class TestLadrunoShellModifierEmit:
    def test_all_defaults_emit_bare_wrapper(self) -> None:
        inner = _plate()
        s = LadrunoShellModifier(inner=inner)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(inner): 3}))
        s._emit(e, tag=9)
        # Sparse emit: an all-defaults wrap carries no flags at all.
        assert e.calls == [
            ("section", ("LadrunoShellModifier", 9, 3), {}),
        ]

    def test_only_non_default_flags_are_emitted(self) -> None:
        inner = _plate()
        s = LadrunoShellModifier(
            inner=inner, f11=0.35, f22=0.35, f12=0.35, mass=1.0,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(inner): 3}))
        s._emit(e, tag=9)
        assert e.calls == [
            (
                "section",
                ("LadrunoShellModifier", 9, 3,
                 "-f11", 0.35, "-f22", 0.35, "-f12", 0.35),
                {},
            )
        ]

    def test_flags_emit_in_canonical_order(self) -> None:
        inner = _plate()
        # Constructed out of order — emit must still be canonical.
        s = LadrunoShellModifier(
            inner=inner, mass=0.9, v23=0.4, f22=0.2, m11=0.3,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(inner): 1}))
        s._emit(e, tag=2)
        assert e.calls[0][1] == (
            "LadrunoShellModifier", 2, 1,
            "-f22", 0.2, "-m11", 0.3, "-v23", 0.4, "-mass", 0.9,
        )

    def test_zero_flag_is_emitted(self) -> None:
        # 0.0 differs from the 1.0 default, so it must reach the deck —
        # a dropped 0.0 would silently restore full stiffness.
        inner = _plate()
        s = LadrunoShellModifier(inner=inner, m11=0.0)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(inner): 1}))
        s._emit(e, tag=2)
        assert e.calls[0][1] == ("LadrunoShellModifier", 2, 1, "-m11", 0.0)

    def test_emit_without_resolver_raises(self) -> None:
        s = LadrunoShellModifier(inner=_plate(), f11=0.35)
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            s._emit(e, tag=1)

    def test_wraps_a_layered_shell_but_warns(self) -> None:
        # The fork accepts any order-8 plate section, so emit still works
        # — but a path-dependent inner is not the supported case.
        a = _FakeND(name="A")
        inner = LayeredShell(
            layers=(ShellLayer(material=a, thickness=0.1),)
        )
        with pytest.warns(ShellModifierNonlinearInnerWarning):
            s = LadrunoShellModifier(inner=inner, f11=0.35)
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(inner): 7}))
        s._emit(e, tag=8)
        assert e.calls[0][1] == ("LadrunoShellModifier", 8, 7, "-f11", 0.35)
        assert s.dependencies() == (inner,)


class TestLadrunoShellModifierInnerIsElastic:
    """Only a linear-elastic plate is the supported inner section."""

    def test_elastic_inner_does_not_warn(self, recwarn) -> None:
        LadrunoShellModifier(inner=_plate(), f11=0.35)
        assert [
            w for w in recwarn
            if issubclass(w.category, ShellModifierNonlinearInnerWarning)
        ] == []

    @pytest.mark.parametrize("cls", [LayeredShell, LayeredShellFiberSection])
    def test_layered_inner_warns_and_names_the_type(self, cls) -> None:
        inner = cls(
            layers=(ShellLayer(material=_FakeND(name="A"), thickness=0.1),)
        )
        with pytest.warns(
            ShellModifierNonlinearInnerWarning, match=cls.__name__,
        ):
            LadrunoShellModifier(inner=inner, f11=0.35)

    def test_warning_explains_the_scaled_strain(self) -> None:
        inner = LayeredShell(
            layers=(ShellLayer(material=_FakeND(name="A"), thickness=0.1),)
        )
        with pytest.warns(
            ShellModifierNonlinearInnerWarning, match=r"S\*e",
        ):
            LadrunoShellModifier(inner=inner)

    def test_an_all_defaults_wrap_of_a_layered_shell_still_warns(self) -> None:
        # The wrapper is a no-op numerically, but the modelling mistake
        # is the wrapping itself, so silence here would be misleading.
        inner = LayeredShell(
            layers=(ShellLayer(material=_FakeND(name="A"), thickness=0.1),)
        )
        with pytest.warns(ShellModifierNonlinearInnerWarning):
            LadrunoShellModifier(inner=inner)
