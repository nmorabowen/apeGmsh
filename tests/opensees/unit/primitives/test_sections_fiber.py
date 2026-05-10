"""Unit tests for the Fiber section + its value objects.

Covers :class:`apeGmsh.opensees.section.fiber.Fiber`,
:class:`RectPatch`, :class:`StraightLayer`, :class:`FiberPoint`.

The Fiber section composes :class:`UniaxialMaterial` references; its
``_emit`` requires a tag resolver attached to the emitter (the open
coordinator question — see :mod:`section._tag_resolver`). Tests use
``set_tag_resolver`` directly to provide a manual mapping.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.types import Primitive, UniaxialMaterial
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.section._tag_resolver import set_tag_resolver
from apeGmsh.opensees.section.fiber import (
    Fiber,
    FiberPoint,
    RectPatch,
    StraightLayer,
)


# ---------------------------------------------------------------------------
# Test-local UniaxialMaterial — same shape as the foundation tests.
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
# RectPatch
# ===========================================================================

class TestRectPatch:
    def test_construct(self) -> None:
        m = _FakeMat(name="core")
        p = RectPatch(
            material=m, ny=10, nz=10,
            yI=-0.5, zI=-0.5, yJ=0.5, zJ=0.5,
        )
        assert p.material is m
        assert p.ny == 10 and p.nz == 10

    def test_negative_subdivisions_rejected(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="ny and nz must be > 0"):
            RectPatch(
                material=m, ny=0, nz=10,
                yI=0, zI=0, yJ=1, zJ=1,
            )


# ===========================================================================
# StraightLayer
# ===========================================================================

class TestStraightLayer:
    def test_construct(self) -> None:
        m = _FakeMat(name="rebar")
        layer = StraightLayer(
            material=m, n_bars=4, area=0.000314,
            yI=-0.4, zI=-0.4, yJ=0.4, zJ=-0.4,
        )
        assert layer.n_bars == 4
        assert layer.area == 0.000314

    def test_n_bars_at_least_one(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="n_bars must be >= 1"):
            StraightLayer(
                material=m, n_bars=0, area=0.001,
                yI=0, zI=0, yJ=1, zJ=0,
            )

    def test_area_positive(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="area must be > 0"):
            StraightLayer(
                material=m, n_bars=1, area=0.0,
                yI=0, zI=0, yJ=0, zJ=0,
            )


# ===========================================================================
# FiberPoint
# ===========================================================================

class TestFiberPoint:
    def test_construct(self) -> None:
        m = _FakeMat(name="x")
        f = FiberPoint(material=m, y=0.1, z=0.2, area=1e-4)
        assert (f.y, f.z, f.area) == (0.1, 0.2, 1e-4)

    def test_area_positive(self) -> None:
        m = _FakeMat(name="x")
        with pytest.raises(ValueError, match="area must be > 0"):
            FiberPoint(material=m, y=0, z=0, area=-1e-4)


# ===========================================================================
# Fiber section
# ===========================================================================

class TestFiberConstruction:
    def test_construct_with_patches(self) -> None:
        m = _FakeMat(name="core")
        p = RectPatch(
            material=m, ny=10, nz=10,
            yI=-0.5, zI=-0.5, yJ=0.5, zJ=0.5,
        )
        s = Fiber(patches=(p,))
        assert s.patches == (p,)
        assert s.fibers == ()
        assert s.layers == ()
        assert s.GJ is None

    def test_construct_with_GJ(self) -> None:
        m = _FakeMat(name="core")
        p = RectPatch(
            material=m, ny=2, nz=2,
            yI=0, zI=0, yJ=1, zJ=1,
        )
        s = Fiber(patches=(p,), GJ=1e9)
        assert s.GJ == 1e9

    def test_empty_section_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="at least one of patches / fibers / layers",
        ):
            Fiber()

    def test_negative_GJ_rejected(self) -> None:
        m = _FakeMat(name="x")
        p = RectPatch(
            material=m, ny=1, nz=1,
            yI=0, zI=0, yJ=1, zJ=1,
        )
        with pytest.raises(ValueError, match="GJ must be > 0"):
            Fiber(patches=(p,), GJ=-1.0)


class TestFiberDependencies:
    def test_dependencies_dedupes_across_kinds(self) -> None:
        a = _FakeMat(name="A")
        b = _FakeMat(name="B")
        c = _FakeMat(name="C")
        # a appears in patches; b in layers; c in fibers; b appears
        # in patches too — dedup in iteration order should produce
        # (a, b, c).
        s = Fiber(
            patches=(
                RectPatch(
                    material=a, ny=1, nz=1,
                    yI=0, zI=0, yJ=1, zJ=1,
                ),
                RectPatch(
                    material=b, ny=1, nz=1,
                    yI=2, zI=2, yJ=3, zJ=3,
                ),
            ),
            layers=(
                StraightLayer(
                    material=b, n_bars=2, area=1e-4,
                    yI=0, zI=0, yJ=1, zJ=0,
                ),
            ),
            fibers=(
                FiberPoint(material=c, y=0, z=0, area=1e-4),
                FiberPoint(material=a, y=0.5, z=0, area=1e-4),
            ),
        )
        deps = s.dependencies()
        assert deps == (a, b, c)


class TestFiberEmit:
    def test_emit_minimal_patch(self) -> None:
        a = _FakeMat(name="A")
        s = Fiber(
            patches=(
                RectPatch(
                    material=a, ny=10, nz=10,
                    yI=-0.5, zI=-0.5, yJ=0.5, zJ=0.5,
                ),
            )
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(a): 1}))
        s._emit(e, tag=3)
        names = [c[0] for c in e.calls]
        assert names == ["section_open", "patch", "section_close"]
        assert e.calls[0] == ("section_open", ("Fiber", 3), {})
        assert e.calls[1] == (
            "patch",
            ("rect", 1, 10, 10, -0.5, -0.5, 0.5, 0.5),
            {},
        )
        assert e.calls[2] == ("section_close", (), {})

    def test_emit_with_GJ(self) -> None:
        a = _FakeMat(name="A")
        s = Fiber(
            patches=(
                RectPatch(
                    material=a, ny=1, nz=1,
                    yI=0, zI=0, yJ=1, zJ=1,
                ),
            ),
            GJ=2e8,
        )
        e = RecordingEmitter()
        set_tag_resolver(e, _resolver_from({id(a): 5}))
        s._emit(e, tag=2)
        assert e.calls[0] == (
            "section_open", ("Fiber", 2, "-GJ", 2e8), {}
        )

    def test_emit_full_with_patches_layers_fibers(self) -> None:
        a = _FakeMat(name="A")
        b = _FakeMat(name="B")
        d = _FakeMat(name="D")
        s = Fiber(
            patches=(
                RectPatch(
                    material=a, ny=10, nz=10,
                    yI=-0.5, zI=-0.5, yJ=0.5, zJ=0.5,
                ),
            ),
            layers=(
                StraightLayer(
                    material=b, n_bars=4, area=3.14e-4,
                    yI=-0.4, zI=-0.4, yJ=0.4, zJ=-0.4,
                ),
            ),
            fibers=(
                FiberPoint(material=d, y=0, z=0, area=1e-3),
            ),
        )
        e = RecordingEmitter()
        set_tag_resolver(
            e, _resolver_from({id(a): 1, id(b): 2, id(d): 3})
        )
        s._emit(e, tag=7)
        names = [c[0] for c in e.calls]
        assert names == [
            "section_open",
            "patch",
            "layer",
            "fiber",
            "section_close",
        ]
        assert e.calls[1] == (
            "patch",
            ("rect", 1, 10, 10, -0.5, -0.5, 0.5, 0.5),
            {},
        )
        assert e.calls[2] == (
            "layer",
            ("straight", 2, 4, 3.14e-4, -0.4, -0.4, 0.4, -0.4),
            {},
        )
        assert e.calls[3] == ("fiber", (0.0, 0.0, 1e-3, 3), {})

    def test_emit_without_resolver_raises(self) -> None:
        a = _FakeMat(name="A")
        s = Fiber(
            patches=(
                RectPatch(
                    material=a, ny=1, nz=1,
                    yI=0, zI=0, yJ=1, zJ=1,
                ),
            )
        )
        e = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            s._emit(e, tag=1)


class TestFiberMisc:
    def test_repr_includes_class_name(self) -> None:
        m = _FakeMat(name="x")
        s = Fiber(
            patches=(
                RectPatch(
                    material=m, ny=1, nz=1,
                    yI=0, zI=0, yJ=1, zJ=1,
                ),
            )
        )
        assert "Fiber" in repr(s)
