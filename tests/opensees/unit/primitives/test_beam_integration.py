"""Unit tests for the :mod:`apeGmsh.opensees.integration` primitives.

Each rule is exercised for construction, validation, dependencies,
and ``_emit`` output via :class:`RecordingEmitter`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees._internal.types import Primitive, Section
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.integration import (
    HingeEndpoint,
    HingeMidpoint,
    HingeRadau,
    HingeRadauTwo,
    Legendre,
    Lobatto,
    NewtonCotes,
    Radau,
    Trapezoidal,
)


# ---------------------------------------------------------------------------
# Test-local section stub.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, kw_only=True, slots=True)
class _FakeSection(Section):
    name: str

    def _emit(self, emitter: Emitter, tag: int) -> None:  # pragma: no cover
        emitter.section("Fake", tag, self.name)

    def dependencies(self) -> tuple[Primitive, ...]:
        return ()


def _resolver(tags: dict[int, int]) -> object:
    def _r(prim: Primitive) -> int:
        return tags[id(prim)]
    return _r


# ===========================================================================
# Uniform-section rules
# ===========================================================================

class TestLobatto:
    def test_construct_and_emit(self) -> None:
        sec = _FakeSection(name="x")
        rule = Lobatto(section=sec, n_ip=5)
        assert rule.section is sec
        assert rule.n_ip == 5
        assert rule.dependencies() == (sec,)

        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec): 7}))
        rule._emit(rec, tag=1)
        assert rec.calls == [("beamIntegration", ("Lobatto", 1, 7, 5), {})]

    def test_n_ip_minimum_is_two(self) -> None:
        sec = _FakeSection(name="x")
        with pytest.raises(ValueError, match="n_ip must be >= 2"):
            Lobatto(section=sec, n_ip=1)


class TestLegendre:
    def test_construct_and_emit(self) -> None:
        sec = _FakeSection(name="x")
        rule = Legendre(section=sec, n_ip=3)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec): 4}))
        rule._emit(rec, tag=2)
        assert rec.calls == [("beamIntegration", ("Legendre", 2, 4, 3), {})]

    def test_n_ip_minimum_is_one(self) -> None:
        sec = _FakeSection(name="x")
        with pytest.raises(ValueError, match="n_ip must be >= 1"):
            Legendre(section=sec, n_ip=0)


class TestNewtonCotes:
    def test_construct_and_emit(self) -> None:
        sec = _FakeSection(name="x")
        rule = NewtonCotes(section=sec, n_ip=4)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec): 5}))
        rule._emit(rec, tag=3)
        assert rec.calls == [("beamIntegration", ("NewtonCotes", 3, 5, 4), {})]


class TestRadau:
    def test_construct_and_emit(self) -> None:
        sec = _FakeSection(name="x")
        rule = Radau(section=sec, n_ip=2)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec): 6}))
        rule._emit(rec, tag=4)
        assert rec.calls == [("beamIntegration", ("Radau", 4, 6, 2), {})]


class TestTrapezoidal:
    def test_construct_and_emit(self) -> None:
        sec = _FakeSection(name="x")
        rule = Trapezoidal(section=sec, n_ip=3)
        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec): 8}))
        rule._emit(rec, tag=5)
        assert rec.calls == [("beamIntegration", ("Trapezoidal", 5, 8, 3), {})]


# ===========================================================================
# Hinge rules — same parameter shape, different type token
# ===========================================================================

_HINGE_TYPES = [
    (HingeRadau,    "HingeRadau"),
    (HingeRadauTwo, "HingeRadauTwo"),
    (HingeMidpoint, "HingeMidpoint"),
    (HingeEndpoint, "HingeEndpoint"),
]


@pytest.mark.parametrize(("cls", "token"), _HINGE_TYPES)
class TestHingeRules:
    def test_construct_and_emit(self, cls: type, token: str) -> None:
        sec_i = _FakeSection(name="plastic_i")
        sec_j = _FakeSection(name="plastic_j")
        sec_e = _FakeSection(name="elastic")
        rule = cls(
            section_i=sec_i, lp_i=0.1,
            section_j=sec_j, lp_j=0.2,
            section_interior=sec_e,
        )
        assert rule.dependencies() == (sec_i, sec_j, sec_e)

        rec = RecordingEmitter()
        set_tag_resolver(rec, _resolver({id(sec_i): 1, id(sec_j): 2, id(sec_e): 3}))
        rule._emit(rec, tag=42)
        # Layout: tag secI lpI secJ lpJ secE
        assert rec.calls == [
            ("beamIntegration", (token, 42, 1, 0.1, 2, 0.2, 3), {}),
        ]

    def test_lp_positive(self, cls: type, token: str) -> None:
        sec = _FakeSection(name="x")
        with pytest.raises(ValueError, match="lp_i must be > 0"):
            cls(
                section_i=sec, lp_i=0.0,
                section_j=sec, lp_j=0.1,
                section_interior=sec,
            )
        with pytest.raises(ValueError, match="lp_j must be > 0"):
            cls(
                section_i=sec, lp_i=0.1,
                section_j=sec, lp_j=-0.05,
                section_interior=sec,
            )

    def test_shared_section_deduped_in_dependencies(
        self, cls: type, token: str,
    ) -> None:
        """When all three sections are the same instance, dependencies()
        returns a single primitive (deduped by id)."""
        sec = _FakeSection(name="shared")
        rule = cls(
            section_i=sec, lp_i=0.1,
            section_j=sec, lp_j=0.1,
            section_interior=sec,
        )
        assert rule.dependencies() == (sec,)
