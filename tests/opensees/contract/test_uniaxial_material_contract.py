"""Family contract gate for typed uniaxial materials (Phase 1A).

Every concrete subclass of ``UniaxialMaterial`` shipped by Phase 1A is
listed in ``ALL_UNIAXIAL`` and verified against the family contract:

  * inherits from :class:`UniaxialMaterial` (and therefore
    :class:`Primitive`),
  * is a frozen, kw-only dataclass,
  * implements ``_emit`` and emits a single
    ``("uniaxialMaterial", ...)`` call,
  * implements ``dependencies`` returning ``()`` (uniaxial leaves),
  * ``__repr__`` includes the class name.

Future uniaxial material slices append their classes to
``ALL_UNIAXIAL`` and add an entry to ``_MINIMAL_PARAMS`` so the smoke
emit covers them automatically.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import Primitive, UniaxialMaterial
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.uniaxial import (
    ENT,
    ASDSteel1D,
    Concrete01,
    Concrete02,
    ElasticMaterial,
    Hysteretic,
    Steel01,
    Steel02,
)


ALL_UNIAXIAL: list[type[UniaxialMaterial]] = [
    Steel01,
    Steel02,
    ASDSteel1D,
    Concrete01,
    Concrete02,
    Hysteretic,
    ElasticMaterial,
    ENT,
]


_MINIMAL_PARAMS: dict[type[UniaxialMaterial], dict[str, Any]] = {
    Steel01: {"fy": 420e6, "E": 200e9, "b": 0.01},
    Steel02: {"fy": 420e6, "E": 200e9, "b": 0.01},
    ASDSteel1D: {"E": 200e9, "sy": 375e6, "su": 480e6, "eu": 0.20},
    Concrete01: {
        "fpc": -30e6, "epsc0": -0.002,
        "fpcu": -25e6, "epsU": -0.006,
    },
    Concrete02: {
        "fpc": -30e6, "epsc0": -0.002,
        "fpcu": -25e6, "epsU": -0.006,
        "lambda_val": 0.1, "ft": 2.5e6, "Ets": 200e6,
    },
    Hysteretic: {
        "s1p": 100e3, "e1p": 0.001, "s2p": 200e3, "e2p": 0.005,
        "s1n": -100e3, "e1n": -0.001, "s2n": -200e3, "e2n": -0.005,
        "pinch_x": 1.0, "pinch_y": 1.0,
        "damage1": 0.0, "damage2": 0.0,
    },
    ElasticMaterial: {"E": 200e9},
    ENT: {"E": 200e9},
}


def _minimal(cls: type[UniaxialMaterial]) -> UniaxialMaterial:
    """Construct a minimal-valid instance of ``cls`` for smoke tests.

    The kwargs dict is class-specific and cannot be statically
    reconciled with each concrete dataclass signature; the
    ``ALL_UNIAXIAL`` roster + per-class ``_MINIMAL_PARAMS`` keep this
    safe at runtime.
    """
    return cls(**_MINIMAL_PARAMS[cls])


@pytest.mark.parametrize("cls", ALL_UNIAXIAL)
class TestUniaxialMaterialContract:
    def test_inherits_from_uniaxial_base(
        self, cls: type[UniaxialMaterial],
    ) -> None:
        assert issubclass(cls, UniaxialMaterial)
        assert issubclass(cls, Primitive)

    def test_is_frozen_kw_only_dataclass(
        self, cls: type[UniaxialMaterial],
    ) -> None:
        assert is_dataclass(cls)
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen
        assert params.kw_only

    def test_implements_emit(self, cls: type[UniaxialMaterial]) -> None:
        assert "_emit" in cls.__dict__
        rec = RecordingEmitter()
        _minimal(cls)._emit(rec, tag=1)
        assert len(rec.calls) == 1
        assert rec.calls[0][0] == "uniaxialMaterial"

    def test_implements_dependencies(
        self, cls: type[UniaxialMaterial],
    ) -> None:
        assert "dependencies" in cls.__dict__
        assert _minimal(cls).dependencies() == ()

    def test_repr_includes_class_name(
        self, cls: type[UniaxialMaterial],
    ) -> None:
        assert cls.__name__ in repr(_minimal(cls))
