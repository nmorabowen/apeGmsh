"""Contract tests for every concrete ``NDMaterial`` subclass.

Phase 1B parametrizes this file over :data:`ALL_ND`. Every primitive
listed here must satisfy the base contract: dataclass shape, the two
abstract methods, type-token in repr, and the ``NDMaterial`` family
parentage.

Adding a new nD material in a future slice should append it to
``ALL_ND``; the parametrized contract suite picks it up automatically.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Callable

import pytest

from apeGmsh.opensees._internal.types import NDMaterial, Primitive
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import (
    DruckerPrager,
    ElasticIsotropic,
    J2Plasticity,
    LadrunoJ2,
    LadrunoJ2Finite,
)


ALL_ND: list[type[NDMaterial]] = [
    ElasticIsotropic,
    J2Plasticity,
    DruckerPrager,
    LadrunoJ2,
    LadrunoJ2Finite,
]


# ---------------------------------------------------------------------------
# Minimal-constructor map: every concrete class needs a known-good
# kwargs payload so the contract tests can instantiate it.
# ---------------------------------------------------------------------------

_MINIMAL_KWARGS: dict[type[NDMaterial], Callable[[], dict[str, Any]]] = {
    ElasticIsotropic: lambda: {"E": 30e9, "nu": 0.2},
    J2Plasticity: lambda: {
        "K": 1.65e8,
        "G": 7.5e7,
        "sig0": 5.0e5,
        "sigInf": 7.0e5,
        "delta": 0.1,
        "H": 1.0e6,
    },
    DruckerPrager: lambda: {
        "K": 80.0e6,
        "G": 60.0e6,
        "sigmaY": 20.0e3,
        "rho": 0.0,
        "rhoBar": 0.0,
        "Kinf": 0.0,
        "Ko": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "H": 0.0,
        "theta": 1.0,
    },
    LadrunoJ2: lambda: {"K": 1.65e8, "G": 7.5e7, "sig0": 5.0e5},
    LadrunoJ2Finite: lambda: {"K": 1.65e8, "G": 7.5e7, "sig0": 5.0e5},
}


def _instantiate(cls: type[NDMaterial]) -> NDMaterial:
    """Build a minimal valid instance of ``cls``."""
    factory = _MINIMAL_KWARGS.get(cls)
    if factory is None:
        raise KeyError(
            f"Add a minimal-kwargs entry for {cls.__name__} to "
            f"tests/opensees/contract/test_nd_material_contract.py"
        )
    return cls(**factory())


# ---------------------------------------------------------------------------
# Contract suite
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_ND)
class TestNDMaterialContract:
    def test_inherits_from_nd_material(self, cls: type[NDMaterial]) -> None:
        assert issubclass(cls, NDMaterial)

    def test_inherits_from_primitive(self, cls: type[NDMaterial]) -> None:
        assert issubclass(cls, Primitive)

    def test_is_frozen_dataclass(self, cls: type[NDMaterial]) -> None:
        assert is_dataclass(cls)
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen
        assert all(f.kw_only for f in fields(cls))

    def test_has_emit(self, cls: type[NDMaterial]) -> None:
        assert hasattr(cls, "_emit")

    def test_has_dependencies(self, cls: type[NDMaterial]) -> None:
        assert hasattr(cls, "dependencies")

    def test_dependencies_returns_tuple(self, cls: type[NDMaterial]) -> None:
        instance = _instantiate(cls)
        deps = instance.dependencies()
        assert isinstance(deps, tuple)

    def test_repr_includes_type_token(self, cls: type[NDMaterial]) -> None:
        instance = _instantiate(cls)
        assert cls.__name__ in repr(instance)

    def test_emit_records_an_nd_material_call(
        self, cls: type[NDMaterial]
    ) -> None:
        """Every nD material must emit through ``Emitter.nDMaterial``."""
        instance = _instantiate(cls)
        emitter = RecordingEmitter()
        instance._emit(emitter, tag=1)
        assert len(emitter.calls) == 1
        method, args, _kwargs = emitter.calls[0]
        assert method == "nDMaterial"
        # Positional args: (mat_type_str, tag, *params)
        assert args[0] == cls.__name__
        assert args[1] == 1
