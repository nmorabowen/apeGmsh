"""Family contract gate for typed analysis-chain primitives (Phase 3C).

Every concrete subclass of any analysis-chain family base
(:class:`ConstraintHandler`, :class:`Numberer`, :class:`LinearSystem`,
:class:`ConvergenceTest`, :class:`SolutionAlgorithm`, :class:`Integrator`,
:class:`Analysis`) shipped by Phase 3C is listed in
:data:`ALL_ANALYSIS_COMPONENTS` and verified against the cross-family
contract:

* inherits from :class:`Primitive` (and from the matching family base)
* is a frozen, kw-only, slotted dataclass
* implements ``_emit`` and ``dependencies``
* ``dependencies()`` on a minimal instance returns ``()``
* ``__repr__`` includes the class name

Future analysis primitives append their classes to one of the
``ALL_*`` family lists and add an entry to :data:`_MINIMAL_PARAMS` —
the contract suite picks them up automatically.
"""
from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

import pytest

from apeGmsh.opensees._internal.types import (
    Analysis,
    ConstraintHandler,
    ConvergenceTest,
    Integrator,
    LinearSystem,
    Numberer,
    Primitive,
    SolutionAlgorithm,
)
from apeGmsh.opensees.analysis.algorithm import (
    BFGS,
    Broyden,
    KrylovNewton,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
)
from apeGmsh.opensees.analysis.algorithm import Linear as AlgorithmLinear
from apeGmsh.opensees.analysis.analysis import (
    Static,
    Transient,
    VariableTransient,
)
from apeGmsh.opensees.analysis.constraint_handler import (
    Lagrange,
    Penalty,
    Transformation,
)
from apeGmsh.opensees.analysis.constraint_handler import (
    Plain as ConstraintsPlain,
)
from apeGmsh.opensees.analysis.integrator import (
    ArcLength,
    CentralDifference,
    DisplacementControl,
    ExplicitDifference,
    HHT,
    LoadControl,
    Newmark,
)
from apeGmsh.opensees.analysis.numberer import AMD, RCM
from apeGmsh.opensees.analysis.numberer import Plain as NumbererPlain
from apeGmsh.opensees.analysis.system import (
    BandGeneral,
    BandSPD,
    FullGeneral,
    Mumps,
    ProfileSPD,
    SparseGeneral,
    UmfPack,
)
from apeGmsh.opensees.analysis.test import (
    EnergyIncr,
    FixedNumIter,
    NormDispIncr,
    NormUnbalance,
    RelativeNormDispIncr,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# ---------------------------------------------------------------------------
# Per-family rosters
# ---------------------------------------------------------------------------

ALL_CONSTRAINT_HANDLERS: list[type[ConstraintHandler]] = [
    ConstraintsPlain,
    Penalty,
    Transformation,
    Lagrange,
]

ALL_NUMBERERS: list[type[Numberer]] = [
    NumbererPlain,
    RCM,
    AMD,
]

ALL_SYSTEMS: list[type[LinearSystem]] = [
    BandGeneral,
    BandSPD,
    ProfileSPD,
    UmfPack,
    Mumps,
    SparseGeneral,
    FullGeneral,
]

ALL_TESTS: list[type[ConvergenceTest]] = [
    NormDispIncr,
    NormUnbalance,
    EnergyIncr,
    FixedNumIter,
    RelativeNormDispIncr,
]

ALL_ALGORITHMS: list[type[SolutionAlgorithm]] = [
    AlgorithmLinear,
    Newton,
    ModifiedNewton,
    NewtonLineSearch,
    KrylovNewton,
    BFGS,
    Broyden,
]

ALL_INTEGRATORS: list[type[Integrator]] = [
    LoadControl,
    DisplacementControl,
    ArcLength,
    Newmark,
    HHT,
    CentralDifference,
    ExplicitDifference,
]

ALL_ANALYSES: list[type[Analysis]] = [
    Static,
    Transient,
    VariableTransient,
]


# Cross-family aggregator — the union of all seven family rosters.
ALL_ANALYSIS_COMPONENTS: list[type[Primitive]] = [
    *ALL_CONSTRAINT_HANDLERS,
    *ALL_NUMBERERS,
    *ALL_SYSTEMS,
    *ALL_TESTS,
    *ALL_ALGORITHMS,
    *ALL_INTEGRATORS,
    *ALL_ANALYSES,
]


# ---------------------------------------------------------------------------
# Minimal kwargs for constructing a smoke instance of each class.
# ---------------------------------------------------------------------------

_MINIMAL_PARAMS: dict[type[Primitive], dict[str, Any]] = {
    # constraint_handler
    ConstraintsPlain: {},
    Penalty: {"alpha_sp": 1e10, "alpha_mp": 1e10},
    Transformation: {},
    Lagrange: {},
    # numberer
    NumbererPlain: {},
    RCM: {},
    AMD: {},
    # system
    BandGeneral: {},
    BandSPD: {},
    ProfileSPD: {},
    UmfPack: {},
    Mumps: {},
    SparseGeneral: {},
    FullGeneral: {},
    # test
    NormDispIncr: {"tol": 1e-6, "max_iter": 10},
    NormUnbalance: {"tol": 1e-6, "max_iter": 10},
    EnergyIncr: {"tol": 1e-6, "max_iter": 10},
    FixedNumIter: {"max_iter": 5},
    RelativeNormDispIncr: {"tol": 1e-6, "max_iter": 10},
    # algorithm
    AlgorithmLinear: {},
    Newton: {},
    ModifiedNewton: {},
    NewtonLineSearch: {"line_search": "Bisection"},
    KrylovNewton: {},
    BFGS: {},
    Broyden: {},
    # integrator
    LoadControl: {"dlam": 0.1},
    DisplacementControl: {"node": 1, "dof": 1, "dU": 0.001},
    ArcLength: {"s": 1.0, "alpha": 0.5},
    Newmark: {"gamma": 0.5, "beta": 0.25},
    HHT: {"alpha": 0.9},
    CentralDifference: {},
    ExplicitDifference: {},
    # analysis
    Static: {},
    Transient: {},
    VariableTransient: {},
}


def _minimal(cls: type[Primitive]) -> Primitive:
    """Construct a minimal-valid instance of ``cls`` for smoke tests."""
    return cls(**_MINIMAL_PARAMS[cls])  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Per-family base-inheritance map (each component must inherit from its
# family base AND from Primitive).
# ---------------------------------------------------------------------------

_FAMILY_BASE: dict[type[Primitive], type[Primitive]] = {}
for c in ALL_CONSTRAINT_HANDLERS:
    _FAMILY_BASE[c] = ConstraintHandler
for n in ALL_NUMBERERS:
    _FAMILY_BASE[n] = Numberer
for s in ALL_SYSTEMS:
    _FAMILY_BASE[s] = LinearSystem
for t in ALL_TESTS:
    _FAMILY_BASE[t] = ConvergenceTest
for a in ALL_ALGORITHMS:
    _FAMILY_BASE[a] = SolutionAlgorithm
for i in ALL_INTEGRATORS:
    _FAMILY_BASE[i] = Integrator
for x in ALL_ANALYSES:
    _FAMILY_BASE[x] = Analysis


# Map family base -> emitter method name. Used to assert each
# primitive's _emit lands on the correct Protocol method.
_FAMILY_EMITTER_METHOD: dict[type[Primitive], str] = {
    ConstraintHandler: "constraints",
    Numberer: "numberer",
    LinearSystem: "system",
    ConvergenceTest: "test",
    SolutionAlgorithm: "algorithm",
    Integrator: "integrator",
    Analysis: "analysis",
}


# ---------------------------------------------------------------------------
# Cross-family contract suite — parametrized over ALL_ANALYSIS_COMPONENTS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", ALL_ANALYSIS_COMPONENTS)
class TestAnalysisComponentContract:
    def test_inherits_from_primitive(self, cls: type[Primitive]) -> None:
        assert issubclass(cls, Primitive)

    def test_inherits_from_family_base(
        self, cls: type[Primitive],
    ) -> None:
        family = _FAMILY_BASE[cls]
        assert issubclass(cls, family), (
            f"{cls.__name__} should inherit from {family.__name__}"
        )

    def test_is_frozen_kw_only_dataclass(
        self, cls: type[Primitive],
    ) -> None:
        assert is_dataclass(cls), f"{cls.__name__} is not a dataclass"
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} dataclass is not frozen"
        assert params.kw_only, f"{cls.__name__} dataclass is not kw_only"

    def test_has_slots(self, cls: type[Primitive]) -> None:
        assert hasattr(cls, "__slots__"), (
            f"{cls.__name__} lacks __slots__"
        )

    def test_implements_emit(self, cls: type[Primitive]) -> None:
        assert "_emit" in cls.__dict__

    def test_implements_dependencies(
        self, cls: type[Primitive],
    ) -> None:
        assert "dependencies" in cls.__dict__

    def test_dependencies_returns_empty_tuple(
        self, cls: type[Primitive],
    ) -> None:
        # All analysis components are leaves — no composed children.
        assert _minimal(cls).dependencies() == ()

    def test_repr_includes_class_name(
        self, cls: type[Primitive],
    ) -> None:
        assert cls.__name__ in repr(_minimal(cls))

    def test_emit_records_correct_protocol_method(
        self, cls: type[Primitive],
    ) -> None:
        rec = RecordingEmitter()
        _minimal(cls)._emit(rec, tag=1)
        assert len(rec.calls) == 1
        family = _FAMILY_BASE[cls]
        expected_method = _FAMILY_EMITTER_METHOD[family]
        assert rec.calls[0][0] == expected_method, (
            f"{cls.__name__}._emit should call "
            f"emitter.{expected_method}(...), "
            f"got {rec.calls[0][0]}"
        )
