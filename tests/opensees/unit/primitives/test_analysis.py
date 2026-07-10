"""Unit tests for ``apeGmsh.opensees.analysis``.

Phase 3C ships the seven analysis-component family modules. Each
class is smoke-tested for:

* construction (defaults + explicit values)
* validation (per-class invariants)
* ``_emit`` records the right call into a :class:`RecordingEmitter`
* ``dependencies()`` returns ``()`` (every analysis primitive is a leaf)
* ``__repr__`` includes the class name

Plus namespace integration: each family's namespace constructs and
registers correctly via ``ops.<family>.<Type>(...)``.

Tests use :class:`RecordingEmitter` only — no openseespy, no gmsh,
no subprocess. Run with::

    pytest tests/opensees/unit/primitives/test_analysis.py -v
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.analysis.algorithm import (
    BFGS,
    Broyden,
    KrylovNewton,
    Linear,
    ModifiedNewton,
    Newton,
    NewtonLineSearch,
)
from apeGmsh.opensees.analysis.analysis import (
    Static,
    Transient,
    VariableTransient,
)
from apeGmsh.opensees.analysis.constraint_handler import (
    Auto as ConstraintsAuto,
)
from apeGmsh.opensees.analysis.constraint_handler import (
    LadrunoContact,
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
    CentralDifferenceLadruno,
    DisplacementControl,
    ExplicitBathe,
    ExplicitBatheLNVD,
    ExplicitDifference,
    HHT,
    CentralDifferenceSMS,
    ExplicitBatheLNVDSMS,
    ExplicitBatheSMS,
    LadrunoArcLength,
    LadrunoDynamicRelaxation,
    LadrunoGeneralizedAlpha,
    LadrunoHHT,
    LadrunoIndirectControl,
    LoadControl,
    Newmark,
)
from apeGmsh.opensees.analysis.numberer import (
    AMD,
    RCM,
    ParallelPlain,
    ParallelRCM,
)
from apeGmsh.opensees.analysis.numberer import Plain as NumbererPlain
from apeGmsh.opensees.analysis.system import (
    BandGeneral,
    BandSPD,
    Diagonal,
    FullGeneral,
    MPIDiagonal,
    Mumps,
    ParallelProfileSPD,
    ProfileSPD,
    SparseGeneral,
    SparseSYM,
    SProfileSPD,
    UmfPack,
)
from apeGmsh.opensees.analysis.test import (
    EnergyIncr,
    FixedNumIter,
    LadrunoStabilizedUnbalance,
    NormDispIncr,
    NormUnbalance,
    RelativeNormDispIncr,
)
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter


def _make_ops() -> apeSees:
    """Construct an apeSees with a stub FEMData (namespaces ignore it)."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


# ===========================================================================
# constraint_handler
# ===========================================================================

class TestConstraintsPlain:
    def test_construction(self) -> None:
        c = ConstraintsPlain()
        assert isinstance(c, ConstraintsPlain)

    def test_emit(self) -> None:
        e = RecordingEmitter()
        ConstraintsPlain()._emit(e, tag=1)
        assert e.calls == [("constraints", ("Plain",), {})]

    def test_dependencies_empty(self) -> None:
        assert ConstraintsPlain().dependencies() == ()

    def test_repr_has_class_name(self) -> None:
        assert "Plain" in repr(ConstraintsPlain())

    def test_emit_ignores_tag(self) -> None:
        e = RecordingEmitter()
        ConstraintsPlain()._emit(e, tag=999)
        assert e.calls[0][1] == ("Plain",)


class TestPenalty:
    def test_construction(self) -> None:
        c = Penalty(alpha_sp=1e10, alpha_mp=1e10)
        assert c.alpha_sp == 1e10
        assert c.alpha_mp == 1e10

    def test_zero_alpha_sp_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_sp must be > 0"):
            Penalty(alpha_sp=0.0, alpha_mp=1e10)

    def test_zero_alpha_mp_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_mp must be > 0"):
            Penalty(alpha_sp=1e10, alpha_mp=0.0)

    def test_negative_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_sp must be > 0"):
            Penalty(alpha_sp=-1.0, alpha_mp=1e10)

    def test_emit(self) -> None:
        e = RecordingEmitter()
        Penalty(alpha_sp=1e10, alpha_mp=1e8)._emit(e, tag=1)
        assert e.calls == [("constraints", ("Penalty", 1e10, 1e8), {})]

    def test_dependencies_empty(self) -> None:
        c = Penalty(alpha_sp=1e10, alpha_mp=1e10)
        assert c.dependencies() == ()


class TestTransformation:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        Transformation()._emit(e, tag=2)
        assert e.calls == [("constraints", ("Transformation",), {})]

    def test_dependencies_empty(self) -> None:
        assert Transformation().dependencies() == ()

    def test_repr_has_class_name(self) -> None:
        assert "Transformation" in repr(Transformation())


class TestLagrange:
    def test_construction_default(self) -> None:
        c = Lagrange()
        assert c.alpha_sp is None
        assert c.alpha_mp is None

    def test_construction_with_alphas(self) -> None:
        c = Lagrange(alpha_sp=1.0, alpha_mp=1.0)
        assert c.alpha_sp == 1.0
        assert c.alpha_mp == 1.0

    def test_one_alpha_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both alpha_sp and alpha_mp"):
            Lagrange(alpha_sp=1.0)

    def test_other_alpha_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both alpha_sp and alpha_mp"):
            Lagrange(alpha_mp=1.0)

    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        Lagrange()._emit(e, tag=1)
        assert e.calls == [("constraints", ("Lagrange",), {})]

    def test_emit_with_alphas(self) -> None:
        e = RecordingEmitter()
        Lagrange(alpha_sp=2.0, alpha_mp=3.0)._emit(e, tag=1)
        assert e.calls == [("constraints", ("Lagrange", 2.0, 3.0), {})]


class TestAuto:
    def test_construction_defaults(self) -> None:
        c = ConstraintsAuto()
        assert c.verbose is False
        assert c.auto_penalty is True
        assert c.auto_penalty_oom == 3.0
        assert c.user_penalty == 0.0

    def test_user_penalty_without_positive_value_raises(self) -> None:
        with pytest.raises(ValueError, match="user_penalty must be > 0"):
            ConstraintsAuto(auto_penalty=False, user_penalty=0.0)

    def test_emit_defaults_minimal(self) -> None:
        e = RecordingEmitter()
        ConstraintsAuto()._emit(e, tag=1)
        # auto_penalty=True with default oom=3.0 — no flags emitted.
        assert e.calls == [("constraints", ("Auto",), {})]

    def test_emit_verbose(self) -> None:
        e = RecordingEmitter()
        ConstraintsAuto(verbose=True)._emit(e, tag=1)
        assert e.calls == [("constraints", ("Auto", "-verbose"), {})]

    def test_emit_custom_auto_penalty_oom(self) -> None:
        e = RecordingEmitter()
        ConstraintsAuto(auto_penalty_oom=5.0)._emit(e, tag=1)
        assert e.calls == [
            ("constraints", ("Auto", "-autoPenalty", 5.0), {}),
        ]

    def test_emit_user_penalty(self) -> None:
        e = RecordingEmitter()
        ConstraintsAuto(
            auto_penalty=False, user_penalty=1e12,
        )._emit(e, tag=1)
        assert e.calls == [
            ("constraints", ("Auto", "-userPenalty", 1e12), {}),
        ]

    def test_emit_verbose_plus_user_penalty(self) -> None:
        e = RecordingEmitter()
        ConstraintsAuto(
            verbose=True, auto_penalty=False, user_penalty=1e12,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "constraints",
                ("Auto", "-verbose", "-userPenalty", 1e12),
                {},
            ),
        ]

    def test_dependencies_empty(self) -> None:
        assert ConstraintsAuto().dependencies() == ()


class TestConstraintsNamespace:
    def test_plain(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Plain()
        assert isinstance(c, ConstraintsPlain)
        assert ops.tag_for(c) == 1

    def test_penalty(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Penalty(alpha_sp=1e10, alpha_mp=1e10)
        assert isinstance(c, Penalty)
        assert c.alpha_sp == 1e10

    def test_transformation(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Transformation()
        assert isinstance(c, Transformation)

    def test_lagrange_default(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Lagrange()
        assert isinstance(c, Lagrange)
        assert c.alpha_sp is None

    def test_lagrange_with_alphas(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Lagrange(alpha_sp=1.0, alpha_mp=1.0)
        assert c.alpha_sp == 1.0

    def test_auto_defaults(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Auto()
        assert isinstance(c, ConstraintsAuto)
        assert c.auto_penalty is True

    def test_auto_user_penalty(self) -> None:
        ops = _make_ops()
        c = ops.constraints.Auto(
            auto_penalty=False, user_penalty=1e12,
        )
        assert c.user_penalty == 1e12


# ===========================================================================
# numberer
# ===========================================================================

class TestNumbererPlain:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        NumbererPlain()._emit(e, tag=1)
        assert e.calls == [("numberer", ("Plain",), {})]

    def test_dependencies_empty(self) -> None:
        assert NumbererPlain().dependencies() == ()


class TestRCM:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        RCM()._emit(e, tag=1)
        assert e.calls == [("numberer", ("RCM",), {})]


class TestAMD:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        AMD()._emit(e, tag=1)
        assert e.calls == [("numberer", ("AMD",), {})]


class TestParallelPlain:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        ParallelPlain()._emit(e, tag=1)
        assert e.calls == [("numberer", ("ParallelPlain",), {})]

    def test_dependencies_empty(self) -> None:
        assert ParallelPlain().dependencies() == ()


class TestParallelRCM:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        ParallelRCM()._emit(e, tag=1)
        assert e.calls == [("numberer", ("ParallelRCM",), {})]

    def test_dependencies_empty(self) -> None:
        assert ParallelRCM().dependencies() == ()


class TestNumbererNamespace:
    def test_plain(self) -> None:
        ops = _make_ops()
        n = ops.numberer.Plain()
        assert isinstance(n, NumbererPlain)

    def test_rcm(self) -> None:
        ops = _make_ops()
        n = ops.numberer.RCM()
        assert isinstance(n, RCM)

    def test_amd(self) -> None:
        ops = _make_ops()
        n = ops.numberer.AMD()
        assert isinstance(n, AMD)

    def test_parallel_plain(self) -> None:
        ops = _make_ops()
        n = ops.numberer.ParallelPlain()
        assert isinstance(n, ParallelPlain)

    def test_parallel_rcm(self) -> None:
        ops = _make_ops()
        n = ops.numberer.ParallelRCM()
        assert isinstance(n, ParallelRCM)


# ===========================================================================
# system
# ===========================================================================

@pytest.mark.parametrize(
    ("cls", "token"),
    [
        (BandGeneral, "BandGeneral"),
        (BandSPD, "BandSPD"),
        (ProfileSPD, "ProfileSPD"),
        (SProfileSPD, "SProfileSPD"),
        (UmfPack, "UmfPack"),
        (Mumps, "Mumps"),
        (SparseGeneral, "SparseGeneral"),
        (SparseSYM, "SparseSYM"),
        (FullGeneral, "FullGeneral"),
        (Diagonal, "Diagonal"),
        (MPIDiagonal, "MPIDiagonal"),
        (ParallelProfileSPD, "ParallelProfileSPD"),
    ],
)
def test_system_emit_records_token(
    cls: type, token: str,
) -> None:
    e = RecordingEmitter()
    cls()._emit(e, tag=1)
    assert e.calls == [("system", (token,), {})]


@pytest.mark.parametrize(
    "cls",
    [
        BandGeneral, BandSPD, ProfileSPD, SProfileSPD, UmfPack,
        Mumps, SparseGeneral, SparseSYM, FullGeneral, Diagonal,
        MPIDiagonal, ParallelProfileSPD,
    ],
)
def test_system_dependencies_empty(cls: type) -> None:
    assert cls().dependencies() == ()


class TestSystemNamespace:
    def test_band_general(self) -> None:
        ops = _make_ops()
        s = ops.system.BandGeneral()
        assert isinstance(s, BandGeneral)

    def test_band_spd(self) -> None:
        ops = _make_ops()
        s = ops.system.BandSPD()
        assert isinstance(s, BandSPD)

    def test_profile_spd(self) -> None:
        ops = _make_ops()
        s = ops.system.ProfileSPD()
        assert isinstance(s, ProfileSPD)

    def test_umfpack(self) -> None:
        ops = _make_ops()
        s = ops.system.UmfPack()
        assert isinstance(s, UmfPack)

    def test_mumps(self) -> None:
        ops = _make_ops()
        s = ops.system.Mumps()
        assert isinstance(s, Mumps)

    def test_sparse_general(self) -> None:
        ops = _make_ops()
        s = ops.system.SparseGeneral()
        assert isinstance(s, SparseGeneral)

    def test_full_general(self) -> None:
        ops = _make_ops()
        s = ops.system.FullGeneral()
        assert isinstance(s, FullGeneral)

    def test_sprofile_spd(self) -> None:
        ops = _make_ops()
        s = ops.system.SProfileSPD()
        assert isinstance(s, SProfileSPD)

    def test_sparse_sym(self) -> None:
        ops = _make_ops()
        s = ops.system.SparseSYM()
        assert isinstance(s, SparseSYM)

    def test_diagonal(self) -> None:
        ops = _make_ops()
        s = ops.system.Diagonal()
        assert isinstance(s, Diagonal)

    def test_mpi_diagonal(self) -> None:
        ops = _make_ops()
        s = ops.system.MPIDiagonal()
        assert isinstance(s, MPIDiagonal)

    def test_parallel_profile_spd(self) -> None:
        ops = _make_ops()
        s = ops.system.ParallelProfileSPD()
        assert isinstance(s, ParallelProfileSPD)


# ===========================================================================
# test (convergence test)
# ===========================================================================

class TestNormDispIncr:
    def test_construction(self) -> None:
        t = NormDispIncr(tol=1e-6, max_iter=10)
        assert t.tol == 1e-6
        assert t.max_iter == 10
        assert t.print_flag == 0
        assert t.norm_type == 2

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            NormDispIncr(tol=0.0, max_iter=10)

    def test_negative_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            NormDispIncr(tol=-1e-6, max_iter=10)

    def test_zero_max_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            NormDispIncr(tol=1e-6, max_iter=0)

    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        NormDispIncr(tol=1e-6, max_iter=10)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("NormDispIncr", 1e-6, 10, 0, 2), {})
        ]

    def test_emit_with_print_flag(self) -> None:
        e = RecordingEmitter()
        NormDispIncr(
            tol=1e-6, max_iter=10, print_flag=1, norm_type=1,
        )._emit(e, tag=1)
        assert e.calls == [
            ("test", ("NormDispIncr", 1e-6, 10, 1, 1), {})
        ]

    def test_dependencies_empty(self) -> None:
        assert NormDispIncr(tol=1e-6, max_iter=10).dependencies() == ()


class TestNormUnbalance:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        NormUnbalance(tol=1e-3, max_iter=20)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("NormUnbalance", 1e-3, 20, 0, 2), {})
        ]

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            NormUnbalance(tol=0.0, max_iter=10)


class TestEnergyIncr:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        EnergyIncr(tol=1e-8, max_iter=15)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("EnergyIncr", 1e-8, 15, 0, 2), {})
        ]

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            EnergyIncr(tol=0.0, max_iter=10)


class TestFixedNumIter:
    def test_construction(self) -> None:
        t = FixedNumIter(max_iter=5)
        assert t.max_iter == 5

    def test_zero_max_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            FixedNumIter(max_iter=0)

    def test_emit(self) -> None:
        e = RecordingEmitter()
        FixedNumIter(max_iter=5)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("FixedNumIter", 5, 0, 2), {})
        ]


class TestRelativeNormDispIncr:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        RelativeNormDispIncr(tol=1e-6, max_iter=10)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("RelativeNormDispIncr", 1e-6, 10, 0, 2), {})
        ]

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            RelativeNormDispIncr(tol=0.0, max_iter=10)


class TestTestNamespace:
    def test_norm_disp_incr(self) -> None:
        ops = _make_ops()
        t = ops.test.NormDispIncr(tol=1e-6, max_iter=10)
        assert isinstance(t, NormDispIncr)
        assert t.tol == 1e-6

    def test_norm_unbalance(self) -> None:
        ops = _make_ops()
        t = ops.test.NormUnbalance(tol=1e-3, max_iter=10)
        assert isinstance(t, NormUnbalance)

    def test_energy_incr(self) -> None:
        ops = _make_ops()
        t = ops.test.EnergyIncr(tol=1e-8, max_iter=10)
        assert isinstance(t, EnergyIncr)

    def test_fixed_num_iter(self) -> None:
        ops = _make_ops()
        t = ops.test.FixedNumIter(max_iter=5)
        assert isinstance(t, FixedNumIter)

    def test_relative_norm_disp_incr(self) -> None:
        ops = _make_ops()
        t = ops.test.RelativeNormDispIncr(tol=1e-6, max_iter=10)
        assert isinstance(t, RelativeNormDispIncr)


# ===========================================================================
# algorithm
# ===========================================================================

class TestAlgorithmLinear:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        Linear()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Linear",), {})]

    def test_emit_factor_once(self) -> None:
        e = RecordingEmitter()
        Linear(factor_once=True)._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Linear", "-FactorOnce"), {})]

    def test_emit_initial_tangent(self) -> None:
        e = RecordingEmitter()
        Linear(tangent="initial")._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Linear", "-initial"), {})]

    def test_emit_tangent_and_factor_once_combine(self) -> None:
        # Linear's parser loops its options, so both flags emit together.
        e = RecordingEmitter()
        Linear(tangent="secant", factor_once=True)._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("Linear", "-secant", "-FactorOnce"), {})
        ]

    def test_dependencies_empty(self) -> None:
        assert Linear().dependencies() == ()


class TestNewton:
    def test_default_tangent(self) -> None:
        a = Newton()
        assert a.tangent == "tangent"

    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        Newton()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Newton",), {})]

    def test_emit_secant(self) -> None:
        e = RecordingEmitter()
        Newton(tangent="secant")._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Newton", "-secant"), {})]

    def test_emit_initial(self) -> None:
        e = RecordingEmitter()
        Newton(tangent="initial")._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Newton", "-initial"), {})]

    def test_emit_initial_then_current_uses_upstream_typo(self) -> None:
        # OpenSees matches the typo'd token, so we must emit it verbatim.
        e = RecordingEmitter()
        Newton(tangent="initialThenCurrent")._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("Newton", "-intialThenCurrent"), {})
        ]

    def test_emit_hall_bare(self) -> None:
        e = RecordingEmitter()
        Newton(tangent="hall")._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Newton", "-hall"), {})]

    def test_emit_hall_with_factors_last(self) -> None:
        e = RecordingEmitter()
        Newton(
            tangent="hall", hall_i_factor=0.1, hall_c_factor=0.9,
        )._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("Newton", "-hall", 0.1, 0.9), {})
        ]

    def test_hall_factors_without_hall_raises(self) -> None:
        with pytest.raises(ValueError, match="require tangent='hall'"):
            Newton(tangent="initial", hall_i_factor=0.1, hall_c_factor=0.9)

    def test_half_hall_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="both hall_i_factor"):
            Newton(tangent="hall", hall_i_factor=0.1)


class TestModifiedNewton:
    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        ModifiedNewton()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("ModifiedNewton",), {})]

    def test_emit_secant(self) -> None:
        e = RecordingEmitter()
        ModifiedNewton(tangent="secant")._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("ModifiedNewton", "-secant"), {})
        ]

    def test_emit_initial(self) -> None:
        e = RecordingEmitter()
        ModifiedNewton(tangent="initial")._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("ModifiedNewton", "-initial"), {})
        ]

    def test_emit_factor_once(self) -> None:
        e = RecordingEmitter()
        ModifiedNewton(factor_once=True)._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("ModifiedNewton", "-FactorOnce"), {})
        ]

    def test_factor_once_with_tangent_raises(self) -> None:
        # The OpenSees parser reads a single option token — combining
        # factor_once with a non-default tangent would drop one flag.
        with pytest.raises(ValueError, match="factor_once cannot be combined"):
            ModifiedNewton(tangent="initial", factor_once=True)

    def test_emit_hall_with_factors(self) -> None:
        e = RecordingEmitter()
        ModifiedNewton(
            tangent="hall", hall_i_factor=0.2, hall_c_factor=0.8,
        )._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("ModifiedNewton", "-hall", 0.2, 0.8), {})
        ]

    def test_factor_once_with_hall_raises(self) -> None:
        with pytest.raises(ValueError, match="factor_once cannot be combined"):
            ModifiedNewton(tangent="hall", factor_once=True)


class TestNewtonLineSearch:
    def test_construction_minimal(self) -> None:
        a = NewtonLineSearch(line_search="Bisection")
        assert a.line_search == "Bisection"
        assert a.tol is None
        assert a.max_iter is None

    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        NewtonLineSearch(line_search="Bisection")._emit(e, tag=1)
        assert e.calls == [
            ("algorithm", ("NewtonLineSearch", "-type", "Bisection"), {})
        ]

    def test_emit_with_tol(self) -> None:
        e = RecordingEmitter()
        NewtonLineSearch(
            line_search="Secant", tol=0.8,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "algorithm",
                ("NewtonLineSearch", "-type", "Secant", "-tol", 0.8),
                {},
            )
        ]

    def test_emit_full(self) -> None:
        e = RecordingEmitter()
        NewtonLineSearch(
            line_search="RegulaFalsi",
            tol=0.8, max_iter=10, min_eta=0.1, max_eta=10.0,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "algorithm",
                (
                    "NewtonLineSearch",
                    "-type", "RegulaFalsi",
                    "-tol", 0.8,
                    "-maxIter", 10,
                    "-minEta", 0.1,
                    "-maxEta", 10.0,
                ),
                {},
            )
        ]

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            NewtonLineSearch(line_search="Bisection", tol=0.0)

    def test_zero_max_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            NewtonLineSearch(line_search="Bisection", max_iter=0)


class TestKrylovNewton:
    def test_construction_minimal(self) -> None:
        a = KrylovNewton()
        assert a.iterate is None
        assert a.increment is None
        assert a.max_dim is None

    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        KrylovNewton()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("KrylovNewton",), {})]

    def test_emit_full(self) -> None:
        e = RecordingEmitter()
        KrylovNewton(
            iterate="current", increment="initial", max_dim=8,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "algorithm",
                (
                    "KrylovNewton",
                    "-iterate", "current",
                    "-increment", "initial",
                    "-maxDim", 8,
                ),
                {},
            )
        ]

    def test_zero_max_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_dim must be >= 1"):
            KrylovNewton(max_dim=0)


class TestBFGS:
    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        BFGS()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("BFGS",), {})]

    def test_emit_with_count(self) -> None:
        e = RecordingEmitter()
        BFGS(count=10)._emit(e, tag=1)
        assert e.calls == [("algorithm", ("BFGS", 10), {})]

    def test_zero_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            BFGS(count=0)


class TestBroyden:
    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        Broyden()._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Broyden",), {})]

    def test_emit_with_count(self) -> None:
        e = RecordingEmitter()
        Broyden(count=5)._emit(e, tag=1)
        assert e.calls == [("algorithm", ("Broyden", 5), {})]


class TestAlgorithmNamespace:
    def test_linear(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Linear()
        assert isinstance(a, Linear)

    def test_linear_factor_once(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Linear(factor_once=True)
        assert a.factor_once is True

    def test_modified_newton_factor_once(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.ModifiedNewton(factor_once=True)
        assert isinstance(a, ModifiedNewton)
        assert a.factor_once is True

    def test_newton_hall(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Newton(
            tangent="hall", hall_i_factor=0.1, hall_c_factor=0.9,
        )
        assert a.tangent == "hall"
        assert a.hall_i_factor == 0.1

    def test_newton_initial_then_current(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Newton(tangent="initialThenCurrent")
        assert a.tangent == "initialThenCurrent"

    def test_newton_default(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Newton()
        assert isinstance(a, Newton)
        assert a.tangent == "tangent"

    def test_newton_secant(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Newton(tangent="secant")
        assert a.tangent == "secant"

    def test_modified_newton(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.ModifiedNewton()
        assert isinstance(a, ModifiedNewton)

    def test_newton_line_search(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.NewtonLineSearch(line_search="Bisection")
        assert isinstance(a, NewtonLineSearch)

    def test_krylov_newton(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.KrylovNewton(max_dim=8)
        assert isinstance(a, KrylovNewton)
        assert a.max_dim == 8

    def test_bfgs(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.BFGS(count=10)
        assert isinstance(a, BFGS)

    def test_broyden(self) -> None:
        ops = _make_ops()
        a = ops.algorithm.Broyden()
        assert isinstance(a, Broyden)


# ===========================================================================
# integrator
# ===========================================================================

class TestLoadControl:
    def test_construction_minimal(self) -> None:
        i = LoadControl(dlam=0.1)
        assert i.dlam == 0.1
        assert i.num_iter is None
        assert i.min_lam is None
        assert i.max_lam is None

    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        LoadControl(dlam=0.1)._emit(e, tag=1)
        assert e.calls == [("integrator", ("LoadControl", 0.1), {})]

    def test_emit_with_num_iter(self) -> None:
        e = RecordingEmitter()
        LoadControl(dlam=0.1, num_iter=5)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("LoadControl", 0.1, 5), {})
        ]

    def test_emit_full(self) -> None:
        e = RecordingEmitter()
        LoadControl(
            dlam=0.1, num_iter=5, min_lam=0.01, max_lam=1.0,
        )._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("LoadControl", 0.1, 5, 0.01, 1.0), {})
        ]

    def test_min_lam_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both min_lam and max_lam"):
            LoadControl(dlam=0.1, num_iter=5, min_lam=0.01)

    def test_min_lam_without_num_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="require num_iter"):
            LoadControl(dlam=0.1, min_lam=0.01, max_lam=1.0)

    def test_min_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_lam must be <= max_lam"):
            LoadControl(dlam=0.1, num_iter=5, min_lam=2.0, max_lam=1.0)

    def test_zero_num_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="num_iter must be >= 1"):
            LoadControl(dlam=0.1, num_iter=0)


class TestDisplacementControl:
    def test_construction_minimal(self) -> None:
        i = DisplacementControl(node=10, dof=1, dU=0.001)
        assert i.node == 10
        assert i.dof == 1
        assert i.dU == 0.001

    def test_zero_dof_raises(self) -> None:
        with pytest.raises(ValueError, match="dof must be >= 1"):
            DisplacementControl(node=10, dof=0, dU=0.001)

    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        DisplacementControl(
            node=10, dof=1, dU=0.001,
        )._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("DisplacementControl", 10, 1, 0.001), {})
        ]

    def test_emit_full(self) -> None:
        e = RecordingEmitter()
        DisplacementControl(
            node=10, dof=1, dU=0.001,
            num_iter=5, min_dU=1e-5, max_dU=1e-2,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("DisplacementControl", 10, 1, 0.001, 5, 1e-5, 1e-2),
                {},
            )
        ]

    def test_min_dU_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both min_dU and max_dU"):
            DisplacementControl(
                node=10, dof=1, dU=0.001, num_iter=5, min_dU=1e-5,
            )

    def test_min_dU_without_num_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="require num_iter"):
            DisplacementControl(
                node=10, dof=1, dU=0.001, min_dU=1e-5, max_dU=1e-2,
            )

    def test_min_above_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_dU must be <= max_dU"):
            DisplacementControl(
                node=10, dof=1, dU=0.001,
                num_iter=5, min_dU=1e-2, max_dU=1e-5,
            )


class TestArcLength:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        ArcLength(s=1.0, alpha=0.5)._emit(e, tag=1)
        assert e.calls == [("integrator", ("ArcLength", 1.0, 0.5), {})]

    def test_zero_s_raises(self) -> None:
        with pytest.raises(ValueError, match="s must be > 0"):
            ArcLength(s=0.0, alpha=0.5)


class TestNewmark:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        Newmark(gamma=0.5, beta=0.25)._emit(e, tag=1)
        assert e.calls == [("integrator", ("Newmark", 0.5, 0.25), {})]

    def test_emit_with_form(self) -> None:
        e = RecordingEmitter()
        Newmark(gamma=0.5, beta=0.25, form="A")._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("Newmark", 0.5, 0.25, "-form", "A"), {})
        ]


class TestHHT:
    def test_emit_alpha_only(self) -> None:
        e = RecordingEmitter()
        HHT(alpha=0.9)._emit(e, tag=1)
        assert e.calls == [("integrator", ("HHT", 0.9), {})]

    def test_emit_with_gamma_beta(self) -> None:
        e = RecordingEmitter()
        HHT(alpha=0.9, gamma=0.5, beta=0.25)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("HHT", 0.9, 0.5, 0.25), {})
        ]

    def test_gamma_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both gamma and beta"):
            HHT(alpha=0.9, gamma=0.5)


class TestCentralDifference:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        CentralDifference()._emit(e, tag=1)
        assert e.calls == [("integrator", ("CentralDifference",), {})]


class TestExplicitDifference:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        ExplicitDifference()._emit(e, tag=1)
        assert e.calls == [("integrator", ("ExplicitDifference",), {})]


# ---------------------------------------------------------------------------
# Fork-only explicit integrators (Ladruno)
# ---------------------------------------------------------------------------

class TestExplicitBathe:
    def test_emit_default(self) -> None:
        e = RecordingEmitter()
        ExplicitBathe()._emit(e, tag=1)
        assert e.calls == [("integrator", ("ExplicitBathe", 0.54), {})]

    def test_emit_all_flags_canonical_order(self) -> None:
        e = RecordingEmitter()
        ExplicitBathe(
            p=0.6,
            cfl=True,
            cfl_abort=True,
            tangent=True,
            recompute=5,
            lump="diagonal",
            verbose=True,
            divergence=2.0,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "ExplicitBathe",
                    0.6,
                    "-cfl",
                    "-cflAbort",
                    "-tangent",
                    "-recompute",
                    5,
                    "-lump",
                    "diagonal",
                    "-verbose",
                    "-divergence",
                    2.0,
                ),
                {},
            )
        ]

    def test_tcl_line(self) -> None:
        e = TclEmitter()
        ExplicitBathe(p=0.54, cfl=True)._emit(e, tag=1)
        assert "integrator ExplicitBathe 0.54 -cfl" in e.lines()

    def test_py_line(self) -> None:
        e = PyEmitter()
        ExplicitBathe(p=0.54, cfl=True)._emit(e, tag=1)
        assert "ops.integrator('ExplicitBathe', 0.54, '-cfl')" in e.lines()

    def test_p_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="p must be in"):
            ExplicitBathe(p=0.0)
        with pytest.raises(ValueError, match="p must be in"):
            ExplicitBathe(p=1.0)

    def test_recompute_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="recompute must be >= 1"):
            ExplicitBathe(recompute=0)

    def test_bad_lump_raises(self) -> None:
        with pytest.raises(ValueError, match="lump must be"):
            ExplicitBathe(lump="banded")  # type: ignore[arg-type]

    def test_nonpositive_divergence_raises(self) -> None:
        with pytest.raises(ValueError, match="divergence factor must be > 0"):
            ExplicitBathe(divergence=0.0)

    def test_dependencies_empty(self) -> None:
        assert ExplicitBathe().dependencies() == ()


class TestExplicitBatheLNVD:
    def test_emit_default_emits_both_positionals(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheLNVD()._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("ExplicitBatheLNVD", 0.54, 0.8), {})
        ]

    def test_emit_all_flags(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheLNVD(
            p=0.6,
            alpha=0.0,
            cfl=True,
            lump="rowsum",
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("ExplicitBatheLNVD", 0.6, 0.0, "-cfl", "-lump", "rowsum"),
                {},
            )
        ]

    def test_py_line(self) -> None:
        e = PyEmitter()
        ExplicitBatheLNVD(p=0.54, alpha=0.8)._emit(e, tag=1)
        assert (
            "ops.integrator('ExplicitBatheLNVD', 0.54, 0.8)" in e.lines()
        )

    def test_alpha_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"alpha .* must be in"):
            ExplicitBatheLNVD(alpha=1.0)
        with pytest.raises(ValueError, match=r"alpha .* must be in"):
            ExplicitBatheLNVD(alpha=-0.1)

    def test_alpha_zero_allowed(self) -> None:
        # alpha == 0 is valid (disables damping) — must not raise.
        assert ExplicitBatheLNVD(alpha=0.0).alpha == 0.0

    def test_dependencies_empty(self) -> None:
        assert ExplicitBatheLNVD().dependencies() == ()


class TestCentralDifferenceLadruno:
    def test_emit_default_no_positional(self) -> None:
        e = RecordingEmitter()
        CentralDifferenceLadruno()._emit(e, tag=1)
        assert e.calls == [("integrator", ("CentralDifferenceLadruno",), {})]

    def test_emit_flags(self) -> None:
        e = RecordingEmitter()
        CentralDifferenceLadruno(cfl=True, lump="diagonal")._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("CentralDifferenceLadruno", "-cfl", "-lump", "diagonal"),
                {},
            )
        ]

    def test_tcl_line(self) -> None:
        e = TclEmitter()
        CentralDifferenceLadruno(cfl=True)._emit(e, tag=1)
        assert "integrator CentralDifferenceLadruno -cfl" in e.lines()

    def test_recompute_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="recompute must be >= 1"):
            CentralDifferenceLadruno(recompute=-1)

    def test_dependencies_empty(self) -> None:
        assert CentralDifferenceLadruno().dependencies() == ()


# ---------------------------------------------------------------------------
# Fork-only path-following / dynamic-relaxation integrators (Ladruno)
# ---------------------------------------------------------------------------

class TestLadrunoArcLength:
    def test_emit_minimal_matches_stock_shape(self) -> None:
        e = RecordingEmitter()
        LadrunoArcLength(s=1.0, alpha=0.5)._emit(e, tag=1)
        assert e.calls == [("integrator", ("LadrunoArcLength", 1.0, 0.5), {})]

    def test_emit_adapt_triple(self) -> None:
        e = RecordingEmitter()
        LadrunoArcLength(
            s=1.0, alpha=0.5, jd=4, ell_min=0.1, ell_max=10.0, p_exp=0.5,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoArcLength", 1.0, 0.5,
                    "-adapt", 4, 0.1, 10.0, "-p", 0.5,
                ),
                {},
            )
        ]

    def test_emit_stabilize_renders_explicit_f_default(self) -> None:
        # The optional -stabilize f is always rendered (fork default 2e-4)
        # to defend the openseespy greedy-read quirk.
        e = RecordingEmitter()
        LadrunoArcLength(s=1.0, alpha=0.5, stabilize=True)._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("LadrunoArcLength", 1.0, 0.5, "-stabilize", 2.0e-4),
                {},
            )
        ]

    def test_emit_stabilize_full(self) -> None:
        e = RecordingEmitter()
        LadrunoArcLength(
            s=1.0, alpha=0.5, stabilize=True, f_target=1e-3,
            adapt_stab=True, c_visc=5.0, mass="lumped", mass_scale=2.0,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoArcLength", 1.0, 0.5,
                    "-stabilize", 1e-3, "-adaptStab", "-cVisc", 5.0,
                    "-mass", "lumped", 2.0,
                ),
                {},
            )
        ]

    def test_emit_mass_unity_no_trailing_scale(self) -> None:
        e = RecordingEmitter()
        LadrunoArcLength(
            s=1.0, alpha=0.5, stabilize=True, mass="unity",
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoArcLength", 1.0, 0.5,
                    "-stabilize", 2.0e-4, "-mass", "unity",
                ),
                {},
            )
        ]

    def test_tcl_line(self) -> None:
        e = TclEmitter()
        LadrunoArcLength(s=2.0, alpha=1.0)._emit(e, tag=1)
        assert "integrator LadrunoArcLength 2.0 1.0" in e.lines()

    def test_zero_s_raises(self) -> None:
        with pytest.raises(ValueError, match="s must be > 0"):
            LadrunoArcLength(s=0.0, alpha=0.5)

    def test_partial_adapt_triple_raises(self) -> None:
        with pytest.raises(ValueError, match="-adapt triple"):
            LadrunoArcLength(s=1.0, alpha=0.5, jd=4)

    def test_adapt_ell_order_raises(self) -> None:
        with pytest.raises(ValueError, match="ell_min must be <= ell_max"):
            LadrunoArcLength(
                s=1.0, alpha=0.5, jd=4, ell_min=10.0, ell_max=1.0,
            )

    def test_stabilize_knob_without_stabilize_raises(self) -> None:
        with pytest.raises(ValueError, match="require stabilize=True"):
            LadrunoArcLength(s=1.0, alpha=0.5, mass="unity")
        with pytest.raises(ValueError, match="require stabilize=True"):
            LadrunoArcLength(s=1.0, alpha=0.5, adapt_stab=True)

    def test_mass_scale_without_lumped_raises(self) -> None:
        with pytest.raises(ValueError, match="mass_scale only applies"):
            LadrunoArcLength(
                s=1.0, alpha=0.5, stabilize=True, mass="unity", mass_scale=2.0,
            )

    def test_dependencies_empty(self) -> None:
        assert LadrunoArcLength(s=1.0, alpha=0.5).dependencies() == ()


class TestLadrunoDynamicRelaxation:
    def test_emit_minimal_no_args(self) -> None:
        e = RecordingEmitter()
        LadrunoDynamicRelaxation()._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("LadrunoDynamicRelaxation",), {})
        ]

    def test_emit_full(self) -> None:
        e = RecordingEmitter()
        LadrunoDynamicRelaxation(
            mass="gershgorin", dt=0.5, recompute=50,
            damping="kinetic", divergence=2.0, verbose=True,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoDynamicRelaxation",
                    "-mass", "gershgorin", "-dt", 0.5, "-recompute", 50,
                    "-damping", "kinetic", "-divergence", 2.0, "-verbose",
                ),
                {},
            )
        ]

    def test_emit_lumped_renders_explicit_scale(self) -> None:
        e = RecordingEmitter()
        LadrunoDynamicRelaxation(mass="lumped")._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("LadrunoDynamicRelaxation", "-mass", "lumped", 1.0),
                {},
            )
        ]

    def test_emit_viscous_renders_explicit_zeta(self) -> None:
        e = RecordingEmitter()
        LadrunoDynamicRelaxation(
            damping="viscous", no_auto_refresh=True, interp=True,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoDynamicRelaxation",
                    "-damping", "viscous", 1.0, "-noAutoRefresh", "-interp",
                ),
                {},
            )
        ]

    def test_recompute_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="recompute must be >= 1"):
            LadrunoDynamicRelaxation(recompute=0)

    def test_zeta_without_viscous_raises(self) -> None:
        with pytest.raises(ValueError, match="zeta only applies"):
            LadrunoDynamicRelaxation(damping="kinetic", zeta=0.5)

    def test_mass_scale_without_lumped_raises(self) -> None:
        with pytest.raises(ValueError, match="mass_scale only applies"):
            LadrunoDynamicRelaxation(mass="gershgorin", mass_scale=2.0)

    def test_dependencies_empty(self) -> None:
        assert LadrunoDynamicRelaxation().dependencies() == ()


class TestLadrunoIndirectControl:
    def test_emit_single_dof(self) -> None:
        e = RecordingEmitter()
        LadrunoIndirectControl(
            incr=0.01, controls=((10, 1, 1.0),),
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                ("LadrunoIndirectControl", 0.01, "-dof", 10, 1, 1.0),
                {},
            )
        ]

    def test_emit_cmod_two_dofs_with_iter(self) -> None:
        e = RecordingEmitter()
        LadrunoIndirectControl(
            incr=0.01,
            controls=((5, 1, 1.0), (7, 1, -1.0)),
            num_iter=4, dmin=0.001, dmax=0.1,
        )._emit(e, tag=1)
        assert e.calls == [
            (
                "integrator",
                (
                    "LadrunoIndirectControl", 0.01,
                    "-dof", 5, 1, 1.0, "-dof", 7, 1, -1.0,
                    "-iter", 4, 0.001, 0.1,
                ),
                {},
            )
        ]

    def test_tcl_line(self) -> None:
        e = TclEmitter()
        LadrunoIndirectControl(
            incr=0.01, controls=((10, 2, 1.0),),
        )._emit(e, tag=1)
        assert "integrator LadrunoIndirectControl 0.01 -dof 10 2 1.0" in e.lines()

    def test_empty_controls_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LadrunoIndirectControl(incr=0.01, controls=())

    def test_dof_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dof must be >= 1"):
            LadrunoIndirectControl(incr=0.01, controls=((10, 0, 1.0),))

    def test_bracket_requires_num_iter(self) -> None:
        with pytest.raises(ValueError, match="dmin/dmax require num_iter"):
            LadrunoIndirectControl(
                incr=0.01, controls=((10, 1, 1.0),), dmin=0.0, dmax=0.1,
            )

    def test_half_bracket_raises(self) -> None:
        with pytest.raises(ValueError, match="both dmin and dmax"):
            LadrunoIndirectControl(
                incr=0.01, controls=((10, 1, 1.0),), num_iter=4, dmin=0.0,
            )

    def test_dependencies_empty(self) -> None:
        i = LadrunoIndirectControl(incr=0.01, controls=((10, 1, 1.0),))
        assert i.dependencies() == ()


class TestIntegratorNamespace:
    def test_load_control(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LoadControl(dlam=0.1)
        assert isinstance(i, LoadControl)
        assert i.dlam == 0.1

    def test_displacement_control(self) -> None:
        ops = _make_ops()
        i = ops.integrator.DisplacementControl(
            node=10, dof=1, dU=0.001,
        )
        assert isinstance(i, DisplacementControl)

    def test_arc_length(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ArcLength(s=1.0, alpha=0.5)
        assert isinstance(i, ArcLength)

    def test_newmark(self) -> None:
        ops = _make_ops()
        i = ops.integrator.Newmark(gamma=0.5, beta=0.25)
        assert isinstance(i, Newmark)

    def test_newmark_form(self) -> None:
        ops = _make_ops()
        i = ops.integrator.Newmark(gamma=0.5, beta=0.25, form="V")
        assert isinstance(i, Newmark)
        assert i.form == "V"

    def test_hht(self) -> None:
        ops = _make_ops()
        i = ops.integrator.HHT(alpha=0.9)
        assert isinstance(i, HHT)

    def test_central_difference(self) -> None:
        ops = _make_ops()
        i = ops.integrator.CentralDifference()
        assert isinstance(i, CentralDifference)

    def test_explicit_difference(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ExplicitDifference()
        assert isinstance(i, ExplicitDifference)

    def test_explicit_bathe(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ExplicitBathe(p=0.54, cfl=True)
        assert isinstance(i, ExplicitBathe)
        assert i.p == 0.54
        assert i.cfl is True

    def test_explicit_bathe_lnvd(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ExplicitBatheLNVD(p=0.54, alpha=0.8)
        assert isinstance(i, ExplicitBatheLNVD)
        assert i.alpha == 0.8

    def test_central_difference_ladruno(self) -> None:
        ops = _make_ops()
        i = ops.integrator.CentralDifferenceLadruno(cfl=True, verbose=True)
        assert isinstance(i, CentralDifferenceLadruno)
        assert i.verbose is True

    def test_ladruno_arc_length(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LadrunoArcLength(
            s=1.0, alpha=0.5, stabilize=True, mass="unity",
        )
        assert isinstance(i, LadrunoArcLength)
        assert i.stabilize is True
        assert i.mass == "unity"

    def test_ladruno_dynamic_relaxation(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LadrunoDynamicRelaxation(
            mass="gershgorin", damping="kinetic",
        )
        assert isinstance(i, LadrunoDynamicRelaxation)
        assert i.mass == "gershgorin"

    def test_ladruno_indirect_control(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LadrunoIndirectControl(
            incr=0.01, controls=[(5, 1, 1.0), (7, 1, -1.0)],
        )
        assert isinstance(i, LadrunoIndirectControl)
        assert i.controls == ((5, 1, 1.0), (7, 1, -1.0))


# ===========================================================================
# analysis (analysis-type)
# ===========================================================================

class TestStatic:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        Static()._emit(e, tag=1)
        assert e.calls == [("analysis", ("Static",), {})]

    def test_dependencies_empty(self) -> None:
        assert Static().dependencies() == ()


class TestTransient:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        Transient()._emit(e, tag=1)
        assert e.calls == [("analysis", ("Transient",), {})]


class TestVariableTransient:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        VariableTransient()._emit(e, tag=1)
        assert e.calls == [("analysis", ("VariableTransient",), {})]


class TestAnalysisNamespace:
    def test_static(self) -> None:
        ops = _make_ops()
        a = ops.analysis.Static()
        assert isinstance(a, Static)

    def test_transient(self) -> None:
        ops = _make_ops()
        a = ops.analysis.Transient()
        assert isinstance(a, Transient)

    def test_variable_transient(self) -> None:
        ops = _make_ops()
        a = ops.analysis.VariableTransient()
        assert isinstance(a, VariableTransient)


# ===========================================================================
# Cross-family namespace integration — distinct tags per kind
# ===========================================================================

class TestAnalysisChainTagAllocation:
    def test_each_family_has_independent_tag_counter(self) -> None:
        """Distinct kind strings keep counters independent."""
        ops = _make_ops()
        c = ops.constraints.Plain()
        n = ops.numberer.Plain()
        s = ops.system.UmfPack()
        t = ops.test.NormDispIncr(tol=1e-6, max_iter=10)
        a = ops.algorithm.Newton()
        i = ops.integrator.LoadControl(dlam=0.1)
        x = ops.analysis.Static()
        # Each family is its own kind, so each is tag 1.
        assert ops.tag_for(c) == 1
        assert ops.tag_for(n) == 1
        assert ops.tag_for(s) == 1
        assert ops.tag_for(t) == 1
        assert ops.tag_for(a) == 1
        assert ops.tag_for(i) == 1
        assert ops.tag_for(x) == 1

    def test_two_constraints_increment_within_kind(self) -> None:
        ops = _make_ops()
        c1 = ops.constraints.Plain()
        c2 = ops.constraints.Transformation()
        assert ops.tag_for(c1) == 1
        assert ops.tag_for(c2) == 2


# ===========================================================================
# Ladruno fork — analysis cluster (integrators / handler / convergence test)
# ===========================================================================

class TestLadrunoHHT:
    def test_emit_alpha_only(self) -> None:
        e = RecordingEmitter()
        LadrunoHHT(alpha=-0.1)._emit(e, tag=1)
        assert e.calls == [("integrator", ("LadrunoHHT", -0.1), {})]

    def test_emit_with_gamma_beta(self) -> None:
        e = RecordingEmitter()
        LadrunoHHT(alpha=-0.1, gamma=0.6, beta=0.3)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("LadrunoHHT", -0.1, 0.6, 0.3), {})
        ]

    def test_gamma_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both gamma and beta"):
            LadrunoHHT(alpha=-0.1, gamma=0.6)

    def test_dependencies_empty(self) -> None:
        assert LadrunoHHT(alpha=-0.1).dependencies() == ()

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LadrunoHHT(alpha=-0.1)
        assert isinstance(i, LadrunoHHT)
        assert ops.tag_for(i) == 1


class TestLadrunoGeneralizedAlpha:
    def test_emit_required_only(self) -> None:
        e = RecordingEmitter()
        LadrunoGeneralizedAlpha(alpha_m=0.4, alpha_f=0.2)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("LadrunoGeneralizedAlpha", 0.4, 0.2), {})
        ]

    def test_emit_with_gamma_beta(self) -> None:
        e = RecordingEmitter()
        LadrunoGeneralizedAlpha(
            alpha_m=0.4, alpha_f=0.2, gamma=0.55, beta=0.28)._emit(e, tag=1)
        assert e.calls == [
            ("integrator",
             ("LadrunoGeneralizedAlpha", 0.4, 0.2, 0.55, 0.28), {})
        ]

    def test_beta_only_raises(self) -> None:
        with pytest.raises(ValueError, match="both gamma and beta"):
            LadrunoGeneralizedAlpha(alpha_m=0.4, alpha_f=0.2, beta=0.28)

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        i = ops.integrator.LadrunoGeneralizedAlpha(alpha_m=0.4, alpha_f=0.2)
        assert isinstance(i, LadrunoGeneralizedAlpha)
        assert ops.tag_for(i) == 1


class TestLadrunoContact:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        LadrunoContact()._emit(e, tag=1)
        assert e.calls == [("constraints", ("LadrunoContact",), {})]

    def test_dependencies_empty(self) -> None:
        assert LadrunoContact().dependencies() == ()

    def test_repr_has_class_name(self) -> None:
        assert "LadrunoContact" in repr(LadrunoContact())

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        c = ops.constraints.LadrunoContact()
        assert isinstance(c, LadrunoContact)
        assert ops.tag_for(c) == 1


class TestLadrunoStabilizedUnbalance:
    def test_emit(self) -> None:
        e = RecordingEmitter()
        LadrunoStabilizedUnbalance(tol=1e-6, max_iter=25)._emit(e, tag=1)
        assert e.calls == [
            ("test", ("LadrunoStabilizedUnbalance", 1e-6, 25, 0, 2), {})
        ]

    def test_emit_with_flags(self) -> None:
        e = RecordingEmitter()
        LadrunoStabilizedUnbalance(
            tol=1e-6, max_iter=25, print_flag=2, norm_type=1)._emit(e, tag=1)
        assert e.calls[0][1] == (
            "LadrunoStabilizedUnbalance", 1e-6, 25, 2, 1)

    def test_zero_tol_raises(self) -> None:
        with pytest.raises(ValueError, match="tol must be > 0"):
            LadrunoStabilizedUnbalance(tol=0.0, max_iter=25)

    def test_bad_max_iter_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            LadrunoStabilizedUnbalance(tol=1e-6, max_iter=0)

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        t = ops.test.LadrunoStabilizedUnbalance(tol=1e-6, max_iter=25)
        assert isinstance(t, LadrunoStabilizedUnbalance)
        assert ops.tag_for(t) == 1


# ===========================================================================
# Ladruno fork — Selective Mass Scaling (SMS) explicit integrators
# ===========================================================================

class TestCentralDifferenceSMS:
    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        CentralDifferenceSMS(dt_target=1e-3)._emit(e, tag=1)
        assert e.calls == [("integrator", ("CentralDifferenceSMS", 1e-3), {})]

    def test_emit_options(self) -> None:
        e = RecordingEmitter()
        CentralDifferenceSMS(
            dt_target=1e-3, max_added_mass=0.1, lump="hrz",
            tangent=True, cfl=True, verbose=True)._emit(e, tag=1)
        assert e.calls[0][1] == (
            "CentralDifferenceSMS", 1e-3, "-maxAddedMass", 0.1,
            "-verbose", "-lump", "hrz", "-tangent", "-cfl",
        )

    def test_emit_consistent_switches_token_and_pcg(self) -> None:
        e = RecordingEmitter()
        CentralDifferenceSMS(
            dt_target=1e-3, consistent=True,
            pcg_tol=1e-8, pcg_max_it=300)._emit(e, tag=1)
        assert e.calls[0][1] == (
            "CentralDifferenceSMSConsistent", 1e-3,
            "-pcgTol", 1e-8, "-pcgMaxIt", 300,
        )

    def test_dt_target_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="dt_target must be > 0"):
            CentralDifferenceSMS(dt_target=0.0)

    def test_pcg_requires_consistent(self) -> None:
        with pytest.raises(ValueError, match="require consistent=True"):
            CentralDifferenceSMS(dt_target=1e-3, pcg_tol=1e-8)

    def test_bad_lump(self) -> None:
        with pytest.raises(ValueError, match="lump must be"):
            CentralDifferenceSMS(dt_target=1e-3, lump="bogus")  # type: ignore[arg-type]

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        i = ops.integrator.CentralDifferenceSMS(dt_target=1e-3)
        assert isinstance(i, CentralDifferenceSMS)
        assert ops.tag_for(i) == 1


class TestExplicitBatheSMS:
    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheSMS(p=0.54, dt_target=2e-4)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("ExplicitBatheSMS", 0.54, 2e-4), {})
        ]

    def test_emit_consistent(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheSMS(
            p=0.54, dt_target=2e-4, consistent=True, pcg_max_it=150)._emit(e, tag=1)
        assert e.calls[0][1] == (
            "ExplicitBatheSMSConsistent", 0.54, 2e-4, "-pcgMaxIt", 150)

    def test_bad_p(self) -> None:
        with pytest.raises(ValueError, match="p must be in"):
            ExplicitBatheSMS(p=1.5, dt_target=2e-4)

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ExplicitBatheSMS(dt_target=2e-4)
        assert isinstance(i, ExplicitBatheSMS)
        assert ops.tag_for(i) == 1


class TestExplicitBatheLNVDSMS:
    def test_emit_minimal(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheLNVDSMS(p=0.54, alpha=0.8, dt_target=2e-4)._emit(e, tag=1)
        assert e.calls == [
            ("integrator", ("ExplicitBatheLNVDSMS", 0.54, 0.8, 2e-4), {})
        ]

    def test_emit_consistent_token(self) -> None:
        e = RecordingEmitter()
        ExplicitBatheLNVDSMS(
            p=0.54, alpha=0.8, dt_target=2e-4, consistent=True)._emit(e, tag=1)
        assert e.calls[0][1][0] == "ExplicitBatheLNVDSMSConsistent"

    def test_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha .* must be in"):
            ExplicitBatheLNVDSMS(p=0.54, alpha=1.0, dt_target=2e-4)

    def test_via_namespace(self) -> None:
        ops = _make_ops()
        i = ops.integrator.ExplicitBatheLNVDSMS(dt_target=2e-4)
        assert isinstance(i, ExplicitBatheLNVDSMS)
        assert ops.tag_for(i) == 1
