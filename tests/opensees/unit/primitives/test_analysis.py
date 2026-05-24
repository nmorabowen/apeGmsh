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
        (UmfPack, "UmfPack"),
        (Mumps, "Mumps"),
        (SparseGeneral, "SparseGeneral"),
        (FullGeneral, "FullGeneral"),
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
        BandGeneral, BandSPD, ProfileSPD, UmfPack,
        Mumps, SparseGeneral, FullGeneral,
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
