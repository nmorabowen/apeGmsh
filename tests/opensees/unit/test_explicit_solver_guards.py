"""Unit tests for the explicit integrator / system / mass compatibility guards.

These exercise ``apeSees._check_explicit_solver_compat`` and
``_warn_if_unguarded_explicit_run`` directly (no openseespy / gmsh) — they
only scan the registered primitives. From the explicit-dynamics design review:

* consistent element mass (``c_mass=True``) + ``Diagonal`` is silently WRONG
  (off-diagonal mass dropped) -> hard error;
* explicit integrator + non-diagonal system is correct-but-slow -> warning;
* ``analyze_explicit`` on an unguarded explicit integrator -> warning.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import OpenSeesExplicitSolverWarning


def _make_ops() -> apeSees:
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


def _truss(ops: apeSees, *, c_mass: bool) -> None:
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e9)
    ops.element.Truss(pg="Bar", A=0.01, material=mat, rho=8000.0, c_mass=c_mass)


# ---------------------------------------------------------------------------
# c_mass + Diagonal -> hard error
# ---------------------------------------------------------------------------

class TestConsistentMassDiagonalRaises:
    def test_consistent_mass_plus_diagonal_raises(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=True)
        ops.system.Diagonal()
        with pytest.raises(BridgeError, match="consistent"):
            ops._check_explicit_solver_compat()

    def test_consistent_mass_plus_mpidiagonal_raises(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=True)
        ops.system.MPIDiagonal()
        with pytest.raises(BridgeError, match="DIAGONAL"):
            ops._check_explicit_solver_compat()

    def test_lumped_mass_plus_diagonal_ok(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=False)
        ops.system.Diagonal()
        ops._check_explicit_solver_compat()  # no raise

    def test_consistent_mass_plus_nondiagonal_ok(self) -> None:
        # Off-diagonal mass is preserved by a banded solver — no error.
        ops = _make_ops()
        _truss(ops, c_mass=True)
        ops.system.ProfileSPD()
        ops._check_explicit_solver_compat()  # no raise


# ---------------------------------------------------------------------------
# explicit integrator + non-diagonal system -> warning (correct but slow)
# ---------------------------------------------------------------------------

class TestExplicitNonDiagonalWarns:
    def test_explicit_plus_banded_warns(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=False)
        ops.system.BandGeneral()
        ops.integrator.ExplicitBathe(p=0.54, cfl=True)
        with pytest.warns(OpenSeesExplicitSolverWarning, match="O\\(N\\)"):
            ops._check_explicit_solver_compat()

    def test_explicit_plus_diagonal_no_warn(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=False)
        ops.system.Diagonal()
        ops.integrator.ExplicitBathe(p=0.54, cfl=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenSeesExplicitSolverWarning)
            ops._check_explicit_solver_compat()  # must not warn

    def test_implicit_plus_banded_no_warn(self) -> None:
        ops = _make_ops()
        _truss(ops, c_mass=False)
        ops.system.BandGeneral()
        ops.integrator.Newmark(gamma=0.5, beta=0.25)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenSeesExplicitSolverWarning)
            ops._check_explicit_solver_compat()  # not explicit -> no warn


# ---------------------------------------------------------------------------
# unguarded explicit run warning
# ---------------------------------------------------------------------------

class TestUnguardedExplicitWarn:
    def test_no_cfl_abort_no_recompute_warns(self) -> None:
        ops = _make_ops()
        ops.integrator.ExplicitBathe(p=0.54, cfl=True)
        with pytest.warns(OpenSeesExplicitSolverWarning, match="stiffening"):
            ops._warn_if_unguarded_explicit_run()

    def test_cfl_abort_set_no_warn(self) -> None:
        ops = _make_ops()
        ops.integrator.ExplicitBathe(p=0.54, cfl=True, cfl_abort=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenSeesExplicitSolverWarning)
            ops._warn_if_unguarded_explicit_run()

    def test_recompute_set_no_warn(self) -> None:
        ops = _make_ops()
        ops.integrator.ExplicitBathe(p=0.54, cfl=True, recompute=10)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", OpenSeesExplicitSolverWarning)
            ops._warn_if_unguarded_explicit_run()
