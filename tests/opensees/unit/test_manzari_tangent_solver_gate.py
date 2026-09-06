"""The Manzari-family consistent-tangent / symmetric-solver gate.

``TanType != 0`` selects the continuum elasto-plastic tangent, which for a
non-associated model is genuinely UNSYMMETRIC.  A half-storage solver reads
only the ``col >= row`` half of each element matrix and converges to a
plausible but wrong answer, so
:func:`validate_manzari_tangent_solver` warns.

Two reasons it fires without anyone touching a deck:
``LadrunoSANISAND`` now defaults to ``tan_type=2``, and the fork parser's
own ``$TanType`` default moved ``0 -> 2`` in fork PR #792 (vanilla
``ManzariDafalias`` stayed at ``0``).

Scope mirrors the ADR-0074 D4 u-p gate exactly — see
``test_ladruno_up_build_gates.py::TestSolverGate`` — but fail-SOFT.
"""
from __future__ import annotations

import warnings

import pytest

from apeGmsh.opensees._internal.build import (
    ManzariTangentSolverWarning,
    validate_manzari_tangent_solver,
)
from apeGmsh.opensees.material.nd import (
    ElasticIsotropic,
    LadrunoSANISAND,
    ManzariDafalias,
)


_LS_KWARGS = {
    "G0": 264.32, "nu": 0.3129, "e_init": 0.6944, "Mc": 1.33090, "c": 0.71,
    "lambda_c": 0.027, "e0": 0.83, "ksi": 0.45, "P_atm": 101.0, "m": 0.005,
    "h0": 1.3, "Ch": 0.968, "nb": 3.5, "A0": 0.05, "nd": 5.75,
    "z_max": 12.5, "cz": 1100.0, "rho": 2.0,
}

SYMMETRIC = [
    "ProfileSPD", "SProfileSPD", "ParallelProfileSPD", "BandSPD",
    "SparseSYM", "Diagonal", "MPIDiagonal",
]
GENERAL = [
    "UmfPack", "SparseGeneral", "FullGeneral", "BandGeneral", "Mumps",
    "Pardiso",
]


def _sys(name: str, **kw):
    import apeGmsh.opensees.analysis.system as sysmod

    return getattr(sysmod, name)(**kw)


def _sanisand(**kw):
    return LadrunoSANISAND(**_LS_KWARGS, **kw)


def _flat(materials, systems, *, enforce=True, partitioned=False):
    validate_manzari_tangent_solver(
        materials, enforce=enforce, staged=False, partitioned=partitioned,
        flat_systems=systems, stage_systems=[],
    )


def _staged(materials, stage_systems, *, enforce=True):
    validate_manzari_tangent_solver(
        materials, enforce=enforce, staged=True, partitioned=False,
        flat_systems=[], stage_systems=stage_systems,
    )


def _silent(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ManzariTangentSolverWarning)
        fn(*a, **kw)


# ── who trips the gate ───────────────────────────────────────────────────────

class TestTrigger:
    def test_default_ladruno_sanisand_trips_it(self) -> None:
        """No one asked for tan_type — the new default is 2."""
        with pytest.warns(ManzariTangentSolverWarning, match="UNSYMMETRIC"):
            _flat([_sanisand()], [_sys("ProfileSPD")])

    def test_elastic_tangent_is_silent(self) -> None:
        _silent(_flat, [_sanisand(tan_type=0)], [_sys("ProfileSPD")])

    def test_vanilla_manzari_default_is_silent(self) -> None:
        """ManzariDafalias keeps vanilla's tan_type=0."""
        md = ManzariDafalias(**_LS_KWARGS)
        assert md.tan_type == 0
        _silent(_flat, [md], [_sys("ProfileSPD")])

    def test_manzari_with_consistent_tangent_trips_it(self) -> None:
        with pytest.warns(ManzariTangentSolverWarning, match="ManzariDafalias"):
            _flat(
                [ManzariDafalias(**_LS_KWARGS, tan_type=2)],
                [_sys("ProfileSPD")],
            )

    def test_unrelated_material_is_silent(self) -> None:
        _silent(_flat, [ElasticIsotropic(E=1e4, nu=0.3)], [_sys("ProfileSPD")])

    def test_no_materials_is_silent(self) -> None:
        _silent(_flat, [], [_sys("ProfileSPD")])

    def test_message_names_the_material_and_its_tan_type(self) -> None:
        with pytest.warns(ManzariTangentSolverWarning) as rec:
            _flat([_sanisand()], [_sys("ProfileSPD")])
        assert "LadrunoSANISAND(tan_type=2)" in str(rec[0].message)


# ── which systems are safe ───────────────────────────────────────────────────

class TestSolvers:
    @pytest.mark.parametrize("sys_name", SYMMETRIC)
    def test_symmetric_or_diagonal_warns(self, sys_name: str) -> None:
        with pytest.warns(ManzariTangentSolverWarning, match="UNSYMMETRIC"):
            _flat([_sanisand()], [_sys(sys_name)])

    @pytest.mark.parametrize("sys_name", GENERAL)
    def test_general_is_silent(self, sys_name: str) -> None:
        _silent(_flat, [_sanisand()], [_sys(sys_name)])

    @pytest.mark.parametrize("cls_name", ["Pardiso", "Mumps"])
    @pytest.mark.parametrize("mtype", ["symmetric", "spd"])
    def test_half_storage_mode_warns(
        self, cls_name: str, mtype: str
    ) -> None:
        with pytest.warns(
            ManzariTangentSolverWarning, match="upper triangle"
        ):
            _flat([_sanisand()], [_sys(cls_name, matrix_type=mtype)])

    def test_last_declared_system_is_the_effective_one(self) -> None:
        _silent(
            _flat, [_sanisand()],
            [_sys("ProfileSPD"), _sys("UmfPack")],
        )


# ── the missing-system branch (OpenSees defaults to ProfileSPD) ─────────────

class TestMissingSystem:
    def test_no_system_on_a_solving_deck_warns(self) -> None:
        with pytest.warns(ManzariTangentSolverWarning, match="ProfileSPD"):
            _flat([_sanisand()], [])

    def test_no_system_without_an_analysis_chain_is_silent(self) -> None:
        """Archival / eigen-only / model-only skeleton never solves."""
        _silent(_flat, [_sanisand()], [], enforce=False)

    def test_no_system_on_a_partitioned_deck_is_silent(self) -> None:
        """ADR-0027 INV-5 auto-emits a general Mumps/UmfPack fallback."""
        _silent(_flat, [_sanisand()], [], partitioned=True)

    def test_declared_symmetric_warns_even_without_a_chain(self) -> None:
        """Declaring the wrong solver is unambiguously a mistake."""
        with pytest.warns(ManzariTangentSolverWarning):
            _flat([_sanisand()], [_sys("ProfileSPD")], enforce=False)


# ── staged decks: each stage owns its chain (wipeAnalysis re-defaults) ──────

class TestStaged:
    def test_per_stage_symmetric_warns_naming_the_stage(self) -> None:
        with pytest.warns(ManzariTangentSolverWarning, match="consolidate"):
            _staged([_sanisand()], [("'consolidate'", _sys("ProfileSPD"))])

    def test_per_stage_general_is_silent(self) -> None:
        _staged([_sanisand()], [("'push'", _sys("UmfPack"))])

    def test_undeclared_stage_warns(self) -> None:
        with pytest.warns(ManzariTangentSolverWarning, match="wipeAnalysis"):
            _staged([_sanisand()], [("'push'", None)])

    def test_undeclared_stage_silent_without_a_chain(self) -> None:
        _silent(_staged, [_sanisand()], [("'push'", None)], enforce=False)

    def test_global_system_is_ignored_in_staged_mode(self) -> None:
        """A globally-registered system is never emitted in staged mode."""
        _silent(
            validate_manzari_tangent_solver, [_sanisand()],
            enforce=True, staged=True, partitioned=False,
            flat_systems=[_sys("ProfileSPD")],
            stage_systems=[("'push'", _sys("UmfPack"))],
        )
