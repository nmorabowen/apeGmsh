"""Unit tests for the ``apeGmsh.opensees.material.nd`` primitives.

Each test class exercises construction, defaults, validation, ``_emit``
through a :class:`RecordingEmitter`, ``dependencies``, and ``__repr__``.
A small final TestClass exercises the bridge namespace integration so
that ``ops.nDMaterial.<Type>(...)`` is verified end-to-end at the
type-system level.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import (
    DruckerPrager,
    ElasticIsotropic,
    J2Plasticity,
    LadrunoJ2,
    LadrunoJ2Finite,
)


# ---------------------------------------------------------------------------
# ElasticIsotropic
# ---------------------------------------------------------------------------

class TestElasticIsotropic:
    def test_construction(self) -> None:
        m = ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
        assert m.E == 30e9
        assert m.nu == 0.2
        assert m.rho == 2400.0

    def test_default_rho_is_zero(self) -> None:
        m = ElasticIsotropic(E=30e9, nu=0.2)
        assert m.rho == 0.0

    def test_emit_records_correct_call(self) -> None:
        m = ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
        emitter = RecordingEmitter()
        m._emit(emitter, tag=7)
        assert emitter.calls == [
            ("nDMaterial", ("ElasticIsotropic", 7, 30e9, 0.2, 2400.0), {})
        ]

    def test_emit_with_default_rho(self) -> None:
        m = ElasticIsotropic(E=200e9, nu=0.3)
        emitter = RecordingEmitter()
        m._emit(emitter, tag=1)
        assert emitter.calls == [
            ("nDMaterial", ("ElasticIsotropic", 1, 200e9, 0.3, 0.0), {})
        ]

    def test_dependencies_is_empty_for_leaf(self) -> None:
        m = ElasticIsotropic(E=30e9, nu=0.2)
        assert m.dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        m = ElasticIsotropic(E=30e9, nu=0.2)
        assert "ElasticIsotropic" in repr(m)

    @pytest.mark.parametrize("bad_E", [0.0, -1.0, -1e9])
    def test_validation_rejects_non_positive_E(self, bad_E: float) -> None:
        with pytest.raises(ValueError, match="E must be > 0"):
            ElasticIsotropic(E=bad_E, nu=0.2)

    @pytest.mark.parametrize("bad_nu", [-0.01, 0.5, 0.6, 1.0])
    def test_validation_rejects_out_of_range_nu(self, bad_nu: float) -> None:
        with pytest.raises(ValueError, match="nu must be in"):
            ElasticIsotropic(E=30e9, nu=bad_nu)

    def test_validation_rejects_negative_rho(self) -> None:
        with pytest.raises(ValueError, match="rho must be >= 0"):
            ElasticIsotropic(E=30e9, nu=0.2, rho=-1.0)

    def test_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        m = ElasticIsotropic(E=30e9, nu=0.2)
        with pytest.raises(FrozenInstanceError):
            m.E = 40e9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# J2Plasticity
# ---------------------------------------------------------------------------

class TestJ2Plasticity:
    @staticmethod
    def _ok_kwargs() -> dict[str, float]:
        return {
            "K": 1.65e8,
            "G": 7.5e7,
            "sig0": 5.0e5,
            "sigInf": 7.0e5,
            "delta": 0.1,
            "H": 1.0e6,
        }

    def test_construction(self) -> None:
        m = J2Plasticity(**self._ok_kwargs())
        assert m.K == 1.65e8
        assert m.G == 7.5e7
        assert m.sig0 == 5.0e5
        assert m.sigInf == 7.0e5
        assert m.delta == 0.1
        assert m.H == 1.0e6

    def test_default_eta_is_zero(self) -> None:
        m = J2Plasticity(**self._ok_kwargs())
        assert m.eta == 0.0

    def test_emit_records_correct_call(self) -> None:
        m = J2Plasticity(**self._ok_kwargs(), eta=0.05)
        emitter = RecordingEmitter()
        m._emit(emitter, tag=3)
        assert emitter.calls == [
            (
                "nDMaterial",
                (
                    "J2Plasticity",
                    3,
                    1.65e8,
                    7.5e7,
                    5.0e5,
                    7.0e5,
                    0.1,
                    1.0e6,
                    0.05,
                ),
                {},
            )
        ]

    def test_emit_with_default_eta(self) -> None:
        m = J2Plasticity(**self._ok_kwargs())
        emitter = RecordingEmitter()
        m._emit(emitter, tag=11)
        # Last param (eta) defaults to 0.0
        assert emitter.calls[0][1][-1] == 0.0

    def test_dependencies_is_empty_for_leaf(self) -> None:
        m = J2Plasticity(**self._ok_kwargs())
        assert m.dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        m = J2Plasticity(**self._ok_kwargs())
        assert "J2Plasticity" in repr(m)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_K(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["K"] = bad
        with pytest.raises(ValueError, match="K must be > 0"):
            J2Plasticity(**kwargs)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_G(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["G"] = bad
        with pytest.raises(ValueError, match="G must be > 0"):
            J2Plasticity(**kwargs)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_sig0(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["sig0"] = bad
        with pytest.raises(ValueError, match="sig0 must be > 0"):
            J2Plasticity(**kwargs)

    def test_validation_rejects_negative_delta(self) -> None:
        kwargs = self._ok_kwargs()
        kwargs["delta"] = -1.0
        with pytest.raises(ValueError, match="delta must be >= 0"):
            J2Plasticity(**kwargs)

    def test_validation_rejects_negative_H(self) -> None:
        kwargs = self._ok_kwargs()
        kwargs["H"] = -1.0
        with pytest.raises(ValueError, match="H must be >= 0"):
            J2Plasticity(**kwargs)

    def test_validation_rejects_negative_eta(self) -> None:
        with pytest.raises(ValueError, match="eta must be >= 0"):
            J2Plasticity(**self._ok_kwargs(), eta=-0.1)


# ---------------------------------------------------------------------------
# DruckerPrager
# ---------------------------------------------------------------------------

class TestDruckerPrager:
    @staticmethod
    def _ok_kwargs() -> dict[str, float]:
        return {
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
        }

    def test_construction(self) -> None:
        m = DruckerPrager(**self._ok_kwargs())
        assert m.K == 80.0e6
        assert m.G == 60.0e6
        assert m.sigmaY == 20.0e3
        assert m.theta == 1.0

    def test_emit_records_correct_call(self) -> None:
        m = DruckerPrager(**self._ok_kwargs())
        emitter = RecordingEmitter()
        m._emit(emitter, tag=42)
        assert emitter.calls == [
            (
                "nDMaterial",
                (
                    "DruckerPrager",
                    42,
                    80.0e6,
                    60.0e6,
                    20.0e3,
                    0.0,  # rho
                    0.0,  # rhoBar
                    0.0,  # Kinf
                    0.0,  # Ko
                    0.0,  # delta1
                    0.0,  # delta2
                    0.0,  # H
                    1.0,  # theta
                ),
                {},
            )
        ]

    def test_dependencies_is_empty_for_leaf(self) -> None:
        m = DruckerPrager(**self._ok_kwargs())
        assert m.dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        m = DruckerPrager(**self._ok_kwargs())
        assert "DruckerPrager" in repr(m)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_K(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["K"] = bad
        with pytest.raises(ValueError, match="K must be > 0"):
            DruckerPrager(**kwargs)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_G(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["G"] = bad
        with pytest.raises(ValueError, match="G must be > 0"):
            DruckerPrager(**kwargs)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_validation_rejects_non_positive_sigmaY(self, bad: float) -> None:
        kwargs = self._ok_kwargs()
        kwargs["sigmaY"] = bad
        with pytest.raises(ValueError, match="sigmaY must be > 0"):
            DruckerPrager(**kwargs)

    @pytest.mark.parametrize(
        "field",
        ["rho", "rhoBar", "Kinf", "Ko", "delta1", "delta2", "H"],
    )
    def test_validation_rejects_negative_nonneg_field(
        self, field: str
    ) -> None:
        kwargs = self._ok_kwargs()
        kwargs[field] = -1.0
        with pytest.raises(ValueError, match=f"{field} must be >= 0"):
            DruckerPrager(**kwargs)

    @pytest.mark.parametrize("bad_theta", [-0.1, 1.1, 2.0])
    def test_validation_rejects_out_of_range_theta(
        self, bad_theta: float
    ) -> None:
        kwargs = self._ok_kwargs()
        kwargs["theta"] = bad_theta
        with pytest.raises(ValueError, match="theta must be in"):
            DruckerPrager(**kwargs)


# ---------------------------------------------------------------------------
# LadrunoJ2 (Ladruno fork — combined Voce+Chaboche von Mises, ND 33011)
# ---------------------------------------------------------------------------

class TestLadrunoJ2:
    def test_construction_defaults(self) -> None:
        m = LadrunoJ2(K=1.65e8, G=7.5e7, sig0=5.0e5)
        assert (m.K, m.G, m.sig0) == (1.65e8, 7.5e7, 5.0e5)
        assert (m.Qinf, m.b, m.Hiso) == (0.0, 0.0, 0.0)
        assert m.backstresses == ()
        assert m.rho == 0.0 and m.lch_ref is None
        assert m.damage is None and m.implex is False

    def test_emit_bare_isotropic(self) -> None:
        rec = RecordingEmitter()
        LadrunoJ2(K=1.65e8, G=7.5e7, sig0=5.0e5)._emit(rec, tag=5)
        assert rec.calls == [
            ("nDMaterial",
             ("LadrunoJ2", 5, 1.65e8, 7.5e7,
              "-iso", "voce", 5.0e5, 0.0, 0.0, 0.0), {}),
        ]

    def test_emit_with_voce_and_kinematic(self) -> None:
        rec = RecordingEmitter()
        LadrunoJ2(
            K=1.65e8, G=7.5e7, sig0=450.0, Qinf=265.0, b=16.93, Hiso=129.24,
            backstresses=[(20000.0, 200.0), (5000.0, 50.0)],
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoJ2", 2, 1.65e8, 7.5e7,
            "-iso", "voce", 450.0, 265.0, 16.93, 129.24,
            "-kin", 2, 20000.0, 200.0, 5000.0, 50.0,
        )

    def test_emit_with_rho_damage_implex(self) -> None:
        rec = RecordingEmitter()
        LadrunoJ2(
            K=1.65e8, G=7.5e7, sig0=450.0,
            rho=7.85e3, lch_ref=0.05,
            damage=(1.0, 1.0, 0.0, 0.99), implex=True,
        )._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "LadrunoJ2", 1, 1.65e8, 7.5e7,
            "-iso", "voce", 450.0, 0.0, 0.0, 0.0,
            "-rho", 7.85e3,
            "-autoRegularization", 0.05,
            "-damage", "lemaitre", 1.0, 1.0, 0.0, 0.99,
            "-implex",
        )

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoJ2(K=1.65e8, G=7.5e7, sig0=5e5).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoJ2" in repr(LadrunoJ2(K=1e8, G=5e7, sig0=5e5))

    def test_rejects_non_positive_sig0(self) -> None:
        with pytest.raises(ValueError, match="sig0 must be > 0"):
            LadrunoJ2(K=1e8, G=5e7, sig0=0.0)

    def test_rejects_too_many_backstresses(self) -> None:
        with pytest.raises(ValueError, match="at most 8"):
            LadrunoJ2(K=1e8, G=5e7, sig0=5e5,
                      backstresses=[(1.0, 1.0)] * 9)

    def test_rejects_non_positive_backstress_modulus(self) -> None:
        with pytest.raises(ValueError, match="modulus C must be > 0"):
            LadrunoJ2(K=1e8, G=5e7, sig0=5e5, backstresses=[(0.0, 1.0)])

    def test_rejects_bad_lemaitre_Dc(self) -> None:
        with pytest.raises(ValueError, match="Dc .* must be in"):
            LadrunoJ2(K=1e8, G=5e7, sig0=5e5, damage=(1.0, 1.0, 0.0, 1.5))

    def test_rejects_bad_lch_ref(self) -> None:
        with pytest.raises(ValueError, match="lch_ref must be > 0"):
            LadrunoJ2(K=1e8, G=5e7, sig0=5e5, lch_ref=0.0)


# ---------------------------------------------------------------------------
# LadrunoJ2Finite (Ladruno fork — finite-strain-native combined J2, ND 33012)
# ---------------------------------------------------------------------------

class TestLadrunoJ2Finite:
    def test_emit_bare(self) -> None:
        rec = RecordingEmitter()
        LadrunoJ2Finite(K=1.65e8, G=7.5e7, sig0=450.0)._emit(rec, tag=9)
        assert rec.calls == [
            ("nDMaterial",
             ("LadrunoJ2Finite", 9, 1.65e8, 7.5e7,
              "-iso", "voce", 450.0, 0.0, 0.0, 0.0), {}),
        ]

    def test_emit_with_kin_rho_implex(self) -> None:
        rec = RecordingEmitter()
        LadrunoJ2Finite(
            K=1.65e8, G=7.5e7, sig0=450.0,
            backstresses=[(20000.0, 200.0)], rho=7.85e3, implex=True,
        )._emit(rec, tag=3)
        assert rec.calls[0][1] == (
            "LadrunoJ2Finite", 3, 1.65e8, 7.5e7,
            "-iso", "voce", 450.0, 0.0, 0.0, 0.0,
            "-kin", 1, 20000.0, 200.0,
            "-rho", 7.85e3,
            "-implex",
        )

    def test_has_no_damage_field(self) -> None:
        # The finite material has no Lemaitre damage (parser rejects it).
        assert not hasattr(LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5), "damage")

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5).dependencies() == ()

    def test_rejects_non_positive_G(self) -> None:
        with pytest.raises(ValueError, match="G must be > 0"):
            LadrunoJ2Finite(K=1e8, G=0.0, sig0=5e5)


# ---------------------------------------------------------------------------
# Cross-cutting: namespace-level integration with the bridge
# ---------------------------------------------------------------------------

def _stub_bridge() -> apeSees:
    """Construct an :class:`apeSees` with a MagicMock FEM stand-in."""
    return apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]


class TestNDMaterialNamespace:
    """Verify the typed namespace methods register and tag correctly."""

    def test_ElasticIsotropic_via_namespace_returns_typed_instance(
        self,
    ) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400.0)
        assert isinstance(m, ElasticIsotropic)
        assert ops.tag_for(m) == 1

    def test_J2Plasticity_via_namespace_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.J2Plasticity(
            K=1.65e8, G=7.5e7,
            sig0=5.0e5, sigInf=7.0e5,
            delta=0.1, H=1.0e6,
        )
        assert isinstance(m, J2Plasticity)
        assert ops.tag_for(m) == 1

    def test_DruckerPrager_via_namespace_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.DruckerPrager(
            K=80e6, G=60e6, sigmaY=20e3,
            rho=0.0, rhoBar=0.0,
            Kinf=0.0, Ko=0.0,
            delta1=0.0, delta2=0.0,
            H=0.0, theta=1.0,
        )
        assert isinstance(m, DruckerPrager)
        assert ops.tag_for(m) == 1

    def test_distinct_nd_materials_get_distinct_tags(self) -> None:
        ops = _stub_bridge()
        m1 = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        m2 = ops.nDMaterial.ElasticIsotropic(E=200e9, nu=0.3)
        assert ops.tag_for(m1) == 1
        assert ops.tag_for(m2) == 2

    def test_LadrunoJ2_via_namespace_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.LadrunoJ2(
            K=1.65e8, G=7.5e7, sig0=450.0, Qinf=265.0, b=16.93, Hiso=129.24,
            backstresses=[(20000.0, 200.0)],
        )
        assert isinstance(m, LadrunoJ2)
        assert m.backstresses == ((20000.0, 200.0),)
        assert ops.tag_for(m) == 1

    def test_LadrunoJ2Finite_via_namespace_returns_typed_instance(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.LadrunoJ2Finite(K=1.65e8, G=7.5e7, sig0=450.0)
        assert isinstance(m, LadrunoJ2Finite)
        assert ops.tag_for(m) == 1
