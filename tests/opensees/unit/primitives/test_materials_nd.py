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
from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import (
    DruckerPrager,
    ElasticIsotropic,
    InitDefGrad,
    J2Plasticity,
    LadrunoCohesiveHingeBiaxial,
    LadrunoConcrete3D,
    LadrunoJ2,
    LadrunoJ2Finite,
    LadrunoRCConcrete,
    LadrunoRCFiniteStrain,
    LogStrain,
    StagedStrain,
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
# LogStrain (Ladruno fork — Hencky finite-strain lift wrapper, ND 33010)
# ---------------------------------------------------------------------------

class TestLogStrain:
    def test_dependencies_is_the_inner(self) -> None:
        inner = ElasticIsotropic(E=30e9, nu=0.2)
        assert LogStrain(inner=inner).dependencies() == (inner,)

    def test_emit_resolves_inner_tag(self) -> None:
        inner = ElasticIsotropic(E=30e9, nu=0.2)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 7 if p is inner else 0)
        LogStrain(inner=inner)._emit(rec, tag=12)
        assert rec.calls == [("nDMaterial", ("LogStrain", 12, 7), {})]

    def test_rejects_non_ndmaterial_inner(self) -> None:
        with pytest.raises(TypeError, match="inner must be an NDMaterial"):
            LogStrain(inner=object())  # type: ignore[arg-type]

    def test_repr_includes_type_token(self) -> None:
        assert "LogStrain" in repr(LogStrain(inner=ElasticIsotropic(E=1e9, nu=0.2)))


# ---------------------------------------------------------------------------
# InitDefGrad (Ladruno fork — finite staged stress-free birth, ND 33013)
# ---------------------------------------------------------------------------

class TestInitDefGrad:
    def test_emit_bare(self) -> None:
        inner = LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 3 if p is inner else 0)
        InitDefGrad(inner=inner)._emit(rec, tag=8)
        assert rec.calls == [("nDMaterial", ("InitDefGrad", 8, 3), {})]

    def test_emit_with_noinitf_and_F0(self) -> None:
        inner = LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 3 if p is inner else 0)
        F0 = (1.01, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        InitDefGrad(inner=inner, no_init_f=True, F0=F0)._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "InitDefGrad", 1, 3, "-noInitF", "-F0", *F0,
        )

    def test_dependencies_is_the_inner(self) -> None:
        inner = LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5)
        assert InitDefGrad(inner=inner).dependencies() == (inner,)

    def test_rejects_bad_F0_length(self) -> None:
        inner = LadrunoJ2Finite(K=1e8, G=5e7, sig0=5e5)
        with pytest.raises(ValueError, match="F0 must have 9 row-major"):
            InitDefGrad(inner=inner, F0=(1.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# StagedStrain (Ladruno fork — small-strain staged stress-free birth, ND 33014)
# ---------------------------------------------------------------------------

class TestStagedStrain:
    def test_emit_bare(self) -> None:
        inner = ElasticIsotropic(E=30e9, nu=0.2)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 5 if p is inner else 0)
        StagedStrain(inner=inner)._emit(rec, tag=9)
        assert rec.calls == [("nDMaterial", ("StagedStrain", 9, 5), {})]

    def test_emit_eps0_is_last_flag(self) -> None:
        inner = ElasticIsotropic(E=30e9, nu=0.2)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 5 if p is inner else 0)
        eps0 = (1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
        StagedStrain(inner=inner, no_init=True, eps0=eps0)._emit(rec, tag=2)
        assert rec.calls[0][1] == ("StagedStrain", 2, 5, "-noInit", "-eps0", *eps0)

    def test_rejects_bad_eps0_length(self) -> None:
        inner = ElasticIsotropic(E=30e9, nu=0.2)
        with pytest.raises(ValueError, match="eps0 must have 6 Voigt"):
            StagedStrain(inner=inner, eps0=(1e-3, 0.0, 0.0))


# ---------------------------------------------------------------------------
# LadrunoConcrete3D (Ladruno fork — CDPM2-grade solid concrete, ND 33017)
# ---------------------------------------------------------------------------

class TestLadrunoConcrete3D:
    def _base(self, **kw: object) -> LadrunoConcrete3D:
        params: dict[str, object] = dict(
            E=30e9, nu=0.2, fc=30e6, ft=3e6, Gf=120.0, Gc=12000.0)
        params.update(kw)
        return LadrunoConcrete3D(**params)  # type: ignore[arg-type]

    def test_construction_defaults(self) -> None:
        m = self._base()
        assert (m.fc, m.ft, m.Gf, m.Gc) == (30e6, 3e6, 120.0, 12000.0)
        assert m.e is None and m.kupfer == 1.16
        assert m.Df == 1.0 and m.As == 2.0 and m.rho == 0.0
        assert m.ct_temper == "none" and m.auto_regularize is False

    def test_emit_minimal_is_just_required(self) -> None:
        rec = RecordingEmitter()
        self._base()._emit(rec, tag=1)
        assert rec.calls == [
            ("nDMaterial",
             ("LadrunoConcrete3D", 1, 30e9, 0.2, 30e6, 3e6, 120.0, 12000.0),
             {}),
        ]

    def test_emit_with_options_in_parser_order(self) -> None:
        rec = RecordingEmitter()
        self._base(
            e=0.7, Df=1.2, As=3.0, rho=2400.0,
            hardening=(0.4, 0.6), ductility=(0.1, 0.004, 2.5, 2e-6),
            lch=0.25, auto_regularize=True, implex=True, eta=0.5,
            ct_temper="alphat", hoop_k=1e7, hoop_fy=4e8,
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoConcrete3D", 2, 30e9, 0.2, 30e6, 3e6, 120.0, 12000.0,
            "-e", 0.7, "-Df", 1.2, "-As", 3.0, "-rho", 2400.0,
            "-hardening", 0.4, 0.6,
            "-ductility", 0.1, 0.004, 2.5, 2e-6,
            "-lch", 0.25, "-autoRegularization", "-implex", "-eta", 0.5,
            "-ctTemper", "alphat", "-hoop", 1e7, "-hoopFy", 4e8,
        )

    def test_emit_kupfer_when_non_default(self) -> None:
        rec = RecordingEmitter()
        self._base(kupfer=1.25)._emit(rec, tag=3)
        assert rec.calls[0][1][8:] == ("-kupfer", 1.25)

    def test_dependencies_is_empty(self) -> None:
        assert self._base().dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoConcrete3D" in repr(self._base())

    def test_rejects_ft_ge_fc(self) -> None:
        with pytest.raises(ValueError, match="need ft < fc"):
            self._base(ft=30e6)

    def test_rejects_non_positive_fc(self) -> None:
        with pytest.raises(ValueError, match="fc, ft must be > 0"):
            self._base(fc=0.0)

    def test_rejects_bad_eccentricity(self) -> None:
        with pytest.raises(ValueError, match=r"e must be in \(0.5, 1\]"):
            self._base(e=0.4)

    def test_rejects_e_and_kupfer_together(self) -> None:
        with pytest.raises(ValueError, match="either e or a non-default kupfer"):
            self._base(e=0.7, kupfer=1.25)

    def test_rejects_As_below_one(self) -> None:
        with pytest.raises(ValueError, match="As must be >= 1"):
            self._base(As=0.5)

    def test_rejects_bad_ct_temper(self) -> None:
        with pytest.raises(ValueError, match="ct_temper must be one of"):
            self._base(ct_temper="bogus")


# ---------------------------------------------------------------------------
# LadrunoRCConcrete / LadrunoRCFiniteStrain (Ladruno fork — RC plastic-damage)
# ---------------------------------------------------------------------------

# A minimal hand-built backbone for deterministic emit assertions.
_RC_RAW: dict[str, object] = dict(
    E=30e9, nu=0.2,
    Ce=(0.0, -1e-3), Cs=(0.0, -30e6),
    Te=(0.0, 1e-4), Ts=(0.0, 3e6),
)


class TestLadrunoRCConcrete:
    def test_emit_minimal_backbone_only(self) -> None:
        rec = RecordingEmitter()
        LadrunoRCConcrete(**_RC_RAW)._emit(rec, tag=1)  # type: ignore[arg-type]
        assert rec.calls[0][1] == (
            "LadrunoRCConcrete", 1, 30e9, 0.2,
            "-Ce", 0.0, -1e-3, "-Cs", 0.0, -30e6,
            "-Te", 0.0, 1e-4, "-Ts", 0.0, 3e6,
        )

    def test_emit_with_flags_in_order(self) -> None:
        rec = RecordingEmitter()
        LadrunoRCConcrete(
            **_RC_RAW, rho=2400.0, tangent="secant",  # type: ignore[arg-type]
            interlock=True, tens_stiff="vc", auto_regularization=0.1,
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoRCConcrete", 2, 30e9, 0.2,
            "-Ce", 0.0, -1e-3, "-Cs", 0.0, -30e6,
            "-Te", 0.0, 1e-4, "-Ts", 0.0, 3e6,
            "-rho", 2400.0, "-secant", "-interlock",
            "-tensStiff", "vc", "-autoRegularization", 0.1,
        )

    def test_emit_damage_lists_when_given(self) -> None:
        rec = RecordingEmitter()
        LadrunoRCConcrete(
            E=30e9, nu=0.2,
            Ce=(0.0, -1e-3), Cs=(0.0, -30e6), Cd=(0.0, 0.5),
            Te=(0.0, 1e-4), Ts=(0.0, 3e6), Td=(0.0, 0.3),
        )._emit(rec, tag=3)
        args = rec.calls[0][1]
        assert "-Cd" in args and "-Td" in args

    def test_from_fc_builds_backbone_and_regularizes(self) -> None:
        m = LadrunoRCConcrete.from_fc(E=30e9, nu=0.2, fc=30e6)
        assert isinstance(m, LadrunoRCConcrete)
        assert len(m.Ce) >= 2 and len(m.Te) >= 2
        assert m.auto_regularization is not None and m.auto_regularization > 0

    def test_from_fc_no_regularize(self) -> None:
        m = LadrunoRCConcrete.from_fc(E=30e9, nu=0.2, fc=30e6, regularize=False)
        assert m.auto_regularization is None

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoRCConcrete(**_RC_RAW).dependencies() == ()  # type: ignore[arg-type]

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoRCConcrete" in repr(
            LadrunoRCConcrete(**_RC_RAW))  # type: ignore[arg-type]

    def test_is_not_finite_strain(self) -> None:
        assert LadrunoRCConcrete.is_finite_strain is False

    def test_rejects_short_backbone(self) -> None:
        with pytest.raises(ValueError, match="backbone needs >= 2 points"):
            LadrunoRCConcrete(
                E=30e9, nu=0.2, Ce=(0.0,), Cs=(0.0,),
                Te=(0.0, 1e-4), Ts=(0.0, 3e6))

    def test_rejects_mismatched_backbone(self) -> None:
        with pytest.raises(ValueError, match="must share length"):
            LadrunoRCConcrete(
                E=30e9, nu=0.2, Ce=(0.0, -1e-3, -2e-3), Cs=(0.0, -30e6),
                Te=(0.0, 1e-4), Ts=(0.0, 3e6))

    def test_rejects_damage_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"damage must be in \[0, 1\)"):
            LadrunoRCConcrete(
                **_RC_RAW, Cd=(0.0, 1.5))  # type: ignore[arg-type]

    def test_rejects_bad_Kc(self) -> None:
        with pytest.raises(ValueError, match=r"Kc must be in \[2/3, 1\]"):
            LadrunoRCConcrete(**_RC_RAW, Kc=0.5)  # type: ignore[arg-type]

    def test_rejects_bad_tangent(self) -> None:
        with pytest.raises(ValueError, match="tangent must be one of"):
            LadrunoRCConcrete(**_RC_RAW, tangent="bogus")  # type: ignore[arg-type]

    def test_rejects_bad_shear_retention(self) -> None:
        with pytest.raises(ValueError, match="shear_retention must be one of"):
            LadrunoRCConcrete(
                **_RC_RAW, shear_retention="bogus")  # type: ignore[arg-type]

    def test_rejects_vc_with_non_positive_c(self) -> None:
        with pytest.raises(ValueError, match="tens_stiff_c must be > 0"):
            LadrunoRCConcrete(
                **_RC_RAW, tens_stiff="vc", tens_stiff_c=0.0)  # type: ignore[arg-type]

    def test_rejects_non_positive_auto_regularization(self) -> None:
        with pytest.raises(ValueError, match="auto_regularization .* must be > 0"):
            LadrunoRCConcrete(
                **_RC_RAW, auto_regularization=0.0)  # type: ignore[arg-type]


class TestLadrunoRCFiniteStrain:
    def test_emits_finite_token(self) -> None:
        rec = RecordingEmitter()
        LadrunoRCFiniteStrain(**_RC_RAW)._emit(rec, tag=1)  # type: ignore[arg-type]
        assert rec.calls[0][1][0] == "LadrunoRCFiniteStrain"

    def test_is_finite_strain(self) -> None:
        assert LadrunoRCFiniteStrain.is_finite_strain is True

    def test_from_fc_returns_finite_subclass(self) -> None:
        m = LadrunoRCFiniteStrain.from_fc(E=30e9, nu=0.2, fc=30e6)
        assert isinstance(m, LadrunoRCFiniteStrain)
        assert m._type == "LadrunoRCFiniteStrain"

    def test_shares_validation_with_rc(self) -> None:
        with pytest.raises(ValueError, match="backbone needs >= 2 points"):
            LadrunoRCFiniteStrain(
                E=30e9, nu=0.2, Ce=(0.0,), Cs=(0.0,),
                Te=(0.0, 1e-4), Ts=(0.0, 3e6))


# ---------------------------------------------------------------------------
# is_finite_strain marker (mirrors the fork's FiniteStrainNDMaterial base)
# ---------------------------------------------------------------------------

class TestFiniteStrainMarker:
    def test_finite_strain_materials_are_marked(self) -> None:
        assert LogStrain.is_finite_strain is True
        assert LadrunoJ2Finite.is_finite_strain is True
        assert InitDefGrad.is_finite_strain is True
        assert LadrunoRCFiniteStrain.is_finite_strain is True

    def test_small_strain_materials_are_not_marked(self) -> None:
        # StagedStrain is the *small-strain* staged wrapper (setTrialStrain),
        # NOT a FiniteStrainNDMaterial — it must stay unmarked.
        assert StagedStrain.is_finite_strain is False
        assert ElasticIsotropic.is_finite_strain is False
        assert LadrunoJ2.is_finite_strain is False
        assert LadrunoConcrete3D.is_finite_strain is False
        assert LadrunoRCConcrete.is_finite_strain is False


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

    def test_LogStrain_wraps_and_inner_tagged_first(self) -> None:
        ops = _stub_bridge()
        inner = ops.nDMaterial.LadrunoJ2(K=1.65e8, G=7.5e7, sig0=450.0)
        wrapped = ops.nDMaterial.LogStrain(inner=inner)
        assert isinstance(wrapped, LogStrain)
        assert ops.tag_for(inner) == 1
        assert ops.tag_for(wrapped) == 2

    def test_LogStrain_accepts_inner_by_name(self) -> None:
        ops = _stub_bridge()
        ops.nDMaterial.LadrunoJ2(K=1.65e8, G=7.5e7, sig0=450.0, name="steel")
        wrapped = ops.nDMaterial.LogStrain(inner="steel")
        assert isinstance(wrapped, LogStrain)

    def test_InitDefGrad_wraps_finite_inner(self) -> None:
        ops = _stub_bridge()
        finite = ops.nDMaterial.LadrunoJ2Finite(K=1.65e8, G=7.5e7, sig0=450.0)
        wrapped = ops.nDMaterial.InitDefGrad(inner=finite)
        assert isinstance(wrapped, InitDefGrad)
        assert ops.tag_for(finite) == 1
        assert ops.tag_for(wrapped) == 2

    def test_StagedStrain_wraps_inner(self) -> None:
        ops = _stub_bridge()
        inner = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2)
        wrapped = ops.nDMaterial.StagedStrain(inner=inner)
        assert isinstance(wrapped, StagedStrain)
        assert ops.tag_for(wrapped) == 2

    def test_LadrunoConcrete3D_via_namespace_returns_typed_instance(
        self,
    ) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.LadrunoConcrete3D(
            E=30e9, nu=0.2, fc=30e6, ft=3e6, Gf=120.0, Gc=12000.0)
        assert isinstance(m, LadrunoConcrete3D)
        assert ops.tag_for(m) == 1

    def test_LadrunoRCConcrete_via_namespace_builds_backbone(self) -> None:
        ops = _stub_bridge()
        # Exercise a deep flag (agg) to confirm the full surface forwards.
        m = ops.nDMaterial.LadrunoRCConcrete(
            E=30e9, nu=0.2, fc=30e6, interlock=True, agg=20.0,
            implex_control=(0.05, 0.01))
        assert isinstance(m, LadrunoRCConcrete)
        assert len(m.Ce) >= 2 and m.interlock is True
        assert m.agg == 20.0 and m.implex_control == (0.05, 0.01)
        assert ops.tag_for(m) == 1

    def test_LadrunoRCFiniteStrain_via_namespace_returns_typed_instance(
        self,
    ) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.LadrunoRCFiniteStrain(E=30e9, nu=0.2, fc=30e6)
        assert isinstance(m, LadrunoRCFiniteStrain)
        assert m.is_finite_strain is True
        assert ops.tag_for(m) == 1

    def test_LadrunoCohesiveHingeBiaxial_via_namespace(self) -> None:
        ops = _stub_bridge()
        m = ops.nDMaterial.LadrunoCohesiveHingeBiaxial(
            Mcz=5e5, Gfz=1200.0, Mcy=3e5, Gfy=900.0)
        assert isinstance(m, LadrunoCohesiveHingeBiaxial)
        assert ops.tag_for(m) == 1


# ---------------------------------------------------------------------------
# LadrunoCohesiveHingeBiaxial (Ladruno fork — coupled Mz-My hinge, ND 33004)
# ---------------------------------------------------------------------------

class TestLadrunoCohesiveHingeBiaxial:
    def test_emit_minimal(self) -> None:
        rec = RecordingEmitter()
        LadrunoCohesiveHingeBiaxial(
            Mcz=5e5, Gfz=1200.0, Mcy=3e5, Gfy=900.0)._emit(rec, tag=1)
        assert rec.calls == [
            ("nDMaterial",
             ("LadrunoCohesiveHingeBiaxial", 1, 5e5, 1200.0, 3e5, 900.0), {}),
        ]

    def test_emit_with_options(self) -> None:
        rec = RecordingEmitter()
        LadrunoCohesiveHingeBiaxial(
            Mcz=5e5, Gfz=1200.0, Mcy=3e5, Gfy=900.0,
            softening="linear", penalty_ratio=500.0, bk_eta=2.0,
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoCohesiveHingeBiaxial", 2, 5e5, 1200.0, 3e5, 900.0,
            "-linear", "-penaltyRatio", 500.0, "-bk", 2.0,
        )

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoCohesiveHingeBiaxial(
            Mcz=1.0, Gfz=1.0, Mcy=1.0, Gfy=1.0).dependencies() == ()

    def test_is_not_finite_strain(self) -> None:
        assert LadrunoCohesiveHingeBiaxial.is_finite_strain is False

    @pytest.mark.parametrize("field", ["Mcz", "Gfz", "Mcy", "Gfy"])
    def test_rejects_non_positive(self, field: str) -> None:
        kwargs = {"Mcz": 5e5, "Gfz": 1200.0, "Mcy": 3e5, "Gfy": 900.0}
        kwargs[field] = 0.0
        with pytest.raises(ValueError, match=f"{field} must be > 0"):
            LadrunoCohesiveHingeBiaxial(**kwargs)

    def test_rejects_bad_bk_eta(self) -> None:
        with pytest.raises(ValueError, match="bk_eta must be > 0"):
            LadrunoCohesiveHingeBiaxial(
                Mcz=5e5, Gfz=1200.0, Mcy=3e5, Gfy=900.0, bk_eta=0.0)
