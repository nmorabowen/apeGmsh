"""Unit tests for the typed uniaxial material primitives (Phase 1A).

For each class shipped by ``apeGmsh.opensees.material.uniaxial``:

  * construction with valid parameters,
  * ``_emit`` records the correct OpenSees command on a
    :class:`RecordingEmitter`,
  * ``dependencies`` is empty (uniaxials are leaves),
  * ``__repr__`` mentions the class name,
  * validation rejects the boundary cases.

The contract gate ``test_uniaxial_material_contract.py`` parametrizes
the family-wide checks; this file exercises per-class behavior.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.uniaxial import (
    ENT,
    ASDSteel1D,
    Concrete01,
    Concrete02,
    ElasticMaterial,
    Hysteretic,
    InitialStress,
    LadrunoBondSlip,
    LadrunoCohesiveHinge,
    LadrunoRebarBuckling,
    LadrunoUniaxialJ2,
    Maxwell,
    Steel01,
    Steel02,
    Viscous,
    ViscousDamper,
)


# ---------------------------------------------------------------------------
# Steel01
# ---------------------------------------------------------------------------

class TestSteel01:
    def test_construction(self) -> None:
        m = Steel01(fy=420e6, E=200e9, b=0.01)
        assert m.fy == 420e6
        assert m.E == 200e9
        assert m.b == 0.01

    def test_emit_records_correct_call(self) -> None:
        m = Steel01(fy=420e6, E=200e9, b=0.01)
        rec = RecordingEmitter()
        m._emit(rec, tag=42)
        assert rec.calls == [
            ("uniaxialMaterial", ("Steel01", 42, 420e6, 200e9, 0.01), {}),
        ]

    def test_emit_with_isotropic_hardening(self) -> None:
        m = Steel01(
            fy=420e6, E=200e9, b=0.01,
            a1=0.1, a2=0.5, a3=0.0, a4=1.0,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "Steel01", 1, 420e6, 200e9, 0.01, 0.1, 0.5, 0.0, 1.0,
        )

    def test_dependencies_is_empty(self) -> None:
        assert Steel01(fy=420e6, E=200e9, b=0.01).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "Steel01" in repr(Steel01(fy=420e6, E=200e9, b=0.01))

    def test_validation_rejects_bad_fy(self) -> None:
        with pytest.raises(ValueError, match="fy"):
            Steel01(fy=-1.0, E=200e9, b=0.01)

    def test_validation_rejects_bad_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            Steel01(fy=420e6, E=0.0, b=0.01)

    def test_validation_rejects_bad_b(self) -> None:
        with pytest.raises(ValueError, match="b"):
            Steel01(fy=420e6, E=200e9, b=1.0)

    def test_isotropic_hardening_all_or_none(self) -> None:
        with pytest.raises(ValueError, match="a1..a4"):
            Steel01(fy=420e6, E=200e9, b=0.01, a1=0.1)


# ---------------------------------------------------------------------------
# Steel02
# ---------------------------------------------------------------------------

class TestSteel02:
    def test_construction(self) -> None:
        m = Steel02(fy=420e6, E=200e9, b=0.01)
        assert m.fy == 420e6
        assert m.R0 == 20.0
        assert m.cR1 == 0.925
        assert m.cR2 == 0.15

    def test_emit_records_correct_call(self) -> None:
        m = Steel02(fy=420e6, E=200e9, b=0.01)
        rec = RecordingEmitter()
        m._emit(rec, tag=42)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("Steel02", 42, 420e6, 200e9, 0.01, 20.0, 0.925, 0.15),
             {}),
        ]

    def test_dependencies_is_empty(self) -> None:
        assert Steel02(fy=420e6, E=200e9, b=0.01).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "Steel02" in repr(Steel02(fy=420e6, E=200e9, b=0.01))

    def test_validation_rejects_bad_fy(self) -> None:
        with pytest.raises(ValueError, match="fy"):
            Steel02(fy=-1.0, E=200e9, b=0.01)

    def test_validation_rejects_bad_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            Steel02(fy=420e6, E=-1.0, b=0.01)

    def test_validation_rejects_bad_b(self) -> None:
        with pytest.raises(ValueError, match="b"):
            Steel02(fy=420e6, E=200e9, b=-0.1)

    def test_validation_rejects_bad_R0(self) -> None:
        with pytest.raises(ValueError, match="R0"):
            Steel02(fy=420e6, E=200e9, b=0.01, R0=0.0)

    def test_isotropic_hardening_all_or_none(self) -> None:
        with pytest.raises(ValueError, match="a1..a4"):
            Steel02(fy=420e6, E=200e9, b=0.01, a1=0.1)

    def test_isotropic_hardening_extends_emit(self) -> None:
        m = Steel02(
            fy=420e6, E=200e9, b=0.01,
            a1=0.1, a2=0.5, a3=0.0, a4=1.0,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1][-4:] == (0.1, 0.5, 0.0, 1.0)

    def test_sig_init_requires_a_quad(self) -> None:
        with pytest.raises(ValueError, match="sig_init"):
            Steel02(fy=420e6, E=200e9, b=0.01, sig_init=1.0)

    def test_sig_init_appends_after_a_quad(self) -> None:
        m = Steel02(
            fy=420e6, E=200e9, b=0.01,
            a1=0.1, a2=0.5, a3=0.0, a4=1.0,
            sig_init=2.5,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1][-1] == 2.5


# ---------------------------------------------------------------------------
# ASDSteel1D
# ---------------------------------------------------------------------------

class TestASDSteel1D:
    def test_construction(self) -> None:
        m = ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.20)
        assert m.E == 200e9
        assert m.sy == 375e6
        assert m.su == 480e6
        assert m.eu == 0.20
        assert m.fracture is False

    def test_emit_minimal_records_correct_call(self) -> None:
        m = ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.20)
        rec = RecordingEmitter()
        m._emit(rec, tag=4)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("ASDSteel1D", 4, 200e9, 375e6, 480e6, 0.20),
             {}),
        ]

    def test_emit_with_fracture_flag(self) -> None:
        m = ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.20, fracture=True)
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "ASDSteel1D", 1, 200e9, 375e6, 480e6, 0.20, "-fracture",
        )

    def test_shared_radius_rides_first_active_feature(self) -> None:
        # buckling has priority over fracture for the shared radius;
        # fracture then carries no radius.
        m = ASDSteel1D(
            E=200e9, sy=375e6, su=480e6, eu=0.20,
            buckling_lch=0.1, fracture=True, radius=0.008,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "ASDSteel1D", 2, 200e9, 375e6, 480e6, 0.20,
            "-buckling", 0.1, 0.008, "-fracture",
        )

    def test_emit_implex_and_controls_tail(self) -> None:
        m = ASDSteel1D(
            E=200e9, sy=375e6, su=480e6, eu=0.20,
            implex=True, implex_control=(0.05, 0.01),
            auto_regularization=True,
            K_alpha=0.5, max_iter=120, tolU=1e-6, tolR=1e-6,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=3)
        assert rec.calls[0][1] == (
            "ASDSteel1D", 3, 200e9, 375e6, 480e6, 0.20,
            "-implex", "-implexControl", 0.05, 0.01,
            "-auto_regularization",
            "-K_alpha", 0.5, "-max_iter", 120, "-tolU", 1e-6,
            "-tolR", 1e-6,
        )

    def test_dependencies_empty_without_slip(self) -> None:
        m = ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.20)
        assert m.dependencies() == ()

    def test_slip_material_is_the_only_dependency(self) -> None:
        bond = Steel01(fy=250e6, E=200e9, b=0.01)
        m = ASDSteel1D(
            E=200e9, sy=375e6, su=480e6, eu=0.20, slip_material=bond,
        )
        assert m.dependencies() == (bond,)

    def test_slip_emit_resolves_material_tag(self) -> None:
        bond = Steel01(fy=250e6, E=200e9, b=0.01)
        m = ASDSteel1D(
            E=200e9, sy=375e6, su=480e6, eu=0.20,
            slip_material=bond, radius=0.008,
        )
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 7 if p is bond else 0)
        m._emit(rec, tag=5)
        assert rec.calls[0][1] == (
            "ASDSteel1D", 5, 200e9, 375e6, 480e6, 0.20,
            "-slip", 7, 0.008,
        )

    def test_repr_includes_type_token(self) -> None:
        m = ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.20)
        assert "ASDSteel1D" in repr(m)

    def test_validation_rejects_nonpositive_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            ASDSteel1D(E=0.0, sy=375e6, su=480e6, eu=0.20)

    def test_validation_rejects_nonpositive_eu(self) -> None:
        with pytest.raises(ValueError, match="eu"):
            ASDSteel1D(E=200e9, sy=375e6, su=480e6, eu=0.0)

    def test_validation_rejects_su_not_above_sy(self) -> None:
        with pytest.raises(ValueError, match="su"):
            ASDSteel1D(E=200e9, sy=480e6, su=480e6, eu=0.20)

    def test_validation_rejects_nonpositive_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            ASDSteel1D(
                E=200e9, sy=375e6, su=480e6, eu=0.20,
                fracture=True, radius=0.0,
            )

    def test_validation_rejects_nonpositive_buckling_lch(self) -> None:
        with pytest.raises(ValueError, match="buckling_lch"):
            ASDSteel1D(
                E=200e9, sy=375e6, su=480e6, eu=0.20, buckling_lch=-1.0,
            )


# ---------------------------------------------------------------------------
# Concrete01
# ---------------------------------------------------------------------------

class TestConcrete01:
    def test_construction(self) -> None:
        m = Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006)
        assert m.fpc == -30e6

    def test_emit_records_correct_call(self) -> None:
        m = Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006)
        rec = RecordingEmitter()
        m._emit(rec, tag=7)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("Concrete01", 7, -30e6, -0.002, -25e6, -0.006),
             {}),
        ]

    def test_dependencies_is_empty(self) -> None:
        m = Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006)
        assert m.dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        m = Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006)
        assert "Concrete01" in repr(m)

    def test_validation_rejects_positive_fpc(self) -> None:
        with pytest.raises(ValueError, match="fpc"):
            Concrete01(fpc=30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006)

    def test_validation_rejects_positive_epsc0(self) -> None:
        with pytest.raises(ValueError, match="epsc0"):
            Concrete01(fpc=-30e6, epsc0=0.002, fpcu=-25e6, epsU=-0.006)

    def test_validation_rejects_positive_fpcu(self) -> None:
        with pytest.raises(ValueError, match="fpcu"):
            Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=25e6, epsU=-0.006)

    def test_validation_rejects_positive_epsU(self) -> None:
        with pytest.raises(ValueError, match="epsU"):
            Concrete01(fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=0.006)

    def test_validation_rejects_epsU_above_epsc0(self) -> None:
        # epsU must be MORE compressive (more negative) than epsc0.
        with pytest.raises(ValueError, match="epsU"):
            Concrete01(fpc=-30e6, epsc0=-0.005, fpcu=-25e6, epsU=-0.001)


# ---------------------------------------------------------------------------
# Concrete02
# ---------------------------------------------------------------------------

class TestConcrete02:
    def test_construction(self) -> None:
        m = Concrete02(
            fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
            lambda_val=0.1, ft=2.5e6, Ets=200e6,
        )
        assert m.lambda_val == 0.1
        assert m.ft == 2.5e6

    def test_emit_records_correct_call(self) -> None:
        m = Concrete02(
            fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
            lambda_val=0.1, ft=2.5e6, Ets=200e6,
        )
        rec = RecordingEmitter()
        m._emit(rec, tag=3)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("Concrete02", 3,
              -30e6, -0.002, -25e6, -0.006, 0.1, 2.5e6, 200e6),
             {}),
        ]

    def test_dependencies_is_empty(self) -> None:
        m = Concrete02(
            fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
            lambda_val=0.1, ft=2.5e6, Ets=200e6,
        )
        assert m.dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        m = Concrete02(
            fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
            lambda_val=0.1, ft=2.5e6, Ets=200e6,
        )
        assert "Concrete02" in repr(m)

    def test_validation_rejects_lambda_above_one(self) -> None:
        with pytest.raises(ValueError, match="lambda_val"):
            Concrete02(
                fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
                lambda_val=1.5, ft=2.5e6, Ets=200e6,
            )

    def test_validation_rejects_negative_lambda(self) -> None:
        with pytest.raises(ValueError, match="lambda_val"):
            Concrete02(
                fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
                lambda_val=-0.1, ft=2.5e6, Ets=200e6,
            )

    def test_validation_rejects_nonpositive_ft(self) -> None:
        with pytest.raises(ValueError, match="ft"):
            Concrete02(
                fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
                lambda_val=0.1, ft=0.0, Ets=200e6,
            )

    def test_validation_rejects_nonpositive_Ets(self) -> None:
        with pytest.raises(ValueError, match="Ets"):
            Concrete02(
                fpc=-30e6, epsc0=-0.002, fpcu=-25e6, epsU=-0.006,
                lambda_val=0.1, ft=2.5e6, Ets=-1.0,
            )


# ---------------------------------------------------------------------------
# Hysteretic
# ---------------------------------------------------------------------------

def _hys(**overrides: float | None) -> Hysteretic:
    """Build a Hysteretic with sensible defaults; ``overrides`` patches them."""
    base: dict[str, float | None] = {
        "s1p": 100e3, "e1p": 0.001, "s2p": 200e3, "e2p": 0.005,
        "s1n": -100e3, "e1n": -0.001, "s2n": -200e3, "e2n": -0.005,
        "pinch_x": 1.0, "pinch_y": 1.0,
        "damage1": 0.0, "damage2": 0.0,
    }
    base.update(overrides)
    # mypy: cannot statically verify the dict spread matches Hysteretic's
    # signature; the test surface enforces it dynamically.
    return Hysteretic(**base)  # type: ignore[arg-type]


class TestHysteretic:
    def test_construction(self) -> None:
        m = _hys()
        assert m.s1p == 100e3
        assert m.beta == 0.0

    def test_emit_records_correct_call(self) -> None:
        m = _hys()
        rec = RecordingEmitter()
        m._emit(rec, tag=11)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("Hysteretic", 11,
              100e3, 0.001, 200e3, 0.005,
              -100e3, -0.001, -200e3, -0.005,
              1.0, 1.0, 0.0, 0.0),
             {}),
        ]

    def test_emit_with_third_envelope_point(self) -> None:
        m = _hys(s3p=250e3, e3p=0.010, s3n=-250e3, e3n=-0.010)
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "Hysteretic", 1,
            100e3, 0.001, 200e3, 0.005, 250e3, 0.010,
            -100e3, -0.001, -200e3, -0.005, -250e3, -0.010,
            1.0, 1.0, 0.0, 0.0,
        )

    def test_emit_with_beta(self) -> None:
        m = _hys(beta=0.25)
        rec = RecordingEmitter()
        m._emit(rec, tag=1)
        assert rec.calls[0][1][-1] == 0.25

    def test_dependencies_is_empty(self) -> None:
        assert _hys().dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "Hysteretic" in repr(_hys())

    def test_validation_rejects_negative_e1p(self) -> None:
        with pytest.raises(ValueError, match="e1p"):
            _hys(e1p=-0.001)

    def test_validation_rejects_positive_e1n(self) -> None:
        with pytest.raises(ValueError, match="e1n"):
            _hys(e1n=0.001)

    def test_validation_rejects_pinch_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="pinch_x"):
            _hys(pinch_x=1.5)

    def test_validation_rejects_negative_damage(self) -> None:
        with pytest.raises(ValueError, match="damage1"):
            _hys(damage1=-0.1)

    def test_validation_rejects_non_monotonic_positive(self) -> None:
        # e2p=0.0005 < e1p=0.001 violates monotonic envelope.
        with pytest.raises(ValueError, match="monotonic"):
            _hys(e2p=0.0005)

    def test_validation_rejects_non_monotonic_negative(self) -> None:
        # e2n=-0.0005 less negative than e1n=-0.001 violates monotonic.
        with pytest.raises(ValueError, match="monotonic"):
            _hys(e2n=-0.0005)

    def test_third_point_all_or_none(self) -> None:
        with pytest.raises(ValueError, match="third envelope"):
            _hys(s3p=250e3)

    def test_third_point_must_extend_positive_envelope(self) -> None:
        with pytest.raises(ValueError, match="monotonic"):
            _hys(s3p=250e3, e3p=0.001, s3n=-250e3, e3n=-0.010)


# ---------------------------------------------------------------------------
# ElasticMaterial — emits "Elastic" as the type token
# ---------------------------------------------------------------------------

class TestElasticMaterial:
    def test_construction(self) -> None:
        m = ElasticMaterial(E=200e9)
        assert m.E == 200e9
        assert m.eta == 0.0

    def test_emit_uses_elastic_type_token(self) -> None:
        m = ElasticMaterial(E=200e9)
        rec = RecordingEmitter()
        m._emit(rec, tag=5)
        # Note: type token is the bare "Elastic", NOT "ElasticMaterial".
        assert rec.calls == [
            ("uniaxialMaterial", ("Elastic", 5, 200e9), {}),
        ]

    def test_emit_appends_eta_when_set(self) -> None:
        m = ElasticMaterial(E=200e9, eta=0.05)
        rec = RecordingEmitter()
        m._emit(rec, tag=5)
        assert rec.calls == [
            ("uniaxialMaterial", ("Elastic", 5, 200e9, 0.05), {}),
        ]

    def test_emit_omits_eta_when_zero(self) -> None:
        m = ElasticMaterial(E=200e9, eta=0.0)
        rec = RecordingEmitter()
        m._emit(rec, tag=5)
        assert rec.calls[0][1] == ("Elastic", 5, 200e9)

    def test_dependencies_is_empty(self) -> None:
        assert ElasticMaterial(E=200e9).dependencies() == ()

    def test_repr_includes_class_name(self) -> None:
        # The Python class name is ElasticMaterial; the OpenSees token
        # is "Elastic". __repr__ shows the Python class name.
        assert "ElasticMaterial" in repr(ElasticMaterial(E=200e9))

    def test_validation_rejects_nonpositive_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            ElasticMaterial(E=-1.0)

    def test_validation_rejects_negative_eta(self) -> None:
        with pytest.raises(ValueError, match="eta"):
            ElasticMaterial(E=200e9, eta=-0.01)


# ---------------------------------------------------------------------------
# ENT
# ---------------------------------------------------------------------------

class TestENT:
    def test_construction(self) -> None:
        assert ENT(E=200e9).E == 200e9

    def test_emit_records_correct_call(self) -> None:
        rec = RecordingEmitter()
        ENT(E=200e9)._emit(rec, tag=9)
        assert rec.calls == [
            ("uniaxialMaterial", ("ENT", 9, 200e9), {}),
        ]

    def test_dependencies_is_empty(self) -> None:
        assert ENT(E=200e9).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "ENT" in repr(ENT(E=200e9))

    def test_validation_rejects_nonpositive_E(self) -> None:
        with pytest.raises(ValueError, match="E"):
            ENT(E=0.0)


# ---------------------------------------------------------------------------
# InitialStress — uniaxial wrapper composing a base uniaxial + sigma_init
# ---------------------------------------------------------------------------

class TestInitialStress:
    def test_construction(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        assert m.base_material is base
        assert m.sigma_init == 125e6

    def test_base_material_is_the_only_dependency(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        assert m.dependencies() == (base,)

    def test_emit_resolves_base_tag(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 7 if p is base else 0)
        m._emit(rec, tag=12)
        assert rec.calls == [
            (
                "uniaxialMaterial",
                ("InitialStressMaterial", 12, 7, 125e6),
                {},
            )
        ]

    def test_emit_without_tag_resolver_raises(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        rec = RecordingEmitter()
        with pytest.raises(RuntimeError, match="tag resolver"):
            m._emit(rec, tag=1)

    def test_repr_includes_type_token(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        assert "InitialStress" in repr(m)

    def test_is_frozen(self) -> None:
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=125e6)
        with pytest.raises(Exception):  # FrozenInstanceError
            m.sigma_init = 0.0  # type: ignore[misc]

    def test_rejects_non_uniaxial_base(self) -> None:
        class FakePrimitive:
            pass

        with pytest.raises(TypeError, match="UniaxialMaterial"):
            InitialStress(
                base_material=FakePrimitive(),  # type: ignore[arg-type]
                sigma_init=125e6,
            )

    def test_accepts_zero_sigma_init(self) -> None:
        # sigma_init=0 is a degenerate but legal call (e.g. parametric
        # sweep over a residual-stress amplitude that starts at zero).
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=0.0)
        assert m.sigma_init == 0.0

    def test_accepts_negative_sigma_init(self) -> None:
        # Compressive residual stress for the central web band in an
        # ECCS welded-I pattern.
        base = Steel02(fy=420e6, E=200e9, b=0.01)
        m = InitialStress(base_material=base, sigma_init=-0.3 * 250e6)
        assert m.sigma_init < 0


class TestInitialStressNamespace:
    def test_namespace_constructs_and_registers(self) -> None:
        from unittest.mock import MagicMock
        from typing import cast
        from apeGmsh.opensees import apeSees

        ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
        base = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
        wrapped = ops.uniaxialMaterial.InitialStress(
            base_material=base, sigma_init=125e6,
        )
        assert isinstance(wrapped, InitialStress)
        # Distinct registered primitives must hold distinct tags; the
        # tag allocator hands them out in registration order.
        assert ops.tag_for(base) == 1
        assert ops.tag_for(wrapped) == 2


# ---------------------------------------------------------------------------
# Viscous / ViscousDamper / Maxwell — rate-dependent (dashpot) materials
# ---------------------------------------------------------------------------

class TestViscous:
    def test_construction_defaults(self) -> None:
        m = Viscous(C=1.0e5)
        assert m.C == 1.0e5
        assert m.alpha == 1.0
        assert m.min_vel == 1.0e-11

    def test_is_rate_dependent(self) -> None:
        assert Viscous.is_rate_dependent is True
        assert Viscous(C=1.0).is_rate_dependent is True

    def test_emit_minimal_omits_default_min_vel(self) -> None:
        m = Viscous(C=1.0e5, alpha=0.5)
        rec = RecordingEmitter()
        m._emit(rec, tag=7)
        assert rec.calls == [
            ("uniaxialMaterial", ("Viscous", 7, 1.0e5, 0.5), {}),
        ]

    def test_emit_appends_custom_min_vel(self) -> None:
        m = Viscous(C=1.0e5, alpha=1.0, min_vel=1.0e-6)
        rec = RecordingEmitter()
        m._emit(rec, tag=3)
        assert rec.calls[0][1] == ("Viscous", 3, 1.0e5, 1.0, 1.0e-6)

    def test_dependencies_is_empty(self) -> None:
        assert Viscous(C=1.0).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "Viscous" in repr(Viscous(C=1.0))

    def test_validation_rejects_bad_C(self) -> None:
        with pytest.raises(ValueError, match="C must be > 0"):
            Viscous(C=0.0)

    def test_validation_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            Viscous(C=1.0, alpha=0.0)

    def test_validation_rejects_bad_min_vel(self) -> None:
        with pytest.raises(ValueError, match="min_vel must be > 0"):
            Viscous(C=1.0, min_vel=0.0)


class TestViscousDamper:
    def test_construction(self) -> None:
        m = ViscousDamper(K=1.0e9, C=1.0e5, alpha=0.5)
        assert (m.K, m.C, m.alpha) == (1.0e9, 1.0e5, 0.5)
        assert m.l_gap is None

    def test_is_rate_dependent(self) -> None:
        assert ViscousDamper.is_rate_dependent is True

    def test_emit_minimal(self) -> None:
        m = ViscousDamper(K=1.0e9, C=1.0e5, alpha=0.5)
        rec = RecordingEmitter()
        m._emit(rec, tag=4)
        assert rec.calls == [
            ("uniaxialMaterial", ("ViscousDamper", 4, 1.0e9, 1.0e5, 0.5), {}),
        ]

    def test_emit_appends_l_gap(self) -> None:
        m = ViscousDamper(K=1.0e9, C=1.0e5, alpha=0.5, l_gap=0.001)
        rec = RecordingEmitter()
        m._emit(rec, tag=4)
        assert rec.calls[0][1] == ("ViscousDamper", 4, 1.0e9, 1.0e5, 0.5, 0.001)

    def test_validation_rejects_bad_K(self) -> None:
        with pytest.raises(ValueError, match="K must be > 0"):
            ViscousDamper(K=0.0, C=1.0, alpha=0.5)

    def test_validation_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha must be > 0"):
            ViscousDamper(K=1.0, C=1.0, alpha=-0.1)


class TestMaxwell:
    def test_construction(self) -> None:
        m = Maxwell(K=1.0e9, C=1.0e5, alpha=0.5, length=2.0)
        assert (m.K, m.C, m.alpha, m.length) == (1.0e9, 1.0e5, 0.5, 2.0)

    def test_is_rate_dependent(self) -> None:
        assert Maxwell.is_rate_dependent is True

    def test_emit(self) -> None:
        m = Maxwell(K=1.0e9, C=1.0e5, alpha=0.5, length=2.0)
        rec = RecordingEmitter()
        m._emit(rec, tag=9)
        assert rec.calls == [
            ("uniaxialMaterial", ("Maxwell", 9, 1.0e9, 1.0e5, 0.5, 2.0), {}),
        ]

    def test_validation_rejects_bad_length(self) -> None:
        with pytest.raises(ValueError, match="length must be > 0"):
            Maxwell(K=1.0, C=1.0, alpha=0.5, length=0.0)


# ---------------------------------------------------------------------------
# LadrunoBondSlip (Ladruno fork — bond-slip axial law, MAT 33002)
# ---------------------------------------------------------------------------

class TestLadrunoBondSlip:
    def _valid(self, **over: float) -> LadrunoBondSlip:
        kw: dict[str, float] = dict(
            tau_max=12.0, s1=1.0, s2=3.0, s3=10.0, tau_f=2.0, alpha=0.4,
        )
        kw.update(over)
        return LadrunoBondSlip(**kw)  # type: ignore[arg-type]

    def test_construction(self) -> None:
        m = self._valid()
        assert m.tau_max == 12.0
        assert (m.s1, m.s2, m.s3) == (1.0, 3.0, 10.0)
        assert m.tau_f == 2.0 and m.alpha == 0.4
        assert m.Gf is None and m.s0 is None

    def test_emit_bare_backbone(self) -> None:
        rec = RecordingEmitter()
        self._valid()._emit(rec, tag=7)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("LadrunoBondSlip", 7, 12.0, 1.0, 3.0, 10.0, 2.0, 0.4), {}),
        ]

    def test_emit_with_Gf_and_s0(self) -> None:
        rec = RecordingEmitter()
        self._valid(Gf=0.5, s0=0.1)._emit(rec, tag=3)
        assert rec.calls[0][1] == (
            "LadrunoBondSlip", 3, 12.0, 1.0, 3.0, 10.0, 2.0, 0.4,
            "-Gf", 0.5, "-s0", 0.1,
        )

    def test_emit_with_only_Gf(self) -> None:
        rec = RecordingEmitter()
        self._valid(Gf=0.5)._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "LadrunoBondSlip", 1, 12.0, 1.0, 3.0, 10.0, 2.0, 0.4, "-Gf", 0.5,
        )

    def test_dependencies_is_empty(self) -> None:
        assert self._valid().dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoBondSlip" in repr(self._valid())

    def test_validation_rejects_bad_tau_max(self) -> None:
        with pytest.raises(ValueError, match="tau_max must be > 0"):
            self._valid(tau_max=0.0)

    def test_validation_rejects_unordered_slips(self) -> None:
        with pytest.raises(ValueError, match="0 < s1 <= s2 <= s3"):
            self._valid(s2=0.5)  # s2 < s1

    def test_validation_rejects_residual_over_peak(self) -> None:
        with pytest.raises(ValueError, match=r"tau_f must be in \[0, tau_max\]"):
            self._valid(tau_f=20.0)

    def test_validation_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
            self._valid(alpha=1.5)

    def test_validation_rejects_bad_Gf(self) -> None:
        with pytest.raises(ValueError, match="Gf must be > 0"):
            self._valid(Gf=0.0)

    def test_validation_rejects_s0_outside_range(self) -> None:
        with pytest.raises(ValueError, match="s0 must satisfy 0 < s0 < s1"):
            self._valid(s0=1.0)  # s0 == s1


# ---------------------------------------------------------------------------
# LadrunoUniaxialJ2 (Ladruno fork — 1D combined-hardening J2, MAT 33000)
# ---------------------------------------------------------------------------

class TestLadrunoUniaxialJ2:
    def test_construction_defaults(self) -> None:
        m = LadrunoUniaxialJ2(E=200e9, sig0=250e6)
        assert m.E == 200e9 and m.sig0 == 250e6
        assert (m.Qinf, m.b, m.Hiso) == (0.0, 0.0, 0.0)
        assert m.backstresses == () and m.damage is None
        assert m.implex is False

    def test_emit_bare_isotropic(self) -> None:
        rec = RecordingEmitter()
        LadrunoUniaxialJ2(E=200e9, sig0=250e6)._emit(rec, tag=4)
        assert rec.calls == [
            ("uniaxialMaterial",
             ("LadrunoUniaxialJ2", 4, 200e9,
              "-iso", "voce", 250e6, 0.0, 0.0, 0.0), {}),
        ]

    def test_emit_with_kinematic_and_damage(self) -> None:
        rec = RecordingEmitter()
        LadrunoUniaxialJ2(
            E=200e9, sig0=420e6, Qinf=100e6, b=10.0, Hiso=1e9,
            backstresses=[(20000e6, 200.0), (5000e6, 50.0)],
            damage=(1.0, 1.0, 0.0, 0.99), implex=True,
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoUniaxialJ2", 2, 200e9,
            "-iso", "voce", 420e6, 100e6, 10.0, 1e9,
            "-kin", 2, 20000e6, 200.0, 5000e6, 50.0,
            "-damage", "lemaitre", 1.0, 1.0, 0.0, 0.99,
            "-implex",
        )

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoUniaxialJ2(E=200e9, sig0=250e6).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoUniaxialJ2" in repr(
            LadrunoUniaxialJ2(E=200e9, sig0=250e6)
        )

    def test_rejects_non_positive_E(self) -> None:
        with pytest.raises(ValueError, match="E must be > 0"):
            LadrunoUniaxialJ2(E=0.0, sig0=250e6)

    def test_rejects_non_positive_sig0(self) -> None:
        with pytest.raises(ValueError, match="sig0 must be > 0"):
            LadrunoUniaxialJ2(E=200e9, sig0=0.0)

    def test_rejects_bad_lemaitre_r(self) -> None:
        with pytest.raises(ValueError, match="Lemaitre r must be > 0"):
            LadrunoUniaxialJ2(E=200e9, sig0=250e6, damage=(0.0, 1.0, 0.0, 0.9))


class TestLadrunoUniaxialJ2Namespace:
    def test_namespace_constructs_and_registers(self) -> None:
        from unittest.mock import MagicMock
        from typing import cast
        from apeGmsh.opensees import apeSees

        ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
        m = ops.uniaxialMaterial.LadrunoUniaxialJ2(
            E=200e9, sig0=420e6, backstresses=[(20000e6, 200.0)],
        )
        assert isinstance(m, LadrunoUniaxialJ2)
        assert m.backstresses == ((20000e6, 200.0),)
        assert ops.tag_for(m) == 1


# ---------------------------------------------------------------------------
# LadrunoRebarBuckling (Ladruno fork — buckling overlay wrapper, MAT 33001)
# ---------------------------------------------------------------------------

class TestLadrunoRebarBuckling:
    def _bar(self) -> Steel02:
        return Steel02(fy=420e6, E=200e9, b=0.01)

    def test_dependencies_is_the_wrapped_material(self) -> None:
        bar = self._bar()
        assert LadrunoRebarBuckling(material=bar).dependencies() == (bar,)

    def test_emit_identity_gate_omits_flags(self) -> None:
        # lsr=0 (default) is the identity gate -> only the wrapped tag.
        bar = self._bar()
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 4 if p is bar else 0)
        LadrunoRebarBuckling(material=bar)._emit(rec, tag=10)
        assert rec.calls == [
            ("uniaxialMaterial", ("LadrunoRebarBuckling", 10, 4), {}),
        ]

    def test_emit_dm_active(self) -> None:
        bar = self._bar()
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 4 if p is bar else 0)
        LadrunoRebarBuckling(
            material=bar, lsr=6.0, fy=420e6, E=200e9, alpha=0.75,
        )._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "LadrunoRebarBuckling", 1, 4,
            "-lsr", 6.0, "-alpha", 0.75, "-fy", 420e6, "-E", 200e9,
        )

    def test_emit_ga_model_and_restraighten_c(self) -> None:
        bar = self._bar()
        rec = RecordingEmitter()
        set_tag_resolver(rec, lambda p: 4 if p is bar else 0)
        LadrunoRebarBuckling(
            material=bar, lsr=6.0, model="ga", E=200e9, reduction=0.3,
            restraighten="c", restraighten_c=0.8,
        )._emit(rec, tag=1)
        assert rec.calls[0][1] == (
            "LadrunoRebarBuckling", 1, 4,
            "-lsr", 6.0, "-model", "ga", "-reduction", 0.3, "-E", 200e9,
            "-restraighten", "c", 0.8,
        )

    def test_rejects_bad_model(self) -> None:
        with pytest.raises(ValueError, match="model must be one of"):
            LadrunoRebarBuckling(material=self._bar(), model="xx")

    def test_rejects_active_without_E(self) -> None:
        with pytest.raises(ValueError, match="E must be > 0 when lsr > 0"):
            LadrunoRebarBuckling(material=self._bar(), lsr=6.0, fy=420e6)

    def test_rejects_dm_active_without_fy(self) -> None:
        with pytest.raises(ValueError, match="fy must be > 0 for model='dm'"):
            LadrunoRebarBuckling(material=self._bar(), lsr=6.0, E=200e9)

    def test_rejects_reduction_out_of_range(self) -> None:
        with pytest.raises(ValueError, match=r"reduction must be in \[0, 1\]"):
            LadrunoRebarBuckling(material=self._bar(), reduction=1.5)

    def test_namespace_wraps_and_tags_inner_first(self) -> None:
        from unittest.mock import MagicMock
        from typing import cast
        from apeGmsh.opensees import apeSees

        ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
        bar = ops.uniaxialMaterial.LadrunoUniaxialJ2(E=200e9, sig0=420e6)
        wrapped = ops.uniaxialMaterial.LadrunoRebarBuckling(
            material=bar, lsr=6.0, fy=420e6, E=200e9,
        )
        assert isinstance(wrapped, LadrunoRebarBuckling)
        assert ops.tag_for(bar) == 1
        assert ops.tag_for(wrapped) == 2


# ---------------------------------------------------------------------------
# LadrunoCohesiveHinge (Ladruno fork — cohesive moment-rotation, MAT 33003)
# ---------------------------------------------------------------------------

class TestLadrunoCohesiveHinge:
    def test_construction_defaults(self) -> None:
        m = LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0)
        assert (m.Mc, m.Gf) == (5e5, 1200.0)
        assert m.penalty is None and m.penalty_ratio == 1000.0
        assert m.softening == "exponential"

    def test_emit_minimal(self) -> None:
        rec = RecordingEmitter()
        LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0)._emit(rec, tag=1)
        assert rec.calls == [
            ("uniaxialMaterial", ("LadrunoCohesiveHinge", 1, 5e5, 1200.0), {}),
        ]

    def test_emit_with_options(self) -> None:
        rec = RecordingEmitter()
        LadrunoCohesiveHinge(
            Mc=5e5, Gf=1200.0, penalty=2e9, penalty_ratio=500.0,
            softening="linear",
        )._emit(rec, tag=2)
        assert rec.calls[0][1] == (
            "LadrunoCohesiveHinge", 2, 5e5, 1200.0,
            "-penalty", 2e9, "-penaltyRatio", 500.0, "-linear",
        )

    def test_dependencies_is_empty(self) -> None:
        assert LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0).dependencies() == ()

    def test_repr_includes_type_token(self) -> None:
        assert "LadrunoCohesiveHinge" in repr(LadrunoCohesiveHinge(Mc=1.0, Gf=1.0))

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_Mc(self, bad: float) -> None:
        with pytest.raises(ValueError, match="Mc must be > 0"):
            LadrunoCohesiveHinge(Mc=bad, Gf=1200.0)

    def test_rejects_non_positive_Gf(self) -> None:
        with pytest.raises(ValueError, match="Gf must be > 0"):
            LadrunoCohesiveHinge(Mc=5e5, Gf=0.0)

    def test_rejects_non_positive_penalty(self) -> None:
        with pytest.raises(ValueError, match="penalty must be > 0"):
            LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0, penalty=0.0)

    def test_rejects_bad_softening(self) -> None:
        with pytest.raises(ValueError, match="softening must be one of"):
            LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0, softening="bogus")

    def test_via_namespace_returns_typed_instance(self) -> None:
        from unittest.mock import MagicMock
        from typing import cast
        from apeGmsh.opensees import apeSees

        ops = apeSees(cast("object", MagicMock(name="FEMData")))  # type: ignore[arg-type]
        m = ops.uniaxialMaterial.LadrunoCohesiveHinge(Mc=5e5, Gf=1200.0)
        assert isinstance(m, LadrunoCohesiveHinge)
        assert ops.tag_for(m) == 1
