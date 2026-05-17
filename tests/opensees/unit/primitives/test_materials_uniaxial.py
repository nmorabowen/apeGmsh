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
    Steel01,
    Steel02,
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
