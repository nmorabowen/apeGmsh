"""Unit tests for the Mander confined-concrete helper (ADR 0044 follow-up).

`ASDConcrete1D.from_mander` bakes triaxial confinement into the compression
backbone (the uniaxial model is confinement-blind) via the Mander/Priestley/Park
1988 model, and returns an explicit-curve `ASDConcrete1D`.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material import _mander as mander
from apeGmsh.opensees.material.uniaxial import ASDConcrete1D


_E, _FC, _FL, _EPS_CU = 30_000.0, 30.0, 5.0, 0.02


# ---------------------------------------------------------------------------
# Mander math (pure functions)
# ---------------------------------------------------------------------------

class TestManderMath:
    def test_confined_strength_enhances_and_matches_formula(self) -> None:
        fcc = mander.confined_strength(_FC, _FL)
        expected = _FC * (2.254 * (1 + 7.94 * _FL / _FC) ** 0.5
                          - 2 * _FL / _FC - 1.254)
        assert fcc == pytest.approx(expected)
        assert fcc > _FC                      # confinement raises strength

    def test_peak_strain_grows_with_confinement(self) -> None:
        fcc = mander.confined_strength(_FC, _FL)
        eps_cc = mander.confined_peak_strain(_FC, fcc, 0.002)
        assert eps_cc > 0.002 * (1 + 5 * (fcc / _FC - 1)) - 1e-12
        assert eps_cc > 0.002                 # more ductile than unconfined

    def test_compression_backbone_shape(self) -> None:
        fcc = mander.confined_strength(_FC, _FL)
        eps_cc = mander.confined_peak_strain(_FC, fcc, 0.002)
        Ce, Cs, Cd = mander.compression_backbone(_E, fcc, eps_cc, _EPS_CU)
        assert Ce[0] == 0.0 and Cs[0] == 0.0
        assert Cs[1] / Ce[1] == pytest.approx(_E)         # elastic anchor slope = E
        assert max(Cs) == pytest.approx(fcc, rel=1e-9)    # peak hits fcc exactly
        assert all(Ce[i] < Ce[i + 1] for i in range(len(Ce) - 1))
        assert all(0.0 <= d < 1.0 for d in Cd)

    def test_rejects_too_soft_peak(self) -> None:
        # E must exceed the secant modulus fcc/eps_cc.
        with pytest.raises(ValueError, match="must exceed the secant modulus"):
            mander.compression_backbone(E=1000.0, fcc=60.0, eps_cc=0.01,
                                        eps_cu=0.02)


# ---------------------------------------------------------------------------
# from_mander
# ---------------------------------------------------------------------------

class TestFromMander:
    def test_fl_and_fcc_paths_agree(self) -> None:
        fcc = mander.confined_strength(_FC, _FL)
        m_fl = ASDConcrete1D.from_mander(E=_E, fc=_FC, fl=_FL, eps_cu=_EPS_CU)
        m_fcc = ASDConcrete1D.from_mander(E=_E, fc=_FC, fcc=fcc, eps_cu=_EPS_CU)
        assert m_fl.Cs == m_fcc.Cs and m_fl.Ce == m_fcc.Ce

    def test_confined_peak_exceeds_unconfined(self) -> None:
        m = ASDConcrete1D.from_mander(E=_E, fc=_FC, fl=_FL, eps_cu=_EPS_CU)
        assert max(m.Cs) > _FC                # confined > unconfined fc

    def test_auto_regularize_off_by_default(self) -> None:
        m = ASDConcrete1D.from_mander(E=_E, fc=_FC, fl=_FL, eps_cu=_EPS_CU)
        assert m.auto_regularize is False
        em = RecordingEmitter()
        m._emit(em, tag=1)
        (_, args, _), = em.calls
        assert "-autoRegularization" not in args   # Mander envelope is physical

    @pytest.mark.parametrize("kwargs,msg", [
        ({"E": _E, "fc": _FC, "eps_cu": _EPS_CU}, "exactly one of fcc"),
        ({"E": _E, "fc": _FC, "fcc": 20.0, "eps_cu": _EPS_CU}, "must exceed the unconfined"),
        ({"E": _E, "fc": _FC, "fl": _FL, "eps_cu": 0.001}, "must exceed the confined peak"),
        ({"E": _E, "fc": _FC, "fl": _FL, "eps_cu": _EPS_CU, "plastic_ratio": 1.5},
         "plastic_ratio must be in"),
    ])
    def test_validation(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ASDConcrete1D.from_mander(**kwargs)

    def test_both_fcc_and_fl_rejected(self) -> None:
        with pytest.raises(ValueError, match="exactly one of fcc"):
            ASDConcrete1D.from_mander(E=_E, fc=_FC, fcc=55.0, fl=_FL,
                                      eps_cu=_EPS_CU)


# ---------------------------------------------------------------------------
# bridge
# ---------------------------------------------------------------------------

class TestBridge:
    def test_confined_concrete1d_registers(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        m = ops.uniaxialMaterial.ConfinedConcrete1D(E=_E, fc=_FC, fl=_FL,
                                                    eps_cu=_EPS_CU)
        assert isinstance(m, ASDConcrete1D)
        assert ops.tag_for(m) == 1
