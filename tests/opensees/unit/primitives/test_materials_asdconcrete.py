"""Unit tests for ``ASDConcrete3D`` and its owned backbone generator.

Validates the ADR-0044 contract without needing a live OpenSees build:
the Python-owned backbone (emitted verbatim as ``-Te/-Ts/...``), the
crack-band reference behaviour, the ``l_max`` snapback formula, the
``-autoRegularization $lch_ref`` emit (the bare flag is a parser error),
and the warn-not-raise element-size guard.
"""
from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material import _asdconcrete_laws as laws
from apeGmsh.opensees.material.nd import ASDConcrete3D, ASDRegularizationWarning
from apeGmsh.opensees.material.uniaxial import ASDConcrete1D


# Representative concrete in N/mm (fc=30 MPa, E=30 GPa).
FC, FT, E, V = 30.0, 3.0, 30_000.0, 0.2
GF = 0.1  # tensile fracture energy per area, N/mm


# ---------------------------------------------------------------------------
# Backbone generator (pure functions)
# ---------------------------------------------------------------------------

class TestBackboneGenerator:
    def test_tension_shape(self) -> None:
        Te, Ts, Td = laws.make_tension(E, FT, GF, lch_ref=50.0)
        assert len(Te) == len(Ts) == len(Td) == 6
        assert Te[0] == 0.0 and Ts[0] == 0.0
        assert Ts[1] == pytest.approx(0.9 * FT)   # f0
        assert Ts[2] == pytest.approx(FT)         # peak
        assert all(Te[i] < Te[i + 1] for i in range(5))   # strictly increasing
        assert all(0.0 <= d < 1.0 for d in Td)

    def test_compression_shape(self) -> None:
        Ce, Cs, Cd = laws.make_compression(E, FC, laws.ceb_fip_Gc(FC, FT, GF),
                                            lch_ref=50.0)
        assert len(Ce) == len(Cs) == len(Cd) == 13
        assert Ce[0] == 0.0 and Cs[0] == 0.0
        assert max(Cs) == pytest.approx(FC, rel=1e-6)   # bezier peak hits fc
        assert all(Ce[i] < Ce[i + 1] for i in range(12))
        assert all(0.0 <= d < 1.0 for d in Cd)

    def test_lch_ref_rescales_softening_inversely(self) -> None:
        # e2 = w2 + f2/E + ep, where only the crack-opening width
        # w2 = (Gf/lch_ref)/ft scales as 1/lch_ref (the elastic offsets are
        # constant). Halving lch_ref doubles w2, so the increase in e2 equals
        # one extra w2 = Gf/(lch_ref*ft) at the original lch_ref.
        Te_a, _, _ = laws.make_tension(E, FT, GF, lch_ref=50.0)
        Te_b, _, _ = laws.make_tension(E, FT, GF, lch_ref=25.0)
        assert Te_b[3] - Te_a[3] == pytest.approx(GF / (50.0 * FT), rel=1e-6)

    def test_l_max_formula(self) -> None:
        assert laws.l_max(E, GF, FT) == pytest.approx(2.0 * E * GF / (FT * FT))

    def test_auto_lch_ref_positive(self) -> None:
        lch = laws.auto_lch_ref(E, FC, FT, laws.ceb_fip_Gf(FC),
                                laws.ceb_fip_Gc(FC, FT, laws.ceb_fip_Gf(FC)))
        assert lch > 0.0


# ---------------------------------------------------------------------------
# from_fc construction
# ---------------------------------------------------------------------------

class TestFromFc:
    def test_builds_full_backbone(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        assert len(m.Te) == 6 and len(m.Ce) == 13
        assert m.lch_ref == 50.0
        assert m.ft == FT and m.Gf == GF

    def test_defaults_ft_and_gf_from_fc(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, lch_ref=50.0)
        assert m.ft == pytest.approx(laws.default_ft(FC))
        assert m.Gf == pytest.approx(laws.ceb_fip_Gf(FC))

    def test_default_lch_ref_is_auto_derived(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC)
        assert m.lch_ref == pytest.approx(
            laws.auto_lch_ref(E, FC, laws.default_ft(FC), laws.ceb_fip_Gf(FC),
                              laws.ceb_fip_Gc(FC, laws.default_ft(FC),
                                              laws.ceb_fip_Gf(FC)))
        )

    def test_larger_Gf_gives_more_ductile_tension(self) -> None:
        tough = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=0.2, lch_ref=50.0)
        brittle = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=0.05, lch_ref=50.0)
        assert tough.Te[3] > brittle.Te[3]   # softening tail extends with Gf

    def test_preview_matches_emitted_curve(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        pv = m.preview_backbone()
        assert pv["Te"] == m.Te and pv["Cs"] == m.Cs and pv["lch_ref"] == m.lch_ref

    @pytest.mark.parametrize("kwargs,msg", [
        ({"E": 0.0, "v": V, "fc": FC}, "E must be > 0"),
        ({"E": E, "v": V, "fc": -1.0}, "fc must be > 0"),
        ({"E": E, "v": V, "fc": FC, "Gf": -0.1}, "Gf must be > 0"),
        ({"E": E, "v": V, "fc": FC, "lch_ref": 0.0}, "lch_ref must be > 0"),
    ])
    def test_validation(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ASDConcrete3D.from_fc(**kwargs)


# ---------------------------------------------------------------------------
# raw construction + validation
# ---------------------------------------------------------------------------

class TestRawConstruction:
    def _curve(self):
        Te, Ts, Td = laws.make_tension(E, FT, GF, 50.0)
        Ce, Cs, Cd = laws.make_compression(E, FC, laws.ceb_fip_Gc(FC, FT, GF), 50.0)
        return dict(Te=tuple(Te), Ts=tuple(Ts), Td=tuple(Td),
                    Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd))

    def test_raw_ok(self) -> None:
        m = ASDConcrete3D(E=E, v=V, lch_ref=50.0, **self._curve())
        assert m.l_max() is None   # no ft/Gf provenance on raw construction

    def test_rejects_bad_Kc(self) -> None:
        with pytest.raises(ValueError, match="Kc must be in"):
            ASDConcrete3D(E=E, v=V, lch_ref=50.0, Kc=0.5, **self._curve())

    def test_rejects_mismatched_lengths(self) -> None:
        c = self._curve()
        c["Ts"] = c["Ts"][:-1]
        with pytest.raises(ValueError, match="tension backbone lists must share"):
            ASDConcrete3D(E=E, v=V, lch_ref=50.0, **c)

    def test_rejects_damage_out_of_range(self) -> None:
        c = self._curve()
        c["Td"] = tuple([*c["Td"][:-1], 1.0])   # d == 1.0 is invalid
        with pytest.raises(ValueError, match="damage must be in"):
            ASDConcrete3D(E=E, v=V, lch_ref=50.0, **c)


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

class TestEmit:
    def test_emit_card_structure(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        em = RecordingEmitter()
        m._emit(em, tag=4)
        (name, args, _), = em.calls
        assert name == "nDMaterial"
        assert args[0] == "ASDConcrete3D" and args[1] == 4
        assert args[2] == E and args[3] == V
        for tok in ("-Te", "-Ts", "-Td", "-Ce", "-Cs", "-Cd", "-Kc"):
            assert tok in args
        # autoRegularization carries the explicit lch_ref (bare flag is illegal)
        i = args.index("-autoRegularization")
        assert args[i + 1] == 50.0

    def test_no_autoreg_when_disabled(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, lch_ref=50.0)
        m = ASDConcrete3D(  # rebuild with the flag off
            E=m.E, v=m.v, Te=m.Te, Ts=m.Ts, Td=m.Td, Ce=m.Ce, Cs=m.Cs,
            Cd=m.Cd, lch_ref=m.lch_ref, auto_regularize=False,
        )
        em = RecordingEmitter()
        m._emit(em, tag=1)
        (_, args, _), = em.calls
        assert "-autoRegularization" not in args


# ---------------------------------------------------------------------------
# l_max element-size guard (warn, never raise)
# ---------------------------------------------------------------------------

class TestElementSizeGuard:
    def test_warns_over_ceiling(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        lm = m.l_max()
        with pytest.warns(ASDRegularizationWarning, match="snapback ceiling"):
            ok = m.check_element_size(lm * 2.0, pg="wall")
        assert ok is False

    def test_silent_within_ceiling(self) -> None:
        m = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")   # any warning fails the test
            assert m.check_element_size(m.l_max() * 0.5) is True

    def test_raw_curve_has_no_ceiling(self) -> None:
        Te, Ts, Td = laws.make_tension(E, FT, GF, 50.0)
        Ce, Cs, Cd = laws.make_compression(E, FC, laws.ceb_fip_Gc(FC, FT, GF), 50.0)
        m = ASDConcrete3D(E=E, v=V, lch_ref=50.0, Te=tuple(Te), Ts=tuple(Ts),
                          Td=tuple(Td), Ce=tuple(Ce), Cs=tuple(Cs), Cd=tuple(Cd))
        assert m.check_element_size(1e9) is True   # unknown ceiling -> OK


# ---------------------------------------------------------------------------
# bridge namespace
# ---------------------------------------------------------------------------

class TestBridgeNamespace:
    def test_registers_and_allocates_tag(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        m = ops.nDMaterial.ASDConcrete3D(E=E, v=V, fc=FC, ft=FT, Gf=GF,
                                         lch_ref=50.0)
        assert isinstance(m, ASDConcrete3D)
        assert ops.tag_for(m) == 1


# ---------------------------------------------------------------------------
# ASDConcrete1D (uniaxial sibling — reuses the same generator)
# ---------------------------------------------------------------------------

class TestASDConcrete1D:
    def test_from_fc_builds_backbone(self) -> None:
        m = ASDConcrete1D.from_fc(E=E, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        assert len(m.Te) == 6 and len(m.Ce) == 13
        assert m.ft == FT and m.Gf == GF and m.lch_ref == 50.0

    def test_backbone_matches_3d_generator(self) -> None:
        # 1-D and 3-D -fc formulas are identical -> same generator output.
        m1 = ASDConcrete1D.from_fc(E=E, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        m3 = ASDConcrete3D.from_fc(E=E, v=V, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        assert m1.Te == m3.Te and m1.Ts == m3.Ts and m1.Ce == m3.Ce

    def test_emit_card_has_no_continuum_tokens(self) -> None:
        m = ASDConcrete1D.from_fc(E=E, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        em = RecordingEmitter()
        m._emit(em, tag=3)
        (name, args, _), = em.calls
        assert name == "uniaxialMaterial"
        assert args[0] == "ASDConcrete1D" and args[1] == 3
        assert args[2] == E                      # E only, no v
        assert "-Kc" not in args and "-cdf" not in args and "-rho" not in args
        i = args.index("-autoRegularization")
        assert args[i + 1] == 50.0

    @pytest.mark.parametrize("kwargs,msg", [
        ({"E": 0.0, "fc": FC}, "E must be > 0"),
        ({"E": E, "fc": -1.0}, "fc must be > 0"),
        ({"E": E, "fc": FC, "Gf": -0.1}, "Gf must be > 0"),
    ])
    def test_validation(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            ASDConcrete1D.from_fc(**kwargs)

    def test_element_size_guard_warns(self) -> None:
        m = ASDConcrete1D.from_fc(E=E, fc=FC, ft=FT, Gf=GF, lch_ref=50.0)
        with pytest.warns(ASDRegularizationWarning, match="snapback ceiling"):
            m.check_element_size(m.l_max() * 2.0)

    def test_bridge_registers(self) -> None:
        ops = apeSees(cast("object", MagicMock(name="FEMData")))
        m = ops.uniaxialMaterial.ASDConcrete1D(E=E, fc=FC, ft=FT, Gf=GF,
                                               lch_ref=50.0)
        assert isinstance(m, ASDConcrete1D)
        assert ops.tag_for(m) == 1
