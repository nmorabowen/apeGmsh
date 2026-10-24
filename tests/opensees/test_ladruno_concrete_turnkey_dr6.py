"""Unit-invariance, calibration and guard verification for DR-6 (Turnkey Concrete).

Tests:
1. 5-unit-system invariance (MPa-mm-N, Pa-m-N, kPa-m-kN, ksi-in-kip, psi-in-lbf)
   for ceb_fip_Gf, LadrunoConcrete3D.from_fc, LadrunoRCConcrete.from_fc, and l_max().
2. Dimensionless pi-groups:
   Pi_t = Gf * Ec / (ft^2 * l_max_t) == 21/20 to < 1e-12
   Pi_c = Gc * Ec / (fc^2 * l_max_c) == (1 + Hp)/2 to < 1e-12
3. check_element_size(h_max) pass/raise behavior.
4. effective_rebar_lch(l_ch, d_b) Grassl 2018 floor.
5. New flags emission (-symTangent, -cyclicHysteresis, -rebar).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.resolvers._reinforce import resolve_reinforce
from apeGmsh.opensees.element import effective_rebar_lch
from apeGmsh.opensees.element._reinforce import effective_rebar_lch as eff_lch_direct
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material._asdconcrete_laws import (
    UNIT_PRESETS,
    ceb_fip_Gf,
    fib_mc2010_concrete_params,
)
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    LadrunoConcrete3D,
    LadrunoRCConcrete,
)

PRESETS_5 = ["MPa-mm-N", "Pa-m-N", "kPa-m-kN", "ksi-in-kip", "psi-in-lbf"]


def test_effective_rebar_lch_import_parity() -> None:
    assert effective_rebar_lch is eff_lch_direct


def test_effective_rebar_lch_grassl_floor() -> None:
    # Coarse mesh (l_ch > 1.5 d_b): keeps l_ch
    assert effective_rebar_lch(50.0, 20.0) == 50.0
    # Fine mesh (l_ch < 1.5 d_b): floored at 1.5 * d_b
    assert effective_rebar_lch(20.0, 20.0) == 30.0
    # Boundary (l_ch == 1.5 d_b)
    assert effective_rebar_lch(30.0, 20.0) == 30.0

    with pytest.raises(ValueError, match="l_ch must be > 0"):
        effective_rebar_lch(0.0, 20.0)
    with pytest.raises(ValueError, match="l_ch must be > 0"):
        effective_rebar_lch(-10.0, 20.0)
    with pytest.raises(ValueError, match="d_b must be > 0"):
        effective_rebar_lch(50.0, 0.0)
    with pytest.raises(ValueError, match="d_b must be > 0"):
        effective_rebar_lch(50.0, -5.0)


def test_ceb_fip_Gf_unit_invariance() -> None:
    fc_base_mpa = 30.0
    gf_ref_n_mm = 0.073 * (fc_base_mpa ** 0.18)

    for u_name in PRESETS_5:
        s_sig, s_len = UNIT_PRESETS[u_name]
        s_gf = s_sig * s_len
        fc_val = fc_base_mpa * s_sig

        gf_val = ceb_fip_Gf(fc_val, units=u_name)
        gf_in_n_mm = gf_val / s_gf
        assert abs(gf_in_n_mm - gf_ref_n_mm) < 1.0e-12

        # Dimensionless ratio check: Gf / (ft * l_ch)
        ft_val = 0.1 * fc_val
        lch_val = 50.0 * s_len
        dimless = gf_val / (ft_val * lch_val)
        ref_dimless = gf_ref_n_mm / ((0.1 * fc_base_mpa) * 50.0)
        assert abs(dimless - ref_dimless) < 1.0e-14


def test_fib_mc2010_params_invariance() -> None:
    fc_base_mpa = 30.0
    ref = fib_mc2010_concrete_params(fc_base_mpa, units="MPa-mm-N")

    for u_name in PRESETS_5:
        s_sig, s_len = UNIT_PRESETS[u_name]
        s_gf = s_sig * s_len
        cur = fib_mc2010_concrete_params(fc_base_mpa * s_sig, units=u_name)

        assert abs(cur.E / s_sig - ref.E) < 1.0e-11
        assert abs(cur.ft / s_sig - ref.ft) < 1.0e-11
        assert abs(cur.Gf / s_gf - ref.Gf) < 1.0e-12
        assert abs(cur.Gc / s_gf - ref.Gc) < 1.0e-11
        assert abs(cur.ec - ref.ec) < 1.0e-15
        assert abs(cur.e - ref.e) < 1.0e-15
        assert cur.Df == ref.Df
        assert cur.hardening == ref.hardening

        # Snap-back lengths scaled to mm
        assert abs(cur.l_max_t / s_len - ref.l_max_t) < 1.0e-11
        assert abs(cur.l_max_c / s_len - ref.l_max_c) < 1.0e-11

        # Dimensionless Pi-groups: exact analytical constants
        pi_t = cur.Gf * cur.E / ((cur.ft ** 2) * cur.l_max_t)
        hp = cur.hardening[1]
        pi_c = cur.Gc * cur.E / ((cur.fc ** 2) * cur.l_max_c)

        assert abs(pi_t - 21.0 / 20.0) < 1.0e-12
        assert abs(pi_c - (1.0 + hp) / 2.0) < 1.0e-12


def test_ladruno_concrete3d_from_fc_invariance() -> None:
    fc_base_mpa = 30.0
    ref = LadrunoConcrete3D.from_fc(fc_base_mpa, units="MPa-mm-N")
    ref_l_max_t, ref_l_max_c = ref.l_max()

    for u_name in PRESETS_5:
        s_sig, s_len = UNIT_PRESETS[u_name]
        s_gf = s_sig * s_len
        cur = LadrunoConcrete3D.from_fc(fc_base_mpa * s_sig, units=u_name)

        assert abs(cur.E / s_sig - ref.E) < 1.0e-11
        assert abs(cur.ft / s_sig - ref.ft) < 1.0e-11
        assert abs(cur.Gf / s_gf - ref.Gf) < 1.0e-12
        assert abs((cur.Gc or 0.0) / s_gf - (ref.Gc or 0.0)) < 1.0e-11
        assert cur.Df == 0.85
        assert cur.hardening == (0.3, 0.01)

        l_max_t, l_max_c = cur.l_max()
        assert abs(l_max_t / s_len - ref_l_max_t) < 1.0e-11
        assert abs(l_max_c / s_len - ref_l_max_c) < 1.0e-11

        # Pi-groups
        pi_t = cur.Gf * cur.E / ((cur.ft ** 2) * l_max_t)
        hp = float(cur.hardening[1])
        gc = cur.Gc if cur.Gc is not None else 150.0 * cur.Gf
        pi_c = gc * cur.E / ((cur.fc ** 2) * l_max_c)

        assert abs(pi_t - 21.0 / 20.0) < 1.0e-12
        assert abs(pi_c - (1.0 + hp) / 2.0) < 1.0e-12


def test_ladruno_rc_concrete_from_fc_invariance() -> None:
    fc_base_mpa = 30.0
    ref = LadrunoRCConcrete.from_fc(fc_base_mpa, units="MPa-mm-N")
    ref_l_max_t, ref_l_max_c = ref.l_max()

    for u_name in PRESETS_5:
        s_sig, s_len = UNIT_PRESETS[u_name]
        s_gf = s_sig * s_len
        cur = LadrunoRCConcrete.from_fc(fc_base_mpa * s_sig, units=u_name)

        assert abs(cur.E / s_sig - ref.E) < 1.0e-11
        assert abs(cur.nu - ref.nu) < 1.0e-15
        assert cur.ft is not None and ref.ft is not None
        assert abs(cur.ft / s_sig - ref.ft) < 1.0e-11
        assert cur.Gf is not None and ref.Gf is not None
        assert abs(cur.Gf / s_gf - ref.Gf) < 1.0e-12
        assert cur.Gc is not None and ref.Gc is not None
        assert abs(cur.Gc / s_gf - ref.Gc) < 1.0e-11
        assert cur.fc is not None and ref.fc is not None

        l_max_t, l_max_c = cur.l_max()
        assert abs(l_max_t / s_len - ref_l_max_t) < 1.0e-11
        assert abs(l_max_c / s_len - ref_l_max_c) < 1.0e-11

        # Pi-groups (LadrunoRCConcrete default tension_law="exp" uses factor 2.0 -> Pi_t = 1/2)
        pi_t = cur.Gf * cur.E / ((cur.ft ** 2) * l_max_t)
        hp = 0.01
        pi_c = cur.Gc * cur.E / ((cur.fc ** 2) * l_max_c)

        assert abs(pi_t - 0.5) < 1.0e-12
        assert abs(pi_c - (1.0 + hp) / 2.0) < 1.0e-12


def test_check_element_size_pass_and_raise() -> None:
    conc = LadrunoConcrete3D.from_fc(30.0)
    l_max_t, l_max_c = conc.l_max()
    l_ceil = min(l_max_t, l_max_c)

    # Within bounds: passes without error
    conc.check_element_size(0.99 * l_ceil)
    conc.check_element_size(l_ceil)

    # Exceeding bounds: raises ValueError
    with pytest.raises(ValueError, match="exceeds snap-back ceiling"):
        conc.check_element_size(1.01 * l_ceil)

    # Exceeding bounds with pg: message includes PG
    with pytest.raises(ValueError, match=r"exceeds snap-back ceiling.*\(PG 'Core'\)"):
        conc.check_element_size(1.01 * l_ceil, pg="Core")

    with pytest.raises(ValueError, match="h_max must be > 0"):
        conc.check_element_size(-5.0)

    # RC concrete
    rc = LadrunoRCConcrete.from_fc(30.0)
    rc_t, rc_c = rc.l_max()
    rc_ceil = min(rc_t, rc_c)

    rc.check_element_size(0.95 * rc_ceil)
    with pytest.raises(ValueError, match="exceeds snap-back ceiling"):
        rc.check_element_size(1.05 * rc_ceil)

    with pytest.raises(ValueError, match=r"exceeds snap-back ceiling.*\(PG 'Shell'\)"):
        rc.check_element_size(1.05 * rc_ceil, pg="Shell")


def test_ladruno_concrete3d_flags_emission() -> None:
    # Default emission omits -symTangent and -cyclicHysteresis
    conc_def = LadrunoConcrete3D.from_fc(30.0)
    rec_def = RecordingEmitter()
    conc_def._emit(rec_def, tag=10)
    args_def = rec_def.calls[0][1]
    assert "-symTangent" not in args_def
    assert "-cyclicHysteresis" not in args_def
    assert "-Df" not in args_def
    assert "-hardening" not in args_def

    # Explicit opt-in emits the flags
    rec = RecordingEmitter()
    conc = LadrunoConcrete3D.from_fc(
        30.0,
        sym_tangent=True,
        cyclic_hysteresis=True,
    )
    conc._emit(rec, tag=1)
    args = rec.calls[0][1]

    assert "-symTangent" in args
    assert "-cyclicHysteresis" in args
    # Defaults 0.85 and (0.3, 0.01) match C++ defaults and are omitted from emit
    assert "-Df" not in args
    assert "-hardening" not in args

    # Non-defaults are emitted
    rec2 = RecordingEmitter()
    conc2 = LadrunoConcrete3D.from_fc(
        30.0,
        Df=0.95,
        hardening=(0.4, 0.02),
    )
    conc2._emit(rec2, tag=2)
    args2 = rec2.calls[0][1]
    assert "-Df" in args2 and 0.95 in args2
    assert "-hardening" in args2 and (0.4, 0.02) == (
        args2[args2.index("-hardening") + 1],
        args2[args2.index("-hardening") + 2],
    )


def test_ladruno_rc_concrete_rebar_emission() -> None:
    # Default emission omits -rebar
    rc_def = LadrunoRCConcrete.from_fc(30.0)
    rec_def = RecordingEmitter()
    rc_def._emit(rec_def, tag=20)
    args_def = rec_def.calls[0][1]
    assert "-rebar" not in args_def

    # Explicit rebar layers are emitted
    rec = RecordingEmitter()
    rc = LadrunoRCConcrete.from_fc(
        30.0,
        rebar=[(0.012, 0.0, 420.0, 200000.0), (0.008, 90.0, 420.0, 200000.0)],
    )
    rc._emit(rec, tag=2)
    args = list(rec.calls[0][1])

    assert args.count("-rebar") == 2
    idx1 = args.index("-rebar")
    assert args[idx1:idx1 + 5] == ["-rebar", 0.012, 0.0, 420.0, 200000.0]


def test_ec_epsc0_attributes_and_overrides() -> None:
    # LadrunoConcrete3D ec and epsc0 alias
    conc = LadrunoConcrete3D.from_fc(30.0)
    assert conc.ec is not None and conc.ec > 0.0
    assert conc.epsc0 == conc.ec

    # Override ec in from_fc kwargs
    conc_over = LadrunoConcrete3D.from_fc(30.0, ec=0.0025)
    assert conc_over.ec == 0.0025
    assert conc_over.epsc0 == 0.0025
    assert conc_over.ductility[1] == pytest.approx(0.0025 - 30.0 / conc_over.E)

    # LadrunoRCConcrete epsc0 and ec alias
    rc = LadrunoRCConcrete.from_fc(30.0)
    assert rc.epsc0 is not None and rc.epsc0 > 0.0
    assert rc.ec == rc.epsc0

    rc_over = LadrunoRCConcrete.from_fc(30.0, ec=0.0028)
    assert rc_over.epsc0 == 0.0028
    assert rc_over.ec == 0.0028


def test_rebar_kwargs_normalization() -> None:
    # Passed via kwargs as nested lists
    rc = LadrunoRCConcrete.from_fc(
        30.0,
        rebar=[[0.015, 0.0, 400.0, 200000.0]],
    )
    assert rc.rebar == ((0.015, 0.0, 400.0, 200000.0),)


def test_ceb_fip_Gf_legacy_pa_autodetect() -> None:
    # 30 MPa in Pa (30e6) without units specified
    gf_auto = ceb_fip_Gf(30.0e6)
    gf_explicit = ceb_fip_Gf(30.0e6, units="Pa-m-N")
    assert abs(gf_auto - gf_explicit) < 1.0e-12
    assert abs(gf_auto - 134.6) < 0.1


def test_ceb_fip_Gf_and_params_2token_alias_pa_autodetect() -> None:
    # 30 MPa in Pa (30e6) with units="MPa-mm" alias
    gf_alias = ceb_fip_Gf(30.0e6, units="MPa-mm")
    gf_explicit = ceb_fip_Gf(30.0e6, units="Pa-m-N")
    assert abs(gf_alias - gf_explicit) < 1.0e-12
    assert abs(gf_alias - 134.6) < 0.1

    params_alias = fib_mc2010_concrete_params(30.0e6, units="MPa-mm")
    params_explicit = fib_mc2010_concrete_params(30.0e6, units="Pa-m-N")
    assert abs(params_alias.E - params_explicit.E) < 1.0e-9
    assert abs(params_alias.Gf - params_explicit.Gf) < 1.0e-12


def test_ladruno_concrete3d_raw_instantiation_defaults_match_cpp() -> None:
    # Raw dataclass instantiation matches C++ defaults (Df=0.85, hardening=(0.3, 0.01))
    conc = LadrunoConcrete3D(
        E=30000.0,
        nu=0.2,
        fc=30.0,
        ft=3.0,
        Gf=0.13,
        Gc=32.5,
    )
    assert conc.Df == 0.85
    assert conc.hardening == (0.3, 0.01)

    # l_max_c uses hp = 0.01 (matching C++ Hp = 0.01)
    # l_max_c = 2 * E * Gc / (fc^2 * (1 + 0.01))
    expected_l_max_c = 2.0 * 30000.0 * 32.5 / (30.0 ** 2 * 1.01)
    _, l_max_c = conc.l_max()
    assert abs(l_max_c - expected_l_max_c) < 1.0e-11

    # Minimal emission omits default Df and hardening
    rec = RecordingEmitter()
    conc._emit(rec, tag=1)
    args = rec.calls[0][1]
    assert "-Df" not in args
    assert "-hardening" not in args


def test_tension_law_sensitivity_l_max() -> None:
    # LadrunoConcrete3D: bilinear -> 20/21, exp/linear -> 2.0
    conc_bilin = LadrunoConcrete3D.from_fc(30.0, tension_law="bilinear")
    l_t_bilin, _ = conc_bilin.l_max()
    pi_t_bilin = conc_bilin.Gf * conc_bilin.E / ((conc_bilin.ft ** 2) * l_t_bilin)
    assert abs(pi_t_bilin - 21.0 / 20.0) < 1.0e-12

    conc_exp = LadrunoConcrete3D.from_fc(30.0, tension_law="exp")
    l_t_exp, _ = conc_exp.l_max()
    pi_t_exp = conc_exp.Gf * conc_exp.E / ((conc_exp.ft ** 2) * l_t_exp)
    assert abs(pi_t_exp - 0.5) < 1.0e-12
    assert abs(l_t_exp / l_t_bilin - 2.0 / (20.0 / 21.0)) < 1.0e-12

    # LadrunoRCConcrete: default ("exp") -> 2.0, bilinear -> 20/21
    rc_exp = LadrunoRCConcrete.from_fc(30.0, tension_law="exp")
    rc_l_t_exp, _ = rc_exp.l_max()
    assert abs(rc_exp.Gf * rc_exp.E / ((rc_exp.ft ** 2) * rc_l_t_exp) - 0.5) < 1.0e-12

    rc_bilin = LadrunoRCConcrete.from_fc(30.0, tension_law="bilinear")
    rc_l_t_bilin, _ = rc_bilin.l_max()
    assert abs(rc_bilin.Gf * rc_bilin.E / ((rc_bilin.ft ** 2) * rc_l_t_bilin) - 21.0 / 20.0) < 1.0e-12


def test_asdconcrete3d_from_fc_units() -> None:
    # In MPa-mm-N: fc = 30.0 MPa
    c_mpa = ASDConcrete3D.from_fc(
        E=30000.0, v=0.2, fc=30.0, units="MPa-mm-N"
    )
    # In kPa-m-kN: fc = 30000.0 kPa (30 MPa), E = 30,000,000 kPa
    c_kpa = ASDConcrete3D.from_fc(
        E=30000.0 * 1000.0, v=0.2, fc=30000.0, units="kPa-m-kN"
    )
    # 1 N/mm == 1 kN/m, so Gf values should match across units
    assert abs(c_kpa.Gf - c_mpa.Gf) < 1.0e-12


def test_resolve_reinforce_invokes_effective_rebar_lch() -> None:
    hex_ids = [1, 2, 3, 4, 5, 6, 7, 8]
    # Small 10 mm cube (nominal l_ch = 10.0)
    hex_xyz = np.array([
        [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10],
    ], dtype=float)
    bar_ids = [10, 11]
    bar_xyz = np.array([[2, 5, 5], [8, 5, 5]], dtype=float)
    bar_seg = [(10, 11)]

    # Case 1: Fine mesh (h = 10.0 mm < 1.5 * d_b = 30.0 mm for d_b = 20.0 mm)
    recs = resolve_reinforce(
        bar_node_ids=bar_ids,
        bar_node_coords=bar_xyz,
        bar_segments=bar_seg,
        host_node_ids=[hex_ids],
        host_node_coords=[hex_xyz],
        host_kinds=["hex8"],
        perfect=1.0e8,
        diameter=20.0,
    )
    assert len(recs) == 2
    for r in recs:
        assert r.effective_lch == pytest.approx(30.0)
        assert r.lch_eff == pytest.approx(30.0)

    # Case 2: Coarse mesh (h = 100.0 mm > 1.5 * d_b = 30.0 mm)
    recs_coarse = resolve_reinforce(
        bar_node_ids=bar_ids,
        bar_node_coords=bar_xyz,
        bar_segments=bar_seg,
        host_node_ids=[hex_ids],
        host_node_coords=[hex_xyz * 10.0],
        host_kinds=["hex8"],
        perfect=1.0e8,
        diameter=20.0,
    )
    for r in recs_coarse:
        assert r.effective_lch == pytest.approx(100.0)
        assert r.lch_eff == pytest.approx(100.0)


def test_ns_wrappers_units_and_ladruno_flags() -> None:
    from unittest.mock import MagicMock
    from apeGmsh.opensees.apesees import apeSees

    ops = apeSees(MagicMock(name="FEMData"))
    # Test ops.nDMaterial.ASDConcrete3D with units
    mat_3d = ops.nDMaterial.ASDConcrete3D(
        E=30000.0, v=0.2, fc=30.0, units="MPa-mm-N", name="c3d"
    )
    assert mat_3d is not None

    # Test ops.uniaxialMaterial.ASDConcrete1D with units
    mat_1d = ops.uniaxialMaterial.ASDConcrete1D(
        E=30000.0, fc=30.0, units="MPa-mm-N", name="c1d"
    )
    assert mat_1d is not None

    # Test ops.nDMaterial.LadrunoConcrete3D with sym_tangent, cyclic_hysteresis, ec
    mat_l3d = ops.nDMaterial.LadrunoConcrete3D(
        E=30000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.13, Gc=32.5,
        sym_tangent=True, cyclic_hysteresis=True, ec=0.0025, name="l3d"
    )
    assert mat_l3d.sym_tangent is True
    assert mat_l3d.cyclic_hysteresis is True
    assert mat_l3d.ec == 0.0025


def test_ladruno_rc_tension_law_validation() -> None:
    # Valid tension laws
    for law in ("exp", "linear", "bilinear", "vc1986", None):
        rc = LadrunoRCConcrete.from_fc(30.0, tension_law=law)
        assert rc.tension_law == law

    # Invalid tension law raises ValueError
    with pytest.raises(ValueError, match="tension_law must be one of"):
        LadrunoRCConcrete.from_fc(30.0, tension_law="quadratic")

