"""DR-6 turnkey concrete calibration: oracles, unit invariance and guards.

Covers:

1. fib MC2010 / EN 1992-1-1 values for a C30 concrete (literature oracle).
2. Hand-derived crack-band snap-back lengths for the fork's two
   LadrunoConcrete3D tension laws, and for a hand-built RC backbone under
   the fork's ``regularize`` floor.
3. Unit invariance across five explicit unit presets (no magnitude
   inference).
4. LadrunoConcrete3D always emits ``-Df`` / ``-hardening``, keeps the fork's
   constructor defaults, and no longer carries flags the pinned fork does
   not parse.
5. ``_LadrunoRC.from_fc`` keeps its legacy calibration unless the caller
   opts into ``calibration="mc2010"``.
6. The backbone generator refuses a point with damage >= 1.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material import _asdconcrete_laws as laws
from apeGmsh.opensees.material._asdconcrete_laws import (
    GC_OVER_GF,
    UNIT_PRESETS,
    ceb_fip_Gf,
    fib_mc2010_concrete_params,
    rc_regularization_lch_max,
)
from apeGmsh.opensees.material.nd import (
    ASDConcrete3D,
    LadrunoConcrete3D,
    LadrunoRCConcrete,
)

PRESETS_5 = ["MPa-mm-N", "Pa-m-N", "kPa-m-kN", "ksi-in-kip", "psi-in-lbf"]


# ---------------------------------------------------------------------------
# 1. Literature oracle: C30 (fck = 30 MPa, fcm = 38 MPa)
# ---------------------------------------------------------------------------

def test_mc2010_c30_matches_published_tables() -> None:
    p = fib_mc2010_concrete_params(38.0, units="MPa-mm-N")
    assert p.fck == pytest.approx(30.0)
    # fib MC2010 Table 5.1-2: C30 fctm = 2.9 MPa.
    assert p.ft == pytest.approx(2.9, abs=0.01)
    # fib MC2010 Table 5.1-7: C30 Eci = 33.6 GPa (quartzite aggregate).
    assert p.E == pytest.approx(33_600.0, rel=2e-3)
    # fib MC2010 Eq. 5.1-9: GF = 73 * 38^0.18 N/m = 140.5 N/m.
    assert p.Gf == pytest.approx(0.1405, abs=5e-4)
    # EN 1992-1-1 Table 3.1: C30/37 eps_c1 = 2.2 per mille.
    assert p.ec == pytest.approx(2.2e-3, abs=5e-5)
    # Nakamura & Higai ratio, not a Model Code value.
    assert p.Gc == pytest.approx(GC_OVER_GF * p.Gf)
    # CDPM2 (Grassl et al. 2013) calibration.
    assert (p.Df, p.hardening, p.As) == (0.85, (0.3, 0.01), 15.0)


def test_mc2010_aggregate_factor_scales_only_the_modulus() -> None:
    q = fib_mc2010_concrete_params(38.0)
    b = fib_mc2010_concrete_params(38.0, aggregate="basalt")
    assert b.E == pytest.approx(1.2 * q.E)  # MC2010 Table 5.1-6
    assert (b.ft, b.Gf) == (q.ft, q.Gf)


def test_eps_c1_is_capped_at_ec2_value() -> None:
    p = fib_mc2010_concrete_params(98.0)  # C90: EC2 caps eps_c1 at 2.8 per mille
    assert p.ec == pytest.approx(max(2.8e-3, 1.6 * p.fc / p.E))


# ---------------------------------------------------------------------------
# 2. Hand-derived snap-back lengths
# ---------------------------------------------------------------------------

def _h_max_from_initial_slope(E: float, slope: float) -> float:
    # A band of width h snaps back once |d sigma / d w| > E / h.
    return E / abs(slope)


def test_exponential_law_snapback_length_is_E_Gf_over_ft2() -> None:
    m = LadrunoConcrete3D(E=30_000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.12,
                          Gc=30.0, tension_law="exp")
    # sigma(w) = ft exp(-w ft / Gf)  (LadrunoConcrete3DKernel.h, law 0)
    dw = 1e-9
    slope = (3.0 * math.exp(-dw * 3.0 / 0.12) - 3.0) / dw
    h = _h_max_from_initial_slope(30_000.0, slope)
    assert h == pytest.approx(30_000.0 * 0.12 / 9.0, rel=1e-6)  # = 400 mm
    l_t, _ = m.l_max()
    assert l_t == pytest.approx(h, rel=1e-6)
    assert l_t == pytest.approx(400.0, rel=1e-6)


@pytest.mark.parametrize("law", ["bilinear", None])
def test_bilinear_law_snapback_length(law: str | None) -> None:
    m = LadrunoConcrete3D(E=30_000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.12,
                          Gc=30.0, tension_law=law)
    # CDPM2 bilinear: ft -> 0.3 ft at w1 = 0.15 wf, wf = Gf / (0.225 ft).
    wf = 0.12 / (0.225 * 3.0)
    slope = (3.0 - 0.3 * 3.0) / (0.15 * wf)
    h = _h_max_from_initial_slope(30_000.0, slope)
    l_t, _ = m.l_max()
    assert l_t == pytest.approx(h, rel=1e-12)
    assert l_t == pytest.approx(400.0 * 20.0 / 21.0, rel=1e-12)


def test_c30_from_fc_snapback_length() -> None:
    m = LadrunoConcrete3D.from_fc(38.0)
    l_t, l_c = m.l_max()
    # (20/21) * 33550.6 * 0.140502 / 2.89647^2, by hand.
    assert l_t == pytest.approx(535.13, abs=0.05)
    assert l_c == pytest.approx(
        2.0 * m.E * m.Gc / (m.fc ** 2 * 1.01), rel=1e-12)  # type: ignore[operator]


def test_rc_regularization_floor_hand_table() -> None:
    # E = 3e4, peak (1e-4, 3), linear softening to zero at 1e-3, no damage.
    # g0 = 3^2 / (2 E) + (1e-3 - 1e-4) * 3 / 2 = 1.5e-3
    # gmin = 1.01 * 3 * 1e-4 / 2 = 1.515e-4  ->  l_max = lref * 9.90099
    x, y = (0.0, 1e-4, 1e-3), (0.0, 3.0, 0.0)
    got = rc_regularization_lch_max(x, y, (), 3.0e4, 50.0)
    assert got == pytest.approx(50.0 * 1.5e-3 / 1.515e-4, rel=1e-12)
    assert rc_regularization_lch_max((0.0, 1e-4), (0.0, 3.0), (), 3e4, 50.0) \
        == math.inf


def test_rc_l_max_reads_the_emitted_tables() -> None:
    rc = LadrunoRCConcrete(
        E=3.0e4, nu=0.2,
        Te=(0.0, 1e-4, 1e-3), Ts=(0.0, 3.0, 0.0),
        Ce=(0.0, 1e-3, 1e-2), Cs=(0.0, 30.0, 0.0),
        auto_regularization=50.0,
    )
    l_t, l_c = rc.l_max()
    assert l_t == pytest.approx(50.0 * 1.5e-3 / 1.515e-4, rel=1e-12)
    g0_c = 30.0 ** 2 / (2 * 3e4) + 9e-3 * 30.0 / 2
    assert l_c == pytest.approx(50.0 * g0_c / (1.01 * 30.0 * 1e-3 / 2), rel=1e-12)
    with pytest.raises(ValueError, match="needs auto_regularization"):
        LadrunoRCConcrete(E=3e4, nu=0.2, Te=rc.Te, Ts=rc.Ts,
                          Ce=rc.Ce, Cs=rc.Cs).l_max()


def test_check_element_size_pass_and_raise() -> None:
    conc = LadrunoConcrete3D.from_fc(38.0)
    ceil = min(conc.l_max())
    conc.check_element_size(ceil)
    with pytest.raises(ValueError, match=r"exceeds snap-back ceiling.*\(PG 'Core'\)"):
        conc.check_element_size(1.01 * ceil, pg="Core")
    with pytest.raises(ValueError, match="h_max must be > 0"):
        conc.check_element_size(-5.0)


# ---------------------------------------------------------------------------
# 3. Explicit units, no magnitude inference
# ---------------------------------------------------------------------------

def test_ceb_fip_Gf_unit_invariance() -> None:
    ref = 0.073 * 30.0 ** 0.18
    for u in PRESETS_5:
        s_sig, s_len = UNIT_PRESETS[u]
        assert ceb_fip_Gf(30.0 * s_sig, units=u) / (s_sig * s_len) \
            == pytest.approx(ref, rel=1e-12)


def test_default_units_never_switch_on_magnitude() -> None:
    # The historical rule, unchanged: fc is read in MPa whatever its size.
    assert ceb_fip_Gf(30.0e6) == pytest.approx(0.073 * (30.0e6) ** 0.18)
    # A psi user passing fc=4000 must declare it.
    p = fib_mc2010_concrete_params(4000.0, units="psi-in-lbf")
    assert p.fc == pytest.approx(4000.0)
    assert p.fcm / UNIT_PRESETS["psi-in-lbf"][0] == pytest.approx(27.579, abs=1e-3)
    with pytest.raises(ValueError, match="unknown units"):
        fib_mc2010_concrete_params(30.0, units="MPa-cm")


def test_asdconcrete3d_from_fc_default_is_unchanged() -> None:
    m = ASDConcrete3D.from_fc(E=30e9, v=0.2, fc=30e6)
    assert m.Gf == pytest.approx(0.073 * (30e6) ** 0.18)


@pytest.mark.parametrize("u", PRESETS_5)
def test_turnkey_unit_invariance(u: str) -> None:
    s_sig, s_len = UNIT_PRESETS[u]
    ref3 = LadrunoConcrete3D.from_fc(38.0)
    cur3 = LadrunoConcrete3D.from_fc(38.0 * s_sig, units=u)
    assert cur3.E / s_sig == pytest.approx(ref3.E, rel=1e-12)
    assert cur3.Gf / (s_sig * s_len) == pytest.approx(ref3.Gf, rel=1e-12)
    for a, b in zip(cur3.l_max(), ref3.l_max()):
        assert a / s_len == pytest.approx(b, rel=1e-12)

    refrc = LadrunoRCConcrete.from_fc(38.0, calibration="mc2010")
    currc = LadrunoRCConcrete.from_fc(38.0 * s_sig, calibration="mc2010", units=u)
    for a, b in zip(currc.l_max(), refrc.l_max()):
        assert a / s_len == pytest.approx(b, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. LadrunoConcrete3D deck contract
# ---------------------------------------------------------------------------

def _args(m: LadrunoConcrete3D) -> tuple[object, ...]:
    rec = RecordingEmitter()
    m._emit(rec, tag=1)
    return tuple(rec.calls[0][1])


def test_raw_defaults_are_the_fork_defaults_and_always_emitted() -> None:
    # Fork pin, LadrunoConcrete3D.cpp: Df = 1.0, qh0 = 0.3, Hp = 0.5.
    m = LadrunoConcrete3D(E=30000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.13, Gc=32.5)
    assert (m.Df, m.hardening) == (1.0, (0.3, 0.5))
    a = _args(m)
    assert a[8:] == ("-Df", 1.0, "-hardening", 0.3, 0.5)


def test_from_fc_emits_the_cdpm2_values_it_reports() -> None:
    m = LadrunoConcrete3D.from_fc(38.0)
    a = _args(m)
    i, j = a.index("-Df"), a.index("-hardening")
    assert a[i:i + 2] == ("-Df", 0.85)
    assert a[j:j + 3] == ("-hardening", 0.3, 0.01)
    assert a[a.index("-As") + 1] == 15.0
    assert "-flowPotential" not in a  # the pinned fork does not parse it


def test_gc_none_emits_the_from_fc_ratio() -> None:
    m = LadrunoConcrete3D(E=30000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.13)
    assert _args(m)[7] == pytest.approx(GC_OVER_GF * 0.13)


def test_flags_the_pinned_fork_does_not_parse_are_gone() -> None:
    base = dict(E=30000.0, nu=0.2, fc=30.0, ft=3.0, Gf=0.13, Gc=32.5)
    for flag in ("sym_tangent", "cyclic_hysteresis"):
        with pytest.raises(TypeError, match=flag):
            LadrunoConcrete3D(**base, **{flag: True})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="rebar"):
        LadrunoRCConcrete.from_fc(
            38.0, calibration="mc2010", rebar=[(0.01, 0.0, 420.0, 2e5)])
    with pytest.raises(ValueError, match="2-tuple"):
        LadrunoConcrete3D(**base, hardening=(0.3, 0.5, 1.0, 2.0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. _LadrunoRC.from_fc back-compat
# ---------------------------------------------------------------------------

def test_rc_from_fc_legacy_is_the_historical_rule() -> None:
    E, fc = 30000.0, 30.0
    m = LadrunoRCConcrete.from_fc(E=E, nu=0.2, fc=fc)
    ft = 0.1 * fc
    Gf = 0.073 * fc ** 0.18
    Gc = laws.ceb_fip_Gc(fc, ft, Gf)
    lch = laws.auto_lch_ref(E, fc, ft, Gf, Gc)
    Te, Ts, Td = laws.make_tension(E, ft, Gf, lch)
    Ce, Cs, Cd = laws.make_compression(E, fc, Gc, lch)
    assert (m.Te, m.Ts, m.Td) == (tuple(Te), tuple(Ts), tuple(Td))
    assert (m.Ce, m.Cs, m.Cd) == (tuple(Ce), tuple(Cs), tuple(Cd))
    assert m.auto_regularization == lch
    assert (m.ft, m.Gf, m.Gc, m.epsc0) == (ft, Gf, Gc, None)


def test_rc_from_fc_legacy_needs_E_and_nu() -> None:
    with pytest.raises(ValueError, match="calibration='mc2010'"):
        LadrunoRCConcrete.from_fc(30.0)
    with pytest.raises(ValueError, match="calibration must be"):
        LadrunoRCConcrete.from_fc(30.0, E=3e4, nu=0.2, calibration="aci")


def test_rc_from_fc_mc2010_opt_in() -> None:
    m = LadrunoRCConcrete.from_fc(38.0, calibration="mc2010")
    p = fib_mc2010_concrete_params(38.0)
    assert (m.E, m.nu, m.ft, m.Gf, m.Gc, m.epsc0) == (
        p.E, p.nu, p.ft, p.Gf, p.Gc, p.ec)
    assert LadrunoRCConcrete.from_fc(
        38.0, calibration="mc2010", ec=0.0028).ec == 0.0028


# ---------------------------------------------------------------------------
# 6. Damage >= 1 is refused, not clamped
# ---------------------------------------------------------------------------

def test_backbone_refuses_full_damage() -> None:
    with pytest.raises(ValueError, match="reaches damage"):
        laws.make_compression(30000.0, 30.0, 50.0, 100.0, fcr=0.0)
