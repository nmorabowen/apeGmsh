"""Analytic checks for the derived stress/strain scalar layer.

Each classical stress state has hand-computable invariants; these lock
the formulas (and the engineering-shear halving) against regressions.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results import _derived


def _stress(**comps: float) -> dict[str, np.ndarray]:
    """One point, one step: {stress_xx: [[v]], ...} shape (1, 1)."""
    return {f"stress_{k}": np.array([[v]], dtype=float) for k, v in comps.items()}


def _strain(**comps: float) -> dict[str, np.ndarray]:
    return {f"strain_{k}": np.array([[v]], dtype=float) for k, v in comps.items()}


def _val(name: str, columns: dict[str, np.ndarray], ndm: int = 3) -> float:
    out = _derived.compute(name, columns, ndm=ndm)
    assert out.shape == (1, 1)
    return float(out[0, 0])


# ---------------------------------------------------------------------
# Registry / plumbing
# ---------------------------------------------------------------------

def test_is_derived():
    assert _derived.is_derived("von_mises_stress")
    assert _derived.is_derived("principal_strain_2")
    assert not _derived.is_derived("stress_xx")
    assert not _derived.is_derived("equivalent_plastic_strain")  # material-state


def test_base_components_clip_by_ndm():
    assert _derived.base_components_for("von_mises_stress", ndm=3) == (
        "stress_xx", "stress_yy", "stress_zz",
        "stress_xy", "stress_yz", "stress_xz",
    )
    assert _derived.base_components_for("von_mises_stress", ndm=2) == (
        "stress_xx", "stress_yy", "stress_xy",
    )
    assert _derived.base_components_for("principal_strain_1", ndm=3)[0] == "strain_xx"


def test_compute_rejects_non_derived():
    with pytest.raises(ValueError):
        _derived.compute("stress_xx", _stress(xx=1.0), ndm=3)


def test_available_derived_completeness_rule():
    # Full 3-D tensor → stress-derived advertised, no strain.
    full3d = ["stress_xx", "stress_yy", "stress_zz",
              "stress_xy", "stress_yz", "stress_xz"]
    adv = _derived.available_derived(full3d)
    assert "von_mises_stress" in adv
    assert "principal_stress_1" in adv
    assert "von_mises_strain" not in adv

    # Genuine 2-D plane (in-plane trio) → still advertised.
    assert "von_mises_stress" in _derived.available_derived(
        ["stress_xx", "stress_yy", "stress_xy"],
    )

    # Partial tensor (one component) → NOT advertised (would zero-fill).
    assert _derived.available_derived(["stress_xx"]) == []

    # Strain trio advertises the strain set.
    adv_e = _derived.available_derived(
        ["strain_xx", "strain_yy", "strain_xy"],
    )
    assert "von_mises_strain" in adv_e
    assert "von_mises_stress" not in adv_e


# ---------------------------------------------------------------------
# Uniaxial stress σ_xx = S
# ---------------------------------------------------------------------

def test_uniaxial_stress():
    S = 7.0
    c = _stress(xx=S)
    assert _val("von_mises_stress", c) == pytest.approx(S)
    assert _val("j2_stress", c) == pytest.approx(S**2 / 3.0)
    assert _val("mean_stress", c) == pytest.approx(S / 3.0)
    assert _val("pressure_hydrostatic", c) == pytest.approx(-S / 3.0)
    assert _val("principal_stress_1", c) == pytest.approx(S)
    assert _val("principal_stress_2", c) == pytest.approx(0.0)
    assert _val("principal_stress_3", c) == pytest.approx(0.0)
    assert _val("tresca_stress", c) == pytest.approx(S)
    assert _val("max_shear_stress", c) == pytest.approx(S / 2.0)


# ---------------------------------------------------------------------
# Pure shear τ_xy = τ
# ---------------------------------------------------------------------

def test_pure_shear_stress():
    t = 5.0
    c = _stress(xy=t)
    assert _val("von_mises_stress", c) == pytest.approx(np.sqrt(3.0) * t)
    assert _val("j2_stress", c) == pytest.approx(t**2)
    assert _val("mean_stress", c) == pytest.approx(0.0)
    assert _val("principal_stress_1", c) == pytest.approx(t)
    assert _val("principal_stress_2", c) == pytest.approx(0.0, abs=1e-9)
    assert _val("principal_stress_3", c) == pytest.approx(-t)
    assert _val("tresca_stress", c) == pytest.approx(2.0 * t)
    assert _val("max_shear_stress", c) == pytest.approx(t)


# ---------------------------------------------------------------------
# Hydrostatic σ = p·I  (pure pressure, zero deviator)
# ---------------------------------------------------------------------

def test_hydrostatic_stress():
    p = 3.0
    c = _stress(xx=p, yy=p, zz=p)
    assert _val("von_mises_stress", c) == pytest.approx(0.0, abs=1e-9)
    assert _val("j2_stress", c) == pytest.approx(0.0, abs=1e-9)
    assert _val("mean_stress", c) == pytest.approx(p)
    assert _val("pressure_hydrostatic", c) == pytest.approx(-p)
    for k in ("principal_stress_1", "principal_stress_2", "principal_stress_3"):
        assert _val(k, c) == pytest.approx(p)
    assert _val("tresca_stress", c) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------
# Advanced invariants — J3, Lode angle, triaxiality (Phase 5)
# ---------------------------------------------------------------------

def _principal_stress_state(s1: float, s2: float, s3: float) -> dict:
    """Diagonal stress state (principal axes aligned with x/y/z)."""
    return _stress(xx=s1, yy=s2, zz=s3)


def test_lode_angle_triaxial_compression():
    # sigma1 > sigma2 = sigma3  →  +30 degrees.
    c = _principal_stress_state(2.0, -1.0, -1.0)
    assert _val("lode_angle", c) == pytest.approx(30.0)
    assert _val("j3_stress", c) == pytest.approx(2.0)   # det(diag(2,-1,-1))


def test_lode_angle_triaxial_extension():
    # sigma1 = sigma2 > sigma3  →  -30 degrees.
    c = _principal_stress_state(1.0, 1.0, -2.0)
    assert _val("lode_angle", c) == pytest.approx(-30.0)


def test_lode_angle_pure_shear_is_zero():
    # Middle principal at the mean → theta = 0.
    c = _principal_stress_state(1.0, 0.0, -1.0)
    assert _val("lode_angle", c) == pytest.approx(0.0, abs=1e-9)


def test_lode_angle_hydrostatic_is_nan():
    c = _principal_stress_state(3.0, 3.0, 3.0)
    assert np.isnan(_val("lode_angle", c))


def test_triaxiality_uniaxial_tension():
    # mean = S/3, von Mises = S  →  eta = 1/3.
    c = _stress(xx=6.0)
    assert _val("stress_triaxiality", c) == pytest.approx(1.0 / 3.0)


def test_triaxiality_hydrostatic_is_nan():
    c = _principal_stress_state(5.0, 5.0, 5.0)  # vм = 0
    assert np.isnan(_val("stress_triaxiality", c))


# ---------------------------------------------------------------------
# Engineering-shear correction — the subtle one
# ---------------------------------------------------------------------

def test_engineering_shear_strain_halved():
    # OpenSees reports strain_xy = engineering gamma. Tensor eps_xy = g/2,
    # so principal strains are +/- g/2, not +/- g.
    g = 0.02
    c = _strain(xy=g)
    assert _val("principal_strain_1", c) == pytest.approx(g / 2.0)
    assert _val("principal_strain_3", c) == pytest.approx(-g / 2.0)
    assert _val("max_shear_strain", c) == pytest.approx(g / 2.0)
    assert _val("volumetric_strain", c) == pytest.approx(0.0, abs=1e-12)


def test_uniaxial_strain():
    e = 0.01
    c = _strain(xx=e)
    assert _val("volumetric_strain", c) == pytest.approx(e)
    assert _val("principal_strain_1", c) == pytest.approx(e)
    assert _val("principal_strain_2", c) == pytest.approx(0.0, abs=1e-12)
    assert _val("principal_strain_3", c) == pytest.approx(0.0, abs=1e-12)
    # eps_vm = sqrt(2/3 e_ij e_ij) = 2/3 * e for uniaxial.
    assert _val("von_mises_strain", c) == pytest.approx(2.0 / 3.0 * e)


# ---------------------------------------------------------------------
# Principal frame — values + eigenvectors (Phase 5, glyph support)
# ---------------------------------------------------------------------

def test_principal_frame_uniaxial():
    w, v = _derived.principal_frame(_stress(xx=7.0))
    np.testing.assert_allclose(w[0, 0], [7.0, 0.0, 0.0], atol=1e-9)
    # eigenvector of the largest principal is the x axis.
    assert abs(abs(v[0, 0, 0, 0]) - 1.0) < 1e-9


def test_principal_frame_pure_shear_directions():
    w, v = _derived.principal_frame(_stress(xy=5.0))
    # principals: +τ, 0, -τ
    np.testing.assert_allclose(w[0, 0], [5.0, 0.0, -5.0], atol=1e-9)
    # σ1 direction is (1,1,0)/√2; σ3 direction is (1,-1,0)/√2 (up to sign)
    d1 = v[0, 0, :, 0]
    d3 = v[0, 0, :, 2]
    assert abs(abs(d1[0]) - abs(d1[1])) < 1e-9 and abs(d1[2]) < 1e-9
    assert abs(abs(d3[0]) - abs(d3[1])) < 1e-9 and abs(d3[2]) < 1e-9
    # eigenvectors are orthonormal
    assert abs(np.dot(d1, d3)) < 1e-9


# ---------------------------------------------------------------------
# Shell-resultant von Mises (Phase 5) — surface stress recovery
# ---------------------------------------------------------------------

def _shell_cols(*, N=(0.0, 0.0, 0.0), M=(0.0, 0.0, 0.0)) -> dict:
    keys = ("xx", "yy", "xy")
    cols = {}
    for k, v in zip(keys, N):
        cols[f"membrane_force_{k}"] = np.array([[v]], dtype=float)
    for k, v in zip(keys, M):
        cols[f"bending_moment_{k}"] = np.array([[v]], dtype=float)
    return cols


def test_shell_pure_membrane():
    # Uniaxial membrane N_xx=N over thickness t → σ=N/t, von Mises = N/t.
    N, t = 100.0, 0.25
    out = _derived.compute_shell("von_mises_shell", _shell_cols(N=(N, 0, 0)), thickness=t)
    assert float(out[0, 0]) == pytest.approx(N / t)


def test_shell_pure_bending():
    # Bending M_xx=M → extreme-fibre σ=6M/t², symmetric top/bottom.
    M, t = 20.0, 0.2
    out = _derived.compute_shell("von_mises_shell", _shell_cols(M=(M, 0, 0)), thickness=t)
    assert float(out[0, 0]) == pytest.approx(6.0 * M / t**2)


def test_shell_envelope_takes_worst_surface():
    # Combined N + M: envelope = max(|N/t + 6M/t²|, |N/t − 6M/t²|).
    N, M, t = 50.0, 10.0, 0.2
    out = _derived.compute_shell(
        "von_mises_shell", _shell_cols(N=(N, 0, 0), M=(M, 0, 0)), thickness=t,
    )
    top = abs(N / t + 6.0 * M / t**2)
    bot = abs(N / t - 6.0 * M / t**2)
    assert float(out[0, 0]) == pytest.approx(max(top, bot))


def test_shell_bad_thickness_raises():
    with pytest.raises(ValueError, match="thickness"):
        _derived.compute_shell("von_mises_shell", _shell_cols(N=(1, 0, 0)), thickness=0.0)


def test_shell_available_and_registry():
    assert _derived.is_shell_derived("von_mises_shell")
    assert not _derived.is_shell_derived("von_mises_stress")
    # advertised only when the six in-plane resultants are stored
    resultants = ["membrane_force_xx", "membrane_force_yy", "membrane_force_xy",
                  "bending_moment_xx", "bending_moment_yy", "bending_moment_xy"]
    assert "von_mises_shell" in _derived.available_derived(resultants)
    assert "von_mises_shell" not in _derived.available_derived(["membrane_force_xx"])


# ---------------------------------------------------------------------
# 2-D plane defaults (out-of-plane component = 0)
# ---------------------------------------------------------------------

def test_2d_plane_stress_uniaxial():
    S = 4.0
    c = _stress(xx=S, yy=0.0, xy=0.0)  # STRESS_2D order
    assert _val("von_mises_stress", c, ndm=2) == pytest.approx(S)
    assert _val("principal_stress_1", c, ndm=2) == pytest.approx(S)
    assert _val("principal_stress_3", c, ndm=2) == pytest.approx(0.0, abs=1e-9)
    assert _val("tresca_stress", c, ndm=2) == pytest.approx(S)


# ---------------------------------------------------------------------
# ν-aware plane-strain / plane-stress out-of-plane recovery (Phase 5)
# ---------------------------------------------------------------------

def _val_plane(name, columns, *, plane, nu, ndm=2):
    out = _derived.compute(name, columns, ndm=ndm, plane=plane, nu=nu)
    return float(out[0, 0])


def test_plane_strain_recovers_sigma_zz():
    # 2-D data, plane strain: σ_zz = ν(σ_xx+σ_yy) fills the third principal.
    S, nu = 10.0, 0.3
    c = _stress(xx=S, yy=0.0, xy=0.0)
    # principals of diag(S, 0, νS) sorted desc → S, νS, 0
    assert _val_plane("principal_stress_1", c, plane="strain", nu=nu) == pytest.approx(S)
    assert _val_plane("principal_stress_2", c, plane="strain", nu=nu) == pytest.approx(nu * S)
    assert _val_plane("principal_stress_3", c, plane="strain", nu=nu) == pytest.approx(0.0, abs=1e-9)
    # mean = (S + 0 + νS)/3
    assert _val_plane("mean_stress", c, plane="strain", nu=nu) == pytest.approx((S + nu * S) / 3.0)


def test_plane_default_is_plane_stress_for_stress():
    # Without plane=, the 2-D stress tensor keeps σ_zz = 0 (plane stress).
    S = 10.0
    c = _stress(xx=S, yy=0.0, xy=0.0)
    assert _val_plane("principal_stress_2", c, plane=None, nu=None) == pytest.approx(0.0, abs=1e-9)
    assert _val_plane("von_mises_stress", c, plane=None, nu=None) == pytest.approx(S)


def test_plane_strain_requires_nu():
    c = _stress(xx=5.0, yy=1.0, xy=0.0)
    with pytest.raises(ValueError, match="nu="):
        _derived.compute("von_mises_stress", c, ndm=2, plane="strain", nu=None)


def test_plane_stress_recovers_epsilon_zz_for_strain():
    # 2-D strain, plane stress: ε_zz = -ν/(1-ν)(ε_xx+ε_yy).
    e, nu = 0.01, 0.25
    c = _strain(xx=e, yy=0.0, xy=0.0)
    ezz = -nu / (1.0 - nu) * e
    # volumetric = εxx+εyy+εzz
    assert _val_plane("volumetric_strain", c, plane="stress", nu=nu) == pytest.approx(e + ezz)


def test_plane_ignored_when_zz_present():
    # Full 3-D tensor: plane/nu are moot (σ_zz already stored), no raise.
    c = _stress(xx=2.0, yy=1.0, zz=3.0, xy=0.5, yz=0.0, xz=0.0)
    got = _derived.compute("von_mises_stress", c, ndm=3, plane="strain", nu=None)
    base = _derived.compute("von_mises_stress", c, ndm=3)
    np.testing.assert_allclose(got, base)


# ---------------------------------------------------------------------
# Vectorization + cross-check against the closed-form von Mises
# ---------------------------------------------------------------------

def test_vectorized_shape_and_formula():
    rng = np.random.default_rng(0)
    T, N = 3, 5
    comps = {f"stress_{k}": rng.standard_normal((T, N))
             for k in ("xx", "yy", "zz", "xy", "yz", "xz")}
    vm = _derived.compute("von_mises_stress", comps, ndm=3)
    assert vm.shape == (T, N)

    sxx, syy, szz = comps["stress_xx"], comps["stress_yy"], comps["stress_zz"]
    sxy, syz, sxz = comps["stress_xy"], comps["stress_yz"], comps["stress_xz"]
    expect = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )
    np.testing.assert_allclose(vm, expect, rtol=1e-12, atol=1e-12)


def test_principal_ordering_descending():
    rng = np.random.default_rng(1)
    comps = {f"stress_{k}": rng.standard_normal((4, 6))
             for k in ("xx", "yy", "zz", "xy", "yz", "xz")}
    p1 = _derived.compute("principal_stress_1", comps, ndm=3)
    p2 = _derived.compute("principal_stress_2", comps, ndm=3)
    p3 = _derived.compute("principal_stress_3", comps, ndm=3)
    assert np.all(p1 >= p2 - 1e-12)
    assert np.all(p2 >= p3 - 1e-12)
