"""Tests — ADR 0078 S4: stress recovery.

Pointwise analytic oracles (uniform N/A, flexural My/I at extreme
fibres, torsional Mzz·r/J on the circle, parabolic 1.5·V/A shear at the
NA), linearity/superposition of the unit-field blend, per-region access
on a composite, the disconnected deferral, and headless matplotlib
smoke tests (Agg).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — never open a window in tests

import numpy as np
import pytest

from apeGmsh.sections import (
    SectionAnalysisError,
    SectionMaterial,
    SectionProperties,
)


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


def _rect(g, b, h, *, x0=0.0, y0=0.0, pg=None):
    tag = g.model.geometry.add_rectangle(x0, y0, 0.0, b, h)
    if pg:
        g.physical.add(2, [tag], name=pg)
    return tag


def _rect_section(g, b=1.0, h=2.0, *, lc=0.08):
    _rect(g, b, h)
    fem = _mesh(g, lc=lc)
    return SectionProperties(fem, name="rect"), b, h


# ─────────────────────────────────────────────────────────────────────
# pointwise analytic oracles (geometric-only mode: E = 1)
# ─────────────────────────────────────────────────────────────────────

def test_pure_axial_uniform(g):
    sec, b, h = _rect_section(g)
    N = 100.0
    st = sec.stress(N=N)
    sig = st.sigma_zz
    assert np.nanmax(np.abs(sig - N / (b * h))) < 1e-9 * N
    # no shear from axial load
    assert np.nanmax(np.abs(st.tau)) < 1e-9 * N


def test_pure_bending_extreme_fibre(g):
    sec, b, h = _rect_section(g)
    M = 50.0
    st = sec.stress(Mxx=M)
    coords = sec._snapshot.coords
    Ixx = b * h**3 / 12
    cy = h / 2
    # σ = M·ȳ/I, tension at +y (top edge)
    top = np.isclose(coords[:, 1], h)
    bot = np.isclose(coords[:, 1], 0.0)
    assert st.sigma_zz[top] == pytest.approx(M * (h - cy) / Ixx, rel=1e-6)
    assert st.sigma_zz[bot] == pytest.approx(-M * cy / Ixx, rel=1e-6)
    # neutral axis: ~zero at mid-height.  Not exactly zero — the
    # unstructured mesh carries a tiny Ixy (~1e-5 relative) and the
    # general bending formula correctly responds to it, so compare
    # against the extreme-fibre stress, not machine epsilon.
    mid = np.isclose(coords[:, 1], cy)
    peak = M * cy / Ixx
    assert np.nanmax(np.abs(st.sigma_zz[mid])) < 1e-4 * peak
    # per-action component bookkeeping
    assert np.allclose(
        st.get("sigma_zz_mxx"), st.sigma_zz, equal_nan=True
    )
    assert np.nanmax(np.abs(st.get("sigma_zz_n"))) == 0.0


def test_myy_sign_convention(g):
    """Positive Myy → tension at +x (documented convention)."""
    sec, b, h = _rect_section(g)
    st = sec.stress(Myy=10.0)
    coords = sec._snapshot.coords
    Iyy = h * b**3 / 12
    right = np.isclose(coords[:, 0], b)
    assert st.sigma_zz[right] == pytest.approx(
        10.0 * (b / 2) / Iyy, rel=1e-6
    )


def test_torsion_circle_boundary(g):
    r = 1.0
    c = g.model.geometry.add_circle(0.0, 0.0, 0.0, r)
    loop = g.model.geometry.add_curve_loop([c])
    g.model.geometry.add_plane_surface([loop])
    fem = _mesh(g, lc=0.1)
    sec = SectionProperties(fem, name="disk")
    Mzz = 20.0
    st = sec.stress(Mzz=Mzz)
    coords = sec._snapshot.coords
    rr = np.hypot(coords[:, 0], coords[:, 1])
    boundary = rr > 0.995 * r
    J = np.pi * r**4 / 2
    # τ = Mzz·r/J on the boundary, tangential
    tau_b = st.tau[boundary]
    assert np.nanmedian(tau_b) == pytest.approx(Mzz * r / J, rel=5e-3)
    # centre: τ → 0
    centre = rr < 0.15 * r
    assert np.nanmax(st.tau[centre]) < 0.2 * Mzz * r / J


def test_shear_parabolic_at_na(g):
    """ν = 0 rectangle: τ_zy from Vy is parabolic with max 1.5·V/A at
    the neutral axis and ~0 at the free edges."""
    sec, b, h = _rect_section(g, lc=0.06)
    V = 30.0
    st = sec.stress(Vy=V)
    coords = sec._snapshot.coords
    mid = np.isclose(coords[:, 1], h / 2)
    edge = np.isclose(coords[:, 1], h) | np.isclose(coords[:, 1], 0.0)
    tau_max = 1.5 * V / (b * h)
    assert np.nanmedian(st.tau_zy[mid]) == pytest.approx(tau_max, rel=5e-3)
    assert np.nanmax(np.abs(st.tau_zy[edge])) < 0.02 * tau_max
    # von Mises composition at the NA: σ = 0 → vm = √3·τ
    vm_mid = st.von_mises[mid]
    assert np.nanmedian(vm_mid) == pytest.approx(
        np.sqrt(3) * tau_max, rel=5e-3
    )


# ─────────────────────────────────────────────────────────────────────
# blend identities
# ─────────────────────────────────────────────────────────────────────

def test_linearity_and_superposition(g):
    sec, _, _ = _rect_section(g, lc=0.12)
    a = sec.stress(N=10.0, Mxx=5.0, Vy=2.0, Mzz=1.0)
    b2 = sec.stress(N=20.0, Mxx=10.0, Vy=4.0, Mzz=2.0)
    assert np.allclose(2 * a.sigma_zz, b2.sigma_zz, equal_nan=True)
    assert np.allclose(2 * a.tau_zx, b2.tau_zx, equal_nan=True)
    # superposition: combined equals the sum of singles
    n_only = sec.stress(N=10.0)
    m_only = sec.stress(Mxx=5.0)
    both = sec.stress(N=10.0, Mxx=5.0)
    assert np.allclose(
        n_only.sigma_zz + m_only.sigma_zz, both.sigma_zz, equal_nan=True
    )
    # unit fields computed once (memoized on the analyzer)
    assert sec._unit_fields is not None


def test_principal_frame_matches_xy_for_upright(g):
    """Upright rectangle: M11 ≡ Mxx (φ = 0)."""
    sec, _, _ = _rect_section(g, lc=0.12)
    s_xy = sec.stress(Mxx=7.0)
    s_11 = sec.stress(M11=7.0)
    assert np.allclose(s_xy.sigma_zz, s_11.sigma_zz, equal_nan=True,
                       atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# composite + per-region access
# ─────────────────────────────────────────────────────────────────────

def test_composite_region_access_and_jump(g):
    """Two stacked strips E=200/10: σ from N jumps at the interface by
    the modular ratio; get(pg=) returns exact per-region values."""
    E1, E2 = 200.0, 10.0
    _rect(g, 1.0, 1.0, pg="stiff")
    _rect(g, 1.0, 1.0, y0=1.0, pg="soft")
    g.model.boolean.fragment([(2, 1)], [(2, 2)], dim=2)
    fem = _mesh(g, lc=0.1)
    sec = SectionProperties(
        fem,
        materials={
            "stiff": SectionMaterial(E=E1, nu=0.3),
            "soft": SectionMaterial(E=E2, nu=0.3),
        },
        name="bimat",
    )
    N = 21.0
    st = sec.stress(N=N)
    EA = E1 * 1.0 + E2 * 1.0
    sig_stiff = st.get("sigma_zz", pg="stiff")
    sig_soft = st.get("sigma_zz", pg="soft")
    assert np.nanmax(np.abs(sig_stiff - E1 * N / EA)) < 1e-9 * N
    assert np.nanmax(np.abs(sig_soft - E2 * N / EA)) < 1e-9 * N
    # NaN outside each region (interior of the other strip)
    coords = sec._snapshot.coords
    deep_soft = coords[:, 1] > 1.5
    assert np.all(np.isnan(sig_stiff[deep_soft]))
    # flat view at the interface keeps the larger-|σ| side (stiff)
    interface = np.isclose(coords[:, 1], 1.0)
    assert np.allclose(
        st.sigma_zz[interface], E1 * N / EA, atol=1e-9 * N
    )
    # unknown names fail loud
    with pytest.raises(KeyError, match="unknown material region"):
        st.get("sigma_zz", pg="ghost")
    with pytest.raises(KeyError, match="unknown stress component"):
        st.get("sigma_wat")


# ─────────────────────────────────────────────────────────────────────
# policy + plotting
# ─────────────────────────────────────────────────────────────────────

def test_disconnected_stress_deferred(g):
    _rect(g, 1.0, 1.0)
    _rect(g, 1.0, 1.0, x0=3.0)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem, disconnected="sum", name="twin")
    with pytest.raises(SectionAnalysisError, match="not yet implemented"):
        sec.stress(Vy=1.0)


def test_plots_headless(g):
    sec, _, _ = _rect_section(g, lc=0.15)
    ax = sec.stress(N=1.0, Vy=1.0).plot("von_mises")
    assert ax.get_title() == "von_mises"
    ax2 = sec.plot_mesh()
    assert ax2 is not None
    ax3 = sec.plot_section()          # centroid + SC + principal axes
    labels = {t.get_text() for t in ax3.get_legend().get_texts()}
    assert "centroid" in labels and "shear centre" in labels
    import matplotlib.pyplot as plt

    plt.close("all")
