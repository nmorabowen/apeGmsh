"""Tests — section-analyzer plot family (ADR 0078 addendum).

``plot_warping`` (ω contour + unit-torsion shear-flow overlay, with the
circle no-warp and disconnected-policy oracles), ``SectionStress.
plot_vector`` / ``plot_mohrs_circle`` (pure-axial Mohr oracle),
``sec.plot()`` overview figure, and the pre-mesh builder preview
``g.sections.plot_faces``.  All headless (Agg).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — never open a window in tests

import numpy as np
import pytest

from apeGmsh.sections import SectionProperties


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


def _rect_sec(g, b=2.0, h=4.0, *, lc=0.3) -> SectionProperties:
    g.model.geometry.add_rectangle(-b / 2, -h / 2, 0.0, b, h)
    return SectionProperties(_mesh(g, lc=lc), name="rect")


def _circle_sec(g, r=1.0, *, lc=0.12) -> SectionProperties:
    geo = g.model.geometry
    c = geo.add_circle(0.0, 0.0, 0.0, r)
    loop = geo.add_curve_loop([c])
    geo.add_plane_surface([loop])
    return SectionProperties(_mesh(g, lc=lc), name="disk")


def _close_all():
    import matplotlib.pyplot as plt

    plt.close("all")


# ─────────────────────────────────────────────────────────────────────
# plot_warping
# ─────────────────────────────────────────────────────────────────────

def test_plot_warping_circle_no_warp_oracle(g):
    """A circular section does not warp: max|ω| ≈ 0 (the exact
    Saint-Venant oracle), and the contour + shear-flow overlay render."""
    sec = _circle_sec(g)
    ax = sec.plot_warping(shear_flow=True)
    assert ax is not None
    omega = np.full(len(sec._snapshot.coords), np.nan)
    for sol in sec._warp_solutions:
        omega[sol.node_rows] = sol.omega
    # scale reference: r² (ω has units of length²·angle)
    assert np.nanmax(np.abs(omega)) < 5e-3
    _close_all()


def test_plot_warping_rectangle_warps(g):
    sec = _rect_sec(g)
    ax = sec.plot_warping()
    assert ax is not None
    omega = np.full(len(sec._snapshot.coords), np.nan)
    for sol in sec._warp_solutions:
        omega[sol.node_rows] = sol.omega
    assert np.nanmax(np.abs(omega)) > 0.1     # a rectangle warps
    _close_all()


def test_plot_warping_disconnected_sum(g):
    """ω plots per part under disconnected='sum'; the shear-flow
    overlay rides the per-part unit stress fields (each part shows its
    GJᵢ/ΣGJ share of the torque)."""
    geo = g.model.geometry
    geo.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    geo.add_rectangle(3.0, 0.0, 0.0, 1.0, 1.0)
    sec = SectionProperties(_mesh(g, lc=0.2), name="twin",
                            disconnected="sum")
    ax = sec.plot_warping()
    assert ax is not None and sec.n_parts == 2
    ax2 = sec.plot_warping(shear_flow=True)
    assert "shear flow" in ax2.get_title()
    _close_all()


# ─────────────────────────────────────────────────────────────────────
# SectionStress.plot_vector / plot_mohrs_circle
# ─────────────────────────────────────────────────────────────────────

def test_plot_vector_combined_and_per_action(g):
    sec = _rect_sec(g)
    st = sec.stress(Vy=350e3, Mzz=2.0e5)
    ax = st.plot_vector()
    assert len(ax.collections) >= 1          # the quiver is there
    ax2 = st.plot_vector("vy")
    assert ax2 is not None
    with pytest.raises(KeyError, match="shear action"):
        st.plot_vector("vz")
    _close_all()


def test_mohrs_circle_pure_axial_oracle(g):
    """Pure N: σ = N/A at every node, τ = 0 → centre σ/2, radius σ/2,
    principal stresses (σ, 0)."""
    sec = _rect_sec(g)                        # A = 8 exactly
    st = sec.stress(N=800e3)
    sigma, tau, _node = st._mohr_state(at=(0.0, 0.0))
    assert sigma == pytest.approx(1.0e5, rel=1e-9)
    assert tau == pytest.approx(0.0, abs=1e-12)
    ax = st.plot_mohrs_circle(at=(0.0, 0.0))
    assert "Mohr" in ax.get_title()
    with pytest.raises(KeyError, match="material region"):
        st.plot_mohrs_circle(at=(0.0, 0.0), pg="nope")
    _close_all()


# ─────────────────────────────────────────────────────────────────────
# overview figure + builder preview
# ─────────────────────────────────────────────────────────────────────

def test_overview_plot_figure(g):
    sec = _rect_sec(g)
    fig = sec.plot()
    assert len(fig.axes) == 2                 # glyph panel + text panel
    _close_all()


def test_builder_plot_faces_preview(g):
    """Pre-mesh geometry preview: outlines for every face + PG-name
    annotations from the builders' auto-PGs."""
    g.sections.rect_face(b=600.0, h=600.0, label="concrete")
    g.sections.W_face(bf=250.0, tf=17.0, h=250.0, tw=10.0,
                      label="steel", translate=(0.0, 100.0))
    ax = g.sections.plot_faces()
    assert len(ax.lines) >= 6                 # rect outline + W outline
    texts = {t.get_text() for t in ax.texts}
    assert {"concrete", "steel"} <= texts
    _close_all()
