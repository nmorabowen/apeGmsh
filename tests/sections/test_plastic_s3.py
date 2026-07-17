"""Tests — ADR 0078 S3: plastic analysis.

Analytic oracles: rectangle S = bh²/4 with shape factor 1.5, circle
S = 4r³/3, asymmetric T-section (equal-area axis + unequal ± shape
factors), a two-material strip hand calc (fy-weighted NA + Mp), the
rotated-rectangle principal-frame path, the fy gates, and the
mixed-fy accessor law.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.sections import (
    CompositeSectionError,
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


def _rect(g, b, h, *, x0=0.0, y0=0.0, pg=None, angles_deg=None):
    tag = g.model.geometry.add_rectangle(
        x0, y0, 0.0, b, h,
        **({"angles_deg": angles_deg, "pivot": (x0, y0, 0.0)}
           if angles_deg else {}),
    )
    if pg:
        g.physical.add(2, [tag], name=pg)
    return tag


STEEL = SectionMaterial(E=200e3, nu=0.3, fy=345.0)


# ─────────────────────────────────────────────────────────────────────
# analytic oracles
# ─────────────────────────────────────────────────────────────────────

def test_rectangle_plastic(g):
    b, h = 2.0, 3.0
    _rect(g, b, h, pg="steel")
    fem = _mesh(g, lc=0.06)
    sec = SectionProperties(fem, materials={"steel": STEEL}, name="rect")
    plas = sec.plastic()

    # plastic NA = equal-area axis = mid-height
    assert plas.y_pc == pytest.approx(h / 2, abs=0.03)
    assert plas.x_pc == pytest.approx(b / 2, abs=0.03)
    # S = b h² / 4 (homogeneous single fy → unprefixed valid)
    assert plas.Sxx == pytest.approx(b * h**2 / 4, rel=1e-3)
    assert plas.Syy == pytest.approx(h * b**2 / 4, rel=1e-3)
    assert plas.Mp_xx == pytest.approx(STEEL.fy * b * h**2 / 4, rel=1e-3)
    # principal frame coincides with x/y for the upright rectangle
    assert plas.S11 == pytest.approx(plas.Sxx, rel=1e-6)
    assert plas.S22 == pytest.approx(plas.Syy, rel=1e-6)
    # shape factor 1.5, symmetric ± sides
    assert plas.sf_xx_plus == pytest.approx(1.5, rel=2e-3)
    assert plas.sf_xx_minus == pytest.approx(1.5, rel=2e-3)
    assert plas.sf_yy_plus == pytest.approx(1.5, rel=2e-3)
    # memoized
    assert sec.plastic() is plas


def test_circle_plastic(g):
    r = 1.0
    c = g.model.geometry.add_circle(0.0, 0.0, 0.0, r)
    loop = g.model.geometry.add_curve_loop([c])
    surf = g.model.geometry.add_plane_surface([loop])
    g.physical.add(2, [surf], name="disk")
    fem = _mesh(g, lc=0.08)
    plas = SectionProperties(
        fem, materials={"disk": SectionMaterial(E=1.0, nu=0.3, fy=1.0)},
        name="disk",
    ).plastic()
    # S = 4 r³ / 3; sf = S / Z = (4/3) / (π/4) = 16 / (3π)
    assert plas.Sxx == pytest.approx(4 * r**3 / 3, rel=2e-3)
    assert plas.sf_xx_plus == pytest.approx(16 / (3 * math.pi), rel=3e-3)
    assert plas.y_pc == pytest.approx(0.0, abs=0.03)


def test_tee_section_plastic(g):
    """T-section: equal-area NA in the flange; ± shape factors differ."""
    bf, tf = 2.0, 0.5          # flange 2 × 0.5, top at y = 2
    tw, hw = 0.5, 1.5          # web 0.5 × 1.5 below it
    fy = 1.0
    _rect(g, tw, hw, x0=-tw / 2, pg="web")            # y ∈ [0, 1.5]
    _rect(g, bf, tf, x0=-bf / 2, y0=hw, pg="flange")  # y ∈ [1.5, 2]
    g.model.boolean.fragment([(2, 1)], [(2, 2)], dim=2)
    fem = _mesh(g, lc=0.05)
    sec = SectionProperties(
        fem,
        materials={
            "web": SectionMaterial(E=1.0, nu=0.3, fy=fy),
            "flange": SectionMaterial(E=1.0, nu=0.3, fy=fy),
        },
        name="tee",
    )
    plas = sec.plastic()

    # equal-area axis: A = 1.75, half = 0.875; flange area 1.0 → NA in
    # the flange, 0.875 of area below it: y_na = 2 − (1 − 0.875) · ...
    # flange spans [1.5, 2] with width 2: area above y = 2 − t is 2t
    # → 2 (2 − y_na) = 0.875 → y_na = 2 − 0.4375 = 1.5625
    y_na = 2.0 - 0.875 / bf
    assert plas.y_pc == pytest.approx(y_na, abs=0.03)
    # Mp about the NA, exact hand integral
    mp = (
        fy * tw * ((y_na - 0.0) ** 2 - (y_na - hw) ** 2) / 2   # web
        + fy * bf * (y_na - hw) ** 2 / 2                        # flange below NA
        + fy * bf * (2.0 - y_na) ** 2 / 2                       # flange above NA
    )
    assert plas.Mp_xx == pytest.approx(mp, rel=2e-3)
    # monosymmetric: the two shape factors differ
    assert plas.sf_xx_plus != pytest.approx(plas.sf_xx_minus, rel=0.05)
    # symmetric about the y axis: x-direction stays symmetric
    assert plas.x_pc == pytest.approx(0.0, abs=0.03)


def test_rotated_rectangle_principal_plastic(g):
    """Rotated rectangle: the principal-frame plastic moduli recover the
    upright values."""
    b, h, ang = 1.0, 3.0, 30.0
    _rect(g, b, h, pg="steel", angles_deg=(0.0, 0.0, ang))
    fem = _mesh(g, lc=0.06)
    plas = SectionProperties(fem, materials={"steel": STEEL}).plastic()

    assert plas.S11 == pytest.approx(b * h**2 / 4, rel=2e-3)
    assert plas.S22 == pytest.approx(h * b**2 / 4, rel=2e-3)
    assert plas.sf_11_plus == pytest.approx(1.5, rel=3e-3)


def test_composite_strip_plastic(g):
    """Two stacked unit strips, fy = 2 below / 1 above: NA where the
    fy-weighted halves balance, Mp from the hand integral."""
    fy1, fy2 = 2.0, 1.0
    _rect(g, 1.0, 1.0, pg="strong")               # y ∈ [0, 1]
    _rect(g, 1.0, 1.0, y0=1.0, pg="weak")         # y ∈ [1, 2]
    g.model.boolean.fragment([(2, 1)], [(2, 2)], dim=2)
    fem = _mesh(g, lc=0.05)
    sec = SectionProperties(
        fem,
        materials={
            "strong": SectionMaterial(E=1.0, nu=0.3, fy=fy1),
            "weak": SectionMaterial(E=1.0, nu=0.3, fy=fy2),
        },
        name="bimetal",
    )
    plas = sec.plastic()

    # total fy·A = 3, half = 1.5 → NA inside the strong strip at
    # fy1 · y_na = 1.5 → y_na = 0.75
    assert plas.y_pc == pytest.approx(0.75, abs=0.03)
    # Mp = 2·∫₀^0.75 (0.75−y) dy + 2·∫_0.75^1 (y−0.75) dy + 1·∫₁² (y−0.75) dy
    mp = (
        fy1 * 0.75**2 / 2 + fy1 * 0.25**2 / 2
        + fy2 * ((2 - 0.75) ** 2 - (1 - 0.75) ** 2) / 2
    )
    assert plas.Mp_xx == pytest.approx(mp, rel=3e-3)
    # mixed fy → unprefixed moduli raise with guidance
    with pytest.raises(CompositeSectionError, match="Mp_xx"):
        _ = plas.Sxx
    assert plas.fy_ref is None
    # same-fy composite (different E) keeps the unprefixed moduli
    sec2 = SectionProperties(
        fem,
        materials={
            "strong": SectionMaterial(E=200.0, nu=0.3, fy=1.0),
            "weak": SectionMaterial(E=10.0, nu=0.3, fy=1.0),
        },
    )
    assert sec2.plastic().fy_ref == 1.0
    _ = sec2.plastic().Sxx  # no raise


# ─────────────────────────────────────────────────────────────────────
# gates
# ─────────────────────────────────────────────────────────────────────

def test_gate_geometric_only(g):
    _rect(g, 1.0, 1.0)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem)
    with pytest.raises(SectionAnalysisError, match="materials="):
        sec.plastic()


def test_gate_missing_fy_names_pg(g):
    _rect(g, 1.0, 1.0, pg="steel")
    _rect(g, 1.0, 1.0, y0=1.0, pg="concrete")
    g.model.boolean.fragment([(2, 1)], [(2, 2)], dim=2)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(
        fem,
        materials={
            "steel": SectionMaterial(E=200e3, nu=0.3, fy=345.0),
            "concrete": SectionMaterial(E=25e3, nu=0.2),   # no fy
        },
    )
    with pytest.raises(SectionAnalysisError, match="concrete"):
        sec.plastic()


def test_analyze_includes_plastic_when_available(g):
    _rect(g, 1.0, 1.0, pg="steel")
    fem = _mesh(g, lc=0.15)
    sec = SectionProperties(fem, materials={"steel": STEEL})
    sec.analyze()
    assert sec._plastic is not None   # populated by analyze()

    # geometric-only analyze() must NOT trip the fy gate
    g2_fem = fem
    sec2 = SectionProperties(g2_fem)
    sec2.analyze()                    # no raise
    assert sec2._plastic is None


# ─────────────────────────────────────────────────────────────────────
# package oracle (dev-only; skipped when not installed)
# ─────────────────────────────────────────────────────────────────────

def test_against_sectionproperties_package(g):
    pytest.importorskip("sectionproperties")
    from sectionproperties.analysis import Section
    from sectionproperties.pre.library import rectangular_section

    b, h = 1.0, 2.0
    ref_geom = rectangular_section(d=h, b=b)
    ref_geom.create_mesh(mesh_sizes=[0.005])
    ref = Section(ref_geom)
    ref.calculate_geometric_properties()
    ref.calculate_plastic_properties()

    _rect(g, b, h, pg="steel")
    fem = _mesh(g, lc=0.05)
    plas = SectionProperties(
        fem, materials={"steel": SectionMaterial(E=1.0, nu=0.3, fy=1.0)}
    ).plastic()

    sxx_ref, syy_ref = ref.get_s()
    assert plas.Sxx == pytest.approx(sxx_ref, rel=2e-3)
    assert plas.Syy == pytest.approx(syy_ref, rel=2e-3)
