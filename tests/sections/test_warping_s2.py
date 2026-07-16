"""Tests — ADR 0078 S2: warping / shear analysis.

Analytic oracles: circle J = πr⁴/2, rectangle J vs the exact series,
rectangle shear-area factor 5/6 at ν = 0, doubly-symmetric shear centre
= centroid, thin-wall channel shear centre.  Policy: disconnected
raise/sum, the unfragmented-touching-faces bug-catch, and the
G-override bound test (strip G→0 recovers the sum, full G recovers the
connected solve).  Package oracle (PyPI sectionproperties) runs when
installed, skips otherwise.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.sections import (
    CompositeSectionError,
    SectionAccuracyWarning,
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


def _rect_torsion_J(b: float, h: float, n_terms: int = 25) -> float:
    """Exact series for a b×h rectangle (b = short side)."""
    if b > h:
        b, h = h, b
    s = sum(
        math.tanh(n * math.pi * h / (2 * b)) / n**5
        for n in range(1, 2 * n_terms, 2)
    )
    return b**3 * h * (1.0 / 3.0 - 64.0 * b / (math.pi**5 * h) * s)


# ─────────────────────────────────────────────────────────────────────
# analytic oracles
# ─────────────────────────────────────────────────────────────────────

def test_circle_torsion_and_shear_centre(g):
    r = 1.0
    c = g.model.geometry.add_circle(0.0, 0.0, 0.0, r)
    loop = g.model.geometry.add_curve_loop([c])
    g.model.geometry.add_plane_surface([loop])
    fem = _mesh(g, lc=0.12)
    sec = SectionProperties(fem, name="disk")
    warp = sec.warping()

    assert warp.J == pytest.approx(math.pi * r**4 / 2, rel=2e-3)
    # doubly symmetric: shear centre = centroid (both definitions)
    assert warp.x_sc == pytest.approx(0.0, abs=1e-4)
    assert warp.y_sc == pytest.approx(0.0, abs=1e-4)
    assert warp.x_sc_t == pytest.approx(0.0, abs=1e-4)
    assert warp.y_sc_t == pytest.approx(0.0, abs=1e-4)
    # symmetry: the two shear areas coincide, cross term huge (≈ no
    # coupling → As_xy → ∞ in the Δ²/κ form; assert the alphas instead)
    assert warp.alpha_x == pytest.approx(warp.alpha_y, rel=1e-3)
    assert 0.5 < warp.alpha_y < 1.0
    # memoized
    assert sec.warping() is warp
    assert warp.parts == ()


def test_rectangle_torsion_series(g):
    b, h = 1.0, 2.0
    _rect(g, b, h)
    fem = _mesh(g, lc=0.08)
    warp = SectionProperties(fem, name="rect").warping()

    assert warp.J == pytest.approx(_rect_torsion_J(b, h), rel=1e-3)
    # doubly symmetric: shear centre at the section centre
    assert warp.x_sc == pytest.approx(b / 2, abs=1e-4)
    assert warp.y_sc == pytest.approx(h / 2, abs=1e-4)
    # monosymmetry constants vanish
    assert warp.beta_x_plus == pytest.approx(0.0, abs=1e-4)
    assert warp.beta_11_plus == pytest.approx(0.0, abs=1e-4)


def test_rectangle_shear_area_nu_zero(g):
    """At ν = 0 the flexural shear distribution over a rectangle is
    exactly parabolic → As/A = 5/6 in both directions."""
    _rect(g, 1.0, 2.0)
    fem = _mesh(g, lc=0.08)
    warp = SectionProperties(fem).warping()

    assert warp.alpha_y == pytest.approx(5.0 / 6.0, rel=1e-3)
    assert warp.alpha_x == pytest.approx(5.0 / 6.0, rel=1e-3)
    assert warp.nu_eff == pytest.approx(0.0, abs=1e-12)


def test_rectangle_geometric_only_J_accessor(g):
    """Geometric-only mode: the unprefixed J is the classic torsion
    constant (GJ carries G = 1/2; the accessor divides it back out)."""
    a = 1.0
    _rect(g, a, a)
    fem = _mesh(g, lc=0.06)
    warp = SectionProperties(fem).warping()
    assert warp.J == pytest.approx(0.1406 * a**4, rel=2e-3)


def test_channel_shear_centre_thin_wall(g):
    """Uniform-thickness thin-wall channel: SC sits e = 3b²/(h + 6b)
    from the web line (outside, away from the flanges)."""
    t = 0.05
    b, h = 1.0, 2.0          # flange width, web height (centrelines)
    # web: x ∈ [0, t], flanges at bottom/top extending +x
    _rect(g, t, h + t)                        # web incl. corners
    _rect(g, b, t, x0=t, y0=0.0)              # bottom flange
    _rect(g, b, t, x0=t, y0=h)                # top flange
    # conformal: fragment the three faces so the mesh connects
    g.model.boolean.fragment([(2, 1)], [(2, 2), (2, 3)], dim=2)
    fem = _mesh(g, lc=0.025)
    warp = SectionProperties(fem, name="channel").warping()

    e = 3 * b**2 / (h + 6 * b)                # thin-wall closed form
    # web centreline at x = t/2; SC at x = t/2 − e
    assert warp.x_sc == pytest.approx(t / 2 - e, abs=0.08 * e)
    # symmetric about mid-height
    assert warp.y_sc == pytest.approx((h + t) / 2, abs=0.01)


# ─────────────────────────────────────────────────────────────────────
# disconnected policy
# ─────────────────────────────────────────────────────────────────────

def test_disconnected_default_raises(g):
    _rect(g, 1.0, 1.0)
    _rect(g, 1.0, 1.0, x0=3.0)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem, name="twin")
    with pytest.raises(SectionAnalysisError, match="2 disconnected"):
        sec.warping()


def test_unfragmented_touching_faces_caught(g):
    """THE authoring bug: two faces that touch but were never
    fragmented → duplicated interface nodes → disconnected mesh.
    A permissive solver would return a silently-garbage J; we raise."""
    _rect(g, 1.0, 1.0)
    _rect(g, 1.0, 1.0, x0=1.0)     # shares the x=1 edge geometrically
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem, name="touching")
    assert sec.n_parts == 2        # not conformal → two components
    with pytest.raises(SectionAnalysisError, match="fragment"):
        sec.warping()


def test_disconnected_sum(g):
    a = 1.0
    _rect(g, a, a)
    _rect(g, a, a, x0=3.0)
    fem = _mesh(g, lc=0.06)
    sec = SectionProperties(fem, name="twin", disconnected="sum")
    warp = sec.warping()

    # GJ = ΣGJᵢ — two unit squares (geometric-only: G = 1/2)
    assert warp.J == pytest.approx(2 * 0.1406 * a**4, rel=2e-3)
    assert len(warp.parts) == 2
    # per-part shear centres at the square centres (authoring axes)
    xs = sorted(p.x_sc for p in warp.parts)
    assert xs[0] == pytest.approx(0.5, abs=1e-3)
    assert xs[1] == pytest.approx(3.5, abs=1e-3)
    # combined = GJ-weighted mean → midpoint by symmetry
    assert warp.x_sc == pytest.approx(2.0, abs=1e-3)
    # shear areas sum: two squares at ν = 0 → 5/6 each
    assert warp.alpha_y == pytest.approx(5.0 / 6.0, rel=2e-3)
    # transformed() propagates into parts: GJ = Σ parts[i].GJ holds on
    # the transformed view too
    t = warp.transformed(e_ref=2.0, g_ref=2.0)
    assert t.GJ == pytest.approx(sum(p.GJ for p in t.parts), rel=1e-12)
    assert t.GJ == pytest.approx(warp.GJ / 2.0, rel=1e-12)


# ─────────────────────────────────────────────────────────────────────
# G-override bound test (ADR: partial shear transfer is authored)
# ─────────────────────────────────────────────────────────────────────

def _three_strip_section(g):
    """Three 1×1 squares in a row, conformal by SHARED points/lines
    (each internal vertical line exists once, used by both loops)."""
    geo = g.model.geometry
    pts = {}
    for i in range(4):
        for jy, y in enumerate((0.0, 1.0)):
            pts[(i, jy)] = geo.add_point(float(i), y, 0.0)
    verts = [geo.add_line(pts[(i, 0)], pts[(i, 1)]) for i in range(4)]
    bots = [geo.add_line(pts[(i, 0)], pts[(i + 1, 0)]) for i in range(3)]
    tops = [geo.add_line(pts[(i, 1)], pts[(i + 1, 1)]) for i in range(3)]
    surfs = []
    for i in range(3):
        loop = geo.add_curve_loop([bots[i], verts[i + 1], tops[i], verts[i]])
        surfs.append(geo.add_plane_surface([loop]))
    g.physical.add(2, [surfs[0]], name="left")
    g.physical.add(2, [surfs[1]], name="strip")
    g.physical.add(2, [surfs[2]], name="right")
    return surfs


@pytest.mark.parametrize("g_strip, target, rel", [
    # rigid strip (chord material): the connected 3×1 rectangle
    (0.5, None, 2e-3),
    # vanishing strip: the two-chord sum (lower bound)
    (1e-9, 2 * 0.1406 * 0.5, 5e-2),
])
def test_g_override_bounds(g, g_strip, target, rel):
    _three_strip_section(g)
    fem = _mesh(g, lc=0.08)
    chord = SectionMaterial(E=1.0, nu=0.0)            # G = 1/2
    strip = SectionMaterial(E=1e-12, nu=0.0, G=g_strip)
    sec = SectionProperties(
        fem,
        materials={"left": chord, "strip": strip, "right": chord},
        name="battened",
    )
    warp = sec.warping()
    if target is None:
        # all-G=1/2 case: J of the full 3×1 rectangle × G
        target = 0.5 * _rect_torsion_J(1.0, 3.0)
    assert warp.GJ == pytest.approx(target, rel=rel)


def test_g_override_monotone(g):
    """GJ grows monotonically with the strip shear modulus.  One mesh,
    three analyzers — materials are analysis-side, not mesh-side."""
    _three_strip_section(g)
    fem = _mesh(g, lc=0.12)
    values = []
    for g_strip in (1e-9, 0.05, 0.5):
        sec = SectionProperties(
            fem,
            materials={
                "left": SectionMaterial(E=1.0, nu=0.0),
                "strip": SectionMaterial(E=1e-12, nu=0.0, G=g_strip),
                "right": SectionMaterial(E=1.0, nu=0.0),
            },
        )
        values.append(sec.warping().GJ)
    assert values[0] < values[1] < values[2]


# ─────────────────────────────────────────────────────────────────────
# naming law + warnings
# ─────────────────────────────────────────────────────────────────────

def test_composite_accessor_law_warping(g):
    _rect(g, 1.0, 1.0, pg="a")
    _rect(g, 1.0, 1.0, y0=1.0, pg="b")
    g.model.boolean.fragment([(2, 1)], [(2, 2)], dim=2)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(
        fem,
        materials={
            "a": SectionMaterial(E=200.0, nu=0.3),
            "b": SectionMaterial(E=10.0, nu=0.3),
        },
    )
    warp = sec.warping()
    with pytest.raises(CompositeSectionError, match="transformed"):
        _ = warp.J
    with pytest.raises(CompositeSectionError):
        _ = warp.Gamma
    # rigidity form + ratios valid
    assert warp.GJ > 0
    assert 0.0 < warp.alpha_y < 1.0
    t = warp.transformed(e_ref=200.0, g_ref=200.0 / 2.6)
    assert t.J == pytest.approx(warp.GJ / (200.0 / 2.6), rel=1e-12)


def test_linear_elements_accuracy_warning(g):
    _rect(g, 1.0, 1.0)
    fem = _mesh(g, lc=0.1, order=1)
    sec = SectionProperties(fem)
    with pytest.warns(SectionAccuracyWarning, match="set_order"):
        warp = sec.warping()
    # linear elements still converge, just slowly — sanity only
    assert 0.10 < warp.J < 0.15


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
    ref.calculate_warping_properties()

    _rect(g, b, h)
    fem = _mesh(g, lc=0.06)
    warp = SectionProperties(fem).warping()

    assert warp.J == pytest.approx(ref.get_j(), rel=2e-3)
    x_sc_ref, y_sc_ref = ref.get_sc()
    # package origin: section corner at (0, 0) as authored here
    assert warp.x_sc == pytest.approx(x_sc_ref, abs=1e-3)
    assert warp.y_sc == pytest.approx(y_sc_ref, abs=1e-3)
    asx, asy = ref.get_as()
    area = b * h
    assert warp.alpha_x == pytest.approx(asx / area, rel=5e-3)
    assert warp.alpha_y == pytest.approx(asy / area, rel=5e-3)
