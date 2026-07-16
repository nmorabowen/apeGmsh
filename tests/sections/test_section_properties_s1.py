"""Tests — ADR 0078 S1: SectionProperties geometric analysis.

Analytic oracles (rectangle / circle / offset / rotated / holed),
composite modulus weighting, geometric-only mode, the disconnected
Steiner behaviour, PG-coverage fail-loud gates, and the unprefixed
accessor law.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.sections import (
    CompositeSectionError,
    SectionMaterial,
    SectionMeshError,
    SectionProperties,
)

REL = 1e-9        # straight-sided meshes: quadrature-exact
REL_CURVED = 1e-3  # curved (circle) boundaries: mesh-convergence bound


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# analytic oracles
# ─────────────────────────────────────────────────────────────────────

def test_rectangle_geometric(g):
    b, h = 2.0, 3.0
    _rect(g, b, h)
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(fem, name="rect")
    geo = sec.geometric()

    assert geo.area == pytest.approx(b * h, rel=REL)
    assert geo.perimeter == pytest.approx(2 * (b + h), rel=REL)
    assert geo.cx == pytest.approx(b / 2, rel=REL)
    assert geo.cy == pytest.approx(h / 2, rel=REL)
    assert geo.Ixx_c == pytest.approx(b * h**3 / 12, rel=REL)
    assert geo.Iyy_c == pytest.approx(h * b**3 / 12, rel=REL)
    assert geo.Ixy_c == pytest.approx(0.0, abs=REL * b * h**3)
    # h > b → major axis is x → phi = 0
    assert geo.phi == pytest.approx(0.0, abs=1e-6)
    assert geo.I11_c == pytest.approx(b * h**3 / 12, rel=REL)
    assert geo.I22_c == pytest.approx(h * b**3 / 12, rel=REL)
    assert geo.Zxx_plus == pytest.approx(b * h**2 / 6, rel=REL)
    assert geo.Zxx_minus == pytest.approx(b * h**2 / 6, rel=REL)
    assert geo.Zyy_plus == pytest.approx(h * b**2 / 6, rel=REL)
    assert geo.rx == pytest.approx(h / math.sqrt(12), rel=REL)
    assert geo.ry == pytest.approx(b / math.sqrt(12), rel=REL)
    # global-axis moments (corner at origin): Ixx_g = b h^3 / 3
    assert geo.Ixx_g == pytest.approx(b * h**3 / 3, rel=REL)
    assert geo.Iyy_g == pytest.approx(h * b**3 / 3, rel=REL)
    assert sec.n_parts == 1


def test_rectangle_tri3_geometric_ok(g):
    """Linear elements are fine for geometric analysis (no warning)."""
    _rect(g, 2.0, 3.0)
    fem = _mesh(g, lc=0.4, order=1)
    geo = SectionProperties(fem).geometric()
    assert geo.area == pytest.approx(6.0, rel=REL)
    assert geo.Ixx_c == pytest.approx(2.0 * 27 / 12, rel=REL)


def test_circle_geometric(g):
    r = 1.0
    c = g.model.geometry.add_circle(0.0, 0.0, 0.0, r)
    loop = g.model.geometry.add_curve_loop([c])
    g.model.geometry.add_plane_surface([loop])
    fem = _mesh(g, lc=0.15)
    geo = SectionProperties(fem, name="disk").geometric()

    assert geo.area == pytest.approx(math.pi * r**2, rel=REL_CURVED)
    assert geo.perimeter == pytest.approx(2 * math.pi * r, rel=5e-3)
    assert geo.cx == pytest.approx(0.0, abs=1e-6)
    assert geo.cy == pytest.approx(0.0, abs=1e-6)
    assert geo.Ixx_c == pytest.approx(math.pi * r**4 / 4, rel=REL_CURVED)
    assert geo.Iyy_c == pytest.approx(math.pi * r**4 / 4, rel=REL_CURVED)


def test_offset_rectangle_steiner(g):
    """Global vs centroidal: Ixx_g = Ixx_c + A·cy²."""
    b, h, y0 = 1.0, 2.0, 5.0
    _rect(g, b, h, y0=y0)
    fem = _mesh(g, lc=0.5)
    geo = SectionProperties(fem).geometric()

    cy = y0 + h / 2
    assert geo.cy == pytest.approx(cy, rel=REL)
    assert geo.Ixx_c == pytest.approx(b * h**3 / 12, rel=REL)
    assert geo.Ixx_g == pytest.approx(b * h**3 / 12 + b * h * cy**2, rel=REL)


def test_rotated_rectangle_principal_axes(g):
    """A rectangle rotated by θ: phi recovers θ, principal I unchanged."""
    b, h, ang = 1.0, 3.0, 30.0
    _rect(g, b, h, angles_deg=(0.0, 0.0, ang))
    fem = _mesh(g, lc=0.4)
    geo = SectionProperties(fem).geometric()

    # major axis of an upright rectangle is x (Ixx > Iyy); rotated by
    # +30° the major axis sits at +30°.
    assert geo.phi == pytest.approx(ang, abs=1e-4)
    assert geo.I11_c == pytest.approx(b * h**3 / 12, rel=REL)
    assert geo.I22_c == pytest.approx(h * b**3 / 12, rel=REL)
    assert geo.Z11_plus == pytest.approx(b * h**2 / 6, rel=1e-9)


def test_holed_rectangle_area_and_perimeter(g):
    """First loop = outer boundary, second = hole; perimeter excludes
    the hole (ADR: exterior walk)."""
    b, h, r = 4.0, 3.0, 0.8
    geo_ns = g.model.geometry
    p1 = geo_ns.add_point(0, 0, 0)
    p2 = geo_ns.add_point(b, 0, 0)
    p3 = geo_ns.add_point(b, h, 0)
    p4 = geo_ns.add_point(0, h, 0)
    l1 = geo_ns.add_line(p1, p2)
    l2 = geo_ns.add_line(p2, p3)
    l3 = geo_ns.add_line(p3, p4)
    l4 = geo_ns.add_line(p4, p1)
    outer = geo_ns.add_curve_loop([l1, l2, l3, l4])
    hole_c = geo_ns.add_circle(b / 2, h / 2, 0.0, r)
    inner = geo_ns.add_curve_loop([hole_c])
    geo_ns.add_plane_surface([outer, inner])
    fem = _mesh(g, lc=0.2)
    geo = SectionProperties(fem, name="holed").geometric()

    assert geo.area == pytest.approx(b * h - math.pi * r**2, rel=REL_CURVED)
    assert geo.perimeter == pytest.approx(2 * (b + h), rel=1e-6)


# ─────────────────────────────────────────────────────────────────────
# materials: composite weighting + the accessor law
# ─────────────────────────────────────────────────────────────────────

def test_composite_two_material_strip(g):
    """Stacked strips: EA / centroid / EIxx_c match the hand calc."""
    E1, E2 = 200.0, 10.0
    s1 = _rect(g, 1.0, 1.0, pg="steel")               # y in [0, 1]
    s2 = _rect(g, 1.0, 1.0, y0=1.0, pg="wood")        # y in [1, 2]
    assert s1 != s2
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(
        fem,
        materials={
            "steel": SectionMaterial(E=E1, nu=0.3),
            "wood": SectionMaterial(E=E2, nu=0.3),
        },
        name="strip",
    )
    geo = sec.geometric()

    EA = E1 * 1.0 + E2 * 1.0
    cy = (E1 * 0.5 + E2 * 1.5) / EA
    EIxx_c = (
        E1 * (1.0 / 12 + (0.5 - cy) ** 2)
        + E2 * (1.0 / 12 + (1.5 - cy) ** 2)
    )
    assert geo.EA == pytest.approx(EA, rel=REL)
    assert geo.cy == pytest.approx(cy, rel=REL)
    assert geo.EIxx_c == pytest.approx(EIxx_c, rel=REL)
    assert geo.area == pytest.approx(2.0, rel=REL)
    assert geo.material_areas == pytest.approx((1.0, 1.0), rel=REL)

    # accessor law: composite → unprefixed raises with guidance
    with pytest.raises(CompositeSectionError, match="transformed"):
        _ = geo.Ixx_c
    with pytest.raises(CompositeSectionError):
        _ = geo.Zxx_plus
    # transformed view: explicit reference modulus
    t = geo.transformed(e_ref=E1)
    assert t.Ixx_c == pytest.approx(EIxx_c / E1, rel=REL)
    assert t.EA == pytest.approx(EA / E1, rel=REL)
    # radii are reference-free — valid straight on the composite
    assert geo.rx == pytest.approx(math.sqrt(EIxx_c / EA), rel=REL)


def test_homogeneous_materials_unprefixed_ok(g):
    """One material (real E): unprefixed accessors divide by it."""
    E = 200e3
    _rect(g, 2.0, 3.0, pg="steel")
    fem = _mesh(g, lc=0.5)
    geo = SectionProperties(
        fem, materials={"steel": SectionMaterial(E=E, nu=0.3)}
    ).geometric()
    assert geo.EIxx_c == pytest.approx(E * 2.0 * 27 / 12, rel=REL)
    assert geo.Ixx_c == pytest.approx(2.0 * 27 / 12, rel=REL)


def test_geometric_only_mode(g):
    """No materials= → unit moduli, classic numbers, accessors valid."""
    _rect(g, 2.0, 3.0)
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(fem)
    assert sec.geometric_only
    geo = sec.geometric()
    assert geo.EA == pytest.approx(geo.area, rel=REL)
    assert geo.Ixx_c == pytest.approx(geo.EIxx_c, rel=REL)


def test_mass_per_unit_length(g):
    rho = 7.85e-9
    _rect(g, 2.0, 3.0, pg="steel")
    fem = _mesh(g, lc=0.5)
    geo = SectionProperties(
        fem,
        materials={"steel": SectionMaterial(E=1.0, nu=0.3, density=rho)},
    ).geometric()
    assert geo.mass == pytest.approx(rho * 6.0, rel=REL)


# ─────────────────────────────────────────────────────────────────────
# disconnected parts (geometric is connectivity-blind)
# ─────────────────────────────────────────────────────────────────────

def test_disconnected_geometric_steiner(g):
    """Two separated unit squares: I about the COMMON centroid includes
    the Steiner terms — no flag needed for geometric()."""
    _rect(g, 1.0, 1.0)                    # x in [0, 1]
    _rect(g, 1.0, 1.0, x0=2.0)            # x in [2, 3]
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(fem, name="twin")
    assert sec.n_parts == 2
    geo = sec.geometric()

    assert geo.area == pytest.approx(2.0, rel=REL)
    assert geo.cx == pytest.approx(1.5, rel=REL)
    # each square: 1/12 + A·(dx=1)² about the common centroid
    assert geo.Iyy_c == pytest.approx(2 * (1.0 / 12 + 1.0), rel=REL)
    assert geo.Ixx_c == pytest.approx(2 * (1.0 / 12), rel=REL)
    # perimeter = both exteriors
    assert geo.perimeter == pytest.approx(8.0, rel=REL)


# ─────────────────────────────────────────────────────────────────────
# input gates
# ─────────────────────────────────────────────────────────────────────

def test_gate_uncovered_elements(g):
    _rect(g, 1.0, 1.0, pg="steel")
    _rect(g, 1.0, 1.0, x0=2.0)            # not in any PG
    fem = _mesh(g, lc=0.5)
    with pytest.raises(SectionMeshError, match="covered by no"):
        SectionProperties(
            fem, materials={"steel": SectionMaterial(E=1.0, nu=0.3)}
        )


def test_gate_double_covered_elements(g):
    tag = _rect(g, 1.0, 1.0)
    g.physical.add(2, [tag], name="a")
    g.physical.add(2, [tag], name="b")
    fem = _mesh(g, lc=0.5)
    with pytest.raises(SectionMeshError, match="claimed by both"):
        SectionProperties(
            fem,
            materials={
                "a": SectionMaterial(E=1.0, nu=0.3),
                "b": SectionMaterial(E=2.0, nu=0.3),
            },
        )


def test_gate_missing_pg(g):
    _rect(g, 1.0, 1.0, pg="steel")
    fem = _mesh(g, lc=0.5)
    with pytest.raises(SectionMeshError, match="ghost"):
        SectionProperties(
            fem, materials={"ghost": SectionMaterial(E=1.0, nu=0.3)}
        )


def test_gate_non_planar(g):
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 2.0, plane="xz")
    fem = _mesh(g, lc=0.5)
    with pytest.raises(SectionMeshError, match="XY plane"):
        SectionProperties(fem)


def test_gate_solid_mesh(g):
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)
    with pytest.raises(SectionMeshError, match="tri3/tri6"):
        SectionProperties(fem)


def test_gate_bad_disconnected_value(g):
    _rect(g, 1.0, 1.0)
    fem = _mesh(g, lc=0.5)
    with pytest.raises(ValueError, match="disconnected"):
        SectionProperties(fem, disconnected="maybe")  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────
# SectionMaterial + display
# ─────────────────────────────────────────────────────────────────────

def test_section_material_validation_and_shear_modulus():
    m = SectionMaterial(E=200.0, nu=0.3)
    assert m.shear_modulus == pytest.approx(200.0 / 2.6)
    m2 = SectionMaterial(E=1e-6, nu=0.0, G=42.0)   # equivalent shear strip
    assert m2.shear_modulus == 42.0
    with pytest.raises(ValueError, match="E must be"):
        SectionMaterial(E=0.0, nu=0.3)
    with pytest.raises(ValueError, match="nu"):
        SectionMaterial(E=1.0, nu=0.5)
    with pytest.raises(ValueError, match="fy"):
        SectionMaterial(E=1.0, nu=0.3, fy=-1.0)


def test_summary_and_repr(g):
    _rect(g, 2.0, 3.0, pg="steel")
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(
        fem, materials={"steel": SectionMaterial(E=200e3, nu=0.3)},
        name="demo",
    )
    text = sec.summary()
    assert "demo" in text and "steel" in text and "EIxx_c" in text
    assert "<pre" in sec._repr_html_()
    geo = sec.geometric()
    assert "GeometricProperties" in geo._repr_html_()
    # memoization: same frozen object back
    assert sec.geometric() is geo
    assert sec.analyze() is sec
