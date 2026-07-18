"""Tests — ADR 0078 S5: flat-face parametric builders (``g.sections.*_face``).

Every builder: exact-area (and where cheap, inertia) oracles through
the geometric-only analyzer, the auto-PG-by-label contract, in-plane
``translate``/``rotate`` placement, the AISC W-shape catalog round
trip (W14×90), and the SRC composite example end-to-end
(cut → fragment → analyzer → ``ComputedSection`` deck line).
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section import ComputedSection
from apeGmsh.sections import (
    CompositeSectionError,
    SectionMaterial,
    SectionProperties,
)


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


# ─────────────────────────────────────────────────────────────────────
# per-builder area / inertia oracles (straight sides → quadrature-exact)
# ─────────────────────────────────────────────────────────────────────

def test_rect_face(g):
    inst = g.sections.rect_face(b=2.0, h=4.0, label="bar")
    assert inst.label == "bar"
    sec = SectionProperties(_mesh(g, lc=0.4), name="bar")
    geo = sec.geometric()
    assert geo.area == pytest.approx(8.0, rel=1e-9)
    assert geo.Ixx_c == pytest.approx(2.0 * 4.0**3 / 12.0, rel=1e-9)
    assert geo.Iyy_c == pytest.approx(4.0 * 2.0**3 / 12.0, rel=1e-9)
    assert (geo.cx, geo.cy) == (
        pytest.approx(0.0, abs=1e-9), pytest.approx(0.0, abs=1e-9),
    )


def test_rect_hollow_face(g):
    g.sections.rect_hollow_face(b=4.0, h=6.0, t=0.5, label="hss")
    geo = SectionProperties(_mesh(g, lc=0.4), name="hss").geometric()
    assert geo.area == pytest.approx(4 * 6 - 3 * 5, rel=1e-9)
    assert geo.Ixx_c == pytest.approx(
        (4 * 6**3 - 3 * 5**3) / 12.0, rel=1e-9
    )
    assert geo.Iyy_c == pytest.approx(
        (6 * 4**3 - 5 * 3**3) / 12.0, rel=1e-9
    )


def test_pipe_face(g):
    g.sections.pipe_face(r=1.0, label="rod")
    geo = SectionProperties(_mesh(g, lc=0.1), name="rod").geometric()
    assert geo.area == pytest.approx(math.pi, rel=2e-3)
    assert geo.Ixx_c == pytest.approx(math.pi / 4.0, rel=5e-3)


def test_pipe_hollow_face(g):
    g.sections.pipe_hollow_face(r=1.0, t=0.3, label="pipe")
    geo = SectionProperties(_mesh(g, lc=0.08), name="pipe").geometric()
    assert geo.area == pytest.approx(math.pi * (1.0 - 0.7**2), rel=2e-3)
    assert geo.Ixx_c == pytest.approx(
        math.pi * (1.0 - 0.7**4) / 4.0, rel=5e-3
    )


def test_angle_face(g):
    g.sections.angle_face(b=3.0, h=4.0, t=0.5, label="L")
    geo = SectionProperties(_mesh(g, lc=0.25), name="L").geometric()
    assert geo.area == pytest.approx(3 * 0.5 + 4 * 0.5 - 0.25, rel=1e-9)


def test_channel_face(g):
    g.sections.channel_face(bf=2.0, tf=0.4, h=3.0, tw=0.3, label="C")
    geo = SectionProperties(_mesh(g, lc=0.15), name="C").geometric()
    assert geo.area == pytest.approx(
        2.0 * 3.8 - 1.7 * 3.0, rel=1e-9
    )


def test_tee_face(g):
    g.sections.tee_face(bf=3.0, tf=0.5, h=4.0, tw=0.4, label="T")
    geo = SectionProperties(_mesh(g, lc=0.2), name="T").geometric()
    assert geo.area == pytest.approx(3 * 0.5 + 4 * 0.4, rel=1e-9)


def test_w_face_area(g):
    bf, tf, h, tw = 14.52, 0.71, 12.6, 0.44
    g.sections.W_face(bf=bf, tf=tf, h=h, tw=tw, label="W")
    geo = SectionProperties(_mesh(g, lc=0.6), name="W").geometric()
    assert geo.area == pytest.approx(2 * bf * tf + h * tw, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# placement + auto-PG contract
# ─────────────────────────────────────────────────────────────────────

def test_translate_moves_centroid(g):
    g.sections.rect_face(b=2.0, h=4.0, label="bar", translate=(10.0, 5.0))
    geo = SectionProperties(_mesh(g, lc=0.4), name="bar").geometric()
    assert geo.cx == pytest.approx(10.0, abs=1e-9)
    assert geo.cy == pytest.approx(5.0, abs=1e-9)


def test_rotate_swaps_axes(g):
    g.sections.rect_face(b=2.0, h=4.0, label="bar", rotate=90.0)
    geo = SectionProperties(_mesh(g, lc=0.4), name="bar").geometric()
    # 90° in-plane: authoring Ixx of the rotated tall rect is the wide one
    assert geo.Ixx_c == pytest.approx(4.0 * 2.0**3 / 12.0, rel=1e-9)
    assert geo.Iyy_c == pytest.approx(2.0 * 4.0**3 / 12.0, rel=1e-9)


def test_auto_pg_feeds_materials(g):
    """The builder's auto-PG (named after ``label``) is exactly what
    ``SectionProperties(materials={label: ...})`` consumes."""
    g.sections.rect_face(b=2.0, h=4.0, label="bar")
    fem = _mesh(g, lc=0.4)
    sec = SectionProperties(
        fem, materials={"bar": SectionMaterial(E=200e3, nu=0.3)},
        name="bar",
    )
    assert sec.geometric().EA == pytest.approx(200e3 * 8.0, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────
# AISC catalog round trip — W14×90 (inches)
# ─────────────────────────────────────────────────────────────────────

def test_w_face_matches_aisc_w14x90(g):
    """W_face → analyzer vs AISC v16 W14×90 catalog values.

    The builder models the fillet-less plate assembly, so A / Ix / Iy
    sit ~1–2 % below catalog (fillets add area near the k-region) and
    J sits below catalog by more (J is fillet-sensitive) — tolerances
    chosen accordingly.  Catalog: A = 26.5 in², Ix = 999 in⁴,
    Iy = 362 in⁴, J = 4.06 in⁴; d = 14.0, bf = 14.5, tf = 0.710,
    tw = 0.440.
    """
    bf, tf, d, tw = 14.5, 0.710, 14.0, 0.440
    g.sections.W_face(bf=bf, tf=tf, h=d - 2 * tf, tw=tw, label="W14X90")
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(fem, name="W14X90")
    geo = sec.geometric()
    assert geo.area == pytest.approx(26.5, rel=0.03)
    assert geo.Ixx_c == pytest.approx(999.0, rel=0.03)
    assert geo.Iyy_c == pytest.approx(362.0, rel=0.03)
    assert sec.warping().J == pytest.approx(4.06, rel=0.15)

    # and straight through the deck lowering (strong axis → Iz)
    es = sec.to_elastic_section(E=29000.0, G=11200.0)
    assert es.Iz == pytest.approx(999.0, rel=0.03)
    assert es.Iy == pytest.approx(362.0, rel=0.03)


# ─────────────────────────────────────────────────────────────────────
# SRC composite example — end to end
# ─────────────────────────────────────────────────────────────────────

def test_src_composite_end_to_end(g):
    """Steel W encased in concrete: faces → cut (keep tool) →
    ``fragment_pair`` → conformal mesh → composite analyzer →
    ``ComputedSection`` deck line.  Exact-area EA cross-check."""
    conc = g.sections.rect_face(b=600.0, h=600.0, label="concrete")
    steel = g.sections.W_face(
        bf=250.0, tf=17.0, h=250.0, tw=10.0, label="steel",
    )
    # Carve the W out of the concrete so the two PGs partition the
    # section, then fragment for a conformal interface.
    g.model.boolean.cut(
        conc.entities[2], steel.entities[2], dim=2, remove_tool=False,
    )
    g.parts.fragment_pair("concrete", "steel", dim=2)
    fem = _mesh(g, lc=40.0)

    sec = SectionProperties(
        fem,
        materials={
            "concrete": SectionMaterial(E=25e3, nu=0.2),
            "steel": SectionMaterial(E=200e3, nu=0.3),
        },
        name="SRC600",
    )
    assert sec.n_parts == 1                      # conformal — one part
    a_w = 2 * 250.0 * 17.0 + 250.0 * 10.0
    ea = 25e3 * (600.0**2 - a_w) + 200e3 * a_w
    assert sec.geometric().EA == pytest.approx(ea, rel=1e-9)

    # composite → explicit reference moduli required, naming the handle
    cs_bad = ComputedSection(analysis=sec)
    with pytest.raises(CompositeSectionError, match="SRC600"):
        cs_bad._emit(TclEmitter(), 1)

    cs = ComputedSection(analysis=sec, E=200e3, G=76.9e3)
    em = TclEmitter()
    cs._emit(em, 1)
    line = em.lines()[-1]
    assert line.startswith("section Elastic 1 200000.0 ")
    assert len(line.split()) == 11
    # transformed-section area rides the deck line
    assert float(line.split()[4]) == pytest.approx(ea / 200e3, rel=1e-9)
