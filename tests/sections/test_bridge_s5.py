"""Tests — ADR 0078 S5: bridge binding (`ComputedSection` +
``to_elastic_section``) through the single shared lowering.

Gate verifies (plan S5): deck **byte-equality** vs a hand-typed
``ElasticSection`` (flat + full deck through ``Lobatto`` /
``forceBeamColumn``), memoization (N references = one solve),
reference-moduli fail-loud rules (geometric-only / homogeneous /
composite), 2-D vs 3-D ``ElasticSection`` form selection, and the
axis-mapping oracles (``Ixx_c → Iz``, ``Iyy_c → Iy``, ``As_y/A →
alphaY``, ``As_x/A → alphaZ``) including the swapped-rectangle
refutation.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.section import ComputedSection, ElasticSection
from apeGmsh.sections import (
    CompositeSectionError,
    SectionMaterial,
    SectionProperties,
)

from tests.opensees.fixtures.fem_stub import make_two_node_beam


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


def _rect_fem(g, b=2.0, h=4.0, *, pg=None, lc=0.3):
    _rect(g, b, h, x0=-b / 2, y0=-h / 2, pg=pg)
    return _mesh(g, lc=lc)


def _two_strip_composite(g):
    """Two conformal 1×2 rectangles side by side (shared LINE, per the
    conformal-fixture law) with different moduli."""
    geo = g.model.geometry
    pts = {}
    for i in range(3):
        for jy, y in enumerate((0.0, 2.0)):
            pts[(i, jy)] = geo.add_point(float(i), y, 0.0)
    verts = [geo.add_line(pts[(i, 0)], pts[(i, 1)]) for i in range(3)]
    bots = [geo.add_line(pts[(i, 0)], pts[(i + 1, 0)]) for i in range(2)]
    tops = [geo.add_line(pts[(i, 1)], pts[(i + 1, 1)]) for i in range(2)]
    surfs = []
    for i in range(2):
        loop = geo.add_curve_loop([bots[i], verts[i + 1], tops[i], verts[i]])
        surfs.append(geo.add_plane_surface([loop]))
    g.physical.add(2, [surfs[0]], name="soft")
    g.physical.add(2, [surfs[1]], name="stiff")
    fem = _mesh(g, lc=0.25)
    return SectionProperties(
        fem,
        materials={
            "soft": SectionMaterial(E=25e3, nu=0.2),
            "stiff": SectionMaterial(E=200e3, nu=0.3),
        },
        name="duo",
    )


# ─────────────────────────────────────────────────────────────────────
# axis mapping (the G-B shared assumption)
# ─────────────────────────────────────────────────────────────────────

def test_axis_mapping_tall_rectangle(g):
    """Tall 2×4 rectangle: strong axis is authoring-x → OpenSees Iz."""
    fem = _rect_fem(g, 2.0, 4.0, pg="bar")
    sec = SectionProperties(
        fem, materials={"bar": SectionMaterial(E=200e3, nu=0.3)},
        name="tall",
    )
    es = sec.to_elastic_section()
    assert es.E == pytest.approx(200e3)
    assert es.G == pytest.approx(200e3 / 2.6, rel=1e-12)
    assert es.A == pytest.approx(8.0, rel=1e-9)
    assert es.Iz == pytest.approx(2.0 * 4.0**3 / 12.0, rel=1e-9)   # Ixx_c
    assert es.Iy == pytest.approx(4.0 * 2.0**3 / 12.0, rel=1e-9)   # Iyy_c
    assert es.Iz > es.Iy
    # shear-area factors ride the same mapping (As_y→alphaY, As_x→alphaZ)
    warp = sec.warping()
    assert es.alphaY == pytest.approx(warp.As_y / 8.0, rel=1e-12)
    assert es.alphaZ == pytest.approx(warp.As_x / 8.0, rel=1e-12)
    assert es.J == pytest.approx(warp.J, rel=1e-12)


def test_axis_mapping_swapped_rectangle_refutation(g):
    """Wide 4×2 rectangle: the mapping must follow the authoring
    orientation (Iz < Iy here), never sort by magnitude."""
    fem = _rect_fem(g, 4.0, 2.0, pg="bar")
    sec = SectionProperties(
        fem, materials={"bar": SectionMaterial(E=200e3, nu=0.3)},
        name="wide",
    )
    es = sec.to_elastic_section()
    assert es.Iz == pytest.approx(4.0 * 2.0**3 / 12.0, rel=1e-9)
    assert es.Iy == pytest.approx(2.0 * 4.0**3 / 12.0, rel=1e-9)
    assert es.Iz < es.Iy


# ─────────────────────────────────────────────────────────────────────
# deck byte-equality
# ─────────────────────────────────────────────────────────────────────

def test_flat_deck_byte_equality_3d(g):
    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bar")
    es = sec.to_elastic_section(E=200e3, G=80e3)
    cs = ComputedSection(analysis=sec, E=200e3, G=80e3)

    e_hand, e_computed = TclEmitter(), TclEmitter()
    es._emit(e_hand, 7)
    cs._emit(e_computed, 7)
    assert e_computed.lines() == e_hand.lines()
    line = e_computed.lines()[-1]
    assert line.startswith("section Elastic 7 ")
    # 3-D form: type + tag + 8 params
    assert len(line.split()) == 11


def test_flat_deck_byte_equality_2d(g):
    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bar")
    es = sec.to_elastic_section(E=200e3, G=80e3, ndm=2)
    cs = ComputedSection(analysis=sec, E=200e3, G=80e3, ndm=2)

    e_hand, e_computed = TclEmitter(), TclEmitter()
    es._emit(e_hand, 7)
    cs._emit(e_computed, 7)
    assert e_computed.lines() == e_hand.lines()
    # 2-D shear-flexible form: E A Iz G alphaY
    assert len(e_computed.lines()[-1].split()) == 8


def test_form_selection_2d_vs_3d(g):
    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bar")
    es3 = sec.to_elastic_section(E=200e3, G=80e3)
    assert (es3.Iy, es3.J, es3.alphaZ) != (None, None, None)
    es2 = sec.to_elastic_section(E=200e3, G=80e3, ndm=2)
    assert es2.Iy is None and es2.J is None and es2.alphaZ is None
    assert es2.G == pytest.approx(80e3) and es2.alphaY is not None
    with pytest.raises(ValueError, match="ndm"):
        sec.to_elastic_section(E=200e3, G=80e3, ndm=4)


def test_full_deck_byte_equality_lobatto_forcebeamcolumn(g, tmp_path):
    """Two identical frame decks — one via ``ComputedSection`` inside
    ``Lobatto`` / ``forceBeamColumn``, one via a hand-typed
    ``ElasticSection`` with the same numbers — are byte-identical."""
    from apeGmsh.opensees import apeSees

    fem = _rect_fem(g, 2.0, 4.0, pg="bar")
    sec = SectionProperties(
        fem, materials={"bar": SectionMaterial(E=200e3, nu=0.3)},
        name="girder",
    )
    hand = sec.to_elastic_section()   # same lowering, eager numbers

    def _deck(section_factory, path):
        frame = make_two_node_beam()
        ops = apeSees(cast("object", frame))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        girder = section_factory(ops)
        integ = ops.beamIntegration.Lobatto(section=girder, n_ip=5)
        ops.element.forceBeamColumn(
            pg="Cols", transf=transf, integration=integ,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        ops.tcl(str(path))
        return path.read_text()

    deck_computed = _deck(
        lambda ops: ops.section.ComputedSection(analysis=sec),
        tmp_path / "computed.tcl",
    )
    deck_hand = _deck(
        lambda ops: ops.section.Elastic(
            E=hand.E, A=hand.A, Iz=hand.Iz, Iy=hand.Iy,
            G=hand.G, J=hand.J, alphaY=hand.alphaY, alphaZ=hand.alphaZ,
        ),
        tmp_path / "hand.tcl",
    )
    assert deck_computed == deck_hand
    assert "section Elastic" in deck_computed


# ─────────────────────────────────────────────────────────────────────
# memoization — N references, one solve
# ─────────────────────────────────────────────────────────────────────

def test_memoization_two_references_one_solve(g, monkeypatch):
    import apeGmsh.sections._analysis as analysis_mod

    fem = _rect_fem(g, 2.0, 4.0)
    calls = {"warping": 0}
    real = analysis_mod.compute_warping

    def counting(*args, **kwargs):
        calls["warping"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(analysis_mod, "compute_warping", counting)

    sec = SectionProperties(fem, name="bar")
    cs_a = ComputedSection(analysis=sec, E=200e3, G=80e3)
    cs_b = ComputedSection(analysis=sec, E=200e3, G=80e3)
    em = TclEmitter()
    cs_a._emit(em, 1)
    cs_b._emit(em, 2)
    sec.to_elastic_section(E=200e3, G=80e3)
    assert calls["warping"] == 1


# ─────────────────────────────────────────────────────────────────────
# reference-moduli rules (fail-loud at emit, naming the handle)
# ─────────────────────────────────────────────────────────────────────

def test_geometric_only_requires_both_moduli(g):
    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bare")
    with pytest.raises(ValueError, match="bare.*E=.*G="):
        sec.to_elastic_section()
    with pytest.raises(ValueError, match="bare.*G="):
        sec.to_elastic_section(E=200e3)
    with pytest.raises(ValueError, match="bare.*E="):
        sec.to_elastic_section(G=80e3)


def test_composite_requires_reference_moduli_at_emit(g):
    sec = _two_strip_composite(g)
    # construction is lazy — no analysis, no error yet
    cs = ComputedSection(analysis=sec)
    em = TclEmitter()
    with pytest.raises(CompositeSectionError, match="duo"):
        cs._emit(em, 1)
    with pytest.raises(CompositeSectionError, match="duo.*G="):
        sec.to_elastic_section(E=200e3)


def test_composite_transformed_lowering_values(g):
    sec = _two_strip_composite(g)
    geo, warp = sec.geometric(), sec.warping()
    e_ref, g_ref = 200e3, 76.9e3
    es = sec.to_elastic_section(E=e_ref, G=g_ref)
    assert es.A == pytest.approx(geo.EA / e_ref, rel=1e-12)
    assert es.Iz == pytest.approx(geo.EIxx_c / e_ref, rel=1e-12)
    assert es.Iy == pytest.approx(geo.EIyy_c / e_ref, rel=1e-12)
    assert es.J == pytest.approx(warp.GJ / g_ref, rel=1e-12)
    assert es.alphaY == pytest.approx(
        (warp.GAs_y / g_ref) / (geo.EA / e_ref), rel=1e-12
    )
    assert es.alphaZ == pytest.approx(
        (warp.GAs_x / g_ref) / (geo.EA / e_ref), rel=1e-12
    )
    # the deck reproduces the analyzer's rigidities exactly
    assert es.E * es.A == pytest.approx(geo.EA, rel=1e-12)
    assert es.G * es.J == pytest.approx(warp.GJ, rel=1e-12)


def test_homogeneous_matches_geometric_only_twin(g):
    """Homogeneous lowering (defaulted moduli) and a geometric-only
    twin lowered with the same explicit moduli agree on the geometry
    — except the shear alphas, which carry the material's ν."""
    fem = _rect_fem(g, 2.0, 4.0, pg="bar")
    mat = SectionMaterial(E=200e3, nu=0.0)   # ν = 0 → twin is exact
    hom = SectionProperties(
        fem, materials={"bar": mat}, name="hom",
    ).to_elastic_section()
    geo = SectionProperties(fem, name="geo").to_elastic_section(
        E=200e3, G=mat.shear_modulus,
    )
    for f in ("E", "A", "Iz", "Iy", "G", "J", "alphaY", "alphaZ"):
        assert getattr(hom, f) == pytest.approx(
            getattr(geo, f), rel=1e-12
        ), f


# ─────────────────────────────────────────────────────────────────────
# registration + construction validation
# ─────────────────────────────────────────────────────────────────────

def test_ns_registration_and_name_channel(g):
    from apeGmsh.opensees import apeSees

    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bar")

    frame = make_two_node_beam()
    ops = apeSees(cast("object", frame))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    cs = ops.section.ComputedSection(
        analysis=sec, E=200e3, G=80e3, name="girder",
    )
    assert isinstance(cs, ComputedSection)
    # registered name resolves through the standard Section channel
    integ = ops.beamIntegration.Lobatto(section="girder", n_ip=5)
    assert integ.section is cs


def test_construction_validation(g):
    fem = _rect_fem(g, 2.0, 4.0)
    sec = SectionProperties(fem, name="bar")
    with pytest.raises(ValueError, match="E must be > 0"):
        ComputedSection(analysis=sec, E=-1.0)
    with pytest.raises(ValueError, match="G must be > 0"):
        ComputedSection(analysis=sec, G=0.0)
    with pytest.raises(ValueError, match="ndm"):
        ComputedSection(analysis=sec, ndm=4)  # type: ignore[arg-type]
    cs = ComputedSection(analysis=sec, E=200e3, G=80e3)
    assert cs.dependencies() == ()
    # resolve() is the inspectable eager view of the same lowering
    assert isinstance(cs.resolve(), ElasticSection)
