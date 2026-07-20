"""Tests — ADR 0078 Amendment A2: ``kind="fiber"`` lowering + gate G-D.

F1: argument-family validation, exact-cover gate, fiber-sum identities
(exact — the Gauss areas are a quadrature partition), deck structure,
dependencies/tag resolution.

F2 (**gate G-D** — handedness of sign-bearing coordinates): the signed
``Σ E·A·y·z = EIxy_c`` identity on a rotated rectangle (a mirror flips
its sign), plus the end-to-end openseespy keystone — moment–curvature
initial slope vs ``EIxx_c``/``EIyy_c`` in both axes and the
``ElasticPP`` plateau vs ``Mp_xx`` in both signs (zeroLengthSection
harness; skipped when openseespy is absent).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.uniaxial import ElasticMaterial
from apeGmsh.opensees.section import ComputedSection
from apeGmsh.sections import SectionMaterial, SectionProperties
from apeGmsh.sections._lowering import lower_to_fiber


def _mesh(g, *, lc: float, order: int = 2):
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    if order > 1:
        g.mesh.generation.set_order(order)
    return g.mesh.queries.get_fem_data(dim=2)


def _rect_sec(g, b=2.0, h=4.0, *, rotate=None, lc=0.3, E=200e3, fy=None):
    g.sections.rect_face(b, h, label="bar", rotate=rotate)
    fem = _mesh(g, lc=lc)
    return SectionProperties(
        fem,
        materials={"bar": SectionMaterial(E=E, nu=0.3, fy=fy)},
        name="bar_sec",
    )


# ─────────────────────────────────────────────────────────────────────
# argument families + exact cover (fail at construction)
# ─────────────────────────────────────────────────────────────────────

def test_kind_validation_and_arg_families(g):
    sec = _rect_sec(g, lc=0.5)
    mat = ElasticMaterial(E=200e3)
    with pytest.raises(ValueError, match="kind must be"):
        ComputedSection(analysis=sec, kind="plastic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fiber-only arguments"):
        ComputedSection(analysis=sec, fibers={"bar": mat})
    with pytest.raises(ValueError, match="fiber-only arguments"):
        ComputedSection(analysis=sec, GJ=1.0)
    with pytest.raises(ValueError, match="no reference moduli"):
        ComputedSection(analysis=sec, kind="fiber",
                        fibers={"bar": mat}, E=200e3)
    with pytest.raises(ValueError, match="elastic lowering only"):
        ComputedSection(analysis=sec, kind="fiber",
                        fibers={"bar": mat}, ndm=2)
    with pytest.raises(ValueError, match="requires"):
        ComputedSection(analysis=sec, kind="fiber")
    with pytest.raises(ValueError, match="GJ must be > 0"):
        ComputedSection(analysis=sec, kind="fiber",
                        fibers={"bar": mat}, GJ=-1.0)


def test_exact_cover_gate(g):
    sec = _rect_sec(g, lc=0.5)
    mat = ElasticMaterial(E=200e3)
    with pytest.raises(ValueError, match=r"missing \['bar'\]"):
        ComputedSection(analysis=sec, kind="fiber", fibers={"ghost": mat})
    with pytest.raises(ValueError, match=r"unknown \['ghost'\]"):
        ComputedSection(analysis=sec, kind="fiber",
                        fibers={"bar": mat, "ghost": mat})


def test_geometric_only_rejected(g):
    g.sections.rect_face(1.0, 1.0, label="bar")
    fem = _mesh(g, lc=0.5)
    sec = SectionProperties(fem, name="geo_only")
    mat = ElasticMaterial(E=1.0)
    with pytest.raises(ValueError, match="geometric-only"):
        ComputedSection(analysis=sec, kind="fiber", fibers={"bar": mat})


# ─────────────────────────────────────────────────────────────────────
# fiber-sum identities (exact) — incl. the SIGNED EIxy mirror-catch
# ─────────────────────────────────────────────────────────────────────

def _fiber_sums(sec):
    """(ΣA, ΣEAy, ΣEAz, ΣEAy², ΣEAz², ΣEAyz) from the lowered data,
    with E per fiber taken from its region's SectionMaterial."""
    data = lower_to_fiber(sec)
    E_by_region = np.array(
        [sec.materials[name].E for name in data.region_names]
    )
    EA = E_by_region[data.region] * data.area
    return (
        float(data.area.sum()),
        float((EA * data.y).sum()),
        float((EA * data.z).sum()),
        float((EA * data.y**2).sum()),
        float((EA * data.z**2).sum()),
        float((EA * data.y * data.z).sum()),
    )


def test_fiber_sums_rotated_rectangle_signed(g):
    """30°-rotated rectangle: ΣEAyz must equal the analyzer's signed
    ``EIxy_c`` — a mirrored lowering (z → −z) flips this sign and
    fails here. This is the G-D mirror-catch."""
    sec = _rect_sec(g, 2.0, 4.0, rotate=30.0, lc=0.25)
    geo = sec.geometric()
    A, EAy, EAz, EAyy, EAzz, EAyz = _fiber_sums(sec)
    assert A == pytest.approx(geo.area, rel=1e-12)
    scale = geo.EA * max(abs(geo.cx), abs(geo.cy), 1.0)
    assert abs(EAy) < 1e-9 * scale and abs(EAz) < 1e-9 * scale
    assert EAyy == pytest.approx(geo.EIxx_c, rel=1e-9)
    assert EAzz == pytest.approx(geo.EIyy_c, rel=1e-9)
    assert abs(geo.EIxy_c) > 0.1 * geo.EIxx_c   # rotation made it live
    assert EAyz == pytest.approx(geo.EIxy_c, rel=1e-9)


def test_fiber_sums_composite_two_regions(g):
    """Two stacked strips E=200/25: per-region E weighting flows into
    the fiber sums exactly."""
    geo_ns = g.model.geometry
    t1 = geo_ns.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    t2 = geo_ns.add_rectangle(0.0, 1.0, 0.0, 1.0, 1.0)
    g.physical.add(2, [t1], name="soft")
    g.physical.add(2, [t2], name="stiff")
    g.model.boolean.fragment([(2, t1)], [(2, t2)], dim=2)
    fem = _mesh(g, lc=0.2)
    sec = SectionProperties(
        fem,
        materials={
            "soft": SectionMaterial(E=25e3, nu=0.2),
            "stiff": SectionMaterial(E=200e3, nu=0.3),
        },
        name="duo",
    )
    geo = sec.geometric()
    A, EAy, EAz, EAyy, EAzz, _ = _fiber_sums(sec)
    assert A == pytest.approx(geo.area, rel=1e-12)
    assert EAyy == pytest.approx(geo.EIxx_c, rel=1e-9)
    assert EAzz == pytest.approx(geo.EIyy_c, rel=1e-9)
    assert abs(EAy) < 1e-9 * geo.EA


# ─────────────────────────────────────────────────────────────────────
# deck structure + dependencies
# ─────────────────────────────────────────────────────────────────────

def test_deck_structure_and_gj_default(g):
    sec = _rect_sec(g, lc=0.4)
    mat = ElasticMaterial(E=200e3)
    cs = ComputedSection(analysis=sec, kind="fiber", fibers={"bar": mat})

    e = TclEmitter()
    set_tag_resolver(e, lambda p: 42)
    cs._emit(e, 9)
    lines = e.lines()
    data = lower_to_fiber(sec)
    gj = sec.warping().GJ
    assert f"section Fiber 9 -GJ {gj} {{" in lines
    assert lines[-1] == "}"
    fiber_lines = [ln for ln in lines if ln.strip().startswith("fiber ")]
    assert len(fiber_lines) == len(data.area)
    # every fiber references the resolved material tag
    assert all(ln.split()[-1] == "42" for ln in fiber_lines)

    # explicit GJ override
    cs2 = ComputedSection(analysis=sec, kind="fiber",
                          fibers={"bar": mat}, GJ=123.0)
    e2 = TclEmitter()
    set_tag_resolver(e2, lambda p: 42)
    cs2._emit(e2, 9)
    assert "section Fiber 9 -GJ 123.0 {" in e2.lines()


def test_dependencies_surface_fiber_materials(g):
    sec = _rect_sec(g, lc=0.5)
    mat = ElasticMaterial(E=200e3)
    cs = ComputedSection(analysis=sec, kind="fiber", fibers={"bar": mat})
    assert cs.dependencies() == (mat,)
    # elastic kind keeps the empty tuple
    assert ComputedSection(analysis=sec, E=1.0, G=1.0).dependencies() == ()


def test_full_bridge_deck(g, tmp_path):
    """kind='fiber' through apeSees: material emits before the section
    block; forceBeamColumn consumes it with zero consumer changes."""
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    sec = _rect_sec(g, lc=0.5)
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    # fiber materials are ordinary dependencies (P11): construct them
    # through the bridge so they are registered
    mat = ops.uniaxialMaterial.ElasticMaterial(E=200e3)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    girder = ops.section.ComputedSection(
        analysis=sec, kind="fiber", fibers={"bar": mat},
    )
    integ = ops.beamIntegration.Lobatto(section=girder, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    path = tmp_path / "fiber.tcl"
    ops.tcl(str(path))
    deck = path.read_text()
    assert deck.index("uniaxialMaterial Elastic") < deck.index("section Fiber")
    assert "forceBeamColumn" in deck and "-GJ" in deck


# ─────────────────────────────────────────────────────────────────────
# gate G-D — openseespy keystone (moment–curvature both axes + Mp±)
# ─────────────────────────────────────────────────────────────────────

def _mc_harness(points, gj, define_materials):
    """zeroLengthSection moment–curvature harness.

    ``define_materials(ops)`` defines the uniaxial laws and returns a
    ``{region_index: mat_tag}`` map.  Node 2's shear/torsion DOFs are
    fixed (a fiber section carries no shear stiffness); axial + both
    rotations stay free.
    """
    ops = pytest.importorskip("openseespy.opensees")
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    ops.node(1, 0.0, 0.0, 0.0)
    ops.node(2, 0.0, 0.0, 0.0)
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(2, 0, 1, 1, 1, 0, 0)   # free: axial (1), rot-y (5), rot-z (6)
    tag_of = define_materials(ops)
    ops.section("Fiber", 1, "-GJ", float(gj))
    for y, z, a, r in points:
        ops.fiber(float(y), float(z), float(a), tag_of[int(r)])
    ops.element("zeroLengthSection", 1, 1, 2, 1)
    return ops


def _points(sec):
    data = lower_to_fiber(sec)
    return list(zip(data.y, data.z, data.area, data.region)), data


def test_gd_elastic_slope_both_axes(g):
    """M–κ initial slope: Mz/κz = EIxx_c and My/κy = EIyy_c — the
    keystone identity, end-to-end through OpenSees (catches an axis
    swap the even-quantity elastic path cannot)."""
    E = 200e3
    sec = _rect_sec(g, 2.0, 4.0, lc=0.3, E=E)
    geo = sec.geometric()
    points, _ = _points(sec)
    gj = sec.warping().GJ

    def elastic_mats(o):
        o.uniaxialMaterial("Elastic", 1, E)
        return {0: 1}

    for dof, EI in ((6, geo.EIxx_c), (5, geo.EIyy_c)):
        ops = _mc_harness(points, gj, elastic_mats)
        M = 1.0e6
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        load = [0.0] * 6
        load[dof - 1] = M
        ops.load(2, *load)
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        assert ops.analyze(1) == 0
        kappa = ops.nodeDisp(2, dof)
        assert M / kappa == pytest.approx(EI, rel=1e-9)
        ops.wipe()


def test_gd_elasticpp_plateau_vs_mp(g):
    """ElasticPP fibers pushed deep past yield: |M| plateaus at the
    analyzer's ``Mp_xx`` in BOTH signs (an asymmetric-response mirror
    would break sign symmetry; magnitude gates the plastic modulus)."""
    E, fy = 200e3, 345.0
    b, h = 2.0, 4.0
    sec = _rect_sec(g, b, h, lc=0.25, E=E, fy=fy)
    plas = sec.plastic()
    points, _ = _points(sec)
    gj = sec.warping().GJ
    kappa_y = 2.0 * fy / (E * h)          # first-yield curvature

    def pp_mats(o):
        o.uniaxialMaterial("ElasticPP", 1, E, fy / E)
        return {0: 1}

    for sign in (+1.0, -1.0):
        ops = _mc_harness(points, gj, pp_mats)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        ops.load(2, 0.0, 0.0, 0.0, 0.0, 0.0, sign)   # reference |Mz| = 1
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Plain")
        ops.integrator("DisplacementControl", 2, 6, sign * kappa_y)
        ops.algorithm("Newton")
        ops.test("NormDispIncr", 1e-10, 25)
        ops.analysis("Static")
        assert ops.analyze(30) == 0        # κ = 30·κ_y — deep plastic
        # reference load has unit magnitude → |M| = load factor
        assert abs(ops.getLoadFactor(1)) == pytest.approx(
            plas.Mp_xx, rel=1e-2
        )
        ops.wipe()


def test_gd_signed_odd_moments_l_section(g):
    """W1 killer (adversarial review) — the Gauss-fiber sums of the
    ODD third moments on an L-section, signed, against composite-
    rectangle closed forms. A coordinated 180° flip of the lowering
    (y→−y AND z→−z) negates Σy³/Σz³ while leaving every even-moment
    and product identity above untouched — this is the one test in
    the G-D family that dies."""
    import numpy as np

    # L-shape: 3×1 leg + 1×3 leg above it (corner at the origin)
    geo_ns = g.model.geometry
    pts = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 4), (0, 4)]
    tags = [geo_ns.add_point(float(x), float(y), 0.0) for x, y in pts]
    lines = [
        geo_ns.add_line(tags[i], tags[(i + 1) % len(tags)])
        for i in range(len(tags))
    ]
    loop = geo_ns.add_curve_loop(lines)
    surf = geo_ns.add_plane_surface([loop])
    g.physical.add(2, [surf], name="L")
    g.mesh.sizing.set_global_size(0.08)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    sec = SectionProperties(
        fem, materials={"L": SectionMaterial(E=1.0, nu=0.0)}, name="L",
    )
    geo = sec.geometric()

    def rect_third(lo, hi, width, c):
        """∫(t−c)³ over t∈[lo,hi] times width (closed form)."""
        return width * ((hi - c) ** 4 - (lo - c) ** 4) / 4.0

    # rects: horizontal leg x∈[0,3],y∈[0,1]; vertical leg x∈[0,1],y∈[1,4]
    y3_hand = rect_third(0.0, 1.0, 3.0, geo.cy) + rect_third(
        1.0, 4.0, 1.0, geo.cy,
    )
    z3_hand = rect_third(0.0, 3.0, 1.0, geo.cx) + rect_third(
        0.0, 1.0, 3.0, geo.cx,
    )
    data = lower_to_fiber(sec)
    y3 = float(np.sum(data.area * data.y**3))
    z3 = float(np.sum(data.area * data.z**3))
    # third moments are cubic — the 3-pt tri rule is degree-2, so the
    # comparison is mesh-converged, not exact; signs are the point
    assert y3_hand != pytest.approx(0.0, abs=1e-3)
    assert z3_hand != pytest.approx(0.0, abs=1e-3)
    assert y3 == pytest.approx(y3_hand, rel=5e-3)
    assert z3 == pytest.approx(z3_hand, rel=5e-3)
