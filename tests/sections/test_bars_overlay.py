"""Tests — ADR 0080 B3: ``bars=`` overlay on ``ComputedSection`` +
**gate G-E**.

Bar coordinates are sign-bearing values crossing the authoring→local
mapping (the G-D class). The gate: signed ``ΣEAyz`` with an asymmetric
corner bar (mirror-catch), the exact M–κ initial-slope identity with
bars (transformed section), and the ``ElasticPP`` bar plateau vs the
hand-computed ``ΣA_s·fy·d`` couple in both signs.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.material.uniaxial import ElasticMaterial
from apeGmsh.opensees.section import Bar, ComputedSection
from apeGmsh.sections import (
    SectionDocument,
    SectionDocumentError,
    SectionMaterial,
    SectionProperties,
)


def _rect_sec(g, b=2.0, h=4.0, *, lc=0.3, E=25e3):
    g.sections.rect_face(b, h, label="conc")
    g.mesh.sizing.set_global_size(lc)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    return SectionProperties(
        fem, materials={"conc": SectionMaterial(E=E, nu=0.2)},
        name="rc_bar_sec",
    )


def _resolved_fiber_arrays(cs):
    """(y, z, area, E) arrays of the resolved Fiber's points."""
    fib = cs.resolve()
    y = np.array([p.y for p in fib.fibers])
    z = np.array([p.z for p in fib.fibers])
    a = np.array([p.area for p in fib.fibers])
    E = np.array([p.material.E for p in fib.fibers])
    return y, z, a, E


# ─────────────────────────────────────────────────────────────────────
# validation + deck structure
# ─────────────────────────────────────────────────────────────────────

def test_bars_validation(g):
    sec = _rect_sec(g, lc=0.6)
    conc = ElasticMaterial(E=25e3)
    steel = ElasticMaterial(E=200e3)
    with pytest.raises(ValueError, match="fiber-only"):
        ComputedSection(
            analysis=sec,
            bars=(Bar(material=steel, x=0.0, y=0.0, area=1.0),),
        )
    with pytest.raises(ValueError, match="area must be > 0"):
        Bar(material=steel, x=0.0, y=0.0, area=0.0)
    cs = ComputedSection(
        analysis=sec, kind="fiber", fibers={"conc": conc},
        bars=(Bar(material=steel, x=0.5, y=3.5, area=2.0),),
    )
    # bar material rides dependencies() for tag resolution (dedup)
    deps = cs.dependencies()
    assert conc in deps and steel in deps and len(deps) == 2


def test_bars_append_to_gauss_fibers(g):
    sec = _rect_sec(g, lc=0.5)
    conc = ElasticMaterial(E=25e3)
    steel = ElasticMaterial(E=200e3)
    plain = ComputedSection(analysis=sec, kind="fiber",
                            fibers={"conc": conc})
    n_gauss = len(plain.resolve().fibers)
    bars = tuple(
        Bar(material=steel, x=x, y=y, area=2.0)
        for x, y in ((-0.7, -1.7), (0.7, -1.7), (-0.7, 1.7), (0.7, 1.7))
    )
    cs = ComputedSection(analysis=sec, kind="fiber",
                         fibers={"conc": conc}, bars=bars)
    fib = cs.resolve()
    assert len(fib.fibers) == n_gauss + 4
    # bars land AFTER the Gauss fibers, mapped about the elastic
    # centroid (rect_face is centred on the origin -> centroid ~(0,0);
    # local y = authoring y, local z = authoring x)
    tail = fib.fibers[-4:]
    got = {(round(p.z, 9), round(p.y, 9)) for p in tail}
    assert got == {
        (-0.7, -1.7), (0.7, -1.7), (-0.7, 1.7), (0.7, 1.7),
    }
    e = TclEmitter()
    set_tag_resolver(e, lambda p: 5)
    cs._emit(e, 9)
    fiber_lines = [ln for ln in e.lines() if ln.strip().startswith("fiber ")]
    assert len(fiber_lines) == n_gauss + 4


# ─────────────────────────────────────────────────────────────────────
# gate G-E — signed identities + M–κ keystone
# ─────────────────────────────────────────────────────────────────────

def test_ge_signed_eayz_mirror_catch(g):
    """ONE off-axis bar: the bars' ΣEAyz contribution equals
    E·A·(y−cy)(x−cx) with its true sign — a mirrored mapping
    (z → −z) flips it."""
    E_c, E_s, A_s = 25e3, 200e3, 2.0
    sec = _rect_sec(g, 2.0, 4.0, lc=0.25, E=E_c)
    geo = sec.geometric()
    conc = ElasticMaterial(E=E_c)
    steel = ElasticMaterial(E=E_s)
    x0, y0 = 0.7, 1.6                     # +z, +y quadrant bar
    cs = ComputedSection(
        analysis=sec, kind="fiber", fibers={"conc": conc},
        bars=(Bar(material=steel, x=x0, y=y0, area=A_s),),
    )
    y, z, a, E = _resolved_fiber_arrays(cs)
    total_eayz = float((E * a * y * z).sum())
    conc_eayz = geo.EIxy_c                # ~0 for the upright rectangle
    bar_eayz = E_s * A_s * (y0 - geo.cy) * (x0 - geo.cx)
    assert bar_eayz > 0.0
    assert total_eayz - conc_eayz == pytest.approx(bar_eayz, rel=1e-12)
    # transformed-section EA and first moments hold too
    assert float((E * a).sum()) == pytest.approx(
        geo.EA + E_s * A_s, rel=1e-9,
    )


def test_ge_mk_slope_exact_with_bars(g):
    """M–κ initial slope through OpenSees == the fiber sum ΣEAy²
    (concrete Gauss fibers + bars as n·A_s steel) — exact identity."""
    ops_py = pytest.importorskip("openseespy.opensees")
    E_c, E_s, A_s = 25e3, 200e3, 2.0
    sec = _rect_sec(g, 2.0, 4.0, lc=0.3, E=E_c)
    conc = ElasticMaterial(E=E_c)
    steel = ElasticMaterial(E=E_s)
    bars = tuple(
        Bar(material=steel, x=x, y=y, area=A_s)
        for x, y in ((-0.7, -1.6), (0.7, -1.6), (-0.7, 1.6), (0.7, 1.6))
    )
    cs = ComputedSection(analysis=sec, kind="fiber",
                         fibers={"conc": conc}, bars=bars)
    y, z, a, E = _resolved_fiber_arrays(cs)
    EI_exact = float((E * a * y * y).sum())

    ops_py.wipe()
    ops_py.model("basic", "-ndm", 3, "-ndf", 6)
    ops_py.node(1, 0.0, 0.0, 0.0)
    ops_py.node(2, 0.0, 0.0, 0.0)
    ops_py.fix(1, 1, 1, 1, 1, 1, 1)
    ops_py.fix(2, 0, 1, 1, 1, 0, 0)
    ops_py.uniaxialMaterial("Elastic", 1, E_c)
    ops_py.uniaxialMaterial("Elastic", 2, E_s)
    gj = sec.warping().GJ
    ops_py.section("Fiber", 1, "-GJ", float(gj))
    fib = cs.resolve()
    for p in fib.fibers:
        tag = 2 if p.material is steel else 1
        ops_py.fiber(float(p.y), float(p.z), float(p.area), tag)
    ops_py.element("zeroLengthSection", 1, 1, 2, 1)
    M = 1.0e6
    ops_py.timeSeries("Linear", 1)
    ops_py.pattern("Plain", 1, 1)
    ops_py.load(2, 0.0, 0.0, 0.0, 0.0, 0.0, M)
    ops_py.system("FullGeneral")
    ops_py.numberer("Plain")
    ops_py.constraints("Plain")
    ops_py.integrator("LoadControl", 1.0)
    ops_py.algorithm("Linear")
    ops_py.analysis("Static")
    assert ops_py.analyze(1) == 0
    kappa = ops_py.nodeDisp(2, 6)
    assert M / kappa == pytest.approx(EI_exact, rel=1e-9)
    ops_py.wipe()


def test_ge_pp_bar_plateau_both_signs(g):
    """Near-zero concrete + ElasticPP bars: the moment plateaus at
    ΣA_s·fy·|ȳ| in BOTH signs (an asymmetric bar layout would break
    sign symmetry if the y mapping flipped)."""
    ops_py = pytest.importorskip("openseespy.opensees")
    E_s, A_s, fy = 200e3, 2.0, 345.0
    sec = _rect_sec(g, 2.0, 4.0, lc=0.4, E=1e-6)   # concrete ~ nothing
    conc = ElasticMaterial(E=1e-6)
    d = 1.6                                        # bars at ȳ = ±1.6
    bars_xy = ((-0.7, -d), (0.7, -d), (-0.7, d), (0.7, d))
    steel = ElasticMaterial(E=E_s)                 # marker for tag pick
    cs = ComputedSection(
        analysis=sec, kind="fiber", fibers={"conc": conc},
        bars=tuple(
            Bar(material=steel, x=x, y=y, area=A_s) for x, y in bars_xy
        ),
    )
    fib = cs.resolve()
    Mp_bars = 4 * A_s * fy * d                     # ΣA_s·fy·|ȳ|
    kappa_y = fy / E_s / d

    for sign in (+1.0, -1.0):
        ops_py.wipe()
        ops_py.model("basic", "-ndm", 3, "-ndf", 6)
        ops_py.node(1, 0.0, 0.0, 0.0)
        ops_py.node(2, 0.0, 0.0, 0.0)
        ops_py.fix(1, 1, 1, 1, 1, 1, 1)
        ops_py.fix(2, 0, 1, 1, 1, 0, 0)
        ops_py.uniaxialMaterial("Elastic", 1, 1e-6)
        ops_py.uniaxialMaterial("ElasticPP", 2, E_s, fy / E_s)
        ops_py.section("Fiber", 1, "-GJ", 1.0)
        for p in fib.fibers:
            tag = 2 if p.material is steel else 1
            ops_py.fiber(float(p.y), float(p.z), float(p.area), tag)
        ops_py.element("zeroLengthSection", 1, 1, 2, 1)
        ops_py.timeSeries("Linear", 1)
        ops_py.pattern("Plain", 1, 1)
        ops_py.load(2, 0.0, 0.0, 0.0, 0.0, 0.0, sign)
        ops_py.system("FullGeneral")
        ops_py.numberer("Plain")
        ops_py.constraints("Plain")
        ops_py.integrator("DisplacementControl", 2, 6, sign * kappa_y)
        ops_py.algorithm("Newton")
        ops_py.test("NormDispIncr", 1e-10, 25)
        ops_py.analysis("Static")
        assert ops_py.analyze(20) == 0             # κ = 20·κ_y
        assert abs(ops_py.getLoadFactor(1)) == pytest.approx(
            Mp_bars, rel=1e-6,
        )
        ops_py.wipe()


# ─────────────────────────────────────────────────────────────────────
# document surface — bars + to_section on a continuum doc
# ─────────────────────────────────────────────────────────────────────

def test_bar_line_expansion():
    doc = SectionDocument.new(name="beam", kind="continuum")
    doc.set_material(
        "conc", E=25e3, nu=0.2,
        uniaxial=("ElasticMaterial", {"E": 25e3}),
    )
    doc.set_material("steel", uniaxial=("ElasticMaterial", {"E": 200e3}))
    doc.add_shape("rect_face", id="conc", b=300.0, h=600.0)
    doc.add_bar_line(material="steel", n=4, area=387.0,
                     start=(-110.0, -260.0), end=(110.0, -260.0))
    pts = doc._expand_bars()
    assert [p["x"] for p in pts] == pytest.approx(
        [-110.0, -110.0 + 220.0 / 3, 110.0 - 220.0 / 3, 110.0]
    )
    assert all(p["y"] == -260.0 for p in pts)
    with pytest.raises(SectionDocumentError, match="must be >= 2"):
        doc.add_bar_line(material="steel", n=1, area=1.0,
                         start=(0, 0), end=(1, 0))
    fib = SectionDocument.new(kind="fiber")
    with pytest.raises(SectionDocumentError, match="continuum-lane"):
        fib.add_bar(material="steel", x=0.0, y=0.0, area=1.0)


def test_continuum_doc_to_section_end_to_end(tmp_path):
    """Continuum document + bars overlay → to_section → registered
    ComputedSection(kind='fiber') with the bars appended; round-trips
    through JSON first."""
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    doc = SectionDocument.new(name="rc_col", kind="continuum")
    doc.set_material(
        "conc", E=25e3, nu=0.2,
        uniaxial=("ElasticMaterial", {"E": 25e3}),
    )
    doc.set_material("steel", uniaxial=("ElasticMaterial", {"E": 200e3}))
    doc.add_shape("rect_face", id="conc", b=400.0, h=400.0)
    for sx in (-1, 1):
        doc.add_bar_line(
            material="steel", n=3, area=510.0,
            start=(sx * 150.0, -150.0), end=(sx * 150.0, 150.0),
        )
    doc.set_mesh(lc=80.0)
    p = tmp_path / "rc.section.json"
    doc.save(p)
    doc = SectionDocument.open(p)

    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    cs = doc.to_section(ops)
    assert isinstance(cs, ComputedSection)
    assert cs.kind == "fiber" and len(cs.bars) == 6
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=cs, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    path = tmp_path / "rc.tcl"
    ops.tcl(str(path))
    deck = path.read_text()
    assert deck.count("uniaxialMaterial Elastic") == 2
    assert "section Fiber" in deck and "-GJ" in deck

    # missing uniaxial role fails loud
    doc2 = SectionDocument.new(name="no_spec", kind="continuum")
    doc2.set_material("conc", E=25e3, nu=0.2)
    doc2.add_shape("rect_face", id="conc", b=1.0, h=1.0)
    doc2.set_mesh(lc=0.5)
    ops2 = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops2.model(ndm=3, ndf=6)
    with pytest.raises(SectionDocumentError, match="no uniaxial spec"):
        doc2.to_section(ops2)


# ─────────────────────────────────────────────────────────────────────
# gate G-E hardening (adversarial review): point-symmetry breakers
# ─────────────────────────────────────────────────────────────────────

def test_ge_axial_coupling_sign_single_bar(g):
    """W1 killer — ONE bar above the centroid: under +Mz with the
    axial DOF free, the axial displacement is ε0 = −(ΣEAy/ΣEA)·κ with
    HAND-computed values (no self-reference). A coordinated 180° flip
    of the emitted section (y→−y, z→−z) flips ΣEAy and therefore the
    axial-displacement sign — this test dies; the symmetric-layout
    tests above cannot see it."""
    ops_py = pytest.importorskip("openseespy.opensees")
    E_c, E_s, A_s = 25e3, 200e3, 2.0
    b, h, y0 = 2.0, 4.0, 1.6
    sec = _rect_sec(g, b, h, lc=0.3, E=E_c)
    conc = ElasticMaterial(E=E_c)
    steel = ElasticMaterial(E=E_s)
    cs = ComputedSection(
        analysis=sec, kind="fiber", fibers={"conc": conc},
        bars=(Bar(material=steel, x=0.0, y=y0, area=A_s),),
    )
    # hand values (rect quadrature is exact; fiber origin at the
    # elastic centroid of the CONCRETE). OpenSees FiberSection3d
    # computes the fibers' AREA centroid (computeCentroid) and
    # measures the axial DOF there: with the strain field solving
    # N=0 under +Mz, ux = κ·(ΣEAy/ΣEA − ȳ_area). Every term flips
    # sign under the 180° double flip.
    EA = E_c * b * h + E_s * A_s
    EAy = E_s * A_s * y0
    EAyy = E_c * b * h**3 / 12.0 + E_s * A_s * y0**2
    EI_eff = EAyy - EAy**2 / EA
    y_bar_area = (A_s * y0) / (b * h + A_s)
    M = 1.0e5
    kappa_hand = M / EI_eff
    eps0_hand = (EAy / EA - y_bar_area) * kappa_hand

    fib = cs.resolve()
    ops_py.wipe()
    ops_py.model("basic", "-ndm", 3, "-ndf", 6)
    ops_py.node(1, 0.0, 0.0, 0.0)
    ops_py.node(2, 0.0, 0.0, 0.0)
    ops_py.fix(1, 1, 1, 1, 1, 1, 1)
    ops_py.fix(2, 0, 1, 1, 1, 0, 0)
    ops_py.uniaxialMaterial("Elastic", 1, E_c)
    ops_py.uniaxialMaterial("Elastic", 2, E_s)
    ops_py.section("Fiber", 1, "-GJ", 1.0)
    for p in fib.fibers:
        tag = 2 if p.material is steel else 1
        ops_py.fiber(float(p.y), float(p.z), float(p.area), tag)
    ops_py.element("zeroLengthSection", 1, 1, 2, 1)
    ops_py.timeSeries("Linear", 1)
    ops_py.pattern("Plain", 1, 1)
    ops_py.load(2, 0.0, 0.0, 0.0, 0.0, 0.0, M)
    ops_py.system("FullGeneral")
    ops_py.numberer("Plain")
    ops_py.constraints("Plain")
    ops_py.integrator("LoadControl", 1.0)
    ops_py.algorithm("Linear")
    ops_py.analysis("Static")
    assert ops_py.analyze(1) == 0
    # SIGNED assertions: bar above the centroid -> positive coupling
    # (flips with the bar side — the point-symmetry breaker)
    assert ops_py.nodeDisp(2, 1) == pytest.approx(eps0_hand, rel=1e-4)
    assert ops_py.nodeDisp(2, 1) > 0.0
    assert ops_py.nodeDisp(2, 6) == pytest.approx(kappa_hand, rel=1e-4)
    ops_py.wipe()


def test_bar_mapping_off_centre_section(g):
    """W4 killer — a translated section: the bar maps about the TRUE
    elastic centroid, not the origin (dropping the −cx/−cy subtraction
    would land the bar at (5.7, 4.6) in local coords)."""
    g.sections.rect_face(2.0, 4.0, label="conc", translate=(5.0, 3.0))
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)
    g.mesh.generation.set_order(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    sec = SectionProperties(
        fem, materials={"conc": SectionMaterial(E=25e3, nu=0.2)},
        name="offset",
    )
    geo = sec.geometric()
    assert (geo.cx, geo.cy) == (
        pytest.approx(5.0, abs=1e-9), pytest.approx(3.0, abs=1e-9),
    )
    steel = ElasticMaterial(E=200e3)
    cs = ComputedSection(
        analysis=sec, kind="fiber",
        fibers={"conc": ElasticMaterial(E=25e3)},
        bars=(Bar(material=steel, x=5.7, y=4.6, area=2.0),),
    )
    tail = cs.resolve().fibers[-1]
    assert (tail.z, tail.y) == (
        pytest.approx(0.7, abs=1e-9), pytest.approx(1.6, abs=1e-9),
    )
