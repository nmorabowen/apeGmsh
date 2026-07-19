"""Tests — ADR 0080 B2: fiber-lane documents + RC templates.

Closed-form expansion oracles (bar counts/positions, exact area
partitions incl. the core/cover split), re-expansion determinism, JSON
round-trip, lane guards, `CircPatch` emission, and the `to_section`
handoff deck golden through a real apeSees bridge.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.sections import SectionDocument, SectionDocumentError
from apeGmsh.sections._rc_templates import expand_template, template_roles


def _rc_doc(*, core_split=False) -> SectionDocument:
    doc = SectionDocument.new(name="col40x40", kind="fiber")
    doc.set_material(
        "conc", E=25e3, nu=0.2,
        uniaxial=("ElasticMaterial", {"E": 25e3}),
    )
    doc.set_material("steel", uniaxial=("ElasticMaterial", {"E": 200e3}))
    mats = (
        {"core": "conc", "cover": "conc", "bars": "steel"}
        if core_split else {"concrete": "conc", "bars": "steel"}
    )
    doc.add_template(
        "rc_rect_column", materials=mats,
        b=400.0, h=400.0, cover=50.0, bars_x=3, bars_y=3,
        bar_area=510.0, core_split=core_split,
    )
    doc.set_GJ(1.0e12)
    return doc


# ─────────────────────────────────────────────────────────────────────
# template expansion oracles (closed-form, exact)
# ─────────────────────────────────────────────────────────────────────

def test_rect_column_bar_layout():
    out = expand_template("rc_rect_column", dict(
        b=400.0, h=500.0, cover=50.0, bars_x=4, bars_y=3,
        bar_area=510.0,
    ))
    yc, zc = 500.0 / 2 - 50.0, 400.0 / 2 - 50.0
    # top + bottom rows as layers of bars_x
    assert [la["n_bars"] for la in out["layers"]] == [4, 4]
    assert out["layers"][0]["yI"] == out["layers"][0]["yJ"] == yc
    assert out["layers"][1]["yI"] == -yc
    assert (out["layers"][0]["zI"], out["layers"][0]["zJ"]) == (-zc, zc)
    # side interiors: (bars_y − 2) rows × 2 sides
    assert len(out["points"]) == 2
    assert {pt["z"] for pt in out["points"]} == {-zc, zc}
    assert all(pt["y"] == 0.0 for pt in out["points"])
    # total steel area exact: 2·bars_x + 2·(bars_y − 2)
    total = sum(la["n_bars"] * la["area"] for la in out["layers"])
    total += sum(pt["area"] for pt in out["points"])
    assert total == pytest.approx((2 * 4 + 2 * 1) * 510.0, rel=1e-15)


def test_rect_column_patch_partition_exact():
    b, h = 400.0, 500.0
    plain = expand_template("rc_rect_column", dict(
        b=b, h=h, cover=50.0, bars_x=2, bars_y=2, bar_area=1.0,
    ))
    split = expand_template("rc_rect_column", dict(
        b=b, h=h, cover=50.0, bars_x=2, bars_y=2, bar_area=1.0,
        core_split=True,
    ))

    def _area(p):
        return abs((p["yJ"] - p["yI"]) * (p["zJ"] - p["zI"]))

    assert sum(_area(p) for p in plain["patches"]) == pytest.approx(b * h)
    assert sum(_area(p) for p in split["patches"]) == pytest.approx(b * h)
    core = [p for p in split["patches"] if p["material"] == "core"]
    assert len(core) == 1
    assert _area(core[0]) == pytest.approx((b - 100.0) * (h - 100.0))
    assert len(split["patches"]) == 5


def test_circ_column_ring():
    d, cover, n = 600.0, 60.0, 8
    out = expand_template("rc_circ_column", dict(
        d=d, cover=cover, n_bars=n, bar_area=510.0, core_split=True,
    ))
    rb = d / 2 - cover
    assert len(out["points"]) == n
    for i, pt in enumerate(out["points"]):
        theta = math.pi / 2 + 2 * math.pi * i / n
        assert pt["y"] == pytest.approx(rb * math.sin(theta), abs=1e-12)
        assert pt["z"] == pytest.approx(rb * math.cos(theta), abs=1e-12)
        assert math.hypot(pt["y"], pt["z"]) == pytest.approx(rb)
    # first bar at the top (+y)
    assert out["points"][0]["y"] == pytest.approx(rb)
    core, cov = out["patches"]
    assert (core["int_rad"], core["ext_rad"]) == (0.0, rb)
    assert (cov["int_rad"], cov["ext_rad"]) == (rb, d / 2)


def test_beam_rows_only():
    out = expand_template("rc_beam", dict(
        b=300.0, h=600.0, cover=40.0, top_bars=2, bottom_bars=4,
        bar_area=387.0,
    ))
    assert [la["n_bars"] for la in out["layers"]] == [2, 4]
    assert out["points"] == []
    assert out["layers"][0]["yI"] == 600.0 / 2 - 40.0


def test_expansion_deterministic_and_cover_editable():
    params = dict(b=400.0, h=400.0, cover=50.0, bars_x=3, bars_y=3,
                  bar_area=510.0)
    a = expand_template("rc_rect_column", dict(params))
    b_ = expand_template("rc_rect_column", dict(params))
    assert a == b_
    edited = expand_template(
        "rc_rect_column", {**params, "cover": 40.0},
    )
    # same counts, moved coordinates
    assert len(edited["layers"]) == len(a["layers"])
    assert len(edited["points"]) == len(a["points"])
    assert edited["layers"][0]["yI"] == 400.0 / 2 - 40.0


def test_template_param_gates():
    with pytest.raises(ValueError, match="cover"):
        expand_template("rc_rect_column", dict(
            b=100.0, h=100.0, cover=60.0, bars_x=2, bars_y=2,
            bar_area=1.0,
        ))
    with pytest.raises(ValueError, match="INCLUDING corners"):
        expand_template("rc_rect_column", dict(
            b=400.0, h=400.0, cover=50.0, bars_x=1, bars_y=2,
            bar_area=1.0,
        ))
    with pytest.raises(ValueError, match="n_bars"):
        expand_template("rc_circ_column", dict(
            d=600.0, cover=60.0, n_bars=3, bar_area=1.0,
        ))
    assert template_roles({"core_split": True}) == ("core", "cover", "bars")
    assert template_roles({}) == ("concrete", "bars")


# ─────────────────────────────────────────────────────────────────────
# document round-trip + recipe + lane guards
# ─────────────────────────────────────────────────────────────────────

def test_fiber_doc_round_trip_and_recipe(tmp_path):
    doc = _rc_doc(core_split=True)
    p = tmp_path / "col.section.json"
    doc.save(p)
    reopened = SectionDocument.open(p)
    assert reopened == doc

    recipe = reopened.build()
    areas = recipe.areas_by_material()
    assert areas["conc"] == pytest.approx(400.0 * 400.0)
    assert areas["steel"] == pytest.approx(8 * 510.0)
    assert recipe.GJ == pytest.approx(1.0e12)


def test_lane_guards():
    fib = SectionDocument.new(kind="fiber")
    with pytest.raises(SectionDocumentError, match="continuum-lane"):
        fib.add_shape("rect_face", id="x", b=1.0, h=1.0)
    with pytest.raises(SectionDocumentError, match="continuum-lane"):
        fib.set_mesh(lc=1.0)
    cont = SectionDocument.new(kind="continuum")
    with pytest.raises(SectionDocumentError, match="fiber-lane"):
        cont.add_point(material="m", y=0.0, z=0.0, area=1.0)
    with pytest.raises(SectionDocumentError, match="fiber-lane"):
        cont.set_GJ(1.0)


def test_material_role_gates():
    doc = SectionDocument.new(kind="fiber")
    with pytest.raises(SectionDocumentError, match="role"):
        doc.set_material("void")
    with pytest.raises(SectionDocumentError, match="come together"):
        doc.set_material("half", E=1.0)
    doc.set_material("ok", uniaxial=("ElasticMaterial", {"E": 1.0}))
    doc.add_point(material="ghost", y=0.0, z=0.0, area=1.0)
    with pytest.raises(SectionDocumentError, match="not in the table"):
        doc.build()


def test_template_role_cover_gate():
    doc = SectionDocument.new(kind="fiber")
    doc.set_material("conc", uniaxial=("ElasticMaterial", {"E": 1.0}))
    with pytest.raises(SectionDocumentError, match="cover roles"):
        doc.add_template(
            "rc_rect_column",
            materials={"concrete": "conc"},   # missing "bars"
            b=400.0, h=400.0, cover=50.0, bars_x=2, bars_y=2,
            bar_area=1.0,
        )


# ─────────────────────────────────────────────────────────────────────
# bridge handoff — deck golden
# ─────────────────────────────────────────────────────────────────────

def test_to_section_deck_golden(tmp_path):
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    doc = _rc_doc(core_split=True)
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    sec = doc.to_section(ops)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    path = tmp_path / "col.tcl"
    ops.tcl(str(path))
    deck = path.read_text()

    # one bridge material per document material name
    assert deck.count("uniaxialMaterial Elastic") == 2
    # core_split rect column: 5 rect patches, 2 layers, 2 side points
    assert deck.count("patch rect") == 5
    assert deck.count("layer straight") == 2
    assert deck.count("    fiber ") == 2
    assert "-GJ 1000000000000.0" in deck
    assert deck.index("uniaxialMaterial Elastic") < deck.index("section Fiber")


def test_circ_patch_emission():
    from apeGmsh.opensees._internal.tag_resolution import set_tag_resolver
    from apeGmsh.opensees.emitter.tcl import TclEmitter
    from apeGmsh.opensees.material.uniaxial import ElasticMaterial
    from apeGmsh.opensees.section import CircPatch, Fiber

    mat = ElasticMaterial(E=25e3)
    with pytest.raises(ValueError, match="int_rad"):
        CircPatch(material=mat, n_circ=8, n_rad=2, yC=0.0, zC=0.0,
                  int_rad=2.0, ext_rad=1.0)
    sec = Fiber(patches=(
        CircPatch(material=mat, n_circ=16, n_rad=4, yC=0.0, zC=0.0,
                  int_rad=0.0, ext_rad=300.0),
    ))
    e = TclEmitter()
    set_tag_resolver(e, lambda p: 7)
    sec._emit(e, 3)
    assert "    patch circ 7 16 4 0.0 0.0 0.0 300.0 0.0 360.0" in e.lines()
