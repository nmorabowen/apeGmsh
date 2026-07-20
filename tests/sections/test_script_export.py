"""Tests — ADR 0080 B4: script export.

The contract: exports are deterministic, and EXECUTING an exported
script reproduces the document (the review lesson: execution tests,
not just golden text — count-based goldens don't test values).
"""
from __future__ import annotations

import pytest

from apeGmsh.sections import SectionDocument


def _src_doc() -> SectionDocument:
    doc = SectionDocument.new(name="SRC600", units="N-mm")
    doc.set_material("concrete", E=25e3, nu=0.2)
    doc.set_material("steel", E=200e3, nu=0.3, fy=345.0)
    doc.add_shape("rect_face", id="concrete", b=600.0, h=600.0)
    doc.add_shape("W_face", id="steel", bf=250.0, tf=17.0, h=250.0, tw=10.0)
    doc.add_embed("concrete", "steel")
    doc.set_mesh(lc=60.0)
    return doc


def _rc_fiber_doc() -> SectionDocument:
    doc = SectionDocument.new(name="col40x40", kind="fiber")
    doc.set_material("conf", uniaxial=("ElasticMaterial", {"E": 30e3}))
    doc.set_material("unconf", uniaxial=("ElasticMaterial", {"E": 20e3}))
    doc.set_material("steel", uniaxial=("ElasticMaterial", {"E": 200e3}))
    doc.add_template(
        "rc_rect_column",
        materials={"core": "conf", "cover": "unconf", "bars": "steel"},
        b=400.0, h=400.0, cover=50.0, bars_x=3, bars_y=3,
        bar_area=510.0, core_split=True,
    )
    doc.set_GJ(1.0e12)
    return doc


def test_export_deterministic(tmp_path):
    doc = _src_doc()
    a = doc.export_script()
    b = doc.export_script(tmp_path / "src.py")
    assert a == b
    assert (tmp_path / "src.py").read_text(encoding="utf-8") == a
    fa = _rc_fiber_doc().export_script()
    fb = _rc_fiber_doc().export_script()
    assert fa == fb


def test_continuum_export_executes_and_matches():
    """Executing the exported script reproduces the document build's
    analyzer numbers (same call order → deterministic mesh)."""
    doc = _src_doc()
    sec_doc = doc.build()

    script = doc.export_script()
    ns: dict = {}
    exec(compile(script, "<export>", "exec"), ns)
    sec_script = ns["sec"]

    gd, gs = sec_doc.geometric(), sec_script.geometric()
    assert gs.EA == pytest.approx(gd.EA, rel=1e-9)
    assert gs.EIxx_c == pytest.approx(gd.EIxx_c, rel=1e-9)
    assert gs.EIyy_c == pytest.approx(gd.EIyy_c, rel=1e-9)
    assert sec_script.warping().GJ == pytest.approx(
        sec_doc.warping().GJ, rel=1e-9,
    )


def test_continuum_export_polygon_and_cut_executes():
    doc = SectionDocument.new(name="holed")
    doc.add_shape("rect_face", id="plate", b=4.0, h=4.0)
    doc.add_polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], id="hole")
    doc.add_cut("plate", "hole", remove_tool=True)
    doc.set_mesh(lc=0.3)
    geo_doc = doc.build().geometric()

    ns: dict = {}
    exec(compile(doc.export_script(), "<export>", "exec"), ns)
    geo_script = ns["sec"].geometric()
    assert geo_script.area == pytest.approx(geo_doc.area, rel=1e-9)
    assert geo_script.area == pytest.approx(12.0, rel=1e-9)


def test_fiber_export_deck_equivalent(tmp_path):
    """build_section(ops) from the exported script produces a deck
    byte-identical to the document's own to_section handoff."""
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    doc = _rc_fiber_doc()

    def deck_via(factory):
        ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        sec = factory(ops)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        ops.element.forceBeamColumn(
            pg="Cols", transf=transf, integration=integ,
        )
        path = tmp_path / f"{factory.__name__}.tcl"
        ops.tcl(str(path))
        return path.read_text(encoding="utf-8")

    ns: dict = {}
    exec(compile(doc.export_script(), "<export>", "exec"), ns)
    build_section = ns["build_section"]

    def via_doc(ops):
        return doc.to_section(ops)

    assert deck_via(build_section) == deck_via(via_doc)


def test_fiber_export_provenance_comment():
    text = _rc_fiber_doc().export_script()
    assert "expanded from template rc_rect_column(" in text
    assert "core_split=True" in text
    assert "def build_section(ops" in text


def test_export_missing_uniaxial_fails_loud():
    doc = SectionDocument.new(name="nospec", kind="fiber")
    doc.set_material("m", E=1.0, nu=0.3)   # continuum role only
    doc.add_point(material="m", y=0.0, z=0.0, area=1.0)
    with pytest.raises(ValueError, match="no uniaxial spec"):
        doc.export_script()
