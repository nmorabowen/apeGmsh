"""Tests — ADR 0080 B7: the catalog picker and the handoff snippet.

Both are fail-soft optional-dependency surfaces, so each gets its
absence path as well as its happy path. The catalog gate is that a
prefill equals apeSteel's own geometry — including the ``h`` trap
(``W_face`` wants the **clear web height**, not the catalog depth
``d``). The handoff gate is that the fiber snippet's deck is
byte-identical to the document's own ``to_section`` handoff: a snippet
that drifts from the bridge is worse than no snippet.
"""
from __future__ import annotations

import sys

import pytest

from apeGmsh.sections import SectionDocument, handoff_snippet
from apeGmsh.sections._catalog import (
    CATALOG_SHAPE,
    catalog_available,
    catalog_labels,
    catalog_shape_params,
)


requires_apesteel = pytest.mark.skipif(
    not catalog_available(), reason="apeSteel not installed"
)


@pytest.fixture
def no_apesteel(monkeypatch):
    """Make ``import apeSteel`` raise, the way an uninstalled optional
    dependency does (a ``None`` in ``sys.modules`` is the documented
    block)."""
    monkeypatch.setitem(sys.modules, "apeSteel", None)


# ─────────────────────────────────────────────────────────────────────
# catalog picker
# ─────────────────────────────────────────────────────────────────────

@requires_apesteel
def test_prefill_matches_apesteel_geometry():
    import apeSteel

    geom = apeSteel.AISCv16Catalog().get_doubly_symmetric_i_geometry(
        "W14X90"
    )
    params = catalog_shape_params("W14X90")
    assert params == {
        "bf": pytest.approx(geom.flange_width_bf),
        "tf": pytest.approx(geom.flange_thickness_tf),
        "h": pytest.approx(geom.web_clear_height_hw),
        "tw": pytest.approx(geom.web_thickness_tw),
    }


@requires_apesteel
def test_h_is_the_clear_web_height_not_the_catalog_depth():
    """The one-line error this picker exists to prevent: typing the
    catalog ``d`` into ``W_face``'s ``h`` builds a section two flange
    thicknesses too deep."""
    import apeSteel

    row = apeSteel.AISCv16Catalog().get_row("W14X90")
    params = catalog_shape_params("W14X90")
    assert params["h"] == pytest.approx(float(row.d) - 2 * params["tf"])
    assert params["h"] != pytest.approx(float(row.d))


@requires_apesteel
def test_prefilled_params_build_a_document_shape():
    """The prefill is exactly ``W_face``'s parameter set — no more, no
    less — so it splats straight into ``add_shape``."""
    doc = SectionDocument.new(name="w14")
    doc.add_shape(CATALOG_SHAPE, id="w", **catalog_shape_params("W14X90"))
    stored = doc.to_dict()["shapes"][0]
    assert stored["shape"] == "W_face"
    assert set(stored["params"]) == {"bf", "tf", "h", "tw"}


@requires_apesteel
def test_labels_enumerate_both_families():
    labels = catalog_labels()
    assert "W14X90" in labels
    assert any(label.startswith("IPE") or label.startswith("HE")
               for label in labels)
    assert len(labels) == len(set(labels))       # deduped


@requires_apesteel
def test_unknown_designation_names_what_was_tried():
    with pytest.raises(ValueError, match="no apeSteel catalog resolves"):
        catalog_shape_params("W99X999")


def test_labels_degrade_to_empty_when_enumeration_breaks(monkeypatch):
    """Enumeration reads a private apeSteel table. If that shape ever
    changes the picker must go free-text, not explode."""
    import apeGmsh.sections._catalog as catalog

    class _Rotted:
        def __init__(self, *a, **k) -> None:
            pass

        def _dataframe(self):
            raise AttributeError("moved")

    fake = type("apeSteel", (), {
        "AISCv16Catalog": _Rotted, "EuropeanIPECatalog": _Rotted,
    })
    monkeypatch.setattr(catalog, "_apesteel", lambda: fake)
    assert catalog_labels() == ()


def test_catalog_absent_is_soft(no_apesteel):
    assert catalog_available() is False
    assert catalog_labels() == ()
    with pytest.raises(ImportError, match="pip install apeSteel"):
        catalog_shape_params("W14X90")


# ─────────────────────────────────────────────────────────────────────
# handoff snippet
# ─────────────────────────────────────────────────────────────────────

def _rc_fiber_doc() -> SectionDocument:
    doc = SectionDocument.new(name="col40x40", kind="fiber")
    doc.set_material("conc", uniaxial=("ElasticMaterial", {"E": 25e3}))
    doc.set_material("steel", uniaxial=("ElasticMaterial", {"E": 200e3}))
    doc.add_template(
        "rc_rect_column", materials={"concrete": "conc", "bars": "steel"},
        b=400.0, h=400.0, cover=50.0, bars_x=3, bars_y=3, bar_area=510.0,
    )
    doc.set_GJ(1.0e12)
    return doc


def _plate_doc(*, bars: bool = False) -> SectionDocument:
    doc = SectionDocument.new(name="plate")
    doc.set_material(
        "steel", E=200e3, nu=0.3,
        uniaxial=("ElasticMaterial", {"E": 200e3}),
    )
    doc.add_shape("rect_face", id="plate", material="steel", b=4.0, h=2.0)
    doc.set_mesh(lc=0.5)
    if bars:
        doc.add_bar(material="steel", x=1.0, y=0.5, area=0.01)
    return doc


@pytest.mark.parametrize(
    "doc", [_rc_fiber_doc(), _plate_doc(), _plate_doc(bars=True)],
    ids=["fiber", "continuum", "continuum-bars"],
)
def test_every_snippet_compiles(doc):
    compile(handoff_snippet(doc), "<handoff>", "exec")


def test_fiber_snippet_deck_equals_the_document_handoff(tmp_path):
    """The gate: exec the snippet on one bridge, call ``to_section`` on
    another, and the emitted decks must match byte for byte."""
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    doc = _rc_fiber_doc()

    def deck_via(factory, tag):
        ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        sec = factory(ops)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        integ = ops.beamIntegration.Lobatto(section=sec, n_ip=3)
        ops.element.forceBeamColumn(
            pg="Cols", transf=transf, integration=integ,
        )
        path = tmp_path / f"{tag}.tcl"
        ops.tcl(str(path))
        return path.read_text(encoding="utf-8")

    def via_snippet(ops):
        ns: dict = {"ops": ops}
        exec(compile(handoff_snippet(doc), "<handoff>", "exec"), ns)
        return ns["col40x40"]

    assert deck_via(via_snippet, "snippet") == deck_via(
        lambda ops: doc.to_section(ops), "doc",
    )


def test_fiber_snippet_carries_template_provenance():
    text = handoff_snippet(_rc_fiber_doc())
    assert "expanded from template rc_rect_column(" in text
    assert "ops.uniaxialMaterial.ElasticMaterial(" in text
    assert "ops.section.Fiber(" in text
    # a snippet, not a module: no def, no import of apeGmsh.sections
    assert "def build_section" not in text


def test_continuum_snippet_reopens_the_document_by_path():
    """Numbers are never hand-copied out of the GUI — the continuum
    handoff re-opens the document and lowers it late."""
    text = handoff_snippet(_plate_doc(), path="C:/models/plate.section.json")
    assert 'SectionDocument.open("C:/models/plate.section.json")' in text
    assert "ops.section.ComputedSection(" in text
    assert "analysis=doc.build()" in text


def test_continuum_snippet_without_a_path_uses_a_placeholder():
    text = handoff_snippet(_plate_doc())
    assert 'SectionDocument.open("plate.section.json")' in text


def test_continuum_with_bars_prefers_the_fiber_lowering():
    """A bars overlay only exists through ``to_section`` (ADR 0080 B3);
    the elastic ``ComputedSection`` would silently drop it, so the
    snippet leads with the right call and demotes the other to a
    comment."""
    text = handoff_snippet(_plate_doc(bars=True))
    assert "doc.to_section(ops" in text
    assert "1 bars entry" in text
    for line in text.splitlines():
        if "ComputedSection(" in line:
            assert line.lstrip().startswith("#")


def test_snippet_variable_is_a_valid_identifier():
    doc = SectionDocument.new(name="600 mm girder", kind="fiber")
    doc.set_material("m", uniaxial=("ElasticMaterial", {"E": 1.0}))
    doc.add_point(material="m", y=1.0, z=0.0, area=1.0)
    text = handoff_snippet(doc)
    compile(text, "<handoff>", "exec")
    assert "s_600_mm_girder = ops.section.Fiber(" in text
