"""Tests — ADR 0080 B1: ``SectionDocument`` continuum lane.

Round-trip identity, the SRC twin-match against a hand-authored
session (same call order → deterministic mesh → analyzer numbers to
1e-9), polygon oracles, translate/rotate flow-through, the version
floor (ADR 0113, 2026-10-04 amendment), and the fail-loud validation surface.
"""
from __future__ import annotations

import json
import warnings

import pytest

import apeGmsh.sections._document as _docmod
from apeGmsh.sections import (
    SECTION_DOC_FLOOR,
    SECTION_DOC_VERSION,
    SectionDocument,
    SectionDocumentError,
    SectionDocumentNewerWarning,
    SectionMaterial,
    SectionProperties,
)


def _src_doc() -> SectionDocument:
    doc = SectionDocument.new(name="SRC600", units="N-mm")
    doc.set_material("concrete", E=25e3, nu=0.2)
    doc.set_material("steel", E=200e3, nu=0.3, fy=345.0)
    doc.add_shape("rect_face", id="concrete", b=600.0, h=600.0)
    doc.add_shape("W_face", id="steel", bf=250.0, tf=17.0, h=250.0, tw=10.0)
    doc.add_embed("concrete", "steel")
    doc.set_mesh(lc=60.0)
    return doc


# ─────────────────────────────────────────────────────────────────────
# persistence + round-trip
# ─────────────────────────────────────────────────────────────────────

def test_json_round_trip_identity(tmp_path):
    doc = _src_doc()
    p = tmp_path / "src600.section.json"
    doc.save(p)
    reopened = SectionDocument.open(p)
    assert reopened == doc
    assert reopened.to_dict()["section_doc_version"] == SECTION_DOC_VERSION
    # save→open→save is byte-stable
    p2 = tmp_path / "again.section.json"
    reopened.save(p2)
    assert p.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")


def _write_version(tmp_path, data, version):
    data = dict(data, section_doc_version=version)
    p = tmp_path / f"v{version}.section.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _open_silently(path):
    """Open and assert no newer-document warning fired."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SectionDocumentNewerWarning)
        return SectionDocument.open(path)


def test_floor_constant_is_writer_owned():
    """ADR 0113 INV-2 for section documents: the floor sits beside the
    version it bounds, same major, never above it; 1.0.0 at the
    2026-10-04 amendment (the version has never been bumped)."""
    floor = tuple(int(p) for p in SECTION_DOC_FLOOR.split("."))
    cur = tuple(int(p) for p in SECTION_DOC_VERSION.split("."))
    assert floor[0] == cur[0]
    assert floor <= cur
    assert SECTION_DOC_FLOOR == "1.0.0"


def test_version_floor_today(tmp_path):
    """The shipped constants (floor 1.0.0, loader 1.0.0): the floor and
    any patch open silently; another major refuses naming the floor; a
    newer same-major minor opens with one warning naming both versions."""
    base = _src_doc().to_dict()
    for version in (SECTION_DOC_FLOOR, SECTION_DOC_VERSION, "1.0.99"):
        doc = _open_silently(_write_version(tmp_path, base, version))
        assert doc.to_dict()["section_doc_version"] == version
    for version in ("0.9.0", "0.99.0", "2.0.0"):
        with pytest.raises(SectionDocumentError, match=r"floor 1\.0\.0"):
            SectionDocument.open(_write_version(tmp_path, base, version))
    with pytest.raises(SectionDocumentError, match="invalid section_doc_version"):
        SectionDocument.open(_write_version(tmp_path, base, "banana"))
    with pytest.warns(SectionDocumentNewerWarning) as rec:
        SectionDocument.open(_write_version(tmp_path, base, "1.1.0"))
    assert len(rec) == 1
    assert "1.1.0" in str(rec[0].message)
    assert SECTION_DOC_VERSION in str(rec[0].message)
    # attributed to the caller of open(), not to apeGmsh internals
    assert rec[0].filename == __file__
    # ... and to the caller of SectionDocument(...) on that path
    with pytest.warns(SectionDocumentNewerWarning) as rec:
        SectionDocument(dict(base, section_doc_version="1.1.0"))
    assert len(rec) == 1
    assert rec[0].filename == __file__


def test_version_floor_edges(tmp_path, monkeypatch):
    """Both edges with a floor above X.0 (floor 1.3, loader 1.5): the
    floor minor and every minor up to the loader open silently;
    floor.minor - 1 refuses naming the floor; loader.minor + 1 opens
    with the warning. Patches are ignored at every edge."""
    monkeypatch.setattr(_docmod, "SECTION_DOC_FLOOR", "1.3.0")
    monkeypatch.setattr(_docmod, "SECTION_DOC_VERSION", "1.5.0")
    base = _src_doc().to_dict()
    for version in ("1.3.0", "1.3.7", "1.4.0", "1.5.0", "1.5.2"):
        _open_silently(_write_version(tmp_path, base, version))
    for version in ("1.2.0", "1.2.9", "1.0.0", "0.5.0", "2.0.0"):
        with pytest.raises(SectionDocumentError, match=r"floor 1\.3\.0"):
            SectionDocument.open(_write_version(tmp_path, base, version))
    with pytest.warns(SectionDocumentNewerWarning, match="1.6.0"):
        SectionDocument.open(_write_version(tmp_path, base, "1.6.0"))


def test_newer_minor_unknown_keys(tmp_path):
    """What a newer same-major document carries (probed on the loader,
    2026-10-04): unknown optional keys at the top level, on a shape and
    in ``mesh`` are tolerated and kept verbatim, so a re-save does not
    lose them, and the build ignores them; a value this loader cannot
    interpret (a new shape kind, a new boolean op, a new material key,
    a new parameter on a known shape) still refuses. Both lanes."""
    data = _src_doc().to_dict()
    data["fillets"] = [{"shape": "steel", "r": 5.0}]
    data["shapes"][0]["chamfer"] = 2.0
    data["mesh"]["algorithm"] = 6
    p = _write_version(tmp_path, data, "1.1.0")
    with pytest.warns(SectionDocumentNewerWarning):
        doc = SectionDocument.open(p)
    assert doc.to_dict() == dict(data, section_doc_version="1.1.0")
    out = tmp_path / "resaved.section.json"
    doc.save(out)
    assert json.loads(out.read_text(encoding="utf-8")) == doc.to_dict()

    # the build ignores them: a 2 x 3 rectangle carrying a "fillet" it
    # cannot honour still integrates to the closed-form area and centroid
    rect = SectionDocument.new(name="r")
    rect.add_shape("rect_face", id="r", b=2.0, h=3.0)
    rect.set_mesh(lc=0.5)
    rdata = rect.to_dict()
    rdata["shapes"][0]["fillet"] = 0.4
    rdata["fillets"] = [{"shape": "r", "r": 0.4}]
    with pytest.warns(SectionDocumentNewerWarning):
        rdoc = SectionDocument.open(_write_version(tmp_path, rdata, "1.1.0"))
    geo = rdoc.build().geometric()
    assert geo.area == pytest.approx(6.0, rel=1e-9)
    assert geo.cx == pytest.approx(0.0, abs=1e-9)
    assert geo.cy == pytest.approx(0.0, abs=1e-9)

    fiber = SectionDocument.new(name="f", kind="fiber").to_dict()
    fiber["shear_springs"] = {"Vy": 1.0}
    with pytest.warns(SectionDocumentNewerWarning):
        fdoc = SectionDocument.open(_write_version(tmp_path, fiber, "1.1.0"))
    assert fdoc.to_dict()["shear_springs"] == {"Vy": 1.0}

    for mutate, match in (
        (lambda d: d["shapes"][0].__setitem__("shape", "hexagon_face"),
         "unknown shape kind"),
        (lambda d: d["booleans"].append({"op": "union", "a": "concrete",
                                         "b": "steel"}),
         "unknown boolean op"),
        (lambda d: d["materials"]["steel"].__setitem__("alpha", 1e-5),
         "unknown keys"),
        # a new parameter on a known shape kind: refused at load (as
        # add_shape refuses it), not a raw TypeError at build time
        (lambda d: d["shapes"][0]["params"].__setitem__("fillet", 5.0),
         r"unknown params \['fillet'\] \(section_doc_version 1\.1\.0"),
    ):
        bad = _src_doc().to_dict()
        mutate(bad)
        with pytest.warns(SectionDocumentNewerWarning):
            with pytest.raises(SectionDocumentError, match=match):
                SectionDocument.open(
                    _write_version(tmp_path, bad, "1.1.0")
                )


def test_lane_mismatch_keys_rejected(tmp_path):
    """A hand-edited doc claiming kind='fiber' but carrying continuum
    keys (and no fiber keys) fails loud on the missing fiber keys."""
    data = _src_doc().to_dict()
    data["kind"] = "fiber"
    p = tmp_path / "f.section.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SectionDocumentError, match="missing document key"):
        SectionDocument.open(p)


# ─────────────────────────────────────────────────────────────────────
# fail-loud validation surface
# ─────────────────────────────────────────────────────────────────────

def test_mutation_gates():
    doc = SectionDocument.new(name="gates")
    with pytest.raises(SectionDocumentError, match="unknown shape"):
        doc.add_shape("hexagon_face", id="x", b=1.0)
    with pytest.raises(SectionDocumentError, match="missing params"):
        doc.add_shape("rect_face", id="x", b=1.0)
    doc.add_shape("rect_face", id="x", b=1.0, h=2.0)
    with pytest.raises(SectionDocumentError, match="duplicate shape id"):
        doc.add_shape("rect_face", id="x", b=1.0, h=2.0)
    with pytest.raises(SectionDocumentError, match="unknown shape"):
        doc.add_embed("x", "ghost")
    with pytest.raises(SectionDocumentError, match="at least 3 points"):
        doc.add_polygon([(0, 0), (1, 0)], id="p")
    with pytest.raises(SectionDocumentError, match="order must be"):
        doc.set_mesh(lc=1.0, order=3)
    with pytest.raises(SectionDocumentError, match="disconnected"):
        doc.set_disconnected("maybe")  # type: ignore[arg-type]


def test_build_gates():
    doc = SectionDocument.new(name="gates")
    doc.add_shape("rect_face", id="bar", b=1.0, h=2.0)
    with pytest.raises(SectionDocumentError, match="set_mesh"):
        doc.build()
    doc.set_mesh(lc=0.5)
    # material named but table empty
    doc2 = SectionDocument.new(name="gates2")
    doc2.add_shape("rect_face", id="bar", material="steel", b=1.0, h=2.0)
    doc2.set_mesh(lc=0.5)
    with pytest.raises(SectionDocumentError, match="table is empty"):
        doc2.build()
    # material missing from a non-empty table
    doc3 = SectionDocument.new(name="gates3")
    doc3.set_material("concrete", E=25e3, nu=0.2)
    doc3.add_shape("rect_face", id="bar", material="steel", b=1.0, h=2.0)
    doc3.set_mesh(lc=0.5)
    with pytest.raises(SectionDocumentError, match="not .* the materials table"):
        doc3.build()


def test_hand_edited_bad_boolean_rejected(tmp_path):
    data = _src_doc().to_dict()
    data["booleans"].append({"op": "weld", "a": "concrete", "b": "steel"})
    p = tmp_path / "bad.section.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SectionDocumentError, match="unknown boolean op"):
        SectionDocument.open(p)


# ─────────────────────────────────────────────────────────────────────
# build oracles
# ─────────────────────────────────────────────────────────────────────

def test_src_twin_matches_hand_authored_session():
    """The document build reproduces the hand-authored SRC session
    (identical call order → deterministic mesh → same numbers)."""
    sec_doc = _src_doc().build()

    from apeGmsh import apeGmsh

    g = apeGmsh(model_name="SRC600", verbose=False)
    g.begin()
    try:
        conc = g.sections.rect_face(600.0, 600.0, label="concrete")
        steel = g.sections.W_face(
            bf=250.0, tf=17.0, h=250.0, tw=10.0, label="steel",
        )
        g.model.boolean.cut(
            conc.entities[2], steel.entities[2], dim=2, remove_tool=False,
        )
        g.parts.fragment_pair("concrete", "steel", dim=2)
        g.mesh.sizing.set_global_size(60.0)
        g.mesh.generation.generate(dim=2)
        g.mesh.generation.set_order(2)
        fem = g.mesh.queries.get_fem_data(dim=2)
    finally:
        g.end()
    sec_hand = SectionProperties(
        fem,
        materials={
            "concrete": SectionMaterial(E=25e3, nu=0.2),
            "steel": SectionMaterial(E=200e3, nu=0.3, fy=345.0),
        },
        name="SRC600",
    )

    gd, gh = sec_doc.geometric(), sec_hand.geometric()
    assert gd.EA == pytest.approx(gh.EA, rel=1e-9)
    assert gd.EIxx_c == pytest.approx(gh.EIxx_c, rel=1e-9)
    assert gd.EIyy_c == pytest.approx(gh.EIyy_c, rel=1e-9)
    wd, wh = sec_doc.warping(), sec_hand.warping()
    assert wd.GJ == pytest.approx(wh.GJ, rel=1e-9)


def test_polygon_oracles():
    """L-shaped polygon in geometric-only mode vs hand integrals."""
    doc = SectionDocument.new(name="Lpoly")
    doc.add_polygon(
        [(0, 0), (3, 0), (3, 1), (1, 1), (1, 4), (0, 4)], id="L",
    )
    doc.set_mesh(lc=0.15)
    sec = doc.build()
    geo = sec.geometric()
    # L = 3x1 leg + 1x3 leg above it
    area = 3.0 * 1.0 + 1.0 * 3.0
    assert geo.area == pytest.approx(area, rel=1e-9)
    cx = (3.0 * 1.5 + 3.0 * 0.5) / area
    cy = (3.0 * 0.5 + 3.0 * 2.5) / area
    assert geo.cx == pytest.approx(cx, rel=1e-9)
    assert geo.cy == pytest.approx(cy, rel=1e-9)


def test_shape_translate_rotate_flow_through():
    """rotate= on a document shape lands in the analyzer exactly like
    the direct builder call (nonzero EIxy on a rotated rectangle)."""
    doc = SectionDocument.new(name="rot")
    doc.add_shape("rect_face", id="bar", b=2.0, h=4.0, rotate=30.0)
    doc.set_mesh(lc=0.4)
    geo = doc.build().geometric()
    Ix, Iy = 2.0 * 4.0**3 / 12.0, 4.0 * 2.0**3 / 12.0
    import math

    two_t = math.radians(60.0)
    # shape rotated by +θ (axes fixed): Ixy' = (Iy − Ix)/2 · sin 2θ
    expected_ixy = (Iy - Ix) / 2.0 * math.sin(two_t)
    assert geo.EIxy_c == pytest.approx(expected_ixy, rel=1e-6)
    assert geo.area == pytest.approx(8.0, rel=1e-9)


def test_geometric_only_and_disconnected_policy():
    doc = SectionDocument.new(name="twin")
    doc.add_shape("rect_face", id="L", b=1.0, h=1.0, translate=(-2.0, 0.0))
    doc.add_shape("rect_face", id="R", b=1.0, h=1.0, translate=(+2.0, 0.0))
    doc.set_mesh(lc=0.2)
    doc.set_disconnected("sum")
    sec = doc.build()
    assert sec.geometric_only
    assert sec.n_parts == 2
    warp = sec.warping()          # "sum" flows through — no raise
    assert len(warp.parts) == 2


def test_polygon_hole_via_cut():
    """Raw cut with a sacrificial polygon tool punches a hole."""
    doc = SectionDocument.new(name="holed")
    doc.add_shape("rect_face", id="plate", b=4.0, h=4.0)
    doc.add_polygon(
        [(-1, -1), (1, -1), (1, 1), (-1, 1)], id="hole",
    )
    doc.add_cut("plate", "hole", remove_tool=True)
    doc.set_mesh(lc=0.25)
    geo = doc.build().geometric()
    assert geo.area == pytest.approx(16.0 - 4.0, rel=1e-9)
