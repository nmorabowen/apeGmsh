"""Tests — ADR 0080 hardening from the adversarial review.

The fail-loud claim, enforced: hand-edited JSON either loads or raises
`SectionDocumentError` naming the problem — never a raw
KeyError/TypeError/ZeroDivisionError from `build()`/`to_section()`,
and never a crash inside the validator itself. Every rule the
mutation API enforces is enforced by the loader too (shared
checkers), and the three loader-optional keys are read tolerantly.
"""
from __future__ import annotations

import json
import math

import pytest

from apeGmsh.sections import (
    SECTION_DOC_VERSION,
    SectionDocument,
    SectionDocumentError,
)


def _write(tmp_path, data):
    p = tmp_path / "doc.section.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _cont_doc() -> dict:
    doc = SectionDocument.new(name="h")
    doc.add_shape("rect_face", id="bar", b=1.0, h=2.0)
    doc.set_mesh(lc=0.5)
    return doc.to_dict()


def _fib_doc() -> dict:
    doc = SectionDocument.new(name="h", kind="fiber")
    doc.set_material("m", uniaxial=("ElasticMaterial", {"E": 1.0}))
    doc.add_point(material="m", y=0.0, z=0.0, area=1.0)
    return doc.to_dict()


# ─────────────────────────────────────────────────────────────────────
# optional keys read tolerantly (were raw KeyError in build())
# ─────────────────────────────────────────────────────────────────────

def test_optional_keys_tolerated(tmp_path):
    d = _cont_doc()
    del d["disconnected"]
    del d["bars"]
    d["mesh"] = {"lc": 0.5}          # no "order" -> defaults to 2
    sec = SectionDocument.open(_write(tmp_path, d)).build()
    assert sec.geometric().area == pytest.approx(2.0, rel=1e-9)

    f = _fib_doc()
    del f["GJ"]
    recipe = SectionDocument.open(_write(tmp_path, f)).build()
    assert recipe.GJ is None


# ─────────────────────────────────────────────────────────────────────
# loader enforces the mutation API's value rules (were raw crashes)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bar", [
    {"kind": "line", "material": "m", "n": 1, "area": 1.0,
     "start": [0, 0], "end": [1, 0]},          # was ZeroDivisionError
    {"kind": "line", "material": "m", "n": 0, "area": 1.0,
     "start": [0, 0], "end": [1, 0]},          # was SILENT bar loss
    {"kind": "point", "material": "m", "x": 0, "y": 0, "area": -5},
    {"kind": "point", "material": "m", "x": "a", "y": 0, "area": 1},
    {"kind": "weld", "material": "m"},
])
def test_bar_junk_refused_at_load(tmp_path, bar):
    d = _cont_doc()
    d["bars"] = [bar]
    with pytest.raises(SectionDocumentError):
        SectionDocument.open(_write(tmp_path, d))


def test_template_junk_refused_at_load(tmp_path):
    base = SectionDocument.new(name="t", kind="fiber")
    base.set_material("c", uniaxial=("ElasticMaterial", {"E": 1.0}))
    base.set_material("s", uniaxial=("ElasticMaterial", {"E": 2.0}))
    base.add_template(
        "rc_rect_column", materials={"concrete": "c", "bars": "s"},
        b=400.0, h=400.0, cover=50.0, bars_x=2, bars_y=2, bar_area=1.0,
    )
    good = base.to_dict()
    for mutate in (
        lambda t: t["params"].__setitem__("cover", 250.0),   # ValueError
        lambda t: t["params"].__setitem__("width", 1.0),     # TypeError
        lambda t: t["params"].__setitem__("b", "wide"),      # TypeError
    ):
        d = json.loads(json.dumps(good))
        mutate(d["templates"][0])
        with pytest.raises(SectionDocumentError, match="template"):
            SectionDocument.open(_write(tmp_path, d))
    # the mutation path wraps identically now (was raw TypeError)
    with pytest.raises(SectionDocumentError, match="invalid params"):
        base.add_template(
            "rc_rect_column", materials={"concrete": "c", "bars": "s"},
            b=400.0, h=400.0, cover=50.0, bars_x=2, bars_y=2,
            bar_area=1.0, width=3.0,
        )


def test_continuum_material_without_e_named_error(tmp_path):
    d = _cont_doc()
    d["materials"] = {"steel": {"uniaxial": {
        "type": "ElasticMaterial", "params": {"E": 1.0}},
    }}
    d["shapes"][0]["material"] = "steel"
    doc = SectionDocument.open(_write(tmp_path, d))
    with pytest.raises(SectionDocumentError, match="no continuum role"):
        doc.build()


@pytest.mark.parametrize("key,value", [
    ("materials", 5),
    ("materials", ["a"]),
    ("materials", {"s": 5}),
    ("shapes", [5]),
    ("shapes", "abc"),
    ("shapes", None),
    ("booleans", [5]),
    ("bars", [5]),
])
def test_container_junk_refused_continuum(tmp_path, key, value):
    d = _cont_doc()
    d[key] = value
    with pytest.raises(SectionDocumentError):
        SectionDocument.open(_write(tmp_path, d))


@pytest.mark.parametrize("key,value", [
    ("patches", [5]),
    ("templates", [5]),
    ("points", [{"material": "m", "y": 0, "z": 0, "area": "big"}]),
    ("layers", [{"kind": "straight", "material": "m", "n_bars": 0,
                 "area": 1.0, "yI": 0, "zI": 0, "yJ": 1, "zJ": 0}]),
    ("patches", [{"kind": "circ", "material": "m", "n_circ": 8,
                  "n_rad": 2, "yC": 0, "zC": 0, "int_rad": 2.0,
                  "ext_rad": 1.0}]),
    ("GJ", True),
])
def test_container_junk_refused_fiber(tmp_path, key, value):
    f = _fib_doc()
    f[key] = value
    with pytest.raises(SectionDocumentError):
        SectionDocument.open(_write(tmp_path, f))


def test_geometry_value_junk_refused_at_load(tmp_path):
    d = _cont_doc()
    d["shapes"][0]["params"]["b"] = "wide"
    with pytest.raises(SectionDocumentError, match="must be a number"):
        SectionDocument.open(_write(tmp_path, d))
    d2 = _cont_doc()
    d2["shapes"].append({
        "id": "p", "shape": "polygon",
        "points": [["a", "b"], [1, 0], [1, 1]],
        "material": None, "translate": [0, 0], "rotate": None,
    })
    with pytest.raises(SectionDocumentError, match="must be a number"):
        SectionDocument.open(_write(tmp_path, d2))
    d3 = _cont_doc()
    d3["shapes"].append({
        "id": "p", "shape": "polygon", "points": "abc",
        "material": None, "translate": [0, 0], "rotate": None,
    })
    with pytest.raises(SectionDocumentError, match="polygon"):
        SectionDocument.open(_write(tmp_path, d3))
    d4 = _cont_doc()
    d4["mesh"] = {"lc": 0.5, "order": 7}
    with pytest.raises(SectionDocumentError, match="order"):
        SectionDocument.open(_write(tmp_path, d4))


# ─────────────────────────────────────────────────────────────────────
# version string edges (the "1.-1.0" hole) + misc
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("version", [
    "1.-1.0", "01.0.0", "1. 0.0", "1.0.-7",
])
def test_version_noncanonical_refused(tmp_path, version):
    d = _cont_doc()
    d["section_doc_version"] = version
    with pytest.raises(SectionDocumentError, match="version"):
        SectionDocument.open(_write(tmp_path, d))


def test_repr_works_on_both_lanes():
    assert "fiber" in repr(SectionDocument.new(kind="fiber"))
    assert "continuum" in repr(SectionDocument.new())


def test_self_boolean_refused(tmp_path):
    doc = SectionDocument.new(name="s")
    doc.add_shape("rect_face", id="a", b=1.0, h=1.0)
    for meth in (doc.add_embed, doc.add_cut, doc.add_fragment_pair):
        with pytest.raises(SectionDocumentError, match="different shapes"):
            meth("a", "a")
    d = _cont_doc()
    d["booleans"] = [{"op": "cut", "target": "bar", "tool": "bar",
                      "remove_tool": True}]
    with pytest.raises(SectionDocumentError, match="different shapes"):
        SectionDocument.open(_write(tmp_path, d))


def test_nonfinite_refused():
    doc = SectionDocument.new(name="n")
    with pytest.raises(SectionDocumentError, match="finite"):
        doc.add_shape("rect_face", id="a", b=math.nan, h=1.0)
    with pytest.raises(SectionDocumentError, match="must be a number"):
        doc.set_material("m", E=True, nu=0.3)


def test_mutation_value_gates_match_loader():
    fib = SectionDocument.new(kind="fiber")
    with pytest.raises(SectionDocumentError, match=">= 1"):
        fib.add_layer_straight(material="m", n_bars=0, area=1.0,
                               yI=0, zI=0, yJ=1, zJ=0)
    with pytest.raises(SectionDocumentError, match="> 0"):
        fib.add_point(material="m", y=0, z=0, area=-3.0)
    with pytest.raises(SectionDocumentError, match="int_rad"):
        fib.add_patch_circ(material="m", n_circ=8, n_rad=2,
                           int_rad=2.0, ext_rad=1.0)


# ─────────────────────────────────────────────────────────────────────
# uniaxial params refuse at to_section, never a raw TypeError (#1354)
# ─────────────────────────────────────────────────────────────────────

def _bridge():
    from typing import cast

    from apeGmsh.opensees import apeSees

    from tests.opensees.fixtures.fem_stub import make_two_node_beam

    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    return ops


_REFUSE_CASES = [
    # a key ElasticMaterial lacks: the bind text must name it (the
    # sorted params list alone is not enough)
    ({"E": 1.0, "fillet": 2.0}, ("unexpected keyword argument 'fillet'",)),
    # its required key missing: the given keys are not blamed. The bind
    # text is asserted in two pieces because 3.12+ says "missing a
    # required keyword-only argument: 'E'" and 3.10/3.11 drop
    # "keyword-only".
    ({}, ("missing a required", "'E'")),
]


@pytest.mark.parametrize("params,bind_texts", _REFUSE_CASES)
def test_uniaxial_params_refuse_at_to_section(tmp_path, params, bind_texts):
    """The loader keeps no bridge signatures, so the document opens;
    ``to_section`` binds the params against the real constructor and
    refuses with the material, the key and the version named."""
    f = _fib_doc()
    f["materials"]["m"]["uniaxial"]["params"] = params
    doc = SectionDocument.open(_write(tmp_path, f))
    with pytest.raises(SectionDocumentError) as ei:
        doc.to_section(_bridge())
    msg = str(ei.value)
    assert "material 'm'" in msg
    assert f"ElasticMaterial() rejects params {sorted(params)}" in msg
    assert all(t in msg for t in bind_texts), msg
    assert "does not take" not in msg
    assert f"section_doc_version {SECTION_DOC_VERSION}" in msg
    # the mutation path lands on the same splat
    doc2 = SectionDocument.new(name="h", kind="fiber")
    doc2.set_material("m", uniaxial=("ElasticMaterial", params))
    doc2.add_point(material="m", y=0.0, z=0.0, area=1.0)
    with pytest.raises(SectionDocumentError) as ei2:
        doc2.to_section(_bridge())
    assert all(t in str(ei2.value) for t in bind_texts), str(ei2.value)


@pytest.mark.parametrize("params,bind_texts", _REFUSE_CASES)
def test_uniaxial_params_refuse_continuum_lane(tmp_path, params, bind_texts):
    """The continuum lane (``_to_computed_fiber``) resolves region and
    bar materials through the same helper: a bad bar-material spec
    refuses after the mesh builds, naming the same three things."""
    doc = SectionDocument.new(name="rc", kind="continuum")
    doc.set_material(
        "conc", E=25e3, nu=0.2, uniaxial=("ElasticMaterial", {"E": 25e3}),
    )
    doc.set_material("steel", uniaxial=("ElasticMaterial", params))
    doc.add_shape("rect_face", id="conc", b=1.0, h=1.0)
    doc.add_bar(material="steel", x=0.0, y=0.0, area=0.01)
    doc.set_mesh(lc=0.5)
    doc = SectionDocument.open(_write(tmp_path, doc.to_dict()))
    with pytest.raises(SectionDocumentError) as ei:
        doc.to_section(_bridge())
    msg = str(ei.value)
    assert "material 'steel'" in msg
    assert f"ElasticMaterial() rejects params {sorted(params)}" in msg
    assert all(t in msg for t in bind_texts), msg
    assert f"section_doc_version {SECTION_DOC_VERSION}" in msg


def test_uniaxial_constructor_typeerror_not_swallowed(tmp_path):
    """Only the keyword-splat failure is translated. A ``TypeError``
    the constructor raises for its own reasons, with the keys accepted,
    propagates untouched (a stub bridge pins the two cases apart)."""
    class _Boom:
        def __init__(self, *, E: float) -> None:
            raise TypeError("inside the constructor")

    class _NS:
        ElasticMaterial = _Boom

    class _Ops:
        uniaxialMaterial = _NS()

    doc = SectionDocument.open(_write(tmp_path, _fib_doc()))
    with pytest.raises(TypeError, match="inside the constructor"):
        doc.to_section(_Ops())
