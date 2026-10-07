"""DXF import: each curve lands in the physical group of its own layer (#1532).

``_DXFImporter._rebuild_layers`` matches the curves that survive
``removeAllDuplicates`` back to their DXF layers geometrically.  The old
key was a bounding box rounded to 8 decimals, and OCC pads
``gmsh.model.getBoundingBox`` by 1e-7, so every curve went to
``_unmatched`` silently.  The oracle here is the DXF the test writes:
each layer's expected endpoint set, read back from gmsh through
``getBoundary``/``getValue`` (exact coordinates, not the padded box).

Warn-as-contract: the clean two-layer import stays silent under
``-W error::WarnDxfLayerMismatch``; a lost entity and a curve duplicated
across layers each warn once.
"""
from __future__ import annotations

import math
import warnings

import gmsh
import pytest

ezdxf = pytest.importorskip("ezdxf")

BEAMS = "Beams"
COLUMNS = "Columns"


def _ends(tag: int) -> set[tuple[float, float, float]]:
    bnd = gmsh.model.getBoundary([(1, tag)], combined=False, oriented=False)
    return {
        tuple(round(float(c), 6) for c in gmsh.model.getValue(0, p, []))
        for _, p in bnd
    }


def _pt(x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
    return (round(x, 6), round(y, 6), round(z, 6))


def _write_two_layer_dxf(path):
    """Two lines and an arc on ``Beams``; a 3-vertex polyline (two
    segments) and a full circle on ``Columns``.  Returns the expected
    endpoint sets per layer, one set per curve."""
    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    doc.layers.add(COLUMNS)
    msp = doc.modelspace()
    msp.add_line((0, 0, 0), (4, 0, 0), dxfattribs={"layer": BEAMS})
    msp.add_line((0, 3, 0), (4, 3, 0), dxfattribs={"layer": BEAMS})
    # Arc from 30 deg to 200 deg: crosses the 90 and 180 deg extremes, so
    # its true box is wider than its chord's.
    msp.add_arc((10, 0, 0), 1.0, 30, 200, dxfattribs={"layer": BEAMS})
    # The column polyline meets both beams at (0, 0) and (0, 3): shared
    # nodes across layers, no shared curve.
    msp.add_lwpolyline([(0, 0), (0, 3), (0, 6)], dxfattribs={"layer": COLUMNS})
    msp.add_circle((20, 20, 0), 2.0, dxfattribs={"layer": COLUMNS})
    doc.saveas(str(path))

    a1, a2 = math.radians(30), math.radians(200)
    expected = {
        BEAMS: [
            {_pt(0, 0), _pt(4, 0)},
            {_pt(0, 3), _pt(4, 3)},
            {_pt(10 + math.cos(a1), math.sin(a1)), _pt(10 + math.cos(a2), math.sin(a2))},
        ],
        COLUMNS: [
            {_pt(0, 0), _pt(0, 3)},
            {_pt(0, 3), _pt(0, 6)},
            {_pt(22, 20)},  # a full circle's boundary is its seam vertex
        ],
    }
    return expected


def test_two_layer_dxf_puts_each_curve_in_its_layer_pg(g, tmp_path):
    path = tmp_path / "two_layers.dxf"
    expected = _write_two_layer_dxf(path)

    layers = g.model.io.load_dxf(path)

    assert set(layers) == {BEAMS, COLUMNS}, layers
    assert "_unmatched" not in layers
    for layer, curves in expected.items():
        tags = layers[layer][1]
        assert len(tags) == len(curves), (layer, tags)
        got = [_ends(t) for t in tags]
        for ends in curves:
            assert ends in got, (layer, ends, got)
        assert set(g.physical.entities(layer)) == set(tags), layer
    assert set(layers[BEAMS][1]).isdisjoint(layers[COLUMNS][1])


def test_two_layer_dxf_import_is_silent(g, tmp_path):
    from apeGmsh.core._model_io import WarnDxfLayerMismatch

    path = tmp_path / "two_layers.dxf"
    _write_two_layer_dxf(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnDxfLayerMismatch)
        g.model.io.load_dxf(path)


def test_scaled_up_drawing_matches_too(g, tmp_path):
    """A plan in mm at 1e5: OCC's absolute 1e-7 pad is still there and
    round-off grows, so the tolerance must scale."""
    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    doc.layers.add(COLUMNS)
    msp = doc.modelspace()
    msp.add_line((100000, 0, 0), (112345.678, 0, 0), dxfattribs={"layer": BEAMS})
    msp.add_arc((150000, 0, 0), 2500.0, 30, 200, dxfattribs={"layer": COLUMNS})
    path = tmp_path / "mm.dxf"
    doc.saveas(str(path))

    layers = g.model.io.load_dxf(path)

    assert "_unmatched" not in layers, layers
    assert len(layers[BEAMS][1]) == 1 and len(layers[COLUMNS][1]) == 1
    assert _ends(layers[BEAMS][1][0]) == {_pt(100000, 0), _pt(112345.678, 0)}


def test_duplicate_curve_within_one_layer_stays_silent(g, tmp_path):
    """Two identical lines on one layer: one curve survives, in that layer,
    and nothing warns (both records matched it)."""
    from apeGmsh.core._model_io import WarnDxfLayerMismatch

    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    msp = doc.modelspace()
    msp.add_line((0, 0, 0), (1, 1, 0), dxfattribs={"layer": BEAMS})
    msp.add_line((0, 0, 0), (1, 1, 0), dxfattribs={"layer": BEAMS})
    path = tmp_path / "dup.dxf"
    doc.saveas(str(path))

    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnDxfLayerMismatch)
        layers = g.model.io.load_dxf(path)
    assert set(layers) == {BEAMS}
    assert len(layers[BEAMS][1]) == 1


def test_curve_duplicated_across_layers_joins_both_and_warns(g, tmp_path):
    from apeGmsh.core._model_io import WarnDxfLayerMismatch

    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    doc.layers.add(COLUMNS)
    msp = doc.modelspace()
    msp.add_line((0, 0, 0), (1, 1, 0), dxfattribs={"layer": BEAMS})
    msp.add_line((0, 0, 0), (1, 1, 0), dxfattribs={"layer": COLUMNS})
    path = tmp_path / "cross.dxf"
    doc.saveas(str(path))

    with pytest.warns(WarnDxfLayerMismatch, match="several layers"):
        layers = g.model.io.load_dxf(path)
    assert layers[BEAMS][1] == layers[COLUMNS][1]
    assert len(layers[BEAMS][1]) == 1
    assert "_unmatched" not in layers


def test_entity_that_matches_no_curve_warns_and_names_the_layer(g, tmp_path):
    """A record no surviving curve matches warns, naming the layer that
    lost it.  gmsh refuses the natural producers (a zero-length line
    raises in ``addLine``), so the record is planted directly."""
    from apeGmsh.core._model_io import (
        WarnDxfLayerMismatch, _DXFImporter, _DxfCurveRecord,
    )

    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    doc.modelspace().add_line((0, 0, 0), (1, 0, 0), dxfattribs={"layer": BEAMS})
    path = tmp_path / "one.dxf"
    doc.saveas(str(path))
    g.model.io.load_dxf(path, create_physical_groups=False)

    importer = _DXFImporter(g.model, 1e-6)
    importer._records.append(_DxfCurveRecord(
        layer=BEAMS, ends=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        bbox=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0), bbox_exact=True,
    ))
    importer._records.append(_DxfCurveRecord(
        layer="Ghost", ends=((9.0, 9.0, 0.0), (9.0, 10.0, 0.0)),
        bbox=(9.0, 9.0, 0.0, 9.0, 10.0, 0.0), bbox_exact=True,
    ))
    with pytest.warns(WarnDxfLayerMismatch, match=r"\['Ghost'\] matched nothing"):
        layers = importer._rebuild_layers()
    assert set(layers) == {BEAMS}


def test_chord_and_arc_with_shared_endpoints_separate_by_box(g, tmp_path):
    """A line and an arc between the same two points: the endpoints tie,
    the exact bounding box decides."""
    doc = ezdxf.new()
    doc.layers.add(BEAMS)
    doc.layers.add(COLUMNS)
    msp = doc.modelspace()
    # Semicircle of radius 1 about (0, 0), from 0 to 180 deg: ends (1,0), (-1,0).
    msp.add_arc((0, 0, 0), 1.0, 0, 180, dxfattribs={"layer": BEAMS})
    msp.add_line((1, 0, 0), (-1, 0, 0), dxfattribs={"layer": COLUMNS})
    path = tmp_path / "chord.dxf"
    doc.saveas(str(path))

    layers = g.model.io.load_dxf(path)

    assert "_unmatched" not in layers, layers
    (arc_tag,) = layers[BEAMS][1]
    (line_tag,) = layers[COLUMNS][1]
    assert gmsh.model.getType(1, arc_tag) == "Circle"
    assert gmsh.model.getType(1, line_tag) == "Line"
