"""Value-type behaviour for the scene_ir IR (ADR 0042, Part 1).

Covers the two guarantees the typed bundles exist to provide:
dtype/contiguity pinning (protects animation from per-frame casts) and
shape validation at emit (malformed diagrams fail loud).  Also pins
the equality contract — array-carrying types are identity-equal, never
element-wise (which would raise on the ambiguous truth value).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ColorSpec,
    GlyphLayer,
    LabelLayer,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    ScalarField,
    VisibilityMask,
)


# --- PointSet -------------------------------------------------------------


def test_pointset_pins_float32_contiguous() -> None:
    src = np.asfortranarray(np.zeros((4, 3), dtype=np.float64))
    ps = PointSet(src)
    assert ps.coords.dtype == np.float32
    assert ps.coords.flags["C_CONTIGUOUS"]
    assert ps.n_points == 4


def test_pointset_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        PointSet(np.zeros((4, 2)))


# --- CellBlocks -----------------------------------------------------------


def test_cellblocks_pins_int64_and_counts() -> None:
    cb = CellBlocks(
        {
            "triangle": np.zeros((2, 3), dtype=np.int32),
            "tetra": np.zeros((5, 4), dtype=np.int64),
        }
    )
    assert cb.blocks["triangle"].dtype == np.int64
    assert cb.n_cells == 7


def test_cellblocks_rejects_1d_block() -> None:
    with pytest.raises(ValueError, match="2-D"):
        CellBlocks({"line": np.arange(6)})


# --- ScalarField ----------------------------------------------------------


def test_scalarfield_requires_1d() -> None:
    ScalarField("stress", np.zeros(10), location="cell")  # ok
    with pytest.raises(ValueError, match="1-D"):
        ScalarField("disp", np.zeros((10, 3)), location="point")


# --- ColorSpec ------------------------------------------------------------


def test_colorspec_by_array_defaults_lut() -> None:
    cs = ColorSpec(mode="by_array", array_name="stress")
    assert isinstance(cs.lut, LutSpec)


def test_colorspec_by_array_requires_name() -> None:
    with pytest.raises(ValueError, match="array_name"):
        ColorSpec(mode="by_array")


def test_colorspec_per_entity_rgb_validated() -> None:
    cs = ColorSpec(mode="per_entity_rgb", entity_rgb=np.zeros((3, 3)))
    assert cs.entity_rgb.dtype == np.float32
    with pytest.raises(ValueError, match="entity_rgb"):
        ColorSpec(mode="per_entity_rgb")


# --- MeshLayer ------------------------------------------------------------


def _mesh() -> MeshLayer:
    return MeshLayer(
        layer_id="m",
        points=PointSet(np.zeros((3, 3))),
        cells=CellBlocks({"triangle": np.array([[0, 1, 2]])}),
        fields=(ScalarField("q", np.zeros(1), location="cell"),),
    )


def test_meshlayer_defaults_and_field_lookup() -> None:
    m = _mesh()
    assert m.color.mode == "solid"
    assert m.visibility.hidden_cells == frozenset()
    assert m.field_named("q") is not None
    assert m.field_named("absent") is None


def test_meshlayer_identity_equality_does_not_raise() -> None:
    # Auto __eq__ on numpy fields would raise on ambiguous truth value;
    # eq=False makes layers identity-equal instead.
    m = _mesh()
    assert m == m
    assert m != _mesh()


# --- GlyphLayer / LabelLayer ---------------------------------------------


def test_glyphlayer_orientations_validated() -> None:
    g = GlyphLayer(
        layer_id="loads",
        positions=PointSet(np.zeros((2, 3))),
        kind="arrow",
        orientations=np.ones((2, 3)),
    )
    assert g.orientations.dtype == np.float32
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        GlyphLayer("g", PointSet(np.zeros((2, 3))), orientations=np.ones((2, 2)))


def test_labellayer_text_count_must_match() -> None:
    LabelLayer("lbl", PointSet(np.zeros((2, 3))), texts=("a", "b"))  # ok
    with pytest.raises(ValueError, match="must match"):
        LabelLayer("lbl", PointSet(np.zeros((2, 3))), texts=("only-one",))


# --- ScalarBarSpec / VisibilityMask --------------------------------------


def test_scalarbar_and_mask_are_plain_values() -> None:
    bar = ScalarBarSpec(key="m", title="Stress", lut=LutSpec("jet", 0, 5))
    assert bar.lut.name == "jet"
    # The spec carries a finished layout — the backend computes nothing
    # (ADR 0081 Part 3).
    assert bar.anchor and bar.extent and bar.title_pt and bar.label_pt
    mask = VisibilityMask(hidden_cells=frozenset({1, 2}))
    assert 2 in mask.hidden_cells
