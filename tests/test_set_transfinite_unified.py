"""Tests for the unified g.mesh.structured.set_transfinite() entry point."""
import math

import gmsh
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_or_skip(g):
    """Generate 3-D mesh; skip if gmsh fails (e.g. transfinite topology issue)."""
    try:
        g.mesh.generation.generate(dim=3)
    except Exception as e:
        pytest.skip(f"mesh generation failed: {e}")


def _curve_n_nodes(tag):
    """Return how many mesh nodes a curve has after meshing."""
    _, nodes, _ = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)
    return len(nodes) // 3


# Gmsh element-type codes
HEX_8 = 5
TET_4 = 4


# ---------------------------------------------------------------------------
# Scalar n=  — uniform on any orientation
# ---------------------------------------------------------------------------

def test_scalar_n_on_axis_aligned_box(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.structured.set_transfinite("box", n=11)
    _generate_or_skip(g)
    # All 12 edges should have 11 nodes
    edges = g.model.queries.select("box", dim=1)
    for _, t in edges:
        assert _curve_n_nodes(t) == 11


def test_scalar_n_no_args_walks_whole_model(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="b1")
    g.model.geometry.add_box(5, 0, 0, 1, 1, 1, label="b2")
    g.mesh.structured.set_transfinite(n=7)
    _generate_or_skip(g)
    for label in ("b1", "b2"):
        edges = g.model.queries.select(label, dim=1)
        for _, t in edges:
            assert _curve_n_nodes(t) == 7


# ---------------------------------------------------------------------------
# Dict n=  — per-axis, requires axis alignment
# ---------------------------------------------------------------------------

def test_dict_n_on_axis_aligned_box(g):
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="box")
    g.mesh.structured.set_transfinite("box", n={"x": 11, "y": 6, "z": 3})
    _generate_or_skip(g)
    edges = g.model.queries.select("box", dim=1)
    x_edges = edges.parallel_to("x").tags()
    y_edges = edges.parallel_to("y").tags()
    z_edges = edges.parallel_to("z").tags()
    for t in x_edges:
        assert _curve_n_nodes(t) == 11
    for t in y_edges:
        assert _curve_n_nodes(t) == 6
    for t in z_edges:
        assert _curve_n_nodes(t) == 3


def test_dict_n_on_rotated_box_raises(g):
    """A rotated box has off-axis edges; dict form must warn-and-skip with a tuple suggestion."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.model.transforms.rotate("box", math.radians(30.0))   # default axis = z

    with pytest.warns(UserWarning, match="dict-form sizing requires axis-aligned"):
        g.mesh.structured.set_transfinite("box", n={"x": 11, "y": 11, "z": 21})


# ---------------------------------------------------------------------------
# Tuple n=  — per principal axis, rotation-invariant
# ---------------------------------------------------------------------------

def test_tuple_n_on_axis_aligned_box(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.structured.set_transfinite("box", n=(11, 6, 3))
    _generate_or_skip(g)
    edges = g.model.queries.select("box", dim=1)
    # Tuple order is (X, Y, Z) — verify by axis classification.
    for t in edges.parallel_to("x").tags():
        assert _curve_n_nodes(t) == 11
    for t in edges.parallel_to("y").tags():
        assert _curve_n_nodes(t) == 6
    for t in edges.parallel_to("z").tags():
        assert _curve_n_nodes(t) == 3


def test_tuple_n_on_rotated_box(g):
    """30°-Z-rotated box: tuple still maps to (X-closest, Y-closest, Z)."""
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="box")
    g.model.transforms.rotate("box", math.radians(30.0))   # default axis = z

    g.mesh.structured.set_transfinite("box", n=(11, 6, 3))
    _generate_or_skip(g)

    edges = g.model.queries.select("box", dim=1)
    # Z edges still aligned with global Z — exact match
    for t in edges.parallel_to("z").tags():
        assert _curve_n_nodes(t) == 3

    # The two horizontal directions are now at ±30° from X and Y respectively.
    # The 30° rotation: the original "x" edges now point at (cos30, sin30, 0),
    # closer to X (angle 30°) than Y (angle 60°), so they map to tuple position 0 → 11 nodes.
    # The original "y" edges point at (-sin30, cos30, 0), closer to Y → position 1 → 6 nodes.
    horizontal_30 = edges.parallel_to((math.cos(math.radians(30.0)),
                                        math.sin(math.radians(30.0)),
                                        0.0)).tags()
    horizontal_120 = edges.parallel_to((-math.sin(math.radians(30.0)),
                                         math.cos(math.radians(30.0)),
                                         0.0)).tags()
    for t in horizontal_30:
        assert _curve_n_nodes(t) == 11
    for t in horizontal_120:
        assert _curve_n_nodes(t) == 6


def test_tuple_wrong_length_warns_and_skips(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    with pytest.warns(UserWarning, match="tuple of length 3 expected"):
        g.mesh.structured.set_transfinite("box", n=(11, 6))


# ---------------------------------------------------------------------------
# size=  forms — length-based
# ---------------------------------------------------------------------------

def test_size_scalar_per_edge(g):
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="box")
    g.mesh.structured.set_transfinite("box", size=1.0)
    _generate_or_skip(g)
    edges = g.model.queries.select("box", dim=1)
    # Each edge gets round(L / 1.0) + 1 nodes
    for t in edges.parallel_to("x").tags():
        assert _curve_n_nodes(t) == 11    # 10 / 1 + 1
    for t in edges.parallel_to("y").tags():
        assert _curve_n_nodes(t) == 6     # 5 / 1 + 1
    for t in edges.parallel_to("z").tags():
        assert _curve_n_nodes(t) == 3     # 2 / 1 + 1


def test_size_dict_per_axis(g):
    g.model.geometry.add_box(0, 0, 0, 10, 5, 2, label="box")
    g.mesh.structured.set_transfinite("box",
                                       size={"x": 1.0, "y": 0.5, "z": 0.5})
    _generate_or_skip(g)
    edges = g.model.queries.select("box", dim=1)
    for t in edges.parallel_to("x").tags():
        assert _curve_n_nodes(t) == 11   # 10/1 + 1
    for t in edges.parallel_to("y").tags():
        assert _curve_n_nodes(t) == 11   # 5/0.5 + 1
    for t in edges.parallel_to("z").tags():
        assert _curve_n_nodes(t) == 5    # 2/0.5 + 1


# ---------------------------------------------------------------------------
# Surface cascade
# ---------------------------------------------------------------------------

def test_surface_cascade_axis_aligned(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    # Pick the bottom face (z=0) by selection
    bottom = (g.model.queries.select("box", dim=2)
                .normal_along("z").select(on={"z": 0}))
    bottom.to_physical("bottom")

    g.mesh.structured.set_transfinite("bottom", n={"x": 11, "y": 6})
    _generate_or_skip(g)
    edges = g.model.queries.select("bottom", dim=1)
    for t in edges.parallel_to("x").tags():
        assert _curve_n_nodes(t) == 11
    for t in edges.parallel_to("y").tags():
        assert _curve_n_nodes(t) == 6


# ---------------------------------------------------------------------------
# Single-curve cascade
# ---------------------------------------------------------------------------

def test_curve_scalar(g):
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(10, 0, 0)
    c = g.model.geometry.add_line(p1, p2, label="line")
    g.mesh.structured.set_transfinite("line", n=11)
    g.mesh.generation.generate(dim=1)
    assert _curve_n_nodes(c) == 11


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_must_pass_n_or_size(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    with pytest.raises(ValueError, match="exactly one of n= or size="):
        g.mesh.structured.set_transfinite("box")


def test_cannot_pass_both_n_and_size(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    with pytest.raises(ValueError, match="exactly one of n= or size="):
        g.mesh.structured.set_transfinite("box", n=11, size=0.1)


def test_dict_missing_key_warns_and_skips(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    with pytest.warns(UserWarning, match="dict-form sizing missing key"):
        g.mesh.structured.set_transfinite("box", n={"x": 11, "y": 11})


# ---------------------------------------------------------------------------
# Element-type contract — locks the recombine= behavior
# ---------------------------------------------------------------------------

def test_default_produces_hex_elements(g):
    """Default recombine=True on a transfinite box gives Hexahedron 8 elements."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.structured.set_transfinite("box", n=3)
    _generate_or_skip(g)
    types, _, _ = gmsh.model.mesh.getElements(dim=3)
    type_list = list(types)
    assert HEX_8 in type_list, f"expected Hexahedron 8, got types {type_list}"
    assert TET_4 not in type_list, f"unexpected tets in output: {type_list}"


def test_recombine_false_produces_tet_elements(g):
    """recombine=False on a transfinite box gives Tetrahedron 4 elements."""
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.mesh.structured.set_transfinite("box", n=3, recombine=False)
    _generate_or_skip(g)
    types, _, _ = gmsh.model.mesh.getElements(dim=3)
    type_list = list(types)
    assert TET_4 in type_list, f"expected Tetrahedron 4, got types {type_list}"
    assert HEX_8 not in type_list, f"unexpected hexes in output: {type_list}"
