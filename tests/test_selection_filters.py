"""Tests for Selection composite and filter engine."""
import math

import gmsh
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_boxes(g):
    """Create two non-overlapping boxes for filter testing."""
    t1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="left")
    t2 = g.model.geometry.add_box(5, 0, 0, 1, 1, 1, label="right")
    return t1, t2


def _make_wireframe(g):
    """Create a simple wireframe for curve-level filter tests."""
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(10, 0, 0)  # horizontal
    p3 = g.model.geometry.add_point(0, 0, 5)   # vertical
    p4 = g.model.geometry.add_point(5, 5, 0)   # diagonal
    c_h = g.model.geometry.add_line(p1, p2)     # horizontal
    c_v = g.model.geometry.add_line(p1, p3)     # vertical
    c_d = g.model.geometry.add_line(p1, p4)     # diagonal
    return c_h, c_v, c_d


# ---------------------------------------------------------------------------
# Volume selection
# ---------------------------------------------------------------------------

def test_select_volumes_all(g):
    """select_volumes() returns all dim=3 entities."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes()
    assert len(sel) == 2


def test_select_volumes_by_tags(g):
    """tags= filter keeps only specified tags."""
    t1, t2 = _make_two_boxes(g)
    sel = g.model.selection.select_volumes(tags=[t1])
    assert len(sel) == 1
    assert sel.tags[0] == t1


def test_select_volumes_exclude_tags(g):
    """exclude_tags= removes specified tags."""
    t1, t2 = _make_two_boxes(g)
    sel = g.model.selection.select_volumes(exclude_tags=[t1])
    assert len(sel) == 1
    assert sel.tags[0] == t2


def test_select_volumes_by_labels(g):
    """labels= glob matches labeled entities."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes(labels="left")
    assert len(sel) == 1

    sel_glob = g.model.selection.select_volumes(labels="*ight")
    assert len(sel_glob) == 1


def test_select_volumes_by_kinds(g):
    """kinds= matches entity kind from metadata."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes(kinds="box")
    assert len(sel) == 2


def test_select_volumes_in_box(g):
    """in_box= spatial filter keeps only entities within bounding box."""
    _make_two_boxes(g)
    # Box around only the left box
    sel = g.model.selection.select_volumes(
        in_box=(-0.5, -0.5, -0.5, 1.5, 1.5, 1.5),
    )
    assert len(sel) == 1


def test_select_volumes_in_sphere(g):
    """in_sphere= keeps entities whose centroid is within radius."""
    _make_two_boxes(g)
    # Sphere around left box centroid (0.5, 0.5, 0.5)
    sel = g.model.selection.select_volumes(
        in_sphere=(0.5, 0.5, 0.5, 1.0),
    )
    assert len(sel) == 1


# ---------------------------------------------------------------------------
# Curve selection
# ---------------------------------------------------------------------------

def test_select_curves_all(g):
    """select_curves() returns all dim=1 entities."""
    _make_wireframe(g)
    sel = g.model.selection.select_curves()
    assert len(sel) >= 3


def test_select_curves_horizontal(g):
    """horizontal=True selects curves perpendicular to Z axis."""
    c_h, c_v, c_d = _make_wireframe(g)
    sel = g.model.selection.select_curves(horizontal=True)
    tags = sel.tags
    assert c_h in tags
    assert c_v not in tags


def test_select_curves_vertical(g):
    """vertical=True selects curves parallel to Z axis."""
    c_h, c_v, c_d = _make_wireframe(g)
    sel = g.model.selection.select_curves(vertical=True)
    tags = sel.tags
    assert c_v in tags
    assert c_h not in tags


# ---------------------------------------------------------------------------
# Point selection
# ---------------------------------------------------------------------------

def test_select_points_on_plane(g):
    """on_plane= filter intersects entity bounding box with a plane."""
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(0, 0, 5)
    g.model.sync()
    sel = g.model.selection.select_points(on_plane=("z", 0.0, 0.1))
    assert (0, p1) in sel
    assert (0, p2) not in sel


def test_select_points_at_point(g):
    """at_point= proximity filter."""
    p1 = g.model.geometry.add_point(1, 2, 3)
    p2 = g.model.geometry.add_point(10, 10, 10)
    g.model.sync()
    sel = g.model.selection.select_points(at_point=(1, 2, 3, 0.5))
    assert (0, p1) in sel
    assert (0, p2) not in sel


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def test_selection_union(g):
    """Union of two selections combines both."""
    t1, t2 = _make_two_boxes(g)
    sel_a = g.model.selection.select_volumes(tags=[t1])
    sel_b = g.model.selection.select_volumes(tags=[t2])
    combined = sel_a | sel_b
    assert len(combined) == 2


def test_selection_intersection(g):
    """Intersection of overlapping selections."""
    t1, t2 = _make_two_boxes(g)
    sel_all = g.model.selection.select_volumes()
    sel_one = g.model.selection.select_volumes(tags=[t1])
    both = sel_all & sel_one
    assert len(both) == 1


def test_selection_difference(g):
    """Difference removes entries in the second selection."""
    t1, t2 = _make_two_boxes(g)
    sel_all = g.model.selection.select_volumes()
    sel_one = g.model.selection.select_volumes(tags=[t1])
    diff = sel_all - sel_one
    assert len(diff) == 1
    assert diff.tags[0] == t2


def test_selection_to_physical(g):
    """to_physical() creates a solver-facing physical group."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes(labels="left")
    sel.to_physical("my_pg")
    # Verify PG exists via raw gmsh
    found = False
    for dim, pg_tag in gmsh.model.getPhysicalGroups(3):
        name = gmsh.model.getPhysicalName(dim, pg_tag)
        if name == "my_pg":
            found = True
            break
    assert found


# ---------------------------------------------------------------------------
# Explicit filter signature — typed kwargs (as of v1.0.3)
# ---------------------------------------------------------------------------

# Every filter name currently accepted by _apply_filters. If this list and
# the method signatures diverge, the smoke test below will fail — and so
# will ``pyright`` in CI, because the methods now use named parameters
# instead of ``**kwargs``.
_ALL_FILTER_KWARGS = dict(
    tags=[],
    exclude_tags=[],
    labels=None,
    kinds=None,
    physical=None,
    in_box=(-100.0, -100.0, -100.0, 100.0, 100.0, 100.0),
    in_sphere=(0.0, 0.0, 0.0, 1000.0),
    on_plane=("z", 0.0, 1.0),
    on_axis=("x", 100.0),
    at_point=(0.0, 0.0, 0.0, 100.0),
    length_range=(0.0, 1e6),
    area_range=(0.0, 1e6),
    volume_range=(0.0, 1e6),
    aligned=("x", 5.0),
    horizontal=False,
    vertical=False,
    predicate=None,
)


@pytest.mark.parametrize(
    "method_name",
    ["select_points", "select_curves", "select_surfaces",
     "select_volumes", "select_all"],
)
def test_select_methods_accept_every_filter(g, method_name):
    """Every filter keyword is a declared parameter on every select_* method.

    Also guards against accidental regression to ``**kwargs`` — a pyright
    pass would fail silently on that change, but this smoke test would
    still run, and a future broken call site would show up here.
    """
    _make_two_boxes(g)
    method = getattr(g.model.selection, method_name)
    # Passing every filter name as a kwarg must not raise TypeError.
    method(**_ALL_FILTER_KWARGS)


def test_selection_filter_accepts_every_filter(g):
    """Selection.filter accepts every filter name as a kwarg."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes()
    sel.filter(**_ALL_FILTER_KWARGS)


def test_unknown_kwarg_raises_on_select(g):
    """Unknown kwargs on select_* now raise TypeError (was silent with **kwargs)."""
    _make_two_boxes(g)
    with pytest.raises(TypeError):
        g.model.selection.select_volumes(not_a_filter=True)  # type: ignore[call-arg]


def test_unknown_kwarg_raises_on_filter(g):
    """Unknown kwargs on Selection.filter now raise TypeError."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes()
    with pytest.raises(TypeError):
        sel.filter(not_a_filter=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Behavioural regression — one filter per family on each method
# ---------------------------------------------------------------------------

def test_select_surfaces_identity_spatial_metrics(g):
    """select_surfaces exercises identity, spatial, and metric filters."""
    t1, _ = _make_two_boxes(g)
    # Identity: tags on volumes -> get surfaces of just the left box via in_box
    sel_id = g.model.selection.select_surfaces(
        in_box=(-0.1, -0.1, -0.1, 1.1, 1.1, 1.1),
    )
    assert len(sel_id) == 6  # faces of the left box
    # Spatial: faces whose bbox intersects the plane z=0.
    # Bottom + 4 sides per box = 5 faces each -> 10 total.
    sel_face = g.model.selection.select_surfaces(
        on_plane=("z", 0.0, 0.01),
    )
    assert len(sel_face) == 10
    # Metric: area_range (each unit face has area 1.0)
    sel_area = g.model.selection.select_surfaces(
        area_range=(0.9, 1.1),
    )
    assert len(sel_area) == 12  # 6 faces * 2 boxes


def test_select_volumes_volume_range(g):
    """volume_range filter keeps volumes with mass in range."""
    _make_two_boxes(g)
    sel = g.model.selection.select_volumes(volume_range=(0.5, 1.5))
    assert len(sel) == 2
    sel_empty = g.model.selection.select_volumes(volume_range=(10.0, 20.0))
    assert len(sel_empty) == 0


def test_select_curves_length_range(g):
    """length_range filter keeps curves with length in range."""
    c_h, c_v, c_d = _make_wireframe(g)
    g.model.sync()
    # c_h is 10 long; c_v is 5 long
    sel = g.model.selection.select_curves(length_range=(4.0, 6.0))
    assert c_v in sel.tags
    assert c_h not in sel.tags


def test_select_curves_aligned(g):
    """aligned filter keeps curves within angular tolerance of an axis."""
    c_h, c_v, c_d = _make_wireframe(g)
    g.model.sync()
    sel = g.model.selection.select_curves(aligned=("x", 5.0))
    assert c_h in sel.tags
    assert c_v not in sel.tags


def test_select_points_on_axis(g):
    """on_axis filter keeps entities whose centroid lies on a coord axis."""
    p_on = g.model.geometry.add_point(5, 0, 0)
    p_off = g.model.geometry.add_point(5, 5, 0)
    g.model.sync()
    sel = g.model.selection.select_points(on_axis=("x", 0.01))
    assert (0, p_on) in sel
    assert (0, p_off) not in sel


def test_select_volumes_predicate(g):
    """predicate escape hatch accepts a (dim, tag) -> bool callable."""
    t1, t2 = _make_two_boxes(g)
    sel = g.model.selection.select_volumes(predicate=lambda d, t: t == t1)
    assert sel.tags == (t1,)


def test_select_volumes_physical(g):
    """physical filter keeps entities in a named physical group."""
    t1, _ = _make_two_boxes(g)
    g.model.selection.select_volumes(tags=[t1]).to_physical("left_vol")
    sel = g.model.selection.select_volumes(physical="left_vol")
    assert sel.tags == (t1,)


def test_select_all_with_dim_and_without(g):
    """select_all(dim=d) matches select_volumes(); dim=-1 spans all dims."""
    _make_two_boxes(g)
    sel_vol = g.model.selection.select_all(dim=3)
    assert len(sel_vol) == 2
    # dim=-1 collects points, curves, surfaces, volumes
    sel_any = g.model.selection.select_all()
    assert sel_any.dim == -1
    assert len(sel_any) > 2


def test_selection_filter_refines(g):
    """Selection.filter narrows an existing selection."""
    t1, t2 = _make_two_boxes(g)
    all_vols = g.model.selection.select_volumes()
    refined = all_vols.filter(tags=[t1])
    assert refined.tags == (t1,)
