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
