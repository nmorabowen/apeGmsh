"""
Tests -- Geometry primitive creation via g.model.geometry.

Each test uses the ``g`` fixture (full apeGmsh session) and verifies:
  - Tag is a positive integer.
  - Entity exists at the expected dimension in the Gmsh model.
  - Metadata is registered with the correct ``kind`` key.
  - Bounding-box sanity for solid primitives.
  - Label PGs are created when ``label=`` is provided.
"""
from __future__ import annotations

import math

import gmsh
import pytest


# =====================================================================
# Helpers
# =====================================================================

def _entity_count(dim: int) -> int:
    """Return how many entities exist at *dim* in the current model."""
    return len(gmsh.model.getEntities(dim))


def _bb(dim: int, tag: int):
    """Return (xmin, ymin, zmin, xmax, ymax, zmax) bounding box."""
    return gmsh.model.getBoundingBox(dim, tag)


# =====================================================================
# Points  (dim = 0)
# =====================================================================

def test_add_point_basic(g):
    n0 = _entity_count(0)
    tag = g.model.geometry.add_point(1.0, 2.0, 3.0)
    assert isinstance(tag, int) and tag > 0
    assert _entity_count(0) > n0


def test_add_point_metadata(g):
    tag = g.model.geometry.add_point(0, 0, 0)
    meta = g.model._metadata[(0, tag)]
    assert meta['kind'] == 'point'


def test_add_point_label(g):
    tag = g.model.geometry.add_point(5.0, 5.0, 0.0, label="corner_pt")
    assert g.labels.has("corner_pt")
    ents = g.labels.entities("corner_pt")
    assert tag in ents


# =====================================================================
# Lines  (dim = 1)
# =====================================================================

def test_add_line_basic(g):
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(1, 0, 0)
    n1 = _entity_count(1)
    tag = g.model.geometry.add_line(p1, p2)
    assert tag > 0
    assert _entity_count(1) > n1
    assert g.model._metadata[(1, tag)]['kind'] == 'line'


def test_add_line_label(g):
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(3, 0, 0)
    tag = g.model.geometry.add_line(p1, p2, label="beam_edge")
    assert g.labels.has("beam_edge")
    assert tag in g.labels.entities("beam_edge")


# =====================================================================
# Arc  (dim = 1)
# =====================================================================

def test_add_arc(g):
    r = 1.0
    start  = g.model.geometry.add_point(r, 0, 0)
    center = g.model.geometry.add_point(0, 0, 0)
    end    = g.model.geometry.add_point(0, r, 0)
    n1 = _entity_count(1)
    tag = g.model.geometry.add_arc(start, center, end)
    assert tag > 0
    assert _entity_count(1) > n1
    assert g.model._metadata[(1, tag)]['kind'] == 'arc'


def test_add_arc_through_point_vs_center(g):
    """through_point=True fits the circle *through* the middle point
    (arch apex up); the default treats it as the circle centre (sag)."""
    gm = g.model.geometry
    L, H1, H2 = 10.0, 4.0, 2.0          # apex at z = H1 + H2 = 6
    gm.add_point(0,   0, H1,      label="sl")   # springline left
    gm.add_point(L/2, 0, H1 + H2, label="apex")
    gm.add_point(L,   0, H1,      label="sr")   # springline right

    # through_point=True: apex lies ON the arc -> bbox reaches z≈6
    arch = gm.add_arc("sl", "apex", "sr", through_point=True)
    _, _, z0, _, _, z1 = _bb(1, arch)
    assert z1 == pytest.approx(H1 + H2, abs=1e-6)   # apex up
    assert z0 == pytest.approx(H1,      abs=1e-6)

    # default (centre-of-circle): apex is the centre -> arc sags below
    sag = gm.add_arc("sl", "apex", "sr")
    _, _, sz0, _, _, sz1 = _bb(1, sag)
    assert sz1 == pytest.approx(H1, abs=1e-6)        # never rises above
    assert sz0 < H1                                  # dips down


# =====================================================================
# Circle  (dim = 1)
# =====================================================================

def test_add_circle_full(g):
    n1 = _entity_count(1)
    tag = g.model.geometry.add_circle(0, 0, 0, 5.0)
    assert tag > 0
    assert _entity_count(1) > n1
    assert g.model._metadata[(1, tag)]['kind'] == 'circle'


def test_add_circle_label(g):
    tag = g.model.geometry.add_circle(0, 0, 0, 2.0, label="hole_edge")
    assert g.labels.has("hole_edge")
    assert tag in g.labels.entities("hole_edge")


# =====================================================================
# Spline  (dim = 1)
# =====================================================================

def test_add_spline(g):
    pts = [
        g.model.geometry.add_point(i, math.sin(i), 0)
        for i in range(4)
    ]
    tag = g.model.geometry.add_spline(pts)
    assert tag > 0
    assert g.model._metadata[(1, tag)]['kind'] == 'spline'


def test_add_spline_too_few_points(g):
    p = g.model.geometry.add_point(0, 0, 0)
    with pytest.raises(ValueError, match="at least 2"):
        g.model.geometry.add_spline([p])


# =====================================================================
# B-Spline  (dim = 1)
# =====================================================================

def test_add_bspline(g):
    pts = [
        g.model.geometry.add_point(float(i), float(i % 2), 0)
        for i in range(4)
    ]
    tag = g.model.geometry.add_bspline(pts)
    assert tag > 0
    assert g.model._metadata[(1, tag)]['kind'] == 'bspline'


# =====================================================================
# Bezier  (dim = 1)
# =====================================================================

def test_add_bezier(g):
    pts = [
        g.model.geometry.add_point(float(i), 0, 0)
        for i in range(3)
    ]
    tag = g.model.geometry.add_bezier(pts)
    assert tag > 0
    assert g.model._metadata[(1, tag)]['kind'] == 'bezier'


def test_add_bezier_too_few_points(g):
    p = g.model.geometry.add_point(0, 0, 0)
    with pytest.raises(ValueError, match="at least 2"):
        g.model.geometry.add_bezier([p])


# =====================================================================
# Curve loop + Plane surface  (dim = 2)
# =====================================================================

def test_add_curve_loop_and_plane_surface(g):
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(1, 0, 0)
    p3 = g.model.geometry.add_point(1, 1, 0)
    p4 = g.model.geometry.add_point(0, 1, 0)
    l1 = g.model.geometry.add_line(p1, p2)
    l2 = g.model.geometry.add_line(p2, p3)
    l3 = g.model.geometry.add_line(p3, p4)
    l4 = g.model.geometry.add_line(p4, p1)
    loop = g.model.geometry.add_curve_loop([l1, l2, l3, l4])
    assert loop > 0

    n2 = _entity_count(2)
    surf = g.model.geometry.add_plane_surface(loop, label="quad_surf")
    assert surf > 0
    assert _entity_count(2) > n2
    assert g.model._metadata[(2, surf)]['kind'] == 'plane_surface'
    assert g.labels.has("quad_surf")


# =====================================================================
# Rectangle  (dim = 2)
# =====================================================================

def test_add_rectangle(g):
    n2 = _entity_count(2)
    tag = g.model.geometry.add_rectangle(0, 0, 0, 4.0, 3.0, label="plate")
    assert tag > 0
    assert _entity_count(2) > n2
    assert g.model._metadata[(2, tag)]['kind'] == 'rectangle'
    assert g.labels.has("plate")
    assert tag in g.labels.entities("plate")


def test_add_rectangle_rotated_about_centre(g):
    # 90 about X tips an XY rectangle into the XZ plane (dy -> dz),
    # rotated about its own centre so the centre stays put.
    tag = g.model.geometry.add_rectangle(
        -2, -1.5, 0, 4.0, 3.0,
        angles_deg=(90.0, 0.0, 0.0),
    )
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(2, tag)
    assert xmax - xmin == pytest.approx(4.0, abs=1e-6)
    assert ymax - ymin == pytest.approx(0.0, abs=1e-6)
    assert zmax - zmin == pytest.approx(3.0, abs=1e-6)
    # centre preserved
    assert (xmin + xmax) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (ymin + ymax) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (zmin + zmax) / 2 == pytest.approx(0.0, abs=1e-6)


def test_add_rectangle_pivot_offset_from_centre(g):
    # 90 about Z, pivot offset = (-dx/2, -dy/2, 0) places the pivot at
    # the bottom-left corner.  After rotation the bottom-left corner
    # must remain at the original (x, y, z).
    tag = g.model.geometry.add_rectangle(
        1.0, 2.0, 0.0, 4.0, 3.0,
        angles_deg=(0.0, 0.0, 90.0),
        pivot=(-2.0, -1.5, 0.0),
    )
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(2, tag)
    assert xmin == pytest.approx(1.0 - 3.0, abs=1e-6)
    assert ymin == pytest.approx(2.0,        abs=1e-6)
    assert xmax == pytest.approx(1.0,        abs=1e-6)
    assert ymax == pytest.approx(2.0 + 4.0,  abs=1e-6)


def test_add_rectangle_radians_matches_degrees(g):
    deg = g.model.geometry.add_rectangle(
        0, 0, 0, 2.0, 1.0, angles_deg=(0.0, 30.0, 0.0),
    )
    rad = g.model.geometry.add_rectangle(
        0, 0, 0, 2.0, 1.0, angles_rad=(0.0, math.radians(30.0), 0.0),
    )
    assert _bb(2, deg) == pytest.approx(_bb(2, rad), abs=1e-9)


def test_add_rectangle_rejects_both_angle_units(g):
    with pytest.raises(ValueError, match="angles_deg or angles_rad"):
        g.model.geometry.add_rectangle(
            0, 0, 0, 1.0, 1.0,
            angles_deg=(10.0, 0.0, 0.0),
            angles_rad=(0.1, 0.0, 0.0),
        )


# =====================================================================
# Box  (dim = 3)
# =====================================================================

def test_add_box(g):
    n3 = _entity_count(3)
    tag = g.model.geometry.add_box(0, 0, 0, 2.0, 3.0, 4.0)
    assert tag > 0
    assert _entity_count(3) > n3
    assert g.model._metadata[(3, tag)]['kind'] == 'box'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    assert xmax - xmin == pytest.approx(2.0, abs=1e-6)
    assert ymax - ymin == pytest.approx(3.0, abs=1e-6)
    assert zmax - zmin == pytest.approx(4.0, abs=1e-6)


def test_add_box_label(g):
    tag = g.model.geometry.add_box(1, 1, 1, 1, 1, 1, label="unit_cube")
    assert g.labels.has("unit_cube")
    assert tag in g.labels.entities("unit_cube")


# =====================================================================
# Sphere  (dim = 3)
# =====================================================================

def test_add_sphere(g):
    tag = g.model.geometry.add_sphere(0, 0, 0, 5.0)
    assert tag > 0
    assert g.model._metadata[(3, tag)]['kind'] == 'sphere'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    assert xmax - xmin == pytest.approx(10.0, abs=1e-6)
    assert ymax - ymin == pytest.approx(10.0, abs=1e-6)
    assert zmax - zmin == pytest.approx(10.0, abs=1e-6)


# =====================================================================
# Cylinder  (dim = 3)
# =====================================================================

def test_add_cylinder(g):
    tag = g.model.geometry.add_cylinder(0, 0, 0, 0, 0, 10.0, 3.0)
    assert tag > 0
    assert g.model._metadata[(3, tag)]['kind'] == 'cylinder'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    assert zmax - zmin == pytest.approx(10.0, abs=1e-6)
    assert xmax - xmin == pytest.approx(6.0, abs=1e-6)  # diameter


# =====================================================================
# Cone  (dim = 3)
# =====================================================================

def test_add_cone(g):
    tag = g.model.geometry.add_cone(0, 0, 0, 0, 0, 5.0, 3.0, 1.0)
    assert tag > 0
    assert g.model._metadata[(3, tag)]['kind'] == 'cone'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    assert zmax - zmin == pytest.approx(5.0, abs=1e-6)
    # base diameter = 6, top diameter = 2 => overall X extent = 6
    assert xmax - xmin == pytest.approx(6.0, abs=1e-6)


# =====================================================================
# Torus  (dim = 3)
# =====================================================================

def test_add_torus(g):
    tag = g.model.geometry.add_torus(0, 0, 0, 5.0, 1.0)
    assert tag > 0
    assert g.model._metadata[(3, tag)]['kind'] == 'torus'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    # total diameter = 2 * (R + r) = 12 (OCC adds small tolerance)
    assert xmax - xmin == pytest.approx(12.0, rel=0.1)
    assert ymax - ymin == pytest.approx(12.0, rel=0.1)
    # thickness = 2 * r = 2
    assert zmax - zmin == pytest.approx(2.0, rel=0.1)


# =====================================================================
# Wedge  (dim = 3)
# =====================================================================

def test_add_wedge(g):
    tag = g.model.geometry.add_wedge(0, 0, 0, 4.0, 3.0, 2.0, 1.0)
    assert tag > 0
    assert g.model._metadata[(3, tag)]['kind'] == 'wedge'
    xmin, ymin, zmin, xmax, ymax, zmax = _bb(3, tag)
    assert xmax - xmin == pytest.approx(4.0, abs=1e-6)
    assert ymax - ymin == pytest.approx(3.0, abs=1e-6)
    assert zmax - zmin == pytest.approx(2.0, abs=1e-6)


# =====================================================================
# Flexible reference resolution on builders
# (int / str-label / (dim, tag) / signed '-label' — same contract the
#  rest of the library honours; raw int tags stay backward compatible)
# =====================================================================

def test_add_line_accepts_label_strings(g):
    g.model.geometry.add_point(0, 0, 0, label="a")
    g.model.geometry.add_point(1, 0, 0, label="b")
    tag = g.model.geometry.add_line("a", "b")
    assert g.model._metadata[(1, tag)]['kind'] == 'line'


def test_add_line_accepts_dimtag_and_int(g):
    p1 = g.model.geometry.add_point(0, 0, 0)
    p2 = g.model.geometry.add_point(1, 0, 0)
    assert g.model.geometry.add_line(p1, p2) > 0          # int (compat)
    assert g.model.geometry.add_line((0, p1), (0, p2)) > 0  # (dim, tag)


def test_add_arc_accepts_label_strings(g):
    g.model.geometry.add_point(1, 0, 0, label="s")
    g.model.geometry.add_point(0, 0, 0, label="c")
    g.model.geometry.add_point(0, 1, 0, label="e")
    tag = g.model.geometry.add_arc("s", "c", "e")
    assert g.model._metadata[(1, tag)]['kind'] == 'arc'


def test_add_curve_loop_signed_label_reversal(g):
    """A leading '-' on a label reverses that curve so the loop closes."""
    gm = g.model.geometry
    gm.add_point(0, 0, 0, label="p1")
    gm.add_point(1, 0, 0, label="p2")
    gm.add_point(1, 1, 0, label="p3")
    gm.add_point(0, 1, 0, label="p4")
    gm.add_line("p1", "p2", label="bottom")
    gm.add_line("p2", "p3", label="right")
    gm.add_line("p3", "p4", label="top")
    gm.add_line("p1", "p4", label="left")   # defined p1->p4; must reverse
    loop = gm.add_curve_loop(["bottom", "right", "top", "-left"], label="q")
    assert g.model._metadata[(1, loop)]['kind'] == 'curve_loop'
    surf = gm.add_plane_surface("q")        # curve loop by label
    assert g.model._metadata[(2, surf)]['kind'] == 'plane_surface'


def test_add_line_unknown_label_fails_loud(g):
    g.model.geometry.add_point(0, 0, 0, label="known")
    with pytest.raises(KeyError, match="not found as a label"):
        g.model.geometry.add_line("missing", "known")


def test_add_line_ambiguous_label_fails_loud(g):
    g.model.geometry.add_point(0, 0, 0, label="dup")
    g.model.geometry.add_point(1, 1, 0, label="dup")
    g.model.geometry.add_point(2, 0, 0, label="solo")
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        g.model.geometry.add_line("dup", "solo")
