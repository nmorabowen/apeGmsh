"""Tests — ADR 0080 B5 drafting aids (``sections/_drafting.py``).

Pure geometry, **no Qt** — snap-candidate resolution, object-over-grid
snap priority and tolerance windows, ortho quadrant selection, the
length/angle lock resolver, and the dynamic-input parse table (with
its rejection cases). These are the drafting-aid contract; the Qt
canvas only draws glyphs and forwards events to them.
"""
from __future__ import annotations

import math

import pytest

from apeGmsh.sections import SectionDocument
from apeGmsh.sections._drafting import (
    DraftingInputError,
    GridSpec,
    SnapCandidate,
    constrain_segment,
    ortho_project,
    parse_dynamic_input,
    resolve_snap,
    shape_outlines,
    snap_candidates,
)


# ─────────────────────────────────────────────────────────────────────
# snap candidates from resolved outlines
# ─────────────────────────────────────────────────────────────────────


def _kinds_at(cands, x, y, *, tol=1e-6):
    return {
        c.kind for c in cands
        if abs(c.x - x) < tol and abs(c.y - y) < tol
    }


def test_rect_face_endpoints_and_midpoints():
    doc = SectionDocument.new(kind="continuum")
    doc.add_shape("rect_face", id="r", b=4.0, h=2.0)
    cands = snap_candidates(doc)
    # four corners
    for cx, cy in ((-2, -1), (2, -1), (2, 1), (-2, 1)):
        assert "endpoint" in _kinds_at(cands, cx, cy)
    # edge midpoints
    assert "midpoint" in _kinds_at(cands, 0, -1)
    assert "midpoint" in _kinds_at(cands, 2, 0)
    # centroid is NOT a snap candidate for a rectangle
    assert _kinds_at(cands, 0, 0) == set()


def test_translate_and_rotate_move_candidates():
    doc = SectionDocument.new(kind="continuum")
    doc.add_shape("rect_face", id="r", b=2.0, h=2.0,
                  translate=(10.0, 0.0), rotate=90.0)
    cands = snap_candidates(doc)
    # unit square corners rotate 90° about origin then shift +10 in x:
    # (1,1)->(-1,1)->(9,1)
    assert "endpoint" in _kinds_at(cands, 9.0, 1.0)
    assert "endpoint" in _kinds_at(cands, 11.0, -1.0)


def test_circle_center_and_quadrants():
    doc = SectionDocument.new(kind="continuum")
    doc.add_shape("pipe_face", id="p", r=3.0, translate=(1.0, 2.0))
    cands = snap_candidates(doc)
    assert "center" in _kinds_at(cands, 1.0, 2.0)
    assert "quadrant" in _kinds_at(cands, 4.0, 2.0)
    assert "quadrant" in _kinds_at(cands, 1.0, 5.0)
    assert "quadrant" in _kinds_at(cands, -2.0, 2.0)


def test_segment_intersection_candidate():
    # an X made of two crossing polygons (thin quads) intersects at 0,0
    doc = SectionDocument.new(kind="continuum")
    doc.add_polygon([(-2, -0.1), (2, 0.1), (2, -0.1), (-2, 0.1)], id="x1")
    cands = snap_candidates(doc)
    assert any(
        c.kind == "intersection" and abs(c.x) < 1e-6 and abs(c.y) < 1e-6
        for c in cands
    )


def test_extra_points_are_endpoints():
    doc = SectionDocument.new(kind="continuum")
    cands = snap_candidates(doc, extra_points=[(5.0, 5.0)])
    assert "endpoint" in _kinds_at(cands, 5.0, 5.0)


def test_hollow_rect_has_inner_and_outer_loops():
    doc = SectionDocument.new(kind="continuum")
    doc.add_shape("rect_hollow_face", id="h", b=10.0, h=10.0, t=1.0)
    cands = snap_candidates(doc)
    assert "endpoint" in _kinds_at(cands, 5.0, 5.0)     # outer corner
    assert "endpoint" in _kinds_at(cands, 4.0, 4.0)     # inner corner


def test_shape_outlines_w_face_is_twelve_vertices():
    out = shape_outlines(
        {"shape": "W_face",
         "params": {"bf": 10.0, "tf": 1.0, "h": 8.0, "tw": 2.0},
         "translate": [0.0, 0.0], "rotate": None}
    )
    assert len(out.polylines) == 1
    assert len(out.polylines[0]) == 12


# ─────────────────────────────────────────────────────────────────────
# resolve_snap — object beats grid, tolerance window
# ─────────────────────────────────────────────────────────────────────


def test_object_snap_beats_grid():
    grid = GridSpec(spacing=1.0)
    cands = [SnapCandidate(0.02, 0.03, "endpoint")]
    got = resolve_snap((0.0, 0.0), cands, grid, tolerance=0.1)
    assert got is not None and got.kind == "endpoint"
    assert got.x == pytest.approx(0.02)


def test_grid_snap_when_no_object_in_range():
    grid = GridSpec(spacing=1.0)
    cands = [SnapCandidate(5.0, 5.0, "endpoint")]   # far away
    got = resolve_snap((0.4, 0.6), cands, grid, tolerance=0.1)
    assert got is not None and got.kind == "grid"
    assert (got.x, got.y) == pytest.approx((0.0, 1.0))


def test_no_snap_without_grid_or_candidate():
    assert resolve_snap((0.4, 0.6), [], None, tolerance=0.1) is None


def test_tolerance_window_excludes_far_candidate():
    cands = [SnapCandidate(0.2, 0.0, "endpoint")]
    assert resolve_snap((0.0, 0.0), cands, None, tolerance=0.1) is None
    assert resolve_snap((0.0, 0.0), cands, None, tolerance=0.3) is not None


def test_nearest_object_wins_then_kind_priority():
    # equal distance → endpoint beats midpoint
    cands = [
        SnapCandidate(0.05, 0.0, "midpoint"),
        SnapCandidate(-0.05, 0.0, "endpoint"),
    ]
    got = resolve_snap((0.0, 0.0), cands, None, tolerance=0.1)
    assert got.kind == "endpoint"
    # strictly nearer midpoint still wins over farther endpoint
    cands = [
        SnapCandidate(0.01, 0.0, "midpoint"),
        SnapCandidate(0.09, 0.0, "endpoint"),
    ]
    got = resolve_snap((0.0, 0.0), cands, None, tolerance=0.1)
    assert got.kind == "midpoint"


def test_grid_spec_rejects_bad_spacing():
    with pytest.raises(DraftingInputError):
        GridSpec(spacing=0.0)
    with pytest.raises(DraftingInputError):
        GridSpec(spacing=-1.0)


# ─────────────────────────────────────────────────────────────────────
# ortho projection — quadrant selection
# ─────────────────────────────────────────────────────────────────────


def test_ortho_horizontal_when_dx_dominates():
    assert ortho_project((0.0, 0.0), (5.0, 1.0)) == (5.0, 0.0)


def test_ortho_vertical_when_dy_dominates():
    assert ortho_project((0.0, 0.0), (1.0, 5.0)) == (0.0, 5.0)


def test_ortho_diagonal_tie_picks_horizontal():
    # |dx| == |dy| → horizontal (dx branch uses >=)
    assert ortho_project((0.0, 0.0), (3.0, 3.0)) == (3.0, 0.0)


def test_ortho_offset_anchor():
    assert ortho_project((2.0, 3.0), (10.0, 4.0)) == (10.0, 3.0)


# ─────────────────────────────────────────────────────────────────────
# constrain_segment — the lock resolver
# ─────────────────────────────────────────────────────────────────────


def test_lock_both_is_fully_determined():
    p = constrain_segment((0.0, 0.0), (99.0, 99.0), length=10.0, angle=30.0)
    assert p[0] == pytest.approx(10.0 * math.cos(math.radians(30)))
    assert p[1] == pytest.approx(10.0 * math.sin(math.radians(30)))


def test_lock_length_projects_onto_circle():
    # cursor direction is +x → vertex at (5, 0)
    p = constrain_segment((0.0, 0.0), (100.0, 0.0), length=5.0)
    assert p == pytest.approx((5.0, 0.0))
    # 45° cursor → vertex on the circle at 45°
    p = constrain_segment((0.0, 0.0), (3.0, 3.0), length=2.0)
    assert p == pytest.approx((2.0 / math.sqrt(2), 2.0 / math.sqrt(2)))


def test_lock_length_cursor_on_anchor_falls_back_to_zero_deg():
    p = constrain_segment((1.0, 1.0), (1.0, 1.0), length=4.0)
    assert p == pytest.approx((5.0, 1.0))


def test_lock_angle_projects_onto_ray():
    # angle 0° ray → vertex is the cursor's x-projection, y from anchor
    p = constrain_segment((0.0, 0.0), (7.0, 3.0), angle=0.0)
    assert p == pytest.approx((7.0, 0.0))
    # 90° ray → vertex takes cursor's y
    p = constrain_segment((0.0, 0.0), (3.0, 7.0), angle=90.0)
    assert p == pytest.approx((0.0, 7.0))


def test_lock_angle_ray_clamps_behind_anchor():
    # cursor behind the 0° ray → clamp to the anchor (distance 0)
    p = constrain_segment((0.0, 0.0), (-5.0, 2.0), angle=0.0)
    assert p == pytest.approx((0.0, 0.0))


def test_no_lock_passes_cursor_through():
    assert constrain_segment((1.0, 1.0), (4.0, 9.0)) == (4.0, 9.0)


def test_lock_rejects_negative_length_and_nonfinite_angle():
    with pytest.raises(DraftingInputError):
        constrain_segment((0.0, 0.0), (1.0, 1.0), length=-1.0)
    with pytest.raises(DraftingInputError):
        constrain_segment((0.0, 0.0), (1.0, 1.0), angle=math.inf)


# ─────────────────────────────────────────────────────────────────────
# parse_dynamic_input — parse table incl. rejections
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, anchor, expected",
    [
        ("10,20", None, (10.0, 20.0)),          # absolute
        ("10,20", (5.0, 5.0), (10.0, 20.0)),    # absolute ignores anchor
        ("-3,4.5", None, (-3.0, 4.5)),
        ("@10,-5", (1.0, 2.0), (11.0, -3.0)),   # relative cartesian
        ("35<0", (0.0, 0.0), (35.0, 0.0)),      # polar along +x
        ("10<90", (0.0, 0.0), (0.0, 10.0)),     # polar along +y
        ("@5<180", (2.0, 2.0), (-3.0, 2.0)),    # @-polar is still relative
    ],
)
def test_parse_accepts(text, anchor, expected):
    got = parse_dynamic_input(text, anchor=anchor)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, anchor",
    [
        ("", None),                 # empty
        ("   ", None),              # whitespace only
        ("abc", None),              # non-numeric, no separator
        ("1,2,3", None),            # too many parts
        ("1;2", None),              # wrong separator
        ("nan,2", None),            # non-finite
        ("inf<30", (0.0, 0.0)),     # non-finite polar length
        ("<30", (0.0, 0.0)),        # missing length
        ("35<", (0.0, 0.0)),        # missing angle
        ("1<2<3", (0.0, 0.0)),      # malformed polar
        ("35<30", None),            # polar with no anchor
        ("@10,5", None),            # relative with no anchor
        ("-5<30", (0.0, 0.0)),      # negative polar length
        (",5", None),               # missing x
        ("5,", None),               # missing y
    ],
)
def test_parse_rejects(text, anchor):
    with pytest.raises(DraftingInputError):
        parse_dynamic_input(text, anchor=anchor)
