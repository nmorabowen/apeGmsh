"""Off-session unit tests for polyline vertex fillet / chamfer (ADR 0097)."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.core._polyline import (
    expand_polyline,
    normalize_polyline_points,
    sharp_untreated_vertices,
)


SQUARE = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]


def test_closed_square_four_lines():
    segs = expand_polyline(SQUARE, closed=True)
    assert len(segs) == 4
    assert all(s.kind == "line" for s in segs)


def test_repeated_closing_vertex_dropped():
    pts = normalize_polyline_points(SQUARE + [SQUARE[0]], closed=True)
    assert pts.shape == (4, 3)


def test_square_fillet_all_corners():
    r = 0.1
    segs = expand_polyline(SQUARE, closed=True, fillet={0: r, 1: r, 2: r, 3: r})
    kinds = [s.kind for s in segs]
    assert kinds.count("line") == 4
    assert kinds.count("arc") == 4
    arcs = [s for s in segs if s.kind == "arc"]
    for a in arcs:
        assert a.center is not None
        np.testing.assert_allclose(
            np.linalg.norm(a.start - a.center), r, atol=1e-9,
        )
        np.testing.assert_allclose(
            np.linalg.norm(a.end - a.center), r, atol=1e-9,
        )


def test_square_corner_fillet_geometry():
    """Vertex 1 at (1,0,0): T1=(0.9,0,0), T2=(1,0.1,0), C=(0.9,0.1,0)."""
    segs = expand_polyline(SQUARE, closed=True, fillet={1: 0.1})
    arcs = [s for s in segs if s.kind == "arc"]
    assert len(arcs) == 1
    a = arcs[0]
    np.testing.assert_allclose(a.start, (0.9, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(a.end, (1.0, 0.1, 0.0), atol=1e-9)
    np.testing.assert_allclose(a.center, (0.9, 0.1, 0.0), atol=1e-9)


def test_chamfer_inserts_line_not_arc():
    segs = expand_polyline(SQUARE, closed=True, chamfer={1: 0.1})
    kinds = [s.kind for s in segs]
    assert kinds.count("line") == 5
    assert kinds.count("arc") == 0


def test_fillet_and_chamfer_same_vertex_raises():
    with pytest.raises(ValueError, match="both fillet"):
        expand_polyline(SQUARE, closed=True, fillet={1: 0.1}, chamfer={1: 0.1})


def test_setback_exceeds_segment_raises():
    with pytest.raises(ValueError, match="meet or exceed"):
        expand_polyline(SQUARE, closed=True, fillet={1: 1.1})


def test_open_endpoint_fillet_raises():
    with pytest.raises(ValueError, match="cannot take a fillet"):
        expand_polyline(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0)], fillet={0: 0.1},
        )


def test_open_middle_fillet_one_arc():
    segs = expand_polyline(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0)], fillet={1: 0.1},
    )
    assert [s.kind for s in segs] == ["line", "arc", "line"]


def test_collinear_fillet_raises():
    with pytest.raises(ValueError, match="collinear"):
        expand_polyline(
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)], fillet={1: 0.1},
        )


def test_sharp_untreated_square_corners():
    pts = normalize_polyline_points(SQUARE, closed=True)
    sharp = sharp_untreated_vertices(pts, closed=True, treated=set())
    assert {i for i, _ in sharp} == {0, 1, 2, 3}
    for _, deg in sharp:
        assert deg == pytest.approx(90.0, abs=0.1)


def test_sharp_treated_vertex_omitted():
    pts = normalize_polyline_points(SQUARE, closed=True)
    sharp = sharp_untreated_vertices(pts, closed=True, treated={1})
    assert 1 not in {i for i, _ in sharp}


def test_need_three_closed_vertices():
    with pytest.raises(ValueError, match="at least 3"):
        expand_polyline([(0, 0, 0), (1, 0, 0)], closed=True)
