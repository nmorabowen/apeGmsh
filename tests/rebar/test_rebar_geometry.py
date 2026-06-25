"""P3 — pure bend/hook geometry math (ADR 0066 §3, §5). Off-session."""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.rebar._geometry import (
    bend_plane, hook_primitives, outward_tangent,
)


def test_bend_plane_basic():
    u, n, fb = bend_plane((0, 0, 1), (1, 0, 0))
    assert np.allclose(u, (1, 0, 0))
    assert np.allclose(n, np.cross((0, 0, 1), (1, 0, 0)))   # (0,1,0)
    assert fb is False
    # u ⟂ t and n ⟂ both
    assert abs(np.dot(u, (0, 0, 1))) < 1e-12


def test_bend_plane_collinear_falls_back():
    u, n, fb = bend_plane((0, 0, 1), (0, 0, 1))     # turn ∥ tangent
    assert fb is True
    assert abs(np.dot(u, (0, 0, 1))) < 1e-12         # still ⟂ tangent


def test_metadata_hook_tail_direction_and_length():
    # 90° hook on an up-bar turning +x → tail runs +x, length == tail
    prims, fb = hook_primitives((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                angle_deg=90, tail=0.3, radius=0.05,
                                true_arc=False)
    assert len(prims) == 1 and prims[0][0] == "line"
    p0, p1 = prims[0][1], prims[0][2]
    assert np.allclose(p1 - p0, (0.3, 0, 0))
    # 180° → tail folds back along -tangent
    prims2, _ = hook_primitives((0, 0, 0), (0, 0, 1), (1, 0, 0),
                                angle_deg=180, tail=0.2, radius=0.05,
                                true_arc=False)
    assert np.allclose(prims2[0][2] - prims2[0][1], (0, 0, -0.2))


def test_true_arc_90_satisfies_addarc_invariant():
    prims, _ = hook_primitives((0, 0, 0), (0, 0, 1), (1, 0, 0),
                               angle_deg=90, tail=0.3, radius=0.05,
                               true_arc=True)
    arcs = [p for p in prims if p[0] == "arc"]
    lines = [p for p in prims if p[0] == "line"]
    assert len(arcs) == 1 and len(lines) == 1
    _, start, center, end = arcs[0]
    R = 0.05
    assert np.linalg.norm(center - start) == pytest.approx(R)
    assert np.linalg.norm(center - end) == pytest.approx(R)     # invariant
    # arc end == start of tail line (shared point → no make_conformal needed)
    assert np.allclose(end, lines[0][1])


def test_true_arc_180_splits_into_two_arcs():
    prims, _ = hook_primitives((0, 0, 0), (0, 0, 1), (1, 0, 0),
                               angle_deg=180, tail=0.2, radius=0.05,
                               true_arc=True)
    arcs = [p for p in prims if p[0] == "arc"]
    assert len(arcs) == 2
    # arc1.end == arc2.start (shared mid vertex)
    assert np.allclose(arcs[0][3], arcs[1][1])
    R = 0.05
    for _, s, c, e in arcs:
        assert np.linalg.norm(c - s) == pytest.approx(R)
        assert np.linalg.norm(c - e) == pytest.approx(R)


def test_outward_tangent_points_out_of_each_end():
    pts = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    assert np.allclose(outward_tangent(pts, at_start=True), (0, 0, -1))
    assert np.allclose(outward_tangent(pts, at_start=False), (0, 0, 1))
