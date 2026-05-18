"""B2 regression: consistent line/surface reduction for higher-order elements.

Tests the pure-math quadrature module (no Gmsh session needed) and the
resolver-level dispatch on ``len(edge)`` / ``len(face)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.core.loads.defs import LineLoadDef, SurfaceLoadDef
from apeGmsh._kernel._consistent_quadrature import (
    integrate_edge,
    integrate_face,
)
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver


# ---------------------------------------------------------------------
# integrate_edge
# ---------------------------------------------------------------------

class TestIntegrateEdge:

    def test_line2_uniform_matches_analytic(self):
        # Straight edge of length L from (0,0,0) to (L,0,0)
        L = 2.5
        coords = np.array([[0, 0, 0], [L, 0, 0]], dtype=float)
        w = integrate_edge(coords, 2)
        # Each end gets L/2
        assert w.shape == (2,)
        np.testing.assert_allclose(w, [L / 2, L / 2], atol=1e-12)

    def test_line3_uniform_straight(self):
        # Straight 3-node edge: ends + midpoint. Consistent reduction
        # under uniform q: (qL/6, 4qL/6, qL/6) → weights are (L/6, 4L/6, L/6).
        L = 3.0
        coords = np.array([
            [0, 0, 0],        # ξ = -1
            [L, 0, 0],        # ξ = +1
            [L / 2, 0, 0],    # ξ =  0
        ], dtype=float)
        w = integrate_edge(coords, 3)
        np.testing.assert_allclose(
            w, [L / 6, L / 6, 4 * L / 6], atol=1e-12,
        )

    def test_line3_sum_equals_length(self):
        # Sum of per-node weights always equals edge length for any shape.
        L = 5.0
        coords = np.array([
            [0, 0, 0], [L, 0, 0], [L / 2, 0, 0],
        ], dtype=float)
        w = integrate_edge(coords, 3)
        assert abs(w.sum() - L) < 1e-12

    def test_line3_curved(self):
        # Curved 3-node edge: mid-node off the line.  Sum of weights
        # approximates true arc length.
        coords = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],    # mid-node lifted
        ], dtype=float)
        w = integrate_edge(coords, 3)
        # True arc length of parabola y = 1 - x² over x ∈ [-1, 1]:
        # ∫ sqrt(1 + 4x²) dx evaluated ≈ 2.9578857...
        # Our 3-pt Gauss gives a close-but-not-exact value; check sign
        # and magnitude.
        assert w.sum() > 2.0  # longer than straight chord (= 2)
        assert np.all(w > 0)  # positive distribution

    def test_unsupported_edge_raises(self):
        coords = np.zeros((4, 3))
        with pytest.raises(NotImplementedError, match="4-node"):
            integrate_edge(coords, 4)


# ---------------------------------------------------------------------
# integrate_face
# ---------------------------------------------------------------------

class TestIntegrateFace:

    def _flat_quad4(self, side: float = 2.0) -> np.ndarray:
        s = side / 2
        return np.array([
            [-s, -s, 0],
            [s, -s, 0],
            [s, s, 0],
            [-s, s, 0],
        ], dtype=float)

    def test_tri3_uniform(self):
        # Right triangle at z=0 with legs of length 1
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        w, n = integrate_face(coords, 3)
        # Area = 1/2; each node gets A/3
        np.testing.assert_allclose(w, [1 / 6, 1 / 6, 1 / 6], atol=1e-12)
        assert abs(w.sum() - 0.5) < 1e-12

    def test_quad4_uniform(self):
        coords = self._flat_quad4(2.0)  # area = 4
        w, n = integrate_face(coords, 4)
        np.testing.assert_allclose(w, [1, 1, 1, 1], atol=1e-12)

    def test_tri6_uniform_flat(self):
        # Right triangle with legs 1; tri6 nodes: 3 corners + 3 midsides.
        coords = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],    # corners
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],  # midsides
        ], dtype=float)
        w, n = integrate_face(coords, 6)
        # Area = 1/2; corners integrate to 0, midsides to A/3 = 1/6.
        np.testing.assert_allclose(
            w, [0, 0, 0, 1 / 6, 1 / 6, 1 / 6], atol=1e-12,
        )
        assert abs(w.sum() - 0.5) < 1e-12

    def test_quad8_uniform_flat(self):
        # 2×2 quad centered at origin → area = 4
        coords = np.array([
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
            [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0],
        ], dtype=float)
        w, n = integrate_face(coords, 8)
        A = 4.0
        # Serendipity: corners get NEGATIVE weight -A/12, midsides A/3
        np.testing.assert_allclose(
            w, [-A / 12] * 4 + [A / 3] * 4, atol=1e-12,
        )
        assert abs(w.sum() - A) < 1e-12

    def test_quad9_uniform_flat(self):
        # 2×2 Lagrangian quad
        coords = np.array([
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
            [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0],
            [0, 0, 0],
        ], dtype=float)
        w, n = integrate_face(coords, 9)
        A = 4.0
        # Lagrangian biquadratic: corners=A/36, midsides=A·4/36=A/9, center=16A/36=4A/9
        np.testing.assert_allclose(
            w,
            [A / 36] * 4 + [4 * A / 36] * 4 + [16 * A / 36],
            atol=1e-12,
        )
        assert abs(w.sum() - A) < 1e-12

    def test_normal_weighted_flat_quad4(self):
        # Flat quad in xy plane → normal is +z; ∫ N_i n̂ |J| dA = A/n · +z
        coords = self._flat_quad4(2.0)
        _, normals = integrate_face(coords, 4)
        expected = np.array([[0, 0, 1]] * 4, dtype=float)  # A/4 = 1 per node
        np.testing.assert_allclose(normals, expected, atol=1e-12)

    def test_unsupported_face_raises(self):
        coords = np.zeros((5, 3))
        with pytest.raises(NotImplementedError, match="5-node"):
            integrate_face(coords, 5)


# ---------------------------------------------------------------------
# resolver dispatch
# ---------------------------------------------------------------------

class TestResolverConsistentDispatch:

    def _make_resolver(self, coords_by_tag):
        tags = np.array(sorted(coords_by_tag), dtype=int)
        coords = np.array([coords_by_tag[int(t)] for t in tags], dtype=float)
        return LoadResolver(tags, coords)

    def test_line2_consistent_matches_tributary(self):
        # Linear edge under uniform q: consistent == tributary
        r = self._make_resolver({1: (0, 0, 0), 2: (4, 0, 0)})
        defn = LineLoadDef(target="x", q_xyz=(0, -10, 0))
        recs_consistent = r.resolve_line_consistent(defn, [[1, 2]])
        recs_tributary = r.resolve_line_tributary(defn, [(1, 2)])
        cons = {r.node_id: r.force_xyz for r in recs_consistent}
        trib = {r.node_id: r.force_xyz for r in recs_tributary}
        assert set(cons) == set(trib)
        for nid in cons:
            np.testing.assert_allclose(cons[nid], trib[nid], atol=1e-12)

    def test_line3_consistent_uniform(self):
        # line3 under q_y = -10, length = 6:
        # F_end = qL/6 = -10 ∗ 6 / 6 = -10;  F_mid = 4qL/6 = -40
        r = self._make_resolver({
            1: (0, 0, 0), 2: (6, 0, 0), 3: (3, 0, 0),
        })
        defn = LineLoadDef(target="x", q_xyz=(0, -10, 0))
        recs = r.resolve_line_consistent(defn, [[1, 2, 3]])
        forces = {r.node_id: r.force_xyz for r in recs}
        np.testing.assert_allclose(forces[1], (0, -10, 0), atol=1e-12)
        np.testing.assert_allclose(forces[2], (0, -10, 0), atol=1e-12)
        np.testing.assert_allclose(forces[3], (0, -40, 0), atol=1e-12)

    def test_quad8_consistent_uniform_pressure(self):
        # 2×2 quad8 in xy plane under pressure p = 5 (pushes into face)
        # F_corner = -p · (−A/12) · ẑ = p·A/12·ẑ (UP, since negative weight
        # gives negative force, and consistent pressure is -p·normals_i)
        r = self._make_resolver({
            1: (-1, -1, 0), 2: (1, -1, 0), 3: (1, 1, 0), 4: (-1, 1, 0),
            5: (0, -1, 0), 6: (1, 0, 0), 7: (0, 1, 0), 8: (-1, 0, 0),
        })
        defn = SurfaceLoadDef(target="x",
                              magnitude=5.0, normal=True)
        recs = r.resolve_surface_consistent(defn, [[1, 2, 3, 4, 5, 6, 7, 8]])
        forces = {r.node_id: np.asarray(r.force_xyz) for r in recs}
        A = 4.0
        p = 5.0
        # For normal=True: F_i = -p · ∫N_i · (cross) dA where cross = +ẑ·|J|
        # ∫ N_corner · |J| dA = -A/12 → F_corner_z = -p · (-A/12) = +pA/12
        # ∫ N_midside · |J| dA = A/3   → F_midside_z = -p·A/3
        corner_fz = p * A / 12      # = 5/3
        midside_fz = -p * A / 3     # = -20/3
        for nid in (1, 2, 3, 4):
            np.testing.assert_allclose(
                forces[nid], (0, 0, corner_fz), atol=1e-10,
            )
        for nid in (5, 6, 7, 8):
            np.testing.assert_allclose(
                forces[nid], (0, 0, midside_fz), atol=1e-10,
            )

    def test_unsupported_edge_node_count_raises(self):
        r = self._make_resolver({
            1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (3, 0, 0),
        })
        defn = LineLoadDef(target="x", q_xyz=(0, -1, 0))
        with pytest.raises(NotImplementedError, match="4-node"):
            r.resolve_line_consistent(defn, [[1, 2, 3, 4]])

    def test_unsupported_face_node_count_raises(self):
        r = self._make_resolver({i: (i, 0, 0) for i in range(1, 6)})
        defn = SurfaceLoadDef(target="x", magnitude=1.0)
        with pytest.raises(NotImplementedError, match="5-node"):
            r.resolve_surface_consistent(defn, [[1, 2, 3, 4, 5]])
