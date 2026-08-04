"""ADR 0086 S1 — dual-mortar kernel unit tests (pure numpy, no gmsh).

Regression targets come from the fork's oracle suite (ADR-62 / P2.1):
partition of unity ≤ 1e-6 per row, and LINEAR PATCH EXACTNESS — the
dual projection must return the exact nodal values of any linear field,
because a linear field lies in the slave space and biorthogonality
makes the projection interpolatory on that space. (The fork's P2.1
comparison: lumped D errs 0.28 on this test; dual lands at 2e-15 —
which is why the kernel has no lumping toggle.)
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.resolvers._mortar import (
    MortarTieError,
    compute_dual_mortar_rows,
)


# ── helpers ─────────────────────────────────────────────────────────

def _coords_fn(coords: dict[int, tuple]) -> callable:
    return lambda t: np.asarray(coords[int(t)], dtype=float)


def _quad4_grid(nx, ny, *, z=0.0, tag0=1, lx=1.0, ly=1.0):
    """(coords, faces) for an nx×ny quad4 grid on [0,lx]×[0,ly] at z."""
    coords, tags = {}, {}
    t = tag0
    for j in range(ny + 1):
        for i in range(nx + 1):
            coords[t] = (lx * i / nx, ly * j / ny, z)
            tags[(i, j)] = t
            t += 1
    faces = []
    for j in range(ny):
        for i in range(nx):
            faces.append([tags[(i, j)], tags[(i + 1, j)],
                          tags[(i + 1, j + 1)], tags[(i, j + 1)]])
    return coords, np.array(faces)


def _quad8_square(*, z=0.0, tag0=100, lx=1.0, ly=1.0):
    """One quad8 facet covering [0,lx]×[0,ly] at z (gmsh ordering)."""
    c = {
        tag0 + 0: (0.0, 0.0, z), tag0 + 1: (lx, 0.0, z),
        tag0 + 2: (lx, ly, z), tag0 + 3: (0.0, ly, z),
        tag0 + 4: (lx / 2, 0.0, z), tag0 + 5: (lx, ly / 2, z),
        tag0 + 6: (lx / 2, ly, z), tag0 + 7: (0.0, ly / 2, z),
    }
    return c, np.array([[tag0 + k for k in range(8)]])


def _assert_linear_patch(rows, coords, a=2.0, b=3.0, c=-5.0, tol=1e-9):
    """P·u_m must reproduce u = a + b·x + c·y at every slave node."""
    for tag_s, m_tags, w in rows:
        um = np.array([a + b * coords[t][0] + c * coords[t][1]
                       for t in m_tags])
        us_exact = a + b * coords[tag_s][0] + c * coords[tag_s][1]
        assert float(w @ um) == pytest.approx(us_exact, abs=tol), \
            f"linear patch failed at slave node {tag_s}"


# ── conforming identity ─────────────────────────────────────────────

def test_conforming_quad4_is_identity():
    mc, mf = _quad4_grid(1, 1, tag0=1)
    sc, sf = _quad4_grid(1, 1, tag0=11)          # same square, own tags
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-6)
    assert len(rows) == 4
    for tag_s, m_tags, w in rows:
        # each slave node coincides with exactly one master node
        big = [(t, wi) for t, wi in zip(m_tags, w) if abs(wi) > 1e-9]
        assert len(big) == 1
        t_m, w_m = big[0]
        assert w_m == pytest.approx(1.0, abs=1e-9)
        assert np.allclose(coords[tag_s][:2], coords[t_m][:2])


# ── non-matching, mixed order: the reason the kernel exists ─────────

def test_quad8_slave_on_quad4_grid_linear_patch():
    mc, mf = _quad4_grid(3, 3, tag0=1)           # 3×3 quad4 master
    sc, sf = _quad8_square(tag0=100)             # one quad8 slave
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-6)
    assert len(rows) == 8                         # all 8 slave nodes tied
    for _, _, w in rows:
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
    _assert_linear_patch(rows, coords)


def test_order_swap_quad4_slaves_on_quad8_master_linear_patch():
    sc, sf = _quad4_grid(3, 3, tag0=1)           # quad4 slaves
    mc, mf = _quad8_square(tag0=100)             # quad8 master
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-6)
    assert len(rows) == 16
    _assert_linear_patch(rows, coords)


def test_tri3_slave_on_quad4_master_linear_patch():
    mc, mf = _quad4_grid(2, 2, tag0=1)
    # two tri3 facets covering the unit square
    sc = {200: (0.0, 0.0, 0.0), 201: (1.0, 0.0, 0.0),
          202: (1.0, 1.0, 0.0), 203: (0.0, 1.0, 0.0)}
    sf = np.array([[200, 201, 202], [200, 202, 203]])
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-6)
    assert len(rows) == 4
    _assert_linear_patch(rows, coords)


def test_rotated_offset_plane_invariance():
    """Same patch, interface rotated out of the axis planes."""
    mc, mf = _quad4_grid(3, 3, tag0=1)
    sc, sf = _quad8_square(tag0=100)
    coords = {**mc, **sc}
    # rotate 30° about x then 20° about y, then translate
    cx, sx = np.cos(0.5236), np.sin(0.5236)
    cy, sy = np.cos(0.3491), np.sin(0.3491)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot, shift = ry @ rx, np.array([2.0, -1.0, 3.0])
    coords3 = {t: rot @ np.asarray(v, float) + shift
               for t, v in coords.items()}
    rows = compute_dual_mortar_rows(
        sf, mf, lambda t: coords3[int(t)], gap_tol=1e-6)
    # linear patch in the ROTATED frame: u = a + b·X + c·Y (world)
    for tag_s, m_tags, w in rows:
        um = np.array([2 + 3 * coords3[t][0] - 5 * coords3[t][1]
                       for t in m_tags])
        us = 2 + 3 * coords3[tag_s][0] - 5 * coords3[tag_s][1]
        assert float(w @ um) == pytest.approx(us, abs=1e-8)


# ── fail-loud paths (ADR 0086 D3: no silent zero-force) ─────────────

def test_tri6_slave_refused():
    mc, mf = _quad4_grid(1, 1, tag0=1)
    sc = {200: (0, 0, 0), 201: (1, 0, 0), 202: (0, 1, 0),
          203: (.5, 0, 0), 204: (.5, .5, 0), 205: (0, .5, 0)}
    sf = np.array([[200, 201, 202, 203, 204, 205]])
    with pytest.raises(MortarTieError, match="tri6 SLAVE"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-6)


def test_gap_beyond_tolerance_refused():
    mc, mf = _quad4_grid(2, 2, tag0=1, z=0.0)
    sc, sf = _quad8_square(tag0=100, z=0.1)      # 0.1 gap
    with pytest.raises(MortarTieError, match="not coincident-flat"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=0.01)


def test_non_convex_slave_refused():
    mc, mf = _quad4_grid(2, 2, tag0=1)
    sc = {200: (0, 0, 0), 201: (1, 0, 0),
          202: (0.2, 0.2, 0), 203: (0, 1, 0)}    # reflex corner
    sf = np.array([[200, 201, 202, 203]])
    with pytest.raises(MortarTieError, match="non-convex"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-6)


def test_no_overlap_refused():
    mc, mf = _quad4_grid(1, 1, tag0=1)                   # [0,1]²
    sc, sf = _quad8_square(tag0=100)
    coords = {**mc, **{t: (v[0] + 5.0, v[1], v[2])      # shifted away
                       for t, v in sc.items()}}
    with pytest.raises(MortarTieError, match="only|no slave/master"):
        compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                 gap_tol=1e-6)


def test_partial_coverage_refused():
    mc, mf = _quad4_grid(1, 1, tag0=1, lx=0.5)   # master covers half
    sc, sf = _quad8_square(tag0=100)             # slave spans [0,1]²
    with pytest.raises(MortarTieError, match="covered"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-6)


def test_outward_override_accepted():
    mc, mf = _quad4_grid(2, 2, tag0=1)
    sc, sf = _quad8_square(tag0=100)
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(
        sf, mf, _coords_fn(coords), gap_tol=1e-6,
        outward=(0.0, 0.0, 1.0))
    _assert_linear_patch(rows, coords)
    with pytest.raises(MortarTieError, match="zero vector"):
        compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                 gap_tol=1e-6, outward=(0, 0, 0))
