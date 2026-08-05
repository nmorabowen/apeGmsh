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

from apeGmsh._kernel.resolvers import _mortar
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


def _linear_patch_error(rows, coords, a=2.0, b=3.0, c=-5.0):
    """Worst |P·u_m − u_s| over the rows for u = a + b·x + c·y."""
    worst = 0.0
    for tag_s, m_tags, w in rows:
        um = np.array([a + b * coords[t][0] + c * coords[t][1]
                       for t in m_tags])
        us_exact = a + b * coords[tag_s][0] + c * coords[tag_s][1]
        worst = max(worst, abs(float(w @ um) - us_exact))
    return worst


def _assert_linear_patch(rows, coords, a=2.0, b=3.0, c=-5.0, tol=1e-9):
    """P·u_m must reproduce u = a + b·x + c·y at every slave node."""
    err = _linear_patch_error(rows, coords, a, b, c)
    assert err < tol, f"linear patch failed, worst error {err:.3e}"


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


# ── R8 test honesty: a patch assertion through EVERY master type ────
#
# Partition of unity alone cannot catch a permuted midside basis — the
# permuted functions still sum to 1 and pass every internal guard.  Only
# a linear-patch assertion routed through that master type sees it, so
# quad4 (above), quad8 (above) and tri6 (here) each need their own.

def _tri6_pair_masters(tag0=1):
    """Two tri6 facets covering [0,1]², split on the 0-2 diagonal."""
    c = {tag0 + 0: (0, 0, 0), tag0 + 1: (1, 0, 0), tag0 + 2: (1, 1, 0),
         tag0 + 3: (0, 1, 0), tag0 + 4: (.5, 0, 0), tag0 + 5: (1, .5, 0),
         tag0 + 6: (.5, 1, 0), tag0 + 7: (0, .5, 0), tag0 + 8: (.5, .5, 0)}
    faces = np.array([
        [tag0 + 0, tag0 + 1, tag0 + 2, tag0 + 4, tag0 + 5, tag0 + 8],
        [tag0 + 0, tag0 + 2, tag0 + 3, tag0 + 8, tag0 + 6, tag0 + 7],
    ])
    return c, faces


def test_tri6_master_linear_patch():
    mc, mf = _tri6_pair_masters(tag0=1)
    sc, sf = _quad4_grid(3, 3, tag0=100)
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-9)
    assert len(rows) == 16
    _assert_linear_patch(rows, coords)


def test_permuted_tri6_midside_connectivity_refused():
    """A cyclically permuted tri6 midside ordering is a hard error.

    ADR-78 R8 flags this as the case partition-of-unity provably cannot
    catch (the permuted functions still sum to 1).  Here the R2 straight
    edge guard catches it structurally instead — permuted midsides are
    no longer at their edge midpoints — so it never reaches the basis.
    """
    mc, _ = _tri6_pair_masters(tag0=1)
    mf_bad = np.array([[1, 2, 3, 6, 9, 5],       # midsides 5,6,9 → 6,9,5
                       [1, 3, 4, 7, 8, 9]])      # midsides 9,7,8 → 7,8,9
    sc, sf = _quad4_grid(3, 3, tag0=100)
    with pytest.raises(MortarTieError, match="curved edge"):
        compute_dual_mortar_rows(sf, mf_bad, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-9)


def test_corrupted_tri6_basis_breaks_the_patch_assertion(monkeypatch):
    """ADR-78 R8 mutation test, kept live instead of run once.

    Corrupt the tri6 *shape functions* (not the connectivity, which the
    R2 guard now catches) by cycling the midside entries.  Σφ stays 1,
    so coverage, the dual row scaling and partition of unity all stay
    clean — only the linear patch sees it.  If this test ever starts
    failing, the tri6 patch assertion above has gone blind.
    """
    good = _mortar.SHAPE_FUNCTIONS[6]
    monkeypatch.setitem(_mortar.SHAPE_FUNCTIONS, 6,
                        lambda xi, eta: good(xi, eta)[[0, 1, 2, 4, 5, 3]])

    mc, mf = _tri6_pair_masters(tag0=1)
    sc, sf = _quad4_grid(3, 3, tag0=100)
    coords = {**mc, **sc}
    rows = compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                    gap_tol=1e-9)
    for _, _, w in rows:                      # every internal guard passes
        assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
    assert _linear_patch_error(rows, coords) > 1e-2


# ── R7: row-by-row cross-check against the fork oracle ──────────────

def _oracle_quad_mesh(nx, ny, *, nps, z=0.0):
    """Mirror of the fork oracle's quad4_mesh / quad8_mesh on [0,1]².

    Node numbering is the oracle's ``nid`` CREATION order (not
    row-major, despite R7's wording) — the reference row below is keyed
    to it.
    """
    nodes, coords, faces = {}, [], []

    def nid(x, y):
        key = (round(x, 12), round(y, 12))
        if key not in nodes:
            nodes[key] = len(coords)
            coords.append((x, y, z))
        return nodes[key]

    hx, hy = 1.0 / nx, 1.0 / ny
    for ey in range(ny):
        for ex in range(nx):
            x0, y0 = ex * hx, ey * hy
            f = [nid(x0, y0), nid(x0 + hx, y0),
                 nid(x0 + hx, y0 + hy), nid(x0, y0 + hy)]
            if nps == 8:
                f += [nid(x0 + hx / 2, y0), nid(x0 + hx, y0 + hy / 2),
                      nid(x0 + hx / 2, y0 + hy), nid(x0, y0 + hy / 2)]
            faces.append(f)
    return coords, faces


#: Fork oracle ``proto_p2_2_quad8_mortar.py`` — ADR-78 R7 reference.
_FORK_P_FRO = 3.927978688773
_FORK_ROW_0 = {0: 1.119341564, 1: -0.090534979, 2: -0.115226337,
               3: -0.090534979, 4: 0.045267490, 5: 0.057613169,
               8: 0.057613169, 9: 0.045267490, 10: -0.028806584}


def test_q4_crosscheck_matches_fork_oracle():
    """ADR-78 R7: 2×2 quad8 slave on a 3×3 quad4 master, P row-by-row.

    This is the deliverable that closes ADR 0086 D2 — the fork kernel is
    the reference implementation and these are its numbers.
    """
    sc, sf = _oracle_quad_mesh(2, 2, nps=8)
    mc, mf = _oracle_quad_mesh(3, 3, nps=4)
    assert (len(sc), len(mc)) == (21, 16)

    s0, m0 = 1000, 1
    coords = {s0 + i: p for i, p in enumerate(sc)}
    coords.update({m0 + j: p for j, p in enumerate(mc)})
    rows = compute_dual_mortar_rows(
        np.array([[s0 + k for k in f] for f in sf]),
        np.array([[m0 + k for k in f] for f in mf]),
        _coords_fn(coords), gap_tol=1e-9)

    p_mat = np.zeros((len(sc), len(mc)))
    for tag_s, m_tags, w in rows:
        p_mat[tag_s - s0, [t - m0 for t in m_tags]] = w

    # Frobenius norm is permutation-invariant, so it checks the whole
    # matrix without depending on the oracle's node numbering.
    assert float(np.linalg.norm(p_mat, "fro")) == \
        pytest.approx(_FORK_P_FRO, abs=1e-11)

    # ...and the corner row node-by-node, against the numbering above.
    # The fork values are quoted to 9 dp, so 1e-9 is the honest bound.
    for j in range(len(mc)):
        assert p_mat[0, j] == pytest.approx(_FORK_ROW_0.get(j, 0.0),
                                            abs=1e-9), f"master node {j}"


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


def test_curved_edge_refused():
    """ADR-78 R2: a midside off its edge midpoint is a hard error.

    All geometry runs on the corner polygon, exact only for straight
    edges.  Before the guard this facet was accepted and left a 0.30
    linear-patch error with every other guard clean.
    """
    mc, mf = _quad4_grid(3, 3, tag0=1)
    sc, sf = _quad8_square(tag0=100)
    sc[104] = (0.6, 0.0, 0.0)                    # was (0.5, 0, 0)
    with pytest.raises(MortarTieError, match="curved edge"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-9)


def test_straight_edge_guard_tolerates_roundoff():
    """The R2 guard must not fire on ordinary floating-point noise."""
    mc, mf = _quad4_grid(3, 3, tag0=1)
    sc, sf = _quad8_square(tag0=100)
    sc[104] = (0.5 + 1e-10, 0.0, 1e-10)          # ~1e-10 « 1e-6 · h
    coords = {**mc, **sc}
    _assert_linear_patch(
        compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                 gap_tol=1e-6),
        coords, tol=1e-8)


def test_overlapping_master_facets_refused():
    """ADR-78 R5.3: coverage counts multiplicity, so masters may not
    overlap each other.

    Two coincident masters over the left half and nothing over the
    right half sum to 100 % coverage.  Before the guard this was
    accepted with Σw = 1 on every row and a machine-exact linear patch
    — the uncovered half was silently *extrapolated*, which is why no
    downstream check can catch it.
    """
    sc = {200: (0, 0, 0), 201: (1, 0, 0), 202: (1, 1, 0), 203: (0, 1, 0)}
    sf = np.array([[200, 201, 202, 203]])
    mc = {1: (0, 0, 0), 2: (0.5, 0, 0), 3: (0.5, 1, 0), 4: (0, 1, 0),
          5: (0, 0, 0), 6: (0.5, 0, 0), 7: (0.5, 1, 0), 8: (0, 1, 0)}
    mf = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    with pytest.raises(MortarTieError, match="overlap each other"):
        compute_dual_mortar_rows(sf, mf, _coords_fn({**mc, **sc}),
                                 gap_tol=1e-9)


def test_edge_adjacent_masters_are_not_self_overlap():
    """The R5.3 guard must not fire on an ordinary conforming master
    mesh, whose facets share edges and clip to zero-area slivers."""
    mc, mf = _quad4_grid(4, 4, tag0=1)           # 16 edge-adjacent masters
    sc, sf = _quad8_square(tag0=100)
    coords = {**mc, **sc}
    _assert_linear_patch(
        compute_dual_mortar_rows(sf, mf, _coords_fn(coords),
                                 gap_tol=1e-9),
        coords)


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
