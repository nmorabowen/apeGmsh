"""Shape-function library tests — partition of unity, corner identity,
and Jacobian-determinant correctness on canonical elements.

Covers every element type in the catalog. Linear types (Line2, Tri3,
Quad4, Tet4, Hex8) plus higher-order additions: Wedge6, Tri6, Quad9,
Tet10, Hex27, Quad8, Hex20.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results._shape_functions import (
    SHAPE_FUNCTIONS_BY_GMSH_CODE,
    compute_jacobian_dets,
    compute_physical_coords,
    get_shape_functions,
    hex8_N, hex8_dN,
    line2_N, line2_dN,
    quad4_N, quad4_dN,
    tet4_N, tet4_dN,
    tri3_N, tri3_dN,
    wedge6_N, wedge6_dN,
    tri6_N, tri6_dN,
    quad9_N, quad9_dN,
    tet10_N, tet10_dN,
    hex27_N, hex27_dN,
    quad8_N, quad8_dN,
    hex20_N, hex20_dN,
)


# =====================================================================
# Catalog
# =====================================================================

_ALL_CODES = (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 16, 17)


def test_catalog_has_every_supported_type():
    for code in _ALL_CODES:
        entry = get_shape_functions(code)
        assert entry is not None, f"Gmsh code {code} missing from catalog"
        N_fn, dN_fn, geom, n_corner = entry
        assert callable(N_fn)
        assert callable(dN_fn)
        assert geom in ("line", "shell", "solid")
        assert n_corner > 0


def test_catalog_unsupported_returns_none():
    assert get_shape_functions(7) is None     # pyramid5 — out of scope
    assert get_shape_functions(8) is None     # line3 — out of scope
    assert get_shape_functions(99) is None


# =====================================================================
# Line2
# =====================================================================

def test_line2_at_endpoints():
    N = line2_N(np.array([[-1.0], [1.0]]))
    np.testing.assert_allclose(N[0], [1.0, 0.0])
    np.testing.assert_allclose(N[1], [0.0, 1.0])


def test_line2_partition_of_unity():
    rng = np.random.default_rng(1)
    nat = rng.uniform(-1, 1, size=(20, 1))
    N = line2_N(nat)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)


def test_line2_world_at_midpoint():
    nat = np.array([[0.0]])
    nodes = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    world = compute_physical_coords(nat, nodes, line2_N)
    np.testing.assert_allclose(world[0, 0], [5.0, 0.0, 0.0])


def test_line2_jacobian_is_half_length():
    """For a 1-D line, the line measure |∂x/∂ξ| equals L/2."""
    nat = np.array([[0.0]])
    nodes = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    j = compute_jacobian_dets(nat, nodes, line2_dN, "line")
    np.testing.assert_allclose(j, 5.0)


# =====================================================================
# Tri3
# =====================================================================

def test_tri3_at_corners():
    N = tri3_N(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0])


def test_tri3_partition_of_unity():
    rng = np.random.default_rng(2)
    nat = rng.uniform(0, 1, size=(20, 2))
    # Constrain to the unit triangle (xi + eta <= 1)
    mask = (nat.sum(axis=1) <= 1.0)
    nat = nat[mask]
    N = tri3_N(nat)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)


def test_tri3_world_at_centroid():
    """Centroid of the unit triangle → centroid of the physical triangle."""
    nat = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    nodes = np.array(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
    )
    world = compute_physical_coords(nat, nodes, tri3_N)
    np.testing.assert_allclose(world[0, 0], [1.0, 1.0, 0.0])


def test_tri3_shell_jacobian_is_2x_area():
    """Shell measure for tri3 = 2 × physical-triangle area."""
    nat = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    nodes = np.array(
        [[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]],
    )
    j = compute_jacobian_dets(nat, nodes, tri3_dN, "shell")
    # Triangle area = 0.5 * 3 * 4 = 6 → shell measure = 12
    np.testing.assert_allclose(j, 12.0)


# =====================================================================
# Quad4
# =====================================================================

def test_quad4_at_corners():
    corners = np.array([
        [-1, -1], [+1, -1], [+1, +1], [-1, +1],
    ], dtype=np.float64)
    N = quad4_N(corners)
    for i in range(4):
        expected = np.zeros(4)
        expected[i] = 1.0
        np.testing.assert_allclose(N[i], expected, atol=1e-12)


def test_quad4_partition_of_unity():
    rng = np.random.default_rng(3)
    nat = rng.uniform(-1, 1, size=(20, 2))
    N = quad4_N(nat)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)


def test_quad4_at_centre():
    """N at natural origin = 1/4 for all four nodes."""
    N = quad4_N(np.array([[0.0, 0.0]]))
    np.testing.assert_allclose(N[0], [0.25, 0.25, 0.25, 0.25])


def test_quad4_world_at_centre():
    nat = np.array([[0.0, 0.0]])
    nodes = np.array([[
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [4.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
    ]])
    world = compute_physical_coords(nat, nodes, quad4_N)
    np.testing.assert_allclose(world[0, 0], [2.0, 1.0, 0.0])


# =====================================================================
# Tet4
# =====================================================================

def test_tet4_at_corners():
    nat = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    N = tet4_N(nat)
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(N[1], [0.0, 1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(N[2], [0.0, 0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(N[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_tet4_partition_of_unity():
    rng = np.random.default_rng(4)
    nat = rng.uniform(0, 1, size=(20, 3))
    mask = (nat.sum(axis=1) <= 1.0)
    nat = nat[mask]
    N = tet4_N(nat)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)


def test_tet4_world_at_centroid():
    nat = np.array([[0.25, 0.25, 0.25]])
    nodes = np.array([[
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]])
    world = compute_physical_coords(nat, nodes, tet4_N)
    np.testing.assert_allclose(world[0, 0], [1.0, 1.0, 1.0])


def test_tet4_solid_jacobian_is_6x_volume():
    """For tet4, det(J) = 6 × physical-tet volume."""
    nat = np.array([[0.25, 0.25, 0.25]])
    # Unit tet (axes-aligned) has volume 1/6 → det(J) = 1
    nodes = np.array([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    j = compute_jacobian_dets(nat, nodes, tet4_dN, "solid")
    np.testing.assert_allclose(j, 1.0)


# =====================================================================
# Hex8
# =====================================================================

def test_hex8_at_corners():
    corners = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float64)
    N = hex8_N(corners)
    for i in range(8):
        expected = np.zeros(8)
        expected[i] = 1.0
        np.testing.assert_allclose(N[i], expected, atol=1e-12)


def test_hex8_at_centre_uniform():
    N = hex8_N(np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(N[0], np.full(8, 1.0 / 8.0))


def test_hex8_partition_of_unity():
    rng = np.random.default_rng(5)
    nat = rng.uniform(-1, 1, size=(20, 3))
    N = hex8_N(nat)
    np.testing.assert_allclose(N.sum(axis=1), 1.0, atol=1e-12)


def test_hex8_world_at_centre():
    nat = np.array([[0.0, 0.0, 0.0]])
    nodes = np.array([[
        [0, 0, 0], [2, 0, 0], [2, 4, 0], [0, 4, 0],
        [0, 0, 6], [2, 0, 6], [2, 4, 6], [0, 4, 6],
    ]], dtype=np.float64)
    world = compute_physical_coords(nat, nodes, hex8_N)
    np.testing.assert_allclose(world[0, 0], [1.0, 2.0, 3.0])


def test_hex8_solid_jacobian_is_8x_volume_unit_cube():
    """For a unit-cube hex8, det(J) at any interior point = 1.

    The natural-cube has volume 8 (in [-1,+1]^3) and the physical cube
    has volume 1; det(J) = 1/8 × volume_phys / volume_nat → here = 1/8.
    Sanity test: at centre, det(J) = 1/8.
    """
    nat = np.array([[0.0, 0.0, 0.0]])
    nodes = np.array([[
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]], dtype=np.float64)
    j = compute_jacobian_dets(nat, nodes, hex8_dN, "solid")
    np.testing.assert_allclose(j, 1.0 / 8.0)


# =====================================================================
# compute_physical_coords vectorization
# =====================================================================

def test_compute_physical_coords_batch_hex8():
    """Multiple elements + multiple IPs in one einsum call."""
    nat = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])    # 2 IPs
    nodes_batch = np.array([
        # Element 0: unit cube
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        # Element 1: same shape, shifted +5 in x
        [[5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0],
         [5, 0, 1], [6, 0, 1], [6, 1, 1], [5, 1, 1]],
    ], dtype=np.float64)
    world = compute_physical_coords(nat, nodes_batch, hex8_N)
    assert world.shape == (2, 2, 3)
    # Element 0, IP 0 (centre): (0.5, 0.5, 0.5)
    np.testing.assert_allclose(world[0, 0], [0.5, 0.5, 0.5])
    # Element 1, IP 0 (centre): (5.5, 0.5, 0.5)
    np.testing.assert_allclose(world[1, 0], [5.5, 0.5, 0.5])
    # Element 0, IP 1 (xi=0.5): centre in y, z + 0.75 in x
    np.testing.assert_allclose(world[0, 1], [0.75, 0.5, 0.5])


# =====================================================================
# Higher-order types — Kronecker delta + partition of unity +
# linear-precision round-trip
# =====================================================================
#
# The linear-precision test is the strongest: it picks a known linear
# field f(p), evaluates it at each node's natural coord to get nodal
# values f_i, then asserts that sum(N_i(query) · f_i) reproduces
# f(query) at random query points. This catches both wrong shape
# functions AND wrong assumed natural coords (which a pure Kronecker
# delta test would not, since it'd be self-consistent against
# whatever natural coords the test used).


# (gmsh_code, N_fn, dN_fn, node_natural_coords, parent_dim, name)
_HIGHER_ORDER = [
    (
        6, wedge6_N, wedge6_dN,
        np.array([
            [0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0],
            [0.0, 0.0, +1.0], [1.0, 0.0, +1.0], [0.0, 1.0, +1.0],
        ]),
        3, "wedge6",
    ),
    (
        9, tri6_N, tri6_dN,
        np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        ]),
        2, "tri6",
    ),
    (
        10, quad9_N, quad9_dN,
        np.array([
            [-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0],
            [ 0.0, -1.0], [+1.0,  0.0], [ 0.0, +1.0], [-1.0,  0.0],
            [ 0.0,  0.0],
        ]),
        2, "quad9",
    ),
    (
        11, tet10_N, tet10_dN,
        np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        ]),
        3, "tet10",
    ),
    (
        12, hex27_N, hex27_dN,
        np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
            [ 0, -1, -1], [-1,  0, -1], [-1, -1,  0],
            [+1,  0, -1], [+1, -1,  0], [ 0, +1, -1],
            [+1, +1,  0], [-1, +1,  0], [ 0, -1, +1],
            [-1,  0, +1], [+1,  0, +1], [ 0, +1, +1],
            [ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0],
            [+1,  0,  0], [ 0, +1,  0], [ 0,  0, +1],
            [ 0,  0,  0],
        ], dtype=np.float64),
        3, "hex27",
    ),
    (
        16, quad8_N, quad8_dN,
        np.array([
            [-1.0, -1.0], [+1.0, -1.0], [+1.0, +1.0], [-1.0, +1.0],
            [ 0.0, -1.0], [+1.0,  0.0], [ 0.0, +1.0], [-1.0,  0.0],
        ]),
        2, "quad8",
    ),
    (
        17, hex20_N, hex20_dN,
        np.array([
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
            [ 0, -1, -1], [-1,  0, -1], [-1, -1,  0],
            [+1,  0, -1], [+1, -1,  0], [ 0, +1, -1],
            [+1, +1,  0], [-1, +1,  0], [ 0, -1, +1],
            [-1,  0, +1], [+1,  0, +1], [ 0, +1, +1],
        ], dtype=np.float64),
        3, "hex20",
    ),
]


def _interior_points(parent_dim: int, gmsh_code: int, n: int = 20):
    """Random ``(n, parent_dim)`` points strictly inside the parent."""
    rng = np.random.default_rng(int(gmsh_code) * 17 + 3)
    if gmsh_code in (9, 11):
        # Barycentric tri / tet — sample, then constrain
        nat = rng.uniform(0.05, 0.95, size=(n * 4, parent_dim))
        mask = nat.sum(axis=1) <= 0.95
        return nat[mask][:n]
    if gmsh_code == 6:
        # Wedge6: tri × line. ξ + η <= 1, ζ in [-1, +1].
        out = []
        while len(out) < n:
            p = rng.uniform(low=[0, 0, -1], high=[1, 1, 1])
            if p[0] + p[1] <= 0.95:
                out.append(p)
        return np.array(out, dtype=np.float64)
    # [-1, +1]^d types
    return rng.uniform(-0.95, 0.95, size=(n, parent_dim))


@pytest.mark.parametrize(
    "gmsh_code,N_fn,dN_fn,node_nat,pdim,name",
    _HIGHER_ORDER,
    ids=[spec[5] for spec in _HIGHER_ORDER],
)
def test_higher_order_kronecker_delta(
    gmsh_code, N_fn, dN_fn, node_nat, pdim, name,
):
    """N evaluated at node i's natural coord = e_i."""
    N = N_fn(node_nat)
    np.testing.assert_allclose(
        N, np.eye(node_nat.shape[0]), atol=1e-12,
        err_msg=f"{name}: Kronecker delta failed",
    )


@pytest.mark.parametrize(
    "gmsh_code,N_fn,dN_fn,node_nat,pdim,name",
    _HIGHER_ORDER,
    ids=[spec[5] for spec in _HIGHER_ORDER],
)
def test_higher_order_partition_of_unity(
    gmsh_code, N_fn, dN_fn, node_nat, pdim, name,
):
    """Sum of shape functions = 1 at every interior point."""
    pts = _interior_points(pdim, gmsh_code)
    N = N_fn(pts)
    np.testing.assert_allclose(
        N.sum(axis=1), 1.0, atol=1e-12,
        err_msg=f"{name}: partition of unity failed",
    )


@pytest.mark.parametrize(
    "gmsh_code,N_fn,dN_fn,node_nat,pdim,name",
    _HIGHER_ORDER,
    ids=[spec[5] for spec in _HIGHER_ORDER],
)
def test_higher_order_linear_precision(
    gmsh_code, N_fn, dN_fn, node_nat, pdim, name,
):
    """For a linear field f(p), sum(N_i(q) · f(node_i)) = f(q).

    This catches both wrong shape functions AND wrong assumed natural
    coords — together they must agree with the field's nodal samples.
    """
    rng = np.random.default_rng(int(gmsh_code) * 7 + 11)
    coeffs = rng.uniform(-3, 3, size=pdim + 1)
    # f(p) = c0 + c1·p[0] + c2·p[1] (+ c3·p[2])
    nodal_vals = coeffs[0] + node_nat @ coeffs[1:]    # (n_nodes,)

    pts = _interior_points(pdim, gmsh_code)
    expected = coeffs[0] + pts @ coeffs[1:]
    N = N_fn(pts)
    actual = N @ nodal_vals
    np.testing.assert_allclose(
        actual, expected, atol=1e-12,
        err_msg=f"{name}: linear precision failed",
    )


@pytest.mark.parametrize(
    "gmsh_code,N_fn,dN_fn,node_nat,pdim,name",
    _HIGHER_ORDER,
    ids=[spec[5] for spec in _HIGHER_ORDER],
)
def test_higher_order_dN_sums_to_zero(
    gmsh_code, N_fn, dN_fn, node_nat, pdim, name,
):
    """Sum of dN over nodes = 0 (consequence of partition of unity)."""
    pts = _interior_points(pdim, gmsh_code)
    dN = dN_fn(pts)
    sums = dN.sum(axis=1)    # (n_ip, parent_dim)
    np.testing.assert_allclose(
        sums, 0.0, atol=1e-12,
        err_msg=f"{name}: dN sum failed",
    )


@pytest.mark.parametrize(
    "gmsh_code,N_fn,dN_fn,node_nat,pdim,name",
    _HIGHER_ORDER,
    ids=[spec[5] for spec in _HIGHER_ORDER],
)
def test_higher_order_dN_matches_finite_diff(
    gmsh_code, N_fn, dN_fn, node_nat, pdim, name,
):
    """Analytic dN matches finite-difference of N at random interior points."""
    pts = _interior_points(pdim, gmsh_code, n=5)
    dN_ana = dN_fn(pts)
    h = 1e-6
    for axis in range(pdim):
        plus = pts.copy();  plus[:,  axis] += h
        minus = pts.copy(); minus[:, axis] -= h
        dN_fd = (N_fn(plus) - N_fn(minus)) / (2 * h)
        np.testing.assert_allclose(
            dN_ana[:, :, axis], dN_fd, atol=1e-7, rtol=1e-6,
            err_msg=f"{name}: dN axis {axis} doesn't match FD",
        )
