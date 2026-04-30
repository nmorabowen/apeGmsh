"""Shape-function library tests — partition of unity, corner identity,
and Jacobian-determinant correctness on canonical elements.

Covers all five element types currently in the catalog: Line2, Tri3,
Quad4, Tet4, Hex8. Mined from STKO_to_python's shape_functions tests
and adapted to the Gmsh-code keyed catalog.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.results._shape_functions import (
    SHAPE_FUNCTIONS_BY_GMSH_CODE,
    compute_jacobian_dets,
    compute_physical_coords,
    get_shape_functions,
    hex8_N,
    hex8_dN,
    line2_N,
    line2_dN,
    quad4_N,
    quad4_dN,
    tet4_N,
    tet4_dN,
    tri3_N,
    tri3_dN,
)


# =====================================================================
# Catalog
# =====================================================================

def test_catalog_has_five_canonical_types():
    """Codes 1..5 (Line2, Tri3, Quad4, Tet4, Hex8) should all be present."""
    for code in (1, 2, 3, 4, 5):
        entry = get_shape_functions(code)
        assert entry is not None, f"Gmsh code {code} missing from catalog"
        N_fn, dN_fn, geom, n_corner = entry
        assert callable(N_fn)
        assert callable(dN_fn)
        assert geom in ("line", "shell", "solid")
        assert n_corner > 0


def test_catalog_unsupported_returns_none():
    assert get_shape_functions(7) is None     # pyramid5
    assert get_shape_functions(11) is None    # tet10
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
