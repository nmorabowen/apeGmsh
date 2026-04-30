"""Unit + slab-level coverage for ``_gauss_extrapolation``.

The matrix-level tests pin the round-trip identity: given nodal values
forming a linear field, evaluating the shape functions at the GPs and
then applying the extrapolation matrix recovers the original nodal
values exactly. This is the algorithm's central claim.

The slab-level tests use a tiny fake-FEM stand-in (``SimpleNamespace``
mimicking just the attributes the extrapolation pipeline reads) so we
can exercise the full ``extrapolate_gauss_slab_to_nodes`` flow without
spinning up a real Gmsh mesh.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh.results._gauss_extrapolation import (
    _build_extrapolation_matrix,
    extrapolate_gauss_slab_to_nodes,
    per_element_max_gp_count,
)
from apeGmsh.results._slabs import GaussSlab


# =====================================================================
# Canonical Gauss-point coordinates
# =====================================================================

_INV_SQRT3 = 1.0 / np.sqrt(3.0)

# 2x2x2 GPs for hex8 — order matches the standard tensor product
_HEX8_GPS = np.array([
    [-_INV_SQRT3, -_INV_SQRT3, -_INV_SQRT3],
    [+_INV_SQRT3, -_INV_SQRT3, -_INV_SQRT3],
    [-_INV_SQRT3, +_INV_SQRT3, -_INV_SQRT3],
    [+_INV_SQRT3, +_INV_SQRT3, -_INV_SQRT3],
    [-_INV_SQRT3, -_INV_SQRT3, +_INV_SQRT3],
    [+_INV_SQRT3, -_INV_SQRT3, +_INV_SQRT3],
    [-_INV_SQRT3, +_INV_SQRT3, +_INV_SQRT3],
    [+_INV_SQRT3, +_INV_SQRT3, +_INV_SQRT3],
], dtype=np.float64)

# 2x2 GPs for quad4
_QUAD4_GPS = np.array([
    [-_INV_SQRT3, -_INV_SQRT3],
    [+_INV_SQRT3, -_INV_SQRT3],
    [-_INV_SQRT3, +_INV_SQRT3],
    [+_INV_SQRT3, +_INV_SQRT3],
], dtype=np.float64)

# Canonical corner coords (natural space)
_HEX8_CORNERS = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float64)
_QUAD4_CORNERS = np.array([
    [-1, -1], [+1, -1], [+1, +1], [-1, +1],
], dtype=np.float64)


# =====================================================================
# Matrix-level: linear-field round-trip
# =====================================================================


def test_hex8_8gp_round_trip_is_exact_for_trilinear_field():
    """For hex8 the 8-node × 8-GP system is square and invertible.

    Build a trilinear nodal field, push it forward through the shape
    functions to get GP values, then extrapolate back. The matrix
    must reproduce the nodal values exactly (up to floating-point
    noise).
    """
    from apeGmsh.results._shape_functions import hex8_N

    # f(x, y, z) = a + bx + cy + dz + e*xy + f*yz + g*zx + h*xyz
    coeffs = np.array([1.5, -2.0, 0.7, 3.1, 0.2, -0.5, 0.9, 0.4])
    x = _HEX8_CORNERS[:, 0]
    y = _HEX8_CORNERS[:, 1]
    z = _HEX8_CORNERS[:, 2]
    nodal = (
        coeffs[0]
        + coeffs[1] * x + coeffs[2] * y + coeffs[3] * z
        + coeffs[4] * x * y + coeffs[5] * y * z + coeffs[6] * z * x
        + coeffs[7] * x * y * z
    )

    A = hex8_N(_HEX8_GPS)                  # (8, 8)
    gp_values = A @ nodal                  # (8,)

    M = _build_extrapolation_matrix(_HEX8_GPS, gmsh_code=5)
    assert M is not None
    assert M.shape == (8, 8)
    np.testing.assert_allclose(M @ gp_values, nodal, atol=1e-12)


def test_quad4_4gp_round_trip_is_exact_for_bilinear_field():
    from apeGmsh.results._shape_functions import quad4_N

    coeffs = np.array([2.0, -1.0, 0.5, 0.3])
    x = _QUAD4_CORNERS[:, 0]
    y = _QUAD4_CORNERS[:, 1]
    nodal = coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * x * y

    A = quad4_N(_QUAD4_GPS)
    gp_values = A @ nodal

    M = _build_extrapolation_matrix(_QUAD4_GPS, gmsh_code=3)
    assert M is not None
    assert M.shape == (4, 4)
    np.testing.assert_allclose(M @ gp_values, nodal, atol=1e-12)


def test_tet4_1gp_assigns_value_to_every_corner():
    """One GP, four nodes — pinv collapses to a constant assignment."""
    nat = np.array([[0.25, 0.25, 0.25]], dtype=np.float64)    # tet centroid
    M = _build_extrapolation_matrix(nat, gmsh_code=4)
    assert M is not None
    assert M.shape == (4, 1)
    # Each entry should be 1 (since N at centroid sums to 1, A is 1x4
    # with row summing to 1; pinv reproduces a column of equal weights).
    out = M @ np.array([7.5])
    np.testing.assert_allclose(out, np.full(4, 7.5), atol=1e-12)


def test_tri3_1gp_assigns_value_to_every_corner():
    nat = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
    M = _build_extrapolation_matrix(nat, gmsh_code=2)
    assert M is not None
    assert M.shape == (3, 1)
    out = M @ np.array([4.2])
    np.testing.assert_allclose(out, np.full(3, 4.2), atol=1e-12)


def test_unsupported_type_returns_none():
    assert _build_extrapolation_matrix(_HEX8_GPS, gmsh_code=99) is None


def test_tet10_falls_back_to_tet4_for_extrapolation():
    """Higher-order types extrapolate via the linear counterpart.

    The viewer's substrate paints corner nodes only (mid-side nodes
    are dropped in build_fem_scene), and a full higher-order pinv
    can produce non-constant nodal fields for a constant GP input.
    Projecting onto the linear corner shape functions sidesteps both.
    """
    # Standard 4-point Gauss rule for tet (Keast / Strang).
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    nat_4gp = np.array([
        [a, a, a],
        [b, a, a],
        [a, b, a],
        [a, a, b],
    ], dtype=np.float64)

    M = _build_extrapolation_matrix(nat_4gp, gmsh_code=11)
    assert M is not None
    # Shape is (n_corner_tet4=4, n_gp=4) — using the linear counterpart.
    assert M.shape == (4, 4)

    # Constant field — every GP value = 7.5; every corner = 7.5.
    nodal = M @ np.full(4, 7.5)
    np.testing.assert_allclose(nodal, np.full(4, 7.5), atol=1e-12)


def test_quad8_falls_back_to_quad4_for_extrapolation():
    inv_sqrt3 = 1.0 / np.sqrt(3.0)
    nat_4gp = np.array([
        [-inv_sqrt3, -inv_sqrt3],
        [+inv_sqrt3, -inv_sqrt3],
        [-inv_sqrt3, +inv_sqrt3],
        [+inv_sqrt3, +inv_sqrt3],
    ], dtype=np.float64)

    M = _build_extrapolation_matrix(nat_4gp, gmsh_code=16)
    assert M is not None
    # 4 corners (Quad4's count), 4 GPs — square invertible.
    assert M.shape == (4, 4)

    # Quad4 + 2x2 GPs is the round-trip-exact case from the linear
    # extrapolation tests; should reproduce a bilinear field exactly.
    from apeGmsh.results._shape_functions import quad4_N
    coeffs = np.array([1.5, -0.7, 2.0, 0.4])
    corners = np.array([[-1, -1], [+1, -1], [+1, +1], [-1, +1]],
                       dtype=np.float64)
    nodal_corner = (
        coeffs[0] + coeffs[1] * corners[:, 0]
        + coeffs[2] * corners[:, 1]
        + coeffs[3] * corners[:, 0] * corners[:, 1]
    )
    A = quad4_N(nat_4gp)
    gp_values = A @ nodal_corner
    np.testing.assert_allclose(M @ gp_values, nodal_corner, atol=1e-12)


# =====================================================================
# Slab-level: end-to-end extrapolation + averaging
# =====================================================================


def _fake_fem(nodes, elements):
    """Build a minimal stand-in for ``FEMData`` with the attributes
    the extrapolation pipeline actually reads.

    nodes: list of (id, (x, y, z))
    elements: list of (id, gmsh_code, [node_ids...])
    """
    node_ids = [nid for nid, _ in nodes]
    coords = np.array([c for _, c in nodes], dtype=np.float64)
    nodes_obj = SimpleNamespace(
        ids=np.asarray(node_ids, dtype=np.int64),
        coords=coords,
    )
    # Each element gets its own group of size 1 for simplicity.
    groups = []
    for eid, code, conn in elements:
        et = SimpleNamespace(code=int(code))
        groups.append(SimpleNamespace(
            element_type=et,
            ids=np.array([eid], dtype=np.int64),
            connectivity=np.asarray([conn], dtype=np.int64),
            __len__=lambda self_: 1,
        ))
    # group must support len() — SimpleNamespace doesn't, so wrap.
    class _Group:
        def __init__(self, et, ids, conn):
            self.element_type = et
            self.ids = ids
            self.connectivity = conn
        def __len__(self):
            return self.ids.size
    groups = [
        _Group(SimpleNamespace(code=int(code)),
               np.array([eid], dtype=np.int64),
               np.asarray([conn], dtype=np.int64))
        for eid, code, conn in elements
    ]
    return SimpleNamespace(nodes=nodes_obj, elements=groups)


def _make_gauss_slab(component, values_T, element_index, natural_coords):
    return GaussSlab(
        component=component,
        values=np.asarray(values_T, dtype=np.float64),
        element_index=np.asarray(element_index, dtype=np.int64),
        natural_coords=np.asarray(natural_coords, dtype=np.float64),
        local_axes_quaternion=None,
        time=np.arange(np.asarray(values_T).shape[0], dtype=np.float64),
    )


def test_single_hex8_recovers_linear_field_exactly():
    """One hex8 with a known linear nodal field. After extrapolation
    each corner should carry its original value."""
    from apeGmsh.results._shape_functions import hex8_N

    # Corner node IDs 1..8 placed at the canonical natural-space corners
    nodes = [(i + 1, _HEX8_CORNERS[i].tolist()) for i in range(8)]
    elements = [(101, 5, [n[0] for n in nodes])]
    fem = _fake_fem(nodes, elements)

    # f(x, y, z) = 5 + 2x - y + 3z (purely linear)
    nodal = (
        5.0 + 2.0 * _HEX8_CORNERS[:, 0]
        - 1.0 * _HEX8_CORNERS[:, 1]
        + 3.0 * _HEX8_CORNERS[:, 2]
    )
    A = hex8_N(_HEX8_GPS)                    # (8, 8)
    gp_values = A @ nodal                    # (8,)

    # element_index has one entry per GP, all == 101
    slab = _make_gauss_slab(
        "stress_xx",
        values_T=gp_values[None, :],         # (T=1, 8)
        element_index=np.full(8, 101, dtype=np.int64),
        natural_coords=_HEX8_GPS,
    )

    node_ids, nodal_out = extrapolate_gauss_slab_to_nodes(slab, fem)
    assert node_ids.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    np.testing.assert_allclose(nodal_out[0], nodal, atol=1e-12)


def test_two_adjacent_hex8_average_at_shared_face():
    """Two hexes sharing a face. Each writes a constant field, but
    different constants. Shared-face nodes get the mean; interior
    nodes carry their element's value verbatim.
    """
    # Two cubes side by side in x: cube A spans x∈[-1,1], cube B spans
    # x∈[1,3]. They share the face at x=1 (4 nodes).
    coords = {
        # Cube A corners
        1: [-1, -1, -1], 2: [+1, -1, -1], 3: [+1, +1, -1], 4: [-1, +1, -1],
        5: [-1, -1, +1], 6: [+1, -1, +1], 7: [+1, +1, +1], 8: [-1, +1, +1],
        # Cube B's far face — its near face reuses 2,3,7,6
        9:  [+3, -1, -1], 10: [+3, +1, -1],
        11: [+3, -1, +1], 12: [+3, +1, +1],
    }
    nodes = [(nid, coords[nid]) for nid in sorted(coords)]
    elem_a = (100, 5, [1, 2, 3, 4, 5, 6, 7, 8])
    elem_b = (200, 5, [2, 9, 10, 3, 6, 11, 12, 7])
    fem = _fake_fem(nodes, [elem_a, elem_b])

    # Constant fields — A writes 10, B writes 20
    val_a, val_b = 10.0, 20.0
    gp_a = np.full(8, val_a, dtype=np.float64)
    gp_b = np.full(8, val_b, dtype=np.float64)
    gp_values = np.concatenate([gp_a, gp_b])

    slab = _make_gauss_slab(
        "stress_xx",
        values_T=gp_values[None, :],
        element_index=np.array([100] * 8 + [200] * 8, dtype=np.int64),
        natural_coords=np.vstack([_HEX8_GPS, _HEX8_GPS]),
    )

    node_ids, nodal_out = extrapolate_gauss_slab_to_nodes(slab, fem)
    nid_to_val = dict(zip(node_ids.tolist(), nodal_out[0].tolist()))

    # Cube A's interior face (x=-1): nodes 1, 4, 5, 8 → val_a
    for n in (1, 4, 5, 8):
        assert nid_to_val[n] == pytest.approx(val_a, abs=1e-12), (
            f"node {n} should have val_a={val_a}, got {nid_to_val[n]}"
        )
    # Cube B's interior face (x=+3): nodes 9, 10, 11, 12 → val_b
    for n in (9, 10, 11, 12):
        assert nid_to_val[n] == pytest.approx(val_b, abs=1e-12), (
            f"node {n} should have val_b={val_b}, got {nid_to_val[n]}"
        )
    # Shared face (x=+1): nodes 2, 3, 6, 7 → mean
    expected_shared = 0.5 * (val_a + val_b)
    for n in (2, 3, 6, 7):
        assert nid_to_val[n] == pytest.approx(expected_shared, abs=1e-12), (
            f"node {n} should carry the mean {expected_shared}, "
            f"got {nid_to_val[n]}"
        )


def test_per_element_max_gp_count_basic():
    """Quick sanity for the n_gp probe used by the dispatcher."""
    eidx = np.array([1, 1, 1, 2, 2, 3], dtype=np.int64)
    slab = _make_gauss_slab(
        "x",
        values_T=np.zeros((1, eidx.size)),
        element_index=eidx,
        natural_coords=np.zeros((eidx.size, 3)),
    )
    assert per_element_max_gp_count(slab) == 3

    eidx_singletons = np.array([10, 20, 30, 40], dtype=np.int64)
    slab_s = _make_gauss_slab(
        "x",
        values_T=np.zeros((1, eidx_singletons.size)),
        element_index=eidx_singletons,
        natural_coords=np.zeros((eidx_singletons.size, 3)),
    )
    assert per_element_max_gp_count(slab_s) == 1


def test_empty_slab_returns_empty_arrays():
    fem = _fake_fem([(1, [0, 0, 0])], [])
    slab = _make_gauss_slab(
        "x",
        values_T=np.zeros((2, 0)),
        element_index=np.zeros(0, dtype=np.int64),
        natural_coords=np.zeros((0, 3)),
    )
    node_ids, nodal = extrapolate_gauss_slab_to_nodes(slab, fem)
    assert node_ids.size == 0
    assert nodal.shape == (2, 0)


def test_time_axis_preserved():
    """T=3 values should produce a (3, N) nodal output."""
    from apeGmsh.results._shape_functions import hex8_N

    nodes = [(i + 1, _HEX8_CORNERS[i].tolist()) for i in range(8)]
    elements = [(101, 5, [n[0] for n in nodes])]
    fem = _fake_fem(nodes, elements)

    nodal_per_step = np.array([
        np.full(8, 1.0),
        np.full(8, 2.0),
        np.full(8, 3.0),
    ])
    A = hex8_N(_HEX8_GPS)
    gp_per_step = nodal_per_step @ A.T    # (3, 8)

    slab = _make_gauss_slab(
        "stress_xx",
        values_T=gp_per_step,
        element_index=np.full(8, 101, dtype=np.int64),
        natural_coords=_HEX8_GPS,
    )

    node_ids, nodal_out = extrapolate_gauss_slab_to_nodes(slab, fem)
    assert nodal_out.shape == (3, 8)
    # Each timestep should recover its own constant.
    np.testing.assert_allclose(nodal_out[0], np.full(8, 1.0), atol=1e-12)
    np.testing.assert_allclose(nodal_out[1], np.full(8, 2.0), atol=1e-12)
    np.testing.assert_allclose(nodal_out[2], np.full(8, 3.0), atol=1e-12)
