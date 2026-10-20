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

import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh.results._gauss_extrapolation import (
    GaussToNodeAccumulator,
    WarnGaussExtrapolationDegraded,
    _build_extrapolation_matrix,
    extrapolate_gauss_slab_per_element,
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
    from apeGmsh.fem._shape_functions import hex8_N

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
    from apeGmsh.fem._shape_functions import quad4_N

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
    from apeGmsh.fem._shape_functions import quad4_N
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
    from apeGmsh.fem._shape_functions import hex8_N

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


def test_accumulator_matches_reference_two_hex8_constant():
    """GaussToNodeAccumulator must reproduce the reference scatter-add.

    The accumulator (the per-frame fast path stamped on contour diagrams)
    pre-builds the scatter structure once and applies it with batched
    matmul + ``np.add.at``. It must be value-identical to the per-element
    dict reference ``extrapolate_gauss_slab_to_nodes`` it replaces. Here
    two hexes share a face writing different constants: shared-face nodes
    get the mean, interior nodes keep their element value.
    """
    from apeGmsh.results._gauss_extrapolation import GaussToNodeAccumulator

    coords = {
        1: [-1, -1, -1], 2: [+1, -1, -1], 3: [+1, +1, -1], 4: [-1, +1, -1],
        5: [-1, -1, +1], 6: [+1, -1, +1], 7: [+1, +1, +1], 8: [-1, +1, +1],
        9:  [+3, -1, -1], 10: [+3, +1, -1],
        11: [+3, -1, +1], 12: [+3, +1, +1],
    }
    nodes = [(nid, coords[nid]) for nid in sorted(coords)]
    fem = _fake_fem(nodes, [
        (100, 5, [1, 2, 3, 4, 5, 6, 7, 8]),
        (200, 5, [2, 9, 10, 3, 6, 11, 12, 7]),
    ])
    element_index = np.array([100] * 8 + [200] * 8, dtype=np.int64)
    natural_coords = np.vstack([_HEX8_GPS, _HEX8_GPS])

    val_a, val_b = 10.0, 20.0
    gp_vals = np.concatenate([np.full(8, val_a), np.full(8, val_b)])

    acc = GaussToNodeAccumulator(element_index, natural_coords, fem)
    nodal = acc.accumulate(gp_vals)
    nid_to_val = dict(zip(acc.node_ids.tolist(), nodal.tolist()))

    for n in (1, 4, 5, 8):           # cube A interior face
        assert nid_to_val[n] == pytest.approx(val_a, abs=1e-9)
    for n in (9, 10, 11, 12):        # cube B interior face
        assert nid_to_val[n] == pytest.approx(val_b, abs=1e-9)
    for n in (2, 3, 6, 7):           # shared face -> mean
        assert nid_to_val[n] == pytest.approx(0.5 * (val_a + val_b), abs=1e-9)


def test_accumulator_matches_reference_arbitrary_field():
    """Lock the two implementations together on a NON-constant field.

    Feeding arbitrary per-GP values exercises the extrapolation matrix
    (not just the averaging) so elements genuinely disagree at the shared
    face. The accumulator output must equal the reference per node.
    """
    from apeGmsh.results._gauss_extrapolation import GaussToNodeAccumulator

    coords = {
        1: [-1, -1, -1], 2: [+1, -1, -1], 3: [+1, +1, -1], 4: [-1, +1, -1],
        5: [-1, -1, +1], 6: [+1, -1, +1], 7: [+1, +1, +1], 8: [-1, +1, +1],
        9:  [+3, -1, -1], 10: [+3, +1, -1],
        11: [+3, -1, +1], 12: [+3, +1, +1],
    }
    nodes = [(nid, coords[nid]) for nid in sorted(coords)]
    fem = _fake_fem(nodes, [
        (100, 5, [1, 2, 3, 4, 5, 6, 7, 8]),
        (200, 5, [2, 9, 10, 3, 6, 11, 12, 7]),
    ])
    element_index = np.array([100] * 8 + [200] * 8, dtype=np.int64)
    natural_coords = np.vstack([_HEX8_GPS, _HEX8_GPS])

    # Deterministic, non-symmetric per-GP values at stress magnitude so the
    # two elements recover different nodal values at the shared face.
    gp_vals = (np.arange(16, dtype=np.float64) - 7.5) * 1.0e7

    acc = GaussToNodeAccumulator(element_index, natural_coords, fem)
    nodal_acc = acc.accumulate(gp_vals)

    slab = _make_gauss_slab(
        "stress_xx",
        values_T=gp_vals[None, :],
        element_index=element_index,
        natural_coords=natural_coords,
    )
    ref_ids, ref_nodal = extrapolate_gauss_slab_to_nodes(slab, fem)

    # Same sorted node-id set, same values per node.
    assert acc.node_ids.tolist() == ref_ids.tolist()
    np.testing.assert_allclose(nodal_acc, ref_nodal[0], rtol=1e-9, atol=1e-3)


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
    from apeGmsh.fem._shape_functions import hex8_N

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


# =====================================================================
# The file's own FEMData (codes = -class_tag), #2240
# =====================================================================
#
# A FEMData read from a result file's MODEL group keys every class by
# ``-class_tag``, which names no topology. Extrapolation looked that code
# up in the gmsh-keyed shape-function catalog, missed, and averaged every
# multi-point element to its mean, silently: a linear field came back flat.
# The oracle is the engine's. The fixtures are one fork solve under
# ``u_i = x_i^2 / 2`` (tests/results/test_gauss_simplex_coords.py, SOLVE),
# so a quadratic element's recorded strain eps_ii at a Gauss point IS that
# point's coordinate x_i, and extrapolating that linear field to the
# corners must give each corner its own coordinate. A bilinear class cannot
# reproduce the field, so it takes the fork's GLOBAL_GP_COORDS instead.

_SIMPLEX_GP = Path(__file__).resolve().parent / "fixtures" / "simplex_gp"
_AXES = {"tri": ("xx", "yy"), "tet": ("xx", "yy", "zz")}


def _file_reader(path: Path):
    from apeGmsh.results.readers._ladruno import LadrunoReader
    from apeGmsh.results.readers._mpco import MPCOReader

    return (LadrunoReader(path) if path.suffix == ".ladruno"
            else MPCOReader(path))


def _file_fem_and_slab(path: Path, component: str, eid: int):
    """The file's own FEMData and one element's slab of ``component``."""
    with warnings.catch_warnings():
        # The readers' own provenance warnings are not under test here.
        warnings.simplefilter("ignore")
        with _file_reader(path) as r:
            fem = r.fem()
            slab = r.read_gauss(
                r.stages()[0].id, component, element_ids=np.array([eid]),
            )
    sel = np.asarray(slab.element_index) == eid
    assert int(sel.sum()) > 1, f"{path.name}: {eid} is not multi-point"
    return fem, _make_gauss_slab(
        component, np.asarray(slab.values)[:, sel],
        np.asarray(slab.element_index)[sel],
        np.asarray(slab.natural_coords)[sel],
    )


def _corner_truth(fem, corner_nids, axis: int) -> np.ndarray:
    pos = {int(n): k for k, n in enumerate(fem.nodes.ids)}
    return np.asarray(fem.nodes.coords)[
        [pos[int(n)] for n in corner_nids], axis]


def _group_code(fem, eid: int) -> int:
    for g in fem.elements:
        if eid in {int(i) for i in g.ids}:
            return int(g.element_type.code)
    raise KeyError(eid)


@pytest.mark.parametrize("ext", ["ladruno", "mpco"])
@pytest.mark.parametrize("stem,eid,cls", [
    ("tri", 2, "SixNodeTri"),
    ("tri", 3, "BezierTri6"),
    ("tri", 4, "LadrunoLST"),
    ("tet", 2, "TenNodeTetrahedron"),
    ("tet", 3, "BezierTet10"),
])
def test_p2_simplex_on_the_files_own_fem_uses_the_gauss_positions(
    ext: str, stem: str, eid: int, cls: str,
) -> None:
    """With the file's own FEMData each P2 simplex corner gets its own
    coordinate, on both paths, with no degradation warning. Before #2240
    every corner got the element mean (an error of 1.3 on these)."""
    path = _SIMPLEX_GP / f"{stem}.{ext}"
    n_corner = 3 if stem == "tri" else 4
    for axis, comp in enumerate(_AXES[stem]):
        fem, slab = _file_fem_and_slab(path, f"strain_{comp}", eid)
        assert _group_code(fem, eid) < 0, "not the file-coded FEMData"
        with warnings.catch_warnings():
            warnings.simplefilter("error", WarnGaussExtrapolationDegraded)
            per = extrapolate_gauss_slab_per_element(slab, fem)
            acc = GaussToNodeAccumulator(
                slab.element_index, slab.natural_coords, fem,
            )
        (corners,) = per.corner_node_ids
        assert corners.size == n_corner, f"{cls}: mid-side nodes kept"
        np.testing.assert_allclose(
            per.values[0][-1], _corner_truth(fem, corners, axis), rtol=0,
            atol=1e-10, err_msg=f"{cls} {comp}",
        )
        np.testing.assert_allclose(
            acc.accumulate(np.asarray(slab.values)[-1]),
            _corner_truth(fem, acc.node_ids, axis), rtol=0, atol=1e-10,
            err_msg=f"{cls} {comp} (accumulator)",
        )


def _global_gp(path: Path, eid: int, n_gp: int) -> np.ndarray:
    """The fork's ``GLOBAL_GP_COORDS`` for one element, ``(n_gp, 3)``."""
    with h5py.File(path, "r") as f:
        for grp in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].values():
            if int(grp["CONNECTIVITY"][0, 0]) == eid:
                g = np.asarray(grp["GLOBAL_GP_COORDS"][0], dtype=np.float64)
                return g.reshape(n_gp, -1)
    raise KeyError(eid)


@pytest.mark.parametrize("stem,eid,cls,n_corner", [
    ("tri", 5, "FourNodeQuad", 4),
    ("tet", 4, "Brick", 8),
])
def test_bilinear_class_on_the_files_own_fem_uses_the_gauss_positions(
    stem: str, eid: int, cls: str, n_corner: int,
) -> None:
    """The fork's GP world positions as the field. A world coordinate is
    (tri)linear in the natural coordinates, so each corner gets its own.
    The file-coded quad and brick were averaged to the mean as well."""
    path = _SIMPLEX_GP / f"{stem}.ladruno"
    fem, slab = _file_fem_and_slab(path, "stress_xx", eid)
    gp_world = _global_gp(path, eid, slab.element_index.size)
    for axis in range(len(_AXES[stem])):
        one = _make_gauss_slab("x", gp_world[None, :, axis],
                               slab.element_index, slab.natural_coords)
        with warnings.catch_warnings():
            warnings.simplefilter("error", WarnGaussExtrapolationDegraded)
            per = extrapolate_gauss_slab_per_element(one, fem)
        (corners,) = per.corner_node_ids
        assert corners.size == n_corner
        np.testing.assert_allclose(
            per.values[0][0], _corner_truth(fem, corners, axis), rtol=0,
            atol=1e-10, err_msg=f"{cls} axis {axis}",
        )


def _file_coded_fem(name: str, code: int, npe: int, dim: int = 3):
    """One file-coded element (code = -class_tag) with id 7, nodes 1..npe."""
    tags = np.arange(1, npe + 1, dtype=np.int64)

    class _Group:
        element_type = SimpleNamespace(code=code, gmsh_name=name,
                                       dim=dim, npe=npe)
        ids = np.array([7], dtype=np.int64)
        connectivity = tags[None, :]

        def __len__(self) -> int:
            return 1

    return SimpleNamespace(
        nodes=SimpleNamespace(ids=tags, coords=np.zeros((npe, 3))),
        elements=[_Group()],
    )


_TRI_GL2 = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])


@pytest.mark.parametrize("nat", [
    np.full((3, 2), np.nan),
    np.zeros((3, 0)),
    np.array([[0.1, np.nan], [0.2, 0.3], [0.4, 0.5]]),
], ids=["all-nan", "no-columns", "partly-nan"])
def test_file_coded_element_without_gp_positions_warns_by_name(
    nat: np.ndarray,
) -> None:
    """Rows with no usable natural coordinates (a reader joined two widths
    and blanked them, or none were recorded) cannot be extrapolated: the
    mean is taken and the class is named, on both paths."""
    fem = _file_coded_fem("SixNodeTri", -209, 6)
    slab = _make_gauss_slab("x", np.array([[1.0, 2.0, 3.0]]),
                            np.full(3, 7), nat)
    with pytest.warns(WarnGaussExtrapolationDegraded,
                      match=r"SixNodeTri \(code -209\).*no natural "
                            r"coordinates"):
        per = extrapolate_gauss_slab_per_element(slab, fem)
    np.testing.assert_allclose(per.values[0][0], 2.0)
    with pytest.warns(WarnGaussExtrapolationDegraded, match="SixNodeTri"):
        GaussToNodeAccumulator(slab.element_index, slab.natural_coords, fem)


def test_file_coded_topology_without_shape_functions_warns_by_name() -> None:
    """A 13-node class with 3-D Gauss points (a P2 pyramid) has no shape
    functions here: warned by name, never a silent mean."""
    fem = _file_coded_fem("SomePyramid13", -999, 13)
    nat = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.3]])
    slab = _make_gauss_slab("x", np.array([[1.0, 3.0]]), np.full(2, 7), nat)
    with pytest.warns(WarnGaussExtrapolationDegraded,
                      match=r"SomePyramid13 \(code -999\).*13-node element "
                            r"with 3-D Gauss points"):
        extrapolate_gauss_slab_per_element(slab, fem)


def test_gmsh_type_without_shape_functions_warns_by_code() -> None:
    """The same rule on a gmsh-coded FEMData (a 5-node pyramid, type 7)."""
    nodes = [(i + 1, [0.0, 0.0, 0.0]) for i in range(5)]
    fem = _fake_fem(nodes, [(11, 7, [1, 2, 3, 4, 5])])
    nat = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.3]])
    slab = _make_gauss_slab("x", np.array([[1.0, 3.0]]), np.full(2, 11), nat)
    with pytest.warns(WarnGaussExtrapolationDegraded,
                      match="gmsh element type 7"):
        extrapolate_gauss_slab_to_nodes(slab, fem)


def test_single_gp_file_coded_element_is_exact_and_silent() -> None:
    """One GP is a constant: no coordinates are needed, nothing warns, and
    a P2 tri still paints only its three corners."""
    fem = _file_coded_fem("SixNodeTri", -209, 6)
    slab = _make_gauss_slab("x", np.array([[4.2]]), np.array([7]),
                            np.array([[1 / 3, 1 / 3]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussExtrapolationDegraded)
        per = extrapolate_gauss_slab_per_element(slab, fem)
    assert per.corner_node_ids[0].tolist() == [1, 2, 3]
    np.testing.assert_allclose(per.values[0][0], 4.2)


_HEX_GL2 = np.array([[a, b, c] for c in (-_INV_SQRT3, _INV_SQRT3)
                     for b in (-_INV_SQRT3, _INV_SQRT3)
                     for a in (-_INV_SQRT3, _INV_SQRT3)])


@pytest.mark.parametrize("npe", [4, 8], ids=["quad4-as-tet4",
                                              "quad8-as-hex8"])
def test_file_coded_rows_wider_than_the_group_dim_warn_by_name(
    npe: int,
) -> None:
    """A 2-D group with 3-wide Gauss rows contradicts itself. Read by the
    rows alone, a 4-node quad becomes a Tet4 and an 8-node quad a Hex8, and
    the corners came back wrong (``[1, 2, 1, 1]`` for ``x = [0, 2, 2, 0]``),
    silently. The shape cannot be told, so it warns by name instead."""
    fem = _file_coded_fem("SomeQuad", -777, npe, dim=2)
    fem.nodes.coords[:4, 0] = [0.0, 2.0, 2.0, 0.0]
    gp_x = 1.0 + _HEX_GL2[:, 0]       # bilinear in x, the quad's own field
    slab = _make_gauss_slab("x", gp_x[None, :], np.full(8, 7), _HEX_GL2)
    with pytest.warns(WarnGaussExtrapolationDegraded,
                      match=r"SomeQuad \(code -777\).*3-D Gauss rows "
                            r"contradict its group's dim=2"):
        extrapolate_gauss_slab_per_element(slab, fem)
    with pytest.warns(WarnGaussExtrapolationDegraded, match="SomeQuad"):
        GaussToNodeAccumulator(slab.element_index, slab.natural_coords, fem)


def test_file_coded_p2_tri_resolves_by_its_gauss_width_not_group_dim() -> None:
    """The group says dim=3 (the class-name guess LadrunoLST gets from a
    ``TOPOLOGY "custom"`` file); its Gauss rows are 2-D, so it is a Tri6,
    not a 6-node wedge."""
    fem = _file_coded_fem("LadrunoLST", -33016, 6, dim=3)
    xy = np.array([[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]])
    fem.nodes.coords[:3, :2] = xy
    u, v = _TRI_GL2[:, 0], _TRI_GL2[:, 1]
    gp_x = (1 - u - v) * xy[0, 0] + u * xy[1, 0] + v * xy[2, 0]
    slab = _make_gauss_slab("x", gp_x[None, :], np.full(3, 7), _TRI_GL2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussExtrapolationDegraded)
        per = extrapolate_gauss_slab_per_element(slab, fem)
    np.testing.assert_allclose(per.values[0][0], xy[:, 0], atol=1e-12)
