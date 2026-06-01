"""Neutral FE basis front-door :mod:`apeGmsh._basis` (recorder plan L2b-2).

Two kinds of check:

* **Math properties** — partition of unity for every family, and (for
  lagrange) Kronecker-delta interpolation at the catalog node coords —
  family-internal, no fixture.
* **Self-describing reconstruction** — evaluate ``B(ξ)`` at a fixture's
  own ``GP_PARAM`` and confirm ``x = B @ X`` reproduces the file's
  ``GLOBAL_GP_COORDS`` to machine precision. This proves the front-door
  matches real fork output (and de-risks the bezier read path, which
  imports the same evaluator).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._basis import BasisError, basis_values

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ladruno"


def _decode(v):
    if isinstance(v, np.ndarray):
        v = v.flat[0]
    return v.decode() if isinstance(v, (bytes, np.bytes_)) else v


# ---------------------------------------------------------------------------
# Partition of unity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "topology, family, order, dim, simplex",
    [
        ("line", "lagrange", 1, 1, False),
        ("quad", "lagrange", 1, 2, False),
        ("quad", "lagrange", 2, 2, False),
        ("hex", "lagrange", 1, 3, False),
        ("hex", "lagrange", 2, 3, False),
        ("tri", "bernstein", 2, 3, True),
        ("tet", "bernstein", 2, 4, True),
        ("quad", "bernstein", 2, 2, False),
    ],
)
def test_partition_of_unity(topology, family, order, dim, simplex) -> None:
    rng = np.random.default_rng(0)
    pts = rng.random((25, dim))
    if simplex:
        pts /= pts.sum(axis=1, keepdims=True)        # barycentric
    elif family == "lagrange":
        pts = pts * 2.0 - 1.0                         # [-1, 1]
    # bernstein tensor lives on [0, 1] — leave as-is.
    R = basis_values(topology=topology, family=family, order=order, xi=pts)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-13)


def test_bernstein_tri6_control_count_and_order() -> None:
    # At vertex ξ=(0,0,1) the pinned P1=ξ₃² is the only nonzero (=1).
    R = basis_values(
        topology="tri", family="bernstein", order=2,
        xi=np.array([[0.0, 0.0, 1.0]]),
    )
    assert R.shape == (1, 6)
    np.testing.assert_allclose(R[0], [1, 0, 0, 0, 0, 0], atol=1e-13)


def test_bernstein_tri6_matches_reference_formulas() -> None:
    # Reference BezierTri6 (bezierFEM), free area coords (ξ₁, ξ₂):
    #   N = [ξ₃², ξ₁², ξ₂², 2ξ₁ξ₃, 2ξ₁ξ₂, 2ξ₂ξ₃],  ξ₃ = 1−ξ₁−ξ₂.
    rng = np.random.default_rng(1)
    free = rng.random((30, 2)) * 0.5            # free coords (ξ₁, ξ₂)
    R = basis_values(topology="tri", family="bernstein", order=2, xi=free)
    a, b = free[:, 0], free[:, 1]
    c = 1.0 - a - b
    ref = np.stack([c * c, a * a, b * b, 2 * a * c, 2 * a * b, 2 * b * c], axis=1)
    np.testing.assert_allclose(R, ref, atol=1e-13)


def test_bernstein_tet10_matches_reference_with_gmsh_swap() -> None:
    # Reference BezierTet10 (bezierFEM), free volume coords (L1, L2, L3):
    # edges (1-2, 2-3, 1-3, 1-4, 3-4, 2-4) — the Larenas N9↔N10 Gmsh swap.
    rng = np.random.default_rng(2)
    free = rng.random((30, 3)) * 0.3            # free coords (L1, L2, L3)
    R = basis_values(topology="tet", family="bernstein", order=2, xi=free)
    l1, l2, l3 = free[:, 0], free[:, 1], free[:, 2]
    l4 = 1.0 - l1 - l2 - l3
    ref = np.stack([
        l1 * l1, l2 * l2, l3 * l3, l4 * l4,
        2 * l1 * l2, 2 * l2 * l3, 2 * l1 * l3,
        2 * l1 * l4, 2 * l3 * l4, 2 * l2 * l4,   # N8, N9(3-4), N10(2-4)
    ], axis=1)
    np.testing.assert_allclose(R, ref, atol=1e-13)
    # Regression guard for the swap: N9 must be the (3-4) edge, N10 the
    # (2-4) edge — pick a point where L2 ≠ L3 so the two differ.
    pt = np.array([[0.1, 0.2, 0.4]])             # L4 = 0.3
    r = basis_values(topology="tet", family="bernstein", order=2, xi=pt)[0]
    np.testing.assert_allclose(r[8], 2 * 0.4 * 0.3)   # N9  = 2·L3·L4
    np.testing.assert_allclose(r[9], 2 * 0.2 * 0.3)   # N10 = 2·L2·L4


def test_bernstein_simplex_accepts_full_barycentric() -> None:
    # 2-col free and 3-col full barycentric agree for the triangle.
    free = np.array([[0.2, 0.3]])
    full = np.array([[0.2, 0.3, 0.5]])
    r_free = basis_values(topology="tri", family="bernstein", order=2, xi=free)
    r_full = basis_values(topology="tri", family="bernstein", order=2, xi=full)
    np.testing.assert_allclose(r_free, r_full, atol=1e-14)


def test_lagrange_quad4_kronecker_delta() -> None:
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    R = basis_values(topology="quad", family="lagrange", order=1, xi=corners)
    np.testing.assert_allclose(R, np.eye(4), atol=1e-13)


def test_unknown_family_raises() -> None:
    with pytest.raises(BasisError, match="family"):
        basis_values(topology="quad", family="hermite", order=1,
                     xi=np.zeros((1, 2)))


def test_unknown_lagrange_topology_order_raises() -> None:
    with pytest.raises(BasisError, match="no lagrange basis"):
        basis_values(topology="pyramid", family="lagrange", order=1,
                     xi=np.zeros((1, 3)))


# ---------------------------------------------------------------------------
# Self-describing reconstruction against fork fixtures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fixture, class_key",
    [
        ("truss2d.ladruno", "12-Truss[1:0]"),          # line2 / lagrange
        ("quad2d.ladruno", "31-FourNodeQuad[201:0]"),  # quad4 / lagrange
    ],
)
def test_reconstructs_global_gp_coords(fixture, class_key) -> None:
    import h5py

    with h5py.File(FIXTURES / fixture, "r") as f:
        me = f["MODEL_STAGE[1]/MODEL/ELEMENTS"][class_key]
        topo = _decode(me.attrs["TOPOLOGY"])
        fam = _decode(me.attrs["FAMILY"])
        order = me.attrs["ORDER"]
        conn = np.asarray(me["CONNECTIVITY"][...], dtype=np.int64)
        gp = np.asarray(me["QUADRATURE"]["GP_PARAM"][...], dtype=np.float64)
        ggc = np.asarray(me["GLOBAL_GP_COORDS"][...], dtype=np.float64)
        nodes = f["MODEL_STAGE[1]/MODEL/NODES"]
        nid = np.asarray(nodes["ID"][...], dtype=np.int64).flatten()
        coords = np.asarray(nodes["COORDINATES"][...], dtype=np.float64)

    id2row = {int(n): i for i, n in enumerate(nid)}
    R = basis_values(topology=topo, family=fam, order=order, xi=gp)
    sdim = coords.shape[1]
    for e in range(conn.shape[0]):
        ctrl = conn[e, 1:]                          # skip the element tag
        Xe = np.stack([coords[id2row[int(c)]] for c in ctrl])
        recon = R @ Xe                              # (nGP, sdim)
        ref = ggc[e].reshape(-1, sdim)
        np.testing.assert_allclose(recon, ref, atol=1e-10)


def test_reconstructs_bezier_tri6_gp_world_coords() -> None:
    """The fork's BezierTri6 writes NO ``GLOBAL_GP_COORDS`` — the reader must
    reconstruct GP world coords via ``B(ξ)`` from the file's bernstein BASIS
    + free-coord ``GP_PARAM``. This locks the reader's ``GP_PARAM``/
    ``CONNECTIVITY``/``COORDINATES`` **plumbing** against real fork output.

    NB — degeneracy: the fixture is **straight-sided** (mid-edge nodes at the
    edge midpoints), so the element is affine and ``P = X``. On affine
    geometry the Bernstein and Lagrange maps coincide, so this 0.0 check is
    *not* a proof of the Bernstein basis (a Lagrange basis would pass too) —
    that proof is ``test_bernstein_tri6_matches_reference_formulas`` at
    interior points. A genuinely curved Bézier element (``P ≠ X``) is
    unreachable in apeGmsh's straight-sided Gmsh pipeline."""
    import h5py

    with h5py.File(FIXTURES / "bezier_tri6.ladruno", "r") as f:
        me = f["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        key = next(iter(me))
        g = me[key]
        assert "GLOBAL_GP_COORDS" not in g          # bezier omits it
        assert _decode(g.attrs["FAMILY"]) == "bernstein"
        topo = _decode(g.attrs["TOPOLOGY"])
        order = g.attrs["ORDER"]
        conn = np.asarray(g["CONNECTIVITY"][...], dtype=np.int64)
        gp = np.asarray(g["QUADRATURE"]["GP_PARAM"][...], dtype=np.float64)
        nodes = f["MODEL_STAGE[1]/MODEL/NODES"]
        nid = np.asarray(nodes["ID"][...], dtype=np.int64).flatten()
        coords = np.asarray(nodes["COORDINATES"][...], dtype=np.float64)

    id2row = {int(n): i for i, n in enumerate(nid)}
    R = basis_values(topology=topo, family="bernstein", order=order, xi=gp)
    np.testing.assert_allclose(R.sum(axis=1), 1.0, atol=1e-13)   # PoU
    Xe = np.stack([coords[id2row[int(c)]] for c in conn[0, 1:]])
    recon = R @ Xe
    # GP_PARAM (1/6,1/6),(2/3,1/6),(1/6,2/3) on the right triangle → these.
    expected = np.array([[1/3, 1/3], [4/3, 1/3], [1/3, 4/3]])
    np.testing.assert_allclose(recon, expected, atol=1e-12)
