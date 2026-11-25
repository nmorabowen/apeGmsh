"""``reduction="consistent"`` on gravity / volume loads is ``∫ N_i dΩ``.

It used to be the equal split for every element (the resolver returned
the tributary vector), which is wrong on every quadratic Lagrange
element and on a distorted hex8 / quad4 / wedge6. Three oracles, none of
which uses apeGmsh's shape functions or quadrature:

* exact rational integrals, from a basis built on gmsh's reference nodes
  (``_exact_weights``), on straight and on curved elements of every type
  the load walk accepts;
* gmsh's own basis functions and Jacobians, on a curved mesh;
* a P2 finite-element solve whose exact solution is quadratic, which a
  consistent vector reproduces at every node and the equal split does not.

The Bernstein side (ADR 0091) is the tributary vector, bit for bit.
"""
from __future__ import annotations

import warnings
from fractions import Fraction as F
from itertools import product
from math import factorial

import gmsh
import numpy as np
import pytest

from apeGmsh._kernel.defs.loads import BodyLoadDef, GravityLoadDef
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver
from apeGmsh.fem._quadrature import gauss_tet_15

# =====================================================================
# Exact rational integration
# =====================================================================
# A polynomial is {(a, b, c): coefficient} in the reference coordinates.

def _mul(p, q):
    out: dict = {}
    for (a, b, c), u in p.items():
        for (d, e, f), v in q.items():
            k = (a + d, b + e, c + f)
            out[k] = out.get(k, F(0)) + u * v
    return out


def _add(p, q, s=1):
    out = dict(p)
    for k, v in q.items():
        out[k] = out.get(k, F(0)) + s * v
    return out


def _diff(p, axis):
    out: dict = {}
    for k, v in p.items():
        if k[axis]:
            m = list(k)
            m[axis] -= 1
            out[tuple(m)] = out.get(tuple(m), F(0)) + v * k[axis]
    return out


def _line(k):
    return F(2, k + 1) if k % 2 == 0 else F(0)


def _integral(p, shape):
    tot = F(0)
    for (a, b, c), v in p.items():
        if shape == "tri":
            i = F(factorial(a) * factorial(b), factorial(a + b + 2))
        elif shape == "tet":
            i = F(factorial(a) * factorial(b) * factorial(c),
                  factorial(a + b + c + 3))
        elif shape == "wedge":
            i = F(factorial(a) * factorial(b),
                  factorial(a + b + 2)) * _line(c)
        elif shape == "quad":
            i = _line(a) * _line(b)
        else:
            i = _line(a) * _line(b) * _line(c)
        tot += v * i
    return tot


def _total(n, d):
    return [m for m in product(range(d + 1), repeat=3)
            if sum(m) <= d and (n == 3 or m[2] == 0)]


def _tensor(n, d):
    return [m for m in product(range(d + 1), repeat=3) if n == 3 or m[2] == 0]


def _serendipity(n):
    """Quadratic serendipity: superlinear degree at most 2."""
    return [m for m in _tensor(n, 2) if sum(k for k in m if k > 1) <= 2]


# gmsh code -> (id, dim, reference shape, monomial space)
_TYPES = {
    2:  ("tri3",   2, "tri",   _total(2, 1)),
    3:  ("quad4",  2, "quad",  _tensor(2, 1)),
    9:  ("tri6",   2, "tri",   _total(2, 2)),
    16: ("quad8",  2, "quad",  _serendipity(2)),
    10: ("quad9",  2, "quad",  _tensor(2, 2)),
    4:  ("tet4",   3, "tet",   _total(3, 1)),
    5:  ("hex8",   3, "hex",   _tensor(3, 1)),
    6:  ("wedge6", 3, "wedge", [m for m in _tensor(3, 1) if m[0] + m[1] <= 1]),
    11: ("tet10",  3, "tet",   _total(3, 2)),
    17: ("hex20",  3, "hex",   _serendipity(3)),
    12: ("hex27",  3, "hex",   _tensor(3, 2)),
}
_CODES = [pytest.param(c, id=t[0]) for c, t in _TYPES.items()]


def _reference_nodes(code):
    """gmsh's reference node coordinates, as exact fractions."""
    _, dim, _, n, local, _ = gmsh.model.mesh.getElementProperties(code)
    ref = [[F(float(v)).limit_denominator(64)
            for v in local[i * dim:(i + 1) * dim]] for i in range(n)]
    return [r + [F(0)] * (3 - dim) for r in ref]


def _basis(code):
    """Nodal basis polynomials: invert the Vandermonde matrix exactly."""
    ref, space = _reference_nodes(code), _TYPES[code][3]
    n = len(ref)
    assert len(space) == n
    rows = [[x[0]**m[0] * x[1]**m[1] * x[2]**m[2] for m in space]
            + [F(int(i == j)) for j in range(n)] for i, x in enumerate(ref)]
    for k in range(n):
        piv = next(r for r in range(k, n) if rows[r][k] != 0)
        rows[k], rows[piv] = rows[piv], rows[k]
        inv = 1 / rows[k][k]
        rows[k] = [v * inv for v in rows[k]]
        for r in range(n):
            if r != k and rows[r][k] != 0:
                f = rows[r][k]
                rows[r] = [v - f * w for v, w in zip(rows[r], rows[k])]
    return [{m: rows[a][n + i] for a, m in enumerate(space) if rows[a][n + i]}
            for i in range(n)]


def _exact_weights(code, coords):
    """``∫ N_i dΩ`` as fractions, for exact node coordinates."""
    _, dim, shape, _ = _TYPES[code]
    N = _basis(code)
    x: list[dict] = [{}, {}, {}]
    for Ni, xi in zip(N, coords):
        for d in range(dim):
            x[d] = _add(x[d], {m: v * xi[d] for m, v in Ni.items()})
    J = [[_diff(x[d], k) for k in range(dim)] for d in range(dim)]
    if dim == 2:
        det = _add(_mul(J[0][0], J[1][1]), _mul(J[0][1], J[1][0]), -1)
    else:
        det = {}
        for (a, b, c), s in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                             ((0, 2, 1), -1), ((1, 0, 2), -1),
                             ((2, 1, 0), -1)):
            det = _add(det, _mul(_mul(J[0][a], J[1][b]), J[2][c]), s)
    return [_integral(_mul(Ni, det), shape) for Ni in N]


def _coords(code, *, curved):
    """Physical nodes: an affine image, plus a deterministic distortion."""
    dim = _TYPES[code][1]
    out = []
    for i, r in enumerate(_reference_nodes(code)):
        x = [2 * r[0] + F(1, 2) * r[1] + 1,
             F(3, 2) * r[1] + F(1, 4) * r[2] - 2,
             r[2] + F(1, 8) * r[0] if dim == 3 else F(0)]
        if curved:
            for d in range(dim):
                x[d] += F((7 * i * i + 5 * d + 3 * i * d) % 11 - 5, 64)
        out.append(x)
    return out


def _resolver(coords):
    n = len(coords)
    return LoadResolver(np.arange(1, n + 1),
                        np.array([[float(v) for v in c] for c in coords]))


def _forces(recs, n):
    out = np.zeros((n, 3))
    for r in recs:
        out[r.node_id - 1] += r.force_xyz
    return out


_B = np.array([3.0, -5.0, 7.0])


def _consistent(kind, coords, dim, **kw):
    conn = [np.arange(1, len(coords) + 1)]
    r = _resolver(coords)
    if kind == "gravity":
        d = GravityLoadDef(target="x", density=2.0, g=tuple(_B / 2.0),
                           reduction="consistent", **kw)
        return r.resolve_gravity_consistent(d, conn, dim)
    d = BodyLoadDef(target="x", force_per_volume=tuple(_B),
                    reduction="consistent", **kw)
    return r.resolve_body_consistent(d, conn, dim)


@pytest.mark.parametrize("kind", ["gravity", "volume"])
@pytest.mark.parametrize("curved", [False, True], ids=["affine", "curved"])
@pytest.mark.parametrize("code", _CODES)
def test_matches_the_exact_integral(g, code, curved, kind):
    coords = _coords(code, curved=curved)
    exact = np.array([float(w) for w in _exact_weights(code, coords)])
    vol = exact.sum()
    assert vol > 0

    got = _forces(_consistent(kind, coords, _TYPES[code][1]), len(coords))

    np.testing.assert_allclose(got, np.outer(exact, _B),
                               atol=2e-13 * vol * np.abs(_B).max(), rtol=0)
    if curved and code not in (2, 4):
        # The curved case must tell the two reductions apart, or it
        # proves nothing about the equal split.
        assert np.abs(exact - vol / len(exact)).max() > 1e-3 * vol


# Shares of a straight-sided element's weight, as published:
# one entry per group of nodes, (how many, share).
_SHARES = {
    2:  [(3, F(1, 3))],
    3:  [(4, F(1, 4))],
    9:  [(3, F(0)), (3, F(1, 3))],
    16: [(4, F(-1, 12)), (4, F(1, 3))],
    10: [(4, F(1, 36)), (4, F(1, 9)), (1, F(4, 9))],
    4:  [(4, F(1, 4))],
    5:  [(8, F(1, 8))],
    6:  [(6, F(1, 6))],
    11: [(4, F(-1, 20)), (6, F(1, 5))],
    17: [(8, F(-1, 8)), (12, F(1, 6))],
    12: [(8, F(1, 216)), (12, F(1, 54)), (6, F(2, 27)), (1, F(8, 27))],
}


@pytest.mark.parametrize("code", _CODES)
def test_closed_form_shares_of_a_straight_element(g, code):
    coords = _coords(code, curved=False)
    share = np.array([float(s) for n, s in _SHARES[code] for _ in range(n)])
    got = _forces(_consistent("volume", coords, _TYPES[code][1]), len(coords))
    total = got[:, 0].sum()
    np.testing.assert_allclose(got[:, 0], share * total,
                               atol=1e-13 * abs(total))


def test_tet_rule_is_exact_to_degree_five():
    P, W = gauss_tet_15()
    for a, b, c in _total(3, 5):
        exact = factorial(a) * factorial(b) * factorial(c) / factorial(a+b+c+3)
        got = float(np.sum(W * P[:, 0]**a * P[:, 1]**b * P[:, 2]**c))
        assert got == pytest.approx(exact, rel=1e-13)


# =====================================================================
# Through the session, on real meshes
# =====================================================================

def _nodal(fem):
    out: dict[int, np.ndarray] = {}
    for r in fem.nodes.loads:
        out[int(r.node_id)] = (out.get(int(r.node_id), np.zeros(3))
                               + np.asarray(r.force_xyz))
    return out


def _gmsh_weights(dim):
    """Per-node ``∫ N_i dΩ`` from gmsh's own basis and Jacobians."""
    out: dict[int, float] = {}
    for et in gmsh.model.mesh.getElementTypes(dim):
        _, conn = gmsh.model.mesh.getElementsByType(et)
        pts, w = gmsh.model.mesh.getIntegrationPoints(et, "Gauss10")
        _, N, _ = gmsh.model.mesh.getBasisFunctions(et, pts, "Lagrange")
        _, det, _ = gmsh.model.mesh.getJacobians(et, pts)
        ng = len(w)
        N = np.asarray(N).reshape(ng, -1)
        det = np.asarray(det).reshape(-1, ng)
        conn = np.asarray(conn).reshape(len(det), -1)
        for row, we in zip(conn, (np.abs(det) * w) @ N):
            for nid, v in zip(row, we):
                out[int(nid)] = out.get(int(nid), 0.0) + float(v)
    return out


@pytest.mark.parametrize("verb", ["gravity", "volume"])
def test_curved_tet10_mesh_against_gmsh(g, verb):
    g.model.geometry.add_sphere(0.0, 0.0, 0.0, 1.0, label="ball")
    g.mesh.sizing.set_global_size(0.7)
    g.mesh.generation.generate(3)
    g.mesh.generation.set_order(2, bubble=False)
    assert {int(t) for t in gmsh.model.mesh.getElementTypes(3)} == {11}
    with g.loads.case("c"):
        if verb == "gravity":
            g.loads.gravity("ball", density=2.0, g=(0.0, 0.0, -9.81),
                            reduction="consistent")
        else:
            g.loads.volume("ball", force_per_volume=(0.0, 0.0, -19.62),
                           reduction="consistent")
    want = _gmsh_weights(3)
    got = _nodal(g.mesh.queries.get_fem_data())

    vol = sum(want.values())
    assert vol == pytest.approx(4 * np.pi / 3, rel=0.02)
    for nid, w in want.items():
        fz = got.get(nid, np.zeros(3))[2]
        assert fz == pytest.approx(-19.62 * w, abs=1e-11 * vol * 19.62)


def _tet10_box(g):
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="soil")
    g.physical.add(3, "soil", name="soil")
    g.mesh.sizing.set_global_size(0.7)
    g.mesh.generation.generate(3)
    g.mesh.generation.set_order(2, bubble=False)


def test_consistent_load_gives_the_exact_p2_solution(g):
    """``-E u'' = -γ`` on a tet10 box fixed at z = 0, free elsewhere.

    The exact ``u = γ(z² - 2z)/(2E)`` is in the P2 space, so Galerkin
    with the consistent vector returns it at every node. The equal split
    (same resultant) does not.
    """
    _tet10_box(g)
    E, gamma = 5.0, 3.0
    for case, red in (("c", "consistent"), ("t", "tributary")):
        with g.loads.case(case):
            g.loads.gravity("soil", density=1.0, g=(0.0, 0.0, -gamma),
                            reduction=red)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(xyz).reshape(-1, 3)
    idx = {int(t): i for i, t in enumerate(tags)}
    _, conn = gmsh.model.mesh.getElementsByType(11)
    conn = np.asarray(conn).reshape(-1, 10)
    pts, w = gmsh.model.mesh.getIntegrationPoints(11, "Gauss4")
    _, dN, _ = gmsh.model.mesh.getBasisFunctions(11, pts, "GradLagrange")
    jac, det, _ = gmsh.model.mesh.getJacobians(11, pts)
    ng = len(w)
    dN = np.asarray(dN).reshape(ng, 10, 3)
    jac = np.asarray(jac).reshape(-1, ng, 3, 3)      # rows: d/du, d/dv, d/dw
    det = np.asarray(det).reshape(-1, ng)
    K = np.zeros((len(tags), len(tags)))
    for e, row in enumerate(conn):
        ii = [idx[int(n)] for n in row]
        for q in range(ng):
            G = dN[q] @ np.linalg.inv(jac[e, q]).T
            K[np.ix_(ii, ii)] += E * w[q] * det[e, q] * (G @ G.T)
    free = xyz[:, 2] > 1e-9
    exact = gamma * (xyz[:, 2]**2 - 2 * xyz[:, 2]) / (2 * E)
    fem = g.mesh.queries.get_fem_data()
    err = {}
    for case in ("c", "t"):
        f = np.zeros(len(tags))
        for r in fem.nodes.loads.by_pattern(case):
            f[idx[int(r.node_id)]] += r.force_xyz[2]
        assert f.sum() == pytest.approx(-gamma)
        u = np.linalg.solve(K[np.ix_(free, free)], f[free])
        err[case] = np.abs(u - exact[free]).max() / np.abs(exact).max()
    assert err["c"] < 1e-10
    assert err["t"] > 1e-3


# =====================================================================
# What must not change: tributary, and the Bézier vector
# =====================================================================

def test_tributary_tet10_is_still_one_tenth_each(g):
    coords = _coords(11, curved=False)
    r = _resolver(coords)
    conn = [np.arange(1, 11)]
    d = GravityLoadDef(target="x", density=2.0, g=tuple(_B / 2.0))
    recs = r.resolve_gravity_tributary(d, conn, 3)
    V = float(sum(_exact_weights(11, coords)))
    np.testing.assert_allclose(_forces(recs, 10), np.tile(_B * V / 10, (10, 1)),
                               rtol=1e-13)
    assert all(rec.basis is None for rec in recs)


@pytest.mark.parametrize("kind", ["gravity", "volume"])
@pytest.mark.parametrize("code", [9, 11], ids=["tri6", "tet10"])
def test_bernstein_is_the_tributary_vector_bit_for_bit(g, code, kind):
    coords, dim = _coords(code, curved=False), _TYPES[code][1]
    n = len(coords)
    r, conn = _resolver(coords), [np.arange(1, n + 1)]
    if kind == "gravity":
        t = r.resolve_gravity_tributary(GravityLoadDef(
            target="x", density=2.0, g=tuple(_B / 2.0)), conn, dim)
    else:
        t = r.resolve_body_tributary(BodyLoadDef(
            target="x", force_per_volume=tuple(_B)), conn, dim)
    b = _consistent(kind, coords, dim, basis="bernstein")
    assert np.array_equal(_forces(b, n), _forces(t, n))
    assert all(rec.basis == "bernstein" for rec in b)


@pytest.mark.parametrize("kind", ["gravity", "volume"])
@pytest.mark.parametrize("code", [3, 16, 10, 5, 6, 17, 12])
def test_bernstein_refuses_a_non_simplex(g, code, kind):
    coords = _coords(code, curved=False)
    with pytest.raises(NotImplementedError, match="simplex"):
        _consistent(kind, coords, _TYPES[code][1], basis="bernstein")


@pytest.mark.parametrize("kind", ["gravity", "volume"])
@pytest.mark.parametrize("code", _CODES)
def test_records_carry_the_basis_only_where_it_matters(g, code, kind):
    recs = _consistent(kind, _coords(code, curved=False), _TYPES[code][1])
    linear = code in (2, 3, 4, 5, 6)
    assert recs
    assert all(rec.basis == (None if linear else "lagrange") for rec in recs)


def test_unknown_cell_width_is_refused_by_count():
    coords = [[F(i), F(i * i), F(i**3)] for i in range(13)]    # pyramid13
    with pytest.raises(NotImplementedError, match="13-node cells"):
        _consistent("volume", coords, 3)


def test_consistent_gravity_needs_a_density():
    d = GravityLoadDef(target="x", density=None, reduction="consistent")
    with pytest.raises(ValueError, match="density"):
        _resolver(_coords_tet4()).resolve_gravity_consistent(
            d, [np.arange(1, 5)], 3)


def _coords_tet4():
    return [[F(0)] * 3, [F(1), F(0), F(0)], [F(0), F(1), F(0)],
            [F(0), F(0), F(1)]]


@pytest.mark.parametrize("verb", ["gravity", "volume"])
@pytest.mark.parametrize("basis, element, warns", [
    ("lagrange",  "TenNodeTetrahedron", None),
    ("lagrange",  "BezierTet10",        "bernstein"),
    ("bernstein", "BezierTet10",        None),
    ("bernstein", "TenNodeTetrahedron", "nodal-value"),
])
def test_build_warns_when_the_basis_mismatches(g, verb, basis, element, warns):
    """The bridge guard (ADR 0091) sees body forces too."""
    from apeGmsh.opensees import apeSees
    from apeGmsh.opensees._internal.build.validators.loads import (
        WarnLoadBasisMismatch,
    )
    from apeGmsh.opensees.emitter.recording import RecordingEmitter
    from apeGmsh.opensees.material.nd import ElasticIsotropic

    _tet10_box(g)
    with g.loads.case("dead"):
        if verb == "gravity":
            g.loads.gravity("soil", density=2.0, reduction="consistent",
                            basis=basis)
        else:
            g.loads.volume("soil", force_per_volume=(0.0, 0.0, -20.0),
                           reduction="consistent", basis=basis)
    fem = g.mesh.queries.get_fem_data(dim=3)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25))
    getattr(ops.element, element)(pg="soil", material=mat)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.from_model("dead")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops.build().emit(RecordingEmitter())
    hits = [str(w.message) for w in caught
            if issubclass(w.category, WarnLoadBasisMismatch)]
    if warns is None:
        assert hits == []
    else:
        assert len(hits) == 1 and warns in hits[0]


@pytest.mark.parametrize("verb", ["gravity", "volume"])
def test_bernstein_needs_the_consistent_reduction(g, verb):
    kw = {"density": 1.0} if verb == "gravity" else {}
    with pytest.raises(ValueError, match="consistent nodal"):
        getattr(g.loads, verb)("x", basis="bernstein", **kw)
