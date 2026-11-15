"""Lattice seeding from FEMData alone (ADR 0134 D2), against closed forms.

Every expected particle set here is computed independently of the seeder: the full lattice
of sub-cell centres, filtered by the analytic inside-predicate of the body's region (the
region of a straight-edged element is its polygon / polyhedron, so the predicate is exact).
The lattice coordinates are dyadic, so ties with a boundary are exact and named per test.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from apeGmsh.opensees._mpm import MPMBody, MPMGrid, MPMSeedError, seed_lattice

from . import _meshes as M

FAMILY_DIM = {"tri3": 2, "quad4": 2, "tet4": 3, "hex8": 3}


def full_lattice(grid: MPMGrid, n: int) -> np.ndarray:
    """Every sub-cell centre, x fastest, then y, then z (Contract-1 order)."""
    spacing = np.asarray(grid.cell_size) / n
    axes = [grid.origin[d] + (np.arange(grid.num_cells[d] * n) + 0.5) * spacing[d]
            for d in range(grid.ndm)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel(order="F") for m in mesh], axis=1)


def box_mesh(family: str, per_axis: int = 2, lo: float = 0.0, hi: float = 1.0):
    d = FAMILY_DIM[family]
    nodes, ids = M.grid_nodes([per_axis] * d, [lo] * d, [hi] * d)
    if family == "tri3":
        return nodes, M.tris(ids)
    if family == "quad4":
        return nodes, M.quads(ids)
    if family == "hex8":
        return nodes, M.hexes(ids)
    return nodes, M.tets(ids, nodes)


def seed(fem, grid, bodies, n, **kw):
    if grid.ndm == 2:
        kw.setdefault("thickness", 1.0)
    return seed_lattice(fem, grid, bodies, n=n, **kw)


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("family", list(FAMILY_DIM))
def test_body_filling_the_grid_gets_every_lattice_point_once(family, n):
    """A body that fills a 4^d grid of 0.25 cells: (4n)^d particles, n^d in every cell,
    volume (h/n)^d each (x 2.5 thickness in 2-D), half-length h/(2n), the body's density.
    The tri3/tet4 diagonals pass through lattice points: each is kept once."""
    d = FAMILY_DIM[family]
    nodes, els = box_mesh(family)
    fem = M.make_fem(nodes, {"rock": els})
    grid = MPMGrid((0.0,) * d, (0.25,) * d, (4,) * d)
    t = 2.5 if d == 2 else None
    s = seed_lattice(fem, grid, [MPMBody("rock", "rock", 3.0)], n=n, thickness=t)

    np.testing.assert_array_equal(s.position, full_lattice(grid, n))
    assert s.num_particles == (4 * n) ** d
    np.testing.assert_array_equal(s.report.counts, np.full((4,) * d, n ** d))
    assert s.report.expected == n ** d and s.report.partial_cells == 0
    vol = (0.25 / n) ** d * (t if t else 1.0)
    np.testing.assert_allclose(s.volume, vol, rtol=1e-15)
    np.testing.assert_allclose(s.volume.sum(), 1.0 * (t if t else 1.0), rtol=1e-12)
    np.testing.assert_allclose(s.gimp_half_length, 0.25 / (2 * n), rtol=1e-15)
    np.testing.assert_array_equal(s.density, 3.0)
    np.testing.assert_array_equal(s.body_id, 0)
    assert s.body_names == ("rock",)
    assert s.report.interface_points == {}


def _tri_pred(p):
    return (p[:, 0] >= -1e-12) & (p[:, 1] >= -1e-12) & (p[:, 0] + p[:, 1] <= 1 + 1e-12)


_QUAD = [(0.0, 0.0), (1.0, 0.1), (0.8, 0.9), (0.1, 0.7)]


def _quad_pred(p):
    ok = np.ones(len(p), dtype=bool)
    for a, b in zip(_QUAD, _QUAD[1:] + _QUAD[:1]):
        cross = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
        ok &= cross >= -1e-12
    return ok


def _tet_pred(p):
    return np.all(p >= -1e-12, axis=1) & (p.sum(axis=1) <= 1 + 1e-12)


def _frustum_pred(p):
    x, y, z = p.T
    return ((z >= 0) & (z <= 1) & (x >= 0.25 * z) & (x <= 1 - 0.25 * z)
            & (y >= 0.25 * z) & (y <= 1 - 0.25 * z))


SHAPES = {
    # family: (nodes, connectivity, predicate)
    "tri3": ({1: (0, 0), 2: (1, 0), 3: (0, 1)}, [1, 2, 3], _tri_pred),
    "quad4": (dict(zip(range(1, 5), _QUAD)), [1, 2, 3, 4], _quad_pred),
    "tet4": ({1: (0, 0, 0), 2: (1, 0, 0), 3: (0, 1, 0), 4: (0, 0, 1)}, [1, 2, 3, 4],
             _tet_pred),
    "hex8": ({1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
              5: (0.25, 0.25, 1), 6: (0.75, 0.25, 1), 7: (0.75, 0.75, 1), 8: (0.25, 0.75, 1)},
             list(range(1, 9)), _frustum_pred),
}


@pytest.mark.parametrize("family", list(SHAPES))
def test_kept_points_are_exactly_the_analytic_region(family):
    """One non-aligned element: the general quad and the frustum hex are not affine, so the
    Newton inverse map decides; the triangle's hypotenuse carries exact ties (kept)."""
    nodes, conn, pred = SHAPES[family]
    d = FAMILY_DIM[family]
    fem = M.make_fem(nodes, {"b": [(family, conn)]})
    grid = MPMGrid((0.0,) * d, (1 / 8,) * d, (8,) * d) if d == 2 else \
        MPMGrid((0.0,) * d, (0.25,) * d, (4,) * d)
    s = seed(fem, grid, [MPMBody("b", "b", 1.0)], n=2)
    lattice = full_lattice(grid, 2)
    want = lattice[pred(lattice)]
    assert 0 < len(want) < len(lattice)
    np.testing.assert_array_equal(s.position, want)

    # the uniformity report bins the same points
    cells = np.floor((want - grid.lo) / np.asarray(grid.cell_size)).astype(int)
    counts = np.zeros(grid.num_cells, dtype=int)
    np.add.at(counts, tuple(cells.T), 1)
    np.testing.assert_array_equal(s.report.counts, counts)
    assert s.report.partial_cells > 0
    assert s.report.counts.max() <= s.report.expected
    assert s.report.full_cells + s.report.partial_cells + s.report.empty_cells == counts.size
    assert len(s.report.partial()) == s.report.partial_cells


@pytest.mark.parametrize("family,n,expect", [
    ("tri3", 2, 4), ("quad4", 3, 9), ("tet4", 2, 8), ("hex8", 3, 27),
])
def test_point_on_a_shared_face_is_counted_once(family, n, expect):
    """A unit cell meshed with 2 elements (2 tris, 2 quads, 2 hexes) or 6 Kuhn tets: the
    lattice puts points on the shared faces (x=y diagonals; x=0.5 for odd n), and each is
    one particle: n^d in total, all distinct."""
    d = FAMILY_DIM[family]
    if family in ("quad4", "hex8"):
        nodes, ids = M.grid_nodes([2] + [1] * (d - 1), [0] * d, [1] * d)
        els = M.quads(ids) if d == 2 else M.hexes(ids)
        lattice_x = (np.arange(n) + 0.5) / n
        assert np.any(np.isclose(lattice_x, 0.5))           # points on the face x = 0.5
    else:
        nodes, ids = M.grid_nodes([1] * d, [0] * d, [1] * d)
        els = M.tris(ids) if d == 2 else M.tets(ids, nodes)
    fem = M.make_fem(nodes, {"b": els})
    grid = MPMGrid((0.0,) * d, (1.0,) * d, (1,) * d)
    s = seed(fem, grid, [MPMBody("b", "b", 1.0)], n=n)
    assert s.num_particles == expect == n ** d
    assert len(np.unique(s.position, axis=0)) == expect
    if family in ("tri3", "tet4"):
        assert np.any(s.position[:, 0] == s.position[:, 1])  # on the shared diagonal


def test_bodies_in_declared_order_then_lattice_order():
    nodes, ids = M.grid_nodes([2, 1], [0, 0], [1, 0.5])
    els = M.quads(ids)
    fem = M.make_fem(nodes, {"left": [els[0]], "right": [els[1]]})
    grid = MPMGrid((0.0, 0.0), (0.5, 0.5), (2, 1))
    s = seed(fem, grid, [MPMBody("right", "right", 2.0), MPMBody("left", "left", 1.0)], n=2)
    assert s.body_names == ("right", "left")
    np.testing.assert_array_equal(s.body_id, [0] * 4 + [1] * 4)
    assert np.all(s.position[:4, 0] > 0.5) and np.all(s.position[4:, 0] < 0.5)
    np.testing.assert_array_equal(s.density, [2.0] * 4 + [1.0] * 4)
    np.testing.assert_array_equal(s.position[:4], [[0.625, 0.125], [0.875, 0.125],
                                                   [0.625, 0.375], [0.875, 0.375]])


# ---------------------------------------------------------------------------
# refusals, by name
# ---------------------------------------------------------------------------

def _unit_square(family="quad4"):
    nodes, els = box_mesh(family, per_axis=1)
    return M.make_fem(nodes, {"b": els})


GRID2 = MPMGrid((0.0, 0.0), (0.5, 0.5), (2, 2))


@pytest.mark.parametrize("family,conn,grid_ndm", [
    ("tri6", [1, 2, 3, 4, 5, 6], 2),
    ("line2", [1, 2], 2),
    ("prism6", [1, 2, 3, 4, 5, 6], 3),
    ("hex8", [1, 2, 3, 4, 5, 6, 7, 8], 2),
    ("tri3", [1, 2, 3], 3),
])
def test_unsupported_family_is_refused_by_name(family, conn, grid_ndm):
    pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.5, 0, 0.5), (0.5, 0.5, 0.5), (0, 0.5, 0.5),
           (1, 1, 1), (0, 1, 1)]
    nodes = {i + 1: p for i, p in enumerate(pts)}
    fem = M.make_fem(nodes, {"b": [(family, conn)]})
    if grid_ndm == 2:
        grid, kw = GRID2, {"thickness": 1.0}
    else:
        grid, kw = MPMGrid((0.0,) * 3, (0.5,) * 3, (2,) * 3), {}
    with pytest.raises(MPMSeedError, match=rf"element family '{family}'.*not supported"):
        seed_lattice(fem, grid, [MPMBody("b", "b", 1.0)], n=2, **kw)


@pytest.mark.parametrize("bad_n", [0, -1, True, 2.0, None])
def test_n_below_one_or_not_an_integer_is_refused(bad_n):
    with pytest.raises(MPMSeedError, match="n .particles per cell per axis"):
        seed_lattice(_unit_square(), GRID2, [MPMBody("b", "b", 1.0)], n=bad_n,
                     thickness=1.0)


@pytest.mark.parametrize("kw,msg", [
    ({"name": "", "pg": "b", "density": 1.0}, "body name"),
    ({"name": "  ", "pg": "b", "density": 1.0}, "body name"),
    ({"name": "b", "pg": "", "density": 1.0}, "physical group name"),
    ({"name": "b", "pg": "b", "density": 0.0}, "density"),
    ({"name": "b", "pg": "b", "density": float("nan")}, "density"),
])
def test_bad_body_declaration_is_refused(kw, msg):
    with pytest.raises(MPMSeedError, match=msg):
        MPMBody(**kw)


@pytest.mark.parametrize("args,msg", [
    (((0.0,), (1.0,), (1,)), "ndm must be 2 or 3"),
    (((0.0, 0.0), (1.0,), (1, 1)), "2 entries each"),
    (((0.0, 0.0), (1.0, 0.0), (1, 1)), "cell_size must be > 0"),
    (((0.0, 0.0), (1.0, 1.0), (1, 0)), "num_cells must be >= 1"),
    (((0.0, 0.0), (1.0, 1.0), (1.5, 1)), "num_cells must be integers"),
    (((0.0, float("inf")), (1.0, 1.0), (1, 1)), "origin must be finite"),
])
def test_bad_grid_is_refused(args, msg):
    with pytest.raises(MPMSeedError, match=msg):
        MPMGrid(*args)


def test_unknown_basis_is_refused():
    with pytest.raises(MPMSeedError, match="basis"):
        MPMGrid((0.0, 0.0), (1.0, 1.0), (1, 1), basis="cubic")


def test_thickness_rules():
    fem = _unit_square()
    body = [MPMBody("b", "b", 1.0)]
    with pytest.raises(MPMSeedError, match="needs thickness"):
        seed_lattice(fem, GRID2, body, n=1)
    with pytest.raises(MPMSeedError, match="thickness must be finite and > 0"):
        seed_lattice(fem, GRID2, body, n=1, thickness=0.0)
    nodes, els = box_mesh("hex8", per_axis=1)
    fem3 = M.make_fem(nodes, {"b": els})
    with pytest.raises(MPMSeedError, match="3-D grid takes none"):
        seed_lattice(fem3, MPMGrid((0.0,) * 3, (1.0,) * 3, (1,) * 3), body, n=1,
                     thickness=1.0)


def test_body_with_no_lattice_point_is_refused_by_name():
    nodes = {1: (0.01, 0.01), 2: (0.05, 0.01), 3: (0.01, 0.05)}
    fem = M.make_fem(nodes, {"sliver": [("tri3", [1, 2, 3])]})
    with pytest.raises(MPMSeedError, match="body 'sliver'.*holds no lattice point"):
        seed(fem, GRID2, [MPMBody("sliver", "sliver", 1.0)], n=1)


@pytest.mark.parametrize("family", ["tri3", "quad4"])
def test_clockwise_2d_elements_seed_like_counter_clockwise_ones(family):
    """A surface meshed with its normal along -z numbers every element clockwise: all
    corner Jacobians negative. That is an orientation, not an inversion: the clockwise mesh
    seeds exactly the particles of the counter-clockwise one (here with points on the
    tri3 diagonals and on the quad4 face x = 1/2 at n = 3)."""
    nodes, ids = M.grid_nodes([2, 2], [0, 0], [1, 1])
    ccw = M.tris(ids) if family == "tri3" else M.quads(ids)
    cw = [(fam, [c[0], *reversed(c[1:])]) for fam, c in ccw]
    grid = MPMGrid((0.0, 0.0), (0.25, 0.25), (4, 4))
    body = [MPMBody("b", "b", 1.0)]
    for n in (2, 3):
        want = seed(M.make_fem(nodes, {"b": ccw}), grid, body, n=n)
        got = seed(M.make_fem(nodes, {"b": cw}), grid, body, n=n)
        assert got.num_particles == (4 * n) ** 2
        np.testing.assert_array_equal(got.position, want.position)
        np.testing.assert_array_equal(got.volume, want.volume)


@pytest.mark.parametrize("family,conn", [
    ("tri3", [1, 2, 5]),                       # collinear: zero area
    ("tri3", [1, 5, 2]),                       # collinear, the other way round
    ("quad4", [1, 2, 4, 3]),                   # bow-tie: mixed-sign corners
    ("quad4", [1, 3, 4, 2]),                   # bow-tie, the other way round
    ("quad4", [1, 2, 6, 4]),                   # non-convex (dart): mixed-sign corners
])
def test_non_positive_jacobian_is_refused_2d(family, conn):
    nodes = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1), 5: (0.5, 0), 6: (0.2, 0.2)}
    fem = M.make_fem(nodes, {"b": [(family, conn)]})
    with pytest.raises(MPMSeedError, match=r"element 1 has a non-positive Jacobian"):
        seed(fem, GRID2, [MPMBody("b", "b", 1.0)], n=1)


@pytest.mark.parametrize("family,conn", [
    ("tet4", [1, 3, 2, 5]),                    # inverted
    ("hex8", [5, 6, 7, 8, 1, 2, 3, 4]),        # top and bottom swapped
])
def test_non_positive_jacobian_is_refused_3d(family, conn):
    nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
             5: (0, 0, 1), 6: (1, 0, 1), 7: (1, 1, 1), 8: (0, 1, 1)}
    fem = M.make_fem(nodes, {"b": [(family, conn)]})
    with pytest.raises(MPMSeedError, match=r"element 1 has a non-positive Jacobian"):
        seed_lattice(fem, MPMGrid((0.0,) * 3, (0.5,) * 3, (2,) * 3),
                     [MPMBody("b", "b", 1.0)], n=1)


def test_overlapping_bodies_are_refused_naming_both_and_the_count():
    """Two bodies on the same unit square, 2 x 2 cells at n = 2: all 16 points shared."""
    nodes, els = box_mesh("quad4", per_axis=1)
    fem = M.make_fem(nodes, {"fill": els, "rock": els})
    with pytest.raises(MPMSeedError) as exc:
        seed(fem, GRID2, [MPMBody("fill", "fill", 1.0), MPMBody("rock", "rock", 1.0)], n=2)
    msg = str(exc.value)
    assert "bodies 'fill' and 'rock' share 16 lattice points at n=2;" in msg
    assert "16 of them are not on an interface between the two" in msg


# ---------------------------------------------------------------------------
# a lattice point two bodies claim (ADR 0134 Amendment 2)
# ---------------------------------------------------------------------------
#
# The lattice below is GRID_I at n = 2: spacing 0.25, points at 0.125, 0.375, 0.625 and
# 0.875 on each axis, all dyadic. The interfaces sit on x = 0.625 (and y = 0.625), so a
# column of lattice points lies exactly on them. The bodies are meshed at 0.125, so every
# lattice point also lies on element facets inside each body: the rule must tell a body's
# own internal facets from its boundary.

GRID_I = {2: MPMGrid((0.0,) * 2, (0.5,) * 2, (2,) * 2),
          3: MPMGrid((0.0,) * 3, (0.5,) * 3, (2,) * 3)}


def boxes_fem(family, boxes, divisions):
    """Each body a structured mesh of its own box ``boxes[name] = (lo, hi)``, with its
    own nodes (two bodies share no node id, so their meshes need not conform)."""
    nodes, bodies, first = {}, {}, 1
    for name, (lo, hi) in boxes.items():
        nd, els = M.box(family, lo, hi, divisions[name], first=first)
        nodes.update(nd)
        bodies[name] = els
        first += len(nd)
    return M.make_fem(nodes, bodies)


def first_body_oracle(grid, n, boxes, order):
    """Closed form: each lattice point goes to the first body in ``order`` whose closed box
    holds it. Returns positions and body ids in Contract-1 order and the per-pair count of
    points a later body gives up to an earlier one."""
    lattice = full_lattice(grid, n)
    held = [np.all((lattice >= boxes[b][0]) & (lattice <= boxes[b][1]), axis=1)
            for b in order]
    owner = np.full(len(lattice), -1)
    for k, mask in enumerate(held):
        owner[(owner < 0) & mask] = k
    position = np.concatenate([lattice[owner == k] for k in range(len(order))])
    body_id = np.concatenate([np.full(np.count_nonzero(owner == k), k)
                              for k in range(len(order))])
    pairs = {}
    for j, mask in enumerate(held):
        for i in range(j):
            k = int(np.count_nonzero(mask & (owner == i)))
            if k:
                pairs[(order[i], order[j])] = k
    return position, body_id, pairs


def _split_box(d):
    """left = [0, 0.625] x [0, 1]^(d-1), right = [0.625, 1] x [0, 1]^(d-1)."""
    rest = [0.0] * (d - 1), [1.0] * (d - 1)
    return {"left": ([0.0, *rest[0]], [0.625, *rest[1]]),
            "right": ([0.625, *rest[0]], [1.0, *rest[1]])}


@pytest.mark.parametrize("conforming", [True, False], ids=["conforming", "nonconforming"])
@pytest.mark.parametrize("first", ["left", "right"])
@pytest.mark.parametrize("family", list(FAMILY_DIM))
def test_a_shared_face_on_the_lattice_goes_to_the_first_declared_body(family, first,
                                                                      conforming):
    """Two bodies share the face x = 0.625, which holds (4)^(d-1) lattice points. The
    model seeds; every one of them belongs to the body declared first, the other body
    keeps its own points only, the total is the union's lattice count (4^d), and swapping
    the declaration order swaps the owner. A conforming mesh shares the interface nodes;
    a non-conforming one meshes the right body coarser, with no node in common."""
    d = FAMILY_DIM[family]
    grid = GRID_I[d]
    boxes = _split_box(d)
    if conforming:
        nodes, els = M.box(family, [0.0] * d, [1.0] * d, [8] * d)
        centre_x = [np.mean([nodes[int(v)][0] for v in conn]) for _, conn in els]
        fem = M.make_fem(nodes, {
            "left": [e for e, c in zip(els, centre_x) if c < 0.625],
            "right": [e for e, c in zip(els, centre_x) if c > 0.625],
        })
    else:
        fem = boxes_fem(family, boxes, {"left": [5] + [8] * (d - 1),
                                        "right": [1] + [3] * (d - 1)})
    order = [first, "right" if first == "left" else "left"]
    s = seed(fem, grid, [MPMBody(b, b, 1.0 + k) for k, b in enumerate(order)], n=2)

    position, body_id, pairs = first_body_oracle(grid, 2, boxes, order)
    assert pairs == {(order[0], order[1]): 4 ** (d - 1)}
    assert s.num_particles == 4 ** d
    np.testing.assert_array_equal(s.position, position)
    np.testing.assert_array_equal(s.body_id, body_id)
    np.testing.assert_array_equal(s.density, 1.0 + body_id)
    on_face = s.position[:, 0] == 0.625
    assert on_face.sum() == 4 ** (d - 1)
    np.testing.assert_array_equal(s.body_id[on_face], 0)
    assert s.report.interface_points == pairs
    np.testing.assert_array_equal(s.report.counts, np.full((2,) * d, 2 ** d))
    assert f"interface points kept by the first-declared body: {order[0]!r} over " \
           f"{order[1]!r}: {4 ** (d - 1)}" in str(s.report)


@pytest.mark.parametrize("family", ["quad4", "tri3"])
def test_four_bodies_meeting_at_a_lattice_point_report_each_pair(family):
    """Four quadrants meet at (0.625, 0.625), a lattice point all four claim; it goes to
    the first. Closed form per pair: q00 keeps 3 points from q10, 3 from q01 and the corner
    from q11; q10 and q01 each keep one point from q11."""
    a, b = [0.0, 0.625], [0.625, 1.0]
    boxes = {"q00": ([a[0], a[0]], [a[1], a[1]]), "q10": ([b[0], a[0]], [b[1], a[1]]),
             "q01": ([a[0], b[0]], [a[1], b[1]]), "q11": ([b[0], b[0]], [b[1], b[1]])}
    div = {"q00": [5, 5], "q10": [3, 5], "q01": [5, 3], "q11": [3, 3]}
    fem = boxes_fem(family, boxes, div)
    order = list(boxes)
    s = seed(fem, GRID_I[2], [MPMBody(q, q, 1.0) for q in order], n=2)
    position, body_id, pairs = first_body_oracle(GRID_I[2], 2, boxes, order)
    assert pairs == {("q00", "q10"): 3, ("q00", "q01"): 3, ("q00", "q11"): 1,
                     ("q10", "q11"): 1, ("q01", "q11"): 1}
    assert s.num_particles == 16
    np.testing.assert_array_equal(s.position, position)
    np.testing.assert_array_equal(s.body_id, body_id)
    assert s.report.interface_points == pairs


@pytest.mark.parametrize("family", list(FAMILY_DIM))
def test_an_overlap_whose_points_all_lie_on_element_facets_is_refused(family):
    """A = [0, 0.625] and B = [0.375, 1] along x overlap on [0.375, 0.625]. Each shared
    lattice point lies on a facet of both meshes, but on x = 0.375 the facet is internal
    to A and on x = 0.625 internal to B: inside a body, not on an interface."""
    d = FAMILY_DIM[family]
    rest = [0.0] * (d - 1), [1.0] * (d - 1)
    boxes = {"A": ([0.0, *rest[0]], [0.625, *rest[1]]),
             "B": ([0.375, *rest[0]], [1.0, *rest[1]])}
    fem = boxes_fem(family, boxes, {"A": [5] + [8] * (d - 1), "B": [5] + [8] * (d - 1)})
    shared = 2 * 4 ** (d - 1)
    with pytest.raises(MPMSeedError) as exc:
        seed(fem, GRID_I[d], [MPMBody("A", "A", 1.0), MPMBody("B", "B", 1.0)], n=2)
    assert (f"bodies 'A' and 'B' share {shared} lattice points at n=2; {shared} of them "
            f"are not on an interface between the two (the bodies overlap there)"
            in str(exc.value))


GRID_S = MPMGrid((0.0, 0.0), (0.5, 0.5), (2, 3))


def _sliver_fem(sliver_lo, sliver_hi):
    """A = [0, 1] x [0, 0.875]; B = a 0.0625-thin sliver on y = 0.875 over x in [0.5, 1]
    plus a block [0, 1] x [1, 1.5] that holds B's own points."""
    nodes_a, els_a = M.box("quad4", [0, 0], [1, 0.875], [8, 7])
    nodes_s, els_s = M.box("quad4", [0.5, sliver_lo], [1, sliver_hi], [4, 1], first=1000)
    nodes_b, els_b = M.box("quad4", [0, 1.0], [1, 1.5], [2, 2], first=2000)
    return M.make_fem({**nodes_a, **nodes_s, **nodes_b}, {"A": els_a, "B": els_s + els_b})


def test_bodies_on_opposite_sides_of_a_boundary_share_it():
    """The sliver lies above y = 0.875 (B's facet faces down, A's up): the two lattice
    points (0.625, 0.875) and (0.875, 0.875) are on the interface and go to A."""
    s = seed(_sliver_fem(0.875, 0.9375), GRID_S,
             [MPMBody("A", "A", 1.0), MPMBody("B", "B", 2.0)], n=2)
    lattice = full_lattice(GRID_S, 2)
    want_a = lattice[lattice[:, 1] <= 0.875]
    want_b = lattice[lattice[:, 1] >= 1.0]
    np.testing.assert_array_equal(s.position, np.concatenate([want_a, want_b]))
    np.testing.assert_array_equal(s.body_id, [0] * 16 + [1] * 8)
    assert s.report.interface_points == {("A", "B"): 2}


def test_bodies_on_the_same_side_of_a_boundary_are_refused():
    """The same sliver below y = 0.875 lies inside A: both bodies' facets on y = 0.875 face
    up, so the two shared points are an overlap, not an interface."""
    with pytest.raises(MPMSeedError, match=r"bodies 'A' and 'B' share 2 lattice points at "
                       r"n=2; 2 of them are not on an interface between the two"):
        seed(_sliver_fem(0.8125, 0.875), GRID_S,
             [MPMBody("A", "A", 1.0), MPMBody("B", "B", 2.0)], n=2)


def test_a_body_left_with_only_interface_points_is_refused():
    """The sliver alone: both its lattice points go to A, so B holds none of its own."""
    nodes_a, els_a = M.box("quad4", [0, 0], [1, 0.875], [8, 7])
    nodes_s, els_s = M.box("quad4", [0.5, 0.875], [1, 0.9375], [4, 1], first=1000)
    fem = M.make_fem({**nodes_a, **nodes_s}, {"A": els_a, "B": els_s})
    with pytest.raises(MPMSeedError, match=r"body 'B' \(physical group 'B'\) holds no "
                       r"lattice point of its own at n=2"):
        seed(fem, GRID_I[2], [MPMBody("A", "A", 1.0), MPMBody("B", "B", 1.0)], n=2)


def test_one_lattice_column_on_a_two_element_interface_seeds_at_n_3():
    """Two bodies meet at x = 0.5; at n = 3 in one unit cell the column x = 0.5 holds 3
    sub-cell centres (y = 1/6, 1/2, 5/6). Before Amendment 2 this model was refused; now
    the 3 points go to 'left', declared first, and 'right' keeps its own 3."""
    nodes, ids = M.grid_nodes([2, 1], [0, 0], [1, 1])
    els = M.quads(ids)
    fem = M.make_fem(nodes, {"left": [els[0]], "right": [els[1]]})
    grid = MPMGrid((0.0, 0.0), (1.0, 1.0), (1, 1))
    s = seed(fem, grid, [MPMBody("left", "left", 1.0), MPMBody("right", "right", 1.0)], n=3)
    assert s.num_particles == 9
    np.testing.assert_array_equal(s.body_id, [0] * 6 + [1] * 3)
    np.testing.assert_allclose(s.position[s.body_id == 1][:, 0], 5 / 6)
    assert s.report.interface_points == {("left", "right"): 3}


def test_mesh_outside_the_grid_is_refused():
    fem = _unit_square()
    grid = MPMGrid((0.0, 0.0), (0.25, 0.25), (2, 2))
    with pytest.raises(MPMSeedError, match="extends outside the grid box"):
        seed(fem, grid, [MPMBody("b", "b", 1.0)], n=1)


def test_unknown_physical_group_is_refused():
    with pytest.raises(MPMSeedError, match="physical group 'nope' is not in this FEMData"):
        seed(_unit_square(), GRID2, [MPMBody("b", "nope", 1.0)], n=1)


def test_duplicate_body_names_are_refused():
    with pytest.raises(MPMSeedError, match="unique; repeated: \\['b'\\]"):
        seed(_unit_square(), GRID2, [MPMBody("b", "b", 1.0), MPMBody("b", "b", 2.0)], n=1)


def test_2d_body_off_one_z_plane_is_refused():
    nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0.3), 4: (0, 1, 0)}
    fem = M.make_fem(nodes, {"b": [("quad4", [1, 2, 3, 4])]})
    with pytest.raises(MPMSeedError, match="one z-plane"):
        seed(fem, GRID2, [MPMBody("b", "b", 1.0)], n=1)
