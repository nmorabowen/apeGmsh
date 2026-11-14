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


def test_interface_through_a_sub_cell_centre_is_refused_naming_both_and_the_count():
    """Two bodies meet at x = 0.5; at n = 3 in one unit cell the column x = 0.5 holds 3
    sub-cell centres (y = 1/6, 1/2, 5/6), each inside both bodies."""
    nodes, ids = M.grid_nodes([2, 1], [0, 0], [1, 1])
    els = M.quads(ids)
    fem = M.make_fem(nodes, {"left": [els[0]], "right": [els[1]]})
    grid = MPMGrid((0.0, 0.0), (1.0, 1.0), (1, 1))
    with pytest.raises(MPMSeedError) as exc:
        seed(fem, grid, [MPMBody("left", "left", 1.0), MPMBody("right", "right", 1.0)], n=3)
    msg = str(exc.value)
    assert "bodies 'left' and 'right' share 3 lattice points at n=3;" in msg
    assert re.search(r"e\.g\. \[0\.5, 0\.16666", msg)
    with pytest.raises(MPMSeedError, match=r"share 1 lattice point at n=1;"):
        seed(fem, grid, [MPMBody("left", "left", 1.0), MPMBody("right", "right", 1.0)], n=1)


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
