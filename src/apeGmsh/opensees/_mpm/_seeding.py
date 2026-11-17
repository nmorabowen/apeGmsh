"""Lattice seeding of MPM particles from FEMData alone (ADR 0134 D2).

The grid declares ``n`` points per cell per axis at the sub-cell centres. A point is kept
for a body when it lies inside an element of the body's physical group, tested on the
FEMData mesh in numpy: barycentric coordinates for ``tri3``/``tet4``, the inverse
isoparametric map (Newton) for ``quad4``/``hex8``. A point on a face shared by two
elements of one body is kept once. A point claimed by two bodies is kept once, by the body
declared first, when it lies on the interface between them (Amendment 2); any other point
claimed by two bodies is refused by name. Every other element family is refused by name.

Charter P3/P9: this module imports neither gmsh nor anything outside ``apeGmsh.mesh``.
``FEMData`` is read through its public ``nodes``/``elements`` surface only.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only (no runtime edge to apeGmsh.mesh)
    from apeGmsh.mesh.FEMData import FEMData

__all__ = [
    "SUPPORTED_FAMILIES",
    "MPMBody",
    "MPMGrid",
    "MPMSeedError",
    "SeededParticles",
    "UniformityReport",
    "seed_lattice",
]

#: Element families lattice seeding supports, per spatial dimension (ADR 0134 D2).
SUPPORTED_FAMILIES: dict[int, tuple[str, ...]] = {2: ("tri3", "quad4"), 3: ("tet4", "hex8")}
BASES = ("gimp", "linear")

#: A point is inside when its barycentric / reference coordinates miss the element by at
#: most this much (reference units). Shared faces are then claimed by both neighbours and
#: deduplicated by lattice index.
_INSIDE_TOL = 1e-10
_NEWTON_TOL = 1e-12
_NEWTON_MAXIT = 50
#: A Newton iterate this far outside the reference cell is outside the element.
_FAR_OUTSIDE = 4.0
#: Candidate (element, lattice point) pairs tested per numpy batch.
_PAIRS_PER_BATCH = 1 << 20

_QUAD_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], dtype=np.float64)
_HEX_CORNERS = np.array(
    [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
     (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    dtype=np.float64,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class MPMSeedError(ValueError):
    """A seeding input or mesh the lattice seeder refuses (ADR 0134 D2)."""


def _finite_tuple(name: str, value: Sequence[float]) -> tuple[float, ...]:
    out = tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in out):
        raise MPMSeedError(f"MPMGrid.{name} must be finite, got {out}")
    return out


@dataclass(frozen=True)
class MPMGrid:
    """The background grid: ``[origin, origin + num_cells * cell_size]`` per axis."""

    origin: tuple[float, ...]
    cell_size: tuple[float, ...]
    num_cells: tuple[int, ...]
    basis: str = "gimp"

    def __post_init__(self) -> None:
        origin = _finite_tuple("origin", self.origin)
        cell = _finite_tuple("cell_size", self.cell_size)
        cells = tuple(self.num_cells)
        if any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) for c in cells):
            raise MPMSeedError(f"MPMGrid.num_cells must be integers, got {cells}")
        cells_i = tuple(int(c) for c in cells)
        ndm = len(origin)
        if ndm not in (2, 3):
            raise MPMSeedError(f"MPMGrid: ndm must be 2 or 3, got origin {origin}")
        if len(cell) != ndm or len(cells_i) != ndm:
            raise MPMSeedError(
                f"MPMGrid: origin, cell_size and num_cells need {ndm} entries each, "
                f"got {origin}, {cell}, {cells_i}")
        if any(h <= 0.0 for h in cell):
            raise MPMSeedError(f"MPMGrid.cell_size must be > 0 on each axis, got {cell}")
        if any(c < 1 for c in cells_i):
            raise MPMSeedError(f"MPMGrid.num_cells must be >= 1 on each axis, got {cells_i}")
        if any(c > np.iinfo(np.int32).max for c in cells_i):
            raise MPMSeedError(f"MPMGrid.num_cells exceeds int32, got {cells_i}")
        if self.basis not in BASES:
            raise MPMSeedError(f"MPMGrid.basis must be one of {BASES}, got {self.basis!r}")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "cell_size", cell)
        object.__setattr__(self, "num_cells", cells_i)

    @property
    def ndm(self) -> int:
        return len(self.origin)

    @property
    def lo(self) -> FloatArray:
        return np.asarray(self.origin, dtype=np.float64)

    @property
    def hi(self) -> FloatArray:
        return self.lo + np.asarray(self.num_cells, dtype=np.float64) * np.asarray(
            self.cell_size, dtype=np.float64)


@dataclass(frozen=True)
class MPMBody:
    """One MPM body: its name, the physical group that holds its elements, its density."""

    name: str
    pg: str
    density: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise MPMSeedError(f"MPMBody: the body name must be a non-empty string, "
                               f"got {self.name!r}")
        if not isinstance(self.pg, str) or not self.pg.strip():
            raise MPMSeedError(f"MPMBody {self.name!r}: the physical group name must be a "
                               f"non-empty string, got {self.pg!r}")
        rho = float(self.density)
        if not math.isfinite(rho) or rho <= 0.0:
            raise MPMSeedError(f"MPMBody {self.name!r}: density must be finite and > 0, "
                               f"got {self.density!r}")
        object.__setattr__(self, "density", rho)


@dataclass(frozen=True)
class UniformityReport:
    """Particles per grid cell against the full count ``n**ndm``.

    ``counts`` has shape ``grid.num_cells``, indexed ``[ix, iy(, iz)]``. A cell on a body's
    boundary holds fewer than ``expected``; a cell holds more only if bodies overlap, which
    seeding refuses, so ``counts <= expected`` always holds.

    ``interface_points`` maps ``(first, later)`` body names to the number of lattice points
    on their interface that the first-declared body kept and the later one gave up (ADR
    0134 Amendment 2). It is empty when no two bodies claim a point.
    """

    counts: IntArray
    expected: int
    interface_points: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def full_cells(self) -> int:
        return int(np.count_nonzero(self.counts == self.expected))

    @property
    def partial_cells(self) -> int:
        return int(np.count_nonzero((self.counts > 0) & (self.counts < self.expected)))

    @property
    def empty_cells(self) -> int:
        return int(np.count_nonzero(self.counts == 0))

    def partial(self) -> IntArray:
        """Indices ``(k, ndm)`` of the partially filled cells."""
        mask = (self.counts > 0) & (self.counts < self.expected)
        return np.argwhere(mask).astype(np.int64)

    def __str__(self) -> str:
        text = (f"{int(self.counts.sum())} particles; {self.full_cells} full cells "
                f"({self.expected} each), {self.partial_cells} partial, "
                f"{self.empty_cells} empty")
        if self.interface_points:
            pairs = ", ".join(f"{a!r} over {b!r}: {k}"
                              for (a, b), k in self.interface_points.items())
            text += f"; interface points kept by the first-declared body: {pairs}"
        return text


@dataclass(frozen=True)
class SeededParticles:
    """A seeded particle set, in Contract-1 order: bodies as declared, then lattice order
    (x fastest, then y, then z) within a body. Ids are ``0 .. N-1`` in that order."""

    grid: MPMGrid
    n: int
    thickness: float | None
    body_names: tuple[str, ...]
    position: FloatArray
    volume: FloatArray
    gimp_half_length: FloatArray
    density: FloatArray
    body_id: NDArray[np.int32]
    report: UniformityReport

    @property
    def num_particles(self) -> int:
        return int(self.volume.shape[0])

    @property
    def ndm(self) -> int:
        return self.grid.ndm


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def seed_lattice(
    fem: FEMData,
    grid: MPMGrid,
    bodies: Sequence[MPMBody],
    *,
    n: int,
    thickness: float | None = None,
) -> SeededParticles:
    """Seed ``n`` particles per cell per axis for each body (ADR 0134 D2).

    ``thickness`` is the plane-strain thickness, required when ``grid.ndm == 2`` and
    refused when it is 3. Each particle gets the volume ``prod(h/n)`` (times the thickness
    in 2D), the GIMP half-lengths ``h/(2n)`` and its body's density.

    A lattice point on the interface of two bodies is kept once, by the body declared first,
    and counted in ``report.interface_points``; a point inside a body that another body
    also claims is refused (ADR 0134 Amendment 2).
    """
    if not isinstance(grid, MPMGrid):
        raise MPMSeedError(f"grid must be an MPMGrid, got {type(grid).__name__}")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise MPMSeedError(f"n (particles per cell per axis) must be an integer >= 1, "
                           f"got {n!r}")
    n = int(n)
    ndm = grid.ndm
    thk = _check_thickness(ndm, thickness)
    bodies = tuple(bodies)
    if not bodies:
        raise MPMSeedError("seed_lattice needs at least one MPMBody")
    for b in bodies:
        if not isinstance(b, MPMBody):
            raise MPMSeedError(f"bodies must be MPMBody instances, got {type(b).__name__}")
    names = [b.name for b in bodies]
    dup = sorted({x for x in names if names.count(x) > 1})
    if dup:
        raise MPMSeedError(f"body names must be unique; repeated: {dup}")

    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    node_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    if node_ids.size == 0:
        raise MPMSeedError("this FEMData holds no nodes; there is nothing to seed")
    order =np.argsort(node_ids, kind="stable")
    sorted_ids = node_ids[order]

    lattice_shape = tuple(c * n for c in grid.num_cells)
    kept: list[IntArray] = []
    for body in bodies:
        lin = _seed_body(fem, body, grid, n, lattice_shape, sorted_ids, order, node_xyz)
        if lin.size == 0:
            raise MPMSeedError(
                f"body {body.name!r} (physical group {body.pg!r}) holds no lattice point "
                f"at n={n} per cell per axis; refine n or the grid")
        kept.append(lin)

    interface = _resolve_shared_points(
        kept, bodies, fem, grid, n, lattice_shape, sorted_ids, order, node_xyz)

    all_lin = np.concatenate(kept)
    idx = np.stack(np.unravel_index(all_lin, lattice_shape, order="F"), axis=1)
    spacing = np.asarray(grid.cell_size, dtype=np.float64) / n
    position = grid.lo + (idx.astype(np.float64) + 0.5) * spacing
    count = all_lin.size
    vol = float(np.prod(spacing)) * (thk if thk is not None else 1.0)
    volume = np.full(count, vol, dtype=np.float64)
    half = np.broadcast_to(spacing / 2.0, (count, ndm)).astype(np.float64)
    body_id = np.concatenate(
        [np.full(k.size, i, dtype=np.int32) for i, k in enumerate(kept)])
    density = np.concatenate(
        [np.full(k.size, b.density, dtype=np.float64) for b, k in zip(bodies, kept)])

    cells = idx // n
    counts = np.bincount(
        np.ravel_multi_index(tuple(cells.T), grid.num_cells, order="F"),
        minlength=int(np.prod(grid.num_cells)),
    ).reshape(grid.num_cells, order="F").astype(np.int64)
    report = UniformityReport(counts=counts, expected=n ** ndm, interface_points=interface)

    return SeededParticles(
        grid=grid, n=n, thickness=thk, body_names=tuple(names),
        position=position, volume=volume, gimp_half_length=half,
        density=density, body_id=body_id, report=report,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _check_thickness(ndm: int, thickness: float | None) -> float | None:
    if ndm == 3:
        if thickness is not None:
            raise MPMSeedError("thickness is the plane-strain thickness; a 3-D grid "
                               f"takes none, got {thickness!r}")
        return None
    if thickness is None:
        raise MPMSeedError("a 2-D (plane-strain) grid needs thickness= (> 0)")
    t = float(thickness)
    if not math.isfinite(t) or t <= 0.0:
        raise MPMSeedError(f"thickness must be finite and > 0, got {thickness!r}")
    return t


def _seed_body(
    fem: Any,
    body: MPMBody,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
    sorted_ids: IntArray,
    order: IntArray,
    node_xyz: FloatArray,
) -> IntArray:
    """Sorted, unique linear lattice indices (Fortran order) kept for one body."""
    found: list[IntArray] = []
    for g in _body_groups(fem, body, grid, sorted_ids, order, node_xyz):
        found.extend(_scan(g.tester, g.lo_e, g.hi_e, grid, n, lattice_shape))
    if not found:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(found))


@dataclass(frozen=True)
class _Group:
    """One element family of a body: its connectivity, bounding boxes and tester."""

    conn: IntArray
    lo_e: FloatArray
    hi_e: FloatArray
    tester: _SimplexTester | _IsoTester


def _body_groups(
    fem: Any,
    body: MPMBody,
    grid: MPMGrid,
    sorted_ids: IntArray,
    order: IntArray,
    node_xyz: FloatArray,
) -> Iterator[_Group]:
    """Validate a body's element groups and yield each one's geometry, group by group."""
    ndm = grid.ndm
    try:
        groups = list(fem.elements.select(pg=body.pg).groups())
    except KeyError as exc:
        raise MPMSeedError(
            f"body {body.name!r}: physical group {body.pg!r} is not in this FEMData "
            f"({exc})") from exc
    if not groups or all(len(g.ids) == 0 for g in groups):
        raise MPMSeedError(f"body {body.name!r}: physical group {body.pg!r} has no elements")

    supported = SUPPORTED_FAMILIES[ndm]
    for g in groups:
        fam = g.element_type.name
        if fam not in supported:
            raise MPMSeedError(
                f"body {body.name!r}: element family {fam!r} in physical group "
                f"{body.pg!r} is not supported for lattice seeding at ndm={ndm} "
                f"(supported: {', '.join(supported)})")

    extent = float(np.max(grid.hi - grid.lo))
    box_tol = 1e-9 * extent
    for g in groups:
        fam = g.element_type.name
        conn = np.asarray(g.connectivity, dtype=np.int64)
        eids = np.asarray(g.ids, dtype=np.int64)
        pos = np.searchsorted(sorted_ids, conn)
        pos_c = np.clip(pos, 0, sorted_ids.size - 1)
        missing = sorted_ids[pos_c] != conn
        if missing.any():
            bad = int(conn[missing][0])
            raise MPMSeedError(f"body {body.name!r}: element connectivity names node "
                               f"{bad}, which FEMData.nodes does not hold")
        xyz = node_xyz[order[pos_c]]                       # (E, npe, 3)
        if ndm == 2:
            z = xyz[..., 2]
            if float(np.ptp(z)) > box_tol:
                raise MPMSeedError(
                    f"body {body.name!r}: a 2-D grid needs the body's mesh in one "
                    f"z-plane; its nodes span z in [{z.min()}, {z.max()}]")
        verts = np.ascontiguousarray(xyz[..., :ndm])       # (E, npe, ndm)
        lo_e = verts.min(axis=1)
        hi_e = verts.max(axis=1)
        out = (lo_e < grid.lo - box_tol) | (hi_e > grid.hi + box_tol)
        if out.any():
            e = int(eids[np.nonzero(out.any(axis=1))[0][0]])
            raise MPMSeedError(
                f"body {body.name!r}: element {e} of physical group {body.pg!r} extends "
                f"outside the grid box [{grid.lo.tolist()}, {grid.hi.tolist()}]")
        if ndm == 2:
            verts, conn = _counter_clockwise(verts, conn)
        tester: _SimplexTester | _IsoTester
        if fam in ("tri3", "tet4"):
            tester = _SimplexTester(verts, eids, body)
        else:
            tester = _IsoTester(verts, eids, body)
        yield _Group(conn=conn, lo_e=lo_e, hi_e=hi_e, tester=tester)


def _counter_clockwise(verts: FloatArray, conn: IntArray) -> tuple[FloatArray, IntArray]:
    """tri3/quad4 vertices (and connectivity) with each clockwise element's node order
    reversed.

    A surface meshed with its normal along -z numbers its elements clockwise, so every
    corner Jacobian is negative. That is an orientation, not an inverted element: lattice
    seeding needs no sign (ADR 0134 D2 keeps the signed check for Gauss seeding). The
    reversal is local, by the shoelace area; a bow-tie or a degenerate element has mixed
    signs or a near-zero Jacobian in either order and the tester still refuses it.
    """
    x = verts[..., 0]
    y = verts[..., 1]
    area2 = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    cw = area2 < 0.0
    if not cw.any():
        return verts, conn
    out = verts.copy()
    out_conn = conn.copy()
    order = [0, *range(verts.shape[1] - 1, 0, -1)]
    out[cw] = verts[cw][:, order, :]
    out_conn[cw] = conn[cw][:, order]
    return out, out_conn


def _scan(
    tester: _SimplexTester | _IsoTester,
    lo_e: FloatArray,
    hi_e: FloatArray,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
) -> list[IntArray]:
    """Test every lattice point in each element's bounding box; return kept indices."""
    kept: list[IntArray] = []
    for e_loc, idx, pts in _candidates(lo_e, hi_e, grid, n, lattice_shape):
        inside = tester.inside(e_loc, pts)
        if inside.any():
            kept.append(np.ravel_multi_index(
                tuple(idx[inside].T), lattice_shape, order="F").astype(np.int64))
    return kept


def _candidates(
    lo_e: FloatArray,
    hi_e: FloatArray,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
) -> Iterator[tuple[IntArray, IntArray, FloatArray]]:
    """Batches ``(element, lattice index, position)`` of the lattice points in each
    element's bounding box, about ``_PAIRS_PER_BATCH`` pairs per batch."""
    ndm = grid.ndm
    spacing = np.asarray(grid.cell_size, dtype=np.float64) / n
    shape = np.asarray(lattice_shape, dtype=np.int64)
    tol = 1e-9 * spacing
    # lattice coordinate i -> origin + (i + 1/2) * spacing
    a = np.ceil((lo_e - tol - grid.lo) / spacing - 0.5).astype(np.int64)
    b = np.floor((hi_e + tol - grid.lo) / spacing - 0.5).astype(np.int64) + 1
    a = np.clip(a, 0, shape)
    b = np.clip(b, 0, shape)
    sizes = np.maximum(b - a, 0)
    per_elem = np.prod(sizes, axis=1)
    start = 0
    n_elem = per_elem.size
    while start < n_elem:
        # grow the batch until it holds about _PAIRS_PER_BATCH pairs (at least one element)
        csum = np.cumsum(per_elem[start:])
        stop = start + max(1, int(np.searchsorted(csum, _PAIRS_PER_BATCH, side="right")))
        sel = np.arange(start, stop)
        counts = per_elem[sel]
        total = int(counts.sum())
        if total:
            e_loc = np.repeat(sel, counts)
            first = np.cumsum(counts) - counts
            k = np.arange(total, dtype=np.int64) - np.repeat(first, counts)
            idx = np.empty((total, ndm), dtype=np.int64)
            for d in range(ndm):
                s = sizes[e_loc, d]
                idx[:, d] = a[e_loc, d] + k % s
                k = k // s
            pts = grid.lo + (idx.astype(np.float64) + 0.5) * spacing
            yield e_loc, idx, pts
        start = stop


class _SimplexTester:
    """tri3 / tet4: barycentric coordinates, ``lambda >= -tol``."""

    def __init__(self, verts: FloatArray, eids: IntArray, body: MPMBody) -> None:
        x0 = verts[:, 0, :]
        edges = np.transpose(verts[:, 1:, :] - x0[:, None, :], (0, 2, 1))  # columns
        det = np.linalg.det(edges)
        ndm = verts.shape[2]
        lmax = np.max(np.linalg.norm(verts[:, :, None, :] - verts[:, None, :, :], axis=-1),
                      axis=(1, 2))
        bad = ~(det > 1e-12 * lmax ** ndm)
        if bad.any():
            i = int(np.nonzero(bad)[0][0])
            raise MPMSeedError(
                f"body {body.name!r}: element {int(eids[i])} has a non-positive Jacobian "
                f"(det {float(det[i]):.6g}); seeding refuses an inverted or degenerate "
                f"element")
        self._x0 = x0
        self._inv = np.linalg.inv(edges)
        nv = ndm + 1
        #: facet k is the one opposite vertex k (where lambda_k = 0)
        self.facet_nodes = np.array([[v for v in range(nv) if v != k] for k in range(nv)])

    def inside(self, e: IntArray, pts: FloatArray) -> NDArray[np.bool_]:
        lam = np.einsum("pij,pj->pi", self._inv[e], pts - self._x0[e])
        lam0 = 1.0 - lam.sum(axis=1)
        inside: NDArray[np.bool_] = (lam.min(axis=1) >= -_INSIDE_TOL) & (lam0 >= -_INSIDE_TOL)
        return inside

    def facets(self, e: IntArray, pts: FloatArray) -> tuple[NDArray[np.bool_], FloatArray]:
        """Which facets each point lies on (``lambda_k <= tol``, inside points only) and
        each facet's outward unit normal ``-grad(lambda_k)``: ``(P, F)`` and ``(P, F, ndm)``."""
        inv = self._inv[e]                                  # row k: grad lambda_(k+1)
        lam = np.einsum("pij,pj->pi", inv, pts - self._x0[e])
        full = np.concatenate([(1.0 - lam.sum(axis=1))[:, None], lam], axis=1)
        inside = full.min(axis=1) >= -_INSIDE_TOL
        on: NDArray[np.bool_] = (full <= _INSIDE_TOL) & inside[:, None]
        grad = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)
        outward: FloatArray = -grad / np.linalg.norm(grad, axis=2, keepdims=True)
        return on, outward


class _IsoTester:
    """quad4 / hex8: the inverse isoparametric map by Newton, ``|xi| <= 1 + tol``."""

    def __init__(self, verts: FloatArray, eids: IntArray, body: MPMBody) -> None:
        ndm = verts.shape[2]
        self._corners = _QUAD_CORNERS if ndm == 2 else _HEX_CORNERS
        self._verts = verts
        self._eids = eids
        self._body = body
        # every corner Jacobian positive (the standard validity check)
        e_all = np.repeat(np.arange(verts.shape[0]), len(self._corners))
        xi = np.tile(self._corners, (verts.shape[0], 1))
        det = np.linalg.det(self._jacobian(e_all, xi)).reshape(verts.shape[0], -1)
        scale = np.max(verts.max(axis=1) - verts.min(axis=1), axis=1) ** ndm
        bad = ~(det.min(axis=1) > 1e-12 * scale)
        if bad.any():
            i = int(np.nonzero(bad)[0][0])
            raise MPMSeedError(
                f"body {body.name!r}: element {int(eids[i])} has a non-positive Jacobian "
                f"at a corner (min det {float(det[i].min()):.6g}); seeding refuses an "
                f"inverted or degenerate element")
        self._scale = scale
        #: facet 2d is xi_d = -1, facet 2d+1 is xi_d = +1
        self.facet_nodes = np.array([np.nonzero(self._corners[:, d] == sgn)[0]
                                     for d in range(ndm) for sgn in (-1.0, 1.0)])

    def _shape(self, xi: FloatArray) -> tuple[FloatArray, FloatArray]:
        """N (P, nn) and dN/dxi (P, nn, ndm) at reference points ``xi`` (P, ndm)."""
        c = self._corners                                   # (nn, ndm)
        ndm = c.shape[1]
        f = 1.0 + xi[:, None, :] * c[None, :, :]            # (P, nn, ndm)
        scale = 0.5 ** ndm
        n_val = scale * np.prod(f, axis=2)
        dn = np.empty(f.shape, dtype=np.float64)
        for k in range(ndm):
            others = np.prod(np.delete(f, k, axis=2), axis=2)
            dn[:, :, k] = scale * c[None, :, k] * others
        return n_val, dn

    def _jacobian(self, e: IntArray, xi: FloatArray) -> FloatArray:
        _, dn = self._shape(xi)
        jac: FloatArray = np.einsum("pai,paj->pij", self._verts[e], dn)
        return jac

    def inside(self, e: IntArray, pts: FloatArray) -> NDArray[np.bool_]:
        return self._invert(e, pts)[0]

    def facets(self, e: IntArray, pts: FloatArray) -> tuple[NDArray[np.bool_], FloatArray]:
        """Which facets each point lies on (``|xi_d| >= 1 - tol``, inside points only) and
        each facet's outward unit normal ``+-grad(xi_d)``: ``(P, F)`` and ``(P, F, ndm)``."""
        inside, xi = self._invert(e, pts)
        npts, ndm = pts.shape
        on = np.zeros((npts, 2 * ndm), dtype=bool)
        outward = np.zeros((npts, 2 * ndm, ndm), dtype=np.float64)
        if inside.any():
            grad = np.linalg.inv(self._jacobian(e[inside], xi[inside]))  # row d: grad xi_d
            unit = grad / np.linalg.norm(grad, axis=2, keepdims=True)
            for d in range(ndm):
                on[:, 2 * d] = xi[:, d] <= -1.0 + _INSIDE_TOL
                on[:, 2 * d + 1] = xi[:, d] >= 1.0 - _INSIDE_TOL
                outward[inside, 2 * d] = -unit[:, d, :]
                outward[inside, 2 * d + 1] = unit[:, d, :]
        on &= inside[:, None]
        return on, outward

    def _invert(self, e: IntArray, pts: FloatArray) -> tuple[NDArray[np.bool_], FloatArray]:
        """The inside flag and the reference coordinates ``xi`` of each point."""
        npts, ndm = pts.shape
        xi = np.zeros((npts, ndm), dtype=np.float64)
        active = np.ones(npts, dtype=bool)
        outside = np.zeros(npts, dtype=bool)
        converged = np.zeros(npts, dtype=bool)
        for _ in range(_NEWTON_MAXIT):
            act = np.nonzero(active)[0]
            if act.size == 0:
                break
            ea = e[act]
            n_val, dn = self._shape(xi[act])
            x = np.einsum("pa,pai->pi", n_val, self._verts[ea])
            jac = np.einsum("pai,paj->pij", self._verts[ea], dn)
            det = np.linalg.det(jac)
            singular = ~(det > 1e-14 * self._scale[ea])
            if singular.any():
                in_box = np.all(np.abs(xi[act]) <= 1.0 + _INSIDE_TOL, axis=1)
                fatal = singular & in_box
                if fatal.any():
                    i = int(act[np.nonzero(fatal)[0][0]])
                    raise MPMSeedError(
                        f"body {self._body.name!r}: element {int(self._eids[e[i]])} has a "
                        f"non-positive Jacobian inside the element; seeding refuses it")
                # singular outside the reference cell: the point is outside this element
                outside[act[singular]] = True
                active[act[singular]] = False
                keep = ~singular
                act, jac, x = act[keep], jac[keep], x[keep]
                if act.size == 0:
                    continue
            step = np.linalg.solve(jac, (x - pts[act])[:, :, None])[:, :, 0]
            xi[act] -= step
            done = np.max(np.abs(step), axis=1) < _NEWTON_TOL
            far = np.max(np.abs(xi[act]), axis=1) > _FAR_OUTSIDE
            converged[act[done]] = True
            outside[act[far & ~done]] = True
            active[act[done | far]] = False
        if active.any():
            i = int(np.nonzero(active)[0][0])
            raise MPMSeedError(
                f"body {self._body.name!r}: the inverse isoparametric map did not converge "
                f"in element {int(self._eids[e[i]])} at point {pts[i].tolist()}")
        inside: NDArray[np.bool_] = converged & ~outside
        inside &= np.all(np.abs(xi) <= 1.0 + _INSIDE_TOL, axis=1)
        return inside, xi


def _resolve_shared_points(
    kept: list[IntArray],
    bodies: Sequence[MPMBody],
    fem: Any,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
    sorted_ids: IntArray,
    order: IntArray,
    node_xyz: FloatArray,
) -> dict[tuple[str, str], int]:
    """Give each lattice point two bodies claim to the first-declared one, or refuse it
    (ADR 0134 Amendment 2). Edits ``kept`` in place; returns the per-pair counts.

    A shared point is on the interface of bodies ``i`` and ``j`` when it lies on a boundary
    facet of each (a facet no other element of that body shares) and two of those facets,
    one per body, face each other (outward normals with a negative dot product). Any other
    shared point lies inside at least one of the two bodies: an overlap, refused by name.
    """
    pairs: dict[tuple[int, int], IntArray] = {}
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            both = np.intersect1d(kept[i], kept[j], assume_unique=True)
            if both.size:
                pairs[(i, j)] = both
    if not pairs:
        return {}

    involved = sorted({b for pair in pairs for b in pair})
    normals: dict[int, tuple[IntArray, FloatArray]] = {}
    for b in involved:
        mine = np.unique(np.concatenate(
            [pts for pair, pts in pairs.items() if b in pair]))
        normals[b] = _boundary_normals(
            fem, bodies[b], grid, n, lattice_shape, sorted_ids, order, node_xyz, mine)

    for (i, j), both in pairs.items():
        facing = _facing(both, normals[i], normals[j])
        if not facing.all():
            bad = both[~facing]
            idx = np.array(np.unravel_index(int(bad[0]), lattice_shape, order="F"))
            spacing = np.asarray(grid.cell_size, dtype=np.float64) / n
            x = grid.lo + (idx + 0.5) * spacing
            raise MPMSeedError(
                f"bodies {bodies[i].name!r} and {bodies[j].name!r} share {both.size} "
                f"lattice point{'s' if both.size != 1 else ''} at n={n}; {bad.size} of them "
                f"{'is' if bad.size == 1 else 'are'} not on an interface between the two "
                f"(the bodies overlap there), e.g. {x.tolist()}; a particle needs one body")

    original = list(kept)
    interface: dict[tuple[str, str], int] = {}
    for j in range(1, len(kept)):
        lost = [pts for (_, jj), pts in pairs.items() if jj == j]
        if not lost:
            continue
        given_up = np.unique(np.concatenate(lost))
        owner = np.full(given_up.size, -1, dtype=np.int64)
        for i in range(j):
            mine = (owner < 0) & np.isin(given_up, original[i], assume_unique=True)
            owner[mine] = i
        for i in np.unique(owner):
            interface[(bodies[int(i)].name, bodies[j].name)] = int(np.count_nonzero(owner == i))
        kept[j] = np.setdiff1d(original[j], given_up, assume_unique=True)
        if kept[j].size == 0:
            raise MPMSeedError(
                f"body {bodies[j].name!r} (physical group {bodies[j].pg!r}) holds no lattice "
                f"point of its own at n={n}: each point it holds lies on its interface with "
                f"an earlier-declared body, which keeps it; refine n or the grid")
    return interface


def _boundary_normals(
    fem: Any,
    body: MPMBody,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
    sorted_ids: IntArray,
    order: IntArray,
    node_xyz: FloatArray,
    points: IntArray,
) -> tuple[IntArray, FloatArray]:
    """For the lattice points ``points`` (sorted linear indices), every boundary facet of
    ``body`` each lies on: rows ``(point, outward unit normal)``, sorted by point."""
    groups = list(_body_groups(fem, body, grid, sorted_ids, order, node_xyz))
    boundary = _boundary_facets(groups)
    rows_lin: list[IntArray] = []
    rows_nrm: list[FloatArray] = []
    for grp, bnd in zip(groups, boundary):
        for e_loc, idx, pts in _candidates(grp.lo_e, grp.hi_e, grid, n, lattice_shape):
            lin = np.ravel_multi_index(tuple(idx.T), lattice_shape, order="F").astype(np.int64)
            pos = np.clip(np.searchsorted(points, lin), 0, points.size - 1)
            sel = points[pos] == lin
            if not sel.any():
                continue
            e_sel = e_loc[sel]
            on, outward = grp.tester.facets(e_sel, pts[sel])
            p_i, f_i = np.nonzero(on & bnd[e_sel])
            rows_lin.append(lin[sel][p_i])
            rows_nrm.append(outward[p_i, f_i])
    if not rows_lin:
        return np.empty(0, dtype=np.int64), np.empty((0, grid.ndm), dtype=np.float64)
    lin_all = np.concatenate(rows_lin)
    srt = np.argsort(lin_all, kind="stable")
    return lin_all[srt], np.concatenate(rows_nrm)[srt]


def _boundary_facets(groups: Sequence[_Group]) -> list[NDArray[np.bool_]]:
    """Per group, ``(E, F)``: the facet belongs to one element of the body only.

    Facets are matched by their sorted node ids across every group of the body, so a
    ``tri3``/``quad4`` mix in 2-D shares its edges; a mesh that does not share nodes
    across a face (non-conforming) shows that face as boundary on both sides.
    """
    keys = [np.sort(grp.conn[:, grp.tester.facet_nodes], axis=2) for grp in groups]
    out: list[NDArray[np.bool_]] = [np.empty((0, 0), dtype=bool)] * len(groups)
    for width in sorted({int(k.shape[2]) for k in keys}):
        members = [gi for gi, k in enumerate(keys) if k.shape[2] == width]
        flat = np.concatenate([keys[gi].reshape(-1, width) for gi in members])
        _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
        once = counts[inverse.ravel()] == 1
        start = 0
        for gi in members:
            size = keys[gi].shape[0] * keys[gi].shape[1]
            out[gi] = once[start:start + size].reshape(keys[gi].shape[:2])
            start += size
    return out


def _facing(
    points: IntArray,
    a: tuple[IntArray, FloatArray],
    b: tuple[IntArray, FloatArray],
) -> NDArray[np.bool_]:
    """Per point: some boundary facet of body ``a`` and some of body ``b`` there have
    outward normals with a negative dot product (the bodies lie on opposite sides)."""
    lin_a, nrm_a = a
    lin_b, nrm_b = b
    lo_a = np.searchsorted(lin_a, points, side="left")
    cnt_a = np.searchsorted(lin_a, points, side="right") - lo_a
    lo_b = np.searchsorted(lin_b, points, side="left")
    cnt_b = np.searchsorted(lin_b, points, side="right") - lo_b
    per = cnt_a * cnt_b
    facing = np.zeros(points.size, dtype=bool)
    total = int(per.sum())
    if total == 0:
        return facing
    p = np.repeat(np.arange(points.size), per)
    k = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(per) - per, per)
    ia = lo_a[p] + k // cnt_b[p]
    ib = lo_b[p] + k % cnt_b[p]
    dots = np.einsum("pi,pi->p", nrm_a[ia], nrm_b[ib])
    np.logical_or.at(facing, p, dots < 0.0)
    return facing
