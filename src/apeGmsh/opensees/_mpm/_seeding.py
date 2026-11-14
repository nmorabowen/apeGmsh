"""Lattice seeding of MPM particles from FEMData alone (ADR 0134 D2).

The grid declares ``n`` points per cell per axis at the sub-cell centres. A point is kept
for a body when it lies inside an element of the body's physical group, tested on the
FEMData mesh in numpy: barycentric coordinates for ``tri3``/``tet4``, the inverse
isoparametric map (Newton) for ``quad4``/``hex8``. A point on a face shared by two
elements of one body is kept once; a point claimed by two bodies is refused. Every other
element family is refused by name.

Charter P3/P9: this module imports neither gmsh nor anything outside ``apeGmsh.mesh``.
``FEMData`` is read through its public ``nodes``/``elements`` surface only.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
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
    """

    counts: IntArray
    expected: int

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
        return (f"{int(self.counts.sum())} particles; {self.full_cells} full cells "
                f"({self.expected} each), {self.partial_cells} partial, "
                f"{self.empty_cells} empty")


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

    _refuse_shared_points(kept, bodies, grid, n, lattice_shape)

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
    report = UniformityReport(counts=counts, expected=n ** ndm)

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
    found: list[IntArray] = []
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
        if fam in ("tri3", "tet4"):
            tester = _SimplexTester(verts, eids, body)
        else:
            tester = _IsoTester(verts, eids, body)
        found.extend(_scan(tester, lo_e, hi_e, grid, n, lattice_shape))
    if not found:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(found))


def _scan(
    tester: _SimplexTester | _IsoTester,
    lo_e: FloatArray,
    hi_e: FloatArray,
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
) -> list[IntArray]:
    """Test every lattice point in each element's bounding box; return kept indices."""
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
    kept: list[IntArray] = []
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
            inside = tester.inside(e_loc, pts)
            if inside.any():
                kept.append(np.ravel_multi_index(
                    tuple(idx[inside].T), lattice_shape, order="F").astype(np.int64))
        start = stop
    return kept


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

    def inside(self, e: IntArray, pts: FloatArray) -> NDArray[np.bool_]:
        lam = np.einsum("pij,pj->pi", self._inv[e], pts - self._x0[e])
        lam0 = 1.0 - lam.sum(axis=1)
        inside: NDArray[np.bool_] = (lam.min(axis=1) >= -_INSIDE_TOL) & (lam0 >= -_INSIDE_TOL)
        return inside


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
        return np.einsum("pai,paj->pij", self._verts[e], dn)

    def inside(self, e: IntArray, pts: FloatArray) -> NDArray[np.bool_]:
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
        inside = converged & ~outside
        inside &= np.all(np.abs(xi) <= 1.0 + _INSIDE_TOL, axis=1)
        return inside


def _refuse_shared_points(
    kept: list[IntArray],
    bodies: Sequence[MPMBody],
    grid: MPMGrid,
    n: int,
    lattice_shape: tuple[int, ...],
) -> None:
    """A lattice point inside two bodies has no owner: refuse it by name."""
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            both = np.intersect1d(kept[i], kept[j], assume_unique=True)
            if both.size:
                idx = np.array(np.unravel_index(int(both[0]), lattice_shape, order="F"))
                spacing = np.asarray(grid.cell_size, dtype=np.float64) / n
                x = grid.lo + (idx + 0.5) * spacing
                raise MPMSeedError(
                    f"bodies {bodies[i].name!r} and {bodies[j].name!r} both hold "
                    f"{both.size} lattice point(s), e.g. {x.tolist()}; a particle needs "
                    f"one body (overlapping bodies, or an interface through a sub-cell "
                    f"centre)")
