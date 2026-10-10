"""
Station soil box: a soil box whose DRM layer sits on the stations of an
``.h5drm`` file (ADR 0118).

One builder makes two configurations from one shared interior:

* **DRM box** (``drm=True``): interior + a one-cell DRM layer between the
  dataset's two station shells + an exterior zone + a fixed or absorbing
  outer boundary. Every DRM-layer node is a station; no other node is
  within the H5DRM matching tolerance of one.
* **Absorbing box** (``drm=False``): the same interior wrapped directly by
  an ASD absorbing skin (base input), no DRM layer, no exterior.

The interior is a structured lattice on the station lines (the dataset's
inner shell and everything inside it), optionally with a hole holding a
**near-field block** (its own spacing, e.g. 2.5 m in plan and 1 m in depth)
tied to the lattice by embedded nodes, and a **pit void** (the building
basement) cut from the near-field block or, without one, from the lattice.
Both configurations build the interior through the same axes, so their
interior meshes are identical node for node.

The frame is the one ``pattern H5DRM`` applies in the fork
(``H5DRMLoadPattern.cpp``, ``do_coordinate_transformation = 1``)::

    xyz_model = T · ((xyz_station − drmbox_x0) · crd_scale) + x0

:class:`SoilLattice.from_h5drm` applies it to the stations once and reads the
lattice off the transformed coordinates, so the model can be in any units,
z-up or z-down, and offset to the building's origin, as long as ``T`` is a
signed permutation that keeps the station depth axis on model z.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import gmsh
import numpy as np

from ._axis1d import Axis1D
from .plane_wave_box import AbsorbingSkinResult, _btype_for, _emit_skin_pgs

if TYPE_CHECKING:
    from .._session import _SessionBase  # pragma: no cover


_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
TOLERANCE_FRACTION = 4e-4
"""Default H5DRM ``distance_tolerance`` as a fraction of the smallest station
gap: 1 mm on a 2.5 m grid in millimetres (1e-3 m in metres)."""
_SKIN = ("L", "R", "F", "K", "B")


# ─────────────────────────────────────────────────────────────────────
# Frame and lattice
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StationFrame:
    """The ``pattern H5DRM`` frame contract for one ``.h5drm`` file."""

    h5drm: str
    crd_scale: float
    transform: tuple[tuple[float, ...], ...]
    """3×3 row-major ``T`` (a signed permutation)."""
    x0: tuple[float, float, float]
    center: tuple[float, float, float]
    """``DRM_Metadata/drmbox_x0`` in station units (the fork's origin)."""
    distance_tolerance: float
    """Node-matching tolerance in model units."""

    def to_model(self, xyz: np.ndarray) -> np.ndarray:
        """Station → model coordinates, in the fork's order of operations."""
        T = np.asarray(self.transform, dtype=float)
        v = (np.asarray(xyz, dtype=float) - np.asarray(self.center)) * self.crd_scale
        return v @ T.T + np.asarray(self.x0, dtype=float)

    def pattern_kwargs(self) -> dict:
        """Keyword arguments for ``ops.pattern.H5DRM(...)`` (add ``factor``)."""
        return {
            "h5drm": self.h5drm,
            "crd_scale": float(self.crd_scale),
            "distance_tolerance": float(self.distance_tolerance),
            "transform": self.transform,
            "x0": self.x0,
        }


@dataclass(frozen=True)
class SoilLattice:
    """Interior lattice lines (model frame): the mesh rule both configurations share.

    ``x`` / ``y`` / ``z`` are the interior lattice lines, ascending, from one
    interior face to the other (for a DRM file: the inner station shell).
    ``surface`` names the free-surface end of ``z`` (``"max"`` for z-up,
    ``"min"`` for z-down). ``layer`` holds the outer-shell lines
    ``(x_lo, x_hi, y_lo, y_hi, z_far)`` of a DRM file; ``None`` otherwise.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    z: tuple[float, ...]
    surface: Literal["max", "min"] = "max"
    layer: tuple[float, float, float, float, float] | None = None
    frame: StationFrame | None = None
    stations: np.ndarray | None = field(default=None, repr=False, compare=False)
    """Station coordinates in the model frame, ``(N, 3)``."""
    internal: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for ax, v in (("x", self.x), ("y", self.y), ("z", self.z)):
            a = np.asarray(v, dtype=float)
            if a.ndim != 1 or len(a) < 2 or np.any(np.diff(a) <= 0):
                raise ValueError(
                    f"SoilLattice: {ax} lines must be >= 2 strictly ascending "
                    f"values, got {v!r}."
                )
        if self.surface not in ("max", "min"):
            raise ValueError(
                f"SoilLattice: surface must be 'max' or 'min', got {self.surface!r}."
            )

    # ── constructors ────────────────────────────────────────────────
    @classmethod
    def regular(
        cls,
        *,
        lo: tuple[float, float, float],
        hi: tuple[float, float, float],
        spacing: float | tuple[float, float, float],
        surface: Literal["max", "min"] = "max",
    ) -> SoilLattice:
        """A uniform interior lattice from a box and a spacing (no file)."""
        hs = (spacing,) * 3 if np.isscalar(spacing) else tuple(spacing)  # type: ignore[arg-type]
        lines = []
        for ax, a, b, h in zip("xyz", lo, hi, hs):
            n = (float(b) - float(a)) / float(h)
            if float(h) <= 0 or n < 1 or abs(n - round(n)) > 1e-6:
                raise ValueError(
                    f"SoilLattice.regular: the {ax} extent {b} - {a} is not a "
                    f"whole number of spacings {h}."
                )
            lines.append(tuple(np.linspace(float(a), float(b), round(n) + 1)))
        return cls(x=lines[0], y=lines[1], z=lines[2], surface=surface)

    @classmethod
    def from_h5drm(
        cls,
        h5drm: str,
        *,
        crd_scale: float = 1000.0,
        transform=None,
        x0: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance_tolerance: float | None = None,
    ) -> SoilLattice:
        """Read the two-shell station grid of an ``.h5drm`` file.

        ``crd_scale`` / ``transform`` / ``x0`` are the values the deck's
        ``pattern H5DRM`` will carry. Accepted: the ShakerMaker DRMBox layout
        (an outer shell on the four sides and the bottom of a lattice, and the
        inner shell one line inward), any spacing per axis, uniform or not.
        Refused, with the counts: anything else, a ``T`` that is not a signed
        permutation or moves the depth axis off model z, inconsistent
        ``internal`` flags.
        """
        T = _check_transform(transform)
        xyz, internal, center = _read_h5drm(h5drm)
        frame0 = StationFrame(
            h5drm=str(h5drm), crd_scale=float(crd_scale), transform=T,
            x0=tuple(float(v) for v in x0),  # type: ignore[arg-type]
            center=center, distance_tolerance=0.0,
        )
        model = frame0.to_model(xyz)
        lines = [_cluster(model[:, i], "xyz"[i]) for i in range(3)]
        idx = np.stack(
            [_nearest_index(lines[i], model[:, i]) for i in range(3)], axis=1)
        nx, ny, nz = (len(v) for v in lines)
        if min(nx, ny) < 4 or nz < 3:
            raise ValueError(
                f"from_h5drm({h5drm!r}): the station lattice is {nx}x{ny}x{nz}; a "
                f"two-shell DRM box needs at least 4 lines in x and y and 3 in z."
            )
        if len(xyz) == nx * ny * nz:
            raise ValueError(
                f"from_h5drm({h5drm!r}): the {len(xyz)} stations fill their "
                f"{nx}x{ny}x{nz} lattice (a complete grid, not the two-shell DRM "
                f"layout): use g.parts.add_DRM_box_from_h5drm for it."
            )
        # the bottom is the z end whose plane is full (the surface end only
        # carries the side shells)
        n_lo = int(np.sum(idx[:, 2] == 0))
        n_hi = int(np.sum(idx[:, 2] == nz - 1))
        full = nx * ny
        if (n_lo == full) == (n_hi == full):
            raise ValueError(
                f"from_h5drm({h5drm!r}): cannot tell the bottom from the free "
                f"surface: the two z-end planes hold {n_lo} and {n_hi} stations "
                f"(a full plane is {full}). Expected exactly one full plane "
                f"(the bottom shell)."
            )
        kb = 0 if n_lo == full else nz - 1
        step = 1 if kb == 0 else -1
        outer, inner = _two_shell_masks(nx, ny, nz, kb, step)
        expected = outer | inner
        got = np.zeros((nx, ny, nz), dtype=int)
        np.add.at(got, (idx[:, 0], idx[:, 1], idx[:, 2]), 1)
        dup = int(np.sum(got > 1))
        missing = int(np.sum(expected & (got == 0)))
        extra = int(np.sum(~expected & (got > 0)))
        if dup or missing or extra:
            raise ValueError(
                f"from_h5drm({h5drm!r}): the {len(xyz)} stations are not the "
                f"two-shell DRM layout of their {nx}x{ny}x{nz} lattice "
                f"(expected {int(expected.sum())}: {int(outer.sum())} outer + "
                f"{int(inner.sum())} inner): {missing} missing, {extra} extra, "
                f"{dup} duplicated."
            )
        want_internal = inner[idx[:, 0], idx[:, 1], idx[:, 2]]
        bad = int(np.sum(want_internal != internal))
        if bad:
            raise ValueError(
                f"from_h5drm({h5drm!r}): {bad} stations carry an 'internal' flag "
                f"that contradicts their shell (outer shell = 0, inner = 1); the "
                f"fork would apply the wrong DRM force sign there."
            )
        gaps = np.concatenate([np.diff(v) for v in lines])
        h_min = float(gaps.min())
        tol = (float(f"{TOLERANCE_FRACTION * h_min:.3g}")
               if distance_tolerance is None else float(distance_tolerance))
        if not (0.0 < tol < 0.5 * h_min):
            raise ValueError(
                f"from_h5drm: distance_tolerance {tol:g} must be in (0, "
                f"{0.5 * h_min:g}) (half the smallest station gap), or no node "
                f"would match or one node would match two stations."
            )
        X, Y, Z = lines
        if step == 1:                       # bottom at z min → surface at max
            zint = Z[1:]
            surface: Literal["max", "min"] = "max"
        else:                               # bottom at z max → surface at min
            zint = Z[:-1]
            surface = "min"
        frame = StationFrame(
            h5drm=frame0.h5drm, crd_scale=frame0.crd_scale, transform=T,
            x0=frame0.x0, center=center, distance_tolerance=tol,
        )
        return cls(
            x=tuple(X[1:-1]), y=tuple(Y[1:-1]), z=tuple(zint), surface=surface,
            layer=(X[0], X[-1], Y[0], Y[-1], Z[kb]),
            frame=frame, stations=model, internal=internal,
        )

    # ── derived ─────────────────────────────────────────────────────
    @property
    def z_surface(self) -> float:
        return float(self.z[-1] if self.surface == "max" else self.z[0])

    @property
    def z_far(self) -> float:
        return float(self.z[0] if self.surface == "max" else self.z[-1])


def _check_transform(transform) -> tuple[tuple[float, ...], ...]:
    T = _IDENTITY if transform is None else tuple(
        tuple(float(c) for c in row) for row in transform)
    A = np.asarray(T, dtype=float)
    ok = (A.shape == (3, 3)
          and np.all(np.isin(A, (-1.0, 0.0, 1.0)))
          and np.all(np.count_nonzero(A, axis=0) == 1)
          and np.all(np.count_nonzero(A, axis=1) == 1))
    if not ok:
        raise ValueError(
            f"station box: transform must be a 3x3 signed permutation "
            f"(axis-aligned; a box cannot be sliced along rotated axes), got {T!r}."
        )
    if A[2, 2] == 0.0:
        raise ValueError(
            f"station box: transform {T!r} moves the station depth axis off "
            f"model z; the free surface must be a z plane."
        )
    return tuple(tuple(0.0 if c == 0 else c for c in row) for row in T)


def _read_h5drm(path: str):
    import h5py

    with h5py.File(path, "r") as f:
        for need in ("DRM_Data/xyz", "DRM_Data/internal", "DRM_Metadata/drmbox_x0"):
            grp, ds = need.split("/")
            if grp not in f or ds not in f[grp]:
                raise ValueError(
                    f"{path!r}: missing {need}, which the fork's H5DRM pattern "
                    f"reads unconditionally."
                )
        xyz = np.asarray(f["DRM_Data/xyz"][:], dtype=float)
        internal = np.asarray(f["DRM_Data/internal"][:]).astype(bool)
        x0c = np.asarray(f["DRM_Metadata/drmbox_x0"][:], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(internal) != len(xyz):
        raise ValueError(
            f"{path!r}: DRM_Data/xyz must be (N, 3) with N internal flags, got "
            f"{xyz.shape} and {internal.shape}."
        )
    return xyz, internal, tuple(float(v) for v in x0c.reshape(3))


def _cluster(vals: np.ndarray, ax: str) -> tuple[float, ...]:
    """Distinct line coordinates along one axis (cluster mean per line)."""
    s = np.sort(np.asarray(vals, dtype=float))
    span = float(s[-1] - s[0])
    tol = 1e-7 * max(span, 1e-300)
    cut = np.flatnonzero(np.diff(s) > tol) + 1
    groups = np.split(s, cut)
    lines = tuple(float(g.mean()) for g in groups)
    if len(lines) > 1:
        widths = max(float(g[-1] - g[0]) for g in groups)
        if widths > 1e-3 * float(np.diff(lines).min()):
            raise ValueError(
                f"station box: the stations do not sit on {ax} lines (a line "
                f"scatters by {widths:g}, comparable to the gap)."
            )
    return lines


def _nearest_index(lines, vals) -> np.ndarray:
    a = np.asarray(lines)
    i = np.clip(np.searchsorted(a, vals), 1, len(a) - 1)
    left = np.abs(vals - a[i - 1]) <= np.abs(vals - a[i])
    return np.where(left, i - 1, i)


def _two_shell_masks(nx, ny, nz, kb, step):
    i = np.arange(nx)[:, None, None]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[None, None, :]
    outer = (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1) | (k == kb)
    outer = np.broadcast_to(outer, (nx, ny, nz))
    inner = ((i == 1) | (i == nx - 2) | (j == 1) | (j == ny - 2) | (k == kb + step))
    inner = np.broadcast_to(inner, (nx, ny, nz)) & ~outer
    return outer, inner


# ─────────────────────────────────────────────────────────────────────
# Specs
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NearField:
    """A near-field block in a hole of the interior lattice, tied to it.

    ``lo`` / ``hi`` must sit on interior lattice lines, reach the free
    surface, and stay at least one lattice cell inside the interior on the
    sides and the bottom (so no near-field node is near a station).
    ``size`` is the target spacing per axis; the block's lines also pass
    through the pit planes and any ``lines`` given per axis.
    ``couple="embedded"`` declares ``g.constraints.embedded`` (the near-field
    interface nodes into the lattice hole faces: ``ASDEmbeddedNodeElement``,
    as the rev2_m36 decks); ``None`` leaves the coupling to the caller.
    """

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    size: float | tuple[float, float, float]
    lines: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] = ((), (), ())
    couple: Literal["embedded"] | None = "embedded"
    stiffness: float | str = "auto"


@dataclass(frozen=True)
class Pit:
    """A void (the building basement): a box that reaches the free surface."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass(frozen=True)
class Exterior:
    """The exterior zone outside the DRM layer (or the interior).

    ``thickness`` is ``t`` or ``(lateral, bottom)``; ``size`` the target
    element size across it (the tangential spacing follows the lattice).
    """

    thickness: float | tuple[float, float]
    size: float


@dataclass(frozen=True)
class StationCheck:
    """Pre-mesh check of the DRM layer against the stations."""

    n_stations: int
    n_drm_nodes: int
    max_distance: float
    tolerance: float


@dataclass(frozen=True)
class StationSoilBoxResult:
    """PG names, frame contract and expected counts of a station soil box."""

    domain_pg: str
    """Every soil hex (lattice + near field + DRM layer + exterior): the
    ``stdBrick`` target."""
    interior_pg: str
    """Lattice interior + near field: identical in both configurations."""
    lattice_pg: str
    nearfield_pg: str = ""
    drm_pg: str = ""
    exterior_pg: str = ""
    free_surface_pg: str = ""
    boundary_pg: str = ""
    """Outer sides + bottom of the soil (dim 2), the ``ops.fix`` target of
    ``boundary="fixed"``; empty when a skin wraps the soil."""
    pit_pgs: dict[str, str] = field(default_factory=dict)
    """``{"bottom", "walls", "all"}`` → dim-2 PG of the pit faces."""
    interface_pgs: tuple[str, str] | None = None
    """``(lattice host faces, near-field faces)``, dim 2."""
    skin: AbsorbingSkinResult | None = None
    frame: StationFrame | None = None
    axes: dict[str, Axis1D] = field(default_factory=dict)
    nearfield_axes: dict[str, Axis1D] = field(default_factory=dict)
    expected_hex: dict[str, int] = field(default_factory=dict)
    """Hex count per volume PG (and ``"skin"``), from the axes."""
    expected_nodes: int = 0
    """Mesh node count (lattice and near-field nodes are not shared)."""
    station_check: StationCheck | None = None


# ─────────────────────────────────────────────────────────────────────
# Axes
# ─────────────────────────────────────────────────────────────────────

def _runs(lines, breaks, tol, region="int"):
    """Split a line sequence into uniform runs, also cutting at ``breaks``."""
    a = np.asarray(lines, dtype=float)
    cuts = {0, len(a) - 1}
    for b in breaks:
        hit = np.flatnonzero(np.abs(a - b) <= tol)
        cuts.add(int(hit[0]))
    g = np.diff(a)
    for i in range(1, len(g)):
        if abs(g[i] - g[i - 1]) > 1e-6 * g[i - 1]:
            cuts.add(i)
    c = sorted(cuts)
    return [(region, float(a[p]), float(a[q]), q - p) for p, q in itertools.pairwise(c)]


def _sized(region, lo, hi, size):
    return (region, float(lo), float(hi), max(1, round((hi - lo) / size)))


def _first_cell(segments, at_lo: bool) -> float:
    _reg, lo, hi, n = segments[0] if at_lo else segments[-1]
    return (hi - lo) / n


def _on_lines(v, lines, tol, what):
    a = np.asarray(lines)
    if np.min(np.abs(a - v)) > tol:
        near = a[np.argsort(np.abs(a - v))[:2]]
        raise ValueError(
            f"station box: {what} = {v:g} is not on an interior lattice line "
            f"(nearest {sorted(float(x) for x in near)}). Move it onto a line, "
            f"or put the feature in a NearField block."
        )


# ─────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────

def build_station_soil_box(
    session: _SessionBase,
    lattice: SoilLattice,
    *,
    drm: bool = True,
    exterior: Exterior | None = None,
    boundary: Literal["fixed", "absorbing", "none"] = "fixed",
    skin_thickness: float | tuple[float, float, float] | None = None,
    nearfield: NearField | None = None,
    pit: Pit | None = None,
    name: str = "soil",
    apply_transfinite: bool = True,
) -> StationSoilBoxResult:
    """Build a station soil box in the live session (see the module docstring)."""
    if boundary not in ("fixed", "absorbing", "none"):
        raise ValueError(
            f"station box: boundary must be 'fixed', 'absorbing' or 'none', got "
            f"{boundary!r}.")
    if drm:
        if lattice.layer is None or lattice.frame is None:
            raise ValueError(
                "station box: drm=True needs a lattice read with "
                "SoilLattice.from_h5drm (it carries the station shells).")
        if exterior is None:
            raise ValueError(
                "station box: drm=True needs an Exterior: a DRM box with no "
                "exterior zone diverges, and a boundary on the outer station "
                "shell would be swept into the DRM force set (ADR 0066).")

    X, Y, Z = (np.asarray(v, dtype=float) for v in (lattice.x, lattice.y, lattice.z))
    up = lattice.surface == "max"
    zs, zf = lattice.z_surface, lattice.z_far
    span = max(X[-1] - X[0], Y[-1] - Y[0], Z[-1] - Z[0])
    tol = 1e-9 * span
    gtol = 1e-6 * span          # geometric face/plane tolerance

    # ── near field / pit validation ────────────────────────────────
    hole = None                 # the lattice hole box (lo, hi)
    if nearfield is not None:
        nlo, nhi = _box(nearfield.lo, nearfield.hi, "NearField")
        for i, ax in enumerate("xyz"):
            lines = (X, Y, Z)[i]
            _on_lines(nlo[i], lines, tol, f"NearField lo[{ax}]")
            _on_lines(nhi[i], lines, tol, f"NearField hi[{ax}]")
        _check_reaches_surface(nlo, nhi, zs, tol, "NearField")
        far_nf = nlo[2] if up else nhi[2]
        if (nlo[0] <= X[0] + tol or nhi[0] >= X[-1] - tol
                or nlo[1] <= Y[0] + tol or nhi[1] >= Y[-1] - tol
                or abs(far_nf - zf) <= tol):
            raise ValueError(
                "station box: the NearField block must stay at least one lattice "
                "cell inside the interior on the sides and the bottom (author "
                "decision A6): its nodes would otherwise sit on the DRM shell or "
                "the skin.")
        hole = (nlo, nhi)
    if pit is not None:
        plo, phi = _box(pit.lo, pit.hi, "Pit")
        _check_reaches_surface(plo, phi, zs, tol, "Pit")
        if nearfield is not None:
            nlo, nhi = hole  # type: ignore[misc]
            far_p, far_n = (plo[2], nlo[2]) if up else (phi[2], nhi[2])
            inside = (nlo[0] < plo[0] - tol and phi[0] < nhi[0] - tol
                      and nlo[1] < plo[1] - tol and phi[1] < nhi[1] - tol
                      and abs(far_p - far_n) > tol
                      and (far_p > far_n if up else far_p < far_n))
            if not inside:
                raise ValueError(
                    "station box: the Pit must sit strictly inside the NearField "
                    "block (sides and bottom).")
        else:
            for i, ax in enumerate("xyz"):
                lines = (X, Y, Z)[i]
                _on_lines(plo[i], lines, tol, f"Pit lo[{ax}]")
                _on_lines(phi[i], lines, tol, f"Pit hi[{ax}]")
            far_p = plo[2] if up else phi[2]
            if (plo[0] <= X[0] + tol or phi[0] >= X[-1] - tol
                    or plo[1] <= Y[0] + tol or phi[1] >= Y[-1] - tol
                    or abs(far_p - zf) <= tol):
                raise ValueError(
                    "station box: the Pit must sit strictly inside the interior.")
            hole = (plo, phi)

    # ── lattice axes ───────────────────────────────────────────────
    hb = [(), (), ()]
    if hole is not None:
        hb = [(hole[0][i], hole[1][i]) for i in range(3)]
    ints = [
        _runs(X, hb[0], tol), _runs(Y, hb[1], tol),
        _runs(Z, [b for b in hb[2] if abs(b - zs) > tol], tol),
    ]
    if exterior is not None:
        t = exterior.thickness
        t_lat, t_bot = (float(t), float(t)) if np.isscalar(t) else map(float, t)  # type: ignore[arg-type]
        if t_lat <= 0 or t_bot <= 0 or exterior.size <= 0:
            raise ValueError("station box: Exterior thickness and size must be > 0.")
    lay = lattice.layer if drm else None

    def lateral(i, lines_int, lo_face, hi_face, skin_lo, skin_hi, ts):
        segs = list(lines_int)
        lo, hi = lines_int[0][1], lines_int[-1][2]
        if lay is not None:
            segs = [("drm", lo_face, lo, 1)] + segs + [("drm", hi, hi_face, 1)]
            lo, hi = lo_face, hi_face
        if exterior is not None:
            segs = ([_sized("ext", lo - t_lat, lo, exterior.size)] + segs
                    + [_sized("ext", hi, hi + t_lat, exterior.size)])
            lo, hi = lo - t_lat, hi + t_lat
        if boundary == "absorbing":
            tl = ts if ts is not None else _first_cell(segs, True)
            th = ts if ts is not None else _first_cell(segs, False)
            segs = [(skin_lo, lo - tl, lo, 1)] + segs + [(skin_hi, hi, hi + th, 1)]
        return Axis1D("xyz"[i], tuple(segs))

    if skin_thickness is None or np.isscalar(skin_thickness):
        tsk = (skin_thickness,) * 3
    else:
        tsk = tuple(skin_thickness)  # type: ignore[arg-type]
    ax = lateral(0, ints[0], *(lay[0:2] if lay else (None, None)), "L", "R", tsk[0])
    ay = lateral(1, ints[1], *(lay[2:4] if lay else (None, None)), "F", "K", tsk[1])

    # z: only the far (bottom) side grows
    zsegs = list(ints[2])
    if up:
        far = zsegs[0][1]
        if lay is not None:
            zsegs = [("drm", lay[4], far, 1)] + zsegs
            far = lay[4]
        if exterior is not None:
            zsegs = [_sized("ext", far - t_bot, far, exterior.size)] + zsegs
            far -= t_bot
        if boundary == "absorbing":
            tb = tsk[2] if tsk[2] is not None else _first_cell(zsegs, True)
            zsegs = [("B", far - tb, far, 1)] + zsegs
    else:
        far = zsegs[-1][2]
        if lay is not None:
            zsegs = zsegs + [("drm", far, lay[4], 1)]
            far = lay[4]
        if exterior is not None:
            zsegs = zsegs + [_sized("ext", far, far + t_bot, exterior.size)]
            far += t_bot
        if boundary == "absorbing":
            tb = tsk[2] if tsk[2] is not None else _first_cell(zsegs, False)
            zsegs = zsegs + [("B", far, far + tb, 1)]
    az = Axis1D("z", tuple(zsegs))

    # ── near-field axes ────────────────────────────────────────────
    nfax: dict[str, Axis1D] = {}
    if nearfield is not None:
        nlo, nhi = hole  # type: ignore[misc]
        sz = ((nearfield.size,) * 3 if np.isscalar(nearfield.size)
              else tuple(nearfield.size))  # type: ignore[arg-type]
        for i, a in enumerate("xyz"):
            br = {nlo[i], nhi[i]}
            br.update(float(v) for v in nearfield.lines[i] if nlo[i] < v < nhi[i])
            if pit is not None:
                br.update((float(pit.lo[i]), float(pit.hi[i])))
            br = sorted(br)
            segs = []
            for p, q in itertools.pairwise(br):
                if q - p <= tol:
                    continue
                inpit = (pit is not None and pit.lo[i] - tol <= p
                         and q <= pit.hi[i] + tol)
                segs.append(_sized("pit" if inpit else "nf", p, q, float(sz[i])))
            nfax[a] = Axis1D(a, tuple(segs))

    # ── geometry ───────────────────────────────────────────────────
    queries = session.model.queries
    physical = session.physical
    p = f"{name}_" if name else ""

    lat_vols = _slice_box(session, ax, ay, az)
    cls: dict[str, list[int]] = {"lattice": [], "drm": [], "ext": []}
    skin_cells: dict[str, list[int]] = {}
    per_vol: list[tuple[int, int, int, int]] = []
    removed: list[int] = []
    for vt in lat_vols:
        c = queries.center_of_mass(vt, dim=3)
        rx, ry, rz = ax.region_of(c[0]), ay.region_of(c[1]), az.region_of(c[2])
        bt = _btype_for(rx, ry, rz)
        if bt:
            skin_cells.setdefault(bt, []).append(vt)
        elif "ext" in (rx, ry, rz):
            cls["ext"].append(vt)
        elif "drm" in (rx, ry, rz):
            cls["drm"].append(vt)
        elif hole is not None and _inside(c, hole[0], hole[1]):
            removed.append(vt)
            continue
        else:
            cls["lattice"].append(vt)
        per_vol.append((vt, ax.count_for(c[0]), ay.count_for(c[1]), az.count_for(c[2])))
    if removed:
        queries.remove(removed, dim=3, recursive=True)

    nf_vols: list[int] = []
    if nearfield is not None:
        before = {int(t) for _d, t in gmsh.model.getEntities(3)}
        nb = _slice_box(session, nfax["x"], nfax["y"], nfax["z"])
        assert not (set(nb) & before)
        pit_cells = []
        for vt in nb:
            c = queries.center_of_mass(vt, dim=3)
            regs = (nfax["x"].region_of(c[0]), nfax["y"].region_of(c[1]),
                    nfax["z"].region_of(c[2]))
            if regs == ("pit", "pit", "pit"):
                pit_cells.append(vt)
                continue
            nf_vols.append(vt)
            per_vol.append((vt, nfax["x"].count_for(c[0]), nfax["y"].count_for(c[1]),
                            nfax["z"].count_for(c[2])))
        if pit_cells:
            queries.remove(pit_cells, dim=3, recursive=True)

    # ── volume PGs ─────────────────────────────────────────────────
    def add(dim, ents, base):
        if not ents:
            return ""
        physical.add(dim, sorted(set(ents)), name=f"{p}{base}")
        return f"{p}{base}"

    soil = cls["lattice"] + nf_vols + cls["drm"] + cls["ext"]
    domain_pg = add(3, soil, "domain")
    interior_pg = add(3, cls["lattice"] + nf_vols, "interior")
    lattice_pg = add(3, cls["lattice"], "lattice")
    nearfield_pg = add(3, nf_vols, "nearfield")
    drm_pg = add(3, cls["drm"], "drm")
    exterior_pg = add(3, cls["ext"], "exterior")

    skin = None
    if skin_cells:
        (s_pg, s_pgs, skin_pgs, skin_all, bottom, by_layer) = _emit_skin_pgs(
            physical, dim=3, n_layers=1, soil_by_layer={0: soil},
            skin_by_layer={0: skin_cells}, pg_name=lambda b: f"{p}{b}",
            soil_pg_name=domain_pg,
        )
        skin = AbsorbingSkinResult(
            soil_pg=s_pg, skin_pgs=skin_pgs, skin_all_pg=skin_all,
            bottom_pgs=bottom, free_surface_pg=f"{p}free_surface",
            axes={"x": ax, "y": ay, "z": az},
            center=(0.5 * (ax.lo + ax.hi), 0.5 * (ay.lo + ay.hi), zs),
            n_layers=1, soil_pgs=s_pgs, skin_pgs_by_layer=by_layer, ndm=3,
        )

    # ── surface PGs ────────────────────────────────────────────────
    def faces_of(vols):
        out = []
        for _d, t in gmsh.model.getBoundary(
                [(3, v) for v in vols], combined=True, oriented=False):
            ft = abs(int(t))
            out.append((ft, queries.center_of_mass(ft, dim=2)))
        return out

    soil_faces = faces_of(soil)
    free = [ft for ft, c in soil_faces if abs(c[2] - zs) <= gtol]
    free_surface_pg = add(2, free, "free_surface")
    boundary_pg = ""
    if boundary != "absorbing":
        dlo = (ax.lo, ay.lo, az.lo)
        dhi = (ax.hi, ay.hi, az.hi)
        zfar_dom = az.lo if up else az.hi
        outer = [ft for ft, c in soil_faces
                 if (abs(c[0] - dlo[0]) <= gtol or abs(c[0] - dhi[0]) <= gtol
                     or abs(c[1] - dlo[1]) <= gtol or abs(c[1] - dhi[1]) <= gtol
                     or abs(c[2] - zfar_dom) <= gtol)]
        boundary_pg = add(2, outer, "boundary")

    pit_pgs: dict[str, str] = {}
    if pit is not None:
        host = nf_vols if nearfield is not None else cls["lattice"]
        plo, phi = _box(pit.lo, pit.hi, "Pit")
        far_p = plo[2] if up else phi[2]
        bottom_f, walls_f = [], []
        for ft, c in faces_of(host):
            if not _inside(c, plo, phi, slack=gtol):
                continue
            if abs(c[2] - far_p) <= gtol:
                bottom_f.append(ft)
            elif (abs(c[0] - plo[0]) <= gtol or abs(c[0] - phi[0]) <= gtol
                  or abs(c[1] - plo[1]) <= gtol or abs(c[1] - phi[1]) <= gtol):
                walls_f.append(ft)
        pit_pgs = {"bottom": add(2, bottom_f, "pit_bottom"),
                   "walls": add(2, walls_f, "pit_walls"),
                   "all": add(2, bottom_f + walls_f, "pit")}

    interface = None
    if nearfield is not None:
        nlo, nhi = hole  # type: ignore[misc]
        far_n = nlo[2] if up else nhi[2]

        def on_nf_planes(c):
            if not _inside(c, nlo, nhi, slack=gtol):
                return False
            return (abs(c[0] - nlo[0]) <= gtol or abs(c[0] - nhi[0]) <= gtol
                    or abs(c[1] - nlo[1]) <= gtol or abs(c[1] - nhi[1]) <= gtol
                    or abs(c[2] - far_n) <= gtol)

        host_f = [ft for ft, c in faces_of(cls["lattice"]) if on_nf_planes(c)]
        emb_f = [ft for ft, c in faces_of(nf_vols) if on_nf_planes(c)]
        interface = (add(2, host_f, "lattice_interface"),
                     add(2, emb_f, "nearfield_interface"))
        if nearfield.couple == "embedded":
            session.constraints.embedded(
                interface[0], interface[1], stiffness=nearfield.stiffness,
                name=f"{p}nearfield_tie",
            )
        elif nearfield.couple is not None:
            raise ValueError(
                f"station box: NearField.couple must be 'embedded' or None, got "
                f"{nearfield.couple!r}.")

    if apply_transfinite:
        structured = session.mesh.structured
        for vt, cx, cy, cz in per_vol:
            structured.set_transfinite((3, vt), n=(cx + 1, cy + 1, cz + 1),
                                       recombine=True)

    # ── expected counts + station check ────────────────────────────
    expected_hex, expected_nodes, drm_nodes = _expected(
        ax, ay, az, hole, nfax, up=up)
    check = None
    if drm and lattice.stations is not None and lattice.frame is not None:
        check = _station_check(drm_nodes, lattice.stations,
                               lattice.frame.distance_tolerance)

    return StationSoilBoxResult(
        domain_pg=domain_pg, interior_pg=interior_pg, lattice_pg=lattice_pg,
        nearfield_pg=nearfield_pg, drm_pg=drm_pg, exterior_pg=exterior_pg,
        free_surface_pg=free_surface_pg, boundary_pg=boundary_pg,
        pit_pgs=pit_pgs, interface_pgs=interface, skin=skin,
        frame=lattice.frame if drm else None,
        axes={"x": ax, "y": ay, "z": az}, nearfield_axes=nfax,
        expected_hex=expected_hex, expected_nodes=expected_nodes,
        station_check=check,
    )


def _box(lo, hi, what):
    a = np.asarray(lo, dtype=float)
    b = np.asarray(hi, dtype=float)
    if a.shape != (3,) or b.shape != (3,) or np.any(b <= a):
        raise ValueError(f"station box: {what} lo/hi must be 3-vectors with hi > lo.")
    return a, b


def _check_reaches_surface(lo, hi, zs, tol, what):
    if not (abs(hi[2] - zs) <= tol or abs(lo[2] - zs) <= tol):
        raise ValueError(
            f"station box: the {what} must reach the free surface z = {zs:g} "
            f"(got z {lo[2]:g}..{hi[2]:g}).")


def _inside(c, lo, hi, slack=0.0) -> bool:
    return all(lo[i] - slack <= c[i] <= hi[i] + slack for i in range(3))


def _slice_box(session, ax: Axis1D, ay: Axis1D, az: Axis1D) -> list[int]:
    """One box sliced at every axis break; returns its sub-volume tags."""
    geom = session.model.geometry
    before = {int(t) for _d, t in gmsh.model.getEntities(3)}
    geom.add_box(ax.lo, ay.lo, az.lo, ax.size, ay.size, az.size)

    def vols():
        return sorted({int(t) for _d, t in gmsh.model.getEntities(3)} - before)

    for a, axis in ((ax, "x"), (ay, "y"), (az, "z")):
        for off in a.slice_offsets():
            geom.slice(target=vols(), axis=axis, offset=float(off))
    return vols()


def _axis_nodes(a: Axis1D):
    """Node coordinates and per-cell region labels of an axis."""
    nodes = [a.segments[0][1]]
    regs: list[str] = []
    for reg, lo, hi, n in a.segments:
        nodes.extend(lo + (hi - lo) * np.arange(1, n + 1) / n)
        regs.extend([reg] * n)
    return np.asarray(nodes, dtype=float), np.asarray(regs)


def _expected(ax, ay, az, hole, nfax, *, up):
    """Hex counts per PG and the node count, from the axes alone."""
    (nx_, rx), (ny_, ry), (nz_, rz) = (_axis_nodes(a) for a in (ax, ay, az))
    RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
    skin = np.isin(RX, ("L", "R")) | np.isin(RY, ("F", "K")) | (RZ == "B")
    ext = ~skin & ((RX == "ext") | (RY == "ext") | (RZ == "ext"))
    drm = ~skin & ~ext & ((RX == "drm") | (RY == "drm") | (RZ == "drm"))
    lat = ~skin & ~ext & ~drm
    if hole is not None:
        cx = 0.5 * (nx_[1:] + nx_[:-1])
        cy = 0.5 * (ny_[1:] + ny_[:-1])
        cz = 0.5 * (nz_[1:] + nz_[:-1])
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        lo, hi = hole
        inh = ((CX > lo[0]) & (CX < hi[0]) & (CY > lo[1]) & (CY < hi[1])
               & (CZ > lo[2]) & (CZ < hi[2]))
        lat &= ~inh
    kept = skin | ext | drm | lat

    def nodes_of(mask):
        n = np.zeros(tuple(s + 1 for s in mask.shape), dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    n[di:di + mask.shape[0], dj:dj + mask.shape[1],
                      dk:dk + mask.shape[2]] |= mask
        return n

    n_lat = int(nodes_of(kept).sum())
    dn = nodes_of(drm)
    I, J, K = np.nonzero(dn)
    drm_nodes = np.stack([nx_[I], ny_[J], nz_[K]], axis=1)
    hexes = {"lattice": int(lat.sum()), "drm": int(drm.sum()),
             "exterior": int(ext.sum()), "skin": int(skin.sum())}
    n_nf = 0
    hexes["nearfield"] = 0
    if nfax:
        (_, qx), (_, qy), (_, qz) = (_axis_nodes(nfax[a]) for a in "xyz")
        QX, QY, QZ = np.meshgrid(qx, qy, qz, indexing="ij")
        nf = ~((QX == "pit") & (QY == "pit") & (QZ == "pit"))
        hexes["nearfield"] = int(nf.sum())
        n_nf = int(nodes_of(nf).sum())
    elif hole is not None:
        pass                                    # pit cut from the lattice
    hexes["interior"] = hexes["lattice"] + hexes["nearfield"]
    hexes["domain"] = hexes["interior"] + hexes["drm"] + hexes["exterior"]
    return hexes, n_lat + n_nf, drm_nodes


def _station_check(drm_nodes, stations, tol) -> StationCheck:
    """Every DRM-layer node is a station and every station a DRM-layer node."""
    d, _ = nearest_station(drm_nodes, stations)
    d2, _ = nearest_station(stations, drm_nodes)
    if len(drm_nodes) != len(stations) or d.max() >= tol or d2.max() >= tol:
        raise RuntimeError(
            f"station box: internal error, the DRM layer ({len(drm_nodes)} "
            f"nodes) does not land on the {len(stations)} stations (max "
            f"distance {max(d.max(), d2.max()):g}, tolerance {tol:g}).")
    return StationCheck(n_stations=len(stations), n_drm_nodes=len(drm_nodes),
                        max_distance=float(max(d.max(), d2.max())), tolerance=tol)


def nearest_station(points, stations, chunk: int = 2048):
    """``(distance, index)`` of the nearest station to each point."""
    P = np.asarray(points, dtype=float)
    S = np.asarray(stations, dtype=float)
    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is in the plot extra
        dist = np.empty(len(P))
        idx = np.empty(len(P), dtype=int)
        for a in range(0, len(P), chunk):
            blk = P[a:a + chunk]
            d2 = ((blk[:, None, :] - S[None, :, :]) ** 2).sum(-1)
            idx[a:a + chunk] = d2.argmin(1)
            dist[a:a + chunk] = np.sqrt(d2[np.arange(len(blk)), idx[a:a + chunk]])
        return dist, idx
    dist, idx = cKDTree(S).query(P)
    return np.asarray(dist), np.asarray(idx)


def h5drm_matches(node_xyz, lattice: SoilLattice, tolerance: float | None = None):
    """The fork's node matching (``H5DRMLoadPattern::node_matching_BruteForce``).

    Every node is tested against its nearest station with ``d < tolerance``
    (strict). Returns ``(matched mask, station index, distance)``.
    """
    if lattice.stations is None or lattice.frame is None:
        raise ValueError("h5drm_matches: the lattice was not read from an .h5drm.")
    tol = lattice.frame.distance_tolerance if tolerance is None else float(tolerance)
    d, i = nearest_station(node_xyz, lattice.stations)
    return d < tol, i, d


__all__ = [
    "TOLERANCE_FRACTION",
    "Exterior",
    "NearField",
    "Pit",
    "SoilLattice",
    "StationCheck",
    "StationFrame",
    "StationSoilBoxResult",
    "build_station_soil_box",
    "h5drm_matches",
    "nearest_station",
]
