"""
DRMBoxFromH5 — structured soil box matched to an ``.h5drm`` station grid (ADR 0066).

Reads a ShakerMaker-style ``.h5drm`` DRM dataset and builds, in the *live
session*, a single transfinite hex box whose nodes land EXACTLY on the dataset
stations, so OpenSees' H5DRM node-matching is trivial.  Tags the soil volume and
the six outer boundary faces (the dataset "b" shell) as physical groups, and
returns the **frame contract** (``crd_scale`` / identity transform / ``x0`` /
centre) so the matching ``ops.pattern.H5DRM(...)`` and the D-3 buffer stay
consistent without the user re-deriving anything.

This is the dataset-keyed sibling of the parametric :class:`~apeGmsh.parts.drm_box.DRMBox`
(an SSI inner/transition/outer absorbing layout — NOT keyed to a dataset).  It
builds directly in the session (no Part/STEP round-trip), like
:func:`~apeGmsh.parts.plane_wave_box.build_plane_wave_box`.  Reference (validated
98/98 node match): ``internal_docs/drm_study/build_drm_model.py``.

Frame contract (the C++ H5DRM transform, ``H5DRMLoadPattern::do_intitialization``):

    ``xyz_model = T · ((xyz_station − drmbox_x0) · crd_scale) + x0``

with ``drmbox_x0`` the box centre read from the file.  Building the model nodes
as ``(xyz_station − drmbox_x0) · crd_scale`` (centred at the lateral origin,
z-down, in metres) makes ``T = I`` / ``x0 = 0`` reproduce them exactly — so the
returned ``transform`` is identity and ``x0`` is zero.

The b/e split is geometric: for a regular DRM grid the dataset boundary stations
(``internal == 0``) are exactly the box's outer-face shell, so the six boundary
face PGs ARE the "b" shell and the soil volume interior is the "e" region.  The
builder cross-checks the geometric shell node count against the dataset
``internal`` flag and warns on a mismatch.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gmsh
import numpy as np

if TYPE_CHECKING:
    from ..core._session import _SessionBase  # pragma: no cover


class WarnDRMGridIrregular(UserWarning):
    """The ``.h5drm`` station set is not a clean regular boundary shell.

    The geometric box-face PGs (the b/e split) assume the dataset's boundary
    stations (``internal == 0``) form the outer shell of a complete regular
    grid.  Fail-soft — H5DRM still matches nodes from the file, but the returned
    boundary PGs may not coincide with the dataset boundary.
    """


# Face keys for the six outer faces (z-down: ``top`` = free surface at z=0,
# ``bottom`` = deepest face at max z).  ``exterior`` (sides+bottom) excludes the
# free surface — the targets a D-3 buffer extends from / a boundary applies to.
_EXTERIOR_KEYS = ("xmin", "xmax", "ymin", "ymax", "bottom")
_ALL_FACE_KEYS = ("xmin", "xmax", "ymin", "ymax", "top", "bottom")

_DEFAULT_NAMES: dict[str, str] = {
    "soil": "drm_soil",
    "boundary_all": "drm_boundary",
    "free_surface": "drm_free_surface",
    "xmin": "drm_face_xmin",
    "xmax": "drm_face_xmax",
    "ymin": "drm_face_ymin",
    "ymax": "drm_face_ymax",
    "top": "drm_face_top",
    "bottom": "drm_face_bottom",
}


@dataclass(frozen=True)
class DRMBoxFromH5Result:
    """Summary of an :func:`build_drm_box_from_h5drm` placement.

    Carries the PGs (so downstream code never touches tags), the **frame
    contract** that the matching ``ops.pattern.H5DRM(...)`` consumes, and the
    grid descriptor.
    """

    soil_pg: str
    """PG name of the interior soil volume (the DRM "e" region)."""
    boundary_pgs: dict[str, str] = field(default_factory=dict)
    """``face-key -> surface PG name`` for the six outer faces (the "b" shell):
    ``xmin``/``xmax``/``ymin``/``ymax``/``top``/``bottom``."""
    boundary_all_pg: str = ""
    """Roll-up surface PG over all six outer faces (the whole "b" shell)."""
    free_surface_pg: str = ""
    """Surface PG of the top face (free surface, z=0)."""
    exterior_pgs: tuple[str, ...] = ()
    """Surface PG names for sides + bottom (NOT the free surface) — the faces a
    D-3 buffer extends from / a far boundary applies to."""

    # ── frame contract (feeds ops.pattern.H5DRM) ──
    crd_scale: float = 1000.0
    transform: tuple[tuple[float, ...], ...] | None = None
    """3×3 row-major rotation; ``None`` == identity (the built-in centred frame)."""
    x0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """``drmbox_x0`` — the box centre in STATION units (the H5DRM transform origin)."""

    # ── grid descriptor ──
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Model-space box origin (m): ``(min_station − center) · crd_scale``."""
    spacing: float = 0.0
    """Uniform grid spacing in model units (m)."""
    counts: tuple[int, int, int] = (0, 0, 0)
    """Node counts ``(nx, ny, nz)`` along each axis."""


def _axis(vals: "np.ndarray") -> tuple[float, float, int, float, float]:
    """Return ``(min, max, count, spacing, ok_uniform)`` for one coordinate axis.

    ``ok_uniform`` is the max relative deviation of the per-step spacing from its
    mean (0.0 == perfectly uniform).
    """
    u = np.unique(np.round(vals, 6))
    if len(u) < 2:
        return float(u.min()), float(u.max()), int(len(u)), 0.0, 0.0
    d = np.diff(u)
    h = float(d.mean())
    dev = float(np.max(np.abs(d - h)) / h) if h > 0 else float("inf")
    return float(u.min()), float(u.max()), int(len(u)), h, dev


def _read_h5drm_grid(
    path: str,
) -> tuple["np.ndarray", "np.ndarray | None", "np.ndarray | None"]:
    """Read station coords, the per-station ``internal`` flag, and ``drmbox_x0``.

    Optional children are probed with ``name in group`` (H5Lexists), never
    ``Group.get`` (see the ``h5py-optional-child`` hazard).
    """
    import h5py

    with h5py.File(path, "r") as f:
        if "DRM_Data" not in f or "xyz" not in f["DRM_Data"]:
            raise ValueError(
                f"{path!r} is not an .h5drm dataset: missing DRM_Data/xyz."
            )
        xyz = np.asarray(f["DRM_Data/xyz"][:], dtype=float)
        internal = (
            np.asarray(f["DRM_Data/internal"][:]).astype(bool)
            if "internal" in f["DRM_Data"]
            else None
        )
        x0c = None
        if "DRM_Metadata" in f and "drmbox_x0" in f["DRM_Metadata"]:
            x0c = np.asarray(f["DRM_Metadata/drmbox_x0"][:], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"{path!r}: DRM_Data/xyz must be (N, 3), got {xyz.shape}."
        )
    return xyz, internal, x0c


def build_drm_box_from_h5drm(
    session: "_SessionBase",
    *,
    h5drm: str,
    crd_scale: float = 1000.0,
    name: str | None = None,
    names: dict[str, str] | None = None,
    apply_transfinite: bool = True,
) -> DRMBoxFromH5Result:
    """Build a structured soil box matched to an ``.h5drm`` station grid.

    See the module docstring and ``g.parts.add_DRM_box_from_h5drm`` for the
    user-facing contract.  Geometry + PGs only — assign the soil material and
    the ``stdBrick`` elements via the apeSees bridge (``ops.nDMaterial`` +
    ``ops.element.stdBrick(pg=result.soil_pg)``), then drive the wavefield with
    ``ops.pattern.H5DRM(h5drm=...)`` whose defaults already match this box.
    """
    xyz, internal, x0c = _read_h5drm_grid(h5drm)

    x0_, x1_, nx, hx, dx_dev = _axis(xyz[:, 0])
    y0_, y1_, ny, hy, dy_dev = _axis(xyz[:, 1])
    z0_, z1_, nz, hz, dz_dev = _axis(xyz[:, 2])

    # ── grid must be a complete, uniform, isotropic regular grid ──
    n_expected = nx * ny * nz
    if xyz.shape[0] != n_expected:
        raise ValueError(
            f"add_DRM_box_from_h5drm: the .h5drm stations are not a complete "
            f"regular grid — got {xyz.shape[0]} stations but the unique-coordinate "
            f"grid is {nx}×{ny}×{nz} = {n_expected}. Only structured grids are "
            f"supported."
        )
    spacings = [h for h, n in ((hx, nx), (hy, ny), (hz, nz)) if n > 1]
    if not spacings:
        raise ValueError("add_DRM_box_from_h5drm: degenerate grid (single node).")
    h = float(np.mean(spacings))
    worst_dev = max(dx_dev, dy_dev, dz_dev)
    if worst_dev > 1e-3 or any(abs(s - h) / h > 1e-3 for s in spacings):
        raise ValueError(
            f"add_DRM_box_from_h5drm: the .h5drm grid spacing is not uniform "
            f"across axes (hx={hx:g}, hy={hy:g}, hz={hz:g}; max per-axis "
            f"deviation {worst_dev:.2e}). Only a single isotropic spacing is "
            f"supported in this slice."
        )

    # ── centre = the file's drmbox_x0 (the H5DRM transform origin) ──
    if x0c is not None and x0c.shape == (3,):
        cx, cy, cz = (float(v) for v in x0c)
    else:
        cx, cy, cz = (x0_ + x1_) / 2.0, (y0_ + y1_) / 2.0, z0_
        warnings.warn(
            "add_DRM_box_from_h5drm: DRM_Metadata/drmbox_x0 missing — centring on "
            "the geometric grid centre; verify it matches the H5DRM transform "
            "origin or pass a matching transform/x0 to ops.pattern.H5DRM.",
            WarnDRMGridIrregular,
            stacklevel=3,
        )

    # model-space box (metres, centred, z-down): node = (station - centre)*scale
    ox, oy, oz = (x0_ - cx) * crd_scale, (y0_ - cy) * crd_scale, (z0_ - cz) * crd_scale
    dx, dy, dz = (x1_ - x0_) * crd_scale, (y1_ - y0_) * crd_scale, (z1_ - z0_) * crd_scale
    h_model = h * crd_scale

    # ── b/e split sanity: boundary stations vs the geometric outer shell ──
    if internal is not None:
        n_shell = nx * ny * nz - max(nx - 2, 0) * max(ny - 2, 0) * max(nz - 2, 0)
        if int((~internal).sum()) != n_shell:
            warnings.warn(
                f"add_DRM_box_from_h5drm: the dataset has {int((~internal).sum())} "
                f"boundary stations but a complete {nx}×{ny}×{nz} grid shell has "
                f"{n_shell} — the geometric boundary face PGs may not coincide "
                f"with the dataset 'b' nodes.",
                WarnDRMGridIrregular,
                stacklevel=3,
            )

    nm = dict(_DEFAULT_NAMES)
    if names:
        nm.update(names)
    if name:
        # ``name`` is a prefix applied to every default PG name.
        nm = {k: f"{name}_{v}" for k, v in nm.items()}

    # ── build the box + tag the soil volume ──
    geom = session.model.geometry
    soil_label = nm["soil"]
    vol = int(geom.add_box(ox, oy, oz, dx, dy, dz, label=soil_label))
    physical = session.physical
    physical.add(3, [vol], name=nm["soil"])

    # ── classify + tag the six outer faces (getBoundary scopes to THIS box) ──
    queries = session.model.queries
    tol = max(h_model, 1.0) * 1e-3
    faces = [int(t) for _d, t in gmsh.model.getBoundary(
        [(3, vol)], combined=False, oriented=False)]
    by_key: dict[str, int] = {}
    for ft in faces:
        fx, fy, fz = queries.center_of_mass(ft, dim=2)
        if abs(fx - ox) < tol:
            by_key["xmin"] = ft
        elif abs(fx - (ox + dx)) < tol:
            by_key["xmax"] = ft
        elif abs(fy - oy) < tol:
            by_key["ymin"] = ft
        elif abs(fy - (oy + dy)) < tol:
            by_key["ymax"] = ft
        elif abs(fz - oz) < tol:
            by_key["top"] = ft            # z-down: min z = free surface
        elif abs(fz - (oz + dz)) < tol:
            by_key["bottom"] = ft         # z-down: max z = deepest face

    boundary_pgs: dict[str, str] = {}
    for key in _ALL_FACE_KEYS:
        ftag = by_key.get(key)
        if ftag is None:
            continue
        physical.add(2, [ftag], name=nm[key])
        boundary_pgs[key] = nm[key]

    all_face_tags = [by_key[k] for k in _ALL_FACE_KEYS if k in by_key]
    if all_face_tags:
        physical.add(2, all_face_tags, name=nm["boundary_all"])

    exterior_pgs = tuple(nm[k] for k in _EXTERIOR_KEYS if k in by_key)

    # ── structured hex constraints (size-based: round(edge/h)+1 == grid count) ──
    if apply_transfinite:
        session.mesh.structured.set_transfinite_box(
            vol, size=h_model, recombine=True,
        )

    return DRMBoxFromH5Result(
        soil_pg=nm["soil"],
        boundary_pgs=boundary_pgs,
        boundary_all_pg=nm["boundary_all"] if all_face_tags else "",
        free_surface_pg=boundary_pgs.get("top", ""),
        exterior_pgs=exterior_pgs,
        crd_scale=float(crd_scale),
        transform=None,
        x0=(0.0, 0.0, 0.0),
        center=(cx, cy, cz),
        origin=(ox, oy, oz),
        spacing=h_model,
        counts=(nx, ny, nz),
    )
