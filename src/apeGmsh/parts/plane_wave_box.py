"""
PlaneWaveBox — structured soil box wrapped by an ASDAbsorbingBoundary skin.

Builds, in the *live session* (not a Part/STEP round-trip), an axis-aligned
structured soil box plus a one-element-thick absorbing **offset shell** on its
five truncation faces (the local +Z top is the free surface and is never
shelled).  Soil + shell are a single plain rectangular block: slicing only at the
region breakpoints yields **18 sub-volumes** — one soil region plus up to 17
skin regions (5 face panels, 4 vertical edges, 4 bottom edges, 4 bottom
corners).  Each skin region is tagged with its OpenSees ``btype`` (the set of
truncation faces it lies outside of, OR-combined) so the bridge fans out one
``ASDAbsorbingBoundary3D`` per element with the shared btype.

This is the shared session-geometry core behind
``g.parts.add_plane_wave_box`` (ADR 0054 / plan_absorbing_skin_ab1.md, slice
AB-1a).  It deliberately does NOT depend on :class:`~apeGmsh.parts.drm_box.DRMBox`
(that serves the Domain Reduction Method) and does NOT use the Part/STEP vehicle
(it builds directly in the session, avoiding the ``setCurrent`` footgun).

btype → axis mapping (proven against the element source and a real STKO export):
``L`` = min-X, ``R`` = max-X, ``F`` = min-Y, ``K`` = max-Y, ``B`` = min-Z
(bottom).  Letters are canonically ordered ``BLRFK``.  Opposite-face combos
(``LR``/``FK``) are illegal and cannot arise from this grid by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gmsh

from ._axis1d import Axis1D

if TYPE_CHECKING:
    from ..core._session import _SessionBase  # pragma: no cover

# Canonical btype letter order — also the OpenSees-accepted set.
_BTYPE_ORDER = "BLRFK"


@dataclass(frozen=True)
class AbsorbingSkinResult:
    """Summary of a :func:`build_plane_wave_box` placement.

    Returned to the user so downstream code can refer to the generated
    physical groups without touching tags.
    """

    soil_pg: str
    """PG name of the intact interior soil volume."""
    skin_pgs: dict[str, str] = field(default_factory=dict)
    """``btype -> PG name`` for every skin region present (e.g.
    ``{"L": "absorbing_L", "LF": "absorbing_LF", "BLF": "absorbing_BLF"}``).
    The bridge emits one ``ASDAbsorbingBoundary3D`` declaration per entry."""
    skin_all_pg: str = ""
    """Roll-up PG over every skin region — the set the staged
    ``s.activate_absorbing()`` flip and the Rayleigh region target."""
    bottom_pgs: tuple[str, ...] = ()
    """Skin PG names whose btype contains ``B`` — the base-input targets."""
    free_surface_pg: str = ""
    """PG (dim 2) of the soil top face at the local free surface (z=0)."""
    axes: dict[str, Axis1D] = field(default_factory=dict)
    """``x``/``y``/``z`` :class:`Axis1D` descriptors for downstream sizing."""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_z: float = 0.0
    """Applied rotation about +Z, in radians (always 0.0 in AB-1a)."""


def _btype_for(rx: str, ry: str, rz: str) -> str:
    """Canonical btype for a sub-volume from its per-axis region labels.

    Returns ``""`` for the interior soil cell.
    """
    faces = []
    if rx in ("L", "R"):
        faces.append(rx)
    if ry in ("F", "K"):
        faces.append(ry)
    if rz == "B":
        faces.append("B")
    return "".join(sorted(faces, key=_BTYPE_ORDER.index))


def _as_size_count(value, who: str) -> tuple[float, int]:
    """Validate and unpack a ``(size, n_elements)`` axis tuple."""
    try:
        size, n = value
    except (TypeError, ValueError):
        raise ValueError(
            f"add_plane_wave_box: {who} must be a (size, n_elements) "
            f"tuple, got {value!r}."
        )
    size_f, n_i = float(size), int(n)
    if size_f <= 0.0:
        raise ValueError(
            f"add_plane_wave_box: {who} size must be > 0, got {size_f}."
        )
    if n_i < 1:
        raise ValueError(
            f"add_plane_wave_box: {who} n_elements must be >= 1, got {n_i}."
        )
    return size_f, n_i


def build_plane_wave_box(
    session: "_SessionBase",
    *,
    x: tuple[float, int],
    y: tuple[float, int],
    z,
    skin_thickness=None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_z_deg: float = 0.0,
    name: str | None = None,
    names: dict[str, str] | None = None,
    apply_transfinite: bool = True,
) -> AbsorbingSkinResult:
    """Build a plane-wave soil box + absorbing skin in the live session.

    See module docstring and ``g.parts.add_plane_wave_box`` for the user-facing
    contract.  AB-1a scope: axis-aligned (``rotation_z_deg == 0``), single soil
    segment per axis (layered Z is AB-1c).
    """
    # ── Fail-loud guards (AB-1a scope) ──────────────────────────────
    if isinstance(z, list):
        raise NotImplementedError(
            "add_plane_wave_box: layered Z (stratigraphy) is AB-1c; pass a "
            "single (depth, n_elements) tuple for z."
        )
    if abs(float(rotation_z_deg)) > 1e-15:
        raise NotImplementedError(
            "add_plane_wave_box: rotation is AB-1c; rotation_z_deg must be 0.0."
        )

    Lx, nx = _as_size_count(x, "x")
    Ly, ny = _as_size_count(y, "y")
    Lz, nz = _as_size_count(z, "z")

    # Skin thickness per axis — default = adjacent soil element size.
    if skin_thickness is None:
        tx, ty, tz = Lx / nx, Ly / ny, Lz / nz
    elif isinstance(skin_thickness, (int, float)):
        tx = ty = tz = float(skin_thickness)
    else:
        try:
            tx, ty, tz = (float(v) for v in skin_thickness)
        except (TypeError, ValueError):
            raise ValueError(
                "add_plane_wave_box: skin_thickness must be None, a scalar, "
                f"or a (tx, ty, tz) tuple, got {skin_thickness!r}."
            )
    for t, ax in ((tx, "x"), (ty, "y"), (tz, "z")):
        if t <= 0.0:
            raise ValueError(
                f"add_plane_wave_box: skin_thickness on {ax} must be > 0, got {t}."
            )

    # ── Axis descriptors (local frame; soil centred laterally, top z=0) ──
    axis_x = Axis1D("x", (
        ("L", -Lx / 2 - tx, -Lx / 2, 1),
        ("soil", -Lx / 2, Lx / 2, nx),
        ("R", Lx / 2, Lx / 2 + tx, 1),
    ))
    axis_y = Axis1D("y", (
        ("F", -Ly / 2 - ty, -Ly / 2, 1),
        ("soil", -Ly / 2, Ly / 2, ny),
        ("K", Ly / 2, Ly / 2 + ty, 1),
    ))
    axis_z = Axis1D("z", (
        ("B", -Lz - tz, -Lz, 1),
        ("soil", -Lz, 0.0, nz),
    ))

    cx, cy, cz = (float(v) for v in center)

    def to_local(world_xyz):
        wx, wy, wz = world_xyz
        return wx - cx, wy - cy, wz - cz

    # Build + slice in the LOCAL frame (centred near the origin, where the
    # slice cutting-plane reliably covers the box — it is sized around the
    # origin, so a box translated far away would slice to nothing), then
    # translate to ``center``.  Mirrors the DRMBox build-local-then-place model.
    geom = session.model.geometry
    before_vols = {int(t) for _d, t in gmsh.model.getEntities(3)}

    geom.add_box(
        axis_x.lo, axis_y.lo, axis_z.lo,
        axis_x.hi - axis_x.lo, axis_y.hi - axis_y.lo, axis_z.hi - axis_z.lo,
    )

    def _box_vols() -> list[int]:
        return sorted({int(t) for _d, t in gmsh.model.getEntities(3)} - before_vols)

    for off in axis_x.slice_offsets():
        geom.slice(target=_box_vols(), axis="x", offset=float(off))
    for off in axis_y.slice_offsets():
        geom.slice(target=_box_vols(), axis="y", offset=float(off))
    for off in axis_z.slice_offsets():
        geom.slice(target=_box_vols(), axis="z", offset=float(off))

    # Place: translate the whole block to the requested center.  Done before
    # PG-tagging / transfinite so no synchronize follows group creation.
    if cx or cy or cz:
        gmsh.model.occ.translate([(3, v) for v in _box_vols()], cx, cy, cz)
        gmsh.model.occ.synchronize()

    # ── Classify the 18 sub-volumes by btype ────────────────────────
    queries = session.model.queries
    soil_vols: list[int] = []
    by_btype: dict[str, list[int]] = {}
    # vtag -> (nx_count, ny_count, nz_count) for the transfinite cascade
    per_vol_counts: list[tuple[int, int, int, int]] = []

    for vtag in _box_vols():
        lx, ly, lz = to_local(queries.center_of_mass(int(vtag), dim=3))
        rx, ry, rz = axis_x.region_of(lx), axis_y.region_of(ly), axis_z.region_of(lz)
        btype = _btype_for(rx, ry, rz)
        if btype:
            by_btype.setdefault(btype, []).append(int(vtag))
        else:
            soil_vols.append(int(vtag))
        per_vol_counts.append((
            int(vtag),
            axis_x.count_for(lx), axis_y.count_for(ly), axis_z.count_for(lz),
        ))

    # ── Physical groups ─────────────────────────────────────────────
    prefix = f"{name}_" if name else ""

    def pg_name(base: str) -> str:
        if names and base in names:
            return str(names[base])
        return f"{prefix}{base}"

    physical = session.physical
    soil_pg = pg_name("soil")
    if soil_vols:
        physical.add(3, soil_vols, name=soil_pg)

    skin_pgs: dict[str, str] = {}
    all_skin_vols: list[int] = []
    for btype in sorted(by_btype, key=lambda b: (len(b), b)):
        vols = by_btype[btype]
        nm = pg_name(f"absorbing_{btype}")
        physical.add(3, vols, name=nm)
        skin_pgs[btype] = nm
        all_skin_vols.extend(vols)

    skin_all_pg = pg_name("absorbing")
    if all_skin_vols:
        physical.add(3, all_skin_vols, name=skin_all_pg)

    bottom_pgs = tuple(
        skin_pgs[bt] for bt in sorted(skin_pgs, key=lambda b: (len(b), b))
        if "B" in bt
    )

    # ── Free surface: soil top faces at local z = 0 ─────────────────
    extent = max(axis_x.hi - axis_x.lo, axis_y.hi - axis_y.lo, axis_z.hi - axis_z.lo)
    z_tol = 1e-6 * extent
    free_faces: list[int] = []
    for _d, ftag in gmsh.model.getEntities(2):
        lx, ly, lz = to_local(queries.center_of_mass(int(ftag), dim=2))
        if abs(lz) > z_tol:
            continue
        if axis_x.region_of(lx) == "soil" and axis_y.region_of(ly) == "soil":
            free_faces.append(int(ftag))
    free_surface_pg = pg_name("free_surface")
    if free_faces:
        physical.add(2, free_faces, name=free_surface_pg)

    # ── Transfinite cascade per sub-volume ──────────────────────────
    if apply_transfinite:
        structured = session.mesh.structured
        for vtag, cnx, cny, cnz in per_vol_counts:
            structured.set_transfinite(
                (3, vtag),
                n=(cnx + 1, cny + 1, cnz + 1),
                recombine=True,
            )

    return AbsorbingSkinResult(
        soil_pg=soil_pg,
        skin_pgs=skin_pgs,
        skin_all_pg=skin_all_pg,
        bottom_pgs=bottom_pgs,
        free_surface_pg=free_surface_pg if free_faces else "",
        axes={"x": axis_x, "y": axis_y, "z": axis_z},
        center=(cx, cy, cz),
        rotation_z=0.0,
    )
