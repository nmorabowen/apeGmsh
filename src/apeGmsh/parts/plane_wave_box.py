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


def _as_xyz(value, who: str) -> tuple[float, float, float]:
    """Validate a scalar-or-``(x, y, z)`` positive-float spec → ``(vx, vy, vz)``.

    ``who`` is the caller-context label used in error messages (e.g.
    ``"add_absorbing_shell: element_size"``).
    """
    if isinstance(value, (int, float)):
        vx = vy = vz = float(value)
    else:
        try:
            vx, vy, vz = (float(t) for t in value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{who} must be a scalar or an (x, y, z) tuple, got {value!r}."
            )
    for v, ax in ((vx, "x"), (vy, "y"), (vz, "z")):
        if v <= 0.0:
            raise ValueError(f"{who} on {ax} must be > 0, got {v}.")
    return vx, vy, vz


_ALL_FACES = ("L", "R", "F", "K", "B")


def _resolve_active_faces(faces) -> set[str]:
    """Validate the optional ``faces=`` subset → a set of active face letters.

    ``None`` → all five truncation faces.  The local +Z top is the free surface
    and is never a valid entry (no ``"T"``).
    """
    if faces is None:
        return set(_ALL_FACES)
    active: set[str] = set()
    for f in faces:
        if f not in _ALL_FACES:
            raise ValueError(
                f"add_absorbing_shell: faces entries must be one of "
                f"{_ALL_FACES} (the +Z top is the free surface and is never "
                f"shelled), got {f!r}."
            )
        active.add(f)
    if not active:
        raise ValueError(
            "add_absorbing_shell: faces must name at least one face."
        )
    return active


def _axes_from_extent(
    extent: tuple[float, float, float, float, float, float],
    sizes: tuple[float, float, float],
    thick: tuple[float, float, float],
    active: set[str],
) -> tuple[Axis1D, Axis1D, Axis1D]:
    """Three world-frame :class:`Axis1D` from a box AABB + element size + skin.

    Soil-segment counts are ``max(1, round(length / size))``; each outer skin
    segment is one element thick.  Outer segments are emitted only for active
    faces; the +Z top is never shelled.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = extent
    sx, sy, sz = sizes
    tx, ty, tz = thick
    nx = max(1, round((xmax - xmin) / sx))
    ny = max(1, round((ymax - ymin) / sy))
    nz = max(1, round((zmax - zmin) / sz))

    x_segs: list[tuple[str, float, float, int]] = []
    if "L" in active:
        x_segs.append(("L", xmin - tx, xmin, 1))
    x_segs.append(("soil", xmin, xmax, nx))
    if "R" in active:
        x_segs.append(("R", xmax, xmax + tx, 1))

    y_segs: list[tuple[str, float, float, int]] = []
    if "F" in active:
        y_segs.append(("F", ymin - ty, ymin, 1))
    y_segs.append(("soil", ymin, ymax, ny))
    if "K" in active:
        y_segs.append(("K", ymax, ymax + ty, 1))

    z_segs: list[tuple[str, float, float, int]] = []
    if "B" in active:
        z_segs.append(("B", zmin - tz, zmin, 1))
    z_segs.append(("soil", zmin, zmax, nz))

    return (
        Axis1D("x", tuple(x_segs)),
        Axis1D("y", tuple(y_segs)),
        Axis1D("z", tuple(z_segs)),
    )


def _tag_and_structure(
    session: "_SessionBase",
    vols: list[int],
    *,
    axis_x: Axis1D,
    axis_y: Axis1D,
    axis_z: Axis1D,
    to_local,
    name: str | None,
    names: dict[str, str] | None,
    apply_transfinite: bool,
    center: tuple[float, float, float],
    soil_pg_name: str | None = None,
) -> AbsorbingSkinResult:
    """Classify volumes by btype, create PGs, apply the transfinite cascade.

    Shared tail of :func:`build_plane_wave_box` (build-then-slice) and
    :func:`build_absorbing_shell` (weld-then-fragment).  ``vols`` is the full
    set of sub-volumes (one soil interior + the skin cells); each is classified
    by its centroid in the local frame.  When ``soil_pg_name`` is given the soil
    interior is reported under that existing name and **no** soil PG is created
    (the caller's box already carries it); otherwise a ``<prefix>soil`` PG is
    made over the interior cell.  The free surface is the soil top face at local
    ``z = axis_z.hi`` (no skin sits above it).
    """
    queries = session.model.queries
    soil_vols: list[int] = []
    by_btype: dict[str, list[int]] = {}
    per_vol_counts: list[tuple[int, int, int, int]] = []

    for vtag in vols:
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

    prefix = f"{name}_" if name else ""

    def pg_name(base: str) -> str:
        if names and base in names:
            return str(names[base])
        return f"{prefix}{base}"

    physical = session.physical
    if soil_pg_name is not None:
        soil_pg = soil_pg_name              # caller's box already carries this PG
    else:
        soil_pg = pg_name("soil")
        if soil_vols:
            physical.add(3, soil_vols, name=soil_pg)

    skin_pgs: dict[str, str] = {}
    all_skin_vols: list[int] = []
    for btype in sorted(by_btype, key=lambda b: (len(b), b)):
        bvols = by_btype[btype]
        nm = pg_name(f"absorbing_{btype}")
        physical.add(3, bvols, name=nm)
        skin_pgs[btype] = nm
        all_skin_vols.extend(bvols)

    skin_all_pg = pg_name("absorbing")
    if all_skin_vols:
        physical.add(3, all_skin_vols, name=skin_all_pg)

    bottom_pgs = tuple(
        skin_pgs[bt] for bt in sorted(skin_pgs, key=lambda b: (len(b), b))
        if "B" in bt
    )

    # ── Free surface: soil top faces at local z = axis_z.hi ─────────
    extent = max(axis_x.hi - axis_x.lo, axis_y.hi - axis_y.lo, axis_z.hi - axis_z.lo)
    z_tol = 1e-6 * extent
    top = axis_z.hi
    free_faces: list[int] = []
    for _d, ftag in gmsh.model.getEntities(2):
        lx, ly, lz = to_local(queries.center_of_mass(int(ftag), dim=2))
        if abs(lz - top) > z_tol:
            continue
        try:  # faces from unrelated geometry fall outside this box's axes
            in_soil = (axis_x.region_of(lx) == "soil"
                       and axis_y.region_of(ly) == "soil")
        except ValueError:
            continue
        if in_soil:
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
        center=center,
        rotation_z=0.0,
    )


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

    # ── Classify sub-volumes, tag PGs, apply the transfinite cascade ─
    return _tag_and_structure(
        session,
        _box_vols(),
        axis_x=axis_x,
        axis_y=axis_y,
        axis_z=axis_z,
        to_local=to_local,
        name=name,
        names=names,
        apply_transfinite=apply_transfinite,
        center=(cx, cy, cz),
    )


def build_absorbing_shell(
    session: "_SessionBase",
    *,
    box,
    element_size,
    skin_thickness=None,
    faces=None,
    name: str | None = None,
    names: dict[str, str] | None = None,
    apply_transfinite: bool = True,
) -> AbsorbingSkinResult:
    """Weld a one-element absorbing skin onto a user's existing soil box.

    See ``g.parts.add_absorbing_shell`` for the user-facing contract (ADR 0054,
    AB-1b).  AB-1b scope: ``box`` resolves to a single axis-aligned *rectangular*
    volume; the skin discretization is **size-based** and (re)applied to box +
    skin after the weld — gmsh cannot report transfinite counts back and the
    ``fragment`` renumbers entities, so the box's prior mesh state is irrelevant
    (this call makes it structured).  ``rotation`` / layered-Z / graded skins are
    AB-1c.
    """
    from apeGmsh.core._helpers import resolve_to_dimtags

    # ── Resolve + validate the box (exactly one rectangular volume) ──
    dts = resolve_to_dimtags(box, default_dim=3, session=session)
    box_vols = [int(t) for d, t in dts if int(d) == 3]
    if len(box_vols) != 1:
        raise ValueError(
            f"add_absorbing_shell: box must resolve to exactly one dim-3 "
            f"volume, got {len(box_vols)} ({box!r}).  AB-1b wraps a single "
            "axis-aligned rectangular soil box."
        )
    box_vol = box_vols[0]

    queries = session.model.queries
    xmin, ymin, zmin, xmax, ymax, zmax = queries.bounding_box(box_vol, dim=3)
    aabb = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
    gmsh.model.occ.synchronize()
    mass = gmsh.model.occ.getMass(3, int(box_vol))
    if aabb <= 0.0 or abs(mass - aabb) > 1e-6 * aabb:
        raise ValueError(
            "add_absorbing_shell: box is not an axis-aligned rectangular "
            f"block (volume {mass:.6g} != bounding-box product {aabb:.6g}).  "
            "AB-1b requires a rectangular box; rotated / curved geometry is "
            "AB-1c."
        )

    sizes = _as_xyz(element_size, "add_absorbing_shell: element_size")
    thick = (
        sizes if skin_thickness is None
        else _as_xyz(skin_thickness, "add_absorbing_shell: skin_thickness")
    )
    active = _resolve_active_faces(faces)

    axis_x, axis_y, axis_z = _axes_from_extent(
        (xmin, ymin, zmin, xmax, ymax, zmax), sizes, thick, active,
    )

    # ── Build the skin slabs (every grid cell except the soil interior) ─
    # The slabs MUST be synchronised before the weld: fragmenting a
    # synced box against unsynced slabs leaves coincident-but-separate
    # faces (duplicate interface nodes ⇒ a disconnected, singular model).
    geom = session.model.geometry
    slab_vols: list[int] = []
    for rx, xlo, xhi, _cx in axis_x.segments:
        for ry, ylo, yhi, _cy in axis_y.segments:
            for rz, zlo, zhi, _cz in axis_z.segments:
                if rx == "soil" and ry == "soil" and rz == "soil":
                    continue  # the user's box — never rebuilt
                slab_vols.append(int(geom.add_box(
                    xlo, ylo, zlo, xhi - xlo, yhi - ylo, zhi - zlo, sync=True,
                )))

    # ── Weld conformally: self-fragment box + slabs (PGs auto-remap) ─
    session.model.boolean.fragment([box_vol, *slab_vols], [], dim=3)

    # ── Collect the welded box + skin volumes (centroid in the outer
    #    block AABB), tolerant of any other geometry in the session ───
    extent_span = max(
        axis_x.hi - axis_x.lo, axis_y.hi - axis_y.lo, axis_z.hi - axis_z.lo,
    )
    tol = 1e-6 * extent_span
    vols: list[int] = []
    for _d, vt in gmsh.model.getEntities(3):
        ccx, ccy, ccz = queries.center_of_mass(int(vt), dim=3)
        if (axis_x.lo - tol <= ccx <= axis_x.hi + tol
                and axis_y.lo - tol <= ccy <= axis_y.hi + tol
                and axis_z.lo - tol <= ccz <= axis_z.hi + tol):
            vols.append(int(vt))

    soil_pg_name = box if isinstance(box, str) else None
    return _tag_and_structure(
        session,
        vols,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_z=axis_z,
        to_local=lambda xyz: xyz,   # world frame (AB-1b is axis-aligned)
        name=name,
        names=names,
        apply_transfinite=apply_transfinite,
        center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)),
        soil_pg_name=soil_pg_name,
    )
