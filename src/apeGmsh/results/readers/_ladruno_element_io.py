"""``ON_ELEMENTS`` decoding for :class:`LadrunoReader` (recorder plan L2b-2).

Where MPCO needs a per-class response catalog to decode element results,
the ``.ladruno`` writer is **self-describing**: every
``RESULTS/ON_ELEMENTS/<token>/<class>[…]`` bucket carries a structured
``COLUMN_MAP`` (one row per output block) plus a ``COMP_NAMES`` attribute
(newline-separated, one CSV line per block). The reader reads component
names **from the file** — this is the seam that sidesteps the per-class
GP-order traps MPCO fights.

``COLUMN_MAP`` row vocabulary (verified, fork build ``605affeb``):

* ``LEVELS`` — descriptor depth: ``0`` ElementOutput · ``1`` GaussPoint ·
  ``2`` SectionOutput · ``3`` FiberOutput · ``4`` Nd/UniaxialMaterial.
* ``GAUSS_ID`` — 0-based GP index for the block, or ``-1`` (element-level).
* ``NUM_COMP`` — components in the block (== len of its ``COMP_NAMES`` line).

``DATA`` is ``[T, nE, total_cols]`` laid out **block-major**: block *i*
occupies columns ``[Σ width_j (j<i) … +width_i)``. Tokens are mapped to
apeGmsh's neutral vocabulary here:

* continuum ``sigmaIJ`` / ``etaIJ`` → ``stress_ij`` / ``strain_ij`` (Gauss);
* beam ``N``/``Vy``/``Vz``/``T``/``My``/``Mz`` (optional ``_<station>``)
  → ``axial_force`` / ``shear_y`` / … (line stations);
* nodal ``P<dof>_<node>`` → ``nodal_resisting_force_{x,y,z}`` (elements).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Token → canonical maps
# =====================================================================

_AXIS = {"1": "x", "2": "y", "3": "z"}

# Beam end / basic forces (localForce, basicForce) — element-level
# (``LEVELS==0``) buckets that carry one block of named columns.
_BEAM_BASE_TO_CANONICAL: dict[str, str] = {
    "N": "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T": "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

# Section stress-resultants (``section.force``, recorder token
# ``LEVELS==2``). One block per integration station; columns named with the
# OpenSees section response codes ``P``/``Vy``/``Vz``/``T``/``My``/``Mz``
# (``P`` is the section axial force — distinct from the element-level ``N``).
_SECTION_FORCE_TO_CANONICAL: dict[str, str] = {
    "P": "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T": "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

# Section generalized strains (``section.deformation``) — conjugate
# work-pair to the resultants, same per-station layout. Tokens are the
# OpenSees deformation codes (``eps``/``kappaZ``/…). Only the 2-D pair
# (``eps``/``kappaZ``) is verified against a fixture; the 3-D tokens follow
# the standard section response order.
_SECTION_DEFORM_TO_CANONICAL: dict[str, str] = {
    "eps": "axial_strain",
    "gammaY": "shear_strain_y",
    "gammaZ": "shear_strain_z",
    "theta": "torsional_strain",
    "kappaY": "curvature_y",
    "kappaZ": "curvature_z",
}

# Fiber-level buckets (``section.fiber.stress`` / ``section.fiber.strain``,
# ``LEVELS==4`` with ``MULTIPLICITY``==num-fibers). Keyed by token, not by
# column name (the column is the scalar uniaxial ``sigma11``/``eps11``).
#
# Two token spellings map to the same canonical: fiber-section beams emit
# under ``section.fiber.*``; layered shells emit under ``material.fiber.*``
# (the recorder swaps ``section``→``material`` for shells, recorder PR #200 —
# the bucket layout is byte-identical, only the token name differs). A model
# carrying both kinds emits both buckets, so reads gather from every present
# spelling (mirrors the MPCO reader's dual-keying, ``_mpco.py`` discover).
_FIBER_TOKEN_TO_CANONICAL: dict[str, str] = {
    "section.fiber.stress": "fiber_stress",
    "section.fiber.strain": "fiber_strain",
    "material.fiber.stress": "fiber_stress",
    "material.fiber.strain": "fiber_strain",
}
# canonical → candidate tokens, ``section.fiber.*`` first (beam convention).
_FIBER_CANONICAL_TO_TOKENS: dict[str, tuple[str, ...]] = {
    "fiber_stress": ("section.fiber.stress", "material.fiber.stress"),
    "fiber_strain": ("section.fiber.strain", "material.fiber.strain"),
}


def section_canonical(token: str) -> Optional[str]:
    """Map a ``section.force`` / ``section.deformation`` column token to a
    neutral name (``P``→``axial_force``, ``kappaZ``→``curvature_z``), else
    ``None``."""
    t = token.strip()
    return _SECTION_FORCE_TO_CANONICAL.get(t) or _SECTION_DEFORM_TO_CANONICAL.get(t)

# Continuum stress/strain tokens come in two flavours across element
# classes: the digit form ``sigma11``/``eta11`` (stock FourNodeQuad etc.;
# the Ladruno solids tag strains ``eps11``) and the axis form
# ``sigma_xx``/``eps_xx``/``gamma_xy`` (BezierTri6 and
# the elements that follow the contract's recommended naming). ``eta`` and
# ``eps``/``epsilon`` are strain; ``gamma_xy`` is engineering shear (the
# same off-diagonal the digit ``eta12`` carries) → ``strain_xy``.
_CONTINUUM_DIGIT_RE = re.compile(
    r"^(?P<kind>sigma|eta|epsp|epsilon|eps)(?P<i>[123])(?P<j>[123])$"
)
_CONTINUUM_AXIS_RE = re.compile(
    r"^(?P<kind>sigma|epsp|epsilon|eps|gamma)_(?P<a>[xyz])(?P<b>[xyz])$"
)
# Token stem → canonical tensor root. ``epsp`` (plastic strain, the
# fork's natural tag following its ``eps11`` total-strain convention)
# is listed before ``eps`` in the alternations above so it wins the
# prefix race.
_KIND_TO_ROOT = {
    "sigma": "stress",
    "eta": "strain",
    "eps": "strain",
    "epsilon": "strain",
    "gamma": "strain",
    "epsp": "plastic_strain",
}
_BEAM_RE = re.compile(r"^(?P<base>[A-Za-z]+?)(?:_(?P<station>\d+))?$")


def continuum_canonical(token: str) -> Optional[str]:
    """Map a continuum token to ``stress_ij`` / ``strain_ij`` /
    ``plastic_strain_ij`` (else None).

    Handles both the digit form (``sigma12``→``stress_xy``,
    ``eta11``/``eps11``→``strain_xx``, ``epsp11``→``plastic_strain_xx``)
    and the axis form (``sigma_xy``→``stress_xy``, ``eps_xx``→
    ``strain_xx``, ``gamma_xy``→``strain_xy``, ``epsp_xx``→
    ``plastic_strain_xx``). Matching is case-insensitive, so the
    out-of-plane ``sigma33`` / ``sigma_zz`` (however the recorder cases
    them) resolve to ``stress_zz``.
    """
    t = token.strip().lower()
    m = _CONTINUUM_DIGIT_RE.match(t)
    if m is not None:
        root = _KIND_TO_ROOT[m.group("kind")]
        return f"{root}_{_AXIS[m.group('i')]}{_AXIS[m.group('j')]}"
    m = _CONTINUUM_AXIS_RE.match(t)
    if m is not None:
        root = _KIND_TO_ROOT[m.group("kind")]
        return f"{root}_{m.group('a')}{m.group('b')}"
    return None


def beam_canonical(token: str) -> Optional[tuple[str, Optional[int]]]:
    """``N_2`` → (``axial_force``, 2); ``N`` → (``axial_force``, None)."""
    m = _BEAM_RE.match(token.strip())
    if m is None:
        return None
    canonical = _BEAM_BASE_TO_CANONICAL.get(m.group("base"))
    if canonical is None:
        return None
    st = m.group("station")
    return canonical, (int(st) if st is not None else None)


# =====================================================================
# COLUMN_MAP block parsing
# =====================================================================

@dataclass(frozen=True)
class _Block:
    """One ``COLUMN_MAP`` row mapped to its slice of ``DATA``.

    Most blocks have ``multiplicity == 1`` (the block is exactly its
    ``comp_names``). Fiber buckets repeat a single column
    (``NUM_COMP==1``) once per fiber, so ``multiplicity`` == fibers and the
    block occupies ``len(comp_names) * multiplicity`` columns of ``DATA``.
    """
    level: int
    gauss_id: int
    comp_names: tuple[str, ...]
    col_start: int          # first DATA column for this block
    multiplicity: int = 1

    @property
    def width(self) -> int:
        return len(self.comp_names) * self.multiplicity


def _decode_str(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def parse_blocks(bucket_grp: "h5py.Group") -> list[_Block]:
    """Parse a bucket's ``COLUMN_MAP`` + ``COMP_NAMES`` into blocks.

    Fails loud if the per-block ``COMP_NAMES`` line count or the summed
    block widths disagree with ``DATA`` — a malformed bucket should
    surface, not silently mis-slice.
    """
    cm = bucket_grp["COLUMN_MAP"]
    levels = np.asarray(cm["LEVELS"][...], dtype=np.int64).flatten()
    gauss = np.asarray(cm["GAUSS_ID"][...], dtype=np.int64).flatten()
    lines = _decode_str(cm.attrs["COMP_NAMES"]).split("\n")
    n_blocks = int(levels.size)
    if "MULTIPLICITY" in cm:
        mult = np.asarray(cm["MULTIPLICITY"][...], dtype=np.int64).flatten()
    else:
        mult = np.ones(n_blocks, dtype=np.int64)
    if len(lines) != n_blocks:
        raise ValueError(
            f"COLUMN_MAP has {n_blocks} rows but COMP_NAMES has "
            f"{len(lines)} lines — malformed .ladruno bucket."
        )
    blocks: list[_Block] = []
    col = 0
    for i in range(n_blocks):
        names = tuple(t.strip() for t in lines[i].split(",") if t.strip())
        m = int(mult[i]) if i < mult.size else 1
        blocks.append(_Block(
            level=int(levels[i]), gauss_id=int(gauss[i]),
            comp_names=names, col_start=col, multiplicity=m,
        ))
        col += len(names) * m
    total = int(np.asarray(bucket_grp["DATA"].shape)[-1])
    if col != total:
        raise ValueError(
            f"COLUMN_MAP block widths sum to {col} but DATA has {total} "
            f"columns — malformed .ladruno bucket."
        )
    return blocks


def _class_prefix(bucket_key: str) -> str:
    """``12-Truss[1:0:0]`` → ``12-Truss`` (strip the bracket suffix)."""
    idx = bucket_key.find("[")
    return bucket_key[:idx] if idx >= 0 else bucket_key


def _gp_param_for(
    model_elements: "Optional[h5py.Group]", bucket_key: str,
) -> Optional[ndarray]:
    """Find ``QUADRATURE/GP_PARAM`` for the bucket's element class."""
    if model_elements is None:
        return None
    prefix = _class_prefix(bucket_key)
    for name in model_elements:
        if _class_prefix(name) != prefix:
            continue
        grp = model_elements[name]
        if "QUADRATURE" in grp and "GP_PARAM" in grp["QUADRATURE"]:
            return np.asarray(
                grp["QUADRATURE"]["GP_PARAM"][...], dtype=np.float64,
            )
    return None


def _select_rows(
    bucket_grp: "h5py.Group", element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray]]":
    """Return ``(row_indices, element_ids)`` after the id filter, or None."""
    all_ids = np.asarray(bucket_grp["ID"][...], dtype=np.int64).flatten()
    if element_ids is None:
        return np.arange(all_ids.size, dtype=np.int64), all_ids
    want = np.asarray(element_ids, dtype=np.int64)
    rows = np.where(np.isin(all_ids, want))[0]
    if rows.size == 0:
        return None
    return rows, all_ids[rows]


# =====================================================================
# Gauss reads (continuum stress / strain)
# =====================================================================

def gauss_available(on_elements: "h5py.Group") -> set[str]:
    out: set[str] = set()
    for token in on_elements:
        for key in on_elements[token]:
            try:
                blocks = parse_blocks(on_elements[token][key])
            except (KeyError, ValueError):
                continue
            for b in blocks:
                # Skip element-level (gauss_id<0) and fiber-expansion
                # (multiplicity>1) blocks — the latter are fiber stress/
                # strain, read via read_fibers, not continuum Gauss points.
                if b.gauss_id < 0 or b.multiplicity != 1:
                    continue
                for name in b.comp_names:
                    c = continuum_canonical(name)
                    if c is not None:
                        out.add(c)
    return out


def read_gauss_slab(
    on_elements: "h5py.Group",
    model_elements: "Optional[h5py.Group]",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumGP], element_index, natural_coords[sumGP, d])``.

    Stitches every bucket whose blocks expose ``component`` at a Gauss
    point. ``None`` if nothing matches.
    """
    values_parts: list[ndarray] = []
    eidx_parts: list[ndarray] = []
    coord_parts: list[ndarray] = []

    for token in on_elements:
        token_grp = on_elements[token]
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            gp_blocks = [
                b for b in blocks if b.gauss_id >= 0 and b.multiplicity == 1
            ]
            if not gp_blocks:
                continue
            # (block, col-within-DATA) pairs that hold the component.
            hits: list[tuple[_Block, int]] = []
            for b in gp_blocks:
                for off, name in enumerate(b.comp_names):
                    if continuum_canonical(name) == component:
                        hits.append((b, b.col_start + off))
            if not hits:
                continue
            sel = _select_rows(bucket, element_ids)
            if sel is None:
                continue
            rows, sel_ids = sel
            data = np.asarray(bucket["DATA"][...], dtype=np.float64)
            gp_param = _gp_param_for(model_elements, key)

            # One slab column per (element, matching GP). GP-major within
            # an element so natural_coords line up with element_index.
            for b, col in hits:
                vals = data[t_idx][:, rows, col]              # (T, E_sel)
                values_parts.append(vals)
                eidx_parts.append(sel_ids)
                if gp_param is not None and b.gauss_id < gp_param.shape[0]:
                    nat = np.tile(gp_param[b.gauss_id], (sel_ids.size, 1))
                else:
                    nat = np.zeros((sel_ids.size, 0), dtype=np.float64)
                coord_parts.append(nat)

    if not values_parts:
        return None
    values = np.concatenate(values_parts, axis=1)
    element_index = np.concatenate(eidx_parts)
    max_dim = max((c.shape[1] for c in coord_parts), default=0)
    coords = (
        np.concatenate(
            [c if c.shape[1] == max_dim
             else np.full((c.shape[0], max_dim), np.nan) for c in coord_parts],
            axis=0,
        )
        if max_dim else np.zeros((element_index.size, 0), dtype=np.float64)
    )
    return values, element_index, coords


# =====================================================================
# Line-station reads (beam internal-force diagrams)
# =====================================================================

def line_station_available(on_elements: "h5py.Group") -> set[str]:
    out: set[str] = set()
    for token in on_elements:
        for key in on_elements[token]:
            try:
                blocks = parse_blocks(on_elements[token][key])
            except (KeyError, ValueError):
                continue
            for b in blocks:
                for name in b.comp_names:
                    if b.level == 2:
                        c = section_canonical(name)
                        if c is not None:
                            out.add(c)
                    else:
                        bc = beam_canonical(name)
                        if bc is not None:
                            out.add(bc[0])
    return out


def read_line_station_slab(
    on_elements: "h5py.Group",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    model_elements: "Optional[h5py.Group]" = None,
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumS], element_index, station_natural_coord)``.

    Two bucket flavours feed the beam line diagram:

    * **element-level** (``localForce`` / ``basicForce``, ``LEVELS==0``) —
      one block of named columns. The ``localForce`` end-force convention
      flips the second station's sign so the diagram is a continuous
      internal-force line (parity with MPCO's ``_mpco_local_force_io``);
      single-station ``basicForce`` is left as-is. Stations are evenly
      spaced over ``[-1, +1]``.
    * **section-level** (``section.force`` / ``section.deformation``,
      ``LEVELS==2``) — one block per integration station. The station's
      natural coordinate is read from the element's ``QUADRATURE/GP_PARAM``
      keyed by ``GAUSS_ID`` (force-based beams), not synthesized.
    """
    values_parts: list[ndarray] = []
    eidx_parts: list[ndarray] = []
    station_parts: list[ndarray] = []

    for token in on_elements:
        token_grp = on_elements[token]
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            if any(b.level == 2 for b in blocks):
                res = _read_section_stations(
                    bucket, blocks, component, key, model_elements,
                    t_idx=t_idx, element_ids=element_ids,
                )
            else:
                res = _read_element_stations(
                    bucket, blocks, component, token,
                    t_idx=t_idx, element_ids=element_ids,
                )
            if res is None:
                continue
            v, ei, st = res
            values_parts.append(v)
            eidx_parts.append(ei)
            station_parts.append(st)

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eidx_parts),
        np.concatenate(station_parts),
    )


def _read_element_stations(
    bucket: "h5py.Group", blocks: list[_Block], component: str, token: str,
    *, t_idx: ndarray, element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Element-level (``localForce``/``basicForce``) line stations."""
    # station (1-based, None→1) → DATA column for this component.
    station_to_col: dict[int, int] = {}
    for b in blocks:
        for off, name in enumerate(b.comp_names):
            bc = beam_canonical(name)
            if bc is None or bc[0] != component:
                continue
            station = bc[1] if bc[1] is not None else 1
            station_to_col[station] = b.col_start + off
    if not station_to_col:
        return None
    sel = _select_rows(bucket, element_ids)
    if sel is None:
        return None
    rows, sel_ids = sel
    data = np.asarray(bucket["DATA"][...], dtype=np.float64)

    stations = sorted(station_to_col)
    n_st = len(stations)
    T = int(np.size(t_idx))
    E = sel_ids.size
    out = np.empty((T, E, n_st), dtype=np.float64)
    for s_i, st in enumerate(stations):
        out[:, :, s_i] = data[t_idx][:, rows, station_to_col[st]]
    # localForce end-force → internal-force sign continuity.
    if token == "localForce" and n_st == 2:
        out[:, :, 1] *= -1.0
    return (
        out.reshape(T, E * n_st),
        np.repeat(sel_ids, n_st),
        np.tile(_station_xi(n_st), E),
    )


def _read_section_stations(
    bucket: "h5py.Group", blocks: list[_Block], component: str,
    bucket_key: str, model_elements: "Optional[h5py.Group]",
    *, t_idx: ndarray, element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Section-level (``section.force``/``section.deformation``) stations.

    Each ``LEVELS==2`` block is one integration station; the station's
    natural coordinate comes from the element's ``GP_PARAM`` keyed by
    ``GAUSS_ID``.
    """
    # gauss_id → DATA column carrying this component.
    gp_to_col: dict[int, int] = {}
    for b in blocks:
        if b.level != 2:
            continue
        for off, name in enumerate(b.comp_names):
            if section_canonical(name) == component:
                gp_to_col[b.gauss_id] = b.col_start + off
    if not gp_to_col:
        return None
    sel = _select_rows(bucket, element_ids)
    if sel is None:
        return None
    rows, sel_ids = sel
    data = np.asarray(bucket["DATA"][...], dtype=np.float64)
    gp_param = _gp_param_for(model_elements, bucket_key)

    gauss_ids = sorted(gp_to_col)
    n_st = len(gauss_ids)
    T = int(np.size(t_idx))
    E = sel_ids.size
    out = np.empty((T, E, n_st), dtype=np.float64)
    xi = np.empty(n_st, dtype=np.float64)
    fallback = _station_xi(n_st)
    for s_i, gid in enumerate(gauss_ids):
        out[:, :, s_i] = data[t_idx][:, rows, gp_to_col[gid]]
        if gp_param is not None and 0 <= gid < gp_param.shape[0]:
            xi[s_i] = float(gp_param[gid, 0])
        else:
            xi[s_i] = fallback[s_i]
    return (
        out.reshape(T, E * n_st),
        np.repeat(sel_ids, n_st),
        np.tile(xi, E),
    )


def _station_xi(n_stations: int) -> ndarray:
    if n_stations == 1:
        return np.array([0.0])
    return np.linspace(-1.0, 1.0, n_stations)


# =====================================================================
# Element reads (token-driven raw blocks)
# =====================================================================
#
# Design (aligned with the parallel effort): element reads are
# **token-driven**, not neutral-canonical. The component IS the file's
# ``ON_ELEMENTS/<token>`` key (``basicForce`` / ``localForce`` /
# ``globalForce`` / ``force``) and the slab carries the element's full raw
# output vector ``(T, E, NUM_COLUMNS)`` in the file's ``COMP_NAMES`` column
# order. This keeps the element level purely file-driven (no ``P<dof>``
# grammar, no neutral remap); the neutral views live on the *other* levels
# — ``line_stations`` (``axial_force``…) and ``gauss`` (``stress_xx``…).
#
# Three details pinned here (flag if the sibling branch chose otherwise):
#   1. npe = raw ``NUM_COLUMNS`` (localForce→12, quad force→8, basicForce→1).
#   2. ``available_components(ELEMENTS)`` lists token keys whose blocks are
#      all element-level (``LEVELS==0``) — excludes Gauss stress/strain.
#   3. ``ElementSlab`` is unextended; column meaning follows the file's
#      ``COMP_NAMES`` (not carried on the slab).


def element_available(on_elements: "h5py.Group") -> set[str]:
    """ON_ELEMENTS token keys that are element-level (all blocks LEVELS==0).

    Token-driven: the token *is* the component. Gauss-level tokens
    (stress/strain, LEVELS==4) are excluded — they belong to read_gauss.
    """
    out: set[str] = set()
    for token in on_elements:
        token_grp = on_elements[token]
        saw_bucket = False
        all_element_level = True
        for key in token_grp:
            try:
                blocks = parse_blocks(token_grp[key])
            except (KeyError, ValueError):
                continue
            saw_bucket = True
            if any(b.level != 0 for b in blocks):
                all_element_level = False
                break
        if saw_bucket and all_element_level:
            out.add(token)
    return out


def read_element_slab(
    on_elements: "h5py.Group",
    token: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray]]":
    """Return ``(values[T, E, ncol], element_ids)`` for ``ON_ELEMENTS/<token>``.

    The raw element output block, sliced to the requested time steps and
    elements, in the file's ``COMP_NAMES`` column order. Buckets of different
    column width under one token (e.g. 2-D vs 3-D beams sharing a
    ``localForce`` token) can't share one ``(T, E, ncol)`` slab — the first
    width wins and mismatched buckets are skipped (homogeneous models, the
    common case, always match).
    """
    if token not in on_elements:
        return None
    token_grp = on_elements[token]
    values_parts: list[ndarray] = []
    eid_parts: list[ndarray] = []
    ncol_ref: "Optional[int]" = None

    for key in token_grp:
        bucket = token_grp[key]
        if "DATA" not in bucket or "ID" not in bucket:
            continue
        sel = _select_rows(bucket, element_ids)
        if sel is None:
            continue
        rows, sel_ids = sel
        data = np.asarray(bucket["DATA"][...], dtype=np.float64)  # (T, E, ncol)
        block = data[t_idx][:, rows, :]                           # (T, E_sel, ncol)
        if ncol_ref is None:
            ncol_ref = block.shape[2]
        elif block.shape[2] != ncol_ref:
            continue
        values_parts.append(block)
        eid_parts.append(sel_ids)

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eid_parts),
    )


# =====================================================================
# Fiber reads (fiber-section stress / strain)
# =====================================================================
#
# A ``.ladruno`` serialises a fiber section's per-fiber state under
# ``ON_ELEMENTS/section.fiber.stress`` (``…strain``) with one ``LEVELS==4``
# block per integration station whose ``MULTIPLICITY`` is the fiber count
# and ``NUM_COMP==1`` (the scalar uniaxial ``sigma11`` / ``eps11``). Fiber
# *geometry* (y, z, area, material) lives in
# ``MODEL/SECTION_ASSIGNMENTS/SECTION_<tag>/{FIBER_DATA, FIBER_MATERIALS}``,
# wired to (element, gauss) by that group's ``ASSIGNMENT[(elem_tag,
# gauss_id)]``. A layered-shell section serialises the same way (layers ==
# fibers), so this path covers both when the writer emits the bucket.


def fiber_available(on_elements: "h5py.Group") -> set[str]:
    """``fiber_stress`` / ``fiber_strain`` present under ``ON_ELEMENTS``."""
    return {
        _FIBER_TOKEN_TO_CANONICAL[token]
        for token in on_elements
        if token in _FIBER_TOKEN_TO_CANONICAL
    }


def _build_fiber_assignment(
    section_assignments: "Optional[h5py.Group]",
) -> "dict[tuple[int, int], tuple[ndarray, ndarray, ndarray, ndarray]]":
    """``(elem_tag, gauss_id)`` → ``(y, z, area, material_tag)`` per fiber."""
    out: dict[tuple[int, int], tuple[ndarray, ndarray, ndarray, ndarray]] = {}
    if section_assignments is None:
        return out
    for name in section_assignments:
        sa = section_assignments[name]
        if "FIBER_DATA" not in sa or "ASSIGNMENT" not in sa:
            continue
        fd = np.asarray(sa["FIBER_DATA"][...], dtype=np.float64).reshape(-1, 3)
        y, z, area = fd[:, 0], fd[:, 1], fd[:, 2]
        if "FIBER_MATERIALS" in sa:
            mats = np.asarray(
                sa["FIBER_MATERIALS"][...], dtype=np.int64,
            ).flatten()
        else:
            mats = np.full(fd.shape[0], -1, dtype=np.int64)
        assign = np.asarray(sa["ASSIGNMENT"][...], dtype=np.int64).reshape(-1, 2)
        for etag, gid in assign:
            out[(int(etag), int(gid))] = (y, z, area, mats)
    return out


def read_fiber_slab(
    on_elements: "h5py.Group",
    model_elements: "Optional[h5py.Group]",
    section_assignments: "Optional[h5py.Group]",
    component: str,
    *,
    t_idx: ndarray,
    element_ids: "Optional[ndarray]",
    gp_indices: "Optional[ndarray]",
) -> "Optional[tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumF], elem_index, gp_index, station_xi, y, z, area, mat)``.

    One slab column per (element, gauss point, fiber). ``None`` if the
    component (``fiber_stress`` / ``fiber_strain``) is absent.
    ``station_xi`` is the per-row station natural coordinate from the
    element's ``QUADRATURE/GP_PARAM`` keyed by ``GAUSS_ID`` (same
    source as the line-stations path); NaN for buckets without a
    recorded quadrature (e.g. layered shells, where the gauss id is a
    surface GP, not a beam station).
    """
    tokens = _FIBER_CANONICAL_TO_TOKENS.get(component)
    if tokens is None:
        return None
    present = [t for t in tokens if t in on_elements]
    if not present:
        return None
    assignment = _build_fiber_assignment(section_assignments)
    want_gp = None if gp_indices is None else set(int(g) for g in gp_indices)

    values_parts: list[ndarray] = []
    ei_parts: list[ndarray] = []
    gp_parts: list[ndarray] = []
    xi_parts: list[ndarray] = []
    y_parts: list[ndarray] = []
    z_parts: list[ndarray] = []
    area_parts: list[ndarray] = []
    mat_parts: list[ndarray] = []

    # Gather from every present spelling — a model with both fiber-section
    # beams (``section.fiber.*``) and layered shells (``material.fiber.*``)
    # emits both buckets.
    for token in present:
        token_grp = on_elements[token]
        # Beam-station natural coords only make sense for the beam
        # spelling — a layered shell's gauss_id is a SURFACE GP, so
        # its GP_PARAM (if any) is not a station along a member.
        is_beam_token = token.startswith("section.fiber")
        for key in token_grp:
            bucket = token_grp[key]
            try:
                blocks = parse_blocks(bucket)
            except (KeyError, ValueError):
                continue
            fiber_blocks = [
                b for b in blocks
                if b.gauss_id >= 0 and len(b.comp_names) == 1
            ]
            if not fiber_blocks:
                continue
            sel = _select_rows(bucket, element_ids)
            if sel is None:
                continue
            rows, sel_ids = sel
            data = np.asarray(bucket["DATA"][...], dtype=np.float64)
            gp_param = (
                _gp_param_for(model_elements, key) if is_beam_token
                else None
            )

            for b in sorted(fiber_blocks, key=lambda bb: bb.gauss_id):
                if want_gp is not None and b.gauss_id not in want_gp:
                    continue
                nfib = b.multiplicity
                # (T, E, nfib) — NUM_COMP==1, so the block is fiber-major.
                block = data[t_idx][:, rows, b.col_start:b.col_start + nfib]
                T = block.shape[0]
                E = sel_ids.size
                values_parts.append(block.reshape(T, E * nfib))
                ei_parts.append(np.repeat(sel_ids, nfib))
                gp_parts.append(np.full(E * nfib, b.gauss_id, dtype=np.int64))
                if (
                    gp_param is not None
                    and 0 <= b.gauss_id < gp_param.shape[0]
                ):
                    xi_val = float(gp_param[b.gauss_id, 0])
                else:
                    xi_val = np.nan
                xi_parts.append(np.full(E * nfib, xi_val, dtype=np.float64))
                # Per-element fiber geometry from the assigned section.
                ys = np.empty(E * nfib, dtype=np.float64)
                zs = np.empty(E * nfib, dtype=np.float64)
                ars = np.empty(E * nfib, dtype=np.float64)
                mts = np.empty(E * nfib, dtype=np.int64)
                for e_i, etag in enumerate(sel_ids):
                    geom = assignment.get((int(etag), b.gauss_id))
                    sl = slice(e_i * nfib, (e_i + 1) * nfib)
                    if geom is not None and geom[0].size == nfib:
                        ys[sl], zs[sl], ars[sl], mts[sl] = geom
                    else:
                        ys[sl] = np.nan
                        zs[sl] = np.nan
                        ars[sl] = np.nan
                        mts[sl] = -1
                y_parts.append(ys)
                z_parts.append(zs)
                area_parts.append(ars)
                mat_parts.append(mts)

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(ei_parts),
        np.concatenate(gp_parts),
        np.concatenate(xi_parts),
        np.concatenate(y_parts),
        np.concatenate(z_parts),
        np.concatenate(area_parts),
        np.concatenate(mat_parts),
    )
