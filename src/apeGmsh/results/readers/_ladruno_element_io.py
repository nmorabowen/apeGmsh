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

# Beam section / end forces (localForce, basicForce, section.force).
_BEAM_BASE_TO_CANONICAL: dict[str, str] = {
    "N": "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T": "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

# Continuum stress/strain tokens come in two flavours across element
# classes: the digit form ``sigma11``/``eta11`` (stock FourNodeQuad etc.)
# and the axis form ``sigma_xx``/``eps_xx``/``gamma_xy`` (BezierTri6 and
# the elements that follow the contract's recommended naming). ``eta`` and
# ``eps``/``epsilon`` are strain; ``gamma_xy`` is engineering shear (the
# same off-diagonal the digit ``eta12`` carries) → ``strain_xy``.
_CONTINUUM_DIGIT_RE = re.compile(r"^(?P<kind>sigma|eta)(?P<i>[123])(?P<j>[123])$")
_CONTINUUM_AXIS_RE = re.compile(
    r"^(?P<kind>sigma|eps|epsilon|gamma)_(?P<a>[xyz])(?P<b>[xyz])$"
)
_BEAM_RE = re.compile(r"^(?P<base>[A-Za-z]+?)(?:_(?P<station>\d+))?$")


def continuum_canonical(token: str) -> Optional[str]:
    """Map a continuum token to ``stress_ij`` / ``strain_ij`` (else None).

    Handles both the digit form (``sigma12``→``stress_xy``,
    ``eta11``→``strain_xx``) and the axis form (``sigma_xy``→``stress_xy``,
    ``eps_xx``→``strain_xx``, ``gamma_xy``→``strain_xy``).
    """
    t = token.strip()
    m = _CONTINUUM_DIGIT_RE.match(t)
    if m is not None:
        root = "stress" if m.group("kind") == "sigma" else "strain"
        return f"{root}_{_AXIS[m.group('i')]}{_AXIS[m.group('j')]}"
    m = _CONTINUUM_AXIS_RE.match(t)
    if m is not None:
        root = "stress" if m.group("kind") == "sigma" else "strain"
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
    """One ``COLUMN_MAP`` row mapped to its slice of ``DATA``."""
    level: int
    gauss_id: int
    comp_names: tuple[str, ...]
    col_start: int          # first DATA column for this block

    @property
    def width(self) -> int:
        return len(self.comp_names)


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
    if len(lines) != n_blocks:
        raise ValueError(
            f"COLUMN_MAP has {n_blocks} rows but COMP_NAMES has "
            f"{len(lines)} lines — malformed .ladruno bucket."
        )
    blocks: list[_Block] = []
    col = 0
    for i in range(n_blocks):
        names = tuple(t.strip() for t in lines[i].split(",") if t.strip())
        blocks.append(_Block(
            level=int(levels[i]), gauss_id=int(gauss[i]),
            comp_names=names, col_start=col,
        ))
        col += len(names)
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
                if b.gauss_id < 0:
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
            gp_blocks = [b for b in blocks if b.gauss_id >= 0]
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
) -> "Optional[tuple[ndarray, ndarray, ndarray]]":
    """Return ``(values[T, sumS], element_index, station_natural_coord)``.

    Decodes beam force buckets (``localForce``, ``basicForce``,
    ``section.force``) from their ``COMP_NAMES`` tokens. The
    ``localForce`` end-force convention flips the second station's sign so
    the diagram is a continuous internal-force line (parity with MPCO's
    ``_mpco_local_force_io``); single-station ``basicForce`` is left as-is.
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
                continue
            sel = _select_rows(bucket, element_ids)
            if sel is None:
                continue
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
            values_parts.append(out.reshape(T, E * n_st))
            eidx_parts.append(np.repeat(sel_ids, n_st))
            station_parts.append(np.tile(_station_xi(n_st), E))

    if not values_parts:
        return None
    return (
        np.concatenate(values_parts, axis=1),
        np.concatenate(eidx_parts),
        np.concatenate(station_parts),
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
