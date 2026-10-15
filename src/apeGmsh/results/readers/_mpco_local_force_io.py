"""``localForce`` element-level decoding for MPCOReader.

Companion to :mod:`_mpco_line_io`. Where that module reads
``RESULTS/ON_ELEMENTS/section.force/`` (force-based beams with
section integration points and ``GP_X`` natural coords), this one
reads the much simpler ``RESULTS/ON_ELEMENTS/localForce/`` bucket
that **stiffness-formulated beams** (``ElasticBeam2d``,
``ElasticBeam3d``) emit — the 6 / 12 end-force components in the
element local frame.

Bucket shape:

* ``META/COMPONENTS``  — single byte-string of the form
  ``"<hdr>.tok_1,tok_2,...,tok_N"`` where each token is
  ``<base>_<station_idx>`` (1-based; some classes drop the underscore,
  ``N1``). For 3-D it's
  ``"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,..,Mz_2"`` (12 columns,
  2 stations × 6 components). For 2-D it's
  ``"0.N_1,V_1,M_1,N_2,V_2,M_2"`` (ElasticBeam2d, ForceBeamColumn2d,
  ElasticTimoshenkoBeam2d) or ``"0.N1,V1,M1,N2,V2,M2"``
  (DispBeamColumn2d and its fork variant): 6 columns, 2 stations × 3.
* ``ID``                — ``(E, 1)`` element IDs.
* ``DATA/STEP_<i>``     — ``(E, NUM_COMPONENTS)`` per-element flat
  values for one step.

Stations live at the parent natural coordinates ``ξ ∈ {-1, +1}``;
``localForce`` carries no GP_X (integration points don't exist for
stiffness beams), and we synthesize the two coords here.

The base→canonical translation::

    N  → axial_force
    Vy → shear_y
    Vz → shear_z
    T  → torsion
    My → bending_moment_y
    Mz → bending_moment_z

plus the 2-D spellings ``V → shear_y`` and ``M → bending_moment_z``,
under the gates in :func:`parse_local_force_layout`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Token → canonical map
# =====================================================================

_LOCAL_FORCE_BASE_TO_CANONICAL: dict[str, str] = {
    "N":  "axial_force",
    "Vy": "shear_y",
    "Vz": "shear_z",
    "T":  "torsion",
    "My": "bending_moment_y",
    "Mz": "bending_moment_z",
}

#: The 2-D spelling of a ``localForce`` bucket: one shear and one moment
#: per end (``N_1,V_1,M_1,..``). A 2-D beam has only ``Vy``/``Mz``. Kept
#: out of the base map so it applies under the gates of
#: :func:`parse_local_force_layout` only.
_LOCAL_FORCE_2D_BASE_TO_CANONICAL: dict[str, str] = {
    "V": "shear_y",
    "M": "bending_moment_z",
}

#: ``N_1`` and ``N1`` both occur (DispBeamColumn2d drops the underscore).
_TOKEN_RE = re.compile(r"^(?P<base>[A-Za-z]+)_?(?P<station>\d+)$")

#: MPCO bracket keys read ``<classTag>-<ClassName>[<rule>]``.
_BRACKET_CLASS_RE = re.compile(r"^\d+-(?P<cls>[^\[]+)\[")


class NoLineStationTokensError(ValueError):
    """A ``localForce`` bucket names no station force or moment at all.

    A truss's ``C1..C4`` (MPCO's names for an untagged response), a
    corotational truss's bare ``N``, a wall's ``Fx_i``: these are element
    force vectors, not beam line stations, so line-station discovery passes
    over them. Every other parse failure is a malformed beam bucket and
    raises.
    """


# =====================================================================
# Bucket descriptor
# =====================================================================

@dataclass(frozen=True)
class _LocalForceBucket:
    """One ``ON_ELEMENTS/localForce/<bracket_key>`` group."""
    bracket_key: str
    n_stations: int
    layout: dict[str, list[int]]   # canonical -> column indices (length = n_stations)


# =====================================================================
# META layout parser
# =====================================================================

def parse_local_force_layout(
    components_raw, *, data_width: int, bracket_key: str = "<unknown>",
) -> tuple[int, dict[str, list[int]]]:
    """Parse ``META/COMPONENTS`` for a localForce bucket.

    Returns ``(n_stations, {canonical: [col_idx_1, col_idx_2, ...]})``
    where each value list has length ``n_stations`` — column indices
    into the per-step ``DATA/STEP_<i>`` row, ordered by station index.

    ``data_width`` is the column count of the bucket's ``DATA/STEP_<i>``
    rows. META must name exactly that many columns, or the bucket is
    refused: a layout that does not cover the data cannot be trusted.

    The 2-D spelling ``V``/``M`` reads as ``shear_y``/``bending_moment_z``
    only when META and DATA agree, the bucket spells no 3-D base
    (``Vy``, ``Vz``, ``T``, ``My``, ``Mz``) beside it, and the bracket's
    element class is not a ``...3d`` class. ComponentElement3d and
    ForceBeamColumnCBDI3d tag their 12-component localForce with the six
    2-D names; MPCO then writes six zero-filled columns, which the class
    gate refuses rather than read as shears and moments.

    Raises :class:`NoLineStationTokensError` when no token names a
    station force or moment, and :class:`ValueError` for every other
    malformed bucket. Other unknown tokens (``Q_1`` of a warping beam) are
    ignored.
    """
    if hasattr(components_raw, "__len__") and not isinstance(
        components_raw, (bytes, str)
    ):
        components_raw = components_raw[0]
    if isinstance(components_raw, bytes):
        components_raw = components_raw.decode("ascii")

    last_dot = components_raw.rfind(".")
    if last_dot < 0:
        raise ValueError(
            f"localForce bucket {bracket_key!r}: META/COMPONENTS "
            f"{components_raw!r} has no header-dot separator."
        )
    csv = components_raw[last_dot + 1:]
    tokens = [t for t in csv.split(",") if t]
    if not tokens:
        raise ValueError(
            f"localForce bucket {bracket_key!r}: empty component list."
        )

    if len(tokens) != data_width:
        raise ValueError(
            f"localForce bucket {bracket_key!r}: META/COMPONENTS names "
            f"{len(tokens)} columns ({components_raw!r}) but its DATA rows "
            f"are {data_width} wide; refusing to guess the layout."
        )

    parsed: list[tuple[int, str, int]] = []      # (col_idx, base, station)
    for col_idx, tok in enumerate(tokens):
        m = _TOKEN_RE.match(tok)
        if m is not None:
            parsed.append((col_idx, m.group("base"), int(m.group("station"))))
    bases = {base for _c, base, _s in parsed}

    base_map = dict(_LOCAL_FORCE_BASE_TO_CANONICAL)
    spelled_2d = bases & set(_LOCAL_FORCE_2D_BASE_TO_CANONICAL)
    if spelled_2d:
        spelled_3d = bases & {"Vy", "Vz", "T", "My", "Mz"}
        if spelled_3d:
            raise ValueError(
                f"localForce bucket {bracket_key!r}: COMPONENTS="
                f"{components_raw!r} mixes the 2-D spelling "
                f"{sorted(spelled_2d)} with the 3-D {sorted(spelled_3d)}; "
                f"the shear and moment axes are ambiguous."
            )
        cls = _BRACKET_CLASS_RE.match(bracket_key)
        if cls is not None and cls.group("cls").lower().endswith("3d"):
            raise ValueError(
                f"localForce bucket {bracket_key!r}: the 3-D element class "
                f"{cls.group('cls')!r} records its localForce under the 2-D "
                f"names {components_raw!r}; its shear and moment axes "
                f"cannot be read from them."
            )
        base_map.update(_LOCAL_FORCE_2D_BASE_TO_CANONICAL)

    # canonical -> {station_index: column_index}
    by_base: dict[str, dict[int, int]] = {}
    for col_idx, base, station in parsed:
        canonical = base_map.get(base)
        if canonical is None:
            continue        # an extra this reader does not map (``Q_1``)
        stations = by_base.setdefault(canonical, {})
        if station in stations:
            raise ValueError(
                f"localForce bucket {bracket_key!r}: COMPONENTS="
                f"{components_raw!r} names {canonical!r} at station "
                f"{station} twice."
            )
        stations[station] = col_idx

    if not by_base:
        raise NoLineStationTokensError(
            f"localForce bucket {bracket_key!r}: no recognized "
            f"force/moment tokens in COMPONENTS={components_raw!r}."
        )

    # Use the maximum station index across canonicals to define n_stations
    # and produce ordered column-index lists. Stations are 1-based;
    # missing stations (e.g. partial recordings) raise so we don't silently
    # produce a sparse layout.
    n_stations = max(max(d) for d in by_base.values())
    layout: dict[str, list[int]] = {}
    for canonical, station_to_col in by_base.items():
        cols = [station_to_col.get(s) for s in range(1, n_stations + 1)]
        if any(c is None for c in cols):
            missing = [s for s in range(1, n_stations + 1)
                       if s not in station_to_col]
            raise ValueError(
                f"localForce bucket {bracket_key!r}: canonical "
                f"{canonical!r} missing station indices {missing} "
                f"(have {sorted(station_to_col)})."
            )
        layout[canonical] = cols    # type: ignore[assignment]

    return n_stations, layout


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_local_force_buckets(
    on_elements_grp: "h5py.Group",
) -> list[_LocalForceBucket]:
    """Walk ``ON_ELEMENTS/localForce/`` and return parsed buckets.

    A malformed bucket raises :class:`ValueError` naming it: no
    ``META/COMPONENTS``, step rows of differing widths, or any refusal of
    :func:`parse_local_force_layout`. Two kinds of bucket are passed over,
    because they hold no line-station values: one with no recorded step,
    and one whose tokens name no station force or moment
    (:class:`NoLineStationTokensError`, a truss's force vector).
    """
    if "localForce" not in on_elements_grp:
        return []
    lf_grp = on_elements_grp["localForce"]
    out: list[_LocalForceBucket] = []
    for bracket_key in lf_grp:
        bucket_grp = lf_grp[bracket_key]
        if "META" not in bucket_grp or "COMPONENTS" not in bucket_grp["META"]:
            raise ValueError(
                f"localForce bucket {bracket_key!r} has no META/COMPONENTS; "
                f"its columns cannot be named."
            )
        widths: set[int] = set()
        if "DATA" in bucket_grp:
            data_grp = bucket_grp["DATA"]
            widths = {
                int(data_grp[k].shape[1]) if len(data_grp[k].shape) == 2
                else -1
                for k in data_grp if k.startswith("STEP_")
            }
        if not widths:
            continue        # no recorded step: nothing to read
        if len(widths) != 1 or -1 in widths:
            raise ValueError(
                f"localForce bucket {bracket_key!r}: DATA/STEP_* rows are "
                f"not one 2-D width (widths {sorted(widths)}, -1 = not 2-D)."
            )
        try:
            n_stations, layout = parse_local_force_layout(
                bucket_grp["META"]["COMPONENTS"][...],
                data_width=widths.pop(), bracket_key=bracket_key,
            )
        except NoLineStationTokensError:
            continue        # an element force vector, not line stations
        out.append(_LocalForceBucket(
            bracket_key=bracket_key,
            n_stations=n_stations,
            layout=layout,
        ))
    return out


def available_components_in_local_force(
    on_elements_grp: "h5py.Group",
) -> set[str]:
    """Aggregate canonicals across every parseable localForce bucket."""
    out: set[str] = set()
    for bucket in discover_local_force_buckets(on_elements_grp):
        out.update(bucket.layout.keys())
    return out


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_local_force_bucket_slab(
    bucket_grp: "h5py.Group",
    bucket: _LocalForceBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Read one component's slab from one localForce bucket.

    Returns
    -------
    (values, element_index, station_natural_coord) or None
        ``values`` shape ``(T, E_sel * n_stations)`` flat per the
        ``LineStationSlab`` schema. Stations are interleaved per
        element: ``[e0_s1, e0_s2, e1_s1, e1_s2, ...]``.
        ``station_natural_coord`` returns the parent ξ for each
        column — ``[-1, +1]`` repeated for n_stations==2.
        Returns ``None`` if no rows survive filtering, the canonical
        isn't in the bucket layout, or the bucket has no recorded
        steps.
    """
    cols = bucket.layout.get(canonical_component)
    if cols is None:
        return None

    all_ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)
    if element_ids is None:
        sel_rows = np.arange(all_ids.size, dtype=np.int64)
        sel_ids = all_ids
    else:
        requested = np.asarray(element_ids, dtype=np.int64)
        mask = np.isin(all_ids, requested)
        sel_rows = np.where(mask)[0]
        sel_ids = all_ids[sel_rows]
        if sel_rows.size == 0:
            return None

    data_grp = bucket_grp["DATA"] if "DATA" in bucket_grp else None
    if data_grp is None:
        return None
    step_keys = sorted(
        (k for k in data_grp.keys() if k.startswith("STEP_")),
        key=lambda s: int(s.split("_", 1)[1]),
    )
    if not step_keys:
        return None

    n_stations = bucket.n_stations
    E = sel_ids.size
    T = int(np.size(t_idx))
    cols_arr = np.asarray(cols, dtype=np.int64)

    # values[t, e, s] = step_arr[sel_rows[e], cols[s]]
    out = np.empty((T, E, n_stations), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        # NumPy fancy indexing: pick rows then cols
        out[i] = step_arr[np.ix_(sel_rows, cols_arr)]

    # Convert OpenSees ``localForce`` (end resisting forces on the
    # element from the joints) to internal-section-force convention so
    # the slab matches what ``section.force`` reports and adjacent
    # elements line up at shared nodes. For ``n_stations == 2``: the
    # raw value at station 2 is the force the joint exerts on the
    # element at xi=+1, i.e. the negative of the internal force on
    # the cross-section there. Multiply station 2 by -1.
    if n_stations == 2:
        out[:, :, 1] *= -1.0

    values = out.reshape(T, E * n_stations)
    element_index = np.repeat(sel_ids, n_stations).astype(np.int64)
    station_xi = _station_natural_coords(n_stations)
    station_natural_coord = np.tile(station_xi, E).astype(np.float64)

    return values, element_index, station_natural_coord


def _station_natural_coords(n_stations: int) -> ndarray:
    """Natural-coordinate vector for a localForce bucket.

    For 2 stations (the only case ElasticBeam* emits today) this is
    ``[-1, +1]``. Generalises linearly if a future MPCO recorder
    writes multi-station local-frame data.
    """
    if n_stations == 1:
        return np.array([0.0])
    return np.linspace(-1.0, 1.0, n_stations)


__all__ = [
    "_LocalForceBucket",
    "NoLineStationTokensError",
    "parse_local_force_layout",
    "discover_local_force_buckets",
    "available_components_in_local_force",
    "read_local_force_bucket_slab",
]
