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
  ``<base>_<station_idx>`` (1-based). For 3-D it's
  ``"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,..,Mz_2"`` (12 columns,
  2 stations × 6 components). For 2-D it's
  ``"0.N_1,Vy_1,Mz_1,N_2,Vy_2,Mz_2"`` (6 columns, 2 stations × 3).
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

_TOKEN_RE = re.compile(r"^(?P<base>[A-Za-z]+)_(?P<station>\d+)$")


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
    components_raw, *, bracket_key: str = "<unknown>",
) -> tuple[int, dict[str, list[int]]]:
    """Parse ``META/COMPONENTS`` for a localForce bucket.

    Returns ``(n_stations, {canonical: [col_idx_1, col_idx_2, ...]})``
    where each value list has length ``n_stations`` — column indices
    into the per-step ``DATA/STEP_<i>`` row, ordered by station index.

    Tokens whose base name isn't in
    :data:`_LOCAL_FORCE_BASE_TO_CANONICAL` are silently ignored (so
    a future MPCO writer can add unknown extras without breaking
    discovery).
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

    # base -> {station_index: column_index}
    by_base: dict[str, dict[int, int]] = {}
    for col_idx, tok in enumerate(tokens):
        m = _TOKEN_RE.match(tok)
        if m is None:
            # Foreign token; skip.
            continue
        base = m.group("base")
        station = int(m.group("station"))
        canonical = _LOCAL_FORCE_BASE_TO_CANONICAL.get(base)
        if canonical is None:
            continue
        by_base.setdefault(canonical, {})[station] = col_idx

    if not by_base:
        raise ValueError(
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

    Buckets whose META can't be parsed are silently skipped — they
    won't show up in ``available_components()`` and reads against
    them return empty.
    """
    if "localForce" not in on_elements_grp:
        return []
    lf_grp = on_elements_grp["localForce"]
    out: list[_LocalForceBucket] = []
    for bracket_key in lf_grp:
        bucket_grp = lf_grp[bracket_key]
        meta = bucket_grp.get("META")
        if meta is None or "COMPONENTS" not in meta:
            continue
        try:
            n_stations, layout = parse_local_force_layout(
                meta["COMPONENTS"][...], bracket_key=bracket_key,
            )
        except ValueError:
            continue
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

    data_grp = bucket_grp.get("DATA")
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
    "parse_local_force_layout",
    "discover_local_force_buckets",
    "available_components_in_local_force",
    "read_local_force_bucket_slab",
]
