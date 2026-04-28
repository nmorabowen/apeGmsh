"""Element-level result decoding for MPCOReader (Phase 11a).

MPCO stores element-level results under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/<token>/<bucket_key>/`` where
``<bucket_key>`` is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]`` and
each bucket holds:

- ``META`` — block-wise self-description of the column layout.
- ``ID`` ``(nElems, 1)`` — element tags (the bucket's row index).
- ``DATA/STEP_<k>`` ``(nElems, NUM_COLUMNS)`` — the flat response for
  every element at step ``k``.

This module discovers the gauss-level buckets a request touches,
validates each bucket against the response catalog, and returns
per-component data as ``(T, E_g, n_GP)`` blocks ready to be packed
into a ``GaussSlab``.

v1 scope (skipped with a clear log message when encountered):

- ``custom_rule_idx != 0`` (force-based beams with user IPs).
- ``header_idx != 0`` (heterogeneous response shapes).
- Element classes not yet in the catalog.

Anything outside the v1 scope simply contributes nothing to the slab —
the user can always extend the catalog and re-read the same file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    CatalogLookupError,
    MPCOElementKey,
    ResponseLayout,
    gauss_routing_for_canonical,
    is_catalogued,
    lookup,
    mpco_gauss_group_aliases,
    parse_mpco_element_key,
    unflatten,
)

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → MPCO -E token translation
# =====================================================================

def canonical_to_gauss_token(canonical: str) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a canonical component.

    Thin alias of :func:`gauss_routing_for_canonical` from the catalog
    module. The MPCO group name is identical to the OpenSees
    ``setResponse`` keyword, so both are returned together.

    Examples
    --------
    ``"stress_xx"``           → ``("stresses", "stress")``
    ``"membrane_force_xy"``   → ``("stresses", "stress")``
    ``"transverse_shear_xz"`` → ``("stresses", "stress")``
    ``"strain_xy"``           → ``("strains", "strain")``
    ``"curvature_xx"``        → ``("strains", "strain")``
    ``"displacement_x"``      → ``None`` (not a GP-level component).
    """
    return gauss_routing_for_canonical(canonical)


# =====================================================================
# Per-bucket descriptor — what the reader walks
# =====================================================================

@dataclass(frozen=True)
class _Bucket:
    """One ``ON_ELEMENTS/<token>/<bracket_key>`` group + its catalog layout.

    ``mpco_group_name`` records which alias group this bucket was
    found under (``stresses`` vs. ``material.stress`` etc.). The read
    site needs it to fetch the bucket dataset because alias groups
    can hold disjoint bucket sets.
    """
    bracket_key: str
    elem_key: MPCOElementKey
    layout: ResponseLayout
    mpco_group_name: str = ""


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_gauss_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_Bucket]]:
    """Walk ``ON_ELEMENTS/<mpco_group>`` and return catalogued GP buckets.

    The MPCO recorder may write continuum stress / strain / damage /
    plastic-strain under more than one group name across builds —
    modern builds use the ``material.X`` keyword family while legacy
    builds used the bare keyword. Some builds emit BOTH groups but
    only populate one; the other is left as an empty placeholder.
    We walk the full alias list from
    :func:`apeGmsh.solvers._element_response.mpco_gauss_group_aliases`
    and accumulate buckets across every group that exists, skipping
    duplicates so the same bucket key is never returned twice.
    Buckets that don't shape-match the catalog are filtered per-bucket.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the ON_ELEMENTS child name we found
        the first non-empty bucket under (the primary keyword if it
        had data, otherwise the first alias that did). Returns the
        primary keyword + empty list when no group has any buckets,
        or ``(None, [])`` if the canonical component has no
        gauss-token mapping at all.
    """
    mapping = canonical_to_gauss_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    candidate_names = mpco_gauss_group_aliases(mpco_group_name)

    out: list[_Bucket] = []
    seen_keys: set[str] = set()
    found_name: str | None = None
    for name in candidate_names:
        if name not in on_elements_grp:
            continue
        token_grp = on_elements_grp[name]
        for bracket_key in token_grp:
            if bracket_key in seen_keys:
                continue
            try:
                elem_key = parse_mpco_element_key(bracket_key)
            except ValueError:
                continue
            if elem_key.custom_rule_idx != 0 or elem_key.header_idx != 0:
                continue
            if not is_catalogued(
                elem_key.class_name, elem_key.int_rule, catalog_token,
            ):
                continue
            layout = lookup(
                elem_key.class_name, elem_key.int_rule, catalog_token,
            )
            out.append(_Bucket(
                bracket_key=bracket_key,
                elem_key=elem_key,
                layout=layout,
                mpco_group_name=name,
            ))
            seen_keys.add(bracket_key)
            if found_name is None:
                found_name = name
    return (found_name or mpco_group_name, out)


# =====================================================================
# META validation
# =====================================================================

def validate_bucket_meta(
    bucket_grp: "h5py.Group",
    layout: ResponseLayout,
    *,
    bracket_key: str,
) -> None:
    """Cross-check the bucket's META against the catalog layout.

    Catches MPCO-version drift early. Validation rules:

    1. ``NUM_COLUMNS == layout.flat_size_per_element``
    2. Per-GP block count is ``n_gauss_points``
    3. Every block contributes ``n_components_per_gp`` columns
    4. Block ``i`` is tagged ``GAUSS_IDS[i] == i`` (sequential GP order)
    5. ``sum(MULTIPLICITY[i] * NUM_COMPONENTS[i]) == NUM_COLUMNS``
    """
    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        raise ValueError(
            f"MPCO bucket {bracket_key!r} is missing the NUM_COLUMNS attribute."
        )
    num_columns = int(_attr_scalar(num_columns_attr))
    expected = layout.flat_size_per_element
    if num_columns != expected:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: NUM_COLUMNS={num_columns} but "
            f"catalog layout for ({layout.class_tag},{layout.n_gauss_points} "
            f"GP × {layout.n_components_per_gp} comp) expects {expected}."
        )

    meta = bucket_grp.get("META")
    if meta is None:
        raise ValueError(
            f"MPCO bucket {bracket_key!r} is missing the META subgroup."
        )

    multiplicity = np.asarray(meta["MULTIPLICITY"][...]).flatten().astype(np.int64)
    gauss_ids = np.asarray(meta["GAUSS_IDS"][...]).flatten().astype(np.int64)
    num_components = np.asarray(meta["NUM_COMPONENTS"][...]).flatten().astype(np.int64)

    if not (multiplicity.size == gauss_ids.size == num_components.size):
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META block arrays have mismatched "
            f"sizes (MULTIPLICITY={multiplicity.size}, "
            f"GAUSS_IDS={gauss_ids.size}, NUM_COMPONENTS={num_components.size})."
        )

    n_gp = layout.n_gauss_points
    if gauss_ids.size != n_gp:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META has {gauss_ids.size} blocks "
            f"but catalog expects {n_gp} (one per GP)."
        )
    if not np.array_equal(gauss_ids, np.arange(n_gp, dtype=np.int64)):
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META GAUSS_IDS={gauss_ids.tolist()} "
            f"is not sequential 0..{n_gp - 1} (catalog assumes GP-slowest order)."
        )
    if not np.all(num_components == layout.n_components_per_gp):
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META NUM_COMPONENTS="
            f"{num_components.tolist()}; expected every block to contribute "
            f"{layout.n_components_per_gp} columns."
        )

    total = int(np.sum(multiplicity * num_components))
    if total != num_columns:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META block-sum {total} != "
            f"NUM_COLUMNS {num_columns}."
        )


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_bucket_slab(
    bucket_grp: "h5py.Group",
    bucket: _Bucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Read one component's slab from one bucket.

    Returns
    -------
    (values, element_index, natural_coords) or None
        ``values``: ``(T, E_g_sel * n_GP)`` flat per the GaussSlab schema.
        ``element_index``: ``(E_g_sel * n_GP,)`` element tag per column.
        ``natural_coords``: ``(E_g_sel * n_GP, dim)`` parent-domain coords.
        Returns ``None`` if no elements survive filtering or the
        component is not in this bucket's layout.
    """
    layout = bucket.layout
    if canonical_component not in layout.component_layout:
        return None
    component_idx = layout.component_layout.index(canonical_component)

    validate_bucket_meta(
        bucket_grp, layout, bracket_key=bucket.bracket_key,
    )

    # Element IDs in row order.
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

    data_grp = bucket_grp["DATA"]
    step_keys = sorted(
        (k for k in data_grp.keys() if k.startswith("STEP_")),
        key=lambda s: int(s.split("_", 1)[1]),
    )
    if not step_keys:
        return None

    n_gp = layout.n_gauss_points
    flat_size = layout.flat_size_per_element
    E_g = sel_ids.size
    T = int(np.size(t_idx))

    # Build flat[T, E_g, flat_size] one step at a time. h5py supports
    # fancy row indexing only via plain numpy slicing on a single axis
    # at a time; we read per-step to stay simple and lazy.
    flat = np.empty((T, E_g, flat_size), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_ds = data_grp[step_keys[int(k)]]
        # Read all rows in one go, then index — faster than fancy h5py
        # row indexing for typical bucket sizes.
        step_arr = np.asarray(step_ds[...], dtype=np.float64)
        flat[i, :, :] = step_arr[sel_rows, :]

    decoded = unflatten(flat, layout)
    per_gp = decoded[canonical_component]   # (T, E_g, n_GP)

    # Pack into (T, sum_GP=E_g * n_GP) — GaussSlab convention.
    values = per_gp.reshape(T, E_g * n_gp)
    element_index = np.repeat(sel_ids, n_gp).astype(np.int64)
    natural_coords = np.tile(layout.natural_coords, (E_g, 1))

    return values, element_index, natural_coords


# =====================================================================
# Internal helpers
# =====================================================================

def _attr_scalar(value):
    """Coerce an h5py attribute value to a scalar (mirrors _mpco._attr_scalar)."""
    if hasattr(value, "size") and getattr(value, "size", 0) == 1:
        return value.item() if hasattr(value, "item") else value[0]
    if hasattr(value, "__len__") and len(value) == 1:
        return value[0]
    return value
