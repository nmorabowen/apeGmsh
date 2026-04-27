"""Nodal-forces element-level result decoding for MPCOReader (Phase 11b).

MPCO stores per-element-node force vectors for closed-form line
elements (``ElasticBeam{2d,3d}``, ``ElasticTimoshenkoBeam{2d,3d}``,
``ModElasticBeam2d``) under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/<frame>/<bucket_key>/`` where
``<frame>`` is ``globalForce`` or ``localForce`` and
``<bucket_key>`` is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]``.
Each bucket holds the standard ``META`` / ``ID`` / ``DATA/STEP_<k>``
triplet, with these closed-form-specific conventions:

- Single META block (no per-IP segmentation): ``MULTIPLICITY = [[1]]``,
  ``GAUSS_IDS = [[-1]]`` (sentinel for "no integration point"),
  ``NUM_COMPONENTS = [[n_nodes * n_components_per_node]]``.
- ``META/COMPONENTS`` lists per-element-node names with a node
  suffix, e.g. ``"0.Px_1,Py_1,Pz_1,Mx_1,My_1,Mz_1,Px_2,...,Mz_2"``
  for a 3D ``globalForce`` (12 cols) or
  ``"0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,...,Mz_2"`` for ``localForce``.
- DATA layout is node-slowest / component-fastest, matching
  :data:`NODAL_FORCE_CATALOG`'s ``component_layout`` order.

This module mirrors :mod:`_mpco_element_io` and :mod:`_mpco_line_io`.
For each requested canonical component (e.g.
``"nodal_resisting_force_x"`` or
``"nodal_resisting_moment_local_z"``) it walks ``ON_ELEMENTS/
<frame>/`` for catalogued classes, validates META, reads DATA, and
emits per-element-node slab parts ready to stitch into an
``ElementSlab(values=(T, E, n_nodes), ...)``.

v1 scope (silent skip when encountered):

- ``custom_rule_idx != 0`` (no custom-rule closed-form beams).
- ``header_idx != 0`` (heterogeneous response shapes).
- Element classes not in :data:`NODAL_FORCE_CATALOG`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    MPCOElementKey,
    NodalForceLayout,
    gauss_routing_for_canonical,
    is_nodal_force_catalogued,
    lookup_nodal_force,
    parse_mpco_element_key,
    unflatten_nodal,
)

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → MPCO -E token translation (nodal_forces topology)
# =====================================================================

def canonical_to_nodal_force_token(
    canonical: str,
) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a nodal-forces component.

    Thin alias of :func:`gauss_routing_for_canonical` with
    ``topology="nodal_forces"``. The MPCO group name is the OpenSees
    recorder keyword (``"globalForce"`` / ``"localForce"``).

    Examples
    --------
    ``"nodal_resisting_force_x"``       → ``("globalForce", "global_force")``
    ``"nodal_resisting_moment_local_z"`` → ``("localForce", "local_force")``
    ``"axial_force"``                    → ``None`` (line-stations
                                                   topology, not here).
    """
    return gauss_routing_for_canonical(canonical, topology="nodal_forces")


# =====================================================================
# Per-bucket descriptor — what the reader walks
# =====================================================================

@dataclass(frozen=True)
class _NodalBucket:
    """One ``ON_ELEMENTS/<frame>/<bracket_key>`` group + its catalog layout."""
    bracket_key: str
    elem_key: MPCOElementKey
    layout: NodalForceLayout


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_nodal_force_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_NodalBucket]]:
    """Walk ``ON_ELEMENTS/<globalForce|localForce>/`` for catalogued buckets.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the ON_ELEMENTS child name we looked
        under (``"globalForce"`` / ``"localForce"``), or ``None`` if
        the canonical component has no nodal-forces routing.
        ``buckets`` is the list whose ``(class_name, catalog_token)``
        is in :data:`NODAL_FORCE_CATALOG` and whose
        ``custom_rule_idx`` / ``header_idx`` are both 0.
    """
    mapping = canonical_to_nodal_force_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [])

    token_grp = on_elements_grp[mpco_group_name]
    out: list[_NodalBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            continue
        if elem_key.custom_rule_idx != 0 or elem_key.header_idx != 0:
            continue
        if not is_nodal_force_catalogued(elem_key.class_name, catalog_token):
            continue
        layout = lookup_nodal_force(elem_key.class_name, catalog_token)
        out.append(_NodalBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            layout=layout,
        ))
    return (mpco_group_name, out)


# =====================================================================
# META validation
# =====================================================================

def validate_nodal_bucket_meta(
    bucket_grp: "h5py.Group",
    layout: NodalForceLayout,
    *,
    bracket_key: str,
) -> None:
    """Cross-check the bucket's META against the catalog layout.

    Closed-form elements have a single META block (no per-IP
    segmentation). Validation rules:

    1. ``NUM_COLUMNS == layout.flat_size_per_element``
    2. Single-block META: ``len(MULTIPLICITY) == 1``
    3. ``GAUSS_IDS[0] == -1`` (sentinel; no integration point)
    4. ``NUM_COMPONENTS[0] == layout.flat_size_per_element``
    5. ``MULTIPLICITY[0] == 1``
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
            f"catalog layout for class_tag={layout.class_tag} expects "
            f"{layout.n_nodes_per_element} nodes × "
            f"{layout.n_components_per_node} comps = {expected}."
        )

    meta = bucket_grp.get("META")
    if meta is None:
        raise ValueError(
            f"MPCO bucket {bracket_key!r} is missing the META subgroup."
        )

    multiplicity = np.asarray(meta["MULTIPLICITY"][...]).flatten().astype(np.int64)
    gauss_ids = np.asarray(meta["GAUSS_IDS"][...]).flatten().astype(np.int64)
    num_components = np.asarray(meta["NUM_COMPONENTS"][...]).flatten().astype(np.int64)

    if not (multiplicity.size == gauss_ids.size == num_components.size == 1):
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: nodal-forces buckets carry "
            f"a single META block; got MULTIPLICITY={multiplicity.size}, "
            f"GAUSS_IDS={gauss_ids.size}, NUM_COMPONENTS={num_components.size}."
        )
    if int(gauss_ids[0]) != -1:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: nodal-forces META expects "
            f"GAUSS_IDS=[-1] (no integration point); got "
            f"GAUSS_IDS=[{int(gauss_ids[0])}]."
        )
    if int(num_components[0]) != expected:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META NUM_COMPONENTS="
            f"{int(num_components[0])} but catalog expects {expected}."
        )
    if int(multiplicity[0]) != 1:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META MULTIPLICITY="
            f"{int(multiplicity[0])} but nodal-forces buckets expect 1."
        )


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_nodal_bucket_slab(
    bucket_grp: "h5py.Group",
    bucket: _NodalBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray] | None:
    """Read one component's slab from one nodal-forces bucket.

    Returns
    -------
    (values, element_ids) or None
        ``values``: ``(T, E_g_sel, n_nodes)`` per the ``ElementSlab``
        schema — last axis is the per-element-node dimension.
        ``element_ids``: ``(E_g_sel,)`` element tag list.
        Returns ``None`` if no elements survive filtering, the
        component is not in the layout, or the bucket has no steps.
    """
    layout = bucket.layout
    if canonical_component not in layout.component_layout:
        return None

    validate_nodal_bucket_meta(
        bucket_grp, layout, bracket_key=bucket.bracket_key,
    )

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

    flat_size = layout.flat_size_per_element
    E_g = sel_ids.size
    T = int(np.size(t_idx))

    flat = np.empty((T, E_g, flat_size), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        flat[i, :, :] = step_arr[sel_rows, :]

    decoded = unflatten_nodal(flat, layout)
    per_node = decoded[canonical_component]   # (T, E_g, n_nodes)

    return per_node, sel_ids


# =====================================================================
# Internal helpers
# =====================================================================

def _attr_scalar(value):
    """Coerce an h5py attribute value to a scalar."""
    if hasattr(value, "size") and getattr(value, "size", 0) == 1:
        return value.item() if hasattr(value, "item") else value[0]
    if hasattr(value, "__len__") and len(value) == 1:
        return value[0]
    return value
