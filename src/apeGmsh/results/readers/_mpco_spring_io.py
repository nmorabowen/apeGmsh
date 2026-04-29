"""Zero-length spring result decoding for MPCOReader (Phase 11d).

MPCO stores ZeroLength / ZeroLengthSection / ZeroLengthND per-spring
results under ``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/basicForce/`` and
``…/deformation/`` bucket groups.  Each bucket key is::

    <classTag>-<ClassName>[<intRule>:<cust>:<hdr>]

where ``<intRule>`` is ``1`` (``Line_GL_1``) for all ZeroLength
family members.  The bucket holds the standard
``META`` / ``ID`` / ``DATA/STEP_<k>`` triplet.

Note that ``basicForce`` (not the plain ``force`` token) carries
per-spring forces.  The ``force`` group instead writes the element's
global resisting force vector — ``2*ndf`` columns
(``P1_x, P1_y, P1_z, P2_x, P2_y, P2_z`` in 3D), one row per element,
with all spring counts merged into a single bucket — which is not
the per-spring layout this reader expects.

Spring-specific conventions
---------------------------
- **Variable component count**: a ZeroLength element configured with
  N springs contributes a row of N scalars per step.  MPCO groups
  elements by the number of springs into separate ``header_idx``
  buckets; within one bucket all elements have the *same* N.
- **META layout**: single block (no per-GP segmentation),
  ``NUM_COMPONENTS = [[N]]``.  ``GAUSS_IDS`` is written as ``[[-1]]``
  for ``basicForce`` / ``deformation`` (the recorder treats these
  responses as "no Gauss point" — the element itself is a point).
  The reader does not validate ``GAUSS_IDS`` since it carries no
  per-spring information.
- **Canonical naming**: ``spring_force_<n>`` for the n-th spring
  force (0-based), ``spring_deformation_<n>`` for the deformation.
  For a 1-spring element both ``spring_force_0`` and the bare root
  ``spring_force`` are valid; the multi-component case mirrors the
  layered-shell ``fiber_stress_<n>`` convention.

Discovery + reading mirror :mod:`_mpco_material_io`: we look up the
ZeroLength class in :data:`ZEROLENGTH_CATALOG`, parse META, extract
the requested spring index's column, and return a per-element value
array ready to be packed into a :class:`SpringSlab`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    ZEROLENGTH_CATALOG,
    ZeroLengthLayout,
    MPCOElementKey,
    gauss_routing_for_canonical,
    is_zerolength_catalogued,
    parse_mpco_element_key,
)
from ._mpco_element_io import _attr_scalar

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → parent token + spring index
# =====================================================================

def _spring_root_and_index(canonical: str) -> tuple[str, int] | None:
    """Split a spring canonical into ``(root, spring_index)``.

    The bare root ``"spring_force"`` (no ``_<n>`` suffix) maps to
    spring index 0 — i.e. it pulls column 0 of *every* matched bucket,
    not just buckets where ``n_springs == 1``.  In a model that mixes
    1-spring and N-spring ZeroLength elements, ``spring_force`` will
    therefore return rows for both groups: the (only) spring of the
    1-spring elements, plus spring 0 of the N-spring elements.  Use
    ``spring_force_<n>`` (or ``ids=`` filtering) when you need to
    target a specific subset.

    Examples
    --------
    ``"spring_force_0"``       → ``("spring_force", 0)``
    ``"spring_force_3"``       → ``("spring_force", 3)``
    ``"spring_deformation_1"`` → ``("spring_deformation", 1)``
    ``"spring_force"``         → ``("spring_force", 0)``  (bare root → index 0)
    ``"stress_xx"``            → ``None``
    """
    for root in ("spring_force", "spring_deformation"):
        if canonical == root:
            return root, 0
        if canonical.startswith(root + "_"):
            suffix = canonical[len(root) + 1:]
            if suffix.isdigit():
                return root, int(suffix)
    return None


def canonical_to_spring_token(
    canonical: str,
) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a spring canonical.

    Thin wrapper of :func:`gauss_routing_for_canonical` with
    ``topology="springs"``, applied to the *root* canonical
    (stripping any ``_<n>`` suffix first).

    Examples
    --------
    ``"spring_force_0"``  → ``("force", "spring_force")``
    ``"spring_force"``    → ``("force", "spring_force")``
    ``"stress_xx"``       → ``None``
    """
    parsed = _spring_root_and_index(canonical)
    if parsed is None:
        return None
    root, _ = parsed
    return gauss_routing_for_canonical(root, topology="springs")


# =====================================================================
# Per-bucket descriptor
# =====================================================================

@dataclass(frozen=True)
class _SpringBucket:
    """One ``ON_ELEMENTS/<token>/<bracket_key>`` group + catalog layout."""
    bracket_key: str
    elem_key: MPCOElementKey
    catalog_layout: ZeroLengthLayout
    mpco_group_name: str     # "force" or "deformation"
    catalog_token: str       # "spring_force" or "spring_deformation"


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_spring_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_SpringBucket]]:
    """Walk ``ON_ELEMENTS/<force|deformation>`` for ZeroLength buckets.

    Parameters
    ----------
    on_elements_grp
        HDF5 group at ``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS``.
    canonical_component
        e.g. ``"spring_force_0"`` or ``"spring_force"``.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is ``"basicForce"`` or ``"deformation"``
        (the ON_ELEMENTS child that holds the data), or ``None`` if
        the canonical has no spring routing. ``buckets`` contains one
        entry per catalogued ZeroLength bucket in that group.
    """
    mapping = canonical_to_spring_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping

    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [])

    token_grp = on_elements_grp[mpco_group_name]
    out: list[_SpringBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            continue
        if elem_key.custom_rule_idx != 0:
            continue
        # header_idx may differ (different spring counts) — we accept all.
        if not is_zerolength_catalogued(elem_key.class_name, catalog_token):
            continue
        from ...solvers._element_response import lookup_zerolength  # noqa: PLC0415
        layout = lookup_zerolength(elem_key.class_name, catalog_token)
        out.append(_SpringBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            catalog_layout=layout,
            mpco_group_name=mpco_group_name,
            catalog_token=catalog_token,
        ))
    return (mpco_group_name, out)


# =====================================================================
# META resolution — n_springs per bucket
# =====================================================================

def resolve_n_springs(
    bucket_grp: "h5py.Group",
    bucket: _SpringBucket,
) -> int:
    """Read META to determine the number of springs in this bucket.

    Validates that:
    - There is exactly one META block (single ``NUM_COMPONENTS`` entry).
    - ``NUM_COLUMNS == n_springs`` (consistent with single-block META).

    Returns the number of springs (columns per element per step).
    """
    meta = bucket_grp.get("META")
    if meta is None:
        raise ValueError(
            f"MPCO spring bucket {bucket.bracket_key!r}: missing META subgroup."
        )

    num_components = np.asarray(meta["NUM_COMPONENTS"][...]).flatten().astype(np.int64)
    if num_components.size != 1:
        raise ValueError(
            f"MPCO spring bucket {bucket.bracket_key!r}: expected a single "
            f"META block (n_springs is fixed per bucket); got "
            f"{num_components.size} blocks."
        )
    n_springs = int(num_components[0])

    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is not None:
        nc = int(_attr_scalar(num_columns_attr))
        if nc != n_springs:
            raise ValueError(
                f"MPCO spring bucket {bucket.bracket_key!r}: "
                f"NUM_COLUMNS={nc} but META reports n_springs={n_springs}."
            )
    return n_springs


def spring_canonicals_in_bucket(
    bucket_grp: "h5py.Group",
    bucket: _SpringBucket,
) -> tuple[str, ...]:
    """Return the spring canonicals present in this bucket.

    For n_springs == 1 the bare root (``spring_force``) is returned;
    for n_springs > 1 indexed names (``spring_force_0``, …,
    ``spring_force_{n-1}``) are returned.  Mirrors the convention used
    by :func:`_mpco_layer_io.parse_layer_meta_components`.
    """
    n = resolve_n_springs(bucket_grp, bucket)
    root = bucket.catalog_token   # "spring_force" or "spring_deformation"
    if n == 1:
        return (root,)
    return tuple(f"{root}_{i}" for i in range(n))


# =====================================================================
# Slab read — one spring index, all elements in one bucket
# =====================================================================

def read_spring_bucket_slab(
    bucket_grp: "h5py.Group",
    bucket: _SpringBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray] | None:
    """Read one spring component from one bucket.

    Returns
    -------
    (values, element_index)
        ``values`` is ``(T, E_g)`` — one column per surviving element
        at each requested timestep. ``element_index`` is ``(E_g,)``
        element IDs. Returns ``None`` when the bucket has no steps,
        no elements survive the filter, or the spring index is out of
        range for this bucket.
    """
    parsed = _spring_root_and_index(canonical_component)
    if parsed is None:
        return None
    _, spring_idx = parsed

    n_springs = resolve_n_springs(bucket_grp, bucket)
    if spring_idx >= n_springs:
        # This bucket has fewer springs than requested — skip.
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

    data_grp = bucket_grp["DATA"]
    step_keys = sorted(
        (k for k in data_grp.keys() if k.startswith("STEP_")),
        key=lambda s: int(s.split("_", 1)[1]),
    )
    if not step_keys:
        return None

    E_g = sel_ids.size
    T = int(np.size(t_idx))

    # DATA shape: (n_elements, n_springs) per step.
    # Column spring_idx is the requested spring.
    values = np.empty((T, E_g), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        values[i, :] = step_arr[sel_rows, spring_idx]

    return values, sel_ids


__all__ = [
    "canonical_to_spring_token",
    "discover_spring_buckets",
    "read_spring_bucket_slab",
    "resolve_n_springs",
    "spring_canonicals_in_bucket",
]
