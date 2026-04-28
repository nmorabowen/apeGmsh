"""Fiber-level result decoding for MPCOReader (Phase 11c).

MPCO stores beam-column fiber-section results under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/section.fiber.stress/<bucket_key>/``
(``section.fiber.strain`` for strain). Each bucket key has the same
shape as a line-stations bucket — ``<classTag>-<ClassName>[1000:<cust>:<hdr>]``
with ``<rule> == 1000`` (CustomIntegrationRule) — and holds the same
``META`` / ``ID`` / ``DATA/STEP_<k>`` triplet, with two key
differences from line-stations:

1. The flat per-element response packs ``(n_IP × n_fibers × 1)``
   floats: each fiber emits a single scalar (axial stress / strain in
   the section's local frame). Total ``NUM_COLUMNS = n_IP * n_fibers``
   for homogeneous fiber sections.

2. The fiber geometry (``y``, ``z``, ``area``, ``material_tag``) lives
   in ``MODEL/SECTION_ASSIGNMENTS/SECTION_<tag>[<ClassType>]``. To
   resolve a bucket's per-fiber metadata, this module walks the
   section-assignment registry to find which section a representative
   element / GP uses, then reads its ``FIBER_DATA`` and
   ``FIBER_MATERIALS`` arrays.

v1 scope (skipped silently when encountered):

- Heterogeneous fiber sections within one bucket (different fiber
  counts per IP or per element). v1 assumes all elements in the
  bucket share an identical-shape fiber section.
- ``header_idx != 0``.
- Sections without ``FIBER_DATA`` (non-fiber sections that somehow
  end up under ``section.fiber.stress`` — should never happen).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    FiberSectionLayout,
    MPCOElementKey,
    gauss_routing_for_canonical,
    is_fiber_catalogued,
    lookup_fiber,
    parse_mpco_element_key,
)
from ._mpco_element_io import _attr_scalar
from ._mpco_line_io import read_gp_x_from_connectivity

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → MPCO -E token translation (fibers topology)
# =====================================================================

def canonical_to_fiber_token(canonical: str) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a fiber-topology component.

    Thin alias of :func:`gauss_routing_for_canonical` with
    ``topology="fibers"``. Beam fiber-section reads route to
    ``section.fiber.*`` group names (no shell keyword swap).

    Examples
    --------
    ``"fiber_stress"`` → ``("section.fiber.stress", "fiber_stress")``
    ``"fiber_strain"`` → ``("section.fiber.strain", "fiber_strain")``
    ``"stress_xx"``    → ``None`` (continuum component, not fiber).
    """
    return gauss_routing_for_canonical(canonical, topology="fibers")


# =====================================================================
# Per-bucket descriptor
# =====================================================================

@dataclass(frozen=True)
class _FiberBucket:
    """One ``ON_ELEMENTS/section.fiber.<X>/<bracket_key>`` group + its catalog entry."""
    bracket_key: str
    elem_key: MPCOElementKey
    fiber_layout: FiberSectionLayout


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_fiber_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_FiberBucket]]:
    """Walk ``ON_ELEMENTS/section.fiber.<X>/`` and return catalogued buckets.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the ON_ELEMENTS child name we looked
        under (``"section.fiber.stress"`` / ``"section.fiber.strain"``)
        or ``None`` if the canonical has no fibers-topology routing.
        ``buckets`` keeps only entries with ``int_rule == Custom``,
        ``header_idx == 0``, and a ``(class_name, token)`` in
        :data:`apeGmsh.solvers._element_response.FIBER_CATALOG`.
    """
    mapping = canonical_to_fiber_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [])

    token_grp = on_elements_grp[mpco_group_name]
    out: list[_FiberBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            continue
        if not elem_key.is_custom_rule:
            # Beam fiber-section buckets always use the custom rule.
            continue
        if elem_key.header_idx != 0:
            continue
        if not is_fiber_catalogued(elem_key.class_name, catalog_token):
            continue
        fiber_layout = lookup_fiber(elem_key.class_name, catalog_token)
        out.append(_FiberBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            fiber_layout=fiber_layout,
        ))
    return (mpco_group_name, out)


# =====================================================================
# SECTION_ASSIGNMENTS lookup → fiber geometry / materials
# =====================================================================

# MPCO section-assignment group names: ``SECTION_<tag>[<ClassType>]``.
_SECTION_KEY_RE = re.compile(r"^SECTION_(?P<tag>-?\d+)\[(?P<class>[^\]]+)\]\s*$")


@dataclass(frozen=True)
class FiberSectionData:
    """Fiber geometry + materials for one MPCO section assignment.

    All arrays are 0-based and aligned: ``fiber_y[k]`` /
    ``fiber_z[k]`` / ``fiber_area[k]`` / ``fiber_material_tag[k]``
    refer to the same fiber ``k`` in the section's local frame.
    """
    section_tag: int
    section_class: str
    fiber_y: ndarray              # (n_fibers,) float64
    fiber_z: ndarray              # (n_fibers,) float64
    fiber_area: ndarray           # (n_fibers,) float64
    fiber_material_tag: ndarray   # (n_fibers,) int64

    @property
    def n_fibers(self) -> int:
        return int(self.fiber_y.size)


def _hydrate_fiber_section(
    sec_grp: "h5py.Group", *, section_tag: int, section_class: str,
) -> FiberSectionData:
    """Read FIBER_DATA / FIBER_MATERIALS off one section group."""
    if "FIBER_DATA" not in sec_grp:
        raise ValueError(
            f"MPCO section ``SECTION_{section_tag}[{section_class}]``: "
            f"FIBER_DATA missing — non-fiber section?"
        )
    fdata = np.asarray(sec_grp["FIBER_DATA"][...], dtype=np.float64)
    if fdata.ndim != 2 or fdata.shape[1] < 3:
        raise ValueError(
            f"MPCO section ``SECTION_{section_tag}[{section_class}]``: "
            f"FIBER_DATA shape {fdata.shape} is not (n_fibers, 3)."
        )
    fmat = (
        np.asarray(sec_grp["FIBER_MATERIALS"][...], dtype=np.int64).flatten()
        if "FIBER_MATERIALS" in sec_grp
        else np.zeros(fdata.shape[0], dtype=np.int64)
    )
    return FiberSectionData(
        section_tag=section_tag, section_class=section_class,
        fiber_y=fdata[:, 0].copy(),
        fiber_z=fdata[:, 1].copy(),
        fiber_area=fdata[:, 2].copy(),
        fiber_material_tag=fmat,
    )


def find_fiber_section_for_element(
    section_assignments_grp: "h5py.Group",
    element_id: int,
    *,
    gp_idx: int = 0,
) -> FiberSectionData:
    """Walk ``SECTION_ASSIGNMENTS`` for the section assigned to one element / GP.

    Each ``SECTION_<tag>[<ClassType>]`` group carries an ``ASSIGNMENT``
    dataset of shape ``(nPairs, 2)`` whose rows are
    ``(elem_tag, 0-based gp_idx)``. The first matching group is
    returned with its fiber data hydrated.

    For bulk reads use :func:`build_fiber_section_index_for_bucket`,
    which scans every ``SECTION_*[*]`` group once and returns an
    ``(elem_id, gp_idx) → FiberSectionData`` lookup. This single-pass
    helper is kept for direct callers who only need a single match.

    Raises
    ------
    ValueError
        If no section group matches, or the matching group is missing
        ``FIBER_DATA``.
    """
    for sec_name in section_assignments_grp:
        m = _SECTION_KEY_RE.match(sec_name)
        if m is None:
            continue
        sec_grp = section_assignments_grp[sec_name]
        if "ASSIGNMENT" not in sec_grp:
            continue
        assign = np.asarray(
            sec_grp["ASSIGNMENT"][...],
        ).reshape(-1, 2).astype(np.int64)
        match = (assign[:, 0] == int(element_id)) & (assign[:, 1] == int(gp_idx))
        if not np.any(match):
            continue
        return _hydrate_fiber_section(
            sec_grp,
            section_tag=int(m.group("tag")),
            section_class=str(m.group("class")),
        )
    raise ValueError(
        f"No SECTION_ASSIGNMENTS entry matches element {element_id}, "
        f"GP {gp_idx} — cannot resolve fiber geometry."
    )


def build_fiber_section_index_for_bucket(
    section_assignments_grp: "h5py.Group",
    *,
    element_ids: ndarray,
    n_ip: int,
) -> dict[tuple[int, int], FiberSectionData]:
    """Build an ``(elem_id, gp_idx) → FiberSectionData`` lookup for one bucket.

    Single pass over ``SECTION_ASSIGNMENTS``: each matching pair of
    ``(elem_id, gp_idx)`` from the bucket's element list × ``range(n_ip)``
    gets bound to the section that owns it. Sections referenced more
    than once share a single hydrated :class:`FiberSectionData` instance.

    Used by :func:`read_fiber_bucket_slab` to support buckets where
    different elements use different fiber sections (a common case
    in real STKO output — multiple beam-column sections in one
    structural model produce one bucket per element class but
    several distinct ``SECTION_<tag>`` entries).
    """
    needed = {
        (int(eid), int(g)) for eid in element_ids for g in range(n_ip)
    }
    out: dict[tuple[int, int], FiberSectionData] = {}
    for sec_name in section_assignments_grp:
        m = _SECTION_KEY_RE.match(sec_name)
        if m is None:
            continue
        sec_grp = section_assignments_grp[sec_name]
        if "ASSIGNMENT" not in sec_grp:
            continue
        assign = np.asarray(
            sec_grp["ASSIGNMENT"][...],
        ).reshape(-1, 2).astype(np.int64)
        # Filter to pairs we care about; hydrate the section once.
        sec_pairs = [
            (int(a), int(b)) for a, b in assign
            if (int(a), int(b)) in needed
        ]
        if not sec_pairs:
            continue
        section_data = _hydrate_fiber_section(
            sec_grp,
            section_tag=int(m.group("tag")),
            section_class=str(m.group("class")),
        )
        for pair in sec_pairs:
            if pair not in out:
                out[pair] = section_data
    missing = needed - out.keys()
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{len(missing)} (element, GP) pairs in the bucket have no "
            f"matching section assignment — first few: {sample}. "
            f"This likely indicates an MPCO file written with a "
            f"different element/section topology than the bucket header."
        )
    return out


# =====================================================================
# Bucket layout resolution
# =====================================================================

@dataclass(frozen=True)
class FiberBucketLayout:
    """Resolved per-bucket layout: IPs × fibers, with per-element sections.

    ``gp_x``: (n_IP,) natural coords in [-1, +1] from connectivity.
    ``section_by_pair``: ``(elem_id, gp_idx) → FiberSectionData`` —
    each (element, IP) maps to the section instance assigned to it
    by ``MODEL/SECTION_ASSIGNMENTS``. Different sections sharing the
    same ``n_fibers`` may coexist in one bucket; the bucket-level
    response shape is still uniform (``NUM_COLUMNS = n_IP × n_fibers``)
    but per-row ``y / z / area / material_tag`` may differ across
    elements.
    ``n_fibers``: enforced uniform across all sections referenced —
    a heterogeneous bucket (different fiber counts) is rejected at
    layout-resolve time.
    """
    gp_x: ndarray                                      # (n_IP,) float64
    section_by_pair: dict[tuple[int, int], FiberSectionData]
    n_fibers: int

    @property
    def n_ip(self) -> int:
        return int(self.gp_x.size)


def resolve_fiber_bucket_layout(
    model_elements_grp: "h5py.Group",
    section_assignments_grp: "h5py.Group",
    bucket_grp: "h5py.Group",
    bucket: _FiberBucket,
) -> FiberBucketLayout:
    """Resolve ``GP_X`` + per-element section geometry for a fiber bucket.

    Builds the (element, IP) → section index in one pass and validates
    that every referenced section has the same ``n_fibers`` (the
    invariant that makes the bucket's flat shape well-defined).
    """
    gp_x = read_gp_x_from_connectivity(model_elements_grp, bucket.elem_key)
    if "ID" not in bucket_grp:
        raise ValueError(
            f"MPCO fiber bucket {bucket.bracket_key!r}: missing ID dataset."
        )
    ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)
    if ids.size == 0:
        raise ValueError(
            f"MPCO fiber bucket {bucket.bracket_key!r}: empty ID array."
        )
    n_ip = int(gp_x.size)
    section_by_pair = build_fiber_section_index_for_bucket(
        section_assignments_grp, element_ids=ids, n_ip=n_ip,
    )
    fiber_counts = {sec.n_fibers for sec in section_by_pair.values()}
    if len(fiber_counts) != 1:
        raise ValueError(
            f"MPCO fiber bucket {bucket.bracket_key!r}: sections "
            f"reference different fiber counts {sorted(fiber_counts)}. "
            f"Heterogeneous fiber sections within a bucket are not "
            f"supported in v1."
        )
    return FiberBucketLayout(
        gp_x=gp_x,
        section_by_pair=section_by_pair,
        n_fibers=fiber_counts.pop(),
    )


# =====================================================================
# Bucket validation
# =====================================================================

def validate_fiber_bucket_meta(
    bucket_grp: "h5py.Group",
    layout: FiberBucketLayout,
    *,
    bracket_key: str,
) -> None:
    """Cross-check NUM_COLUMNS against the resolved IP × fiber count.

    Lighter validation than the gauss path's :func:`validate_bucket_meta`
    — META blocks for fiber sections compress over the
    fiber/material levels and are not block-by-block checkable
    against a fixed catalog (the catalog declares only the
    structural identity of the response).
    """
    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        raise ValueError(
            f"MPCO fiber bucket {bracket_key!r}: missing NUM_COLUMNS attr."
        )
    num_columns = int(_attr_scalar(num_columns_attr))
    expected = layout.n_ip * layout.n_fibers
    if num_columns != expected:
        raise ValueError(
            f"MPCO fiber bucket {bracket_key!r}: NUM_COLUMNS={num_columns} "
            f"!= n_IP * n_fibers = {layout.n_ip} * {layout.n_fibers} "
            f"= {expected}. Heterogeneous fiber sections within a bucket "
            f"are not supported in v1."
        )


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_fiber_bucket_slab(
    bucket_grp: "h5py.Group",
    model_elements_grp: "h5py.Group",
    section_assignments_grp: "h5py.Group",
    bucket: _FiberBucket,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
    gp_indices: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray,
           ndarray, ndarray, ndarray, ndarray] | None:
    """Read one fiber-component slab from one bucket.

    Returns
    -------
    (values, element_index, gp_index, y, z, area, material_tag) | None
        ``values``: ``(T, sum_F)`` flat — F sweeps element-slow, GP-mid,
        fiber-fastest matching the FiberSlab schema.
        Location vectors all of shape ``(sum_F,)``.
        Returns ``None`` if no elements survive filtering or the
        bucket has no recorded steps.
    """
    layout = resolve_fiber_bucket_layout(
        model_elements_grp, section_assignments_grp, bucket_grp, bucket,
    )
    validate_fiber_bucket_meta(
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

    # GP filter (subset of n_IP).
    n_ip = layout.n_ip
    if gp_indices is None:
        gp_sel = np.arange(n_ip, dtype=np.int64)
    else:
        gp_sel = np.asarray(gp_indices, dtype=np.int64)
        if gp_sel.size == 0:
            return None
        if int(gp_sel.max()) >= n_ip or int(gp_sel.min()) < 0:
            raise ValueError(
                f"gp_indices {gp_sel.tolist()} out of range [0, {n_ip})."
            )

    data_grp = bucket_grp["DATA"]
    step_keys = sorted(
        (k for k in data_grp.keys() if k.startswith("STEP_")),
        key=lambda s: int(s.split("_", 1)[1]),
    )
    if not step_keys:
        return None

    n_fibers = layout.n_fibers
    flat_size = n_ip * n_fibers
    E_g = sel_ids.size
    T = int(np.size(t_idx))

    flat = np.empty((T, E_g, flat_size), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        flat[i, :, :] = step_arr[sel_rows, :]

    # Reshape (T, E, n_IP * n_fibers) → (T, E, n_IP, n_fibers).
    per_fiber = flat.reshape(T, E_g, n_ip, n_fibers)

    # Apply GP filter on axis 2.
    if gp_indices is not None:
        per_fiber = per_fiber[:, :, gp_sel, :]
    n_gp_sel = gp_sel.size

    # Pack into FiberSlab axes: outermost element, then GP, then fiber.
    values = np.ascontiguousarray(
        per_fiber.reshape(T, E_g * n_gp_sel * n_fibers),
    )

    # Per-row location vectors. Element / GP indices follow the same
    # outer-element / GP-mid / fiber-fast tiling as the values; y / z
    # / area / material_tag are looked up per (element, GP) from the
    # bucket's section index (different elements may use different
    # fiber sections — same n_fibers, different geometry).
    element_index = np.repeat(sel_ids, n_gp_sel * n_fibers).astype(np.int64)
    gp_index = np.tile(np.repeat(gp_sel, n_fibers), E_g).astype(np.int64)

    sum_f = E_g * n_gp_sel * n_fibers
    y = np.empty(sum_f, dtype=np.float64)
    z = np.empty(sum_f, dtype=np.float64)
    area = np.empty(sum_f, dtype=np.float64)
    material_tag = np.empty(sum_f, dtype=np.int64)
    cursor = 0
    for eid in sel_ids:
        for gp in gp_sel:
            sec = layout.section_by_pair[(int(eid), int(gp))]
            stop = cursor + n_fibers
            y[cursor:stop] = sec.fiber_y
            z[cursor:stop] = sec.fiber_z
            area[cursor:stop] = sec.fiber_area
            material_tag[cursor:stop] = sec.fiber_material_tag
            cursor = stop

    return values, element_index, gp_index, y, z, area, material_tag


__all__ = [
    "FiberBucketLayout",
    "FiberSectionData",
    "canonical_to_fiber_token",
    "discover_fiber_buckets",
    "find_fiber_section_for_element",
    "read_fiber_bucket_slab",
    "resolve_fiber_bucket_layout",
    "validate_fiber_bucket_meta",
]
