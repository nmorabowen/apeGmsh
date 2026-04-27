"""Line-stations element-level result decoding for MPCOReader (Phase 11b).

MPCO stores beam-column section forces under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/section.force/<bucket_key>/``
where ``<bucket_key>`` is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]``
with ``<rule> == 1000`` (``CustomIntegrationRule``). Each bucket
holds the same ``META`` / ``ID`` / ``DATA/STEP_<k>`` triplet as
gauss-level buckets, with two crucial differences:

1. ``GP_X`` — per-element natural coordinates in ``[-1, +1]`` —
   lives as an *attribute* on the connectivity dataset at
   ``MODEL/ELEMENTS/<classTag>-<ClassName>[<rule>:<cust>]`` (no
   trailing header index), not on the results bucket itself.
2. ``META/COMPONENTS`` carries per-IP section response component
   names (``P``, ``Mz``, ``My``, ``T``, ``Vy``, ``Vz``) rather than
   stress / strain tensor indices. Catalog entries declare only
   the structural identity (class tag, parent domain); the
   concrete ``ResponseLayout`` is resolved per-bucket from
   ``GP_X`` plus the parsed section codes.

This module mirrors :mod:`_mpco_element_io` for line-stations
buckets. It walks ``ON_ELEMENTS/section.force/`` for a requested
canonical component, parses each bucket's META to recover section
codes, fetches ``GP_X`` from the connectivity dataset, builds a
concrete :class:`ResponseLayout` via :func:`resolve_layout_from_gp_x`,
and emits per-element-station slab parts ready to stitch into a
``LineStationSlab``.

v1 scope (skipped silently when encountered):

- ``header_idx != 0`` — heterogeneous response shapes within one
  bucket. STKO uses ``hdrIdx`` to disambiguate distinct META
  layouts that would otherwise collide. v1 reads ``:0`` only.
- Element classes not in :data:`CUSTOM_RULE_CATALOG`.
- Sections with response codes outside
  :data:`SECTION_RESPONSE_TO_CANONICAL` (warping, asymmetric
  bending — codes 15+).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    CustomRuleLayout,
    MPCOElementKey,
    ResponseLayout,
    gauss_routing_for_canonical,
    is_custom_rule_catalogued,
    lookup_custom_rule,
    parse_mpco_element_key,
    resolve_layout_from_gp_x,
    unflatten,
)
from ._mpco_element_io import _attr_scalar, validate_bucket_meta

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → MPCO -E token translation (line-stations topology)
# =====================================================================

def canonical_to_line_station_token(
    canonical: str,
) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a line-stations component.

    Thin alias of :func:`gauss_routing_for_canonical` with
    ``topology="line_stations"``. The MPCO group name is identical
    to the OpenSees recorder keyword (``"section.force"``).

    Examples
    --------
    ``"axial_force"``       → ``("section.force", "section_force")``
    ``"bending_moment_z"``  → ``("section.force", "section_force")``
    ``"shear_y"``           → ``("section.force", "section_force")``
    ``"stress_xx"``         → ``None`` (gauss-level component, not
                                       a line-station one).
    """
    return gauss_routing_for_canonical(canonical, topology="line_stations")


# =====================================================================
# MPCO section component names → SECTION_RESPONSE_* codes
# =====================================================================
#
# OpenSees ``MPCORecorder.cpp`` writes the section's response codes
# as short tokens in META/COMPONENTS. The mapping is fixed across
# all SectionForceDeformation subclasses — a section's
# ``getType()`` Vector picks values from this set, in the section's
# own order. Codes match
# ``SRC/material/section/SectionForceDeformation.h:52–57``.

MPCO_SECTION_NAME_TO_CODE: dict[str, int] = {
    "P": 2,        # SECTION_RESPONSE_P
    "Mz": 1,       # SECTION_RESPONSE_MZ
    "My": 4,       # SECTION_RESPONSE_MY
    "T": 6,        # SECTION_RESPONSE_T
    "Vy": 3,       # SECTION_RESPONSE_VY
    "Vz": 5,       # SECTION_RESPONSE_VZ
}


# =====================================================================
# Per-bucket descriptor — what the reader walks
# =====================================================================

@dataclass(frozen=True)
class _LineBucket:
    """One ``ON_ELEMENTS/section.force/<bracket_key>`` group + its catalog entry."""
    bracket_key: str
    elem_key: MPCOElementKey
    custom: CustomRuleLayout


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_line_station_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_LineBucket]]:
    """Walk ``ON_ELEMENTS/section.force/`` and return catalogued buckets.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the ON_ELEMENTS child name we looked
        under (always ``"section.force"`` in v1), or ``None`` if the
        canonical component has no line-stations routing. ``buckets``
        is the list whose ``(class_name, "section_force")`` is in
        :data:`CUSTOM_RULE_CATALOG`, ``int_rule == Custom``, and
        ``header_idx == 0``.
    """
    mapping = canonical_to_line_station_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [])

    token_grp = on_elements_grp[mpco_group_name]
    out: list[_LineBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            continue
        if not elem_key.is_custom_rule:
            # Standard-rule bucket under section.force? Out of scope.
            continue
        if elem_key.header_idx != 0:
            # Heterogeneous-section bucket — out of scope for v1.
            continue
        if not is_custom_rule_catalogued(elem_key.class_name, catalog_token):
            continue
        custom = lookup_custom_rule(elem_key.class_name, catalog_token)
        out.append(_LineBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            custom=custom,
        ))
    return (mpco_group_name, out)


# =====================================================================
# GP_X retrieval + META section-code parsing
# =====================================================================

def _connectivity_bracket_key(elem_key: MPCOElementKey) -> str:
    """Reconstruct the MODEL/ELEMENTS bracket key (no ``:hdr``).

    Results bucket keys are 3-field
    (``<rule>:<cust>:<hdr>``); connectivity keys are 2-field
    (``<rule>:<cust>``). The connectivity dataset carries the
    ``GP_X`` attribute we need to resolve the per-element layout.
    """
    return (
        f"{elem_key.class_tag}-{elem_key.class_name}"
        f"[{elem_key.int_rule}:{elem_key.custom_rule_idx}]"
    )


def read_gp_x_from_connectivity(
    model_elements_grp: "h5py.Group",
    elem_key: MPCOElementKey,
) -> ndarray:
    """Fetch the per-bucket ``GP_X`` natural-coordinate array.

    All elements in a bucket share the same ``(rule_type, x_vector)``
    custom-rule index, so a single ``GP_X`` array covers the whole
    bucket. Values are in parent ξ ∈ ``[-1, +1]``.
    """
    conn_key = _connectivity_bracket_key(elem_key)
    if conn_key not in model_elements_grp:
        raise ValueError(
            f"MPCO connectivity dataset {conn_key!r} missing from "
            f"MODEL/ELEMENTS; cannot resolve line-stations layout for "
            f"bucket {elem_key.class_tag}-{elem_key.class_name}."
        )
    conn_ds = model_elements_grp[conn_key]
    gp_x_attr = conn_ds.attrs.get("GP_X")
    if gp_x_attr is None:
        raise ValueError(
            f"MPCO connectivity dataset {conn_key!r} is missing the "
            f"GP_X attribute. Custom-rule beam-columns must carry "
            f"per-element integration-point coordinates."
        )
    return np.asarray(gp_x_attr, dtype=np.float64).flatten()


def parse_section_codes_from_meta(
    bucket_grp: "h5py.Group",
    *,
    bracket_key: str,
) -> tuple[int, ...]:
    """Parse META/COMPONENTS to extract per-IP section response codes.

    META/COMPONENTS is a single byte-string of the form::

        "<pathInts>.<comps>;<pathInts>.<comps>;..."

    where each ``;``-separated segment describes one IP, ``<pathInts>``
    are descriptor levels (3 ints for section.force on a beam, e.g.
    ``"0.1.2"``), and ``<comps>`` is a comma-separated list of MPCO
    section component tokens (``"P,Mz,My,T,Vy,Vz"``).

    v1 requires homogeneous sections — every IP segment must produce
    the same code tuple — so we return one tuple representing the
    whole bucket.
    """
    meta = bucket_grp.get("META")
    if meta is None or "COMPONENTS" not in meta:
        raise ValueError(
            f"MPCO bucket {bracket_key!r} is missing META/COMPONENTS; "
            f"cannot decode line-stations layout."
        )
    raw = meta["COMPONENTS"][...]
    if hasattr(raw, "__len__") and not isinstance(raw, (bytes, str)):
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("ascii")

    segments = [seg for seg in raw.split(";") if seg]
    if not segments:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META/COMPONENTS is empty."
        )

    parsed = [
        _parse_one_segment(seg, bracket_key=bracket_key)
        for seg in segments
    ]
    first = parsed[0]
    for i, codes in enumerate(parsed[1:], start=1):
        if codes != first:
            raise ValueError(
                f"MPCO bucket {bracket_key!r}: heterogeneous section "
                f"codes per IP (IP 0 = {first}, IP {i} = {codes}). "
                f"v1 supports homogeneous sections only."
            )
    return first


def _parse_one_segment(segment: str, *, bracket_key: str) -> tuple[int, ...]:
    """Parse one ``<pathInts>.<compsCsv>`` segment into a code tuple.

    The component-CSV is the substring after the *last* dot. Path
    integers are dropped (caller doesn't need them in v1; they
    describe the descriptor hierarchy that produced the segment).
    """
    last_dot = segment.rfind(".")
    if last_dot < 0:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META segment {segment!r} "
            f"has no dot separator."
        )
    comps_csv = segment[last_dot + 1:]
    names = [n for n in comps_csv.split(",") if n]
    if not names:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: META segment {segment!r} "
            f"has no component names after the dot."
        )
    try:
        return tuple(MPCO_SECTION_NAME_TO_CODE[n] for n in names)
    except KeyError as e:
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: unknown section component "
            f"name {e!s} in META segment {segment!r}. Known names: "
            f"{sorted(MPCO_SECTION_NAME_TO_CODE)}."
        ) from None


# =====================================================================
# Per-bucket layout resolution + cached read
# =====================================================================

def resolve_bucket_layout(
    model_elements_grp: "h5py.Group",
    bucket_grp: "h5py.Group",
    bucket: _LineBucket,
) -> tuple[ResponseLayout, ndarray]:
    """Build the concrete ``ResponseLayout`` for one line-stations bucket.

    Returns the resolved layout and the bucket's ``GP_X`` array (kept
    separately so the caller can reuse it as ``station_natural_coord``
    without re-reading).
    """
    gp_x = read_gp_x_from_connectivity(model_elements_grp, bucket.elem_key)
    section_codes = parse_section_codes_from_meta(
        bucket_grp, bracket_key=bucket.bracket_key,
    )
    layout = resolve_layout_from_gp_x(bucket.custom, gp_x, section_codes)
    return layout, gp_x


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_line_bucket_slab(
    bucket_grp: "h5py.Group",
    model_elements_grp: "h5py.Group",
    bucket: _LineBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Read one component's slab from one line-stations bucket.

    Returns
    -------
    (values, element_index, station_natural_coord) or None
        ``values``: ``(T, E_g_sel * n_IP)`` flat per the
        ``LineStationSlab`` schema.
        ``element_index``: ``(E_g_sel * n_IP,)`` element tag per
        column (each tag repeated ``n_IP`` times).
        ``station_natural_coord``: ``(E_g_sel * n_IP,)`` parent ξ
        per column (``GP_X`` tiled across selected elements).
        Returns ``None`` if no elements survive filtering, the
        component is not in the resolved layout, or the bucket has
        no recorded steps.
    """
    layout, gp_x = resolve_bucket_layout(
        model_elements_grp, bucket_grp, bucket,
    )
    if canonical_component not in layout.component_layout:
        return None

    validate_bucket_meta(
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

    n_ip = layout.n_gauss_points
    flat_size = layout.flat_size_per_element
    E_g = sel_ids.size
    T = int(np.size(t_idx))

    flat = np.empty((T, E_g, flat_size), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(data_grp[step_keys[int(k)]][...], dtype=np.float64)
        flat[i, :, :] = step_arr[sel_rows, :]

    decoded = unflatten(flat, layout)
    per_ip = decoded[canonical_component]    # (T, E_g, n_IP)

    values = per_ip.reshape(T, E_g * n_ip)
    element_index = np.repeat(sel_ids, n_ip).astype(np.int64)
    station_natural_coord = np.tile(gp_x, E_g).astype(np.float64)

    return values, element_index, station_natural_coord


# Re-export for symmetry with _mpco_element_io.
__all__ = [
    "MPCO_SECTION_NAME_TO_CODE",
    "canonical_to_line_station_token",
    "discover_line_station_buckets",
    "parse_section_codes_from_meta",
    "read_gp_x_from_connectivity",
    "read_line_bucket_slab",
    "resolve_bucket_layout",
    "_attr_scalar",
]
