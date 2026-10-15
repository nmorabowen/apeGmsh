"""Line-stations element-level result decoding for MPCOReader (Phase 11b).

MPCO stores beam-column section forces under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/section.force/<bucket_key>/``
and section deformations under ``…/section.deformation/<bucket_key>/``,
where ``<bucket_key>`` is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]``
with ``<rule> == 1000`` (``CustomIntegrationRule``). Each bucket
holds the same ``META`` / ``ID`` / ``DATA/STEP_<k>`` triplet as
gauss-level buckets, with two crucial differences:

1. ``GP_X`` — per-element natural coordinates in ``[-1, +1]`` —
   lives as an *attribute* on the connectivity dataset at
   ``MODEL/ELEMENTS/<classTag>-<ClassName>[<rule>:<cust>]`` (no
   trailing header index), not on the results bucket itself.
2. ``META/COMPONENTS`` carries per-IP section response component
   names (``P``, ``Mz``, ``My``, ``T``, ``Vy``, ``Vz`` for forces;
   ``eps``, ``kappaZ``, ``kappaY``, ``theta``, ``gammaY``, ``gammaZ``
   for deformations) rather than stress / strain tensor indices. Catalog entries declare only
   the structural identity (class tag, parent domain); the
   concrete ``ResponseLayout`` is resolved per-bucket from
   ``GP_X`` plus the parsed section codes.

This module mirrors :mod:`_mpco_element_io` for line-stations
buckets. It walks ``ON_ELEMENTS/section.force/`` (or
``section.deformation/``) for a requested canonical component, parses
each bucket's META to recover section codes, fetches ``GP_X`` from the
connectivity dataset, builds a concrete :class:`ResponseLayout` via
:func:`resolve_layout_from_gp_x`, and emits per-element-station slab
parts ready to stitch into a ``LineStationSlab``.

Every header index is read (``:0``, ``:1``, ...: MPCO splits one
class's elements by section component list), and so is a custom-rule
class outside :data:`CUSTOM_RULE_CATALOG`. A bucket that cannot be
decoded raises when read: no ``GP_X`` (the fork's rule-0 beams), or a
section code outside :data:`SECTION_RESPONSE_TO_CANONICAL` (warping,
asymmetric bending — codes 15+, tagged ``Unknown``).

Not silent: a section-deformation canonical requested from a file with
no ``section.deformation`` group raises ``LookupError`` naming the
group (#2053); a section-force canonical may still come from
``localForce``, so its missing group reads empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...opensees._response_catalog import (
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
#
# The tokens are the ``ResponseType`` tags of
# ``SectionForceDeformation::setResponse`` (fork 79e062367,
# ``SRC/material/section/SectionForceDeformation.cpp`` L651–672 for
# ``deformation``, L704–725 for ``force``). No section class on the
# beam path overrides those tags, so every section kind shares them:
# a 2-D fiber / elastic section ``[P, Mz]`` writes ``eps,kappaZ``; a
# 3-D fiber section with torsion ``[P, Mz, My, T]`` writes
# ``eps,kappaZ,kappaY,theta``; a shear-deformable section
# (SectionAggregator with Vy/Vz, Timoshenko elastic sections) adds
# ``gammaY`` / ``gammaZ``. The two vocabularies are disjoint, and a
# bucket must use the one its ON_ELEMENTS group names.

MPCO_SECTION_FORCE_NAME_TO_CODE: dict[str, int] = {
    "P": 2,        # SECTION_RESPONSE_P
    "Mz": 1,       # SECTION_RESPONSE_MZ
    "My": 4,       # SECTION_RESPONSE_MY
    "T": 6,        # SECTION_RESPONSE_T
    "Vy": 3,       # SECTION_RESPONSE_VY
    "Vz": 5,       # SECTION_RESPONSE_VZ
}

MPCO_SECTION_DEFORMATION_NAME_TO_CODE: dict[str, int] = {
    "eps": 2,      # SECTION_RESPONSE_P  — axial strain
    "kappaZ": 1,   # SECTION_RESPONSE_MZ — curvature about z
    "kappaY": 4,   # SECTION_RESPONSE_MY — curvature about y
    "theta": 6,    # SECTION_RESPONSE_T  — twist rate
    "gammaY": 3,   # SECTION_RESPONSE_VY — shear strain y
    "gammaZ": 5,   # SECTION_RESPONSE_VZ — shear strain z
}

#: Every section component name the reader decodes, either kind.
MPCO_SECTION_NAME_TO_CODE: dict[str, int] = {
    **MPCO_SECTION_FORCE_NAME_TO_CODE,
    **MPCO_SECTION_DEFORMATION_NAME_TO_CODE,
}

#: ON_ELEMENTS group → (``resolve_layout_from_gp_x`` kind, its names).
_SECTION_KIND_BY_GROUP: dict[str, tuple[str, dict[str, int]]] = {
    "section.force": ("force", MPCO_SECTION_FORCE_NAME_TO_CODE),
    "section.deformation": (
        "deformation", MPCO_SECTION_DEFORMATION_NAME_TO_CODE),
}

#: Groups whose canonicals have no other source in an MPCO file.
#: ``section.force`` is absent from this set: ``localForce`` supplies
#: elastic beams' end stations when it is missing.
_GROUPS_WITHOUT_FALLBACK: frozenset[str] = frozenset({"section.deformation"})


# =====================================================================
# Per-bucket descriptor — what the reader walks
# =====================================================================

@dataclass(frozen=True)
class _LineBucket:
    """One ``ON_ELEMENTS/<section group>/<bracket_key>`` group + its catalog entry.

    ``kind`` is ``"force"`` under ``section.force`` and
    ``"deformation"`` under ``section.deformation``; it picks both the
    META name vocabulary and the canonical names.
    """
    bracket_key: str
    elem_key: MPCOElementKey
    custom: CustomRuleLayout
    kind: str


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_line_station_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_LineBucket]]:
    """Walk ``ON_ELEMENTS/<section group>/`` and return catalogued buckets.

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the ON_ELEMENTS child name we looked
        under (``"section.force"`` or ``"section.deformation"``), or
        ``None`` if the canonical component has no line-stations
        routing. ``buckets`` is the list whose ``(class_name,
        catalog_token)`` is in :data:`CUSTOM_RULE_CATALOG`,
        ``int_rule == Custom``, and ``header_idx == 0``.

    Raises
    ------
    LookupError
        The canonical routes to ``section.deformation`` and the file
        has no such ON_ELEMENTS group: the run did not record
        ``-E section.deformation``, and nothing else in an MPCO file
        holds section deformations. MPCO creates the group for every
        requested ``-E`` result even when no element of this partition
        answers, so its absence means "not recorded", never "no
        beams here". ``LookupError``, not ``ValueError``: callers that
        skip undecodable buckets on ``ValueError`` must not swallow a
        missing recording.
    """
    mapping = canonical_to_line_station_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in _SECTION_KIND_BY_GROUP:
        raise ValueError(
            f"line-station component {canonical_component!r} routes to "
            f"MPCO group {mpco_group_name!r}, which this reader cannot "
            f"decode (known: {sorted(_SECTION_KIND_BY_GROUP)})."
        )
    kind = _SECTION_KIND_BY_GROUP[mpco_group_name][0]
    if mpco_group_name not in on_elements_grp:
        if mpco_group_name in _GROUPS_WITHOUT_FALLBACK:
            raise LookupError(
                f"MPCO file has no ON_ELEMENTS/{mpco_group_name} bucket, "
                f"so it cannot supply line-station component "
                f"{canonical_component!r}. Record it with "
                f"'-E {mpco_group_name}' (recorded groups: "
                f"{sorted(on_elements_grp)})."
            )
        return (mpco_group_name, [])

    token_grp = on_elements_grp[mpco_group_name]
    out: list[_LineBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            continue
        catalogued = is_custom_rule_catalogued(
            elem_key.class_name, catalog_token)
        if not elem_key.is_custom_rule and not catalogued:
            # A non-beam answering "section" (a zero-length section,
            # say): its rule has no stations, so it is not a
            # line-station element.
            continue
        # Every header index is read: MPCO splits one class's elements
        # into ``:0``, ``:1``, ... when their sections answer with
        # different component lists. A class outside
        # the catalogue is read too: section components depend on the
        # section, not the element class, and an undecodable bucket
        # raises at read time (no GP_X, unknown names).
        custom = (
            lookup_custom_rule(elem_key.class_name, catalog_token)
            if catalogued else
            CustomRuleLayout(
                class_tag=elem_key.class_tag,
                coord_system="isoparametric_1d",
            )
        )
        out.append(_LineBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            custom=custom,
            kind=kind,
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
            f"GP_X attribute, so the station positions of element class "
            f"{elem_key.class_name!r} (integration rule "
            f"{elem_key.int_rule}) are unknown and its line-station "
            f"results cannot be read. Custom-rule beam-columns must "
            f"carry per-element integration-point coordinates."
        )
    return np.asarray(gp_x_attr, dtype=np.float64).flatten()


def parse_section_codes_from_meta(
    bucket_grp: "h5py.Group",
    *,
    bracket_key: str,
    kind: str = "force",
) -> tuple[int, ...]:
    """Parse META/COMPONENTS to extract per-IP section response codes.

    META/COMPONENTS is a single byte-string of the form::

        "<pathInts>.<comps>;<pathInts>.<comps>;..."

    where each ``;``-separated segment describes one IP, ``<pathInts>``
    are descriptor levels (3 ints for section.force on a beam, e.g.
    ``"0.1.2"``), and ``<comps>`` is a comma-separated list of MPCO
    section component tokens (``"P,Mz,My,T,Vy,Vz"`` under
    ``kind="force"``, ``"eps,kappaZ,kappaY,theta,gammaY,gammaZ"``
    under ``kind="deformation"``). A name from the other kind's
    vocabulary raises: it means the bucket sits under the wrong group.

    v1 requires homogeneous sections — every IP segment must produce
    the same code tuple — so we return one tuple representing the
    whole bucket.
    """
    names_to_code = _names_for_kind(kind)
    meta = bucket_grp["META"] if "META" in bucket_grp else None
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
        _parse_one_segment(
            seg, bracket_key=bracket_key, kind=kind,
            names_to_code=names_to_code,
        )
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


def _names_for_kind(kind: str) -> dict[str, int]:
    for group_kind, names in _SECTION_KIND_BY_GROUP.values():
        if group_kind == kind:
            return names
    raise ValueError(
        f"section response kind must be one of "
        f"{sorted(k for k, _ in _SECTION_KIND_BY_GROUP.values())}; "
        f"got {kind!r}."
    )


def _parse_one_segment(
    segment: str,
    *,
    bracket_key: str,
    kind: str,
    names_to_code: dict[str, int],
) -> tuple[int, ...]:
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
    unknown = [n for n in names if n not in names_to_code]
    if unknown:
        if all(n in MPCO_SECTION_NAME_TO_CODE for n in unknown):
            raise ValueError(
                f"MPCO bucket {bracket_key!r}: section {kind} bucket "
                f"carries component name(s) {unknown} of the other "
                f"section response kind in META segment {segment!r}. "
                f"Known {kind} names: {sorted(names_to_code)}."
            )
        raise ValueError(
            f"MPCO bucket {bracket_key!r}: unknown section component "
            f"name(s) {unknown} in META segment {segment!r}. Known "
            f"{kind} names: {sorted(names_to_code)}."
        )
    return tuple(names_to_code[n] for n in names)


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
        bucket_grp, bracket_key=bucket.bracket_key, kind=bucket.kind,
    )
    layout = resolve_layout_from_gp_x(
        bucket.custom, gp_x, section_codes, kind=bucket.kind,
    )
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
    "MPCO_SECTION_DEFORMATION_NAME_TO_CODE",
    "MPCO_SECTION_FORCE_NAME_TO_CODE",
    "MPCO_SECTION_NAME_TO_CODE",
    "canonical_to_line_station_token",
    "discover_line_station_buckets",
    "parse_section_codes_from_meta",
    "read_gp_x_from_connectivity",
    "read_line_bucket_slab",
    "resolve_bucket_layout",
    "_attr_scalar",
]
