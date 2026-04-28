"""Layered-shell result decoding for MPCOReader (Phase 11c).

MPCO stores layered-shell fiber-section results under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/material.fiber.stress/<bucket_key>/``
(``material.fiber.strain`` for strain). The on-disk keyword is
``material.fiber.*``, not ``section.fiber.*``: MPCO swaps the user's
recorder token for shell elements (``utils::shell::isShellElementTag``
in MPCORecorder.cpp), and our routing tables encode that swap so the
fibers / layers topologies route to the correct group name without
the read site needing to know about it.

Each bucket key is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]``
with ``<rule>`` equal to the *surface* integration rule (e.g. 201 for
Quad_GL_2) — layered shells reuse the host shell's surface GP rule —
and holds the same ``META`` / ``ID`` / ``DATA/STEP_<k>`` triplet as
fiber/line buckets, with these specifics:

1. The flat per-element response packs ``(n_sgp × n_layers × n_sub_gp × 1)``
   floats: a single scalar (axial-in-layer stress / strain) per
   ``(surface_GP, layer, sub_GP)``. v1 supports ``n_sub_gp == 1``
   only — the standard ``LayeredShellFiberSection`` integration.
2. Per-layer thickness + material tag come from the assigned section
   in ``MODEL/SECTION_ASSIGNMENTS``. The section's ``FIBER_DATA``
   stores ``(0, 0, thickness)`` rows for layered shells (fiber y/z
   are reused for through-thickness location, but apeGmsh's
   ``LayerSlab`` exposes only thickness — y/z are not user-facing
   for layers).
3. Per-element local-axes quaternions come from
   ``MODEL/LOCAL_AXES/<connectivity_key>/QUATERNIONS``. They are
   broadcast to per-row in the ``LayerSlab.local_axes_quaternion``
   field for caller convenience (small arrays — layers buckets are
   surface elements only).

v1 scope (skipped silently when encountered):

- ``n_sub_gp != 1`` (multi-IP-per-layer integration extensions).
- ``header_idx != 0``.
- Heterogeneous layered sections within one bucket (different layer
  counts per element).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...solvers._element_response import (
    LayeredShellLayout,
    MPCOElementKey,
    gauss_routing_for_canonical,
    is_layer_catalogued,
    lookup_layer,
    mpco_layer_group_aliases,
    parse_mpco_element_key,
)
from ._mpco_element_io import _attr_scalar
from ._mpco_fiber_io import _SECTION_KEY_RE

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Canonical → MPCO -E token translation (layers topology)
# =====================================================================

def canonical_to_layer_token(canonical: str) -> tuple[str, str] | None:
    """Return ``(mpco_group_name, catalog_token)`` for a layers-topology component.

    Bare canonicals (``fiber_stress`` / ``fiber_strain``) route via
    :func:`gauss_routing_for_canonical` with ``topology="layers"``.
    Index-suffixed canonicals (``fiber_stress_0``, …) recover the
    parent name first — the bucket-level routing is parent-driven;
    only at slab read time does the index pick a column.

    Examples
    --------
    ``"fiber_stress"``    → ``("material.fiber.stress", "fiber_stress")``
    ``"fiber_strain"``    → ``("material.fiber.strain", "fiber_strain")``
    ``"fiber_stress_3"``  → ``("material.fiber.stress", "fiber_stress")``
    """
    direct = gauss_routing_for_canonical(canonical, topology="layers")
    if direct is not None:
        return direct
    # Strip an index suffix and retry against the parent.
    for parent in ("fiber_stress", "fiber_strain"):
        if canonical.startswith(parent + "_"):
            suffix = canonical[len(parent) + 1:]
            if suffix.isdigit():
                return gauss_routing_for_canonical(parent, topology="layers")
    return None


# =====================================================================
# Per-bucket descriptor
# =====================================================================

@dataclass(frozen=True)
class _LayerBucket:
    """One ``ON_ELEMENTS/<token>/<bracket_key>`` layered-shell bucket.

    ``mpco_group_name`` records the alias under which the bucket was
    found (``material.fiber.stress`` vs. ``section.fiber.stress``).
    The read site uses it to fetch the bucket dataset because the
    keyword swap doesn't always fire.
    """
    bracket_key: str
    elem_key: MPCOElementKey
    layer_layout: LayeredShellLayout
    mpco_group_name: str = ""


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_layer_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_LayerBucket]]:
    """Walk MPCO group(s) for the layers topology and return catalogued buckets.

    The MPCO recorder may write layered-shell results under either
    ``material.fiber.<X>`` (the keyword-swapped name from
    ``utils::shell::isShellElementTag``) or ``section.fiber.<X>``
    (when the swap doesn't fire — some shell class tags are excluded
    from the swap list, or the user explicitly wrote the unswapped
    token in the recorder command). Discovery walks both group names
    via :func:`mpco_layer_group_aliases` and accumulates buckets
    whose class is in ``LAYER_CATALOG``. Beam fiber-section buckets
    that happen to share the ``section.fiber.<X>`` group are filtered
    out by the catalog check (they're in ``FIBER_CATALOG``, not
    ``LAYER_CATALOG``).

    Returns
    -------
    (mpco_group_name, buckets)
        ``mpco_group_name`` is the alias we found the first matching
        bucket under (or the primary keyword if none matched).
        Returns ``(None, [])`` if the canonical has no layers-topology
        routing at all.
    """
    mapping = canonical_to_layer_token(canonical_component)
    if mapping is None:
        return (None, [])
    mpco_group_name, catalog_token = mapping
    candidate_names = mpco_layer_group_aliases(mpco_group_name)

    out: list[_LayerBucket] = []
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
            if elem_key.header_idx != 0:
                continue
            if not is_layer_catalogued(elem_key.class_name, catalog_token):
                continue
            layer_layout = lookup_layer(elem_key.class_name, catalog_token)
            if elem_key.int_rule != layer_layout.surface_int_rule:
                continue
            out.append(_LayerBucket(
                bracket_key=bracket_key,
                elem_key=elem_key,
                layer_layout=layer_layout,
                mpco_group_name=name,
            ))
            seen_keys.add(bracket_key)
            if found_name is None:
                found_name = name
    return (found_name or mpco_group_name, out)


# =====================================================================
# SECTION_ASSIGNMENTS lookup → layer thickness / materials
# =====================================================================

@dataclass(frozen=True)
class LayerSectionData:
    """Per-layer thickness + material for a layered shell section.

    Layer arrays are 0-based and aligned: ``layer_thickness[k]`` /
    ``layer_material_tag[k]`` refer to the same layer ``k`` (counted
    from the section's bottom face outward).
    """
    section_tag: int
    section_class: str
    layer_thickness: ndarray         # (n_layers,) float64
    layer_material_tag: ndarray      # (n_layers,) int64

    @property
    def n_layers(self) -> int:
        return int(self.layer_thickness.size)


def _hydrate_layer_section(
    sec_grp: "h5py.Group", *, section_tag: int, section_class: str,
) -> LayerSectionData:
    """Read FIBER_DATA / FIBER_MATERIALS off one layered-section group."""
    if "FIBER_DATA" not in sec_grp:
        raise ValueError(
            f"MPCO section ``SECTION_{section_tag}[{section_class}]``: "
            f"FIBER_DATA missing."
        )
    fdata = np.asarray(sec_grp["FIBER_DATA"][...], dtype=np.float64)
    if fdata.ndim != 2 or fdata.shape[1] < 3:
        raise ValueError(
            f"MPCO section ``SECTION_{section_tag}[{section_class}]``: "
            f"FIBER_DATA shape {fdata.shape} is not (n_layers, 3)."
        )
    fmat = (
        np.asarray(
            sec_grp["FIBER_MATERIALS"][...], dtype=np.int64,
        ).flatten()
        if "FIBER_MATERIALS" in sec_grp
        else np.zeros(fdata.shape[0], dtype=np.int64)
    )
    return LayerSectionData(
        section_tag=section_tag, section_class=section_class,
        layer_thickness=fdata[:, 2].copy(),
        layer_material_tag=fmat,
    )


def find_layered_section_for_element(
    section_assignments_grp: "h5py.Group",
    element_id: int,
    *,
    surface_gp_idx: int = 0,
) -> LayerSectionData:
    """Walk ``SECTION_ASSIGNMENTS`` for the layered section assigned to one element.

    ASSIGNMENT rows for layered shells are ``(elem_tag, surface_gp)``
    — the second column is the *surface* GP index, not a
    through-thickness IP. Layer thickness lives in
    ``FIBER_DATA[:, 2]`` (columns 0 / 1 are 0 for axial-only
    layered shells).

    For bulk reads use :func:`build_layer_section_index_for_bucket`.
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
        match = (
            (assign[:, 0] == int(element_id))
            & (assign[:, 1] == int(surface_gp_idx))
        )
        if not np.any(match):
            continue
        return _hydrate_layer_section(
            sec_grp,
            section_tag=int(m.group("tag")),
            section_class=str(m.group("class")),
        )
    raise ValueError(
        f"No SECTION_ASSIGNMENTS entry matches element {element_id}, "
        f"surface GP {surface_gp_idx} — cannot resolve layered section."
    )


def build_layer_section_index_for_bucket(
    section_assignments_grp: "h5py.Group",
    *,
    element_ids: ndarray,
    n_surface_gp: int,
) -> dict[tuple[int, int], LayerSectionData]:
    """Build an ``(elem_id, surface_gp_idx) → LayerSectionData`` lookup.

    Single pass over ``SECTION_ASSIGNMENTS`` analogous to
    :func:`apeGmsh.results.readers._mpco_fiber_io.build_fiber_section_index_for_bucket`.
    Supports buckets where different shell elements use different
    layered sections (matching n_layers).
    """
    needed = {
        (int(eid), int(g))
        for eid in element_ids
        for g in range(n_surface_gp)
    }
    out: dict[tuple[int, int], LayerSectionData] = {}
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
        sec_pairs = [
            (int(a), int(b)) for a, b in assign
            if (int(a), int(b)) in needed
        ]
        if not sec_pairs:
            continue
        section_data = _hydrate_layer_section(
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
            f"{len(missing)} (element, surface GP) pairs in the bucket "
            f"have no matching section assignment — first few: {sample}."
        )
    return out


# =====================================================================
# LOCAL_AXES lookup → per-element quaternions
# =====================================================================

# Identity quaternion (w, x, y, z) — used as a fallback when LOCAL_AXES
# is absent. The convention matches OpenSees / ASDEA's quaternion
# layout: scalar-first.
_IDENTITY_QUATERNION: ndarray = np.array(
    [1.0, 0.0, 0.0, 0.0], dtype=np.float64,
)


def _connectivity_bracket_key(elem_key: MPCOElementKey) -> str:
    """Reconstruct the MODEL/{ELEMENTS,LOCAL_AXES} bracket key (no ``:hdr``)."""
    return (
        f"{elem_key.class_tag}-{elem_key.class_name}"
        f"[{elem_key.int_rule}:{elem_key.custom_rule_idx}]"
    )


def read_local_axes_quaternions(
    local_axes_grp: "h5py.Group | None",
    elem_key: MPCOElementKey,
    element_ids: ndarray,
) -> ndarray:
    """Look up per-element quaternions, falling back to identity.

    Returns
    -------
    quaternions : (n, 4) float64
        One quaternion per requested element, scalar-first
        ``(w, x, y, z)``. Identity ``(1, 0, 0, 0)`` is returned for
        any element without a recorded quaternion (or when
        ``local_axes_grp`` is ``None`` — older MPCO writers).
    """
    n = element_ids.size
    out = np.tile(_IDENTITY_QUATERNION, (n, 1))
    if local_axes_grp is None:
        return out
    key = _connectivity_bracket_key(elem_key)
    if key not in local_axes_grp:
        return out
    grp = local_axes_grp[key]
    if "QUATERNIONS" not in grp or "ID" not in grp:
        return out
    quat_arr = np.asarray(
        grp["QUATERNIONS"][...], dtype=np.float64,
    ).reshape(-1, 4)
    qid_arr = np.asarray(grp["ID"][...]).flatten().astype(np.int64)
    if quat_arr.shape[0] != qid_arr.size:
        # Inconsistent dataset — fall back to identity.
        return out
    qid_to_row = {int(q): i for i, q in enumerate(qid_arr)}
    for k, eid in enumerate(element_ids):
        row = qid_to_row.get(int(eid))
        if row is not None:
            out[k, :] = quat_arr[row, :]
    return out


# =====================================================================
# Bucket layout resolution
# =====================================================================

@dataclass(frozen=True)
class LayerBucketLayout:
    """Resolved per-bucket layout: surface-GPs × layers × components.

    ``section_by_pair`` maps ``(elem_id, surface_gp_idx) → LayerSectionData``
    so different shell elements within one bucket can use different
    layered sections (with the same ``n_layers`` — heterogeneous
    layer counts are rejected at layout-resolve time).

    ``component_layout`` and ``n_components_per_cell`` come from the
    bucket's ``META/COMPONENTS``. A LayeredShellFiberSection
    typically writes a 5-component plane-stress + transverse-shear
    vector per (surface_GP, layer) under generic ``UnknownStress``
    labels; we surface them via index-suffixed canonicals
    (``fiber_stress_0`` … ``fiber_stress_4``). A 1-component bucket
    keeps the bare ``fiber_stress`` / ``fiber_strain`` canonical.
    """
    n_surface_gp: int             # from RESPONSE_CATALOG via class+rule
    section_by_pair: dict[tuple[int, int], LayerSectionData]
    n_layers: int
    component_layout: tuple[str, ...]
    n_components_per_cell: int
    n_sub_gp: int = 1             # v1 fixes this at 1


def parse_layer_meta_components(
    components_blob: bytes | str | np.ndarray,
    *,
    parent_token: str,
    bracket_key: str,
) -> tuple[tuple[str, ...], int]:
    """Parse ``META/COMPONENTS`` for a layered-shell bucket.

    ASDShellQ4 with LayeredShellFiberSection writes one
    ``;``-separated segment per surface GP, each declaring
    ``n_components_per_cell`` symbols (e.g. 5 ``UnknownStress*``
    labels). Every segment must agree.

    Returns
    -------
    (component_layout, n_components_per_cell)
        For a 1-component bucket: ``("fiber_stress",), 1`` (bare
        canonical preserved). For multi-component: index-suffixed
        canonicals like ``("fiber_stress_0", …, "fiber_stress_4")``.

    Raises
    ------
    ValueError
        If META is empty, malformed, or heterogeneous across surface
        GPs.
    """
    if isinstance(components_blob, np.ndarray):
        if components_blob.size != 1:
            raise ValueError(
                f"MPCO layer bucket {bracket_key!r}: META/COMPONENTS "
                f"array has {components_blob.size} entries; expected 1."
            )
        components_blob = components_blob[0]
    if isinstance(components_blob, bytes):
        raw = components_blob.decode("ascii")
    elif isinstance(components_blob, str):
        raw = components_blob
    else:
        raise ValueError(
            f"MPCO layer bucket {bracket_key!r}: META/COMPONENTS has "
            f"unexpected type {type(components_blob)}."
        )

    segments = [seg for seg in raw.split(";") if seg]
    if not segments:
        raise ValueError(
            f"MPCO layer bucket {bracket_key!r}: META/COMPONENTS empty."
        )

    parsed: list[tuple[str, ...]] = []
    for seg in segments:
        last_dot = seg.rfind(".")
        if last_dot < 0:
            raise ValueError(
                f"MPCO layer bucket {bracket_key!r}: META segment "
                f"{seg!r} has no dot separator."
            )
        symbols = tuple(s for s in seg[last_dot + 1:].split(",") if s)
        if not symbols:
            raise ValueError(
                f"MPCO layer bucket {bracket_key!r}: META segment "
                f"{seg!r} has no component symbols."
            )
        parsed.append(symbols)

    first = parsed[0]
    for i, syms in enumerate(parsed[1:], start=1):
        if syms != first:
            raise ValueError(
                f"MPCO layer bucket {bracket_key!r}: heterogeneous "
                f"per-surface-GP components (GP 0 = {first}, GP {i} = "
                f"{syms}). v1 supports only homogeneous component "
                f"layout per bucket."
            )

    n = len(first)
    if n == 1:
        return (parent_token,), 1
    # Multi-component: use index-suffixed canonicals. The META
    # symbols (``UnknownStress``, ``UnknownStress(1)``, …) carry no
    # semantic meaning STKO recognises, so an index suffix is the
    # honest representation.
    canonicals = tuple(f"{parent_token}_{i}" for i in range(n))
    return canonicals, n


def _surface_gp_count(elem_key: MPCOElementKey) -> int:
    """Look up the standard surface GP count for a shell class+rule.

    Uses the gauss-level :data:`RESPONSE_CATALOG` entry — every shell
    class has a fixed surface-GP rule (ASDShellQ4 → 4, ShellMITC9 → 9,
    ASDShellT3 → 3, etc.). The layer bucket reuses the same surface
    rule so the count is the same.
    """
    from ...solvers._element_response import RESPONSE_CATALOG
    # We try ``"stress"`` token; every layered shell has a gauss-level
    # stress entry. Fall back to ``"strain"`` for completeness.
    for token in ("stress", "strain"):
        key = (elem_key.class_name, elem_key.int_rule, token)
        layout = RESPONSE_CATALOG.get(key)
        if layout is not None:
            return layout.n_gauss_points
    raise ValueError(
        f"No gauss-level RESPONSE_CATALOG entry for class "
        f"{elem_key.class_name!r}, rule {elem_key.int_rule}. "
        f"Layered shell read needs the surface-GP count."
    )


def resolve_layer_bucket_layout(
    section_assignments_grp: "h5py.Group",
    bucket_grp: "h5py.Group",
    bucket: _LayerBucket,
) -> LayerBucketLayout:
    """Resolve surface-GP count, section geometry, and per-cell components.

    Walks ``META/COMPONENTS`` to determine ``n_components_per_cell``
    (e.g. 5 for ASDShellQ4 + LayeredShellFiberSection's
    plane-stress + transverse-shear vector). The bucket's
    ``NUM_COLUMNS`` attribute is cross-checked against
    ``n_sgp × n_layers × n_sub_gp × n_components_per_cell``.
    """
    if "ID" not in bucket_grp:
        raise ValueError(
            f"MPCO layer bucket {bucket.bracket_key!r}: missing ID dataset."
        )
    ids = np.asarray(bucket_grp["ID"][...]).flatten().astype(np.int64)
    if ids.size == 0:
        raise ValueError(
            f"MPCO layer bucket {bucket.bracket_key!r}: empty ID array."
        )
    n_sgp = _surface_gp_count(bucket.elem_key)
    section_by_pair = build_layer_section_index_for_bucket(
        section_assignments_grp, element_ids=ids, n_surface_gp=n_sgp,
    )
    layer_counts = {sec.n_layers for sec in section_by_pair.values()}
    if len(layer_counts) != 1:
        raise ValueError(
            f"MPCO layer bucket {bucket.bracket_key!r}: sections "
            f"reference different layer counts {sorted(layer_counts)}. "
            f"Heterogeneous layered sections within a bucket are not "
            f"supported in v1."
        )
    n_layers = layer_counts.pop()

    # Parse META to get per-cell component count + canonical names.
    # ``parent_token`` derives from the catalog entry's token: the
    # _LayerBucket carries the layout, but we need to know whether
    # this is fiber_stress or fiber_strain. The bucket's catalog
    # entry's token is recorded indirectly: layer_layout doesn't
    # carry it, but the element_key + LAYER_CATALOG mapping does.
    # Simplest approach: derive parent token from the on-disk group
    # name if it ends with ``stress`` / ``strain``; default to
    # ``fiber_stress``.
    if bucket.mpco_group_name.endswith("strain"):
        parent_token = "fiber_strain"
    else:
        parent_token = "fiber_stress"

    meta = bucket_grp.get("META")
    if meta is None or "COMPONENTS" not in meta:
        # Fallback: assume a single-component bucket. The validator
        # will catch any shape mismatch downstream.
        component_layout = (parent_token,)
        n_per_cell = 1
    else:
        component_layout, n_per_cell = parse_layer_meta_components(
            meta["COMPONENTS"][...],
            parent_token=parent_token,
            bracket_key=bucket.bracket_key,
        )

    return LayerBucketLayout(
        n_surface_gp=n_sgp,
        section_by_pair=section_by_pair,
        n_layers=n_layers,
        component_layout=component_layout,
        n_components_per_cell=n_per_cell,
    )


# =====================================================================
# Bucket validation
# =====================================================================

def validate_layer_bucket_meta(
    bucket_grp: "h5py.Group",
    layout: LayerBucketLayout,
    *,
    bracket_key: str,
) -> None:
    """Cross-check NUM_COLUMNS against the resolved bucket shape."""
    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        raise ValueError(
            f"MPCO layer bucket {bracket_key!r}: missing NUM_COLUMNS attr."
        )
    num_columns = int(_attr_scalar(num_columns_attr))
    expected = (
        layout.n_surface_gp * layout.n_layers * layout.n_sub_gp
        * layout.n_components_per_cell
    )
    if num_columns != expected:
        raise ValueError(
            f"MPCO layer bucket {bracket_key!r}: NUM_COLUMNS={num_columns} "
            f"!= n_sgp * n_layers * n_sub_gp * n_components_per_cell = "
            f"{layout.n_surface_gp} * {layout.n_layers} * "
            f"{layout.n_sub_gp} * {layout.n_components_per_cell} = "
            f"{expected}. Heterogeneous layered sections (or multi-"
            f"sub-GP integration) are not supported in v1."
        )


# =====================================================================
# Slab read — one bucket
# =====================================================================

def read_layer_bucket_slab(
    bucket_grp: "h5py.Group",
    section_assignments_grp: "h5py.Group",
    local_axes_grp: "h5py.Group | None",
    bucket: _LayerBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
    gp_indices: ndarray | None,
    layer_indices: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray,
           ndarray, ndarray] | None:
    """Read one layer-component slab from one bucket.

    ``canonical_component`` selects which META column to surface:
    bare ``fiber_stress`` / ``fiber_strain`` for single-component
    buckets; ``fiber_stress_<i>`` / ``fiber_strain_<i>`` for
    multi-component (e.g. ASDShellQ4 + LayeredShellFiberSection's
    5-component plane-stress vector).

    Returns
    -------
    (values, element_index, gp_index, layer_index, sub_gp_index,
     thickness, local_axes_quaternion) | None
        ``values``: ``(T, sum_L)`` flat — L sweeps element-slow,
        surface-GP-mid, layer-fast, sub-GP-fastest matching
        the LayerSlab schema.
        Returns ``None`` if no rows survive filtering, the bucket
        has no recorded steps, or the canonical is not in this
        bucket's META.
    """
    layout = resolve_layer_bucket_layout(
        section_assignments_grp, bucket_grp, bucket,
    )
    validate_layer_bucket_meta(
        bucket_grp, layout, bracket_key=bucket.bracket_key,
    )
    if canonical_component not in layout.component_layout:
        return None
    comp_idx = layout.component_layout.index(canonical_component)

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

    n_sgp = layout.n_surface_gp
    n_layers = layout.n_layers
    n_sub = layout.n_sub_gp

    # Filters.
    if gp_indices is None:
        sgp_sel = np.arange(n_sgp, dtype=np.int64)
    else:
        sgp_sel = np.asarray(gp_indices, dtype=np.int64)
        if sgp_sel.size == 0:
            return None
        if int(sgp_sel.max()) >= n_sgp or int(sgp_sel.min()) < 0:
            raise ValueError(
                f"gp_indices {sgp_sel.tolist()} out of range [0, {n_sgp})."
            )
    if layer_indices is None:
        lyr_sel = np.arange(n_layers, dtype=np.int64)
    else:
        lyr_sel = np.asarray(layer_indices, dtype=np.int64)
        if lyr_sel.size == 0:
            return None
        if int(lyr_sel.max()) >= n_layers or int(lyr_sel.min()) < 0:
            raise ValueError(
                f"layer_indices {lyr_sel.tolist()} out of range "
                f"[0, {n_layers})."
            )

    data_grp = bucket_grp["DATA"]
    step_keys = sorted(
        (k for k in data_grp.keys() if k.startswith("STEP_")),
        key=lambda s: int(s.split("_", 1)[1]),
    )
    if not step_keys:
        return None

    n_per_cell = layout.n_components_per_cell
    flat_size = n_sgp * n_layers * n_sub * n_per_cell
    E_g = sel_ids.size
    T = int(np.size(t_idx))

    flat = np.empty((T, E_g, flat_size), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        flat[i, :, :] = step_arr[sel_rows, :]

    # Reshape (T, E, n_sgp * n_lyr * n_sub * n_per_cell)
    # → (T, E, n_sgp, n_lyr, n_sub, n_per_cell). The MPCO column
    # order is sgp-slowest, layer-mid, sub_gp-then-component-fast
    # (matching the META MULTIPLICITY=n_layers, NUM_COMPONENTS=K
    # block structure).
    reshaped = flat.reshape(T, E_g, n_sgp, n_layers, n_sub, n_per_cell)
    # Pick the requested canonical component.
    per_layer = reshaped[:, :, :, :, :, comp_idx]

    # Apply filters.
    per_layer = per_layer[:, :, sgp_sel, :, :]
    per_layer = per_layer[:, :, :, lyr_sel, :]
    n_sgp_sel = sgp_sel.size
    n_lyr_sel = lyr_sel.size

    # Pack into LayerSlab axes.
    sum_l = E_g * n_sgp_sel * n_lyr_sel * n_sub
    values = np.ascontiguousarray(per_layer.reshape(T, sum_l))

    # Per-element-row quaternions, then broadcast across (sgp × lyr × sub).
    quats = read_local_axes_quaternions(
        local_axes_grp, bucket.elem_key, sel_ids,
    )                                                     # (E_g, 4)

    # Index vectors built outer-fastest using nested np.tile/repeat:
    # row k corresponds to (e, sgp, lyr, sub) with sub fastest.
    block_per_element = n_sgp_sel * n_lyr_sel * n_sub
    block_per_sgp = n_lyr_sel * n_sub
    block_per_layer = n_sub

    element_index = np.repeat(sel_ids, block_per_element).astype(np.int64)
    gp_index = np.tile(
        np.repeat(sgp_sel, block_per_sgp), E_g,
    ).astype(np.int64)
    layer_index = np.tile(
        np.repeat(lyr_sel, block_per_layer), E_g * n_sgp_sel,
    ).astype(np.int64)
    sub_gp_index = np.tile(
        np.arange(n_sub, dtype=np.int64),
        E_g * n_sgp_sel * n_lyr_sel,
    )

    # Thickness comes per (element, surface_gp) section assignment —
    # different elements may use different layered sections (same
    # n_layers, different thickness profile). Walk the selected rows
    # and hydrate per (element, surface_gp).
    sum_l = E_g * n_sgp_sel * n_lyr_sel * n_sub
    thickness = np.empty(sum_l, dtype=np.float64)
    block_per_pair = n_lyr_sel * n_sub
    cursor = 0
    for eid in sel_ids:
        for sgp in sgp_sel:
            sec = layout.section_by_pair[(int(eid), int(sgp))]
            thickness_per_layer = sec.layer_thickness[lyr_sel]
            thickness[cursor:cursor + block_per_pair] = np.repeat(
                thickness_per_layer, n_sub,
            )
            cursor += block_per_pair

    # Quaternion is per-element (broadcast across sgp × lyr × sub).
    local_axes_quaternion = np.repeat(
        quats, block_per_element, axis=0,
    )                                                    # (sum_L, 4)

    return (values, element_index, gp_index, layer_index,
            sub_gp_index, thickness, local_axes_quaternion)


__all__ = [
    "LayerBucketLayout",
    "LayerSectionData",
    "canonical_to_layer_token",
    "discover_layer_buckets",
    "find_layered_section_for_element",
    "read_layer_bucket_slab",
    "read_local_axes_quaternions",
    "resolve_layer_bucket_layout",
    "validate_layer_bucket_meta",
]
