"""Material-state result decoding for MPCOReader.

Damage-tracking constitutive models (ASDConcrete, plastic-damage
formulations, etc.) expose per-GP scalar state variables under MPCO
group names like ``damage`` / ``material.damage`` and
``equivalentPlasticStrain`` / ``material.equivalentPlasticStrain``.
Unlike the continuum stress/strain buckets, the *number* of values
per GP is **material-dependent**:

- A single-state damage material writes one scalar per GP.
- A tension/compression damage-plasticity material (ASDConcrete and
  family) writes two scalars per GP, named ``d+`` / ``d-`` (damage)
  or ``PLE+`` / ``PLE-`` (plastic strain) in
  ``META/COMPONENTS``.

The :data:`apeGmsh.solvers._element_response.RESPONSE_CATALOG`
declares fixed component layouts per ``(class, int_rule, token)``,
which doesn't fit material-state's META-driven shape. This module
takes the alternate path used by fiber/layer reads: re-use the
class+int_rule entry from the catalog only for its **GP layout**
(n_GP, natural_coords, coord_system, class_tag), then build a
concrete :class:`ResponseLayout` per bucket from the bucket's META.

Component naming
----------------

Multi-segment buckets get one canonical per segment, prefixed by the
parent token. Known META symbols map to apeGmsh suffixes via
:data:`apeGmsh.results._vocabulary.MPCO_MATERIAL_SYMBOL_TO_CANONICAL_SUFFIX`
(``d+`` → ``tension``, ``d-`` → ``compression``, etc.). Unknown
symbols fall back to indexed suffixes (``damage_0``, ``damage_1``).
A single-segment bucket maps to the bare parent canonical
(``damage`` / ``equivalent_plastic_strain``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from .._vocabulary import MPCO_MATERIAL_SYMBOL_TO_CANONICAL_SUFFIX
from ...solvers._element_response import (
    RESPONSE_CATALOG,
    MPCOElementKey,
    ResponseLayout,
    is_catalogued,
    lookup,
    mpco_gauss_group_aliases,
    parse_mpco_element_key,
)
from ._mpco_element_io import _attr_scalar

if TYPE_CHECKING:
    import h5py


# =====================================================================
# Token routing — the parent token (damage / equivalent_plastic_strain)
# determines the MPCO group; canonicals strip a suffix to recover it.
# =====================================================================

# Recognised parent tokens and their MPCO primary group names.
_PARENT_TOKEN_TO_MPCO_GROUP: dict[str, str] = {
    "damage": "damage",
    "equivalent_plastic_strain": "equivalentPlasticStrain",
}


def parent_token_for_canonical(canonical: str) -> str | None:
    """Recover the parent material-state token from a canonical name.

    Examples
    --------
    ``"damage"``                              → ``"damage"``
    ``"damage_tension"``                      → ``"damage"``
    ``"damage_compression"``                  → ``"damage"``
    ``"damage_3"``                            → ``"damage"``
    ``"equivalent_plastic_strain"``           → ``"equivalent_plastic_strain"``
    ``"equivalent_plastic_strain_tension"``   → ``"equivalent_plastic_strain"``
    ``"stress_xx"``                           → ``None``
    """
    for parent in _PARENT_TOKEN_TO_MPCO_GROUP:
        if canonical == parent:
            return parent
        if canonical.startswith(parent + "_"):
            return parent
    return None


# =====================================================================
# META segment parsing
# =====================================================================

def _normalise_symbol(symbol: str, *, fallback_index: int) -> str:
    """Map an MPCO META symbol to an apeGmsh canonical suffix.

    Known symbols (``d+`` / ``d-`` / ``PLE+`` / ``PLE-``) get
    semantic suffixes (``tension`` / ``compression``); unknown symbols
    fall back to the segment's 0-based index.
    """
    return MPCO_MATERIAL_SYMBOL_TO_CANONICAL_SUFFIX.get(
        symbol.strip(), str(fallback_index),
    )


def parse_meta_components(
    components_blob: bytes | str,
    *,
    parent_token: str,
    bracket_key: str,
) -> tuple[tuple[str, ...], int]:
    """Parse ``META/COMPONENTS`` for a material-state bucket.

    The blob is one ``;``-separated string per IP, each segment
    structured as ``"<descriptor_path>.<comp1>,<comp2>,..."``. Every
    IP must list the same components for the bucket to have a
    well-defined per-GP shape (v1 invariant; heterogeneous IPs raise).

    Returns
    -------
    (component_layout, n_components_per_gp)
        ``component_layout`` is the apeGmsh canonical names in the
        order they appear in the file (e.g.
        ``("damage_tension", "damage_compression")`` for a ``d+,d-``
        bucket on token=``damage``). For a single-component
        bucket emitting just one symbol the bare parent canonical
        is returned (``("damage",)``).

    Raises
    ------
    ValueError
        If META is empty, malformed, or heterogeneous across IPs.
    """
    if isinstance(components_blob, np.ndarray):
        if components_blob.size != 1:
            raise ValueError(
                f"MPCO material-state bucket {bracket_key!r}: "
                f"META/COMPONENTS array has {components_blob.size} "
                f"entries; expected 1."
            )
        components_blob = components_blob[0]
    if isinstance(components_blob, bytes):
        raw = components_blob.decode("ascii")
    elif isinstance(components_blob, str):
        raw = components_blob
    else:
        raise ValueError(
            f"MPCO material-state bucket {bracket_key!r}: "
            f"META/COMPONENTS has unexpected type {type(components_blob)}."
        )

    segments = [seg for seg in raw.split(";") if seg]
    if not segments:
        raise ValueError(
            f"MPCO material-state bucket {bracket_key!r}: "
            f"META/COMPONENTS is empty."
        )

    parsed_per_ip: list[tuple[str, ...]] = []
    for seg in segments:
        last_dot = seg.rfind(".")
        if last_dot < 0:
            raise ValueError(
                f"MPCO material-state bucket {bracket_key!r}: META "
                f"segment {seg!r} has no dot separator."
            )
        comps_csv = seg[last_dot + 1:]
        symbols = tuple(s for s in comps_csv.split(",") if s)
        if not symbols:
            raise ValueError(
                f"MPCO material-state bucket {bracket_key!r}: META "
                f"segment {seg!r} has no component symbols after "
                f"the dot."
            )
        parsed_per_ip.append(symbols)

    first = parsed_per_ip[0]
    for i, syms in enumerate(parsed_per_ip[1:], start=1):
        if syms != first:
            raise ValueError(
                f"MPCO material-state bucket {bracket_key!r}: "
                f"heterogeneous components across IPs (IP 0 = "
                f"{first}, IP {i} = {syms}). v1 supports only "
                f"homogeneous component layout per bucket."
            )

    if len(first) == 1:
        # Single-component material — use the bare parent canonical.
        return (parent_token,), 1
    canonicals = tuple(
        f"{parent_token}_{_normalise_symbol(sym, fallback_index=i)}"
        for i, sym in enumerate(first)
    )
    # Detect duplicate suffixes after normalisation (e.g. an
    # unrecognised symbol's index colliding with a known one).
    if len(set(canonicals)) != len(canonicals):
        raise ValueError(
            f"MPCO material-state bucket {bracket_key!r}: META symbols "
            f"{first} produced duplicate canonicals {canonicals}."
        )
    return canonicals, len(canonicals)


# =====================================================================
# Per-bucket descriptor + layout resolution
# =====================================================================

@dataclass(frozen=True)
class _MaterialBucket:
    """One ``ON_ELEMENTS/<token>/<bracket_key>`` group + alias info.

    ``mpco_group_name`` is the alias under which the bucket lives
    (``damage`` vs. ``material.damage``). The catalog ``layout`` is
    the *stress* layout for the same (class, int_rule) — used only
    for its GP layout (n_GP, natural_coords); component_layout is
    irrelevant and gets overridden per-bucket from META.
    """
    bracket_key: str
    elem_key: MPCOElementKey
    catalog_gp_layout: ResponseLayout    # stress entry for the same class+rule
    mpco_group_name: str
    parent_token: str                    # "damage" / "equivalent_plastic_strain"


def _stress_gp_layout_for(elem_key: MPCOElementKey) -> ResponseLayout | None:
    """Look up the stress catalog entry for the same class + rule.

    Material-state buckets share GP layout (n_GP and natural_coords)
    with their stress sibling. We don't have a dedicated catalog
    entry for damage / eps_pl shape (META carries the per-component
    structure), but the per-class GP coordinates are still needed
    for the slab's natural_coords.
    """
    if not is_catalogued(elem_key.class_name, elem_key.int_rule, "stress"):
        return None
    return lookup(elem_key.class_name, elem_key.int_rule, "stress")


# =====================================================================
# Bucket discovery
# =====================================================================

def discover_material_state_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_MaterialBucket]]:
    """Walk MPCO group(s) for the parent token and return matching buckets.

    Aliases (``damage`` + ``material.damage``, etc.) are walked in
    order; duplicate bracket keys across alias groups dedup. Each
    returned bucket carries its source ``mpco_group_name`` so the
    read site can fetch the right dataset path.

    A bucket is included only if its (class, int_rule) is in the
    stress catalog (gives us the GP coordinates).
    """
    parent = parent_token_for_canonical(canonical_component)
    if parent is None:
        return (None, [])
    primary = _PARENT_TOKEN_TO_MPCO_GROUP[parent]
    candidate_names = mpco_gauss_group_aliases(primary)

    out: list[_MaterialBucket] = []
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
            gp_layout = _stress_gp_layout_for(elem_key)
            if gp_layout is None:
                continue
            out.append(_MaterialBucket(
                bracket_key=bracket_key,
                elem_key=elem_key,
                catalog_gp_layout=gp_layout,
                mpco_group_name=name,
                parent_token=parent,
            ))
            seen_keys.add(bracket_key)
            if found_name is None:
                found_name = name
    return (found_name or primary, out)


# =====================================================================
# Layout resolution + slab read
# =====================================================================

def resolve_material_state_layout(
    bucket_grp: "h5py.Group",
    bucket: _MaterialBucket,
) -> ResponseLayout:
    """Build a concrete :class:`ResponseLayout` from META.

    Validates META block structure (one block per GP, equal
    NUM_COMPONENTS) and cross-checks the bucket's
    ``NUM_COLUMNS`` attribute against ``n_GP × n_components_per_gp``.
    """
    meta = bucket_grp.get("META")
    if meta is None or "COMPONENTS" not in meta:
        raise ValueError(
            f"MPCO material-state bucket {bucket.bracket_key!r}: "
            f"missing META/COMPONENTS — cannot resolve layout."
        )
    components_blob = meta["COMPONENTS"][...]
    component_layout, n_per_gp = parse_meta_components(
        components_blob,
        parent_token=bucket.parent_token,
        bracket_key=bucket.bracket_key,
    )
    n_gp = bucket.catalog_gp_layout.n_gauss_points
    expected_columns = n_gp * n_per_gp

    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        raise ValueError(
            f"MPCO material-state bucket {bucket.bracket_key!r}: "
            f"missing NUM_COLUMNS attribute."
        )
    num_columns = int(_attr_scalar(num_columns_attr))
    if num_columns != expected_columns:
        raise ValueError(
            f"MPCO material-state bucket {bucket.bracket_key!r}: "
            f"NUM_COLUMNS={num_columns} but META resolves to "
            f"n_GP * n_components_per_gp = {n_gp} * {n_per_gp} = "
            f"{expected_columns}."
        )

    return ResponseLayout(
        n_gauss_points=n_gp,
        natural_coords=bucket.catalog_gp_layout.natural_coords,
        coord_system=bucket.catalog_gp_layout.coord_system,
        n_components_per_gp=n_per_gp,
        component_layout=component_layout,
        class_tag=bucket.catalog_gp_layout.class_tag,
    )


def read_material_state_bucket_slab(
    bucket_grp: "h5py.Group",
    bucket: _MaterialBucket,
    canonical_component: str,
    *,
    t_idx: ndarray,
    element_ids: ndarray | None,
) -> tuple[ndarray, ndarray, ndarray] | None:
    """Read one component's slab from one material-state bucket.

    Returns
    -------
    (values, element_index, natural_coords) or None
        Same shape conventions as :func:`_mpco_element_io.read_bucket_slab`.
        Returns ``None`` when the bucket has no recorded steps, when
        no elements survive filtering, or when the requested
        canonical is not present in this bucket's META layout.
    """
    layout = resolve_material_state_layout(bucket_grp, bucket)
    if canonical_component not in layout.component_layout:
        return None
    comp_idx = layout.component_layout.index(canonical_component)
    n_gp = layout.n_gauss_points
    n_per_gp = layout.n_components_per_gp

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

    # Read all components, then pick the requested one. The flat
    # column layout is GP-slowest / component-fastest:
    #     col(gp, k) = gp * n_per_gp + k
    flat = np.empty((T, E_g, n_gp * n_per_gp), dtype=np.float64)
    for i, k in enumerate(t_idx):
        step_arr = np.asarray(
            data_grp[step_keys[int(k)]][...], dtype=np.float64,
        )
        flat[i, :, :] = step_arr[sel_rows, :]
    reshaped = flat.reshape(T, E_g, n_gp, n_per_gp)
    per_gp = reshaped[:, :, :, comp_idx]   # (T, E_g, n_GP)
    values = per_gp.reshape(T, E_g * n_gp)

    element_index = np.repeat(sel_ids, n_gp).astype(np.int64)
    natural_coords = np.tile(layout.natural_coords, (E_g, 1))

    return values, element_index, natural_coords


def material_state_canonicals_in_bucket(
    bucket_grp: "h5py.Group",
    bucket: _MaterialBucket,
) -> tuple[str, ...]:
    """List the apeGmsh canonicals present in this bucket's META.

    Cheap helper for :func:`available_components` discovery —
    returns the resolved layout's component_layout without reading
    any DATA datasets.
    """
    layout = resolve_material_state_layout(bucket_grp, bucket)
    return layout.component_layout


__all__ = [
    "discover_material_state_buckets",
    "material_state_canonicals_in_bucket",
    "parent_token_for_canonical",
    "parse_meta_components",
    "read_material_state_bucket_slab",
    "resolve_material_state_layout",
]
