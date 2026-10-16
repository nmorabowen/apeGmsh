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

Every header index is read (``:0``, ``:1``, ...): MPCO opens a new
header when one class's elements write different META, and each header
bucket is checked against its own ``NUM_COLUMNS`` and META by
:func:`validate_bucket_meta` before it is read.

v1 scope (skipped when encountered):

- ``custom_rule_idx != 0`` (force-based beams with user IPs).
- Element classes not yet in the catalog.

Anything outside the v1 scope simply contributes nothing to the slab —
the user can always extend the catalog and re-read the same file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ..._vocabulary import STRAIN_2D, STRESS_2D
from ...opensees._response_catalog import (
    RESPONSE_CATALOG,
    IntRule,
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
    #: The bracket's rule was 0, so ``layout`` is the class's one
    #: catalogued rule, chosen by class name rather than named by the file.
    rule_from_class: bool = False


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
    :func:`apeGmsh.opensees._response_catalog.mpco_gauss_group_aliases`
    and accumulate buckets across every group that exists, skipping
    duplicates so the same bucket key is never returned twice.
    Buckets that don't shape-match the catalog are filtered per-bucket.
    Every header index is returned; each header bucket's width comes
    from its own ``NUM_COLUMNS`` and its META is validated at read.

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
    # The ON_ELEMENTS group is named by the keyword the deck asked for,
    # so ``-E stress`` lands under ``stress``, not ``stresses``.
    if catalog_token not in candidate_names:
        candidate_names = candidate_names + (catalog_token,)

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
            if elem_key.custom_rule_idx != 0:
                continue
            found = catalog_layout(
                elem_key, catalog_token, where=f"{name}/{bracket_key}",
            )
            if found is None:
                continue
            layout = _maybe_extend_plane_layout(found, token_grp[bracket_key])
            out.append(_Bucket(
                bracket_key=bracket_key,
                elem_key=elem_key,
                layout=layout,
                mpco_group_name=name,
                rule_from_class=(
                    elem_key.int_rule == IntRule.NoIntegrationRule
                ),
            ))
            seen_keys.add(bracket_key)
            if found_name is None:
                found_name = name
    return (found_name or mpco_group_name, out)


# The fork's MPCO recorder writes the bracket rule as 0
# (NoIntegrationRule) for every Ladruno-band class tag — LadrunoBrick20 is
# ``33018-LadrunoBrick20[0:0:0]`` — so the bracket cannot be the catalog's
# ``int_rule`` key. No catalog row is keyed on rule 0, and every catalogued
# (class, token) has at most one non-Custom rule, so a rule-0 bucket takes
# the class's one row. Its width and GP blocks are still checked against
# that row at read time, so a bucket that does not fit refuses loudly.

def catalog_layout(
    elem_key: MPCOElementKey, catalog_token: str, *, where: str,
) -> ResponseLayout | None:
    """The catalog row for one bucket, or ``None`` for an uncatalogued class.

    Raises ``ValueError`` for a rule-0 bucket whose class the catalog
    places at more than one Gauss rule under this token: which one the
    file holds cannot be told from the bracket.
    """
    cls = elem_key.class_name
    if is_catalogued(cls, elem_key.int_rule, catalog_token):
        return lookup(cls, elem_key.int_rule, catalog_token)
    if elem_key.int_rule != IntRule.NoIntegrationRule:
        return None
    rows = [
        lay for (c, rule, tok), lay in RESPONSE_CATALOG.items()
        if c == cls and tok == catalog_token and rule != IntRule.Custom
    ]
    if len(rows) > 1:
        raise ValueError(
            f"MPCO bucket {where!r} carries integration rule 0, and "
            f"RESPONSE_CATALOG has {len(rows)} rules for class {cls!r} / "
            f"token {catalog_token!r}. Refusing to guess which one the "
            f"file holds."
        )
    return rows[0] if rows else None


# =====================================================================
# Out-of-plane (σ_zz) width adaptation
# =====================================================================

# A plane element that records the out-of-plane component stores one
# extra column per Gauss point beyond the 3-component catalog layout:
# ``[σxx, σyy, σxy, σzz]`` (likewise strain). The recorded file is
# self-describing via ``NUM_COLUMNS``; when it is exactly one component
# wider than the catalog's 3-component plane stress/strain layout we
# extend the layout in place so ``stress_zz`` / ``strain_zz`` is read as
# the 4th column. 3-component files are untouched (backward compatible).
_PLANE_OUT_OF_PLANE: dict[tuple[str, ...], str] = {
    STRESS_2D: "stress_zz",
    STRAIN_2D: "strain_zz",
}


def _maybe_extend_plane_layout(
    layout: ResponseLayout, bucket_grp: "h5py.Group",
) -> ResponseLayout:
    """Extend a 3-component plane layout to 4 when the bucket stores σ_zz.

    Returns ``layout`` unchanged unless it is a 3-component plane
    stress/strain layout AND the bucket's ``NUM_COLUMNS`` equals
    ``n_gauss_points * (n_components_per_gp + 1)`` — i.e. the recorder
    appended one out-of-plane component per Gauss point.
    """
    extra = _PLANE_OUT_OF_PLANE.get(layout.component_layout)
    if extra is None:
        return layout
    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        return layout
    try:
        stored = int(_attr_scalar(num_columns_attr))
    except Exception:
        return layout
    widened = layout.n_gauss_points * (layout.n_components_per_gp + 1)
    if stored != widened:
        return layout
    return replace(
        layout,
        n_components_per_gp=layout.n_components_per_gp + 1,
        component_layout=layout.component_layout + (extra,),
    )


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

    meta = bucket_grp["META"] if "META" in bucket_grp else None
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


_GENERIC_COMP_RE = re.compile(r"^C\d+$")


# =====================================================================
# META column names → catalog canonicals
# =====================================================================

# The labels MPCO writes for a Gauss bucket's columns, by catalog canonical
# (dumped from fork build 79e0623670 for FourNodeQuad, SSPquad, Brick,
# BbarBrick, SSPbrick, LadrunoBrick, ShellMITC4, ASDShellQ4, ShellDKGQ,
# ShellDKGT and ASDShellT3): continuum stress ``sigma11,sigma22,sigma33,
# sigma12,sigma23,sigma13`` (plane: ``sigma11,sigma22,sigma12``, plus
# ``sigma33`` last for ``stressPlaneStrain``); continuum strain ``eps11..``
# (3-D) or ``eta11..`` (plane); shell resultants ``p11,p22,p12,m11,m22,m12,
# q1,q2``; shell strains ``eps11,eps22,gamma12,theta11,theta22,theta33,
# gamma13,gamma23``. A material that tags no names writes ``C1..Cn``
# (LadrunoConcrete3D under ``material.stress``), and those columns take the
# catalog's order. Lookup is case-insensitive.
_XYZ = {"x": "1", "y": "2", "z": "3"}

_SHELL_LABELS: dict[str, frozenset[str]] = {
    "membrane_force_xx": frozenset({"p11"}),
    "membrane_force_yy": frozenset({"p22"}),
    "membrane_force_xy": frozenset({"p12", "p21"}),
    "bending_moment_xx": frozenset({"m11"}),
    "bending_moment_yy": frozenset({"m22"}),
    "bending_moment_xy": frozenset({"m12", "m21"}),
    "transverse_shear_xz": frozenset({"q1", "q13"}),
    "transverse_shear_yz": frozenset({"q2", "q23"}),
    "membrane_strain_xx": frozenset({"eps11"}),
    "membrane_strain_yy": frozenset({"eps22"}),
    "membrane_strain_xy": frozenset({"gamma12", "eps12"}),
    "curvature_xx": frozenset({"theta11"}),
    "curvature_yy": frozenset({"theta22"}),
    "curvature_xy": frozenset({"theta33", "theta12"}),
    "transverse_shear_strain_xz": frozenset({"gamma13"}),
    "transverse_shear_strain_yz": frozenset({"gamma23"}),
}

_TENSOR_RE = re.compile(r"^(?P<root>stress|strain)_(?P<a>[xyz])(?P<b>[xyz])$")


def _labels_for(canonical: str) -> frozenset[str]:
    """Every META label (lower case) that names ``canonical``."""
    if canonical in _SHELL_LABELS:
        return _SHELL_LABELS[canonical]
    m = _TENSOR_RE.match(canonical)
    if m is None:
        return frozenset({canonical.lower()})
    a, b = m.group("a"), m.group("b")
    ij = {_XYZ[a] + _XYZ[b], _XYZ[b] + _XYZ[a]}
    ab = {a + b, b + a}
    stems_digit: tuple[str, ...]
    stems_axis: tuple[str, ...]
    if m.group("root") == "stress":
        stems_digit, stems_axis = ("sigma",), ("sigma_", "s", "stress_")
    else:
        stems_digit = ("eps", "eta", "epsilon") + (("gamma",) if a != b else ())
        stems_axis = ("eps_", "epsilon_", "strain_") + (
            ("gamma_",) if a != b else ())
    return frozenset(
        {s + d for s in stems_digit for d in ij}
        | {s + x for s in stems_axis for x in ab}
    )


def _meta_column_labels(
    bucket_grp: "h5py.Group", *, where: str,
) -> list[tuple[str, ...]]:
    """The column labels of each META block, in block order."""
    meta = bucket_grp["META"]
    if "COMPONENTS" not in meta:
        raise ValueError(
            f"MPCO bucket {where!r}: META carries no COMPONENTS, so the "
            f"names of its columns are unknown. Refusing to read it by "
            f"position."
        )
    raw = np.asarray(meta["COMPONENTS"][...]).flatten()
    if raw.size != 1:
        raise ValueError(
            f"MPCO bucket {where!r}: META/COMPONENTS holds {raw.size} "
            f"entries; expected 1."
        )
    text = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
    return [
        tuple(s.strip() for s in seg.rpartition(".")[2].split(",") if s.strip())
        for seg in text.split(";") if seg
    ]


def resolve_gauss_component_names(
    bucket_grp: "h5py.Group", layout: ResponseLayout, *, where: str,
) -> ResponseLayout:
    """``layout`` with its component names in the bucket's own META order.

    Each column label must name exactly one of the catalog's components
    (``sigma33`` → ``stress_zz`` wherever it sits). MPCO's unnamed
    ``C1..Cn`` columns keep the catalog order, which RESPONSE_CATALOG
    defines for them.

    Raises
    ------
    ValueError
        When a label names none (or several) of the catalog's components,
        when two labels name the same one, when the blocks disagree, or
        when META carries no column names: the bucket's columns would be
        read under names its own META contradicts.
    """
    blocks = _meta_column_labels(bucket_grp, where=where)
    expected = list(layout.component_layout)
    if len(blocks) != layout.n_gauss_points or any(
        b != blocks[0] for b in blocks[1:]
    ):
        raise ValueError(
            f"MPCO bucket {where!r}: META names {len(blocks)} Gauss blocks "
            f"with differing columns; the catalog layout {expected} needs "
            f"{layout.n_gauss_points} blocks with one column list."
        )
    labels = blocks[0]
    if len(labels) != layout.n_components_per_gp:
        raise ValueError(
            f"MPCO bucket {where!r}: META names {len(labels)} columns per "
            f"Gauss point {list(labels)}; the catalog layout has "
            f"{layout.n_components_per_gp} {expected}."
        )
    if all(_GENERIC_COMP_RE.match(lab) for lab in labels) and [
        int(lab[1:]) for lab in labels
    ] == list(range(1, len(labels) + 1)):
        return layout
    names: list[str] = []
    for lab in labels:
        hits = [c for c in expected if lab.lower() in _labels_for(c)]
        if len(hits) != 1:
            raise ValueError(
                f"MPCO bucket {where!r}: META column {lab!r} (of "
                f"{list(labels)}) names {len(hits)} of the catalog "
                f"components {expected} for class tag {layout.class_tag}. "
                f"Refusing to read its columns under names its META "
                f"contradicts."
            )
        names.append(hits[0])
    if len(set(names)) != len(names):
        raise ValueError(
            f"MPCO bucket {where!r}: META columns {list(labels)} name the "
            f"catalog components {names}, one of them twice."
        )
    if tuple(names) == layout.component_layout:
        return layout
    return replace(layout, component_layout=tuple(names))


def is_unnamed_element_block(
    bucket_grp: "h5py.Group", layout: ResponseLayout,
) -> bool:
    """Whether the bucket is one unnamed element-level block of the catalog's width.

    An element that tags no Gauss points or component names (LadrunoLST)
    is written as a single element-level block (``GAUSS_IDS == [-1]``)
    labelled ``C1..Cn``. Its columns can only be named by the catalog, so
    it is read by the catalog layout when, and only when, its width is
    exactly the layout's. A block with its own names (Truss ``N``) or
    another width is not one; :func:`validate_bucket_meta` then refuses it.
    """
    if "META" not in bucket_grp:
        return False
    meta = bucket_grp["META"]
    if not all(k in meta for k in (
        "MULTIPLICITY", "GAUSS_IDS", "NUM_COMPONENTS", "COMPONENTS",
    )):
        return False
    num_columns_attr = bucket_grp.attrs.get("NUM_COLUMNS")
    if num_columns_attr is None:
        return False
    width = layout.flat_size_per_element
    if int(_attr_scalar(num_columns_attr)) != width:
        return False
    gauss_ids = np.asarray(meta["GAUSS_IDS"][...]).flatten()
    multiplicity = np.asarray(meta["MULTIPLICITY"][...]).flatten()
    num_components = np.asarray(meta["NUM_COMPONENTS"][...]).flatten()
    if (gauss_ids.tolist() != [-1] or multiplicity.tolist() != [1]
            or num_components.tolist() != [width]):
        return False
    raw = np.asarray(meta["COMPONENTS"][...]).flatten()
    if raw.size != 1:
        return False
    text = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
    names = text.rpartition(".")[2].split(",")
    return len(names) == width and all(
        _GENERIC_COMP_RE.match(n) for n in names
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

    if not is_unnamed_element_block(bucket_grp, layout):
        try:
            validate_bucket_meta(
                bucket_grp, layout, bracket_key=bucket.bracket_key,
            )
        except ValueError as exc:
            if not bucket.rule_from_class:
                raise
            raise ValueError(
                f"{exc} The bracket carries integration rule 0, so this "
                f"{layout.n_gauss_points}-GP layout is the only rule "
                f"RESPONSE_CATALOG has for {bucket.elem_key.class_name!r}, "
                f"resolved by class name, not one the file named. The "
                f"elements were integrated with another rule (another "
                f"formulation of the same class, such as one with a "
                f"reduced rule), so MPCO cannot say where their Gauss "
                f"points are. Record this element with the fork's native "
                f"recorder, which stores each element's own Gauss points "
                f"and reads it correctly."
            ) from exc
        layout = resolve_gauss_component_names(
            bucket_grp, layout,
            where=f"{bucket.mpco_group_name}/{bucket.bracket_key}",
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
