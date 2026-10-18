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

A bucket with a nonzero custom-rule index (``209-SixNodeTri[1000:2:0]``)
is left unread and named in one ``MPCOBucketSkippedWarning``. MPCO opens
a new custom-rule index for each distinct set of Gauss-point positions
(``GP_X``), so the catalog's fixed coordinates cannot be trusted for it,
and this reader does not yet check ``GP_X``.

A bucket whose class has no catalog row for the token is left unread
too, and named the same way (#2160): the catalog is what places its
Gauss points, so without a row its values cannot be returned.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
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
            what = f"gauss discovery of {canonical_component!r}"
            try:
                elem_key = parse_mpco_element_key(bracket_key)
            except ValueError as exc:
                warn_unparseable_bucket_key(
                    token_grp[bracket_key], exc, what=what,
                )
                seen_keys.add(bracket_key)
                continue
            if elem_key.custom_rule_idx != 0:
                warn_unread_custom_rule_bucket(
                    token_grp[bracket_key], elem_key, what=what,
                )
                seen_keys.add(bracket_key)
                continue
            found = catalog_layout(
                elem_key, catalog_token, where=f"{name}/{bracket_key}",
            )
            if found is None:
                warn_uncatalogued_bucket(
                    token_grp[bracket_key], elem_key, catalog_token,
                    what=what,
                )
                seen_keys.add(bracket_key)
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


def warn_unread_custom_rule_bucket(
    bucket_grp: "h5py.Group",
    elem_key: MPCOElementKey,
    *,
    what: str,
) -> None:
    """Name a bucket with a nonzero custom-rule index, by path and class.

    MPCO opens a new custom-rule index for each distinct set of
    Gauss-point positions (``GP_X``), so RESPONSE_CATALOG's fixed
    coordinates cannot be trusted for it, and the bucket is left unread.
    One ``MPCOBucketSkippedWarning`` per bucket; the other buckets still
    read.
    """
    warn_unread_bucket(
        bucket_grp,
        what=what,
        reason=(
            f"class {elem_key.class_name!r} carries custom integration "
            f"rule {elem_key.custom_rule_idx} (bracket rule "
            f"{elem_key.int_rule}). MPCO numbers a custom rule per set of "
            f"Gauss-point positions (GP_X), which this reader does not "
            f"verify, so the catalog's Gauss points cannot be assumed"
        ),
    )


def warn_uncatalogued_bucket(
    bucket_grp: "h5py.Group",
    elem_key: MPCOElementKey,
    catalog_token: str,
    *,
    what: str,
) -> None:
    """Name a standard-rule bucket whose class has no catalog row.

    RESPONSE_CATALOG is what gives a bucket its Gauss points and column
    layout, so a ``(class, rule, token)`` it lacks is left unread. One
    ``MPCOBucketSkippedWarning`` per bucket; the other buckets still
    read, and a catalog row for the class makes the same file read.
    """
    warn_unread_bucket(
        bucket_grp,
        what=what,
        reason=(
            f"class {elem_key.class_name!r} at integration rule "
            f"{elem_key.int_rule} has no RESPONSE_CATALOG row for token "
            f"{catalog_token!r}, which is what places its Gauss points"
        ),
    )


def warn_unparseable_bucket_key(
    bucket_grp: "h5py.Group", exc: ValueError, *, what: str,
) -> None:
    """Name a bucket whose bracket key does not parse (#2165).

    Without a parsed key the reader has no class, rule or header to look
    the bucket up by, so it is left unread. One
    ``MPCOBucketSkippedWarning`` per key; the other buckets still read.
    """
    warn_unread_bucket(
        bucket_grp,
        what=what,
        reason=f"its bracket key does not parse ({exc})",
    )


def warn_unread_bucket(
    bucket_grp: "h5py.Group", *, what: str, reason: str,
) -> None:
    """One ``MPCOBucketSkippedWarning`` naming one bucket (path, reason, file).

    Called through one reason-specific wrapper from a discovery function:
    ``stacklevel`` points past this helper, the wrapper and the discovery
    function, at the reader method's caller.
    """
    from ._mpco import MPCOBucketSkippedWarning
    warnings.warn(
        f"MPCO {what} in {Path(bucket_grp.file.filename).name} left 1 "
        f"bucket(s) unread, so their elements return no values: "
        f"{bucket_grp.name}: {reason}.",
        MPCOBucketSkippedWarning, stacklevel=5,
    )


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

# The labels MPCO writes for a Gauss bucket's columns are the
# ``ResponseType`` tags of the element's or the material's ``setResponse``.
# Swept from the fork source (c744d6e70, ``SRC/element`` and
# ``SRC/material/nD|section``) for every catalogued class and the ND
# materials and shell sections they forward to, and dumped live from build
# 79e0623670:
#
# * continuum stress: ``sigma11..sigma13`` (any index order, e.g. BrickUP's
#   ``sigma13,sigma23``), ``sigma_xx..sigma_zx`` (BezierTet10, BezierTri6),
#   ``sig11`` (FSAM); plane ``sigma11,sigma22,sigma12`` plus ``sigma33``
#   for ``stressPlaneStrain``;
# * continuum strain: ``eps11..`` (3-D), ``eta11..`` (plane, NDMaterial),
#   ``e11,e22,e12`` (VonPapaDamage), ``e1212``, ``eps_xx``/``gamma_xy``
#   (BezierTet10, BezierTri6);
# * shell resultants: ``p11,p22,p12,m11,m22,m12,q1,q2``, with ``p1212`` for
#   the membrane shear on ShellMITC9 / ShellDKGT / the thermal shells; the
#   section spelling ``Fxx,Fyy,Fxy,Mxx,Myy,Mxy,Vxz,Vyz``;
# * shell strains: ``eps11,eps22,gamma12,theta11,theta22,theta33,gamma13,
#   gamma23``; the section spelling ``epsXX,epsYY,epsXY,kappaXX,kappaYY,
#   kappaXY,gammaXZ,gammaYZ``.
#
# A response that names nothing gets MPCO's ``C1..Cn`` (LadrunoConcrete3D
# under ``material.stress``) or NDMaterial's ``UnknownStress`` /
# ``UnknownStrain`` (a material of another width), and those columns take
# the catalog's order. Lookup is case-insensitive.
_XYZ = {"x": "1", "y": "2", "z": "3"}

_SHELL_LABELS: dict[str, frozenset[str]] = {
    "membrane_force_xx": frozenset({"p11", "fxx"}),
    "membrane_force_yy": frozenset({"p22", "fyy"}),
    "membrane_force_xy": frozenset({"p12", "p21", "p1212", "fxy"}),
    "bending_moment_xx": frozenset({"m11", "mxx"}),
    "bending_moment_yy": frozenset({"m22", "myy"}),
    "bending_moment_xy": frozenset({"m12", "m21", "mxy"}),
    "transverse_shear_xz": frozenset({"q1", "q13", "vxz"}),
    "transverse_shear_yz": frozenset({"q2", "q23", "vyz"}),
    "membrane_strain_xx": frozenset({"eps11", "epsxx"}),
    "membrane_strain_yy": frozenset({"eps22", "epsyy"}),
    "membrane_strain_xy": frozenset({"gamma12", "eps12", "epsxy"}),
    "curvature_xx": frozenset({"theta11", "kappaxx"}),
    "curvature_yy": frozenset({"theta22", "kappayy"}),
    "curvature_xy": frozenset({"theta33", "theta12", "kappaxy"}),
    "transverse_shear_strain_xz": frozenset({"gamma13", "gammaxz"}),
    "transverse_shear_strain_yz": frozenset({"gamma23", "gammayz"}),
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
        stems_digit, stems_axis = ("sigma", "sig"), ("sigma_", "s", "stress_")
    else:
        stems_digit = ("eps", "eta", "epsilon", "e") + (
            ("gamma",) if a != b else ())
        stems_axis = ("eps_", "epsilon_", "strain_") + (
            ("gamma_",) if a != b else ())
    return frozenset(
        {s + d for s in stems_digit for d in ij}
        | {s + d + d for s in stems_digit for d in ij}    # e1212, p1212
        | {s + x for s in stems_axis for x in ab}
    )


#: Labels that name nothing: MPCO's ``C<n>`` and NDMaterial's
#: ``UnknownStress`` / ``UnknownStrain`` (MPCO may suffix repeats ``(n)``).
_UNNAMED_LABEL_RE = re.compile(
    r"^(?:C\d+|Unknown(?:Stress|Strain)?(?:\(\d+\))?)$")


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
    if all(_UNNAMED_LABEL_RE.match(lab) for lab in labels):
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


def _warn_unnamed_columns(
    bucket_grp: "h5py.Group", canonical_component: str, exc: ValueError,
) -> None:
    """Name a bucket whose META labels refuse the catalog's names.

    The bucket is left unread and the read goes on with the others: one
    bucket a reader cannot name never fails a read for every class.
    """
    from ._mpco import MPCOBucketSkippedWarning
    warnings.warn(
        f"MPCO gauss read of {canonical_component!r} in "
        f"{Path(bucket_grp.file.filename).name} left 1 bucket(s) unread, so "
        f"their elements return no values: {bucket_grp.name}: {exc}",
        MPCOBucketSkippedWarning, stacklevel=4,
    )


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
        component is not in this bucket's layout, and when the bucket's
        META labels refuse the catalog's names
        (:func:`resolve_gauss_component_names`); that bucket is named in
        an ``MPCOBucketSkippedWarning`` and the other buckets still read.
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
        try:
            layout = resolve_gauss_component_names(
                bucket_grp, layout,
                where=f"{bucket.mpco_group_name}/{bucket.bracket_key}",
            )
        except ValueError as exc:
            _warn_unnamed_columns(bucket_grp, canonical_component, exc)
            return None

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
