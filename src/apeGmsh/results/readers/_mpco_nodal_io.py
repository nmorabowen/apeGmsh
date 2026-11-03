"""Nodal-forces element-level result decoding for MPCOReader (Phase 11b).

MPCO stores per-element-node force vectors for line elements (the
closed-form ``ElasticBeam{2d,3d}``, ``ElasticTimoshenkoBeam{2d,3d}``,
``ModElasticBeam2d``, and the frame beams ``ForceBeamColumn``,
``DispBeamColumn`` and ``LadrunoDispBeamColumn``) under
``MODEL_STAGE[…]/RESULTS/ON_ELEMENTS/<frame>/<bucket_key>/`` where
``<frame>`` is ``globalForce`` or ``localForce`` and
``<bucket_key>`` is ``<classTag>-<ClassName>[<rule>:<cust>:<hdr>]``.
Each bucket holds the standard ``META`` / ``ID`` / ``DATA/STEP_<k>``
triplet, with these conventions:

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

A frame beam's bucket carries a custom-rule key
(``74-ForceBeamColumn3d[1000:1:0]``): its integration rule names the
sections, never the end-force layout, so any ``custom_rule_idx`` is read
and META validation guards the layout.

Every header index is read (``:0``, ``:1``, ...). MPCO opens a new
header when elements of one class write a different META, so a
``header_idx > 0`` bucket is read only once its own META spells the
catalog layout: its ``NUM_COLUMNS`` and META blocks
(:func:`validate_nodal_bucket_meta`), and its ``META/COMPONENTS``
column names, node count and order
(:func:`header_bucket_meta_mismatch`).

A bucket the reader is unable to decode is listed by
:func:`scan_nodal_force_buckets` with its reason, and
:func:`warn_skipped_nodal_buckets` names its class
(:class:`NodalForceBucketSkippedWarning`):

- a bucket key that does not parse;
- an element class absent from :data:`NODAL_FORCE_CATALOG`;
- a ``header_idx > 0`` bucket whose META does not spell the catalog
  layout, or carries no column names to check it by.

MPCO writes each element once, to exactly one bucket. A read that would
return one element id twice, from two buckets (header indices, custom
rules or classes) or from one bucket's ``ID``, raises
:class:`~._mpco_refusals.MPCOElementInTwoBuckets` (#2173). The check
covers the ids the read returns, as :func:`read_nodal_bucket_slab`
selects them, and reads no ``ID`` a second time.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

from ...opensees._response_catalog import (
    MPCOElementKey,
    NodalForceLayout,
    gauss_routing_for_canonical,
    is_nodal_force_catalogued,
    lookup_nodal_force,
    parse_mpco_element_key,
    unflatten_nodal,
)
from ._buckets import NodalForceBucketSkippedWarning, _attr_scalar
from ._ladruno_element_io import (
    decode_local_nodal_force_columns,
    decode_nodal_force_columns,
)
from ._mpco_refusals import refuse_element_in_two_buckets

if TYPE_CHECKING:
    from collections.abc import Callable

    import h5py

    from typing import TypeAlias

    from ...opensees._readback import RotationRoles

    #: What :meth:`RotationMask.attach` takes for the model's DOF roles:
    #: called once, on the first read that needs them.
    RolesSource: TypeAlias = "Callable[[], RotationRoles]"


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

class _ReturnedIds:
    """The element ids one read has returned so far, by bucket path.

    Every bucket of one :func:`scan_nodal_force_buckets` shares one
    ledger, so the buckets of one scan serve one read.
    :func:`read_nodal_bucket_slab` adds each bucket's selected ids, and
    an id seen twice raises
    :class:`~._mpco_refusals.MPCOElementInTwoBuckets`.
    """

    def __init__(self) -> None:
        self._by_path: dict[str, ndarray] = {}
        self._seen = np.empty(0, dtype=np.int64)

    def add(self, bucket_grp: "h5py.Group", ids: ndarray) -> None:
        path = bucket_grp.name
        if path in self._by_path:
            raise RuntimeError(
                f"MPCO bucket {path} was read twice from one scan; "
                f"scan_nodal_force_buckets again for each read."
            )
        file = bucket_grp.file.filename
        refuse_element_in_two_buckets({path: ids}, file=file)
        if self._seen.size and np.isin(ids, self._seen).any():
            refuse_element_in_two_buckets(
                {**self._by_path, path: ids}, file=file,
            )
        self._by_path[path] = ids
        self._seen = np.concatenate([self._seen, ids])


@dataclass(frozen=True)
class _NodalBucket:
    """One ``ON_ELEMENTS/<frame>/<bracket_key>`` group + its catalog layout."""
    bracket_key: str
    elem_key: MPCOElementKey
    layout: NodalForceLayout
    #: Shared by every bucket of one scan: the ids its read has returned.
    returned: _ReturnedIds = field(compare=False, repr=False)


# =====================================================================
# Bucket discovery
# =====================================================================


@dataclass(frozen=True)
class _SkippedNodalBucket:
    """One ``ON_ELEMENTS/<frame>/<bracket_key>`` group the reader leaves unread."""
    bracket_key: str
    class_name: str        # the raw key when the key does not parse
    reason: str


def scan_nodal_force_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_NodalBucket], list[_SkippedNodalBucket]]:
    """Walk ``ON_ELEMENTS/<globalForce|localForce>/``, sorting every bucket.

    Returns
    -------
    (mpco_group_name, buckets, skipped)
        ``mpco_group_name`` is the ON_ELEMENTS child name we looked
        under (``"globalForce"`` / ``"localForce"``), or ``None`` if
        the canonical component has no nodal-forces routing.
        ``buckets`` holds the buckets whose ``(class_name,
        catalog_token)`` is in :data:`NODAL_FORCE_CATALOG`, at any
        ``custom_rule_idx``: every ``header_idx`` 0 bucket, and each
        ``header_idx > 0`` bucket whose own META spells the catalog
        layout (:func:`header_bucket_meta_mismatch`). ``skipped``
        holds every other bucket with the reason it is left unread.
        The buckets share one :class:`_ReturnedIds`, so they serve one
        read: :func:`read_nodal_bucket_slab` refuses an element id they
        return twice.
    """
    mapping = canonical_to_nodal_force_token(canonical_component)
    if mapping is None:
        return (None, [], [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [], [])

    token_grp = on_elements_grp[mpco_group_name]
    returned = _ReturnedIds()
    out: list[_NodalBucket] = []
    skipped: list[_SkippedNodalBucket] = []
    for bracket_key in token_grp:
        try:
            elem_key = parse_mpco_element_key(bracket_key)
        except ValueError:
            skipped.append(_SkippedNodalBucket(
                bracket_key, bracket_key,
                "its bucket key does not parse as "
                "'<tag>-<Class>[<rule>:<cust>:<hdr>]'",
            ))
            continue
        if not is_nodal_force_catalogued(elem_key.class_name, catalog_token):
            skipped.append(_SkippedNodalBucket(
                bracket_key, elem_key.class_name,
                f"({elem_key.class_name!r}, {catalog_token!r}) is absent "
                f"from NODAL_FORCE_CATALOG",
            ))
            continue
        layout = lookup_nodal_force(elem_key.class_name, catalog_token)
        if elem_key.header_idx != 0:
            mismatch = header_bucket_meta_mismatch(
                token_grp[bracket_key], layout,
                class_name=elem_key.class_name, bracket_key=bracket_key,
            )
            if mismatch is not None:
                skipped.append(_SkippedNodalBucket(
                    bracket_key, elem_key.class_name,
                    f"it is response header {elem_key.header_idx} of the "
                    f"class and {mismatch}",
                ))
                continue
        out.append(_NodalBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            layout=layout,
            returned=returned,
        ))
    return (mpco_group_name, out, skipped)


def discover_nodal_force_buckets(
    on_elements_grp: "h5py.Group",
    *,
    canonical_component: str,
) -> tuple[str | None, list[_NodalBucket]]:
    """The readable buckets of :func:`scan_nodal_force_buckets`.

    For listings (``available_components``); a read uses the scan, so it
    can warn about the buckets it leaves unread.
    """
    token, buckets, _skipped = scan_nodal_force_buckets(
        on_elements_grp, canonical_component=canonical_component,
    )
    return (token, buckets)


def warn_skipped_nodal_buckets(
    token_grp: "h5py.Group",
    skipped: list[_SkippedNodalBucket],
    *,
    component: str,
    element_ids: ndarray | None,
    source: str,
) -> None:
    """Warn once per unread bucket that holds a requested element.

    ``element_ids`` are the requested OpenSees tags (``None`` requests
    every element). A bucket whose ``ID`` holds none of them is
    irrelevant to the read and stays quiet.
    """
    for s in skipped:
        grp = token_grp[s.bracket_key]
        if element_ids is not None and "ID" in grp:
            ids = np.asarray(grp["ID"][...]).ravel().astype(np.int64)
            if not np.isin(ids, np.asarray(element_ids, dtype=np.int64)).any():
                continue
        warnings.warn(
            f"{source}: the MPCO bucket {token_grp.name}/{s.bracket_key} "
            f"(element class {s.class_name!r}) is left unread because "
            f"{s.reason}; its elements read no {component!r} rows.",
            NodalForceBucketSkippedWarning, stacklevel=3,
        )


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

    meta = bucket_grp["META"] if "META" in bucket_grp else None
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


def _meta_column_names(bucket_grp: "h5py.Group") -> list[str] | None:
    """The column names of a single-block bucket's ``META/COMPONENTS``.

    ``None`` when META carries no ``COMPONENTS``. Raises
    :class:`ValueError` when it holds other than one entry or one block.
    """
    meta = bucket_grp["META"]
    if "COMPONENTS" not in meta:
        return None
    raw = np.asarray(meta["COMPONENTS"][...]).flatten()
    if raw.size != 1:
        raise ValueError(
            f"its META/COMPONENTS holds {raw.size} entries, not 1"
        )
    text = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
    blocks = [seg for seg in text.split(";") if seg]
    if len(blocks) != 1:
        raise ValueError(
            f"its META/COMPONENTS names {len(blocks)} blocks, not 1"
        )
    return [
        n.strip() for n in blocks[0].rpartition(".")[2].split(",")
        if n.strip()
    ]


def header_bucket_meta_mismatch(
    bucket_grp: "h5py.Group",
    layout: NodalForceLayout,
    *,
    class_name: str,
    bracket_key: str,
) -> str | None:
    """Why a ``header_idx > 0`` bucket's own META does not spell ``layout``.

    ``None`` when it does: ``NUM_COLUMNS`` and the META blocks pass
    :func:`validate_nodal_bucket_meta`, and the ``META/COMPONENTS`` names
    decode, column by column, to ``layout``'s components at each of its
    ``n_nodes_per_element`` nodes, node-slowest in catalog order. A
    ``globalForce`` name is decoded as the ``.ladruno`` reader decodes it
    (``Px_1``, ``Mz_2``: :func:`decode_nodal_force_columns`), and a
    ``localForce`` name likewise (``N_1``, ``Vy_1``, the 2-D ``V_1`` or
    ``V1``: :func:`decode_local_nodal_force_columns`).

    Otherwise the reason, for :class:`NodalForceBucketSkippedWarning`:
    a META the catalog contradicts, or a META with no column names to
    check (no ``COMPONENTS``, or MPCO's unnamed ``C1..Cn``), which leaves
    the order of its columns unknown.
    """
    try:
        validate_nodal_bucket_meta(bucket_grp, layout, bracket_key=bracket_key)
        names = _meta_column_names(bucket_grp)
    except (KeyError, ValueError) as exc:
        return f"its META contradicts the catalog layout: {exc}"
    expected = list(layout.component_layout)
    n_nodes = layout.n_nodes_per_element
    where = (
        f"the {n_nodes}-node {layout.frame} layout NODAL_FORCE_CATALOG "
        f"gives {class_name!r} ({expected} at each node)"
    )
    if names is None:
        return (
            f"its META carries no COMPONENTS, so its columns cannot be "
            f"checked against {where}"
        )
    if len(names) != layout.flat_size_per_element:
        return (
            f"its META names {len(names)} columns {names[:12]}, not the "
            f"{layout.flat_size_per_element} of {where}"
        )
    if layout.frame == "local":
        try:
            decode_local_nodal_force_columns(names, class_name)
        except ValueError as exc:
            return f"its META column names do not match the catalog: {exc}"
        return None
    try:
        columns, undecoded = decode_nodal_force_columns(
            names, ndm=2 if layout.n_components_per_node == 3 else 3,
        )
    except ValueError as exc:
        return f"its META column names {names[:12]} do not decode: {exc}"
    k_per_node = layout.n_components_per_node
    nodes = np.arange(n_nodes, dtype=np.int64)
    if undecoded or set(columns) != set(expected) or any(
        not np.array_equal(columns[c], nodes * k_per_node + k)
        for k, c in enumerate(expected)
    ):
        return f"its META columns {names[:12]} do not spell {where}"
    return None


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

    Raises
    ------
    MPCOElementInTwoBuckets
        When an element id this read returns was already returned by
        another bucket of the same scan, or repeats in this bucket's
        ``ID``.
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
    bucket.returned.add(bucket_grp, sel_ids)

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
# Modes of vibration — ``ON_NODES/MODES_OF_VIBRATION(U|R)``
# =====================================================================
#
# Both the MPCO recorder and the fork's Ladruno recorder write the same
# modal layout (measured on fork build 79e0623670):
#
#   <stage>/RESULTS/ON_NODES/MODES_OF_VIBRATION(U)/
#       ID                        (nNodes, 1)
#       DATA/STEP_<s>/            group, attrs STEP, TIME
#           MODE_<k>              (nNodes, nComp), k 0-based, attrs
#                                 MODE, LAMBDA, OMEGA, FREQUENCY, PERIOD
#
# ``MODES_OF_VIBRATION(R)`` holds the rotational columns in the same shape.
# ``STEP_<s>`` is the commit step at which the recorder saw a new eigen
# solution, so one stage may hold more than one modal set.


# =====================================================================
# Rotations a node does not have
# =====================================================================
#
# Neither ``.mpco`` nor ``.ladruno`` records a node's ndf. Over a mixed
# model (a tet footing under a shell wall, a quad block beside a 2-D
# frame) the recorder writes 0.0 into the rotation columns of a node
# that has no rotational DOF, and a viewer draws that 0.0 as a field.
# With the model's ``/opensees/nodes_ndf`` attached, the reader writes
# NaN there. Without it the reader warns once
# when the stage holds a continuum element, until the fork records
# per-node ndf in the file.


# =====================================================================
# Internal helpers
# =====================================================================

