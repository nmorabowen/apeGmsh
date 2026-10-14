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

from ...opensees._response_catalog import (
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
# Modes of vibration — ``ON_NODES/MODES_OF_VIBRATION(U|R)`` (#1946)
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


@dataclass(frozen=True)
class ModeEntry:
    """One recorded mode: where it lives and its eigen data."""

    step_key: str          # "STEP_<s>" under the modal group's DATA
    mode_key: str          # "MODE_<k>" (k 0-based)
    mode_index: int        # 1-based, the ``capture_modes`` convention
    eigenvalue: float      # LAMBDA = omega^2
    frequency_hz: float
    period_s: float


def _modal_groups(on_nodes: "h5py.Group") -> "list[str]":
    from ._mpco_translation import is_modal_result_name
    return [n for n in on_nodes if is_modal_result_name(n)]


def _keyed(group: "h5py.Group", prefix: str) -> "list[str]":
    keys = [k for k in group if k.startswith(prefix)]
    return sorted(keys, key=lambda k: int(k[len(prefix):]))


def _components_of(group: "h5py.Group") -> "list[str]":
    if "COMPONENTS" not in group.attrs:
        return []
    raw = _attr_scalar(group.attrs["COMPONENTS"])
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def eigen_scalars(
    lam: float, mode_index: int, source: str,
) -> "tuple[float, float]":
    """``(frequency_hz, period_s)`` from an eigenvalue.

    The ``DomainCapture.capture_modes`` / ``ParallelModalResult.to_native``
    rule: a non-positive eigenvalue (a rigid-body mechanism or an
    unconverged eigensolve) warns and reads as 0 Hz / 0 s rather than
    passing as a valid mode.
    """
    import math
    import warnings

    if lam > 0:
        omega = math.sqrt(lam)
        return omega / (2.0 * math.pi), 2.0 * math.pi / omega
    warnings.warn(
        f"{source}: mode {mode_index} has a non-positive eigenvalue "
        f"({lam:.6g}); this indicates a spurious/unstable mode (rigid-body "
        "mechanism or unconverged eigensolve). Reading frequency=0 / "
        "period=0 for it.",
        UserWarning,
        stacklevel=3,
    )
    return 0.0, 0.0


def _mode_sets_of(
    data: "h5py.Group", group_name: str, source: str,
) -> "list[tuple[str, list[tuple[str, float]]]]":
    sets: "list[tuple[str, list[tuple[str, float]]]]" = []
    for step_key in _keyed(data, "STEP_"):
        modes: "list[tuple[str, float]]" = []
        for mode_key in _keyed(data[step_key], "MODE_"):
            attrs = data[step_key][mode_key].attrs
            if "LAMBDA" not in attrs:
                raise ValueError(
                    f"{source}: ON_NODES/{group_name}/DATA/{step_key}/"
                    f"{mode_key} has no LAMBDA attribute; the eigenvalue "
                    "of this mode is not recorded."
                )
            modes.append((mode_key, float(_attr_scalar(attrs["LAMBDA"]))))
        if modes:
            sets.append((step_key, modes))
    return sets


def discover_modes(
    on_nodes: "h5py.Group", *, source: str,
) -> "list[tuple[str, list[ModeEntry]]]":
    """Every modal set under ``on_nodes``, as ``[(step_key, [ModeEntry])]``.

    Sets are ordered by step, modes by index. The translational group is
    the reference when present; the rotational group, when both exist,
    must list the same steps, modes and eigenvalues, or the file is
    refused. A ``MODE_<k>`` without a ``LAMBDA`` attribute is refused too:
    a mode stage never invents its eigenvalue.
    """
    names = sorted(_modal_groups(on_nodes), key=lambda n: (not n.endswith("(U)"), n))
    if not names:
        return []
    per_group = []
    for name in names:
        if "DATA" not in on_nodes[name]:
            raise ValueError(
                f"{source}: ON_NODES/{name} has no DATA group; the modal "
                "record is incomplete."
            )
        per_group.append(_mode_sets_of(on_nodes[name]["DATA"], name, source))
    for name, other in zip(names[1:], per_group[1:]):
        if other != per_group[0]:
            raise ValueError(
                f"{source}: ON_NODES/{name} lists different modes or "
                f"eigenvalues than ON_NODES/{names[0]}; the modal record is "
                "inconsistent."
            )
    out: "list[tuple[str, list[ModeEntry]]]" = []
    for step_key, modes in per_group[0]:
        entries = []
        for mode_key, lam in modes:
            index = int(mode_key[len("MODE_"):]) + 1
            f_hz, t_s = eigen_scalars(lam, index, source)
            entries.append(ModeEntry(
                step_key=step_key, mode_key=mode_key, mode_index=index,
                eigenvalue=lam, frequency_hz=f_hz, period_s=t_s,
            ))
        out.append((step_key, entries))
    return out


def modal_components(on_nodes: "h5py.Group") -> "list[str]":
    """Canonical mode-shape components the modal groups carry."""
    from ._mpco_translation import canonical_modal_component

    out: "set[str]" = set()
    for name in _modal_groups(on_nodes):
        for label in _components_of(on_nodes[name]):
            canon = canonical_modal_component(name, label)
            if canon is not None:
                out.add(canon)
    return sorted(out)


def read_mode_shape(
    on_nodes: "h5py.Group",
    entry: ModeEntry,
    component: str,
    *,
    node_ids: "ndarray | None" = None,
) -> "tuple[ndarray, ndarray] | None":
    """``(values[nSel], node_ids[nSel])`` of one mode's ``component``.

    ``None`` when no modal group carries ``component`` for this mode.
    """
    from ._mpco_translation import canonical_modal_component

    for name in _modal_groups(on_nodes):
        grp = on_nodes[name]
        cols = [
            i for i, lab in enumerate(_components_of(grp))
            if canonical_modal_component(name, lab) == component
        ]
        if not cols:
            continue
        data = grp["DATA"]
        if entry.step_key not in data or entry.mode_key not in data[entry.step_key]:
            return None
        ids = np.asarray(grp["ID"][...], dtype=np.int64).flatten()
        rows = np.arange(ids.size)
        if node_ids is not None:
            rows = np.flatnonzero(
                np.isin(ids, np.asarray(node_ids, dtype=np.int64)),
            )
        block = data[entry.step_key][entry.mode_key]
        col = np.asarray(block[:, cols[0]], dtype=np.float64)
        return col[rows], ids[rows]
    return None


def mode_stage_layout(
    host_ids: "list[str]",
    host_names: "list[str]",
    host_sets: "list[list[tuple[str, list[ModeEntry]]]]",
) -> "list[tuple[int, str, str, ModeEntry]]":
    """``(host_ordinal, stage_id, name, entry)`` for every recorded mode.

    Ids are ``<host_id>_mode_<k>``, or ``<host_id>_step_<s>_mode_<k>``
    when the host holds more than one modal set. Names are ``mode_<k>``
    (the ``capture_modes`` names) when the whole file holds one modal
    set, else ``<host name>/STEP_<s>/mode_<k>`` so every name is unique.
    """
    n_sets = sum(len(sets) for sets in host_sets)
    out: "list[tuple[int, str, str, ModeEntry]]" = []
    for h, (hid, hname, sets) in enumerate(zip(host_ids, host_names, host_sets)):
        for step_key, entries in sets:
            for e in entries:
                if len(sets) == 1:
                    sid = f"{hid}_mode_{e.mode_index}"
                else:
                    sid = f"{hid}_{step_key.lower()}_mode_{e.mode_index}"
                name = (
                    f"mode_{e.mode_index}" if n_sets == 1
                    else f"{hname}/{step_key}/mode_{e.mode_index}"
                )
                out.append((h, sid, name, e))
    return out


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
