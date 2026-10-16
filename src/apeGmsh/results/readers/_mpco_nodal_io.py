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

A bucket the reader is unable to decode is listed by
:func:`scan_nodal_force_buckets` with its reason, and
:func:`warn_skipped_nodal_buckets` names its class
(:class:`NodalForceBucketSkippedWarning`):

- a bucket key that does not parse;
- ``header_idx != 0`` (a second response header for the class);
- an element class absent from :data:`NODAL_FORCE_CATALOG`.
"""
from __future__ import annotations

import warnings
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
    from pathlib import Path

    import h5py

    from .._slabs import NodeSlab


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

class NodalForceBucketSkippedWarning(UserWarning):
    """An MPCO nodal-force bucket the reader is unable to decode: the
    elements it holds read no rows."""


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
        catalog_token)`` is in :data:`NODAL_FORCE_CATALOG` and whose
        ``header_idx`` is 0, at any ``custom_rule_idx``. ``skipped``
        holds every other bucket with the reason it is left unread.
    """
    mapping = canonical_to_nodal_force_token(canonical_component)
    if mapping is None:
        return (None, [], [])
    mpco_group_name, catalog_token = mapping
    if mpco_group_name not in on_elements_grp:
        return (mpco_group_name, [], [])

    token_grp = on_elements_grp[mpco_group_name]
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
        if elem_key.header_idx != 0:
            skipped.append(_SkippedNodalBucket(
                bracket_key, elem_key.class_name,
                f"it is response header {elem_key.header_idx} of the "
                f"class, and only header 0 is decoded",
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
        out.append(_NodalBucket(
            bracket_key=bracket_key,
            elem_key=elem_key,
            layout=layout,
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


@dataclass(frozen=True)
class ModeEntry:
    """One recorded mode: where it lives and its eigen data."""

    step_key: str          # "STEP_<s>" under the modal group's DATA
    mode_key: str          # "MODE_<k>" (k 0-based)
    mode_index: int        # 1-based, as in a native results file
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


#: Marker in the warning :func:`eigen_scalars` raises; the multi-partition
#: readers key on it to warn once per mode for the whole part set.
NON_POSITIVE_EIGENVALUE = "non-positive eigenvalue"


def eigen_scalars(
    lam: float, mode_index: int, source: str,
) -> "tuple[float, float]":
    """``(frequency_hz, period_s)`` from an eigenvalue.

    The ``ParallelModalResult.to_native`` rule: a non-positive eigenvalue (a rigid-body mechanism or an
    unconverged eigensolve) warns and reads as 0 Hz / 0 s rather than
    passing as a valid mode.
    """
    import math
    import warnings

    if lam > 0:
        omega = math.sqrt(lam)
        return omega / (2.0 * math.pi), 2.0 * math.pi / omega
    warnings.warn(
        f"{source}: mode {mode_index} has a {NON_POSITIVE_EIGENVALUE} "
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
    (the names a native results file uses) when the whole file holds one modal
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
# Rotations a node does not have
# =====================================================================
#
# Neither ``.mpco`` nor ``.ladruno`` records a node's ndf. Over a mixed
# model (a tet footing under a shell wall, a quad block beside a 2-D
# frame) the recorder writes 0.0 into the rotation columns of a node
# that has no rotational DOF, and a viewer draws that 0.0 as a field.
# With the model's ``/opensees/nodes_ndf`` attached, the reader writes
# NaN there, as DomainCapture does. Without it the reader warns once
# when the stage holds a continuum element, until the fork records
# per-node ndf in the file.


class AbsentDofWarning(UserWarning):
    """A rotation read without per-node ndf may hold 0.0 at nodes that
    have no rotational DOF (#1971)."""


def load_nodes_ndf(model_h5: "str | Path") -> "dict[int, int] | None":
    """``{node: ndf}`` from ``model_h5``'s ``/opensees/nodes_ndf``.

    ``None`` when the archive predates that group.
    """
    from ...opensees.emitter import h5_reader

    with h5_reader.open(str(model_h5)) as m:
        return m.nodes_ndf()


def _continuum_classes() -> "frozenset[str]":
    """C++ element classes whose nodes may lack rotational DOFs.

    Two derived sources, so a class joins through either table: the
    RESPONSE_CATALOG classes whose ``stress`` is a continuum tensor, and
    the element registry's ``nd``-material entries (by ``cpp_class_name``,
    else the key). The registry adds the u-p family, whose nodes carry a
    pressure DOF in a slot the recorder writes as a rotation.
    """
    from ...opensees._element_capabilities import _ELEM_REGISTRY
    from ...opensees._response_catalog import RESPONSE_CATALOG

    catalog = {
        cls for (cls, _rule, token), layout in RESPONSE_CATALOG.items()
        if token == "stress" and layout.component_layout[0] == "stress_xx"
    }
    registry = {
        spec.cpp_class_name or key for key, spec in _ELEM_REGISTRY.items()
        if spec.mat_family == "nd"
    }
    return frozenset(catalog | registry)


def _has_continuum_class(model_elements: "h5py.Group | None") -> bool:
    """True when ``MODEL/ELEMENTS`` holds a ``<tag>-<Class>[..]`` entry of
    a continuum class."""
    if model_elements is None:
        return False
    solids = _continuum_classes()
    return any(
        key.partition("-")[2].partition("[")[0] in solids
        for key in model_elements
    )


def model_node_ids(stage_groups: "list[h5py.Group]") -> "set[int]":
    """Every node id under the stages' ``MODEL/NODES/ID``."""
    out: "set[int]" = set()
    for grp in stage_groups:
        if "MODEL" in grp and "NODES" in grp["MODEL"] and "ID" in grp["MODEL/NODES"]:
            out.update(int(n) for n in np.asarray(grp["MODEL/NODES/ID"][...]).flat)
    return out


_NO_MODEL = (
    "no per-node ndf is attached. Pass model_h5= (the run's archive, with "
    "/opensees/nodes_ndf) to read NaN at those nodes; a partitioned read "
    "does not attach it yet"
)


class RotationMask:
    """Per-reader NaN mask for rotation columns a node lacks.

    :meth:`attach` takes the model's per-node ndf. :meth:`absent` then
    names the nodes whose DOF layout has no slot for a component, and
    :meth:`apply` writes NaN there. Without an attached ndf, nothing is
    masked, and the first rotation read over a continuum class warns.
    """

    def __init__(self, source: str) -> None:
        self._source = source
        self._nodes_ndf: "dict[int, int] | None" = None
        self._ndm = 3
        self._warned = False
        self._why_unattached = _NO_MODEL

    def attach(
        self, nodes_ndf: "dict[int, int] | None", ndm: int,
        file_nodes: "set[int]",
    ) -> None:
        """Use ``nodes_ndf`` if it describes every node in ``file_nodes``.

        ``None`` is an archive older than ``/opensees/nodes_ndf``. A model
        that misses some of the file's nodes is another model (the bind
        warns ``ModelFemMismatchWarning`` for it). Neither is used; the
        warning then names which it was.
        """
        if nodes_ndf is None:
            self._why_unattached = (
                "model_h5 is an archive older than /opensees/nodes_ndf, so "
                "it holds no per-node ndf. Re-save the model with this "
                "apeGmsh to read NaN at those nodes"
            )
            return
        missing = sorted(file_nodes - {int(n) for n in nodes_ndf})
        if missing:
            self._why_unattached = (
                f"model_h5's /opensees/nodes_ndf misses nodes "
                f"{missing[:10]} of this file, so it is another model's "
                "and its ndf is not used. Pass the archive of the run "
                "that wrote this file"
            )
            return
        self._nodes_ndf = {int(n): int(d) for n, d in nodes_ndf.items()}
        self._ndm = int(ndm)

    def absent(
        self, component: str, node_ids: "ndarray",
        model_elements: "h5py.Group | None",
    ) -> "ndarray | None":
        """``(N,)`` bool: the nodes without ``component``'s DOF, or
        ``None`` when nothing is masked (not a rotation, or no ndf)."""
        from ...opensees._recorder_translate import node_component_spatial

        spatial = node_component_spatial(component)
        if spatial is None or spatial[1] < 3:
            return None                 # not a rotational component
        slot = spatial[1]
        if self._nodes_ndf is None:
            if not self._warned and _has_continuum_class(model_elements):
                self._warned = True
                import warnings
                warnings.warn(
                    f"{self._source}: {component!r} is read over continuum "
                    "elements, whose nodes may have no rotational DOF; the "
                    "recorder writes 0.0 there (or a pressure, at a u-p "
                    "node), so a value may be an absent DOF. "
                    f"{self._why_unattached}. The fork recording per-node "
                    "ndf is the permanent fix (#2015).",
                    AbsentDofWarning, stacklevel=5,
                )
            return None
        from ...opensees._internal.build.recorders import _node_layout

        ids = np.asarray(node_ids, dtype=np.int64)
        missing = sorted({int(n) for n in ids} - self._nodes_ndf.keys())
        if missing:
            raise ValueError(
                f"{self._source}: {component!r} has nodes "
                f"{missing[:10]} that are neither in the file's MODEL/NODES "
                "nor in the model's /opensees/nodes_ndf, so whether their "
                "column is a value or padding is unknown."
            )
        return np.array([
            slot not in _node_layout(self._nodes_ndf[int(n)], self._ndm)
            for n in ids
        ], dtype=bool)

    def apply(
        self, slab: "NodeSlab", model_elements: "h5py.Group | None",
    ) -> "NodeSlab":
        """``slab`` with NaN at the nodes :meth:`absent` names."""
        from dataclasses import replace

        absent = self.absent(slab.component, slab.node_ids, model_elements)
        if absent is None or not absent.any():
            return slab
        values = np.where(absent[None, :], np.nan, slab.values)
        return replace(slab, values=values)


def attach_nodes_ndf(
    reader: object, single: type, model_h5: "str | Path", ndm: int,
) -> None:
    """Attach ``model_h5``'s per-node ndf to ``reader`` (#1971).

    ``single`` is the one-file reader class, which carries a
    :class:`RotationMask`. A partitioned reader is left as it is: its
    part readers have no ndf, so they warn on a rotation read over
    continuum elements (:class:`AbsentDofWarning`).
    """
    if not isinstance(reader, single):
        return
    reader.attach_nodes_ndf(          # type: ignore[attr-defined]
        load_nodes_ndf(model_h5), ndm,
    )


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
