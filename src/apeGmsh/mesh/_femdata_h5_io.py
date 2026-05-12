"""Broker-side neutral-zone writer for ``model.h5``.

Phase 8.5 makes the :class:`apeGmsh.mesh.FEMData` broker write the
neutral-zone groups that the master plan
([architecture/phase-8-untangle.md §3](../opensees/architecture/phase-8-untangle.md))
places at the root of ``model.h5``:

* ``/meta``                — file-level metadata (schema_version, ndm,
                             snapshot_id, …).
* ``/nodes``               — ids, coords.
* ``/elements/{type}``     — per-type ids + connectivity.
* ``/physical_groups``     — top-level index for viewer discovery.
* ``/labels``              — apeGmsh-internal labels.
* ``/constraints/{kind}``  — MP-style records, symmetric compound shape.
* ``/loads/{kind}/{pattern}``  — per-pattern load records.
* ``/masses``              — per-node mass vectors.

The companion ``mesh/_femdata_native_io.py`` writes a FEMData snapshot
under a ``/model/`` SUB-group inside results files — different layout,
different consumer (master plan §7 Q2: "Keep both").  ``_femdata_h5_io``
targets the ROOT of a fresh model.h5; ``_femdata_native_io`` targets a
named sub-group inside an existing results file.

Public entry points:

* :func:`write_fem_h5` — open a fresh file at ``path``, write meta +
  neutral zone, close.  This is what ``FEMData.to_h5(path)`` delegates
  to.
* :func:`write_neutral_zone` — write the seven neutral-zone groups
  into an already-open :class:`h5py.File`.  Used by the bridge in
  Phase 8.5 commit 4 to compose neutral + ``/opensees/`` in one file.
* :func:`write_meta` — write ``/meta`` attrs.  Caller-owned so the
  bridge can stamp its own ``schema_version`` / ``ndf`` while the
  broker fills in the geometry-derived attrs.

Reader-side helpers live in
:mod:`apeGmsh.opensees.emitter.h5_reader` (Phase 8.5 commit 3 extends
it with typed accessors for the new groups).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

from ._record_h5 import (
    element_load_payload_dtype,
    interpolation_payload_dtype,
    make_record_dtype,
    mass_payload_dtype,
    node_group_payload_dtype,
    node_pair_payload_dtype,
    node_to_surface_payload_dtype,
    nodal_load_payload_dtype,
    sp_payload_dtype,
    surface_coupling_payload_dtype,
)

if TYPE_CHECKING:
    from .FEMData import FEMData


__all__ = [
    "NEUTRAL_SCHEMA_VERSION",
    "write_fem_h5",
    "write_meta",
    "write_neutral_zone",
]


#: Schema version stamped by :func:`write_fem_h5` and the standalone
#: ``FEMData.to_h5(path)`` flow.  Phase 8.5 added the neutral zone
#: (`2.0.0 → 2.1.0`); Phase 8.6 added the ``fem_eids`` dataset under
#: each ``/opensees/element_meta/{type_token}/`` group
#: (`2.1.0 → 2.2.0`).  Broker-only files (no `/opensees/...`) still
#: stamp the current minor — the field is additive and old readers
#: tolerate its absence.
NEUTRAL_SCHEMA_VERSION: str = "2.2.0"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def write_fem_h5(
    fem: "FEMData",
    path: str,
    *,
    schema_version: str = NEUTRAL_SCHEMA_VERSION,
    model_name: str = "",
    apegmsh_version: str = "",
    ndf: int = 0,
) -> None:
    """Write a fresh ``model.h5`` with the neutral zone.

    No ``/opensees/`` content is emitted — absent enrichment is the
    right "no solver" signal.
    """
    import h5py

    with h5py.File(path, "w") as f:
        write_meta(
            fem, f,
            schema_version=schema_version,
            model_name=model_name,
            apegmsh_version=apegmsh_version,
            ndf=ndf,
        )
        write_neutral_zone(fem, f)


def write_meta(
    fem: "FEMData",
    f: Any,
    *,
    schema_version: str,
    model_name: str = "",
    apegmsh_version: str = "",
    ndf: int = 0,
) -> None:
    """Create ``/meta`` and stamp the file-level attrs.

    Caller-owned so the bridge can supply its own ``ndf`` /
    ``schema_version``.  Broker-only writes pass ``ndf=0``.
    """
    meta = f.create_group("meta")
    meta.attrs["schema_version"] = schema_version
    meta.attrs["apeGmsh_version"] = apegmsh_version
    meta.attrs["created_iso"] = datetime.now(tz=timezone.utc).isoformat()
    meta.attrs["ndm"] = int(_derive_ndm(fem))
    meta.attrs["ndf"] = int(ndf)
    meta.attrs["snapshot_id"] = str(fem.snapshot_id)
    meta.attrs["model_name"] = str(model_name)


def write_neutral_zone(fem: "FEMData", f: Any) -> None:
    """Write the seven neutral-zone groups into an open HDF5 file.

    Does NOT write ``/meta`` — the caller owns that, so the bridge
    can stamp its own ``schema_version`` / ``ndf`` while the broker
    just contributes geometry.
    """
    _write_nodes(fem, f)
    _write_elements(fem, f)
    _write_physical_groups(fem, f)
    _write_labels(fem, f)
    _write_constraints(fem, f)
    _write_loads(fem, f)
    _write_masses(fem, f)


# ---------------------------------------------------------------------------
# Per-group writers
# ---------------------------------------------------------------------------


def _derive_ndm(fem: "FEMData") -> int:
    """Best-effort spatial dimension from the broker's element types."""
    try:
        dims = [int(t.dim) for t in fem.info.types]
        if dims:
            return max(dims)
    except (AttributeError, ValueError):
        pass
    return 3


def _write_nodes(fem: "FEMData", f: Any) -> None:
    """Write ``/nodes/{ids, coords}`` from ``fem.nodes``."""
    nodes_grp = f.create_group("nodes")
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    node_coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    nodes_grp.create_dataset("ids", data=node_ids)
    nodes_grp.create_dataset("coords", data=node_coords)


def _write_elements(fem: "FEMData", f: Any) -> None:
    """Write ``/elements/{type}/{ids, connectivity}`` per element type.

    ``{type}`` is the broker's GMSH-style alias (``tet4``, ``hex8``,
    ``triangle3``, …).  These deliberately do NOT match the bridge's
    OpenSees type tokens (``forceBeamColumn``, ``FourNodeTetrahedron``);
    the two namespaces serve different consumers and live in
    different zones (root vs ``/opensees/element_meta``).
    """
    elements_grp = f.create_group("elements")
    for elem_group in fem.elements:
        if elem_group.ids.size == 0:
            continue
        type_name = elem_group.type_name.replace("/", "_")
        sub = elements_grp.create_group(type_name)
        et = elem_group.element_type
        sub.attrs["code"] = int(et.code)
        sub.attrs["gmsh_name"] = str(et.gmsh_name)
        sub.attrs["npe"] = int(et.npe)
        sub.attrs["dim"] = int(et.dim)
        sub.attrs["order"] = int(et.order)
        sub.create_dataset("ids", data=np.asarray(elem_group.ids, dtype=np.int64))
        sub.create_dataset(
            "connectivity",
            data=np.asarray(elem_group.connectivity, dtype=np.int64),
        )


def _write_physical_groups(fem: "FEMData", f: Any) -> None:
    """Write ``/physical_groups/{name}/{node_ids, node_coords, element_ids}``.

    Union of node-side and element-side PG taxonomies — each (dim, tag)
    pair appears once with both ``node_ids`` and ``element_ids`` (when
    the dim supports elements).  Omitted entirely if neither side
    declared any PGs.
    """
    _write_named_index_at_root(
        fem, f, group_name="physical_groups",
        node_side=getattr(fem.nodes, "physical", None),
        element_side=getattr(fem.elements, "physical", None),
    )


def _write_labels(fem: "FEMData", f: Any) -> None:
    """Write ``/labels/{name}/{node_ids, node_coords, element_ids}``.

    Same shape as ``/physical_groups``; the only difference is the
    source-side label set.
    """
    _write_named_index_at_root(
        fem, f, group_name="labels",
        node_side=getattr(fem.nodes, "labels", None),
        element_side=getattr(fem.elements, "labels", None),
    )


def _write_named_index_at_root(
    fem: "FEMData",
    f: Any,
    *,
    group_name: str,
    node_side: Any,
    element_side: Any,
) -> None:
    """Combine node-side + element-side named groups under a root index."""
    node_keys = _safe_get_all(node_side)
    elem_keys = _safe_get_all(element_side)
    all_keys = list(dict.fromkeys(node_keys + elem_keys))
    if not all_keys:
        return

    parent = f.create_group(group_name)
    seen_safe: set[str] = set()
    for dim, tag in all_keys:
        name = _safe_get_name(node_side, dim, tag) or _safe_get_name(
            element_side, dim, tag,
        ) or f"_unnamed_{dim}_{tag}"
        safe = name.replace("/", "_")
        if safe in seen_safe:
            safe = f"{safe}__{dim}_{tag}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = int(dim)
        sub.attrs["tag"] = int(tag)
        sub.attrs["name"] = name

        nids, ncoords = _named_node_arrays(node_side, dim, tag)
        sub.create_dataset("node_ids", data=nids)
        sub.create_dataset("node_coords", data=ncoords)

        if dim >= 1:
            eids = _named_element_ids(element_side, dim, tag)
            if eids.size > 0:
                sub.create_dataset("element_ids", data=eids)


def _safe_get_all(group_set: Any) -> list[tuple[int, int]]:
    if group_set is None:
        return []
    try:
        keys = group_set.get_all()
    except (AttributeError, TypeError):
        return []
    return [(int(d), int(t)) for d, t in keys]


def _safe_get_name(group_set: Any, dim: int, tag: int) -> str | None:
    if group_set is None:
        return None
    try:
        name = group_set.get_name(dim, tag)
    except (AttributeError, KeyError, ValueError):
        return None
    return None if name is None else str(name)


def _named_node_arrays(
    group_set: Any, dim: int, tag: int,
) -> tuple[np.ndarray, np.ndarray]:
    if group_set is None:
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
        )
    try:
        nids = np.asarray(group_set.node_ids((dim, tag)), dtype=np.int64)
        ncoords = np.asarray(
            group_set.node_coords((dim, tag)), dtype=np.float64,
        )
    except (KeyError, ValueError, AttributeError):
        return (
            np.array([], dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
        )
    return nids, ncoords


def _named_element_ids(group_set: Any, dim: int, tag: int) -> np.ndarray:
    if group_set is None:
        return np.array([], dtype=np.int64)
    try:
        eids = np.asarray(
            group_set.element_ids((dim, tag)), dtype=np.int64,
        )
    except (KeyError, ValueError, AttributeError):
        return np.array([], dtype=np.int64)
    return eids


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def _write_constraints(fem: "FEMData", f: Any) -> None:
    """Write ``/constraints/{kind}`` datasets using the symmetric compound.

    Iterates the broker's node-side and element-side constraint
    composites separately, binning each record by ``kind``.  Per-kind
    datasets use a per-record-type payload dtype from
    :mod:`apeGmsh.mesh._record_h5`.
    """
    from .records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
        SurfaceCouplingRecord,
    )

    by_kind: dict[str, list[Any]] = {}

    def _bucket(rec: Any) -> None:
        kind = getattr(rec, "kind", None)
        if kind is None:
            return
        by_kind.setdefault(str(kind), []).append(rec)

    node_constraints = getattr(fem.nodes, "constraints", None)
    if node_constraints is not None:
        for rec in node_constraints:
            _bucket(rec)
    elem_constraints = getattr(fem.elements, "constraints", None)
    if elem_constraints is not None:
        for rec in elem_constraints:
            _bucket(rec)

    if not by_kind:
        return

    parent = f.create_group("constraints")
    for kind, records in by_kind.items():
        safe_kind = kind.replace("/", "_")
        first = records[0]
        if isinstance(first, NodePairRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_pair_payload_dtype(), _encode_node_pair,
                target_kind="node",
            )
        elif isinstance(first, NodeGroupRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_group_payload_dtype(), _encode_node_group,
                target_kind="node",
            )
        elif isinstance(first, InterpolationRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                interpolation_payload_dtype(), _encode_interpolation,
                target_kind="node",
            )
        elif isinstance(first, SurfaceCouplingRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                surface_coupling_payload_dtype(),
                _encode_surface_coupling,
                target_kind="element",
            )
        elif isinstance(first, NodeToSurfaceRecord):
            _write_kind_dataset(
                parent, safe_kind, kind, records,
                node_to_surface_payload_dtype(),
                _encode_node_to_surface,
                target_kind="node",
            )
        else:
            # Unknown record type — preserve the kind name but log via
            # an attribute so consumers can detect we lost detail.
            sub = parent.create_group(safe_kind)
            sub.attrs["__deviation__"] = (
                f"unrecognised record type {type(first).__name__}; "
                f"{len(records)} records skipped"
            )


def _write_kind_dataset(
    parent: Any,
    safe_kind: str,
    kind_label: str,
    records: list[Any],
    payload_dtype: np.dtype,
    encode_payload: Any,
    *,
    target_kind: str,
) -> None:
    """Build the symmetric-compound rows for one kind and emit the dataset."""
    outer = make_record_dtype(payload_dtype)
    rows = np.empty(len(records), dtype=outer)
    for i, rec in enumerate(records):
        rows[i] = (
            target_kind,
            _target_for(rec, target_kind),
            kind_label,
            encode_payload(rec),
        )
    parent.create_dataset(safe_kind, data=rows)


def _target_for(rec: Any, target_kind: str) -> str:
    """Best-effort string identifier for ``target`` (per symmetric contract)."""
    if target_kind == "node":
        for attr in ("slave_node", "master_node"):
            v = getattr(rec, attr, None)
            if v is not None:
                return str(int(v))
    elif target_kind == "element":
        # Surface coupling: pick the first slave node as a stand-in
        # identifier (no single "element id" applies — the constraint
        # spans many).  Consumers walk the payload for full info.
        slaves = getattr(rec, "slave_nodes", None)
        if slaves:
            return str(int(slaves[0]))
    return ""


def _encode_node_pair(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    offset = rec.offset
    offset_arr: tuple[float, ...]
    if offset is None:
        offset_arr = (nan, nan, nan)
    else:
        offset_arr = tuple(float(x) for x in np.asarray(offset).reshape(-1)[:3])
    penalty = float(rec.penalty_stiffness) if rec.penalty_stiffness is not None else nan
    return (
        int(rec.master_node),
        int(rec.slave_node),
        np.asarray(rec.dofs, dtype=np.int64),
        offset_arr,
        penalty,
    )


def _encode_node_group(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    offsets = rec.offsets
    if offsets is None:
        offsets_flat = np.array([], dtype=np.float64)
    else:
        offsets_flat = np.asarray(offsets, dtype=np.float64).reshape(-1)
    plane = rec.plane_normal
    plane_arr: tuple[float, ...]
    if plane is None:
        plane_arr = (nan, nan, nan)
    else:
        plane_arr = tuple(float(x) for x in np.asarray(plane).reshape(-1)[:3])
    return (
        int(rec.master_node),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.dofs, dtype=np.int64),
        offsets_flat,
        plane_arr,
    )


def _encode_interpolation(rec: Any) -> tuple[Any, ...]:
    nan = float("nan")
    weights = rec.weights
    if weights is None:
        weights_arr = np.array([], dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    pp = rec.projected_point
    pp_arr = (
        tuple(float(x) for x in np.asarray(pp).reshape(-1)[:3])
        if pp is not None else (nan, nan, nan)
    )
    pc = rec.parametric_coords
    pc_arr = (
        tuple(float(x) for x in np.asarray(pc).reshape(-1)[:2])
        if pc is not None else (nan, nan)
    )
    return (
        int(rec.slave_node),
        np.asarray(rec.master_nodes, dtype=np.int64),
        weights_arr,
        np.asarray(rec.dofs, dtype=np.int64),
        pp_arr,
        pc_arr,
    )


def _encode_surface_coupling(rec: Any) -> tuple[Any, ...]:
    op = rec.mortar_operator
    op_shape: tuple[int, ...]
    if op is None:
        op_arr = np.array([], dtype=np.float64)
        op_shape = (0, 0)
    else:
        op_np = np.asarray(op, dtype=np.float64)
        op_shape = tuple(int(s) for s in op_np.shape[:2])
        if len(op_shape) < 2:
            op_shape = (op_shape[0] if op_shape else 0, 0)
        op_arr = op_np.reshape(-1)
    return (
        np.asarray(rec.master_nodes, dtype=np.int64),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.dofs, dtype=np.int64),
        op_shape,
        op_arr,
    )


def _encode_node_to_surface(rec: Any) -> tuple[Any, ...]:
    coords = rec.phantom_coords
    if coords is None:
        coords_flat = np.array([], dtype=np.float64)
    else:
        coords_flat = np.asarray(coords, dtype=np.float64).reshape(-1)
    return (
        int(rec.master_node),
        np.asarray(rec.slave_nodes, dtype=np.int64),
        np.asarray(rec.phantom_nodes, dtype=np.int64),
        coords_flat,
        np.asarray(rec.dofs, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------


def _write_loads(fem: "FEMData", f: Any) -> None:
    """Write ``/loads/{kind}/{pattern}`` per pattern + kind.

    Nodal loads land under ``/loads/nodal/{pattern}/``; element loads
    under ``/loads/element/{pattern}/``.  SP (single-point) records
    land under ``/loads/sp/{pattern_or_default}`` for symmetry with
    the other load kinds.
    """
    nodal_loads = getattr(fem.nodes, "loads", None)
    elem_loads = getattr(fem.elements, "loads", None)
    sp_loads = getattr(fem.nodes, "sp", None)

    has_nodal = bool(nodal_loads) if nodal_loads is not None else False
    has_elem = bool(elem_loads) if elem_loads is not None else False
    has_sp = bool(sp_loads) if sp_loads is not None else False

    if not (has_nodal or has_elem or has_sp):
        return

    parent = f.create_group("loads")

    if has_nodal:
        _write_nodal_loads(parent.create_group("nodal"), nodal_loads)
    if has_elem:
        _write_element_loads(parent.create_group("element"), elem_loads)
    if has_sp:
        _write_sp_loads(parent.create_group("sp"), sp_loads)


def _write_nodal_loads(parent: Any, load_set: Any) -> None:
    nan = float("nan")
    outer = make_record_dtype(nodal_load_payload_dtype())
    for pattern in load_set.patterns():
        records = load_set.by_pattern(pattern)
        if not records:
            continue
        rows = np.empty(len(records), dtype=outer)
        for i, rec in enumerate(records):
            force = rec.force_xyz or (nan, nan, nan)
            moment = rec.moment_xyz or (nan, nan, nan)
            rows[i] = (
                "node", str(int(rec.node_id)), "nodal",
                (int(rec.node_id), tuple(float(x) for x in force),
                 tuple(float(x) for x in moment)),
            )
        safe = str(pattern).replace("/", "_") or "default"
        parent.create_dataset(safe, data=rows)


def _write_element_loads(parent: Any, load_set: Any) -> None:
    outer = make_record_dtype(element_load_payload_dtype())
    for pattern in load_set.patterns():
        records = load_set.by_pattern(pattern)
        if not records:
            continue
        rows = np.empty(len(records), dtype=outer)
        for i, rec in enumerate(records):
            params_json = json.dumps(rec.params, default=_json_default)
            rows[i] = (
                "element", str(int(rec.element_id)), "element",
                (int(rec.element_id), str(rec.load_type), params_json),
            )
        safe = str(pattern).replace("/", "_") or "default"
        parent.create_dataset(safe, data=rows)


def _write_sp_loads(parent: Any, sp_set: Any) -> None:
    outer = make_record_dtype(sp_payload_dtype())
    # SPSet has no pattern attr per record beyond the LoadRecord base;
    # group all records under a single ``default`` dataset.
    rows = np.empty(len(sp_set), dtype=outer)
    for i, rec in enumerate(sp_set):
        rows[i] = (
            "node", str(int(rec.node_id)), "sp",
            (
                int(rec.node_id), int(rec.dof),
                float(rec.value), int(bool(rec.is_homogeneous)),
            ),
        )
    parent.create_dataset("default", data=rows)


def _json_default(obj: Any) -> Any:
    """Fallback for ``json.dumps`` on non-JSON types (numpy scalars)."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Masses
# ---------------------------------------------------------------------------


def _write_masses(fem: "FEMData", f: Any) -> None:
    """Write ``/masses`` — one symmetric-compound row per :class:`MassRecord`."""
    mass_set = getattr(fem.nodes, "masses", None)
    if not mass_set:
        return

    outer = make_record_dtype(mass_payload_dtype())
    rows = np.empty(len(mass_set), dtype=outer)
    for i, rec in enumerate(mass_set):
        mass_tuple = tuple(float(x) for x in tuple(rec.mass)[:6])
        if len(mass_tuple) < 6:
            mass_tuple = mass_tuple + (0.0,) * (6 - len(mass_tuple))
        rows[i] = (
            "node", str(int(rec.node_id)), "mass",
            (int(rec.node_id), mass_tuple),
        )
    f.create_dataset("masses", data=rows)
