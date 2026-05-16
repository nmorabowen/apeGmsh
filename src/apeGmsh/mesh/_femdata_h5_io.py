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
* ``/mesh_selections``     — post-mesh selection sets (Phase 8.7).
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
    "read_fem_h5",
    "write_fem_h5",
    "write_meta",
    "write_neutral_zone",
]


#: Schema version stamped by :func:`write_fem_h5` and the standalone
#: ``FEMData.to_h5(path)`` flow.  Phase 8.5 added the neutral zone
#: (`2.0.0 → 2.1.0`); Phase 8.6 added the ``fem_eids`` dataset under
#: each ``/opensees/element_meta/{type_token}/`` group
#: (`2.1.0 → 2.2.0`).  Phase 8.7 commit 2 added the
#: ``/mesh_selections/`` neutral-zone group, mirroring
#: ``/physical_groups`` for post-mesh selection sets
#: (`2.3.0 → 2.4.0`).  Broker-only files (no `/opensees/...`) still
#: stamp the current minor — the field is additive and old readers
#: tolerate its absence.
NEUTRAL_SCHEMA_VERSION: str = "2.4.0"


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
    _write_mesh_selections(fem, f)
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


def _write_mesh_selections(fem: "FEMData", f: Any) -> None:
    """Write ``/mesh_selections/{name}/{node_ids, node_coords, element_ids}``.

    Mirrors ``/physical_groups`` and ``/labels`` shape so the same
    ``H5Reader._read_named_index`` helper handles all three.  Sourced
    from :attr:`FEMData.mesh_selection` (a
    :class:`apeGmsh.mesh.MeshSelectionSet.MeshSelectionStore` captured
    at ``get_fem_data()`` time).  Omitted entirely when the snapshot
    has no selection store or the store is empty — absence is the
    right "no selections" signal.

    Added in Phase 8.7 commit 2 (schema 2.3.0 → 2.4.0, additive) so
    the viewer's ``selection=`` selector round-trips through
    ``model.h5``.
    """
    store = getattr(fem, "mesh_selection", None)
    if store is None:
        return
    try:
        keys = store.get_all()
    except (AttributeError, TypeError):
        return
    if not keys:
        return

    parent = f.create_group("mesh_selections")
    seen_safe: set[str] = set()
    for dim, tag in keys:
        d, t = int(dim), int(tag)
        try:
            name = store.get_name(d, t)
        except (KeyError, ValueError, AttributeError):
            name = ""
        if not name:
            name = f"_unnamed_{d}_{t}"
        safe = name.replace("/", "_")
        if safe in seen_safe:
            safe = f"{safe}__{d}_{t}"
        seen_safe.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = d
        sub.attrs["tag"] = t
        sub.attrs["name"] = name

        try:
            node_data = store.get_nodes(d, t)
            nids = np.asarray(node_data["tags"], dtype=np.int64)
            ncoords = np.asarray(node_data["coords"], dtype=np.float64)
        except (KeyError, ValueError, AttributeError):
            nids = np.array([], dtype=np.int64)
            ncoords = np.zeros((0, 3), dtype=np.float64)
        sub.create_dataset("node_ids", data=nids)
        sub.create_dataset("node_coords", data=ncoords)

        if d >= 1:
            try:
                elem_data = store.get_elements(d, t)
                eids = np.asarray(elem_data["element_ids"], dtype=np.int64)
            except (KeyError, ValueError, AttributeError):
                eids = np.array([], dtype=np.int64)
            if eids.size > 0:
                sub.create_dataset("element_ids", data=eids)


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


# ===========================================================================
# Reader (Phase session-save 2 — root-layout inverse of write_fem_h5)
# ===========================================================================
#
# Mirrors the writer's seven neutral-zone groups plus ``/meta``.  Schema
# major version is checked; minor differences are tolerated (additive).
# Companion to :func:`write_fem_h5` — together they form the contract
# that ``apeGmsh(save_to=...)`` round-trips through.


def read_fem_h5(path: str) -> "FEMData":
    """Reconstruct a :class:`FEMData` from a root-layout ``model.h5``.

    Inverse of :func:`write_fem_h5`.  Reads the seven neutral-zone
    groups (``/nodes``, ``/elements``, ``/physical_groups``,
    ``/labels``, ``/mesh_selections``, ``/constraints``, ``/loads``,
    ``/masses``) plus ``/meta``, and rebuilds a fully populated
    ``FEMData`` snapshot.  Empty groups are skipped — absence is the
    right "no records" signal on disk.

    Parameters
    ----------
    path : str
        Path to a model.h5 written by ``g.save()``,
        ``apeGmsh(save_to=...)``, or :func:`write_fem_h5`.

    Raises
    ------
    apeGmsh.opensees.emitter.h5_reader.SchemaVersionError
        If ``/meta/schema_version`` major != 2.
    apeGmsh.opensees.emitter.h5_reader.MalformedH5Error
        If ``/meta`` is missing.
    """
    import h5py
    from apeGmsh.opensees.emitter.h5_reader import (
        MalformedH5Error,
        SchemaVersionError,
    )

    from ._element_types import ElementGroup, ElementTypeInfo, make_type_info
    from ._group_set import LabelSet, PhysicalGroupSet
    from .FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

    with h5py.File(path, "r") as f:
        # -- meta + schema check --
        if "meta" not in f:
            raise MalformedH5Error(
                f"{path}: missing /meta group; not an apeGmsh model.h5"
            )
        version = str(f["meta"].attrs.get("schema_version", ""))
        if not version:
            raise MalformedH5Error(
                f"{path}: /meta/schema_version attribute is empty"
            )
        try:
            major = int(version.split(".", 1)[0])
        except ValueError as exc:
            raise MalformedH5Error(
                f"{path}: /meta/schema_version {version!r} is not "
                "semver-shaped"
            ) from exc
        if major != 2:
            raise SchemaVersionError(
                f"{path}: schema_version={version} (major {major}) is "
                "not supported by read_fem_h5 (expected major 2)"
            )

        # -- nodes --
        nodes_grp = f["nodes"]
        node_ids = np.asarray(nodes_grp["ids"][...], dtype=np.int64)
        node_coords = np.asarray(nodes_grp["coords"][...], dtype=np.float64)

        # -- elements (per-type subgroups) --
        element_groups: dict[int, ElementGroup] = {}
        types_meta: list[ElementTypeInfo] = []
        elem_grp = f.get("elements")
        if elem_grp is not None:
            for type_name in sorted(elem_grp.keys()):
                sub = elem_grp[type_name]
                if not hasattr(sub, "keys"):
                    continue
                ids = np.asarray(sub["ids"][...], dtype=np.int64)
                conn = np.asarray(sub["connectivity"][...], dtype=np.int64)
                attrs = sub.attrs
                npe = int(
                    attrs.get("npe", conn.shape[1] if conn.ndim == 2 else 0)
                )
                info = make_type_info(
                    code=int(attrs.get("code", 0)),
                    gmsh_name=str(attrs.get("gmsh_name", type_name)),
                    dim=int(attrs.get("dim", 0)),
                    order=int(attrs.get("order", 1)),
                    npe=npe,
                    count=ids.shape[0],
                )
                types_meta.append(info)
                element_groups[info.code] = ElementGroup(
                    element_type=info, ids=ids, connectivity=conn,
                )

        # -- physical_groups + labels (root-level union of node + elem sides) --
        node_pgs, elem_pgs = _read_named_index_at_root(f.get("physical_groups"))
        node_labels, elem_labels = _read_named_index_at_root(f.get("labels"))

        # -- mesh_selections --
        mesh_selection = _read_mesh_selections(f.get("mesh_selections"))

        # -- constraints (split node-side vs element-side by record type) --
        node_constraints, elem_constraints = _read_constraints(
            f.get("constraints")
        )

        # -- loads --
        nodal_loads, element_loads, sp_records = _read_loads(f.get("loads"))

        # -- masses --
        mass_records = _read_masses(f.get("masses"))

        # -- assemble composites --
        nodes = NodeComposite(
            node_ids=node_ids,
            node_coords=node_coords,
            physical=PhysicalGroupSet(node_pgs),
            labels=LabelSet(node_labels),
            constraints=node_constraints,
            loads=nodal_loads,
            sp=sp_records,
            masses=mass_records,
        )
        elements = ElementComposite(
            groups=element_groups,
            physical=PhysicalGroupSet(elem_pgs),
            labels=LabelSet(elem_labels),
            constraints=elem_constraints,
            loads=element_loads,
        )
        info = MeshInfo(
            n_nodes=len(node_ids),
            n_elems=sum(len(g) for g in element_groups.values()),
            bandwidth=0,            # not round-tripped (writer doesn't store it)
            types=types_meta,
        )
        return FEMData(
            nodes=nodes, elements=elements, info=info,
            mesh_selection=mesh_selection,
        )


def _read_named_index_at_root(
    parent: Any,
) -> tuple[dict[tuple[int, int], dict], dict[tuple[int, int], dict]]:
    """Read a root ``/physical_groups`` or ``/labels`` index.

    The writer combines node-side + element-side keys into one root
    group per (dim, tag). Reading splits them back: every key lands in
    both dicts (the node-side gets node_ids/coords; the element-side
    additionally gets element_ids when the writer recorded any).
    """
    node_dict: dict[tuple[int, int], dict] = {}
    elem_dict: dict[tuple[int, int], dict] = {}
    if parent is None:
        return node_dict, elem_dict

    for safe_name in parent.keys():
        sub = parent[safe_name]
        if not hasattr(sub, "keys"):
            continue
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        name = str(attrs.get("name", safe_name))

        nids = (
            np.asarray(sub["node_ids"][...], dtype=np.int64)
            if "node_ids" in sub
            else np.array([], dtype=np.int64)
        )
        ncoords = (
            np.asarray(sub["node_coords"][...], dtype=np.float64)
            if "node_coords" in sub
            else np.zeros((0, 3), dtype=np.float64)
        )

        node_dict[(dim, tag)] = {
            "name": name,
            "node_ids": nids,
            "node_coords": ncoords,
        }
        elem_info: dict = {
            "name": name,
            "node_ids": nids,
            "node_coords": ncoords,
        }
        if "element_ids" in sub:
            elem_info["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )
        elem_dict[(dim, tag)] = elem_info

    return node_dict, elem_dict


def _read_mesh_selections(parent: Any):
    """Reconstruct a ``MeshSelectionStore`` from ``/mesh_selections``."""
    if parent is None:
        return None
    from .MeshSelectionSet import MeshSelectionStore

    sets: dict[tuple[int, int], dict] = {}
    for safe_name in parent.keys():
        sub = parent[safe_name]
        if not hasattr(sub, "keys"):
            continue
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        info: dict = {"name": str(attrs.get("name", safe_name))}

        if "node_ids" in sub:
            info["node_ids"] = np.asarray(
                sub["node_ids"][...], dtype=np.int64,
            )
            info["node_coords"] = np.asarray(
                sub["node_coords"][...], dtype=np.float64,
            )
        if "element_ids" in sub:
            info["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )

        sets[(dim, tag)] = info

    if not sets:
        return None
    return MeshSelectionStore(sets)


# ---------------------------------------------------------------------------
# Constraint decoders
# ---------------------------------------------------------------------------


def _read_constraints(
    parent: Any,
) -> tuple[list[Any], list[Any]]:
    """Decode ``/constraints/{kind}`` datasets into node + element record lists.

    Routing matches the writer:

    * ``NodePair*``, ``NodeGroup*``, ``NodeToSurface*`` → node-side
    * ``Interpolation*``, ``SurfaceCoupling*``         → element-side
    """
    from .records._constraints import (
        InterpolationRecord,
        NodeGroupRecord,
        NodePairRecord,
        NodeToSurfaceRecord,
        SurfaceCouplingRecord,
    )

    node_records: list[Any] = []
    elem_records: list[Any] = []
    if parent is None:
        return node_records, elem_records

    # Maps payload-field signature → (decoder, target_list)
    NODE_PAIR_FIELDS = {
        "master_node", "slave_node", "dofs", "offset", "penalty_stiffness",
    }
    NODE_GROUP_FIELDS = {
        "master_node", "slave_nodes", "dofs", "offsets", "plane_normal",
    }
    INTERPOLATION_FIELDS = {
        "slave_node", "master_nodes", "weights", "dofs",
        "projected_point", "parametric_coords",
    }
    SURFACE_COUPLING_FIELDS = {
        "master_nodes", "slave_nodes", "dofs",
        "mortar_operator_shape", "mortar_operator",
    }
    NODE_TO_SURFACE_FIELDS = {
        "master_node", "slave_nodes", "phantom_nodes",
        "phantom_coords", "dofs",
    }

    for kind_name in parent.keys():
        ds = parent[kind_name]
        if hasattr(ds, "keys"):
            # Skipped/unknown record type was written as a group with
            # __deviation__ attr — nothing to reconstruct.
            continue

        rows = ds[...]
        if rows.shape == ():       # scalar → make 1-D
            rows = np.array([rows])
        if rows.size == 0:
            continue

        payload_fields = set(rows.dtype["payload"].names or ())

        if payload_fields == NODE_PAIR_FIELDS:
            for row in rows:
                node_records.append(_decode_node_pair(row, NodePairRecord))
        elif payload_fields == NODE_GROUP_FIELDS:
            for row in rows:
                node_records.append(_decode_node_group(row, NodeGroupRecord))
        elif payload_fields == INTERPOLATION_FIELDS:
            for row in rows:
                elem_records.append(
                    _decode_interpolation(row, InterpolationRecord)
                )
        elif payload_fields == SURFACE_COUPLING_FIELDS:
            for row in rows:
                elem_records.append(
                    _decode_surface_coupling(row, SurfaceCouplingRecord)
                )
        elif payload_fields == NODE_TO_SURFACE_FIELDS:
            for row in rows:
                node_records.append(
                    _decode_node_to_surface(row, NodeToSurfaceRecord)
                )
        # else: unknown payload schema — skip silently (forward-compat)

    return node_records, elem_records


def _kind(row: Any) -> str:
    return _str(row["payload_kind"])


def _str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _opt_vec3(arr: np.ndarray) -> np.ndarray | None:
    """Return arr if any component is finite, else None (NaN-sentinel decode)."""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)[:3]
    if not np.any(np.isfinite(a)):
        return None
    return a


def _opt_vec2(arr: np.ndarray) -> np.ndarray | None:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)[:2]
    if not np.any(np.isfinite(a)):
        return None
    return a


def _opt_scalar(value: float) -> float | None:
    v = float(value)
    if np.isnan(v):
        return None
    return v


def _decode_node_pair(row: Any, cls: type) -> Any:
    p = row["payload"]
    return cls(
        kind=_kind(row),
        master_node=int(p["master_node"]),
        slave_node=int(p["slave_node"]),
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        offset=_opt_vec3(p["offset"]),
        penalty_stiffness=_opt_scalar(p["penalty_stiffness"]),
    )


def _decode_node_group(row: Any, cls: type) -> Any:
    p = row["payload"]
    slaves = np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)
    offsets_flat = np.asarray(p["offsets"], dtype=np.float64).reshape(-1)
    offsets = (
        offsets_flat.reshape(-1, 3)
        if offsets_flat.size and offsets_flat.size == 3 * len(slaves)
        else None
    )
    return cls(
        kind=_kind(row),
        master_node=int(p["master_node"]),
        slave_nodes=[int(x) for x in slaves],
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        offsets=offsets,
        plane_normal=_opt_vec3(p["plane_normal"]),
    )


def _decode_interpolation(row: Any, cls: type) -> Any:
    p = row["payload"]
    weights_flat = np.asarray(p["weights"], dtype=np.float64).reshape(-1)
    weights = weights_flat if weights_flat.size > 0 else None
    return cls(
        kind=_kind(row),
        slave_node=int(p["slave_node"]),
        master_nodes=[
            int(x) for x in np.asarray(p["master_nodes"]).reshape(-1)
        ],
        weights=weights,
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        projected_point=_opt_vec3(p["projected_point"]),
        parametric_coords=_opt_vec2(p["parametric_coords"]),
    )


def _decode_surface_coupling(row: Any, cls: type) -> Any:
    p = row["payload"]
    shape = tuple(int(x) for x in np.asarray(p["mortar_operator_shape"]))
    flat = np.asarray(p["mortar_operator"], dtype=np.float64).reshape(-1)
    if shape == (0, 0) or flat.size == 0:
        operator = None
    else:
        operator = flat.reshape(shape)
    return cls(
        kind=_kind(row),
        master_nodes=[
            int(x) for x in np.asarray(p["master_nodes"]).reshape(-1)
        ],
        slave_nodes=[
            int(x) for x in np.asarray(p["slave_nodes"]).reshape(-1)
        ],
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
        mortar_operator=operator,
    )


def _decode_node_to_surface(row: Any, cls: type) -> Any:
    p = row["payload"]
    slaves = np.asarray(p["slave_nodes"], dtype=np.int64).reshape(-1)
    coords_flat = np.asarray(p["phantom_coords"], dtype=np.float64).reshape(-1)
    if coords_flat.size and coords_flat.size == 3 * len(slaves):
        phantom_coords = coords_flat.reshape(-1, 3)
    else:
        phantom_coords = None
    return cls(
        kind=_kind(row),
        master_node=int(p["master_node"]),
        slave_nodes=[int(x) for x in slaves],
        phantom_nodes=[
            int(x) for x in np.asarray(p["phantom_nodes"]).reshape(-1)
        ],
        phantom_coords=phantom_coords,
        dofs=[int(x) for x in np.asarray(p["dofs"]).reshape(-1)],
    )


# ---------------------------------------------------------------------------
# Load decoders
# ---------------------------------------------------------------------------


def _read_loads(
    parent: Any,
) -> tuple[list[Any], list[Any], list[Any]]:
    """Decode ``/loads/{nodal|element|sp}/{pattern}`` datasets."""
    from .records._loads import ElementLoadRecord, NodalLoadRecord, SPRecord

    nodal: list[Any] = []
    element: list[Any] = []
    sp: list[Any] = []
    if parent is None:
        return nodal, element, sp

    nodal_grp = parent.get("nodal")
    if nodal_grp is not None:
        for pattern_safe in nodal_grp.keys():
            ds = nodal_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                force = tuple(
                    float(x) for x in np.asarray(p["force_xyz"]).reshape(-1)[:3]
                )
                moment = tuple(
                    float(x) for x in np.asarray(p["moment_xyz"]).reshape(-1)[:3]
                )
                nodal.append(NodalLoadRecord(
                    pattern=_str(pattern_safe),
                    node_id=int(p["node_id"]),
                    force_xyz=force if any(np.isfinite(force)) else None,
                    moment_xyz=moment if any(np.isfinite(moment)) else None,
                ))

    elem_grp = parent.get("element")
    if elem_grp is not None:
        for pattern_safe in elem_grp.keys():
            ds = elem_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                params_str = _str(p["params_json"])
                params = json.loads(params_str) if params_str else {}
                element.append(ElementLoadRecord(
                    pattern=_str(pattern_safe),
                    element_id=int(p["element_id"]),
                    load_type=_str(p["load_type"]),
                    params=params,
                ))

    sp_grp = parent.get("sp")
    if sp_grp is not None:
        for pattern_safe in sp_grp.keys():
            ds = sp_grp[pattern_safe]
            rows = np.atleast_1d(ds[...])
            for row in rows:
                p = row["payload"]
                sp.append(SPRecord(
                    pattern=_str(pattern_safe) if pattern_safe != "default" else "default",
                    node_id=int(p["node_id"]),
                    dof=int(p["dof"]),
                    value=float(p["value"]),
                    is_homogeneous=bool(int(p["is_homogeneous"])),
                ))

    return nodal, element, sp


# ---------------------------------------------------------------------------
# Mass decoder
# ---------------------------------------------------------------------------


def _read_masses(ds: Any) -> list[Any]:
    """Decode the ``/masses`` dataset into ``MassRecord`` objects."""
    from .records._masses import MassRecord

    out: list[Any] = []
    if ds is None:
        return out
    rows = np.atleast_1d(ds[...])
    for row in rows:
        p = row["payload"]
        mass_arr = np.asarray(p["mass"], dtype=np.float64).reshape(-1)[:6]
        if mass_arr.size < 6:
            mass_arr = np.concatenate(
                [mass_arr, np.zeros(6 - mass_arr.size, dtype=np.float64)]
            )
        out.append(MassRecord(
            node_id=int(p["node_id"]),
            mass=tuple(float(x) for x in mass_arr),
        ))
    return out
