"""Read-side row dataclasses + decoders for the viewer.

These are the viewer-facing tuples produced by both
:meth:`apeGmsh.viewers.data.ViewerData.from_fem` and
:meth:`~apeGmsh.viewers.data.ViewerData.from_h5`.  They are
deliberately NOT imported from :mod:`apeGmsh.mesh.records` — that
package owns the write-side authoritative record types, and importing
them here would re-establish the forbidden ``viewers/`` →
``mesh/`` coupling (Phase 8.7 acceptance criterion).

The duplication is intentional and documented in
[ADR 0014](../../opensees/architecture/decisions/0014-viewer-is-pure-h5-consumer.md):
the schema document (``architecture/h5-schema.md``) is the single
source-of-truth for the on-disk contract; row dataclasses here echo
the field names so the schema seam stays one-directional.

If a new ``payload_kind`` lands in the writer (``mesh/_record_h5.py``)
without a matching read-side decoder here, :func:`decode_constraint_row`
raises :class:`ViewerDataDecodeError` naming the unknown kind.  The
parity test in ``tests/viewers/data/test_viewer_data.py`` keeps the
two sides honest.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray


class ViewerDataDecodeError(ValueError):
    """Raised when a model.h5 record row has an unknown payload_kind.

    Carries the kind in the message so a missing read-side decoder is
    easy to diagnose without re-running with a debugger.
    """


# =====================================================================
# Load row dataclasses
# =====================================================================


@dataclass(frozen=True)
class NodalLoadRow:
    """Resolved point load at a node.

    Mirrors :class:`apeGmsh.mesh.records._loads.NodalLoadRecord` fields
    (DOF-agnostic 3D spatial vectors)."""
    node_id: int
    force_xyz: tuple[float, float, float] | None
    moment_xyz: tuple[float, float, float] | None
    pattern: str
    name: str | None = None


@dataclass(frozen=True)
class ElementLoadRow:
    """Resolved element load (beam uniform, surface pressure, …).

    ``params`` is the decoded dict — from the live ``params`` attribute
    when sourced from a FEMData, or decoded from the ``params_json``
    field when sourced from ``model.h5``."""
    element_id: int
    load_type: str
    params: dict[str, Any]
    pattern: str
    name: str | None = None


@dataclass(frozen=True)
class SPRow:
    """Single-point constraint row (prescribed displacement / fix)."""
    node_id: int
    dof: int
    value: float
    is_homogeneous: bool
    pattern: str
    name: str | None = None


# =====================================================================
# Mass row
# =====================================================================


@dataclass(frozen=True)
class MassRow:
    """Resolved per-node 6-DOF mass (mx, my, mz, Ixx, Iyy, Izz)."""
    node_id: int
    mass: tuple[float, float, float, float, float, float]
    name: str | None = None


# =====================================================================
# Constraint row dataclasses
# =====================================================================


@dataclass(frozen=True)
class NodePairRow:
    """One master ↔ one slave (equal_dof, rigid_beam, rigid_rod, penalty)."""
    kind: str
    master_node: int
    slave_node: int
    dofs: tuple[int, ...]
    offset: tuple[float, float, float] | None = None
    penalty_stiffness: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class NodeGroupRow:
    """One master ↔ many slaves (rigid_diaphragm, rigid_body, kinematic_coupling)."""
    kind: str
    master_node: int
    slave_nodes: tuple[int, ...]
    dofs: tuple[int, ...]
    offsets: ndarray | None = None       # (n_slaves, 3) or None
    plane_normal: tuple[float, float, float] | None = None
    name: str | None = None


@dataclass(frozen=True)
class InterpolationRow:
    """Slave node interpolated from master face (tie, distributing, embedded)."""
    kind: str
    slave_node: int
    master_nodes: tuple[int, ...]
    weights: ndarray | None
    dofs: tuple[int, ...]
    projected_point: tuple[float, float, float] | None = None
    parametric_coords: tuple[float, float] | None = None
    name: str | None = None


@dataclass(frozen=True)
class SurfaceCouplingRow:
    """Surface-to-surface coupling (tied_contact, mortar).

    The per-slave ``slave_records`` list mirrors the write-side
    :class:`SurfaceCouplingRecord` shape.  On the h5 path, this list
    is empty — the symmetric compound stores only the high-level
    operator fields; per-slave reconstruction lives on the FEM side."""
    kind: str
    slave_records: tuple[InterpolationRow, ...]
    mortar_operator: ndarray | None
    master_nodes: tuple[int, ...]
    slave_nodes: tuple[int, ...]
    dofs: tuple[int, ...]
    name: str | None = None


@dataclass(frozen=True)
class NodeToSurfaceRow:
    """6-DOF node ↔ 3-DOF surface coupling via phantom nodes."""
    kind: str
    master_node: int
    slave_nodes: tuple[int, ...]
    phantom_nodes: tuple[int, ...]
    phantom_coords: ndarray | None
    rigid_link_records: tuple[NodePairRow, ...]
    equal_dof_records: tuple[NodePairRow, ...]
    dofs: tuple[int, ...]
    name: str | None = None


ConstraintRow = (
    NodePairRow | NodeGroupRow | InterpolationRow
    | SurfaceCouplingRow | NodeToSurfaceRow
)


# =====================================================================
# FEM-side converters (record → row)
# =====================================================================


def _opt_tuple3(v: Any) -> tuple[float, float, float] | None:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        return None
    if np.any(np.isnan(arr)):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _tuple_int(v: Any) -> tuple[int, ...]:
    if v is None:
        return ()
    return tuple(int(x) for x in v)


def nodal_load_row_from_record(rec: Any) -> NodalLoadRow:
    return NodalLoadRow(
        node_id=int(rec.node_id),
        force_xyz=_opt_tuple3(rec.force_xyz),
        moment_xyz=_opt_tuple3(rec.moment_xyz),
        pattern=str(rec.pattern),
        name=rec.name,
    )


def element_load_row_from_record(rec: Any) -> ElementLoadRow:
    return ElementLoadRow(
        element_id=int(rec.element_id),
        load_type=str(rec.load_type),
        params=dict(rec.params or {}),
        pattern=str(rec.pattern),
        name=rec.name,
    )


def sp_row_from_record(rec: Any) -> SPRow:
    return SPRow(
        node_id=int(rec.node_id),
        dof=int(rec.dof),
        value=float(rec.value),
        is_homogeneous=bool(rec.is_homogeneous),
        pattern=str(rec.pattern),
        name=rec.name,
    )


def mass_row_from_record(rec: Any) -> MassRow:
    m = tuple(float(x) for x in rec.mass)
    if len(m) != 6:
        m = (m + (0.0,) * 6)[:6]
    return MassRow(
        node_id=int(rec.node_id),
        mass=m,  # type: ignore[arg-type]
        name=getattr(rec, "name", None),
    )


def node_pair_row_from_record(rec: Any) -> NodePairRow:
    return NodePairRow(
        kind=str(rec.kind),
        master_node=int(rec.master_node),
        slave_node=int(rec.slave_node),
        dofs=_tuple_int(rec.dofs),
        offset=_opt_tuple3(rec.offset),
        penalty_stiffness=(
            None if rec.penalty_stiffness is None
            else float(rec.penalty_stiffness)
        ),
        name=rec.name,
    )


def node_group_row_from_record(rec: Any) -> NodeGroupRow:
    offsets = None
    if rec.offsets is not None:
        offsets = np.asarray(rec.offsets, dtype=np.float64).reshape(-1, 3)
    return NodeGroupRow(
        kind=str(rec.kind),
        master_node=int(rec.master_node),
        slave_nodes=_tuple_int(rec.slave_nodes),
        dofs=_tuple_int(rec.dofs),
        offsets=offsets,
        plane_normal=_opt_tuple3(rec.plane_normal),
        name=rec.name,
    )


def interpolation_row_from_record(rec: Any) -> InterpolationRow:
    return InterpolationRow(
        kind=str(rec.kind),
        slave_node=int(rec.slave_node),
        master_nodes=_tuple_int(rec.master_nodes),
        weights=(
            None if rec.weights is None
            else np.asarray(rec.weights, dtype=np.float64)
        ),
        dofs=_tuple_int(rec.dofs),
        projected_point=_opt_tuple3(rec.projected_point),
        parametric_coords=(
            None if rec.parametric_coords is None
            else (
                float(rec.parametric_coords[0]),
                float(rec.parametric_coords[1]),
            )
        ),
        name=rec.name,
    )


def surface_coupling_row_from_record(rec: Any) -> SurfaceCouplingRow:
    return SurfaceCouplingRow(
        kind=str(rec.kind),
        slave_records=tuple(
            interpolation_row_from_record(s) for s in rec.slave_records
        ),
        mortar_operator=(
            None if rec.mortar_operator is None
            else np.asarray(rec.mortar_operator, dtype=np.float64)
        ),
        master_nodes=_tuple_int(rec.master_nodes),
        slave_nodes=_tuple_int(rec.slave_nodes),
        dofs=_tuple_int(rec.dofs),
        name=rec.name,
    )


def node_to_surface_row_from_record(rec: Any) -> NodeToSurfaceRow:
    return NodeToSurfaceRow(
        kind=str(rec.kind),
        master_node=int(rec.master_node),
        slave_nodes=_tuple_int(rec.slave_nodes),
        phantom_nodes=_tuple_int(rec.phantom_nodes),
        phantom_coords=(
            None if rec.phantom_coords is None
            else np.asarray(rec.phantom_coords, dtype=np.float64)
        ),
        rigid_link_records=tuple(
            node_pair_row_from_record(p) for p in rec.rigid_link_records
        ),
        equal_dof_records=tuple(
            node_pair_row_from_record(p) for p in rec.equal_dof_records
        ),
        dofs=_tuple_int(rec.dofs),
        name=rec.name,
    )


def constraint_row_from_record(rec: Any) -> ConstraintRow:
    """Dispatch a FEM-side ConstraintRecord subclass to its row type.

    Used by :meth:`ViewerData.from_fem`; raises
    :class:`ViewerDataDecodeError` for unknown subclasses to surface
    schema drift quickly.
    """
    cls_name = type(rec).__name__
    if cls_name == "NodePairRecord":
        return node_pair_row_from_record(rec)
    if cls_name == "NodeGroupRecord":
        return node_group_row_from_record(rec)
    if cls_name == "InterpolationRecord":
        return interpolation_row_from_record(rec)
    if cls_name == "SurfaceCouplingRecord":
        return surface_coupling_row_from_record(rec)
    if cls_name == "NodeToSurfaceRecord":
        return node_to_surface_row_from_record(rec)
    raise ViewerDataDecodeError(
        f"unknown FEM constraint record class {cls_name!r}; "
        "viewers/data/_records.py needs a new converter"
    )


# =====================================================================
# H5-side decoders (compound row → row)
# =====================================================================
#
# Each decoder takes the inner ``payload`` field of a symmetric
# compound row (after caller has read ``payload_kind`` / ``target``).
# The compound rows themselves are h5py.void scalars; payload fields
# are accessed by name and (for vlen) yield ndarrays.


def _decode_payload_dofs(payload: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in payload["dofs"])


def _decode_offset(payload: Any) -> tuple[float, float, float] | None:
    arr = np.asarray(payload["offset"], dtype=np.float64)
    if np.any(np.isnan(arr)):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _decode_penalty(payload: Any) -> float | None:
    val = float(payload["penalty_stiffness"])
    if np.isnan(val):
        return None
    return val


def _decode_plane_normal(payload: Any) -> tuple[float, float, float] | None:
    arr = np.asarray(payload["plane_normal"], dtype=np.float64)
    if np.any(np.isnan(arr)):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _decode_nodal_force(payload: Any, field: str) -> tuple[float, float, float] | None:
    arr = np.asarray(payload[field], dtype=np.float64)
    if np.any(np.isnan(arr)):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def decode_node_pair_row(payload: Any, kind: str) -> NodePairRow:
    return NodePairRow(
        kind=kind,
        master_node=int(payload["master_node"]),
        slave_node=int(payload["slave_node"]),
        dofs=_decode_payload_dofs(payload),
        offset=_decode_offset(payload),
        penalty_stiffness=_decode_penalty(payload),
    )


def decode_node_group_row(payload: Any, kind: str) -> NodeGroupRow:
    flat_off = np.asarray(payload["offsets"], dtype=np.float64).reshape(-1)
    offsets = flat_off.reshape(-1, 3) if flat_off.size else None
    return NodeGroupRow(
        kind=kind,
        master_node=int(payload["master_node"]),
        slave_nodes=tuple(int(x) for x in payload["slave_nodes"]),
        dofs=_decode_payload_dofs(payload),
        offsets=offsets,
        plane_normal=_decode_plane_normal(payload),
    )


def decode_interpolation_row(payload: Any, kind: str) -> InterpolationRow:
    weights = np.asarray(payload["weights"], dtype=np.float64)
    pp = np.asarray(payload["projected_point"], dtype=np.float64)
    pc = np.asarray(payload["parametric_coords"], dtype=np.float64)
    return InterpolationRow(
        kind=kind,
        slave_node=int(payload["slave_node"]),
        master_nodes=tuple(int(x) for x in payload["master_nodes"]),
        weights=weights if weights.size else None,
        dofs=_decode_payload_dofs(payload),
        projected_point=(
            None if np.any(np.isnan(pp))
            else (float(pp[0]), float(pp[1]), float(pp[2]))
        ),
        parametric_coords=(
            None if np.any(np.isnan(pc))
            else (float(pc[0]), float(pc[1]))
        ),
    )


def decode_surface_coupling_row(payload: Any, kind: str) -> SurfaceCouplingRow:
    flat_op = np.asarray(payload["mortar_operator"], dtype=np.float64).reshape(-1)
    shape = tuple(int(x) for x in payload["mortar_operator_shape"])
    if shape == (0, 0) or flat_op.size == 0:
        operator = None
    else:
        operator = flat_op.reshape(shape)
    return SurfaceCouplingRow(
        kind=kind,
        slave_records=(),    # not stored in compound (see _record_h5.py docstring)
        mortar_operator=operator,
        master_nodes=tuple(int(x) for x in payload["master_nodes"]),
        slave_nodes=tuple(int(x) for x in payload["slave_nodes"]),
        dofs=_decode_payload_dofs(payload),
    )


def decode_node_to_surface_row(payload: Any, kind: str) -> NodeToSurfaceRow:
    coords_flat = np.asarray(payload["phantom_coords"], dtype=np.float64).reshape(-1)
    phantom_coords = coords_flat.reshape(-1, 3) if coords_flat.size else None
    return NodeToSurfaceRow(
        kind=kind,
        master_node=int(payload["master_node"]),
        slave_nodes=tuple(int(x) for x in payload["slave_nodes"]),
        phantom_nodes=tuple(int(x) for x in payload["phantom_nodes"]),
        phantom_coords=phantom_coords,
        rigid_link_records=(),  # not stored — derivable from the high-level fields
        equal_dof_records=(),
        dofs=_decode_payload_dofs(payload),
    )


# Per-payload-kind dispatch tables.  Schema 2.4.0 / Phase 8.7 — every
# constraint kind documented in ``architecture/h5-schema.md`` resolves
# to one of these decoders.
_NODE_PAIR_KINDS = frozenset({
    "equal_dof", "rigid_beam", "rigid_beam_stiff", "rigid_rod", "penalty",
})
_NODE_GROUP_KINDS = frozenset({
    "rigid_diaphragm", "rigid_body", "kinematic_coupling",
})
_INTERPOLATION_KINDS = frozenset({
    "tie", "distributing", "embedded",
})
_SURFACE_COUPLING_KINDS = frozenset({
    "tied_contact", "mortar",
})
_NODE_TO_SURFACE_KINDS = frozenset({
    "node_to_surface", "node_to_surface_spring",
})


def decode_constraint_row(row: Any) -> ConstraintRow:
    """Decode one outer compound row into the matching constraint row.

    ``row`` is one entry of a ``/constraints/{kind}`` dataset — an
    h5py.void carrying ``target_kind`` / ``target`` / ``payload_kind`` /
    ``payload`` fields per the symmetric contract.
    """
    payload_kind = _utf8(row["payload_kind"])
    payload = row["payload"]
    if payload_kind in _NODE_PAIR_KINDS:
        return decode_node_pair_row(payload, payload_kind)
    if payload_kind in _NODE_GROUP_KINDS:
        return decode_node_group_row(payload, payload_kind)
    if payload_kind in _INTERPOLATION_KINDS:
        return decode_interpolation_row(payload, payload_kind)
    if payload_kind in _SURFACE_COUPLING_KINDS:
        return decode_surface_coupling_row(payload, payload_kind)
    if payload_kind in _NODE_TO_SURFACE_KINDS:
        return decode_node_to_surface_row(payload, payload_kind)
    raise ViewerDataDecodeError(
        f"unknown constraint payload_kind {payload_kind!r}; "
        "viewers/data/_records.py needs a new decoder"
    )


def decode_nodal_load_row(row: Any, pattern: str) -> NodalLoadRow:
    payload = row["payload"]
    return NodalLoadRow(
        node_id=int(payload["node_id"]),
        force_xyz=_decode_nodal_force(payload, "force_xyz"),
        moment_xyz=_decode_nodal_force(payload, "moment_xyz"),
        pattern=pattern,
    )


def decode_element_load_row(row: Any, pattern: str) -> ElementLoadRow:
    payload = row["payload"]
    params_blob = _utf8(payload["params_json"])
    try:
        params: dict[str, Any] = json.loads(params_blob) if params_blob else {}
    except json.JSONDecodeError:
        params = {}
    return ElementLoadRow(
        element_id=int(payload["element_id"]),
        load_type=_utf8(payload["load_type"]),
        params=params,
        pattern=pattern,
    )


def decode_sp_row(row: Any, pattern: str) -> SPRow:
    payload = row["payload"]
    return SPRow(
        node_id=int(payload["node_id"]),
        dof=int(payload["dof"]),
        value=float(payload["value"]),
        is_homogeneous=bool(int(payload["is_homogeneous"])),
        pattern=pattern,
    )


def decode_mass_row(row: Any) -> MassRow:
    payload = row["payload"]
    arr = np.asarray(payload["mass"], dtype=np.float64).reshape(-1)
    if arr.size != 6:
        arr = np.concatenate([arr, np.zeros(6 - arr.size)])[:6]
    return MassRow(
        node_id=int(payload["node_id"]),
        mass=(
            float(arr[0]), float(arr[1]), float(arr[2]),
            float(arr[3]), float(arr[4]), float(arr[5]),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utf8(v: Any) -> str:
    """Decode an h5py vlen-utf8 / bytes / str compound field to str."""
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)
