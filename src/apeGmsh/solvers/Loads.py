"""
Loads — solver-agnostic load definitions, records, and resolver.

Mirrors the architecture of :mod:`solvers.Constraints`:

* :class:`LoadDef` subclasses describe **intent** at the geometry/PG/part
  level (pre-mesh).
* :class:`LoadRecord` subclasses describe **resolved** facts at the
  node/element level (post-mesh).
* :class:`LoadResolver` converts defs to records using mesh data.

The composite that wraps this (:class:`LoadsComposite`) lives in
``core/LoadsComposite.py``.  This file has no Gmsh dependency, no
session plumbing, and no factory methods — pure data + math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    pass


# ======================================================================
# LoadDef hierarchy (pre-mesh, intent)
# ======================================================================

@dataclass
class LoadDef:
    """Base class for all load definitions."""
    kind: str
    target: object                  # part label, PG name, label name, mesh selection, or DimTag list
    pattern: str = "default"        # pattern grouping name
    name: str | None = None
    reduction: str = "tributary"    # "tributary" | "consistent"
    target_form: str = "nodal"      # "nodal" | "element"
    target_source: str = "auto"     # "auto" | "pg" | "label"


@dataclass
class PointLoadDef(LoadDef):
    """Concentrated force/moment at a node (or set of nodes).

    All targeted nodes receive the **same** force/moment.  Use
    :meth:`PointLoadDef.force_xyz` for translational forces and
    :meth:`PointLoadDef.moment_xyz` for moments (3D rotational DOFs).
    Either may be ``None``.
    """
    kind: str = field(init=False, default="point")
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None


@dataclass
class LineLoadDef(LoadDef):
    """Distributed load along a 1-D entity (curve / beam element).

    Two ways to specify:

    * ``magnitude`` + ``direction`` — scalar magnitude (force per unit
      length) and a direction vector or axis name (``"x"``, ``"y"``,
      ``"z"``).
    * ``q_xyz`` — explicit ``(qx, qy, qz)`` force-per-length vector.
    """
    kind: str = field(init=False, default="line")
    magnitude: float = 0.0
    direction: object = (0.0, 0.0, -1.0)   # tuple or "x"/"y"/"z"
    q_xyz: tuple[float, float, float] | None = None


@dataclass
class SurfaceLoadDef(LoadDef):
    """Pressure or traction on a 2-D entity.

    * ``normal=True``: scalar pressure perpendicular to the face
      (positive into the face).
    * ``normal=False``: vector traction in the given direction.
    """
    kind: str = field(init=False, default="surface")
    magnitude: float = 0.0
    normal: bool = True
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


@dataclass
class GravityLoadDef(LoadDef):
    """Body load from gravity = ρ·g over a volume.

    If ``density`` is ``None``, the solver bridge is expected to read
    density from the assigned material/section.
    """
    kind: str = field(init=False, default="gravity")
    g: tuple[float, float, float] = (0.0, 0.0, -9.81)
    density: float | None = None


@dataclass
class BodyLoadDef(LoadDef):
    """Generic per-volume body force vector (force per unit volume)."""
    kind: str = field(init=False, default="body")
    force_per_volume: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FaceLoadDef(LoadDef):
    """Concentrated force/moment at face centroid, distributed to face nodes.

    ``force_xyz`` is split equally among all face nodes (``F / N``).
    ``moment_xyz`` is converted to statically equivalent nodal forces
    via a least-norm distribution such that ``Sum(r_i x f_i) = M`` and
    ``Sum(f_i) = 0``.

    Use this instead of a reference node + coupling when you only need
    to apply a load to a face without structural coupling to another
    element.
    """
    kind: str = field(init=False, default="face_load")
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None


@dataclass
class FaceSPDef(LoadDef):
    """Prescribed displacement/rotation at face centroid, mapped to face nodes.

    Maps a rigid-body motion at the face centroid to per-node
    displacements using ``u_i = disp_xyz + rot_xyz x r_i``.

    When both ``disp_xyz`` and ``rot_xyz`` are ``None``, all values are
    zero and the result is a homogeneous fix.

    Parameters
    ----------
    dofs : list[int]
        Restraint mask — ``1`` for constrained DOFs, ``0`` for free.
    disp_xyz : tuple or None
        Prescribed translation at the face centroid.
    rot_xyz : tuple or None
        Prescribed rotation about the face centroid.
    """
    kind: str = field(init=False, default="face_sp")
    dofs: list[int] = field(default_factory=lambda: [1, 1, 1])
    disp_xyz: tuple[float, float, float] | None = None
    rot_xyz: tuple[float, float, float] | None = None


# ======================================================================
# LoadRecord hierarchy (post-mesh, resolved)
# ======================================================================

@dataclass
class LoadRecord:
    """Base class for all resolved load records."""
    kind: str
    pattern: str = "default"
    name: str | None = None


@dataclass
class NodalLoadRecord(LoadRecord):
    """Force and/or moment at a single node.

    ``force_xyz`` and ``moment_xyz`` are pure 3D spatial vectors (or
    ``None`` when absent). The record is DOF-agnostic — mapping onto
    a solver's DOF space is the caller's responsibility.
    """
    kind: str = field(init=False, default="nodal")
    node_id: int = 0
    force_xyz: tuple[float, float, float] | None = None
    moment_xyz: tuple[float, float, float] | None = None


@dataclass
class ElementLoadRecord(LoadRecord):
    """Element-level load command (e.g. ``eleLoad -beamUniform``)."""
    kind: str = field(init=False, default="element")
    element_id: int = 0
    load_type: str = ""             # "beamUniform", "surfacePressure", "bodyForce", ...
    params: dict = field(default_factory=dict)


@dataclass
class SPRecord(LoadRecord):
    """Single-point constraint: prescribed displacement or homogeneous fix.

    One record per DOF per node.  When ``is_homogeneous`` is ``True``
    the downstream emitter can use ``ops.fix()``; otherwise it must use
    ``ops.sp(node, dof, value)``.
    """
    kind: str = field(init=False, default="sp")
    node_id: int = 0
    dof: int = 1                    # 1-based DOF index
    value: float = 0.0
    is_homogeneous: bool = True


# ======================================================================
# Helpers
# ======================================================================

def _direction_vec(direction) -> ndarray:
    """Resolve a direction (axis name or vector) to a length-3 ndarray."""
    if isinstance(direction, str):
        ax = direction.lower()
        if ax == "x":
            return np.array([1.0, 0.0, 0.0])
        if ax == "y":
            return np.array([0.0, 1.0, 0.0])
        if ax == "z":
            return np.array([0.0, 0.0, 1.0])
        raise ValueError(f"Unknown axis name: {direction!r}")
    return np.asarray(direction, dtype=float)


def _to_force6(
    force_xyz: tuple | None,
    moment_xyz: tuple | None,
) -> ndarray:
    """Expand user-supplied force/moment input to a length-6 ndarray.

    Purely internal accumulator format (Fx, Fy, Fz, Mx, My, Mz).
    No DOF awareness — zero components are legal, the resolver
    collapses them to ``None`` at record-build time.
    """
    vec = np.zeros(6, dtype=float)
    if force_xyz is not None:
        f = np.asarray(force_xyz, dtype=float)
        if f.shape[0] == 2:
            vec[:2] = f
        elif f.shape[0] == 3:
            vec[:3] = f
        else:
            raise ValueError(
                f"force_xyz must be length 2 or 3, got {f.shape[0]}"
            )
    if moment_xyz is not None:
        m = np.asarray(moment_xyz, dtype=float)
        if m.shape[0] == 1:
            vec[5] = m[0]
        elif m.shape[0] == 3:
            vec[3:6] = m
        else:
            raise ValueError(
                f"moment_xyz must be length 1 or 3, got {m.shape[0]}"
            )
    return vec


def _accumulate_nodal(
    accum: dict[int, ndarray],
    node_id: int,
    force6: ndarray,
) -> None:
    """Accumulate a length-6 force vector into the per-node sum."""
    if node_id in accum:
        accum[node_id] += force6
    else:
        accum[node_id] = force6.copy()


def _accum_to_records(
    accum: dict[int, ndarray],
    *,
    pattern: str,
    name: str | None,
) -> list[NodalLoadRecord]:
    """Convert an accumulator dict to a list of NodalLoadRecord.

    Splits the length-6 accumulator into separate ``force_xyz`` and
    ``moment_xyz`` fields. Zero sub-vectors are stored as ``None``
    so downstream consumers can skip them cheaply.
    """
    out: list[NodalLoadRecord] = []
    for nid, vec in accum.items():
        f = tuple(float(v) for v in vec[:3])
        m = tuple(float(v) for v in vec[3:6])
        force_xyz = f if any(abs(v) > 0.0 for v in f) else None
        moment_xyz = m if any(abs(v) > 0.0 for v in m) else None
        if force_xyz is None and moment_xyz is None:
            continue
        out.append(NodalLoadRecord(
            pattern=pattern,
            name=name,
            node_id=int(nid),
            force_xyz=force_xyz,
            moment_xyz=moment_xyz,
        ))
    return out


# ======================================================================
# LoadResolver
# ======================================================================

class LoadResolver:
    """Convert :class:`LoadDef` instances to :class:`LoadRecord` lists.

    Pure mesh math — receives raw arrays and a DOF context, returns
    record lists.  No Gmsh queries (the composite handles target
    resolution before calling here).
    """

    def __init__(
        self,
        node_tags: ndarray,
        node_coords: ndarray,
        elem_tags: ndarray | None = None,
        connectivity: ndarray | None = None,
    ) -> None:
        self.node_tags = np.asarray(node_tags, dtype=np.int64)
        self.node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        self.elem_tags = (
            np.asarray(elem_tags, dtype=np.int64) if elem_tags is not None else None
        )
        self.connectivity = (
            np.asarray(connectivity, dtype=np.int64) if connectivity is not None else None
        )
        # Lookup helpers
        self._node_to_idx = {int(t): i for i, t in enumerate(self.node_tags)}
        if self.elem_tags is not None:
            self._elem_to_idx = {int(t): i for i, t in enumerate(self.elem_tags)}
        else:
            self._elem_to_idx = {}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def coords_of(self, node_id: int) -> ndarray:
        return self.node_coords[self._node_to_idx[int(node_id)]]

    def edge_length(self, n1: int, n2: int) -> float:
        return float(np.linalg.norm(self.coords_of(n1) - self.coords_of(n2)))

    def face_area(self, node_ids: list[int]) -> float:
        """Polygonal face area via fan triangulation from node[0]."""
        if len(node_ids) < 3:
            return 0.0
        p0 = self.coords_of(node_ids[0])
        area = 0.0
        for i in range(1, len(node_ids) - 1):
            p1 = self.coords_of(node_ids[i])
            p2 = self.coords_of(node_ids[i + 1])
            area += 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        return area

    def face_normal(self, node_ids: list[int]) -> ndarray:
        """Outward normal estimate from the first three nodes."""
        if len(node_ids) < 3:
            return np.array([0.0, 0.0, 1.0])
        p0 = self.coords_of(node_ids[0])
        p1 = self.coords_of(node_ids[1])
        p2 = self.coords_of(node_ids[2])
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])

    def element_volume(self, conn_row: ndarray) -> float:
        """Approximate element volume from its node coordinates.

        Tet4: actual volume.  Hex8: approximate via 6 tets.
        For other element types, returns the convex hull bbox volume
        as a coarse fallback.
        """
        n = len(conn_row)
        pts = np.array([self.coords_of(int(nid)) for nid in conn_row])
        if n == 4:
            v = np.abs(np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0]))) / 6.0
            return float(v)
        if n == 8:
            # Decompose hex into 6 tets sharing one diagonal
            tets = [
                (0, 1, 2, 5), (0, 2, 3, 7), (0, 5, 2, 6),
                (0, 5, 6, 7), (0, 7, 2, 6), (0, 4, 5, 7),
            ]
            tot = 0.0
            for a, b, c, d in tets:
                tot += np.abs(np.dot(
                    pts[b] - pts[a],
                    np.cross(pts[c] - pts[a], pts[d] - pts[a]),
                )) / 6.0
            return float(tot)
        # Fallback: bounding box volume
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        return float(np.prod(mx - mn))

    # ------------------------------------------------------------------
    # Tributary reduction
    # ------------------------------------------------------------------

    def resolve_point(
        self,
        defn: PointLoadDef,
        node_set: set[int],
    ) -> list[NodalLoadRecord]:
        """Apply the same force/moment to every node in *node_set*."""
        force6 = _to_force6(defn.force_xyz, defn.moment_xyz)
        accum: dict[int, ndarray] = {}
        for nid in node_set:
            _accumulate_nodal(accum, nid, force6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_line_tributary(
        self,
        defn: LineLoadDef,
        edges: list[tuple[int, int]],
    ) -> list[NodalLoadRecord]:
        """Distribute a line load by length-weighted nodal share.

        *edges* is a list of (node_a, node_b) pairs covering the loaded
        curve.  Each node receives ``magnitude * Σ(adjacent_edge_len/2)``
        in the direction vector.
        """
        if defn.q_xyz is not None:
            q = np.asarray(defn.q_xyz, dtype=float)
        else:
            q = defn.magnitude * _direction_vec(defn.direction)
        accum: dict[int, ndarray] = {}
        for n1, n2 in edges:
            half_L = 0.5 * self.edge_length(n1, n2)
            f3 = q * half_L
            f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
            _accumulate_nodal(accum, n1, f6)
            _accumulate_nodal(accum, n2, f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_surface_tributary(
        self,
        defn: SurfaceLoadDef,
        faces: list[list[int]],
    ) -> list[NodalLoadRecord]:
        """Distribute a surface load by tributary area.

        *faces* is a list of node-id lists (one per face element).
        For each face, total = ``magnitude * area`` and is split
        equally among the face's nodes.  ``normal=True`` projects
        along the face normal; otherwise the explicit direction vector.
        """
        accum: dict[int, ndarray] = {}
        for face in faces:
            A = self.face_area(face)
            if A <= 0:
                continue
            if defn.normal:
                n = self.face_normal(face)
                # Convention: positive magnitude = pressure pushing into face
                f3 = -defn.magnitude * A * n
            else:
                d = np.asarray(defn.direction, dtype=float)
                d = d / (np.linalg.norm(d) + 1e-30)
                f3 = defn.magnitude * A * d
            per_node = f3 / len(face)
            f6 = np.array([per_node[0], per_node[1], per_node[2], 0.0, 0.0, 0.0])
            for nid in face:
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_gravity_tributary(
        self,
        defn: GravityLoadDef,
        elements: list[ndarray],
    ) -> list[NodalLoadRecord]:
        """Distribute body weight equally to element nodes.

        *elements* is a list of connectivity rows (each is an array
        of node IDs).  Each element contributes ``ρ·V·g`` total,
        split equally among its nodes.
        """
        if defn.density is None:
            raise ValueError(
                "GravityLoadDef requires explicit density for tributary "
                "reduction. Either set density= or use target_form='element' "
                "to defer to the solver."
            )
        g_vec = np.asarray(defn.g, dtype=float)
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_volume(conn_row)
            if V <= 0:
                continue
            f3 = defn.density * V * g_vec
            per_node = f3 / len(conn_row)
            f6 = np.array([per_node[0], per_node[1], per_node[2], 0.0, 0.0, 0.0])
            for nid in conn_row:
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_body_tributary(
        self,
        defn: BodyLoadDef,
        elements: list[ndarray],
    ) -> list[NodalLoadRecord]:
        """Distribute a per-volume body force equally to element nodes."""
        bf = np.asarray(defn.force_per_volume, dtype=float)
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_volume(conn_row)
            if V <= 0:
                continue
            f3 = bf * V
            per_node = f3 / len(conn_row)
            f6 = np.array([per_node[0], per_node[1], per_node[2], 0.0, 0.0, 0.0])
            for nid in conn_row:
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    # ------------------------------------------------------------------
    # Consistent reduction (variational, shape-function based)
    # ------------------------------------------------------------------

    def resolve_line_consistent(
        self,
        defn: LineLoadDef,
        edges: list[tuple[int, int]],
        edge_orders: list[int] | None = None,
    ) -> list[NodalLoadRecord]:
        """Consistent line-load reduction.

        For linear (2-node) edges this is identical to tributary
        (``f_i = q·L/2``).  For quadratic (3-node) edges:
        ``f_end = q·L/6``, ``f_mid = 4q·L/6``.

        *edge_orders* (optional) — list of node counts per edge
        (2 = linear, 3 = quadratic).  When None, all edges are
        treated as linear.
        """
        # NOTE: For now, the composite always passes 2-node edges
        # because we don't have higher-order edge connectivity in
        # general.  For linear edges the consistent reduction matches
        # the tributary one, so we delegate directly.  When quadratic
        # edges land here, re-introduce the per-edge shape-function
        # integration using ``q = defn.q_xyz`` (or
        # ``defn.magnitude * _direction_vec(defn.direction)``).
        return self.resolve_line_tributary(defn, edges)

    def resolve_surface_consistent(
        self,
        defn: SurfaceLoadDef,
        faces: list[list[int]],
    ) -> list[NodalLoadRecord]:
        """Consistent surface load using element shape functions.

        Tri3 / Quad4: equal split (matches tributary).
        Higher-order types fall back to tributary with a warning.
        """
        # Tri3 and Quad4 give the same result as tributary because
        # their shape functions integrate to A/n.
        return self.resolve_surface_tributary(defn, faces)

    def resolve_gravity_consistent(
        self,
        defn: GravityLoadDef,
        elements: list[ndarray],
    ) -> list[NodalLoadRecord]:
        """Consistent gravity reduction.

        For tet4 / hex8 with constant density, the consistent vector
        equals the tributary vector (each node gets V/n × ρ × g).
        """
        return self.resolve_gravity_tributary(defn, elements)

    # ------------------------------------------------------------------
    # Element-form output (eleLoad-style commands)
    # ------------------------------------------------------------------

    def resolve_line_element(
        self,
        defn: LineLoadDef,
        element_ids: list[int],
    ) -> list[ElementLoadRecord]:
        """Emit one ElementLoadRecord per beam element with beamUniform params."""
        if defn.q_xyz is not None:
            qx, qy, qz = defn.q_xyz
        else:
            v = defn.magnitude * _direction_vec(defn.direction)
            qx, qy, qz = float(v[0]), float(v[1]), float(v[2])
        out: list[ElementLoadRecord] = []
        for eid in element_ids:
            out.append(ElementLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                element_id=int(eid),
                load_type="beamUniform",
                params={"wx": qx, "wy": qy, "wz": qz},
            ))
        return out

    def resolve_surface_element(
        self,
        defn: SurfaceLoadDef,
        element_ids: list[int],
    ) -> list[ElementLoadRecord]:
        """Emit one ElementLoadRecord per face element with surfacePressure."""
        out: list[ElementLoadRecord] = []
        for eid in element_ids:
            out.append(ElementLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                element_id=int(eid),
                load_type="surfacePressure",
                params={
                    "p": float(defn.magnitude),
                    "normal": bool(defn.normal),
                    "direction": tuple(float(v) for v in defn.direction),
                },
            ))
        return out

    def resolve_gravity_element(
        self,
        defn: GravityLoadDef,
        element_ids: list[int],
    ) -> list[ElementLoadRecord]:
        """Emit one ElementLoadRecord per volume element with bodyForce."""
        out: list[ElementLoadRecord] = []
        for eid in element_ids:
            out.append(ElementLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                element_id=int(eid),
                load_type="bodyForce",
                params={
                    "g": tuple(float(v) for v in defn.g),
                    "density": defn.density,
                },
            ))
        return out

    def resolve_body_element(
        self,
        defn: BodyLoadDef,
        element_ids: list[int],
    ) -> list[ElementLoadRecord]:
        """Emit one ElementLoadRecord per volume element with bodyForce."""
        out: list[ElementLoadRecord] = []
        for eid in element_ids:
            out.append(ElementLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                element_id=int(eid),
                load_type="bodyForce",
                params={"bf": tuple(float(v) for v in defn.force_per_volume)},
            ))
        return out

    # ------------------------------------------------------------------
    # Face load / face SP
    # ------------------------------------------------------------------

    def resolve_face_load(
        self,
        defn: FaceLoadDef,
        face_node_ids: list[int],
    ) -> list[NodalLoadRecord]:
        """Distribute centroidal force/moment to face nodes.

        ``force_xyz``: equal share ``F / N`` per node.
        ``moment_xyz``: least-norm nodal forces satisfying
        ``Sum(f_i) = 0`` and ``Sum(r_i x f_i) = M``.
        """
        N = len(face_node_ids)
        if N == 0:
            return []
        accum: dict[int, ndarray] = {}

        # Force contribution: equal split
        if defn.force_xyz is not None:
            f_total = np.asarray(defn.force_xyz, dtype=float)
            per_node = f_total / N
            f6 = np.array([per_node[0], per_node[1], per_node[2],
                           0.0, 0.0, 0.0])
            for nid in face_node_ids:
                _accumulate_nodal(accum, nid, f6)

        # Moment contribution: least-norm distribution
        if defn.moment_xyz is not None:
            moment_forces = self._moment_to_nodal_forces(
                defn.moment_xyz, face_node_ids)
            for nid, f3 in moment_forces.items():
                f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
                _accumulate_nodal(accum, nid, f6)

        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def _moment_to_nodal_forces(
        self,
        moment_xyz: tuple[float, float, float],
        node_ids: list[int],
    ) -> dict[int, ndarray]:
        """Least-norm nodal forces for a moment about the face centroid.

        Builds the 6 x 3N equilibrium matrix A:
        - rows 0-2: ``Sum(f_i) = 0``
        - rows 3-5: ``Sum(r_i x f_i) = M``

        Solves for ``f = A^T (A A^T)^{-1} b`` (least-norm solution).
        """
        coords = np.array([self.coords_of(nid) for nid in node_ids])
        centroid = coords.mean(axis=0)
        arms = coords - centroid
        N = len(node_ids)

        A = np.zeros((6, 3 * N))
        for i in range(N):
            # Force equilibrium: Sum(f_i) = 0
            A[0:3, 3 * i:3 * i + 3] = np.eye(3)
            # Moment equilibrium: Sum(r_i x f_i) = M
            r = arms[i]
            A[3, 3 * i:3 * i + 3] = [0.0, -r[2], r[1]]
            A[4, 3 * i:3 * i + 3] = [r[2], 0.0, -r[0]]
            A[5, 3 * i:3 * i + 3] = [-r[1], r[0], 0.0]

        b = np.array([0.0, 0.0, 0.0,
                       moment_xyz[0], moment_xyz[1], moment_xyz[2]])

        AAt = A @ A.T  # 6x6
        f_flat = A.T @ np.linalg.solve(AAt, b)

        result: dict[int, ndarray] = {}
        for i, nid in enumerate(node_ids):
            result[nid] = f_flat[3 * i:3 * i + 3]
        return result

    def resolve_face_sp(
        self,
        defn: FaceSPDef,
        face_node_ids: list[int],
    ) -> list[SPRecord]:
        """Map centroidal rigid-body motion to per-node SP constraints.

        For each constrained DOF *d* and each node *i*:
        ``u_i = disp_xyz + rot_xyz x r_i``, then emit
        ``SPRecord(node_id=i, dof=d, value=u_i[d-1])``.
        """
        if not face_node_ids:
            return []

        coords = np.array([self.coords_of(nid) for nid in face_node_ids])
        centroid = coords.mean(axis=0)

        u0 = np.asarray(defn.disp_xyz, dtype=float) if defn.disp_xyz else np.zeros(3)
        theta = np.asarray(defn.rot_xyz, dtype=float) if defn.rot_xyz else np.zeros(3)

        out: list[SPRecord] = []
        for i, nid in enumerate(face_node_ids):
            r_i = coords[i] - centroid
            u_i = u0 + np.cross(theta, r_i)
            for d_idx, mask in enumerate(defn.dofs):
                if mask != 1:
                    continue
                val = float(u_i[d_idx])
                out.append(SPRecord(
                    pattern=defn.pattern,
                    name=defn.name,
                    node_id=int(nid),
                    dof=d_idx + 1,
                    value=val,
                    is_homogeneous=(abs(val) < 1e-30),
                ))
        return out
