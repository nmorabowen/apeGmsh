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
    target: object                  # part label, PG name, mesh selection name, or DimTag list
    pattern: str = "default"        # pattern grouping name
    name: str | None = None
    reduction: str = "tributary"    # "tributary" | "consistent"
    target_form: str = "nodal"      # "nodal" | "element"


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
    """Force/moment vector at a single node."""
    kind: str = field(init=False, default="nodal")
    node_id: int = 0
    forces: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # (Fx,Fy,Fz,Mx,My,Mz)


@dataclass
class ElementLoadRecord(LoadRecord):
    """Element-level load command (e.g. ``eleLoad -beamUniform``)."""
    kind: str = field(init=False, default="element")
    element_id: int = 0
    load_type: str = ""             # "beamUniform", "surfacePressure", "bodyForce", ...
    params: dict = field(default_factory=dict)


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


def _pad_force(
    force_xyz: tuple | None,
    moment_xyz: tuple | None,
    ndf: int,
) -> tuple:
    """Build a (Fx,Fy,Fz,Mx,My,Mz) tuple padded for *ndf*.

    Used by :class:`LoadResolver` to ensure NodalLoadRecord.forces
    has consistent shape regardless of how the user specified the load.

    The result is always length 6.  The solver bridge slices to
    ``ndf`` when emitting commands.
    """
    fx = fy = fz = 0.0
    mx = my = mz = 0.0
    if force_xyz is not None:
        if len(force_xyz) == 2:
            fx, fy = force_xyz
        elif len(force_xyz) == 3:
            fx, fy, fz = force_xyz
        else:
            raise ValueError(f"force_xyz must be length 2 or 3, got {len(force_xyz)}")
    if moment_xyz is not None:
        if ndf < 4 and any(abs(v) > 0 for v in moment_xyz):
            raise ValueError(
                f"Cannot apply moment to a model with ndf={ndf} (no rotational DOFs)."
            )
        if len(moment_xyz) == 1:
            mz = moment_xyz[0]
        elif len(moment_xyz) == 3:
            mx, my, mz = moment_xyz
        else:
            raise ValueError(f"moment_xyz must be length 1 or 3, got {len(moment_xyz)}")
    return (fx, fy, fz, mx, my, mz)


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
    """Convert an accumulator dict to a list of NodalLoadRecord."""
    out: list[NodalLoadRecord] = []
    for nid, vec in accum.items():
        rec = NodalLoadRecord(
            pattern=pattern,
            name=name,
            node_id=int(nid),
            forces=tuple(float(v) for v in vec),
        )
        out.append(rec)
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
        ndf: int = 6,
    ) -> None:
        self.node_tags = np.asarray(node_tags, dtype=np.int64)
        self.node_coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        self.elem_tags = (
            np.asarray(elem_tags, dtype=np.int64) if elem_tags is not None else None
        )
        self.connectivity = (
            np.asarray(connectivity, dtype=np.int64) if connectivity is not None else None
        )
        self.ndf = int(ndf)
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
        force6 = np.array(_pad_force(defn.force_xyz, defn.moment_xyz, self.ndf))
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
        if defn.q_xyz is not None:
            q = np.asarray(defn.q_xyz, dtype=float)
        else:
            q = defn.magnitude * _direction_vec(defn.direction)
        accum: dict[int, ndarray] = {}
        # NOTE: For now, the composite always passes 2-node edges
        # because we don't have higher-order edge connectivity in
        # general.  This branch matches resolve_line_tributary().
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
