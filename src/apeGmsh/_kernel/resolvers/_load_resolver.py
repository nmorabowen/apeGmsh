"""LoadResolver — pure mesh math that turns LoadDef into LoadRecord.

The resolver receives raw arrays (node tags, coords, optional
connectivity) and a load definition, and returns a list of resolved
records.  No Gmsh queries (the composite handles target resolution
before calling here), no solver imports.

Inputs:
    :class:`apeGmsh.core.loads.defs.LoadDef` subclasses.
Outputs:
    :class:`apeGmsh.mesh.records._loads.NodalLoadRecord`,
    :class:`~apeGmsh.mesh.records._loads.ElementLoadRecord`, and
    :class:`~apeGmsh.mesh.records._loads.SPRecord`.

Shape-function quadrature for the consistent reduction lives in
:mod:`apeGmsh.mesh._consistent_quadrature`.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from apeGmsh._kernel.defs.loads import (
    BodyLoadDef,
    FaceLoadDef,
    FaceSPDef,
    GravityLoadDef,
    LineLoadDef,
    PointLoadDef,
    SurfaceLoadDef,
)
from apeGmsh._kernel.records._loads import (
    ElementLoadRecord,
    NodalLoadRecord,
    SPRecord,
)


# ======================================================================
# Helpers
# ======================================================================

def _direction_vec(direction: object) -> ndarray:
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


def _project_in_plane(vec: ndarray, n: ndarray) -> ndarray:
    """In-plane part of *vec* on the plane with normal *n* (ADR 0050 shear).

    Returns ``vec - (vec·n_hat) n_hat`` — the component of the global
    reference vector lying in the face's tangent plane. Raises
    ``ValueError`` when that projection vanishes (vec is essentially
    normal to the face), so an in-plane shear is undefined there.
    """
    nn = float(np.linalg.norm(n))
    n_hat = n / nn if nn > 1e-30 else n
    d_t = vec - float(np.dot(vec, n_hat)) * n_hat
    v_norm = float(np.linalg.norm(vec))
    dt_norm = float(np.linalg.norm(d_t))
    if v_norm < 1e-30 or dt_norm < 1e-9 * v_norm:
        raise ValueError(
            f"surface.shear: the reference vector "
            f"{tuple(round(float(x), 6) for x in vec)} is (near-)normal to a "
            f"loaded face — its in-plane projection vanishes, so an in-plane "
            f"shear is undefined there. Use surface.pressure for a normal "
            f"load, or pass a vector with an in-plane component."
        )
    return d_t


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
        # Static shape: each slice is exactly 3 floats (guaranteed by
        # _accumulate_nodal's 6-element vectors). Casting to a 3-tuple
        # lets mypy's tuple[float, float, float] type match.
        force_xyz: tuple[float, float, float] | None = (
            (f[0], f[1], f[2]) if any(abs(v) > 0.0 for v in f) else None
        )
        moment_xyz: tuple[float, float, float] | None = (
            (m[0], m[1], m[2]) if any(abs(v) > 0.0 for v in m) else None
        )
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

    def element_measure(self, conn_row: ndarray, dim: int) -> float:
        """Geometric measure of a continuum element.

        ``dim == 3`` → element volume; ``dim == 2`` → element area
        (from the corner nodes, so higher-order tri6/quad8/quad9 use
        their straight-edge corners). Lets gravity / body tributary
        lumping serve 2D models (per-area intensity) and 3D models
        (per-volume intensity) through one code path.
        """
        if dim == 2:
            n = len(conn_row)
            if n in (3, 6):            # tri3 / tri6 → 3 corners
                corners = conn_row[:3]
            elif n in (4, 8, 9):       # quad4 / quad8 / quad9 → 4 corners
                corners = conn_row[:4]
            else:
                corners = conn_row
            return self.face_area([int(x) for x in corners])
        return self.element_volume(conn_row)

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

    def resolve_line_per_edge_tributary(
        self,
        defn: LineLoadDef,
        items: list[tuple[int, int, ndarray]],
    ) -> list[NodalLoadRecord]:
        """Tributary line-load reduction with a per-edge force-per-length.

        *items* is a list of ``(n1, n2, q_xyz)`` triples; each ``q_xyz``
        is the force-per-length vector applied to that single edge.
        Used by the composite for ``normal=True`` loads where the
        direction varies edge-by-edge.
        """
        accum: dict[int, ndarray] = {}
        for n1, n2, q in items:
            half_L = 0.5 * self.edge_length(n1, n2)
            f3 = np.asarray(q, dtype=float) * half_L
            f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
            _accumulate_nodal(accum, n1, f6)
            _accumulate_nodal(accum, n2, f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_surface_tributary(
        self,
        defn: SurfaceLoadDef,
        faces: list[list[int]],
        outwards: list[ndarray] | None = None,
    ) -> list[NodalLoadRecord]:
        """Distribute a surface load by tributary area.

        *faces* is a list of node-id lists (one per face element).
        For each face, total = ``magnitude * area`` and is split
        equally among the face's nodes.  ``normal=True`` projects
        along the face normal; otherwise the explicit direction vector.

        ``outwards``, when given, supplies a per-face physical
        outward unit normal that overrides the connectivity-derived
        :meth:`face_normal`.  This is needed for embedded crack
        faces (whose connectivity normal can disagree with physical
        outward) and for tilted faces with unpredictable connectivity
        orientation; the composite layer fills it in via
        :meth:`LoadsComposite._face_outward_normals`.  When ``None``,
        the connectivity normal is used (preserving backward compat
        for direct callers that don't go through the composite).
        """
        accum: dict[int, ndarray] = {}
        for i, face in enumerate(faces):
            A = self.face_area(face)
            if A <= 0:
                continue
            if defn.mode == "pressure":
                n = (
                    np.asarray(outwards[i], dtype=float)
                    if outwards is not None
                    else self.face_normal(face)
                )
                # Convention: positive magnitude = pressure pushing into face
                f3 = -defn.magnitude * A * n
            elif defn.mode == "shear":
                # In-plane traction: project the global reference vector
                # onto this face's tangent plane (the plane is sign-
                # independent, so the connectivity normal suffices).
                f3 = _project_in_plane(
                    np.asarray(defn.direction, dtype=float),
                    self.face_normal(face),
                ) * A
            else:  # traction
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
        dim: int = 3,
    ) -> list[NodalLoadRecord]:
        """Distribute body weight equally to element nodes.

        *elements* is a list of connectivity rows (each is an array
        of node IDs).  Each element contributes ``ρ·measure·g`` total,
        split equally among its nodes, where *measure* is the element
        volume for ``dim == 3`` and its area for ``dim == 2`` (so
        ``density`` is mass-per-volume in 3D, mass-per-area in 2D).
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
            V = self.element_measure(conn_row, dim)
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
        dim: int = 3,
    ) -> list[NodalLoadRecord]:
        """Distribute a body force equally to element nodes.

        ``force_per_volume`` is multiplied by the element measure —
        volume for ``dim == 3``, area for ``dim == 2`` (so the
        supplied intensity is force-per-area in a 2D model).
        """
        bf = np.asarray(defn.force_per_volume, dtype=float)
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_measure(conn_row, dim)
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
        edges: list,
    ) -> list[NodalLoadRecord]:
        """Consistent line-load reduction via shape-function integration.

        *edges* is a list of node-id sequences; each sequence's length
        determines element order:

            2 nodes  -> line2 (linear), 2-pt Gauss
            3 nodes  -> line3 (quadratic), 3-pt Gauss

        Any other node count raises :class:`NotImplementedError` rather
        than silently producing wrong numbers.
        """
        from .._consistent_quadrature import integrate_edge

        if defn.q_xyz is not None:
            q = np.asarray(defn.q_xyz, dtype=float)
        else:
            q = defn.magnitude * _direction_vec(defn.direction)
        accum: dict[int, ndarray] = {}
        for edge in edges:
            edge = list(edge)
            coords = np.array([self.coords_of(n) for n in edge])
            weights = integrate_edge(coords, len(edge))
            for i, nid in enumerate(edge):
                f3 = q * float(weights[i])
                f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_line_per_edge_consistent(
        self,
        defn: LineLoadDef,
        items: list[tuple[list[int], ndarray]],
    ) -> list[NodalLoadRecord]:
        """Consistent line-load reduction with a per-edge force-per-length.

        *items* is a list of ``(node_seq, q_xyz)`` pairs; each
        ``q_xyz`` is treated as constant along that edge.  Shape-
        function integration is otherwise identical to
        :meth:`resolve_line_consistent`.
        """
        from .._consistent_quadrature import integrate_edge

        accum: dict[int, ndarray] = {}
        for edge, q in items:
            edge = list(edge)
            coords = np.array([self.coords_of(n) for n in edge])
            weights = integrate_edge(coords, len(edge))
            q_arr = np.asarray(q, dtype=float)
            for i, nid in enumerate(edge):
                f3 = q_arr * float(weights[i])
                f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_line_per_edge_consistent_varying(
        self,
        defn: LineLoadDef,
        items: list,
    ) -> list[NodalLoadRecord]:
        """Consistent reduction of a **spatially varying** line load.

        *items* is a list of ``(node_seq, dir_vec, scalar_fn)`` tuples.
        For each edge the scalar force-per-length ``scalar_fn(xyz)`` is
        integrated against the shape functions at the element's Gauss
        points (:func:`integrate_edge_scaled`) and applied along the
        per-edge ``dir_vec`` (the in-plane normal for ``normal=True``,
        or the direction vector otherwise).  This is the exact
        consistent load vector to quadrature order, so a varying
        magnitude does not over/undershoot the way a single midpoint
        sample does.
        """
        from .._consistent_quadrature import integrate_edge_scaled

        accum: dict[int, ndarray] = {}
        for edge, dir_vec, scalar_fn in items:
            edge = list(edge)
            coords = np.array([self.coords_of(n) for n in edge])
            weights = integrate_edge_scaled(
                coords, len(edge), scalar_fn)
            d = np.asarray(dir_vec, dtype=float)
            for i, nid in enumerate(edge):
                f3 = d * float(weights[i])
                f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_surface_consistent(
        self,
        defn: SurfaceLoadDef,
        faces: list[list[int]],
    ) -> list[NodalLoadRecord]:
        """Consistent surface load via shape-function integration.

        Each *face* is a node-id sequence whose length determines the
        face type:

            3 -> tri3, 4 -> quad4, 6 -> tri6, 8 -> quad8, 9 -> quad9

        For ``mode="pressure"`` the pressure follows the curved face
        normal evaluated at each Gauss point.  Any other node count
        raises :class:`NotImplementedError`.

        For ``mode="shear"`` the global reference vector is projected
        onto each face's average tangent plane (exact for flat faces;
        for curved higher-order faces the face-average normal is used).
        """
        from .._consistent_quadrature import integrate_face

        d = None
        if defn.mode == "traction":
            d = np.asarray(defn.direction, dtype=float)
            d = d / (np.linalg.norm(d) + 1e-30)
        accum: dict[int, ndarray] = {}
        for face in faces:
            face = list(face)
            coords = np.array([self.coords_of(n) for n in face])
            weights, normals = integrate_face(coords, len(face))
            d_shear = None
            if defn.mode == "shear":
                # Face-average normal (∫ n dA over the face), then the
                # in-plane part of the reference vector against it.
                d_shear = _project_in_plane(
                    np.asarray(defn.direction, dtype=float),
                    np.sum(normals, axis=0),
                )
            for i, nid in enumerate(face):
                if defn.mode == "pressure":
                    # Positive magnitude = pressure pushing into face.
                    f3 = -defn.magnitude * normals[i]
                elif defn.mode == "shear":
                    # weights[i] = ∫ N_i dA; d_shear is the full in-plane
                    # traction vector (per area).
                    f3 = float(weights[i]) * d_shear
                else:  # traction
                    f3 = defn.magnitude * float(weights[i]) * d
                f6 = np.array([f3[0], f3[1], f3[2], 0.0, 0.0, 0.0])
                _accumulate_nodal(accum, int(nid), f6)
        return _accum_to_records(accum, pattern=defn.pattern, name=defn.name)

    def resolve_gravity_consistent(
        self,
        defn: GravityLoadDef,
        elements: list[ndarray],
        dim: int = 3,
    ) -> list[NodalLoadRecord]:
        """Consistent gravity reduction.

        For tet4 / hex8 with constant density, the consistent vector
        equals the tributary vector (each node gets V/n × ρ × g).
        """
        return self.resolve_gravity_tributary(defn, elements, dim)

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

    def resolve_line_element_varying(
        self,
        defn: LineLoadDef,
        items: list[tuple[int, tuple[float, float, float]]],
    ) -> list[ElementLoadRecord]:
        """Per-element ``beamUniform`` for a spatially varying line load.

        *items* is a list of ``(element_id, (wx, wy, wz))`` pairs — the
        composite has already sampled the callable ``magnitude`` at each
        element's midpoint, so each element gets its own constant
        ``beamUniform`` (the only thing OpenSees ``eleLoad`` supports).
        """
        out: list[ElementLoadRecord] = []
        for eid, (wx, wy, wz) in items:
            out.append(ElementLoadRecord(
                pattern=defn.pattern,
                name=defn.name,
                element_id=int(eid),
                load_type="beamUniform",
                params={"wx": float(wx), "wy": float(wy),
                        "wz": float(wz)},
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
                    "normal": defn.mode == "pressure",
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
        faces: list[list[int]] | None = None,
        outwards: list[ndarray] | None = None,
    ) -> list[NodalLoadRecord]:
        """Distribute centroidal force/moment to face nodes.

        ``force_xyz``: equal share ``F / N`` per node.
        ``moment_xyz``: least-norm nodal forces satisfying
        ``Sum(f_i) = 0`` and ``Sum(r_i x f_i) = M``.
        ``magnitude`` + ``normal``/``direction``: equivalent force_xyz
        derived from face geometry.  Requires ``faces`` (per-element
        node-id lists) when ``normal=True`` so the area-weighted
        average normal can be computed.

        ``outwards``, when given, supplies a per-face physical outward
        unit normal that overrides the connectivity-derived
        :meth:`face_normal` in the area-weighted average.  See the
        ``outwards=`` discussion on
        :meth:`resolve_surface_tributary`.
        """
        N = len(face_node_ids)
        if N == 0:
            return []
        accum: dict[int, ndarray] = {}

        # Magnitude path: derive an equivalent force_xyz from
        # face geometry, then equal-split like force_xyz.
        if defn.magnitude != 0.0:
            if defn.normal:
                if not faces:
                    raise ValueError(
                        "face_load with normal=True requires face "
                        "element information; got empty `faces`."
                    )
                weighted = np.zeros(3)
                total_area = 0.0
                for i, face in enumerate(faces):
                    A = self.face_area(face)
                    if A <= 0:
                        continue
                    n = (
                        np.asarray(outwards[i], dtype=float)
                        if outwards is not None
                        else self.face_normal(face)
                    )
                    weighted += A * n
                    total_area += A
                w_norm = float(np.linalg.norm(weighted))
                if w_norm < 1e-30 or total_area <= 0:
                    raise ValueError(
                        "face_load(normal=True): degenerate face "
                        "geometry (zero area or null average normal)."
                    )
                # Sign convention: magnitude * +n_avg.  Positive
                # magnitude acts along the average outward face normal.
                # Pass a negative magnitude for an "into-face"
                # (pressure-like) load.
                f_total = defn.magnitude * (weighted / w_norm)
            else:
                if defn.direction is None:
                    raise ValueError(
                        "face_load(magnitude=...) requires either "
                        "normal=True or an explicit direction= vector."
                    )
                d = np.asarray(defn.direction, dtype=float)
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-30:
                    raise ValueError(
                        "face_load(direction=...): null direction vector."
                    )
                f_total = defn.magnitude * (d / d_norm)
            per_node = f_total / N
            f6 = np.array([per_node[0], per_node[1], per_node[2],
                           0.0, 0.0, 0.0])
            for nid in face_node_ids:
                _accumulate_nodal(accum, nid, f6)

        # Explicit force_xyz: equal split
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
        faces: list[list[int]] | None = None,
        outwards: list[ndarray] | None = None,
    ) -> list[SPRecord]:
        """Map centroidal rigid-body motion to per-node SP constraints.

        For each constrained DOF *d* and each node *i*:
        ``u_i = disp_xyz + rot_xyz x r_i``, then emit
        ``SPRecord(node_id=i, dof=d, value=u_i[d-1])``.

        ``magnitude`` + ``normal``/``direction`` derive an additional
        translation contribution (along ``+n_avg`` for ``normal=True``,
        otherwise along the normalised ``direction``).  Requires
        ``faces`` when ``normal=True``.

        ``outwards``, when given, supplies a per-face physical outward
        unit normal that overrides the connectivity-derived
        :meth:`face_normal`.  See
        :meth:`resolve_surface_tributary`.
        """
        if not face_node_ids:
            return []

        coords = np.array([self.coords_of(nid) for nid in face_node_ids])
        centroid = coords.mean(axis=0)

        u0 = np.asarray(defn.disp_xyz, dtype=float) if defn.disp_xyz else np.zeros(3)
        theta = np.asarray(defn.rot_xyz, dtype=float) if defn.rot_xyz else np.zeros(3)

        # Magnitude path: derive an equivalent disp_xyz along the
        # face normal or an explicit direction.
        if defn.magnitude != 0.0:
            if defn.normal:
                if not faces:
                    raise ValueError(
                        "face_sp with normal=True requires face element "
                        "information; got empty `faces`."
                    )
                weighted = np.zeros(3)
                for i, face in enumerate(faces):
                    A = self.face_area(face)
                    if A <= 0:
                        continue
                    n = (
                        np.asarray(outwards[i], dtype=float)
                        if outwards is not None
                        else self.face_normal(face)
                    )
                    weighted += A * n
                w_norm = float(np.linalg.norm(weighted))
                if w_norm < 1e-30:
                    raise ValueError(
                        "face_sp(normal=True): degenerate face geometry."
                    )
                u0 = u0 + defn.magnitude * (weighted / w_norm)
            else:
                if defn.direction is None:
                    raise ValueError(
                        "face_sp(magnitude=...) requires normal=True or "
                        "an explicit direction= vector."
                    )
                d = np.asarray(defn.direction, dtype=float)
                d_norm = float(np.linalg.norm(d))
                if d_norm < 1e-30:
                    raise ValueError(
                        "face_sp(direction=...): null direction vector."
                    )
                u0 = u0 + defn.magnitude * (d / d_norm)

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

    def resolve_point_sp(
        self,
        defn,
        node_ids: list[int],
    ) -> list[SPRecord]:
        """Prescribed displacement/rotation applied directly at nodes.

        For each targeted node and each constrained DOF *d*
        (``defn.dofs[d] == 1``) emit ``SPRecord(node_id, dof=d+1,
        value=values[d])`` — the value taken from ``defn.values``
        (``None`` → homogeneous 0). No centroid / rigid-body mapping:
        the value is applied verbatim at every node.
        """
        if not node_ids:
            return []
        vals = defn.values
        out: list[SPRecord] = []
        for nid in node_ids:
            for d_idx, mask in enumerate(defn.dofs):
                if mask != 1:
                    continue
                val = float(vals[d_idx]) if vals is not None else 0.0
                out.append(SPRecord(
                    pattern=defn.pattern,
                    name=defn.name,
                    node_id=int(nid),
                    dof=d_idx + 1,
                    value=val,
                    is_homogeneous=(abs(val) < 1e-30),
                ))
        return out


__all__ = ["LoadResolver"]
