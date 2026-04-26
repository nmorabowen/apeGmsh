"""
Stage 2 — :class:`ConstraintResolver`.

Converts geometry-level :mod:`constraint definitions
<apeGmsh.solvers._constraint_defs>` into concrete
:mod:`records <apeGmsh.solvers._constraint_records>` by attaching
mesh data (node tags, coordinates, connectivity) and running the
appropriate geometric search / projection per constraint kind.

The resolver works with raw NumPy arrays and is solver-agnostic —
nothing in here imports Gmsh or OpenSees.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy import ndarray

from ._constraint_defs import (
    DistributingCouplingDef,
    EqualDOFDef,
    KinematicCouplingDef,
    MortarDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TieDef,
    TiedContactDef,
)
from ._constraint_geom import (
    SHAPE_FUNCTIONS,
    _SpatialIndex,
    _is_inside_parametric,
    _project_point_to_face,
)
from ._constraint_records import (
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    NodeToSurfaceRecord,
    SurfaceCouplingRecord,
)
from ._kinds import ConstraintKind


def _barycentric_tri3(
    p: ndarray,
    corners: ndarray,
) -> tuple[ndarray, float | None, ndarray]:
    """Barycentric coordinates of *p* in a tri3 with 3D *corners*.

    Returns ``(weights, excess, parametric)`` where ``weights`` are
    the three shape-function values summing to 1, ``excess`` is
    ``-min(weight)`` (0 when p is inside; positive when outside), and
    ``parametric`` is ``(v, w)`` corresponding to corners 1 and 2.

    Projects the point onto the triangle plane (handles embedded
    nodes that are slightly off-plane in 2D meshes).
    """
    A, B, C = corners[0], corners[1], corners[2]
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-30:
        return np.zeros(3), None, np.zeros(2)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    weights = np.array([u, v, w], dtype=float)
    excess = -float(weights.min())
    return weights, max(excess, 0.0), np.array([v, w], dtype=float)


def _barycentric_tet4(
    p: ndarray,
    corners: ndarray,
) -> tuple[ndarray, float | None, ndarray]:
    """Barycentric coordinates of *p* in a tet4 with 3D *corners*.

    Returns ``(weights, excess, parametric)`` where ``weights`` are
    the four shape-function values summing to 1, ``excess`` is
    ``-min(weight)`` (0 when p is inside; positive when outside), and
    ``parametric`` is ``(v, w, x)`` corresponding to corners 1, 2, 3.
    """
    A, B, C, D = corners[0], corners[1], corners[2], corners[3]
    M = np.column_stack([B - A, C - A, D - A])
    rhs = p - A
    det = float(np.linalg.det(M))
    if abs(det) < 1e-30:
        return np.zeros(4), None, np.zeros(3)
    try:
        coeffs = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(4), None, np.zeros(3)
    v, w, x = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    u = 1.0 - v - w - x
    weights = np.array([u, v, w, x], dtype=float)
    excess = -float(weights.min())
    return weights, max(excess, 0.0), np.array([v, w, x], dtype=float)


class ConstraintResolver:
    """
    Converts constraint definitions into resolved records.

    The resolver works with raw numpy arrays of node coordinates
    and connectivity — it does NOT depend on Gmsh or any solver.
    This makes it fully portable.

    Parameters
    ----------
    node_tags : ndarray, shape (n_nodes,)
        Node tags (IDs) from the mesh.
    node_coords : ndarray, shape (n_nodes, 3)
        Nodal coordinates.
    elem_tags : ndarray, shape (n_elems,)
        Element tags.
    connectivity : ndarray, shape (n_elems, n_nodes_per_elem)
        Element connectivity (node tags).
    face_connectivity : list of ndarray, optional
        Element face connectivity for surface elements.
        If ``None``, the resolver extracts faces from the
        volume connectivity.
    """

    def __init__(
        self,
        node_tags: ndarray,
        node_coords: ndarray,
        elem_tags: ndarray | None = None,
        connectivity: ndarray | None = None,
    ) -> None:
        self.node_tags = np.asarray(node_tags, dtype=int)
        self.node_coords = np.asarray(node_coords, dtype=float)

        # Tag -> index mapping
        self._tag_to_idx: dict[int, int] = {
            int(t): i for i, t in enumerate(self.node_tags)
        }

        self.elem_tags = (
            np.asarray(elem_tags, dtype=int) if elem_tags is not None
            else None
        )
        self.connectivity = (
            np.asarray(connectivity, dtype=int) if connectivity is not None
            else None
        )

        # Running high-water mark for phantom node tag generation.
        # Each resolve_node_to_surface() call advances this so that
        # multiple calls never produce overlapping phantom tag ranges.
        self._next_phantom_tag: int = int(self.node_tags.max()) + 1

        # KD-tree for spatial queries (built lazily)
        self._tree = None

    @property
    def tree(self):
        """Lazily build a KD-tree for nearest-neighbour queries."""
        if self._tree is None:
            self._tree = _SpatialIndex(self.node_coords)
        return self._tree

    def _coords_of(self, tag: int) -> ndarray:
        """Get coordinates of a node by tag."""
        return self.node_coords[self._tag_to_idx[tag]]

    def _nodes_near(
        self,
        point: ndarray | Sequence[float],
        radius: float,
    ) -> list[int]:
        """Find node tags within *radius* of *point*."""
        point = np.asarray(point, dtype=float)
        indices = self.tree.query_ball_point(point, radius)
        return [int(self.node_tags[i]) for i in indices]

    def _closest_node(
        self,
        point: ndarray | Sequence[float],
    ) -> tuple[int, float]:
        """Find the closest node tag and distance to *point*."""
        point = np.asarray(point, dtype=float)
        dist, idx = self.tree.query(point)
        return int(self.node_tags[idx]), float(dist)

    def _closest_node_in_set(
        self,
        point: ndarray | Sequence[float],
        candidates: set[int] | list[int],
    ) -> tuple[int, float]:
        """Find the closest node to *point* inside a candidate tag set."""
        candidate_list = sorted(int(tag) for tag in candidates)
        if not candidate_list:
            return self._closest_node(point)

        point = np.asarray(point, dtype=float)
        coords = np.array([self._coords_of(tag) for tag in candidate_list])
        dists = np.linalg.norm(coords - point, axis=1)
        idx = int(np.argmin(dists))
        return candidate_list[idx], float(dists[idx])

    def _match_node_pairs(
        self,
        master_tags: set[int],
        slave_tags: set[int],
        tolerance: float,
    ) -> list[tuple[int, int]]:
        """
        Find co-located (master, slave) node pairs within tolerance.

        Returns list of (master_tag, slave_tag) tuples.
        """
        # Build sub-tree from master nodes
        master_list = sorted(master_tags)
        if not master_list:
            return []
        master_coords = np.array([
            self._coords_of(t) for t in master_list
        ])
        master_tree = _SpatialIndex(master_coords)

        pairs = []
        for st in sorted(slave_tags):
            sc = self._coords_of(st)
            dist, idx = master_tree.query(sc)
            if dist <= tolerance:
                mt = master_list[idx]
                if mt != st:     # don't pair a node with itself
                    pairs.append((mt, st))

        return pairs

    # ------------------------------------------------------------------
    # Resolve methods — one per constraint level
    # ------------------------------------------------------------------

    def resolve_equal_dof(
        self,
        defn: EqualDOFDef,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> list[NodePairRecord]:
        """
        Resolve an EqualDOF definition into node pair records.

        Parameters
        ----------
        defn : EqualDOFDef
        master_nodes : set[int]
            Node tags belonging to the master instance.
        slave_nodes : set[int]
            Node tags belonging to the slave instance.
        """
        pairs = self._match_node_pairs(
            master_nodes, slave_nodes, defn.tolerance,
        )
        dofs = defn.dofs or [1, 2, 3, 4, 5, 6]
        return [
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                name=defn.name,
                master_node=mt,
                slave_node=st,
                dofs=list(dofs),
            )
            for mt, st in pairs
        ]

    def resolve_rigid_link(
        self,
        defn: RigidLinkDef,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> list[NodePairRecord]:
        """
        Resolve a rigid link definition.

        If ``master_point`` is specified, find the closest master node.
        Then link all slave nodes to that master via rigid offset.
        """
        # Find master node
        if defn.master_point is not None:
            master_tag, _ = self._closest_node_in_set(defn.master_point, master_nodes)
        else:
            if master_nodes:
                coords = np.array([self._coords_of(t) for t in master_nodes])
                centroid = coords.mean(axis=0)
                master_tag, _ = self._closest_node_in_set(centroid, master_nodes)
            else:
                centroid = self.node_coords.mean(axis=0)
                master_tag, _ = self._closest_node(centroid)

        master_xyz = self._coords_of(master_tag)
        kind = f"rigid_{defn.link_type}"

        if kind == ConstraintKind.RIGID_BEAM:
            dofs = [1, 2, 3, 4, 5, 6]
        else:
            dofs = [1, 2, 3]

        records = []
        for st in sorted(slave_nodes):
            if st == master_tag:
                continue
            slave_xyz = self._coords_of(st)
            offset = slave_xyz - master_xyz
            records.append(NodePairRecord(
                kind=kind,
                name=defn.name,
                master_node=master_tag,
                slave_node=st,
                dofs=list(dofs),
                offset=offset,
            ))
        return records

    def resolve_penalty(
        self,
        defn: PenaltyDef,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> list[NodePairRecord]:
        """Resolve a penalty definition into node pair records."""
        pairs = self._match_node_pairs(
            master_nodes, slave_nodes, defn.tolerance,
        )
        dofs = defn.dofs or [1, 2, 3, 4, 5, 6]
        return [
            NodePairRecord(
                kind=ConstraintKind.PENALTY,
                name=defn.name,
                master_node=mt,
                slave_node=st,
                dofs=list(dofs),
                penalty_stiffness=defn.stiffness,
            )
            for mt, st in pairs
        ]

    def resolve_rigid_diaphragm(
        self,
        defn: RigidDiaphragmDef,
        all_nodes: set[int],
    ) -> NodeGroupRecord:
        """
        Resolve a rigid diaphragm.

        Collects all nodes within ``plane_tolerance`` of the diaphragm
        plane, then the closest to ``master_point`` becomes master.
        """
        normal = np.asarray(defn.plane_normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        mp = np.asarray(defn.master_point, dtype=float)
        d = np.dot(normal, mp)

        # Collect nodes near the plane
        plane_nodes = []
        for tag in all_nodes:
            c = self._coords_of(tag)
            dist_to_plane = abs(np.dot(normal, c) - d)
            if dist_to_plane <= defn.plane_tolerance:
                plane_nodes.append(tag)

        if not plane_nodes:
            return NodeGroupRecord(
                kind=ConstraintKind.RIGID_DIAPHRAGM,
                name=defn.name,
                dofs=list(defn.constrained_dofs),
            )

        # Find master: closest to master_point
        master_tag, _ = self._closest_node(mp)
        if master_tag not in plane_nodes:
            # Pick the closest plane node instead
            dists = [np.linalg.norm(self._coords_of(t) - mp)
                     for t in plane_nodes]
            master_tag = plane_nodes[int(np.argmin(dists))]

        slave_tags = [t for t in plane_nodes if t != master_tag]
        master_xyz = self._coords_of(master_tag)
        offsets = np.array([
            self._coords_of(t) - master_xyz for t in slave_tags
        ]) if slave_tags else None

        return NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            name=defn.name,
            master_node=master_tag,
            slave_nodes=slave_tags,
            dofs=list(defn.constrained_dofs),
            offsets=offsets,
            plane_normal=normal,
        )

    def resolve_kinematic_coupling(
        self,
        defn: KinematicCouplingDef | RigidBodyDef,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> NodeGroupRecord:
        """
        Resolve kinematic coupling or rigid body constraint.
        """
        master_tag, _ = self._closest_node_in_set(defn.master_point, master_nodes)
        master_xyz = self._coords_of(master_tag)

        slaves = sorted(slave_nodes - {master_tag})
        offsets = np.array([
            self._coords_of(t) - master_xyz for t in slaves
        ]) if slaves else None

        if isinstance(defn, RigidBodyDef):
            dofs = [1, 2, 3, 4, 5, 6]
        else:
            dofs = list(defn.dofs)

        return NodeGroupRecord(
            kind=defn.kind,
            name=defn.name,
            master_node=master_tag,
            slave_nodes=slaves,
            dofs=dofs,
            offsets=offsets,
        )

    def resolve_tie(
        self,
        defn: TieDef,
        master_face_conn: ndarray,
        slave_nodes: set[int],
    ) -> list[InterpolationRecord]:
        """
        Resolve a surface tie via closest-point projection.

        For each slave node, find the closest master face, project
        onto it, and compute shape function weights.

        Parameters
        ----------
        defn : TieDef
        master_face_conn : ndarray, shape (n_faces, n_nodes_per_face)
            Connectivity of master surface element faces (node tags).
        slave_nodes : set[int]
            Slave node tags to project.

        Returns
        -------
        list[InterpolationRecord]
        """
        dofs = defn.dofs or [1, 2, 3]
        records = []

        # Pre-compute face centroids for quick nearest-face search
        n_faces = master_face_conn.shape[0]
        n_fpn = master_face_conn.shape[1]
        face_centroids = np.zeros((n_faces, 3))
        face_coords_list = []
        for fi in range(n_faces):
            nodes = master_face_conn[fi]
            coords = np.array([self._coords_of(int(n)) for n in nodes])
            face_coords_list.append(coords)
            face_centroids[fi] = coords.mean(axis=0)

        face_tree = _SpatialIndex(face_centroids)

        for st in sorted(slave_nodes):
            s_xyz = self._coords_of(st)

            # Find K nearest face centroids, try projection on each
            K = min(5, n_faces)
            _, face_indices = face_tree.query(s_xyz, k=K)
            if isinstance(face_indices, (int, np.integer)):
                face_indices = [face_indices]

            best_dist = float('inf')
            best_record = None

            for fi in face_indices:
                fi = int(fi)
                fc = face_coords_list[fi]
                fn = master_face_conn[fi]

                try:
                    xi_eta, proj, dist = _project_point_to_face(s_xyz, fc)
                except Exception:
                    continue

                if dist > defn.tolerance:
                    continue

                if not _is_inside_parametric(xi_eta, n_fpn):
                    continue

                if dist < best_dist:
                    best_dist = dist
                    shape_fn = SHAPE_FUNCTIONS[n_fpn]
                    weights = shape_fn(xi_eta[0], xi_eta[1])

                    best_record = InterpolationRecord(
                        kind=ConstraintKind.TIE,
                        name=defn.name,
                        slave_node=st,
                        master_nodes=[int(n) for n in fn],
                        weights=weights,
                        dofs=list(dofs),
                        projected_point=proj,
                        parametric_coords=xi_eta,
                    )

            if best_record is not None:
                records.append(best_record)

        return records

    def resolve_distributing(
        self,
        defn: DistributingCouplingDef,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> InterpolationRecord:
        """
        Resolve a distributing coupling.

        Computes weights for each slave node based on weighting scheme.
        """
        dofs = defn.dofs or [1, 2, 3]
        master_tag, _ = self._closest_node_in_set(defn.master_point, master_nodes)

        slave_list = sorted(slave_nodes - {master_tag})
        n = len(slave_list)

        if defn.weighting == "uniform":
            weights = np.ones(n) / n
        else:
            # Area weighting: approximate by Voronoi-like partition
            # (simplified: weight by inverse distance from centroid)
            coords = np.array([self._coords_of(t) for t in slave_list])
            centroid = coords.mean(axis=0)
            dists = np.linalg.norm(coords - centroid, axis=1)
            dists = np.maximum(dists, 1e-12)
            w = 1.0 / dists
            weights = w / w.sum()

        return InterpolationRecord(
            kind=ConstraintKind.DISTRIBUTING,
            name=defn.name,
            slave_node=master_tag,        # "slave" is the ref point here
            master_nodes=slave_list,       # "masters" are the surface nodes
            weights=weights,
            dofs=list(dofs),
        )

    def resolve_tied_contact(
        self,
        defn: TiedContactDef,
        master_face_conn: ndarray,
        slave_face_conn: ndarray,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> SurfaceCouplingRecord:
        """
        Resolve a full surface-to-surface tie.

        Projects slave nodes onto master faces AND master nodes onto
        slave faces (bidirectional), then keeps the best projection
        for each node.
        """
        dofs = defn.dofs or [1, 2, 3]

        # Forward: slave nodes -> master faces
        tie_fwd = TieDef(
            master_label=defn.master_label,
            slave_label=defn.slave_label,
            tolerance=defn.tolerance,
            dofs=dofs,
        )
        fwd_records = self.resolve_tie(
            tie_fwd, master_face_conn, slave_nodes,
        )

        # Backward: master nodes -> slave faces
        tie_bwd = TieDef(
            master_label=defn.slave_label,
            slave_label=defn.master_label,
            tolerance=defn.tolerance,
            dofs=dofs,
        )
        bwd_records = self.resolve_tie(
            tie_bwd, slave_face_conn, master_nodes,
        )

        all_records = fwd_records + bwd_records
        return SurfaceCouplingRecord(
            kind=ConstraintKind.TIED_CONTACT,
            name=defn.name,
            slave_records=all_records,
            master_nodes=sorted(master_nodes),
            slave_nodes=sorted(slave_nodes),
            dofs=list(dofs),
        )

    def resolve_mortar(
        self,
        defn: MortarDef,
        master_face_conn: ndarray,
        slave_face_conn: ndarray,
        master_nodes: set[int],
        slave_nodes: set[int],
    ) -> SurfaceCouplingRecord:
        """
        Resolve a mortar coupling.

        .. note::

           The mortar operator requires numerical integration over
           the overlapping surface segments -- a significant algorithm.
           This implementation provides the *architecture* (the
           SurfaceCouplingRecord with mortar_operator field) but uses
           a **simplified** node-to-surface projection as a placeholder.

           For production mortar coupling, implement the segment-based
           integration following Puso & Laursen (2004) or
           Popp et al. (2010).
        """
        dofs = defn.dofs or [1, 2, 3]

        # Placeholder: use tied_contact projection as approximation
        tied = TiedContactDef(
            master_label=defn.master_label,
            slave_label=defn.slave_label,
            tolerance=10.0,   # generous for mortar
            dofs=dofs,
        )
        tied_result = self.resolve_tied_contact(
            tied,
            master_face_conn, slave_face_conn,
            master_nodes, slave_nodes,
        )

        # Build approximate mortar operator from interpolation records
        m_list = sorted(master_nodes)
        s_list = sorted(slave_nodes)
        m_idx = {t: i for i, t in enumerate(m_list)}
        s_idx = {t: i for i, t in enumerate(s_list)}
        nd = len(dofs)

        B = np.zeros((len(s_list) * nd, len(m_list) * nd))
        for rec in tied_result.slave_records:
            if rec.slave_node in s_idx:
                si = s_idx[rec.slave_node]
                w = rec.weights if rec.weights is not None else np.zeros(0)
                for j, mn in enumerate(rec.master_nodes):
                    if mn in m_idx:
                        mi = m_idx[mn]
                        for d in range(nd):
                            B[si * nd + d, mi * nd + d] = w[j]

        return SurfaceCouplingRecord(
            kind=ConstraintKind.MORTAR,
            name=defn.name,
            slave_records=tied_result.slave_records,
            mortar_operator=B,
            master_nodes=m_list,
            slave_nodes=s_list,
            dofs=list(dofs),
        )

    def resolve_node_to_surface(
        self,
        defn: NodeToSurfaceDef,
        master_tag: int,
        slave_nodes: set[int],
    ) -> NodeToSurfaceRecord:
        """
        Resolve a 6-DOF node to 3-DOF surface coupling.

        Steps:

        1. Use the master node tag directly (already resolved from
           ``master_label`` as bare node tag).
        2. Generate phantom node tags — one per slave, starting at
           ``max(all_existing_tags) + 1``.
        3. Build rigid-beam records: master -> each phantom.
        4. Build equalDOF records: each phantom -> original slave
           (translations only).

        Parameters
        ----------
        defn : NodeToSurfaceDef
        master_tag : int
            The 6-DOF master node tag (dim=0).
        slave_nodes : set[int]
            Node tags belonging to the slave surface (dim=2, 3-DOF).

        Returns
        -------
        NodeToSurfaceRecord
        """

        master_xyz = self._coords_of(master_tag)
        slave_list = sorted(slave_nodes - {master_tag})
        dofs = defn.dofs or [1, 2, 3]

        # -- 2. Generate phantom node tags (unique across calls) --
        start = self._next_phantom_tag
        phantom_tags = list(range(start, start + len(slave_list)))
        self._next_phantom_tag = start + len(slave_list)

        phantom_coords = np.array([
            self._coords_of(t) for t in slave_list
        ])

        # -- 3. Rigid beam: master -> phantom --
        # No dofs list: OpenSees `rigidLink('beam', ...)` picks DOFs
        # from the model's ndf at emit time. The caller's DOF space is
        # not known at resolve time and apeGmsh refuses to guess.
        rigid_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            slave_xyz = self._coords_of(slave_tag)
            offset = slave_xyz - master_xyz
            rigid_records.append(NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                name=defn.name,
                master_node=master_tag,
                slave_node=phantom_tag,
                offset=offset,
            ))

        # -- 4. EqualDOF: phantom -> slave (translations only) --
        edof_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            edof_records.append(NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                name=defn.name,
                master_node=phantom_tag,
                slave_node=slave_tag,
                dofs=list(dofs),
            ))

        return NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE,
            name=defn.name,
            master_node=master_tag,
            slave_nodes=slave_list,
            phantom_nodes=phantom_tags,
            phantom_coords=phantom_coords,
            rigid_link_records=rigid_records,
            equal_dof_records=edof_records,
            dofs=list(dofs),
        )

    def resolve_embedded(
        self,
        defn,
        host_elems: ndarray,
        embedded_nodes: set[int] | list[int],
    ) -> list[InterpolationRecord]:
        """
        Resolve an embedded-element constraint.

        Each embedded node is located inside a host element (tri3 in
        2D or tet4 in 3D) via barycentric coordinates. The resulting
        shape-function weights couple the embedded node to the host
        element's corner nodes, matching the kinematics of
        ``ASDEmbeddedNodeElement`` in OpenSees.

        Parameters
        ----------
        defn : EmbeddedDef
            Only ``defn.tolerance`` and ``defn.name`` are consulted.
        host_elems : ndarray, shape (n_elems, 3 | 4)
            Node-tag connectivity of the host elements. A row of 3
            is treated as tri3; a row of 4 is treated as tet4.
        embedded_nodes : iterable of int
            Node tags to embed.

        Returns
        -------
        list[InterpolationRecord]
            One record per embedded node successfully located.
        """
        host_elems = np.asarray(host_elems, dtype=int)
        if host_elems.ndim != 2 or host_elems.shape[0] == 0:
            return []

        npe = int(host_elems.shape[1])
        if npe not in (3, 4):
            raise ValueError(
                f"resolve_embedded: host elements must be tri3 (npe=3) "
                f"or tet4 (npe=4), got npe={npe}"
            )

        # Pre-compute corner coords per element and centroids for the
        # nearest-element search.
        n_elems = host_elems.shape[0]
        host_coords = np.zeros((n_elems, npe, 3), dtype=float)
        for ei in range(n_elems):
            for ni in range(npe):
                host_coords[ei, ni] = self._coords_of(int(host_elems[ei, ni]))
        centroids = host_coords.mean(axis=1)
        centroid_tree = _SpatialIndex(centroids)

        tol = float(defn.tolerance)
        # Barycentric out-of-element tolerance is unitless; keep it
        # small so we don't falsely claim a node sits inside an element
        # it is only grazing.
        bary_tol = 1e-6

        records: list[InterpolationRecord] = []
        K = min(16, n_elems)

        for en in sorted(int(t) for t in embedded_nodes):
            p = self._coords_of(en)
            _, cand = centroid_tree.query(p, k=K)
            if isinstance(cand, (int, np.integer)):
                cand = [int(cand)]
            else:
                cand = [int(c) for c in np.atleast_1d(cand)]

            best_record: InterpolationRecord | None = None
            best_excess = float("inf")

            for ei in cand:
                corners = host_coords[ei]
                if npe == 3:
                    weights, excess, xi_eta = _barycentric_tri3(p, corners)
                    parametric = xi_eta
                else:
                    weights, excess, xi_etz = _barycentric_tet4(p, corners)
                    parametric = xi_etz

                if excess is None:
                    continue

                # "Inside" when all barycentric coords are non-negative
                # within bary_tol. Take the first hit; if none is fully
                # inside, keep the one with the smallest excess so the
                # caller can inspect via the log.
                if excess < best_excess:
                    best_excess = excess
                    best_record = InterpolationRecord(
                        kind=ConstraintKind.EMBEDDED,
                        name=defn.name,
                        slave_node=en,
                        master_nodes=[int(t) for t in host_elems[ei]],
                        weights=weights,
                        dofs=[1, 2, 3],
                        projected_point=p.copy(),
                        parametric_coords=parametric,
                    )
                if excess <= bary_tol:
                    break

            if best_record is None:
                continue

            # Use defn.tolerance as a soft gate on barycentric excess
            # scaled by a characteristic host edge length. We simply
            # accept any located record; the caller (composite) decides
            # whether to warn.
            records.append(best_record)

        return records

    def resolve_node_to_surface_spring(
        self,
        defn: "NodeToSurfaceSpringDef",
        master_tag: int,
        slave_nodes: set[int],
    ) -> NodeToSurfaceRecord:
        """
        Resolve a spring-variant 6-DOF → 3-DOF surface coupling.

        Identical phantom-node generation and equalDOF records as
        :meth:`resolve_node_to_surface`. The only difference is that
        the master → phantom rigid-link records are tagged with
        ``kind='rigid_beam_stiff'`` so they are routed through
        ``stiff_beam_groups()`` at emission time (becoming stiff
        ``elasticBeamColumn`` elements) instead of
        ``rigid_link_groups()`` (which would emit ``rigidLink`` and
        hit the ill-conditioning described in
        :class:`NodeToSurfaceSpringDef`).
        """

        master_xyz = self._coords_of(master_tag)
        slave_list = sorted(slave_nodes - {master_tag})
        dofs = defn.dofs or [1, 2, 3]

        start = self._next_phantom_tag
        phantom_tags = list(range(start, start + len(slave_list)))
        self._next_phantom_tag = start + len(slave_list)

        phantom_coords = np.array([
            self._coords_of(t) for t in slave_list
        ])

        # Stiff beams: master → phantom. Same structure as the
        # constraint-based variant but tagged with a distinct kind so
        # the mesh iterators can route them to the element emission
        # path.
        stiff_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            slave_xyz = self._coords_of(slave_tag)
            offset = slave_xyz - master_xyz
            stiff_records.append(NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM_STIFF,
                name=defn.name,
                master_node=master_tag,
                slave_node=phantom_tag,
                offset=offset,
            ))

        edof_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            edof_records.append(NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                name=defn.name,
                master_node=phantom_tag,
                slave_node=slave_tag,
                dofs=list(dofs),
            ))

        return NodeToSurfaceRecord(
            kind=ConstraintKind.NODE_TO_SURFACE_SPRING,
            name=defn.name,
            master_node=master_tag,
            slave_nodes=slave_list,
            phantom_nodes=phantom_tags,
            phantom_coords=phantom_coords,
            rigid_link_records=stiff_records,
            equal_dof_records=edof_records,
            dofs=list(dofs),
        )


__all__ = ["ConstraintResolver"]
