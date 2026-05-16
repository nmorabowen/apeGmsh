"""MassResolver — pure mesh math that turns MassDef into MassRecord.

Receives raw arrays (node tags, coords, optional connectivity) plus a
mass definition, and returns a list of resolved per-node
:class:`~apeGmsh.mesh.records._masses.MassRecord` entries.  No Gmsh
queries (the composite handles target resolution before calling
here), no solver imports.

Two reduction strategies are supported:

* **lumped** — equal split per node on the translational DOFs listed
  in ``defn.dofs`` (default: 1, 2, 3) plus optional fixed rotational
  inertia from ``defn.rotational``.
* **consistent** — HRZ (Hinton–Rock–Zienkiewicz) diagonal-scaling
  lumping.  Each element's total mass is distributed in proportion to
  ``∫ N_I² dΩ_ref`` (normalized), via :mod:`apeGmsh.fem._hrz`.  For
  first-order elements (tet4, hex8, tri3, quad4, line2, wedge6) the
  weights are all ``1/n`` so this is *bit-identical to lumped*; for
  higher-order elements (tet10, hex20, hex27, tri6, quad8, quad9)
  mid-edge / face / center nodes get their physically-correct larger
  share instead of an incorrect equal split.

  HRZ still emits a strictly *diagonal* nodal mass — it is the correct
  diagonalized mass, not a true consistent matrix.  The off-diagonal
  node coupling of a genuine consistent matrix has no destination
  through ``ops.mass``; for that, use the element's own ``-cMass``
  flag at the OpenSees level and skip this composite for that region.
  The HRZ weights are computed on the reference element (affine
  Jacobian) — exact for parallelepiped/affine elements, a standard
  approximation for curved higher-order ones.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from apeGmsh.core.masses.defs import (
    LineMassDef,
    PointMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)
from apeGmsh.fem._hrz import hrz_weights, surface_code, volume_code
from apeGmsh.mesh.records._masses import MassRecord


# ======================================================================
# Helpers
# ======================================================================

def _accumulate(accum: dict[int, ndarray], node_id: int, vec6: ndarray) -> None:
    """Accumulate a length-6 mass vector into the per-node sum."""
    if node_id in accum:
        accum[node_id] += vec6
    else:
        accum[node_id] = vec6.copy()


def _accum_to_records(accum: dict[int, ndarray], *, name: str | None) -> list[MassRecord]:
    out: list[MassRecord] = []
    for nid, vec in accum.items():
        out.append(MassRecord(
            node_id=int(nid),
            mass=tuple(float(v) for v in vec),
            name=name,
        ))
    return out


def _build_vec6(
    m_share: float,
    dofs: list[int] | tuple[int, ...] | None,
    rotational: tuple[float, float, float] | None,
) -> ndarray:
    """Assemble a length-6 mass vector honoring ``dofs`` + ``rotational``.

    Translational positions are 0,1,2 (1-based DOFs 1,2,3); rotational
    are 3,4,5 (DOFs 4,5,6).  Composes cleanly: ``dofs=[1,2]`` zeroes the
    z-translational slot but leaves the rotational tuple alone.
    """
    vec = np.zeros(6)
    if dofs is None:
        vec[0] = vec[1] = vec[2] = m_share
    else:
        for d in dofs:
            vec[d - 1] = m_share
    if rotational is not None:
        vec[3] = float(rotational[0])
        vec[4] = float(rotational[1])
        vec[5] = float(rotational[2])
    return vec


# ======================================================================
# MassResolver
# ======================================================================

class MassResolver:
    """Convert :class:`MassDef` instances to :class:`MassRecord` lists.

    Pure mesh math — receives raw arrays, returns record lists.
    The composite handles target -> mesh-entity resolution before
    calling these methods.
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
        self._node_to_idx = {int(t): i for i, t in enumerate(self.node_tags)}

    # ------------------------------------------------------------------
    # Geometry helpers (shared with LoadResolver semantics)
    # ------------------------------------------------------------------

    def coords_of(self, node_id: int) -> ndarray:
        return self.node_coords[self._node_to_idx[int(node_id)]]

    def edge_length(self, n1: int, n2: int) -> float:
        return float(np.linalg.norm(self.coords_of(n1) - self.coords_of(n2)))

    def face_area(self, node_ids: list[int]) -> float:
        if len(node_ids) < 3:
            return 0.0
        p0 = self.coords_of(node_ids[0])
        area = 0.0
        for i in range(1, len(node_ids) - 1):
            p1 = self.coords_of(node_ids[i])
            p2 = self.coords_of(node_ids[i + 1])
            area += 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))
        return area

    def element_volume(self, conn_row: ndarray) -> float:
        n = len(conn_row)
        pts = np.array([self.coords_of(int(nid)) for nid in conn_row])
        if n == 4:
            v = np.abs(np.dot(pts[1] - pts[0], np.cross(pts[2] - pts[0], pts[3] - pts[0]))) / 6.0
            return float(v)
        if n == 8:
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
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        return float(np.prod(mx - mn))

    # ------------------------------------------------------------------
    # Lumped mass — translational only, equal split per node
    # ------------------------------------------------------------------

    def resolve_point_lumped(
        self,
        defn: PointMassDef,
        node_set: set[int],
    ) -> list[MassRecord]:
        """Apply the same point mass to every node in *node_set*.

        Honors ``defn.dofs`` (translational mask) and
        ``defn.rotational`` (rotational inertia tuple).
        """
        vec = _build_vec6(float(defn.mass), defn.dofs, defn.rotational)
        accum: dict[int, ndarray] = {}
        for nid in node_set:
            _accumulate(accum, nid, vec)
        return _accum_to_records(accum, name=defn.name)

    def resolve_line_lumped(
        self,
        defn: LineMassDef,
        edges: list[tuple[int, int]],
    ) -> list[MassRecord]:
        """Distribute line mass by tributary length.

        Each node receives ``ρₗ × Σ(adjacent_edge_length / 2)`` of
        translational mass on the DOFs listed in ``defn.dofs``
        (default: 1, 2, 3).  ``defn.rotational`` (if set) attaches
        a fixed rotational inertia to every receiving node.
        """
        rho_l = float(defn.linear_density)
        dofs = defn.dofs
        rot = defn.rotational
        accum: dict[int, ndarray] = {}
        for n1, n2 in edges:
            half_L = 0.5 * self.edge_length(n1, n2)
            m_share = rho_l * half_L
            vec = _build_vec6(m_share, dofs, rot)
            _accumulate(accum, n1, vec)
            _accumulate(accum, n2, vec)
        return _accum_to_records(accum, name=defn.name)

    def resolve_surface_lumped(
        self,
        defn: SurfaceMassDef,
        faces: list[list[int]],
    ) -> list[MassRecord]:
        """Distribute surface mass by tributary area.

        Honors ``defn.dofs`` (translational mask) and
        ``defn.rotational`` (rotational inertia tuple).
        """
        rho_a = float(defn.areal_density)
        dofs = defn.dofs
        rot = defn.rotational
        accum: dict[int, ndarray] = {}
        for face in faces:
            A = self.face_area(face)
            if A <= 0:
                continue
            m_share = rho_a * A / len(face)
            vec = _build_vec6(m_share, dofs, rot)
            for nid in face:
                _accumulate(accum, int(nid), vec)
        return _accum_to_records(accum, name=defn.name)

    def resolve_volume_lumped(
        self,
        defn: VolumeMassDef,
        elements: list[ndarray],
    ) -> list[MassRecord]:
        """Distribute volume mass equally to element nodes (lumped).

        Honors ``defn.dofs`` (translational mask) and
        ``defn.rotational`` (rotational inertia tuple).
        """
        rho = float(defn.density)
        dofs = defn.dofs
        rot = defn.rotational
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_volume(conn_row)
            if V <= 0:
                continue
            m_share = rho * V / len(conn_row)
            vec = _build_vec6(m_share, dofs, rot)
            for nid in conn_row:
                _accumulate(accum, int(nid), vec)
        return _accum_to_records(accum, name=defn.name)

    # ------------------------------------------------------------------
    # Consistent mass — full ∫ρ N N dV (couples nodes within element)
    # ------------------------------------------------------------------

    def resolve_point_consistent(
        self,
        defn: PointMassDef,
        node_set: set[int],
    ) -> list[MassRecord]:
        """Point mass is unambiguous — same as lumped."""
        return self.resolve_point_lumped(defn, node_set)

    def resolve_line_consistent(
        self,
        defn: LineMassDef,
        edges: list[tuple[int, int]],
    ) -> list[MassRecord]:
        """Consistent line mass for 2-node line element.

        For a line element with constant linear density ρₗ and length L::

            M_consistent = ρₗL/6 · [[2, 1], [1, 2]]

        Each node receives the diagonal entry (2ρₗL/6) plus a coupling
        term to the other node (ρₗL/6).  Lumped is `M = ρₗL/2 · I`,
        so the diagonal sum is the same — but consistent has
        off-diagonal coupling.

        Per-node accumulation matches lumped at the diagonal level
        (2ρₗL/6 + ρₗL/6 = ρₗL/2).  This implementation emits the
        diagonal-equivalent (which matches lumped numerically for
        2-node lines) — true off-diagonal coupling requires the
        solver to support consistent element mass via ``-cMass``
        or equivalent.
        """
        rho_l = float(defn.linear_density)
        accum: dict[int, ndarray] = {}
        for n1, n2 in edges:
            L = self.edge_length(n1, n2)
            # Consistent diagonal contribution: (2/6 + 1/6)·ρₗL = ρₗL/2
            m_share = 0.5 * rho_l * L
            vec = np.array([m_share, m_share, m_share, 0.0, 0.0, 0.0])
            _accumulate(accum, n1, vec)
            _accumulate(accum, n2, vec)
        return _accum_to_records(accum, name=defn.name)

    def _hrz_distribute(
        self,
        accum: dict[int, ndarray],
        conn,
        m_elem: float,
        gmsh_code: "int | None",
        dofs,
        rot,
    ) -> None:
        """Spread one element's total mass to its nodes via HRZ weights.

        HRZ (Hinton–Rock–Zienkiewicz) lumping distributes ``m_elem`` in
        proportion to ``∫ N_I² dΩ_ref`` (normalized).  For first-order
        elements every weight is ``1/n`` so this is bit-identical to
        equal-share.  For higher-order elements mid-edge / face / center
        nodes get their physically-correct larger share.

        Falls back to equal-share if ``gmsh_code`` is unknown (an
        element type not in the shape-function catalog).
        """
        n = len(conn)
        if gmsh_code is None:
            weights = [1.0 / n] * n
        else:
            weights = hrz_weights(gmsh_code)
            if len(weights) != n:
                # Connectivity node count disagrees with the catalog
                # entry (unexpected ordering / partial element) — stay
                # safe with equal-share rather than mis-weighting.
                weights = [1.0 / n] * n
        for w, nid in zip(weights, conn):
            _accumulate(accum, int(nid), _build_vec6(m_elem * w, dofs, rot))

    def resolve_surface_consistent(
        self,
        defn: SurfaceMassDef,
        faces: list[list[int]],
    ) -> list[MassRecord]:
        """HRZ-lumped surface mass.

        Distributes ``ρ_a · A`` per face to its nodes using HRZ
        weights keyed by node count (tri3/quad4 → equal-share by
        construction; tri6/quad8/quad9 → mid-side nodes correctly
        weighted).  Honors ``defn.dofs`` and ``defn.rotational``.
        """
        rho_a = float(defn.areal_density)
        dofs = defn.dofs
        rot = defn.rotational
        accum: dict[int, ndarray] = {}
        for face in faces:
            A = self.face_area(face)
            if A <= 0:
                continue
            self._hrz_distribute(
                accum, face, rho_a * A, surface_code(len(face)), dofs, rot,
            )
        return _accum_to_records(accum, name=defn.name)

    def resolve_volume_consistent(
        self,
        defn: VolumeMassDef,
        elements: list[ndarray],
    ) -> list[MassRecord]:
        """HRZ-lumped volume mass.

        Distributes ``ρ · V`` per element to its nodes using HRZ
        weights keyed by node count (tet4/hex8/wedge6 → equal-share by
        construction; tet10/hex20/hex27 → higher-order nodes correctly
        weighted).  Honors ``defn.dofs`` and ``defn.rotational``.

        The element volume still comes from :meth:`element_volume`,
        which is exact for tet4 / hex8 and a bounding-box approximation
        for higher-order types — proper isoparametric volume
        integration is a separate follow-up.  The HRZ *distribution*
        is correct regardless.
        """
        rho = float(defn.density)
        dofs = defn.dofs
        rot = defn.rotational
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_volume(conn_row)
            if V <= 0:
                continue
            self._hrz_distribute(
                accum, conn_row, rho * V,
                volume_code(len(conn_row)), dofs, rot,
            )
        return _accum_to_records(accum, name=defn.name)


__all__ = ["MassResolver"]
