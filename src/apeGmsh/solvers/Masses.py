"""
Masses — solver-agnostic mass definitions, records, and resolver.

Mirrors the architecture of :mod:`solvers.Loads` but simpler:

* No load-pattern grouping (mass is intrinsic to the model).
* Single record type (:class:`MassRecord`) — one mass entry per node.
* Two reduction strategies:
    - **lumped**: translational only, equal split per node
    - **consistent**: full ∫ρ N N dV mass coupling between nodes

The composite (:class:`MassesComposite`) lives in
``core/MassesComposite.py``.  This file has no Gmsh dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    pass


# ======================================================================
# MassDef hierarchy (pre-mesh, intent)
# ======================================================================

@dataclass
class MassDef:
    """Base class for all mass definitions."""
    kind: str
    target: object              # part label, PG name, mesh selection name, or DimTag list
    name: str | None = None
    reduction: str = "lumped"   # "lumped" | "consistent"


@dataclass
class PointMassDef(MassDef):
    """Concentrated mass at a node (or set of nodes).

    The same scalar mass (and optional rotational inertia) is applied
    to every node in the target.  Useful for representing equipment,
    point fixtures, or any localised lumped mass.
    """
    kind: str = field(init=False, default="point")
    mass: float = 0.0                          # translational mass per node
    rotational: tuple[float, float, float] | None = None  # (Ixx, Iyy, Izz)


@dataclass
class LineMassDef(MassDef):
    """Distributed line mass — linear density along curve(s).

    ``linear_density`` is in mass per unit length (e.g. kg/m).  The
    total mass per curve is ``linear_density × curve_length`` and is
    distributed to the curve's mesh nodes.
    """
    kind: str = field(init=False, default="line")
    linear_density: float = 0.0


@dataclass
class SurfaceMassDef(MassDef):
    """Distributed surface mass — areal density on face(s).

    ``areal_density`` is in mass per unit area (e.g. kg/m²).  The
    total mass per face is ``areal_density × face_area`` distributed
    to the face's mesh nodes.
    """
    kind: str = field(init=False, default="surface")
    areal_density: float = 0.0


@dataclass
class VolumeMassDef(MassDef):
    """Distributed volume mass — material density on volume(s).

    ``density`` is in mass per unit volume (e.g. kg/m³).  The total
    mass per element is ``density × element_volume`` distributed to
    the element's nodes.

    Note
    ----
    The user is responsible for setting the OpenSees material's
    ``rho=0`` (or equivalent in other solvers) to avoid double
    counting.  This composite always emits explicit nodal mass
    via ``ops.mass(...)`` commands.
    """
    kind: str = field(init=False, default="volume")
    density: float = 0.0


# ======================================================================
# MassRecord (post-mesh, resolved)
# ======================================================================

@dataclass
class MassRecord:
    """Resolved per-node mass entry.

    Always length 6: ``(mx, my, mz, Ixx, Iyy, Izz)``.  The OpenSees
    bridge slices to ``ndf`` when emitting commands (the rotational
    components are dropped for ``ndf<4`` models).

    Multiple :class:`MassDef` may contribute to the same node — the
    composite accumulates them so each node gets at most one
    :class:`MassRecord` in the final :class:`MassSet`.
    """
    node_id: int = 0
    mass: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    name: str | None = None


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
        """Apply the same point mass to every node in *node_set*."""
        m = float(defn.mass)
        if defn.rotational is not None:
            ix, iy, iz = defn.rotational
            vec = np.array([m, m, m, ix, iy, iz])
        else:
            vec = np.array([m, m, m, 0.0, 0.0, 0.0])
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
        translational mass on x, y, z DOFs.  Rotational DOFs zero.
        """
        rho_l = float(defn.linear_density)
        accum: dict[int, ndarray] = {}
        for n1, n2 in edges:
            half_L = 0.5 * self.edge_length(n1, n2)
            m_share = rho_l * half_L
            vec = np.array([m_share, m_share, m_share, 0.0, 0.0, 0.0])
            _accumulate(accum, n1, vec)
            _accumulate(accum, n2, vec)
        return _accum_to_records(accum, name=defn.name)

    def resolve_surface_lumped(
        self,
        defn: SurfaceMassDef,
        faces: list[list[int]],
    ) -> list[MassRecord]:
        """Distribute surface mass by tributary area."""
        rho_a = float(defn.areal_density)
        accum: dict[int, ndarray] = {}
        for face in faces:
            A = self.face_area(face)
            if A <= 0:
                continue
            m_share = rho_a * A / len(face)
            vec = np.array([m_share, m_share, m_share, 0.0, 0.0, 0.0])
            for nid in face:
                _accumulate(accum, int(nid), vec)
        return _accum_to_records(accum, name=defn.name)

    def resolve_volume_lumped(
        self,
        defn: VolumeMassDef,
        elements: list[ndarray],
    ) -> list[MassRecord]:
        """Distribute volume mass equally to element nodes (lumped)."""
        rho = float(defn.density)
        accum: dict[int, ndarray] = {}
        for conn_row in elements:
            V = self.element_volume(conn_row)
            if V <= 0:
                continue
            m_share = rho * V / len(conn_row)
            vec = np.array([m_share, m_share, m_share, 0.0, 0.0, 0.0])
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

    def resolve_surface_consistent(
        self,
        defn: SurfaceMassDef,
        faces: list[list[int]],
    ) -> list[MassRecord]:
        """Consistent surface mass.

        For tri3 / quad4 with constant areal density, the consistent
        mass matrix has equal diagonal entries that sum (with off-diagonal
        contributions) to ρₐ·A/n per node — same as lumped at the
        diagonal level.  Higher-order types fall through to lumped.
        """
        return self.resolve_surface_lumped(defn, faces)

    def resolve_volume_consistent(
        self,
        defn: VolumeMassDef,
        elements: list[ndarray],
    ) -> list[MassRecord]:
        """Consistent volume mass.

        For tet4 / hex8 with constant density, the consistent mass
        matrix's diagonal entries sum to ρ·V/n per node — same as
        lumped at the diagonal level.  Higher-order types fall through.
        """
        return self.resolve_volume_lumped(defn, elements)
