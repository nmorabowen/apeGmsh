"""
Constraints — Solver-agnostic multi-point constraint engine.
=============================================================

Two-stage pipeline:

    Stage 1  ─  **Definition** (pre-mesh, geometry level)
    ─────────────────────────────────────────────────────────
    The user declares *intent*:  "these two instances are tied
    at this interface" or "this node is rigidly linked to that
    surface."  Definitions are lightweight dataclasses that
    carry the geometry-level information.

    Stage 2  ─  **Resolution** (post-mesh)
    ─────────────────────────────────────────────────────────
    After meshing, :class:`ConstraintResolver` converts each
    definition into concrete :class:`ConstraintRecord` objects
    containing actual node tags, DOFs, weights, and offset
    vectors.  These records are **solver-agnostic** — any
    solver adapter (OpenSees, Abaqus, Code_Aster, …) can
    consume them.

Constraint taxonomy
~~~~~~~~~~~~~~~~~~~

**Level 1 — Node-to-Node** (1 master, 1 slave)

=================  ================================================
``equal_dof``      u_slave = u_master  (selected DOFs)
``rigid_beam``     slave follows master as rigid bar (6-DOF coupling
                   with rotational offset)
``rigid_rod``      only translations coupled through rigid bar
``penalty``        soft spring K_p between two nodes
=================  ================================================

**Level 2 — Node-to-Group** (1 master, N slaves)

=================  ================================================
``rigid_diaphragm``  in-plane DOFs of all slaves follow master
``rigid_body``       all 6 DOFs follow master
``kinematic``        general: user picks which DOFs
=================  ================================================

**Level 2b — Mixed-DOF coupling** (1 6-DOF master, N 3-DOF slaves)

=================  ================================================
``node_to_surface`` 6-DOF master -> phantom nodes -> 3-DOF solid slaves
                   via rigid link + equalDOF (translations only)
=================  ================================================

**Level 3 — Node-to-Surface** (1 node, 1 element face)

=================  ================================================
``tie``            u_slave = Σ N_i · u_master_i  (shape function
                   interpolation on closest master face)
``distributing``   force at master distributed to slave surface
``embedded``       embedded element nodes follow host field
=================  ================================================

**Level 4 — Surface-to-Surface**

=================  ================================================
``tied_contact``   all nodes on surface A tied to surface B
``mortar``         Lagrange multiplier space on interface
=================  ================================================

All constraints ultimately express the linear MPC equation:

    u_slave = C · u_master

where C is the constraint transformation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy import ndarray


# =====================================================================
# Stage 1 — Constraint Definitions (pre-mesh, geometry-level intent)
# =====================================================================

@dataclass
class ConstraintDef:
    """Base class for all constraint definitions."""
    kind: str
    master_label: str          # Assembly instance label
    slave_label: str           # Assembly instance label
    name: str | None = None    # optional user name


# ── Level 1: Node-to-Node ────────────────────────────────────────────

@dataclass
class EqualDOFDef(ConstraintDef):
    """
    Co-located nodes share selected DOFs.

    After meshing, the resolver finds node pairs within *tolerance*
    on the interface between master and slave instances, and produces
    one :class:`NodePairRecord` per pair.

    Parameters
    ----------
    dofs : list[int] or None
        DOF numbers to constrain (1-based: 1=ux, 2=uy, 3=uz,
        4=rx, 5=ry, 6=rz).  ``None`` = all DOFs.
    tolerance : float
        Spatial distance (in model units) within which two nodes
        are considered co-located.
    master_entities : list of (dim, tag), optional
        Limit the master search to specific geometric entities.
    slave_entities : list of (dim, tag), optional
        Limit the slave search to specific geometric entities.
    """
    kind: str = field(init=False, default="equal_dof")
    dofs: list[int] | None = None
    tolerance: float = 1e-6
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None


@dataclass
class RigidLinkDef(ConstraintDef):
    """
    Rigid bar connecting master and slave nodes.

    ``rigid_beam``  ->  full 6-DOF coupling (translations + rotations)::

        u_s = u_m + θ_m × r       (translations)
        θ_s = θ_m                  (rotations)

    ``rigid_rod``  ->  translations only, rotations independent::

        u_s = u_m + θ_m × r
        (θ_s free)

    Parameters
    ----------
    link_type : ``"beam"`` or ``"rod"``
    master_point : (x,y,z) or None
        Explicit master node location.  If None, found by proximity.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves.
    tolerance : float
        For auto-detecting master node by proximity.
    """
    kind: str = field(init=False, default="rigid_link")
    link_type: str = "beam"
    master_point: tuple[float, float, float] | None = None
    slave_entities: list[tuple[int, int]] | None = None
    tolerance: float = 1e-6


@dataclass
class PenaltyDef(ConstraintDef):
    """
    Soft spring between co-located node pairs.

    Numerically approximates EqualDOF when K -> ∞.  Useful when
    hard constraints cause ill-conditioning.

    Parameters
    ----------
    stiffness : float
        Penalty spring stiffness (force/length units).
    dofs : list[int] or None
        DOFs to penalise.
    tolerance : float
        Node-matching tolerance.
    """
    kind: str = field(init=False, default="penalty")
    stiffness: float = 1e10
    dofs: list[int] | None = None
    tolerance: float = 1e-6
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None


# ── Level 2: Node-to-Group ───────────────────────────────────────────

@dataclass
class RigidDiaphragmDef(ConstraintDef):
    """
    In-plane rigid body constraint.  All slave nodes at a given
    plane follow the master node for in-plane DOFs.

    Classic use: floor slabs in multi-story buildings — all nodes
    at a floor elevation share in-plane translation + rotation
    about the out-of-plane axis.

    Parameters
    ----------
    master_point : (x, y, z)
        Master node location (typically center of mass).
    plane_normal : (nx, ny, nz)
        Normal to the diaphragm plane.  (0,0,1) = horizontal floor.
    constrained_dofs : list[int]
        DOFs constrained in-plane.  For a horizontal floor with
        Z as vertical: [1, 2, 6]  (ux, uy, rz).
    plane_tolerance : float
        Distance from the plane within which nodes are collected.
    """
    kind: str = field(init=False, default="rigid_diaphragm")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    plane_normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    constrained_dofs: list[int] = field(default_factory=lambda: [1, 2, 6])
    plane_tolerance: float = 1.0


@dataclass
class RigidBodyDef(ConstraintDef):
    """
    Full rigid body constraint: all 6 DOFs of every slave node
    follow the master.

    Parameters
    ----------
    master_point : (x, y, z)
        Master node location.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves.
    """
    kind: str = field(init=False, default="rigid_body")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None


@dataclass
class KinematicCouplingDef(ConstraintDef):
    """
    Generalised master-slave: user picks which DOFs.

    This is the parent of rigid_diaphragm and rigid_body —
    they are special cases with pre-set DOF lists.

    Parameters
    ----------
    master_point : (x, y, z)
        Master node location.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves.
    dofs : list[int]
        DOFs to couple.
    """
    kind: str = field(init=False, default="kinematic_coupling")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


# ── Level 3: Node-to-Surface ─────────────────────────────────────────

@dataclass
class TieDef(ConstraintDef):
    """
    Surface tie via shape function interpolation.

    Each slave node is projected onto the closest master element
    face.  Its DOFs are constrained to the master face via::

        u_slave = Σ  N_i(ξ,η) · u_master_i

    where N_i are the shape functions of the master face element
    evaluated at the projected parametric coordinates.

    This is what Abaqus ``*TIE`` does.  It preserves displacement
    continuity even with non-matching meshes.

    Parameters
    ----------
    master_entities : list of (dim, tag)
        Master surface entities.
    slave_entities : list of (dim, tag)
        Slave surface entities (nodes on these are projected).
    dofs : list[int] or None
        DOFs to tie.  None = all translational DOFs [1,2,3].
    tolerance : float
        Maximum projection distance.  Slave nodes farther than
        this from the master surface are skipped.
    """
    kind: str = field(init=False, default="tie")
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] | None = None
    tolerance: float = 1.0


@dataclass
class DistributingCouplingDef(ConstraintDef):
    """
    Distributing coupling: load at master is distributed to slave
    surface as consistent nodal forces.

    The slave surface deforms freely (not rigidly).  Weights are
    computed so that force and moment equilibrium are preserved.

    Parameters
    ----------
    master_point : (x, y, z)
        Reference point where the load is applied.
    slave_entities : list of (dim, tag)
        Surface entities that receive the distributed load.
    dofs : list[int] or None
        DOFs to couple.
    weighting : ``"uniform"`` or ``"area"``
        How to distribute: uniform gives equal weights; area
        weights by tributary area (more physical).
    """
    kind: str = field(init=False, default="distributing")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] | None = None
    weighting: str = "uniform"


@dataclass
class EmbeddedDef(ConstraintDef):
    """
    Embedded element constraint: nodes of a lower-dimensional
    element (beam, truss) are constrained to the displacement
    field of a higher-dimensional host element (solid).

    Used for reinforcement in concrete, stiffeners in shells, etc.

    Parameters
    ----------
    host_entities : list of (dim, tag)
        Host volume/surface entities.
    embedded_entities : list of (dim, tag)
        Embedded line/surface entities.
    tolerance : float
        Search tolerance for finding the host element.
    """
    kind: str = field(init=False, default="embedded")
    host_entities: list[tuple[int, int]] | None = None
    embedded_entities: list[tuple[int, int]] | None = None
    tolerance: float = 1.0


# ── Level 2b: Mixed-DOF coupling ────────────────────────────────────

@dataclass
class NodeToSurfaceDef(ConstraintDef):
    """
    6-DOF node to 3-DOF surface coupling via phantom (duplicate) nodes.

    Connects a 6-DOF master node (beam, frame, or any reference
    point) to a group of 3-DOF slave nodes on a surface (solid
    elements) through an intermediate layer of phantom nodes that
    carry full 6-DOF kinematics.

    The resolver:

    1. **Duplicates** each slave node -> creates phantom node tags
       at the same coordinates (6-DOF intermediaries).
    2. **Rigid links** master -> each phantom node (``rigid_beam``),
       propagating rotational effects through the offset arm::

           u_phantom = u_master + θ_master × r

    3. **EqualDOF** phantom -> original slave, translations only
       ``[1, 2, 3]`` (rotations discarded since the solid has none).

    This is the standard technique for mixed-dimensionality
    coupling (Abaqus ``*COUPLING, KINEMATIC`` on solids; OpenSees
    manual rigid-link + equalDOF pattern).

    Unlike other constraint definitions that take string labels,
    this one accepts **bare tags**:

    - ``master_label``: node tag (int, dim=0) — the 6-DOF node.
    - ``slave_label``: surface entity tag (int, dim=2) — the
      Gmsh surface whose nodes become the 3-DOF slaves.

    Parameters
    ----------
    master_point : (x, y, z) or None
        Explicit master node location.  If None, the node tag in
        ``master_label`` is used directly.
    dofs : list[int] or None
        Translational DOFs coupled to the solid.  Default [1, 2, 3].
    tolerance : float
        For auto-detecting master node by proximity.
    """
    kind: str = field(init=False, default="node_to_surface")
    master_point: tuple[float, float, float] | None = None
    dofs: list[int] | None = None
    tolerance: float = 1e-6


# ── Level 4: Surface-to-Surface ──────────────────────────────────────

@dataclass
class TiedContactDef(ConstraintDef):
    """
    Full surface-to-surface tie.  Every node on the slave surface
    is tied to the master surface via shape function interpolation.
    Bidirectional — also checks master nodes against slave faces.

    Parameters
    ----------
    master_entities : list of (dim, tag)
    slave_entities : list of (dim, tag)
    dofs : list[int] or None
    tolerance : float
    """
    kind: str = field(init=False, default="tied_contact")
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] | None = None
    tolerance: float = 1.0


@dataclass
class MortarDef(ConstraintDef):
    """
    Mortar coupling: Lagrange multiplier space on the interface.

    Mathematically rigorous surface-to-surface coupling that
    satisfies the inf-sup condition.  More accurate than node-to-
    surface tie for non-matching meshes.

    The coupling operator B is computed by numerical integration
    over the overlapping surface segments::

        B_ij = ∫_Γ  ψ_i · N_j  dΓ

    where ψ_i are the Lagrange multiplier basis functions (defined
    on the slave side) and N_j are the master shape functions.

    Parameters
    ----------
    master_entities : list of (dim, tag)
    slave_entities : list of (dim, tag)
    dofs : list[int] or None
    integration_order : int
        Gauss quadrature order for the coupling integral.
    """
    kind: str = field(init=False, default="mortar")
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] | None = None
    integration_order: int = 2


# =====================================================================
# Stage 2 — Constraint Records (post-mesh, resolved)
# =====================================================================

@dataclass
class ConstraintRecord:
    """
    Base for all resolved constraint records.

    Every record expresses (or can be expanded to) the general
    linear MPC equation:  u_slave = C · u_master.
    """
    kind: str
    name: str | None = None


@dataclass
class NodePairRecord(ConstraintRecord):
    """
    One master node ↔ one slave node.

    Covers: ``equal_dof``, ``rigid_beam``, ``rigid_rod``, ``penalty``.

    Attributes
    ----------
    master_node : int
        Master node tag (from mesh).
    slave_node : int
        Slave node tag (from mesh).
    dofs : list[int]
        Constrained DOFs (1-based).
    offset : ndarray or None
        Rigid arm vector r = x_slave − x_master.  Present for
        rigid link types; ``None`` for equal_dof.
    penalty_stiffness : float or None
        For penalty type only.
    """
    master_node: int = 0
    slave_node: int = 0
    dofs: list[int] = field(default_factory=list)
    offset: ndarray | None = None
    penalty_stiffness: float | None = None

    def constraint_matrix(self, ndof: int = 6) -> ndarray:
        """
        Build the constraint transformation matrix C such that
        u_slave[dofs] = C · u_master[all_dofs].

        For equal_dof: C is a selection matrix (rows of identity).
        For rigid_beam: C includes the skew-symmetric offset matrix.

        Parameters
        ----------
        ndof : int
            DOFs per node (default 6 for shell/beam).

        Returns
        -------
        ndarray of shape (len(dofs), ndof)
        """
        n = len(self.dofs)
        C = np.zeros((n, ndof))

        if self.kind == "equal_dof" or self.kind == "penalty":
            # u_slave_i = u_master_i
            for row, dof in enumerate(self.dofs):
                C[row, dof - 1] = 1.0

        elif self.kind in ("rigid_beam", "rigid_rod"):
            # u_s = u_m + θ_m × r
            #
            # In matrix form for translations (DOFs 1-3):
            #   [u_s]   [I  | -[r×]] [u_m ]
            #   [   ] = [   |      ] [    ]
            #   [θ_s]   [0  |   I  ] [θ_m ]  (beam only)
            #
            # Skew-symmetric matrix of r:
            #   [r×] = [ 0   -rz   ry]
            #          [ rz   0   -rx]
            #          [-ry   rx   0 ]
            r = self.offset if self.offset is not None else np.zeros(3)
            rx, ry, rz = r

            skew = np.array([
                [ 0,  -rz,  ry],
                [ rz,  0,  -rx],
                [-ry,  rx,   0],
            ])

            for row, dof in enumerate(self.dofs):
                idx = dof - 1
                if idx < 3:
                    # Translation: u_s_i = u_m_i + (skew · θ_m)_i
                    C[row, idx] = 1.0                   # I term
                    C[row, 3:6] = -skew[idx, :]         # -[r×] · θ_m
                elif idx < 6 and self.kind == "rigid_beam":
                    # Rotation (beam only): θ_s = θ_m
                    C[row, idx] = 1.0

        return C


@dataclass
class NodeGroupRecord(ConstraintRecord):
    """
    One master node ↔ multiple slave nodes.

    Covers: ``rigid_diaphragm``, ``rigid_body``,
    ``kinematic_coupling``.

    Attributes
    ----------
    master_node : int
    slave_nodes : list[int]
    dofs : list[int]
        DOFs constrained for all slaves.
    offsets : ndarray
        Array of shape (n_slaves, 3) — offset vector for each slave.
    plane_normal : ndarray or None
        For rigid_diaphragm: normal to the constraint plane.
    """
    master_node: int = 0
    slave_nodes: list[int] = field(default_factory=list)
    dofs: list[int] = field(default_factory=list)
    offsets: ndarray | None = None
    plane_normal: ndarray | None = None

    def expand_to_pairs(self) -> list[NodePairRecord]:
        """
        Expand this group constraint into individual
        :class:`NodePairRecord` objects — one per slave node.

        This is the most common consumption path: most solvers
        implement group constraints as loops of pair constraints
        (e.g., OpenSees ``rigidDiaphragm`` or repeated ``equalDOF``).
        """
        pairs = []
        for i, sn in enumerate(self.slave_nodes):
            offset = self.offsets[i] if self.offsets is not None else None
            if self.kind == "rigid_diaphragm":
                pair_kind = "rigid_beam"
            elif self.kind == "rigid_body":
                pair_kind = "rigid_beam"
            else:
                pair_kind = "kinematic_coupling"

            pairs.append(NodePairRecord(
                kind=pair_kind,
                name=self.name,
                master_node=self.master_node,
                slave_node=sn,
                dofs=list(self.dofs),
                offset=offset,
            ))
        return pairs


@dataclass
class InterpolationRecord(ConstraintRecord):
    """
    One slave node interpolated from a master element face.

    Covers: ``tie``, ``distributing``, ``embedded``.

    The constraint equation is::

        u_slave = Σ  w_i · u_master_i

    where w_i are the interpolation weights (shape function values
    at the projected parametric coordinates on the master face).

    Attributes
    ----------
    slave_node : int
    master_nodes : list[int]
        Nodes of the master element face (ordered).
    weights : ndarray
        Shape function values N_i(ξ,η) — same length as
        ``master_nodes``.  Sum to 1.0 for partition of unity.
    dofs : list[int]
    projected_point : ndarray or None
        Physical coordinates of the projection onto the master face
        (useful for verification / visualisation).
    parametric_coords : ndarray or None
        (ξ, η) on the master face.
    """
    slave_node: int = 0
    master_nodes: list[int] = field(default_factory=list)
    weights: ndarray | None = None
    dofs: list[int] = field(default_factory=list)
    projected_point: ndarray | None = None
    parametric_coords: ndarray | None = None

    def constraint_matrix(self, ndof: int = 3) -> ndarray:
        """
        Build the constraint matrix C of shape
        (ndof, n_master_nodes * ndof).

        u_slave[i] = Σ_j  w_j · u_master_j[i]   for each DOF i
        """
        n_master = len(self.master_nodes)
        n_dof = len(self.dofs)
        C = np.zeros((n_dof, n_master * n_dof))
        w = self.weights if self.weights is not None else np.ones(n_master) / n_master
        for row, dof in enumerate(self.dofs):
            for j in range(n_master):
                C[row, j * n_dof + row] = w[j]
        return C


@dataclass
class SurfaceCouplingRecord(ConstraintRecord):
    """
    Surface-to-surface coupling operator.

    Covers: ``tied_contact``, ``mortar``.

    The coupling is stored as a sparse set of interpolation
    records (one per slave node for tied_contact), or as the
    full mortar operator matrix B.

    Attributes
    ----------
    slave_records : list[InterpolationRecord]
        Per-slave-node interpolation data (for tied_contact).
    mortar_operator : ndarray or None
        Dense coupling matrix B (for mortar method).
        Shape: (n_slave_dofs, n_master_dofs).
    master_nodes : list[int]
        All master nodes involved.
    slave_nodes : list[int]
        All slave nodes involved.
    dofs : list[int]
    """
    slave_records: list[InterpolationRecord] = field(default_factory=list)
    mortar_operator: ndarray | None = None
    master_nodes: list[int] = field(default_factory=list)
    slave_nodes: list[int] = field(default_factory=list)
    dofs: list[int] = field(default_factory=list)


@dataclass
class NodeToSurfaceRecord(ConstraintRecord):
    """
    Compound record for 6-DOF node to 3-DOF surface coupling via phantom nodes.

    This record encapsulates the three-step coupling:

    1. Phantom nodes duplicated from the original slave positions.
    2. Rigid links from the 6-DOF master to each phantom node.
    3. EqualDOF from each phantom node to the original slave (translations only).

    Solvers consume this by:
    - Creating the phantom nodes (6-DOF, same coords as slaves).
    - Emitting ``rigid_beam`` constraints master -> phantom.
    - Emitting ``equal_dof`` constraints phantom -> slave for DOFs [1,2,3].

    Attributes
    ----------
    master_node : int
        The 6-DOF master node tag.
    slave_nodes : list[int]
        Original 3-DOF slave node tags (from the surface mesh).
    phantom_nodes : list[int]
        Generated 6-DOF phantom node tags (one per slave, same
        coordinates).  Tag generation is handled by the resolver
        using an offset above the maximum existing node tag.
    phantom_coords : ndarray
        Coordinates of phantom nodes, shape (n_slaves, 3).
        Identical to the slave coordinates.
    rigid_link_records : list[NodePairRecord]
        Master -> phantom rigid beam records (with offset vectors).
    equal_dof_records : list[NodePairRecord]
        Phantom -> slave equalDOF records (translations only).
    dofs : list[int]
        Translational DOFs coupled to the surface (default [1,2,3]).
    """
    master_node: int = 0
    slave_nodes: list[int] = field(default_factory=list)
    phantom_nodes: list[int] = field(default_factory=list)
    phantom_coords: ndarray | None = None
    rigid_link_records: list[NodePairRecord] = field(default_factory=list)
    equal_dof_records: list[NodePairRecord] = field(default_factory=list)
    dofs: list[int] = field(default_factory=lambda: [1, 2, 3])

    def expand(self) -> list[NodePairRecord]:
        """
        Flatten into individual :class:`NodePairRecord` objects.

        Returns the rigid link records followed by the equalDOF
        records — the natural emission order for solvers.
        """
        return list(self.rigid_link_records) + list(self.equal_dof_records)


# =====================================================================
# Shape functions for master face interpolation (Level 3 & 4)
# =====================================================================

def _shape_tri3(xi: float, eta: float) -> ndarray:
    """Shape functions for 3-node triangle in area coords."""
    return np.array([1.0 - xi - eta, xi, eta])


def _shape_quad4(xi: float, eta: float) -> ndarray:
    """Shape functions for 4-node quad at (ξ, η) ∈ [-1,1]²."""
    return 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])


def _shape_tri6(xi: float, eta: float) -> ndarray:
    """Shape functions for 6-node triangle."""
    zeta = 1.0 - xi - eta
    return np.array([
        zeta * (2 * zeta - 1),
        xi * (2 * xi - 1),
        eta * (2 * eta - 1),
        4 * zeta * xi,
        4 * xi * eta,
        4 * eta * zeta,
    ])


def _shape_quad8(xi: float, eta: float) -> ndarray:
    """Shape functions for 8-node serendipity quad."""
    N = np.zeros(8)
    # Corner nodes
    for i, (xi_i, eta_i) in enumerate([
        (-1, -1), (1, -1), (1, 1), (-1, 1)
    ]):
        N[i] = 0.25 * (1 + xi_i * xi) * (1 + eta_i * eta) * \
               (xi_i * xi + eta_i * eta - 1)
    # Mid-side nodes
    N[4] = 0.5 * (1 - xi**2) * (1 - eta)
    N[5] = 0.5 * (1 + xi) * (1 - eta**2)
    N[6] = 0.5 * (1 - xi**2) * (1 + eta)
    N[7] = 0.5 * (1 - xi) * (1 - eta**2)
    return N


# Map: number of face nodes -> shape function evaluator
SHAPE_FUNCTIONS = {
    3: _shape_tri3,
    4: _shape_quad4,
    6: _shape_tri6,
    8: _shape_quad8,
}


class _SpatialIndex:
    """Small nearest-neighbour wrapper with a SciPy-free fallback."""

    def __init__(self, coords: ndarray) -> None:
        self._coords = np.asarray(coords, dtype=float)

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            self._tree = None
        else:
            self._tree = cKDTree(self._coords)

    def query_ball_point(self, point: ndarray, radius: float) -> list[int]:
        point = np.asarray(point, dtype=float)
        if self._tree is not None:
            return list(self._tree.query_ball_point(point, radius))

        dists = np.linalg.norm(self._coords - point, axis=1)
        return np.flatnonzero(dists <= radius).astype(int).tolist()

    def query(self, point: ndarray, k: int = 1):
        point = np.asarray(point, dtype=float)
        if self._tree is not None:
            return self._tree.query(point, k=k)

        dists = np.linalg.norm(self._coords - point, axis=1)
        order = np.argsort(dists)
        if k == 1:
            idx = int(order[0])
            return float(dists[idx]), idx

        order = order[:k]
        return dists[order], order


# =====================================================================
# Geometric utilities
# =====================================================================

def _project_point_to_face(
    point: ndarray,
    face_coords: ndarray,
) -> tuple[ndarray, ndarray, float]:
    """
    Project a point onto an element face.

    Uses Newton iteration to find the parametric coordinates (ξ, η)
    that minimise the distance from the point to the face surface.

    Parameters
    ----------
    point : ndarray, shape (3,)
        The point to project.
    face_coords : ndarray, shape (n_nodes, 3)
        Physical coordinates of the face nodes.

    Returns
    -------
    xi_eta : ndarray, shape (2,)
        Parametric coordinates of the projection.
    projected : ndarray, shape (3,)
        Physical coordinates of the projected point on the face.
    distance : float
        Distance from the original point to the projection.
    """
    n_nodes = face_coords.shape[0]
    shape_fn = SHAPE_FUNCTIONS.get(n_nodes)
    if shape_fn is None:
        raise ValueError(
            f"No shape function for {n_nodes}-node face."
        )

    # Initial guess: face centroid in parametric space
    if n_nodes in (3, 6):
        xi, eta = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi, eta = 0.0, 0.0

    # Newton iteration (typically converges in 3-5 iterations)
    for _ in range(20):
        N = shape_fn(xi, eta)
        x_param = N @ face_coords           # (3,)
        residual = x_param - point           # (3,)

        # Numerical derivatives of shape functions
        eps = 1e-8
        N_xi  = (shape_fn(xi + eps, eta) - shape_fn(xi - eps, eta)) / (2 * eps)
        N_eta = (shape_fn(xi, eta + eps) - shape_fn(xi, eta - eps)) / (2 * eps)

        dx_dxi  = N_xi  @ face_coords       # (3,)
        dx_deta = N_eta @ face_coords        # (3,)

        # 2×2 system:  J^T J  [dξ, dη]^T = -J^T r
        J = np.column_stack([dx_dxi, dx_deta])   # (3, 2)
        JtJ = J.T @ J                             # (2, 2)
        Jtr = J.T @ residual                       # (2,)

        det = JtJ[0, 0] * JtJ[1, 1] - JtJ[0, 1] * JtJ[1, 0]
        if abs(det) < 1e-30:
            break
        inv = np.array([
            [ JtJ[1, 1], -JtJ[0, 1]],
            [-JtJ[1, 0],  JtJ[0, 0]],
        ]) / det

        delta = -inv @ Jtr
        xi  += delta[0]
        eta += delta[1]

        if np.linalg.norm(delta) < 1e-12:
            break

    N_final = shape_fn(xi, eta)
    projected = np.asarray(N_final @ face_coords)
    distance = float(np.linalg.norm(projected - point))

    return np.array([xi, eta]), projected, distance


def _is_inside_parametric(
    xi_eta: ndarray,
    n_nodes: int,
    tol: float = 0.05,
) -> bool:
    """
    Check if parametric coordinates are inside the element face
    (with a small tolerance for numerical rounding).
    """
    xi, eta = xi_eta
    if n_nodes in (3, 6):
        # Triangle: ξ ≥ 0, η ≥ 0, ξ + η ≤ 1
        return (xi >= -tol and eta >= -tol and
                xi + eta <= 1.0 + tol)
    else:
        # Quad: ξ ∈ [-1,1], η ∈ [-1,1]
        return (abs(xi) <= 1.0 + tol and abs(eta) <= 1.0 + tol)


# =====================================================================
# Constraint Resolver
# =====================================================================

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
                kind="equal_dof",
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

        if kind == "rigid_beam":
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
                kind="penalty",
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
                kind="rigid_diaphragm",
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
            kind="rigid_diaphragm",
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
                        kind="tie",
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
            kind="distributing",
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
            kind="tied_contact",
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
            kind="mortar",
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

        # -- 2. Generate phantom node tags --
        max_tag = int(self.node_tags.max())
        phantom_tags = list(range(max_tag + 1, max_tag + 1 + len(slave_list)))

        phantom_coords = np.array([
            self._coords_of(t) for t in slave_list
        ])

        # -- 3. Rigid beam: master -> phantom --
        rigid_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            slave_xyz = self._coords_of(slave_tag)
            offset = slave_xyz - master_xyz
            rigid_records.append(NodePairRecord(
                kind="rigid_beam",
                name=defn.name,
                master_node=master_tag,
                slave_node=phantom_tag,
                dofs=[1, 2, 3, 4, 5, 6],
                offset=offset,
            ))

        # -- 4. EqualDOF: phantom -> slave (translations only) --
        edof_records = []
        for phantom_tag, slave_tag in zip(phantom_tags, slave_list):
            edof_records.append(NodePairRecord(
                kind="equal_dof",
                name=defn.name,
                master_node=phantom_tag,
                slave_node=slave_tag,
                dofs=list(dofs),
            ))

        return NodeToSurfaceRecord(
            kind="node_to_surface",
            name=defn.name,
            master_node=master_tag,
            slave_nodes=slave_list,
            phantom_nodes=phantom_tags,
            phantom_coords=phantom_coords,
            rigid_link_records=rigid_records,
            equal_dof_records=edof_records,
            dofs=list(dofs),
        )
