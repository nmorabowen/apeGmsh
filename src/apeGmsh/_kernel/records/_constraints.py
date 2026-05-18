"""
Stage 2 — Constraint Records (post-mesh, resolved).

These dataclasses carry the concrete mesh-level outputs of constraint
resolution: node tags, shape-function weights, offset vectors, and
phantom-node bookkeeping. Records are solver-agnostic — any adapter
(OpenSees, Abaqus, Code_Aster, …) can consume them.

All records ultimately express the linear MPC equation::

    u_slave = C · u_master
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from ._kinds import ConstraintKind


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

        if self.kind in (ConstraintKind.EQUAL_DOF, ConstraintKind.PENALTY):
            # u_slave_i = u_master_i
            for row, dof in enumerate(self.dofs):
                C[row, dof - 1] = 1.0

        elif self.kind in (ConstraintKind.RIGID_BEAM, ConstraintKind.RIGID_ROD):
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
                elif idx < 6 and self.kind == ConstraintKind.RIGID_BEAM:
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
            if self.kind == ConstraintKind.RIGID_DIAPHRAGM:
                pair_kind = ConstraintKind.RIGID_BEAM
            elif self.kind == ConstraintKind.RIGID_BODY:
                pair_kind = ConstraintKind.RIGID_BEAM
            else:
                pair_kind = ConstraintKind.KINEMATIC_COUPLING

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


__all__ = [
    "ConstraintRecord",
    "NodePairRecord",
    "NodeGroupRecord",
    "InterpolationRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
]
