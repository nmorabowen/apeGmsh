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
from typing import ClassVar

import numpy as np
from numpy import ndarray

from .._coupling_control import CouplingControl  # noqa: F401  (re-exported)
from ._kinds import ConstraintKind


@dataclass
class ConstraintRecord:
    """
    Base for all resolved constraint records.

    Every record expresses (or can be expanded to) the general
    linear MPC equation:  u_slave = C · u_master.

    ADR 0038 §"Tag-reference rewrite checklist" — every concrete
    subclass below declares a ``tag_rewrite_spec`` class attribute
    (``ClassVar``) naming the tag-bearing + name-bearing fields the
    Phase 3B.2a compose rewriter must offset / namespace-prefix.  The
    base class has no spec on its own — it is never instantiated bare.
    """
    kind: str
    name: str | None = None

    # The base class has no tag-bearing fields.  Concrete subclasses
    # override this ClassVar; the compose rewriter iterates the
    # registry and applies each spec uniformly.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": (),
        "tag_fields_array": (),
        "name_fields": (),
    }


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
    #: Retained-node DOFs for ``equal_dof_mixed`` ONLY — paired by index
    #: with :attr:`dofs` (the constrained-node DOFs).  ``None`` for every
    #: other kind, where retained and constrained DOFs are identical and
    #: :attr:`dofs` alone suffices.  ``len(master_dofs) == len(dofs)``.
    master_dofs: list[int] | None = None

    # ADR 0038 §"Tag-reference rewrite checklist" — master_node and
    # slave_node are tag-references; ``name`` is the optional caller
    # label that gets namespace-prefixed.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("master_node", "slave_node"),
        "tag_fields_array": (),
        "name_fields": ("name",),
    }

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
    #: Explicit fork-coupling knobs (kinematic_coupling / RBE2 only;
    #: ``None`` for rigid_diaphragm / rigid_body, which ignore it).
    control: "CouplingControl | None" = None
    #: ``rigid_body`` only — emit the fork ``element LadrunoRigidBody``
    #: (over ``{master_node, *slave_nodes}``) instead of the rigidLink
    #: chain. ``False`` for every other kind.
    as_element: bool = False
    #: Total body mass for the ``as_element`` LadrunoRigidBody (``-mass``);
    #: ``None`` ⇒ condense from the slaves' nodal mass.
    mass: float | None = None
    #: Initial body-frame angular velocity for the ``as_element``
    #: LadrunoRigidBody (``-omega``, explicit-dynamics IC); ``None`` ⇒ none.
    omega: tuple[float, float, float] | None = None

    # ADR 0038 §"Tag-reference rewrite checklist" — master_node (scalar)
    # and slave_nodes (array) per the cover set.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("master_node",),
        "tag_fields_array": ("slave_nodes",),
        "name_fields": ("name",),
    }

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
    excess : float or None
        Barycentric excess of the slave node relative to the host
        element — ``0.0`` when the slave is strictly inside, positive
        when outside (extrapolation; the magnitude is how far outside
        in barycentric coordinates).  Populated by ``resolve_embedded``;
        ``None`` for records produced by other code paths.  Enables
        downstream tolerance gating and post-resolution introspection.
    """
    slave_node: int = 0
    master_nodes: list[int] = field(default_factory=list)
    weights: ndarray | None = None
    dofs: list[int] = field(default_factory=list)
    projected_point: ndarray | None = None
    parametric_coords: ndarray | None = None
    excess: float | None = None
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False
    #: Enforcement route (ADR 0068 §1) for ``tie`` / ``tied_contact``:
    #: "penalty" (ASDEmbeddedNodeElement, default) | "penalty_al"
    #: (LadrunoEmbeddedNode) | "equation" (per-DOF EQ_Constraint expansion,
    #: ``u_d(slave)=Σ wᵢ·u_d(mᵢ)``).  Plain passthrough — compose copies it
    #: verbatim (it is neither a tag nor a name).
    enforce: str = "penalty"
    #: Explicit fork-coupling knobs (distributing / RBE3 only; ``None`` for
    #: tie / embedded, which use the stiffness/rotational/pressure fields).
    control: "CouplingControl | None" = None

    # ADR 0038 §"Tag-reference rewrite checklist" — slave_node (scalar)
    # and master_nodes (array) per the cover set; the InterpolationRecord
    # is the host-element-tag carrier for embedded/tied constraints, but
    # the resolved record dataclass stores the full master_nodes list, so
    # there's no separate host_element_tag field to rewrite here.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("slave_node",),
        "tag_fields_array": ("master_nodes",),
        "name_fields": ("name",),
    }

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
class ReinforceTieRecord(ConstraintRecord):
    """One resolved ``LadrunoEmbeddedRebar`` tie (Ladruno fork).

    Carries the inverse-map result for a single rebar node plus the
    pass-through tie parameters, so the bridge build step can emit
    ``element LadrunoEmbeddedRebar`` (via the R0 ``embedded_rebar_args``
    builder, resolving ``bond`` by name → tag). Solver-agnostic — no
    OpenSees imports here.

    Attributes
    ----------
    rebar_node
        The rebar (slave) mesh node tag.
    host_nodes
        The host element's node tags the weights couple to (8 for a hex8
        host, 4 for tet4 — the ``-shape`` host node list).
    weights
        Shape-function weights ``Nᵢ(ξ)`` at the rebar point (sum to 1),
        parallel to ``host_nodes``.
    direction
        Unit bar axis ``d̂`` at this node (from the rebar segment).
    bond_scale
        ``π·d_b·L_trib`` (``None`` for the perfect-bond law).
    bond
        ``LadrunoBondSlip`` material **name** for the axial law, or
        ``None`` when ``perfect`` is set.
    perfect
        Perfect-bond axial penalty ``kAxial`` (or ``None`` for bond).
    kt, kt_alpha, enforce
        Transverse-penalty + enforcement pass-throughs.
    excess, in_bounds
        Inverse-map diagnostics (excess > tol with ``snap`` ⇒ extrapolated).
    """

    rebar_node: int = 0
    host_nodes: list[int] = field(default_factory=list)
    weights: ndarray | None = None
    direction: ndarray | None = None
    bond_scale: float | None = None
    bond: str | None = None
    perfect: float | None = None
    kt: float | None = None
    kt_alpha: float | None = None
    enforce: str = "penalty"
    bipenalty: bool = False
    dtcr: float | None = None
    excess: float | None = None
    in_bounds: bool = True

    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("rebar_node",),
        "tag_fields_array": ("host_nodes",),
        "name_fields": ("name", "bond"),
    }


@dataclass
class EmbedTieRecord(ConstraintRecord):
    """One resolved ``LadrunoEmbeddedNode`` tie (Ladruno fork).

    The isotropic sibling of :class:`ReinforceTieRecord`: it ties a single
    constrained node into the host element it falls inside (via the same
    guarded inverse map), with no bar axis, bond law, or tributary length.
    The bridge build step emits ``element LadrunoEmbeddedNode`` via the
    ``embedded_node_args`` builder. Solver-agnostic — no OpenSees imports.

    Attributes
    ----------
    node
        The constrained (slave) mesh node tag.
    host_nodes
        The host element's node tags the weights couple to (8 for hex8,
        4 for tet4 — the ``-shape`` host node list).
    weights
        Shape-function weights ``Nᵢ(ξ)`` at the node (sum to 1), parallel
        to ``host_nodes``.
    k, k_alpha, enforce
        Isotropic penalty + enforcement pass-throughs (``-k`` / ``-kAlpha``).
    bipenalty, dtcr
        Explicit bipenalty critical-time-step control.
    staged
        ``True`` (default) → g0 stress-free birth (no ``-absolute``);
        ``False`` → emit ``-absolute`` (legacy absolute tie).
    excess, in_bounds
        Inverse-map diagnostics (excess > tol with ``snap`` ⇒ extrapolated).
    """

    node: int = 0
    host_nodes: list[int] = field(default_factory=list)
    weights: ndarray | None = None
    k: float | None = None
    k_alpha: float | None = None
    enforce: str = "penalty"
    bipenalty: bool = False
    dtcr: float | None = None
    staged: bool = True
    excess: float | None = None
    in_bounds: bool = True

    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("node",),
        "tag_fields_array": ("host_nodes",),
        "name_fields": ("name",),
    }


@dataclass
class ContactRecord(ConstraintRecord):
    """One resolved fork contact interaction (`contactSurface` + `contact`).

    A face-to-face contact between two meshed surfaces, emitted as the fork's
    `contactSurface tag (-master|-slave|-slave-segments) …` pair plus the
    `contact tag master slave …` verb (and the `LadrunoContact` handler). The
    master is always a faceted surface; the slave is a node set (NTS) or a
    faceted surface (mortar). Solver-agnostic — no OpenSees imports here.

    **Serial-only.** The fork contact subsystem is not parallel (the handler's
    sendSelf/recvSelf are stubs), so contact records carry no partition
    tag-rewrite — a contact model emits/runs in a single domain.

    Attributes
    ----------
    formulation
        ``"nts"`` (node-to-segment) or ``"mortar"`` (segment-to-segment ALM).
    master_faces, master_nps
        The master surface's flat face connectivity ``(n_faces, nps)`` and the
        per-facet node count ``nps`` (3=tri, 4=quad).
    slave_nodes
        NTS slave node tags (``None`` for mortar).
    slave_faces, slave_nps
        Mortar slave faceted connectivity + stride (``None``/0 for NTS).
    outward
        Unit outward normal ``(ox, oy, oz)`` toward the slave half-space, or
        ``None`` (let the fork auto-derive).
    kn, kt, mu
        NTS penalty (normal/tangential) + friction.
    eps_n, eps_t, cohesion, tau_max, aug_tol, max_aug, ngp, tie
        Mortar ALM penalty / friction-cone / augmentation controls + mesh-tie.
    soft, visc, consistent_tan, geom_tan
        Extension modifiers (ADR 0073): ``soft`` = explicit Courant-stable SOFT
        penalty (``True`` ⇒ fork default SOFSCL 0.10, or a float SOFSCL);
        ``visc`` = viscous normal-stabilisation coefficient μ_c; ``consistent_tan``
        = non-symmetric consistent friction tangent (needs an unsymmetric
        solver); ``geom_tan`` = NTS ∂n/∂u geometric normal tangent (NTS-only).
    """

    formulation: str = "nts"
    master_faces: ndarray | None = None
    master_nps: int = 0
    slave_nodes: list[int] | None = None
    slave_faces: ndarray | None = None
    slave_nps: int = 0
    outward: tuple | None = None
    kn: float | str | None = None
    kt: float | None = None
    mu: float | None = None
    eps_n: float | str | None = None
    eps_t: float | str | None = None
    cohesion: float | None = None
    tau_max: float | None = None
    aug_tol: float | None = None
    max_aug: int | None = None
    ngp: int | None = None
    tie: bool = False
    soft: float | bool | None = None
    visc: float | None = None
    consistent_tan: bool = False
    geom_tan: bool = False

    # Serial-only subsystem — no partition tag rewrite (see class docstring).
    tag_rewrite_spec: ClassVar[dict] = {"name_fields": ("name",)}


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

    # ADR 0038 §"Tag-reference rewrite checklist" — master_nodes /
    # slave_nodes arrays.  ``slave_records`` is a list of
    # :class:`InterpolationRecord` and is rewritten recursively by the
    # compose engine (each child has its own spec).
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": (),
        "tag_fields_array": ("master_nodes", "slave_nodes"),
        "name_fields": ("name",),
        "nested_records": ("slave_records",),
    }


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

    # ADR 0038 §"Tag-reference rewrite checklist" — master_node (scalar)
    # plus slave_nodes / phantom_nodes (arrays).  The nested
    # ``rigid_link_records`` / ``equal_dof_records`` are
    # :class:`NodePairRecord` lists; the compose engine rewrites them
    # recursively via their own spec.
    tag_rewrite_spec: ClassVar[dict] = {
        "tag_fields_scalar": ("master_node",),
        "tag_fields_array": ("slave_nodes", "phantom_nodes"),
        "name_fields": ("name",),
        "nested_records": ("rigid_link_records", "equal_dof_records"),
    }

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
    "ReinforceTieRecord",
    "SurfaceCouplingRecord",
    "NodeToSurfaceRecord",
]
