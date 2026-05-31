"""
Stage 1 — Constraint Definitions (pre-mesh, geometry-level intent).

These dataclasses describe *what* the user wants tied together at the
geometry level. They carry no node tags. After meshing, the
:class:`~apeGmsh.mesh._constraint_resolver.ConstraintResolver`
converts each definition into concrete records
(see :mod:`apeGmsh.mesh.records._constraints`).

See :mod:`apeGmsh.mesh._constraint_resolver` for the resolver layer
and :mod:`apeGmsh.mesh.records` for the full constraint taxonomy
(Level 1-4) and resolved record types.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _validate_asd_embedded_options(
    rotational: bool,
    pressure: bool,
    stiffness_p: float | None,
    kind: str,
) -> None:
    """Fail-loud validation of ASDEmbeddedNodeElement-bound flags.

    Mirrors the C++ parser's mutual-exclusion check at
    ``ASDEmbeddedNodeElement.cpp:276`` and refuses combinations that
    OpenSees would either reject at parse time or silently ignore.
    """
    if rotational and pressure:
        raise ValueError(
            f"{kind}: rotational and pressure are mutually exclusive "
            f"(ASDEmbeddedNodeElement parser rejects both -rot and -p)."
        )
    if stiffness_p is not None and not pressure:
        raise ValueError(
            f"{kind}: stiffness_p (-KP) is only meaningful when "
            f"pressure=True; OpenSees ignores -KP outside the u-p path."
        )


@dataclass
class ConstraintDef:
    """Base class for all constraint definitions."""
    kind: str
    master_label: str          # Assembly instance label
    slave_label: str           # Assembly instance label
    name: str | None = None    # optional user name


# ── Single-point constraint (no master/slave) ────────────────────────

@dataclass
class BCDef:
    """Homogeneous single-point constraint — fix a pattern to ground.

    Deliberately **not** a :class:`ConstraintDef` subclass: a boundary
    condition is an *essential* (Dirichlet) constraint with no master
    and no slave, so the master/slave contract every other constraint
    obeys does not apply.  After meshing it resolves — via the same
    dimension-agnostic target→nodes path the load/SP family uses — to
    one homogeneous :class:`~apeGmsh.mesh.records._loads.SPRecord`
    (``value=0.0, is_homogeneous=True``) per restrained DOF per node,
    landing in ``fem.nodes.sp`` (the same channel as
    ``g.displacements.surface``), **not** ``fem.nodes.constraints``.

    Parameters
    ----------
    target : str or list[(dim, tag)]
        Pattern to fix.  Resolved label → physical group → raw tags
        (or a mesh selection), exactly like ``g.displacements.surface``.
    target_source : str
        ``"label"`` / ``"pg"`` / ``"tag"`` / ``"auto"`` — produced by
        ``LoadsComposite._coalesce_target``; selects the resolution
        path at mesh time.
    dofs : list[int]
        Restraint **mask** (``1`` = constrained, ``0`` = free), in
        DOF order ``[ux, uy, uz, rx, ry, rz]``.  Default ``[1, 1, 1]``.
        This is the OpenSees ``ops.fix`` / ``face_sp`` convention —
        *not* the index-list convention of ``equal_dof(dofs=...)``.
    name : str or None
        Friendly name shown in summaries / the viewer.
    """
    target: str | list[tuple[int, int]]
    target_source: str = "auto"
    dofs: list[int] = field(default_factory=lambda: [1, 1, 1])
    name: str | None = None
    kind: str = field(init=False, default="bc")


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
        If given, the master is the *nearest* node in the master set
        to this point.  If ``None``, the master is the node nearest
        the master set's centroid.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves.
    tolerance : float
        Reserved.  **Not currently enforced** for master selection
        (the nearest node is taken unconditionally); kept for API
        stability and a future proximity-gated check.
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
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p, "TieDef",
        )


@dataclass
class DistributingCouplingDef(ConstraintDef):
    """
    Distributing coupling (RBE3-style force distribution).

    .. warning::

       **Not implemented.**  ``g.constraints.distributing_coupling``
       raises ``NotImplementedError``.  A correct RBE3 distributes a
       master force/moment so that ``ΣF`` and ``Σr×F`` are preserved
       while the surface deforms freely; the prior implementation was
       a mislabelled *kinematic* mean (and its ``"area"`` weighting
       was inverse-distance-from-centroid, not tributary area).  This
       dataclass is retained for a future correct implementation.

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
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p,
            "DistributingCouplingDef",
        )


@dataclass
class EmbeddedDef(ConstraintDef):
    """
    Embedded element constraint: nodes of a lower-dimensional
    element (beam, truss) are constrained to the displacement
    field of a higher-dimensional host element (solid).

    Used for reinforcement in concrete, stiffeners in shells, etc.

    Parameters
    ----------
    host_entities : list of (dim, tag), optional
        Host volume/surface entities.  Settable via
        ``g.constraints.embedded(..., host_entities=...)``; when
        omitted the whole ``host_label`` is used.
    embedded_entities : list of (dim, tag), optional
        Embedded line/surface entities.  Settable via
        ``embedded(..., embedded_entities=...)``; when omitted the
        whole ``embedded_label`` is used.
    tolerance : float
        Maximum **dimensionless barycentric excess** allowed when
        locating an embedded node inside a host element.  ``0.0``
        (the default) means strictly inside; ``0.05`` allows ~5%
        extrapolation; ``inf`` accepts everything (the pre-Phase-2
        behaviour).  An embedded node whose excess exceeds this
        threshold raises ``ValueError`` from the resolver naming the
        offending slave node and its excess — fail-loud, since
        accepting an extrapolated node silently produces an
        ``ASDEmbeddedNodeElement`` with negative shape-function
        weights and the wrong physics.
    host_coupling : {"linear"}
        Reserved keyword that pins the coupling kinematics for this
        embed.  Only ``"linear"`` is currently accepted: the embedded
        node is coupled to **3 or 4 corner nodes** of a host tri/tet
        sub-element via linear barycentric shape functions, matching
        the kinematics of OpenSees ``ASDEmbeddedNodeElement``.

        For non-simplex / higher-order hosts (tri6, tet10, quad4,
        quad8, quad9, hex8, hex20, prism6, prism15, pyramid5,
        pyramid13) the
        ``ConstraintsComposite._collect_host_subelements`` collector
        decomposes the host into linear sub-tris / sub-tets using
        corner nodes only and ignores midside nodes.  Consequence:
        the embedded coupling does NOT see the host's native
        bilinear / trilinear / quadratic displacement field — only
        a linear projection over the corner subset that brackets
        the embedded point.

        Per-hex asymmetry: two embedded nodes inside the same hex8
        may couple to *different* 4-corner subsets depending on
        which of the 6 Kuhn sub-tets contains each one.  This is
        geometrically correct under linear coupling but can surprise
        readers of the resolved records.

        The keyword is reserved (not just documented) so that a
        future ``"trilinear"`` / ``"biquadratic"`` option can be
        added without changing the public API; pre-existing models
        will keep producing identical numerical results because
        ``"linear"`` stays the default.
    """
    kind: str = field(init=False, default="embedded")
    host_entities: list[tuple[int, int]] | None = None
    embedded_entities: list[tuple[int, int]] | None = None
    tolerance: float = 0.0
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False
    host_coupling: str = "linear"

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p,
            "EmbeddedDef",
        )
        if self.host_coupling != "linear":
            raise ValueError(
                f"EmbeddedDef: host_coupling={self.host_coupling!r} "
                f"is reserved but not yet implemented.  Only "
                f"'linear' is currently accepted (coupling to 3 or "
                f"4 corner nodes via barycentric shape functions, "
                f"matching ASDEmbeddedNodeElement).  Future values "
                f"('trilinear' for hex hosts, 'biquadratic' for "
                f"quadratic hosts) would require a new OpenSees "
                f"element class that supports higher-order coupling."
            )


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
    dofs : list[int] or None
        Translational DOFs coupled to the solid.  Default [1, 2, 3].
    master_point : (x, y, z) or None
        **Ignored.**  The master is taken directly from the
        ``master_label`` node tag (this def uses bare tags, see
        above); there is no proximity master-detection.  Retained
        only for dataclass/API stability.
    tolerance : float
        **Ignored** for the same reason.  Retained for API
        stability.
    """
    kind: str = field(init=False, default="node_to_surface")
    master_point: tuple[float, float, float] | None = None
    dofs: list[int] | None = None
    tolerance: float = 1e-6


@dataclass
class NodeToSurfaceSpringDef(NodeToSurfaceDef):
    """
    Spring-based variant of :class:`NodeToSurfaceDef`.

    Same topology as ``NodeToSurfaceDef`` — a 6-DOF master node is
    coupled to the 3-DOF nodes of a surface through an intermediate
    layer of phantom nodes — but the master → phantom link is emitted
    downstream as a **stiff** ``elasticBeamColumn`` element instead of
    a kinematic ``rigidLink('beam', …)`` constraint.

    Why this variant exists
    -----------------------
    The standard ``NodeToSurfaceDef`` uses rigidLink + equalDOF. That
    chain works perfectly for most cases — rigid load transfer,
    prescribed translations at a master, fully-fixed masters — but
    breaks down when all three of the following are true:

    * The master has **free rotational DOFs** (fork support, free
      bending rotations at a simply-supported end).
    * A **moment** is applied directly to those free rotation DOFs.
    * The slave side is a solid element with ``ndf=3`` (tet4, hex8,
      …), so the rigid-link constraint back-propagates stiffness to
      the master rotations only through kinematic coupling — no
      element attaches directly to ``master.ry`` / ``master.rz``.

    Under those conditions the reduced stiffness matrix becomes
    ill-conditioned and OpenSees's solver fails with
    *"numeric analysis returns 1 -- UmfpackGenLinSolver::solve"*.

    The spring variant fixes it by giving the master's rotation DOFs
    **direct element stiffness**: each master → phantom link becomes
    a stiff ``elasticBeamColumn`` element whose 6-DOF stiffness matrix
    contributes terms on the master's rotation diagonal regardless of
    any constraint handler gymnastics. Conditioning stays good.

    Trade-offs
    ----------
    * **Pro** — robust for fork supports + moment loading.
    * **Pro** — element-level stiffness is directly assembled into K,
      so no penalty factor to tune.
    * **Con** — each master → phantom link is now an **element**, so
      the element count grows by ``n_slaves`` per coupling. For a
      typical face with ~30 slave nodes this is ~30 extra
      ``elasticBeamColumn`` elements per ``node_to_surface_spring``
      call. Negligible in solve time.
    * **Con** — approximate-rigid rather than truly rigid: the stiff
      beams have finite stiffness, so there is a tiny compliance in
      the coupling. Choose the section properties so they are orders
      of magnitude stiffer than the downstream elements.

    Parameters
    ----------
    Inherited from :class:`NodeToSurfaceDef`.

    See Also
    --------
    NodeToSurfaceDef : constraint-based variant.
    """
    kind: str = field(init=False, default="node_to_surface_spring")


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
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p,
            "TiedContactDef",
        )


@dataclass
class MortarDef(ConstraintDef):
    """
    Mortar coupling: Lagrange-multiplier space on the interface.

    .. warning::

       **Not implemented.**  ``g.constraints.mortar`` raises
       ``NotImplementedError``.  A correct mortar operator is
       ``Bᵢⱼ = ∫_Γ ψᵢ·Nⱼ dΓ`` (segment integration, dual basis,
       inf-sup/LBB).  The prior implementation was a ``tied_contact``
       collocation tie with a hardcoded unit-dependent
       ``tolerance=10.0`` mislabelled ``MORTAR``.  This dataclass is
       retained for a future correct implementation; use
       ``tied_contact`` for a collocation-based non-matching tie.

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


__all__ = [
    "ConstraintDef",
    "BCDef",
    "EqualDOFDef",
    "RigidLinkDef",
    "PenaltyDef",
    "RigidDiaphragmDef",
    "RigidBodyDef",
    "KinematicCouplingDef",
    "TieDef",
    "DistributingCouplingDef",
    "EmbeddedDef",
    "NodeToSurfaceDef",
    "NodeToSurfaceSpringDef",
    "TiedContactDef",
    "MortarDef",
]
