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

from .._coupling_control import CouplingControl


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


#: Valid ``enforce=`` routes for surface-coupling ties (ADR 0068 §1).
#:   "penalty"    → ASDEmbeddedNodeElement (penalty element, default)
#:   "penalty_al" → LadrunoEmbeddedNode    (penalty + AL + bipenalty, fork)
#:   "equation"   → EQ_Constraint          (exact; Lagrange/LadrunoProjection)
_TIE_ENFORCE_MODES = ("penalty", "penalty_al", "equation")


def _validate_tie_enforce(
    enforce: str,
    *,
    rotational: bool,
    pressure: bool,
    stiffness_p: float | None,
    control: "CouplingControl | None",
    kind: str,
) -> None:
    """Validate ``enforce=`` and the route-specific knob compatibility
    (ADR 0068 §1, INV-3).

    Routes & their knobs:
      * ``"penalty"``    → ASDEmbeddedNodeElement: ``stiffness``/
        ``stiffness_p``/``rotational``/``pressure``; NO ``control``.
      * ``"penalty_al"`` → LadrunoEmbeddedNode: ``control`` (CouplingControl
        — ``-k``/``-kAlpha``/``-host``/``-enforce al``/``-bipenalty``/
        ``-absolute``); translations-only in v1 (no ``rotational``/
        ``pressure``/``stiffness_p``).
      * ``"equation"``   → EQ_Constraint: exact, handler-enforced;
        translations-only, no penalty/control knobs.
    """
    if enforce not in _TIE_ENFORCE_MODES:
        raise ValueError(
            f"{kind}: enforce must be one of {_TIE_ENFORCE_MODES}, got "
            f"{enforce!r}."
        )
    # control (CouplingControl) configures the LadrunoEmbeddedNode element —
    # only meaningful on the penalty_al route.
    if control is not None and enforce != "penalty_al":
        raise ValueError(
            f"{kind}: control= (CouplingControl) configures the "
            f"LadrunoEmbeddedNode 'penalty_al' route and is not valid with "
            f"enforce={enforce!r}. Use enforce='penalty_al', or drop control."
        )
    # equation + penalty_al are TRANSLATIONS-ONLY routes (v1): the
    # ASDEmbeddedNodeElement penalty knobs don't apply.
    if enforce in ("equation", "penalty_al"):
        bad = []
        if rotational:
            bad.append("rotational")
        if pressure:
            bad.append("pressure")
        if stiffness_p is not None:
            bad.append("stiffness_p")
        if bad:
            route = ("an exact EQ_Constraint" if enforce == "equation"
                     else "the LadrunoEmbeddedNode tie")
            raise ValueError(
                f"{kind}: {', '.join(bad)} is an ASDEmbeddedNodeElement-only "
                f"option and cannot be combined with enforce={enforce!r} "
                f"({route} ties translations only in v1). Use "
                f"enforce='penalty', or configure the penalty_al element via "
                f"control=CouplingControl(...)."
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
class EqualDOFMixedDef(ConstraintDef):
    """
    Co-located nodes share *differently-numbered* DOFs.

    The mixed analog of :class:`EqualDOFDef`: instead of tying DOF ``i``
    on the master to DOF ``i`` on the slave, each entry of
    :attr:`dof_pairs` ties an explicit ``(retained_dof, constrained_dof)``
    couple — e.g. master ``ux`` to slave ``rz``.  Resolves, like
    ``equal_dof``, to one :class:`NodePairRecord` per co-located pair
    (kind ``equal_dof_mixed``, carrying both
    :attr:`~NodePairRecord.master_dofs` and
    :attr:`~NodePairRecord.dofs`), and emits
    ``ops.equalDOF_Mixed(R, C, numDOF, RDOF1, CDOF1, ...)`` downstream.

    Parameters
    ----------
    dof_pairs : list[(int, int)]
        ``(retained_dof, constrained_dof)`` couples, 1-based
        (``1=ux, 2=uy, 3=uz, 4=rx, 5=ry, 6=rz``).  The two members of a
        couple may differ (that is the whole point of the mixed form);
        an all-equal list is just :class:`EqualDOFDef` spelled the long way.
    tolerance : float
        Spatial distance (model units) within which two nodes are
        considered co-located.  Same semantics as
        :attr:`EqualDOFDef.tolerance`.
    master_entities : list of (dim, tag), optional
        Limit the master (retained) search to specific geometric entities.
    slave_entities : list of (dim, tag), optional
        Limit the slave (constrained) search to specific geometric entities.
    """
    kind: str = field(init=False, default="equal_dof_mixed")
    dof_pairs: list[tuple[int, int]] = field(default_factory=list)
    tolerance: float = 1e-6
    master_entities: list[tuple[int, int]] | None = None
    slave_entities: list[tuple[int, int]] | None = None

    def __post_init__(self) -> None:
        if not self.dof_pairs:
            raise ValueError(
                "equal_dof_mixed: dof_pairs is required and must be "
                "non-empty — each entry a (retained_dof, constrained_dof) "
                "1-based couple."
            )
        for pair in self.dof_pairs:
            if len(pair) != 2:
                raise ValueError(
                    f"equal_dof_mixed: each dof_pairs entry must be a "
                    f"(retained_dof, constrained_dof) couple, got {pair!r}."
                )
            rdof, cdof = pair
            for label, d in (("retained", rdof), ("constrained", cdof)):
                if not isinstance(d, int) or isinstance(d, bool) or d < 1:
                    raise ValueError(
                        f"equal_dof_mixed: {label} DOF must be a 1-based "
                        f"int, got {d!r} in {pair!r}."
                    )


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

    By default the body is emitted as a chain of ``rigidLink "beam"``
    constraints (master → each slave). Set ``as_element=True`` to emit the
    fork ``element LadrunoRigidBody`` instead (class tag 33015, **3D
    only**): the whole node set ``{master, *slaves}`` becomes one 6-DOF
    rigid body with a private internal centre-of-mass node and condensed
    mass — which the rigidLink chain cannot represent (no body mass, no
    CoM, no explicit-dynamics support). **Fork-only:** the element line
    emits on any build but needs the Ladruno fork to run.

    Parameters
    ----------
    master_point : (x, y, z)
        Master node location.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves.
    as_element : bool, default False
        Emit ``element LadrunoRigidBody`` over ``{master, *slaves}`` (3D
        only) instead of the ``rigidLink`` chain.
    mass : float or None
        Total body mass for the ``as_element`` form (``-mass``); ``None``
        condenses the mass from the slaves' own nodal mass. Ignored by the
        ``rigidLink`` form (raises if set without ``as_element``).
    """
    kind: str = field(init=False, default="rigid_body")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None
    as_element: bool = False
    mass: float | None = None

    def __post_init__(self) -> None:
        if self.mass is not None:
            if not self.as_element:
                raise ValueError(
                    "rigid_body: mass= only applies to the as_element=True "
                    "(LadrunoRigidBody) form; the rigidLink chain has no "
                    "body mass. Pass as_element=True, or drop mass."
                )
            if self.mass < 0:
                raise ValueError(
                    f"rigid_body: mass must be >= 0, got {self.mass!r}."
                )


@dataclass
class KinematicCouplingDef(ConstraintDef):
    """
    RBE2 / kinematic coupling — a reference (master) node rigidly drives
    a set of slave nodes.

    Emitted as the Ladruno-fork ``element LadrunoKinematicCoupling``
    (class tag 33012): a penalty rigid-body driver with the correct
    moment-arm transport ``u_i = u_R + θ_R × d_i`` (so an *offset*
    reference is handled rigidly — unlike the old ``equalDOF`` expansion,
    which ignored the lever arm). **Fork-only:** the deck emits on any
    build, but running it needs the Ladruno fork; stock OpenSees fails
    loud at the element line.

    Parameters
    ----------
    master_point : (x, y, z)
        Reference (master) node location — must carry the rotational DOFs
        (ndf 6 in 3D / 3 in 2D); the fork refuses a too-small reference.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become slaves (may mix 3- and
        6-DOF nodes — the element resolves the ragged layout).
    dofs : list[int] or None
        1-based dependent components to tie on each slave (``-dof``).
        ``None`` (default) ties *every DOF the slave has* (the element's
        own default), which is the right behaviour for a mixed 3/6-DOF
        slave set; pass an explicit list to restrict (e.g. ``[1, 2, 3]``
        for translations only).
    """
    kind: str = field(init=False, default="kinematic_coupling")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None
    dofs: list[int] | None = None
    #: Explicit penalty / enforcement knobs (see :class:`CouplingControl`).
    control: CouplingControl = field(default_factory=CouplingControl)


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
    #: Enforcement route (ADR 0068 §1): "penalty" (ASDEmbeddedNodeElement,
    #: default) | "penalty_al" (LadrunoEmbeddedNode) | "equation"
    #: (EQ_Constraint, exact — Lagrange/LadrunoProjection handler).
    enforce: str = "penalty"
    #: LadrunoEmbeddedNode penalty/AL/bipenalty knobs (ADR 0068 P4) — only
    #: with enforce="penalty_al"; reuses the RBE2/RBE3 CouplingControl.
    control: CouplingControl | None = None

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p, "TieDef",
        )
        _validate_tie_enforce(
            self.enforce, rotational=self.rotational, pressure=self.pressure,
            stiffness_p=self.stiffness_p, control=self.control, kind="TieDef",
        )


@dataclass
class DistributingCouplingDef(ConstraintDef):
    """
    RBE3 / distributing coupling — a reference (dependent) node is the
    weighted-average rigid-body fit of a set of independent nodes, and a
    load applied at the reference is distributed to the set as a
    statically-equivalent force pattern, **adding no stiffness** to the
    independents (the set stays free to deform).

    Emitted as the Ladruno-fork ``element LadrunoDistributingCoupling``
    (class tag 33011). It is the inverse-role sibling of
    :class:`KinematicCouplingDef` (RBE2): there the single node is the
    rigid *master*; here it is the flexible *dependent*. **Fork-only:**
    the deck emits on any build, but running it needs the Ladruno fork.

    Parameters
    ----------
    master_point : (x, y, z)
        Location of the reference (dependent) node R — must carry the
        rotational DOFs (ndf 6 in 3D / 3 in 2D) so a moment transmits.
    slave_entities : list of (dim, tag), optional
        Geometric entities whose nodes become the **independent** set
        (translations-only is fine — the fit injects no rotational
        stiffness into them).
    weighting : ``"uniform"`` | ``"area"``
        ``"uniform"`` ⇒ equal weights (``-w`` omitted ⇒ the fork
        element's equal-weight default). ``"area"`` ⇒ the resolver
        computes per-independent **tributary areas** over the slave
        surface faces (each face's area split equally among its nodes —
        the ``g.loads`` surface-tributary lumping model) and stores
        them on the record's ``weights`` ⇒ ``-w w1..wN`` emitted in the
        sorted independent order.
    """
    kind: str = field(init=False, default="distributing")
    master_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slave_entities: list[tuple[int, int]] | None = None
    weighting: str = "uniform"
    #: Explicit penalty / enforcement knobs (see :class:`CouplingControl`).
    control: CouplingControl = field(default_factory=CouplingControl)


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


@dataclass
class ReinforceDef(ConstraintDef):
    """Embedded reinforcement: tie a pre-meshed rebar line PG into a
    non-matching solid host via the Ladruno ``LadrunoEmbeddedRebar``
    coupling element (Mode P penalty).

    The geometry-side def captured by ``g.reinforce(...)``. Unlike
    :class:`EmbeddedDef` (which Kuhn-decomposes every host to a corner
    sub-tet and ties to 3-4 corners), the reinforcement resolver inverts
    each rebar node into the **actual** host element (hex8 → 8-node
    trilinear weights, tet4 → barycentric) via the guarded inverse map
    (:mod:`apeGmsh._kernel.geometry._inverse_map`), and emits one
    ``LadrunoEmbeddedRebar`` per rebar node with an anisotropic axial
    (bond-slip or perfect) + transverse-penalty tie.

    The rebar ``corotTruss`` itself and the steel / bond materials are
    declared **separately** on the bridge; this def only references the
    ``bond`` material by name (Option B layering — the geometry composite
    never resolves OpenSees tags).

    Parameters
    ----------
    master_label
        The solid host PG (``host=``).
    slave_label
        The pre-meshed rebar **line** PG (``bars=``).
    bond
        Name of a ``LadrunoBondSlip`` material for the axial law
        (``-bond``). Mutually exclusive with ``perfect``.
    perfect
        Perfect-bond axial penalty ``kAxial`` (``-perfect``). Mutually
        exclusive with ``bond``.
    bar_diameter, bar_area
        Bar geometry for ``bondScale = π·d_b·L_trib``. Supply one;
        ``bar_area`` derives ``d_b = 2·sqrt(A/π)``. Required for ``bond``.
    kt, kt_alpha
        Transverse penalty (``-kt`` / ``-ktAlpha``); ``"auto"`` (default)
        scales it to the host stiffness.
    enforce
        ``"penalty"`` (default) or ``"al"`` (augmented Lagrangian).
    bipenalty, dtcr
        Explicit bipenalty critical-time-step control (``-bipenalty
        -dtcr``). ``bipenalty=True`` + a positive ``dtcr`` budget keeps the
        coupling stiffness from shrinking the explicit critical step below
        ``dtcr``. Penalty-enforcement only (the fork auto-disables it under
        ``enforce="al"``); the ``-wcap`` host-frequency form is deferred
        with the ``-xi`` path.
    tolerance
        Acceptance threshold on the inverse-map barycentric/parametric
        excess (ADR 20 D3).
    snap
        ``False`` (default) → a rebar node outside every host raises;
        ``True`` → project it onto the nearest host + warn.
    """

    kind: str = field(init=False, default="reinforce")
    host_entities: list[tuple[int, int]] | None = None
    bars_entities: list[tuple[int, int]] | None = None
    bond: str | None = None
    perfect: float | None = None
    bar_diameter: float | None = None
    bar_area: float | None = None
    kt: float | None = None
    kt_alpha: float | None = None
    enforce: str = "penalty"
    bipenalty: bool = False
    dtcr: float | None = None
    tolerance: float = 1.0e-6
    snap: bool = False

    def __post_init__(self) -> None:
        if (self.bond is None) == (self.perfect is None):
            raise ValueError(
                "ReinforceDef: supply exactly one axial law — bond (a "
                "LadrunoBondSlip material name) or perfect (a kAxial value)"
            )
        if self.enforce not in ("penalty", "al"):
            raise ValueError(
                f"ReinforceDef: enforce must be 'penalty' or 'al', got "
                f"{self.enforce!r}"
            )
        # Explicit bipenalty critical-time-step control (R3). The `-shape`
        # path supports the user-supplied `-dtcr <dt>` budget only; `-wcap`
        # reads the host frequency and needs the `-host`/`-xi` form (deferred
        # with `-xi`/`-kt auto`). bipenalty is auto-disabled under augmented
        # Lagrangian — the fork gates it on penalty enforcement.
        if self.bipenalty:
            if self.dtcr is None:
                raise ValueError(
                    "ReinforceDef: bipenalty=True needs an explicit dtcr "
                    "(the critical-time-step budget); the -wcap host-query "
                    "form is deferred with the -xi path."
                )
            if self.enforce != "penalty":
                raise ValueError(
                    "ReinforceDef: bipenalty is gated on enforce='penalty' "
                    "(the fork auto-disables it under augmented Lagrangian)."
                )
        if self.dtcr is not None:
            if not self.bipenalty:
                raise ValueError(
                    "ReinforceDef: dtcr is only valid with bipenalty=True."
                )
            if self.dtcr <= 0:
                raise ValueError(
                    f"ReinforceDef: dtcr must be > 0, got {self.dtcr!r}"
                )
        # v1 emits the `-shape` path (apeGmsh-computed weights, host-element-
        # tag-free). `-kt auto` reads the host's getInitialStiff and needs the
        # `-host`/`-xi` form — deferred with the `-xi` optimisation. A numeric
        # kt or None (→ the fork's default transverse penalty) is supported.
        if self.kt == "auto":  # type: ignore[comparison-overlap]
            raise ValueError(
                "ReinforceDef: kt='auto' needs the `-xi` host-query path "
                "(host stiffness scaling), which is deferred; pass a numeric "
                "kt or leave it None (fork default transverse penalty)."
            )
        if self.bond is not None and self.bar_diameter is None and self.bar_area is None:
            raise ValueError(
                "ReinforceDef: a bond law needs the bar geometry for "
                "bondScale = π·d_b·L_trib — supply bar_diameter or bar_area"
            )
        if self.bar_area is not None and self.bar_area <= 0:
            raise ValueError(
                f"ReinforceDef: bar_area must be > 0, got {self.bar_area!r}"
            )
        if self.bar_diameter is not None and self.bar_diameter <= 0:
            raise ValueError(
                f"ReinforceDef: bar_diameter must be > 0, got "
                f"{self.bar_diameter!r}"
            )

    @property
    def diameter(self) -> float | None:
        """Resolved bar diameter — explicit ``bar_diameter`` or derived
        from ``bar_area`` (``d = 2·sqrt(A/π)``)."""
        if self.bar_diameter is not None:
            return self.bar_diameter
        if self.bar_area is not None:
            from math import pi, sqrt
            return 2.0 * sqrt(self.bar_area / pi)
        return None


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

    One-directional (slave conforms to master): an earlier bidirectional
    variant was removed because projecting master nodes onto slave faces as
    well produced cyclic / over-determined MPCs the constraint handler
    cannot satisfy.

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
    #: Enforcement route (ADR 0068 §1): "penalty" | "penalty_al" | "equation".
    enforce: str = "penalty"
    #: LadrunoEmbeddedNode knobs (penalty_al only); see :class:`CouplingControl`.
    control: CouplingControl | None = None

    def __post_init__(self) -> None:
        _validate_asd_embedded_options(
            self.rotational, self.pressure, self.stiffness_p,
            "TiedContactDef",
        )
        _validate_tie_enforce(
            self.enforce, rotational=self.rotational, pressure=self.pressure,
            stiffness_p=self.stiffness_p, control=self.control,
            kind="TiedContactDef",
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
    "EqualDOFMixedDef",
    "RigidLinkDef",
    "PenaltyDef",
    "RigidDiaphragmDef",
    "RigidBodyDef",
    "KinematicCouplingDef",
    "TieDef",
    "DistributingCouplingDef",
    "EmbeddedDef",
    "ReinforceDef",
    "NodeToSurfaceDef",
    "NodeToSurfaceSpringDef",
    "TiedContactDef",
    "MortarDef",
]
