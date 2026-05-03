"""
ConstraintsComposite -- Define and resolve kinematic constraints.

Two-stage pipeline:

1. **Define** (pre-mesh): factory methods store :class:`ConstraintDef`
   objects describing geometric intent.
2. **Resolve** (post-mesh): :meth:`resolve` delegates to
   :class:`ConstraintResolver` (in ``solvers/Constraints.py``) with
   caller-provided node/face maps.  Dependency-injected -- this module
   never imports PartsRegistry.

Usage::

    g.constraints.equal_dof("beam", "slab", tolerance=1e-3)
    g.constraints.tie("beam", "slab", master_entities=[(2, 5)])

    fem = g.mesh.queries.get_fem_data(dim=2)
    nm  = g.parts.build_node_map(fem.nodes.ids, fem.nodes.coords)
    fm  = g.parts.build_face_map(nm)
    recs = g.constraints.resolve(
        fem.nodes.ids, fem.nodes.coords, node_map=nm, face_map=fm,
    )
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.mesh._record_set import NodeConstraintSet as ConstraintSet
from apeGmsh.solvers.Constraints import (
    ConstraintDef,
    ConstraintRecord,
    ConstraintResolver,
    EqualDOFDef,
    RigidLinkDef,
    PenaltyDef,
    RigidDiaphragmDef,
    RigidBodyDef,
    KinematicCouplingDef,
    TieDef,
    DistributingCouplingDef,
    EmbeddedDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    TiedContactDef,
    MortarDef,
)

_DISPATCH: dict[type, str] = {
    EqualDOFDef:             "_resolve_node_pair",
    RigidLinkDef:            "_resolve_node_pair",
    PenaltyDef:              "_resolve_node_pair",
    RigidDiaphragmDef:       "_resolve_diaphragm",
    RigidBodyDef:            "_resolve_kinematic",
    KinematicCouplingDef:    "_resolve_kinematic",
    TieDef:                  "_resolve_face_slave",
    DistributingCouplingDef: "_resolve_kinematic",
    EmbeddedDef:             "_resolve_embedded",
    # Both the constraint- and spring-based variants go through the
    # same composite-level lookup; the final call to the resolver is
    # dispatched off ``_RESOLVER_METHOD`` at runtime so the lookup
    # code is not duplicated.
    NodeToSurfaceDef:        "_resolve_node_to_surface",
    NodeToSurfaceSpringDef:  "_resolve_node_to_surface",
    TiedContactDef:          "_resolve_face_both",
    MortarDef:               "_resolve_face_both",
}

_RESOLVER_METHOD: dict[type, str] = {
    EqualDOFDef:             "resolve_equal_dof",
    RigidLinkDef:            "resolve_rigid_link",
    PenaltyDef:              "resolve_penalty",
    RigidDiaphragmDef:       "resolve_rigid_diaphragm",
    RigidBodyDef:            "resolve_kinematic_coupling",
    KinematicCouplingDef:    "resolve_kinematic_coupling",
    TieDef:                  "resolve_tie",
    DistributingCouplingDef: "resolve_distributing",
    NodeToSurfaceDef:        "resolve_node_to_surface",
    NodeToSurfaceSpringDef:  "resolve_node_to_surface_spring",
    TiedContactDef:          "resolve_tied_contact",
    MortarDef:               "resolve_mortar",
}

_FACE_TYPES = (TieDef, TiedContactDef, MortarDef)

_ConstraintT = TypeVar("_ConstraintT", bound=ConstraintDef)


class ConstraintsComposite:
    """Solver-agnostic kinematic-constraint composite — declare on
    geometry, resolve to nodes after meshing.

    Two-stage pipeline
    ------------------
    1. **Declare** (pre-mesh): the factory methods on this composite
       (``equal_dof``, ``rigid_link``, ``rigid_diaphragm``, ``tie``, …)
       store :class:`~apeGmsh.solvers.Constraints.ConstraintDef`
       dataclasses describing *intent* at the geometry level. Defs
       carry no node tags and survive remeshing.
    2. **Resolve** (post-mesh): :meth:`resolve` (called automatically
       by :meth:`Mesh.queries.get_fem_data`) walks the def list and
       hands each one to
       :class:`~apeGmsh.solvers.Constraints.ConstraintResolver`,
       which produces concrete
       :class:`~apeGmsh.solvers.Constraints.ConstraintRecord` objects
       — actual node tags, weights, and offset vectors.

    The resolved records land on the FEM broker:

    * **node-pair / node-group / node_to_surface** records →
      ``fem.nodes.constraints``
    * **surface-coupling / interpolation** records →
      ``fem.elements.constraints``

    Constraint taxonomy
    -------------------
    Five tiers, ordered by topology and the role each plays in a
    structural model:

    ============= ===================================================== =================================
    Tier          Methods                                               Record family
    ============= ===================================================== =================================
    1 — Pair      :meth:`equal_dof`, :meth:`rigid_link`,                ``NodePairRecord``
                  :meth:`penalty`
    2 — Group     :meth:`rigid_diaphragm`, :meth:`rigid_body`,          ``NodeGroupRecord``
                  :meth:`kinematic_coupling`
    2b — Mixed    :meth:`node_to_surface`,                              ``NodeToSurfaceRecord``
                  :meth:`node_to_surface_spring`                        (+ phantom nodes)
    3 — Surface   :meth:`tie`, :meth:`distributing_coupling`,           ``InterpolationRecord``
                  :meth:`embedded`
    4 — Contact   :meth:`tied_contact`, :meth:`mortar`                  ``SurfaceCouplingRecord``
    ============= ===================================================== =================================

    All constraints ultimately express the linear MPC equation
    ``u_slave = C · u_master``. Tiers differ in **how** ``C`` is
    built — by node co-location (Tier 1), kinematic transformation
    around a master point (Tier 2), shape-function interpolation
    (Tier 3), or numerical integration on the interface (Tier 4).

    Target identification
    ---------------------
    Most methods identify their master and slave sides by **part
    label** (a key of ``g.parts._instances``). :meth:`_add_def`
    validates both labels against the registry and raises
    ``KeyError`` on a typo::

        g.constraints.tie(master_label="column",
                          slave_label="slab",
                          master_entities=[(2, 13)],   # optional scope
                          slave_entities=[(2, 17)])

    Optional ``master_entities`` / ``slave_entities`` (list of
    ``(dim, tag)``) narrow the search to a subset of the part's
    entities — useful when a part has many surfaces and only one is
    the interface.

    Exceptions to the part-label scheme
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    * :meth:`node_to_surface` and :meth:`node_to_surface_spring`
      take **bare tags** instead. The ``master`` is a Gmsh point
      entity (``dim=0``) and ``slave`` is one or more surface
      entities (``dim=2``). Both arguments accept ``int``, ``str``,
      or ``(dim, tag)``; label validation is skipped.
    * :meth:`embedded` uses ``host_label`` / ``embedded_label`` to
      mirror the host/embedded vocabulary, but the lookup logic
      otherwise matches the part-label scheme.

    Resolution semantics
    --------------------
    :meth:`resolve` is dependency-injected — it never imports
    ``PartsRegistry``. The caller (typically
    ``Mesh.queries.get_fem_data``) supplies:

    * ``node_map``: ``{part_label → set[int]}`` of mesh node tags
    * ``face_map``: ``{part_label → ndarray(F, n_per_face)}``
      built only when surface constraints (Tier 3 / 4) are present.

    See Also
    --------
    apeGmsh.solvers.Constraints :
        Module-level taxonomy and theory.
    apeGmsh.solvers._constraint_defs :
        Stage-1 dataclasses with full per-method theory.
    apeGmsh.solvers._constraint_resolver.ConstraintResolver :
        Stage-2 implementation.
    apeGmsh.mesh._record_set.NodeConstraintSet :
        Iteration helpers (``rigid_link_groups``, ``equal_dofs``,
        ``rigid_diaphragms``, ``pairs``).

    Examples
    --------
    Declare a mix of constraints, mesh, and read out grouped
    rigid-link masters for OpenSees emission::

        with apeGmsh(model_name="frame") as g:
            # Tier 1 — co-located nodes share x/y/z
            g.constraints.equal_dof("col", "beam", dofs=[1, 2, 3])

            # Tier 2 — slab nodes follow the centre-of-mass node
            g.constraints.rigid_diaphragm(
                "slab", "slab_master",
                master_point=(2.5, 2.5, 3.0),
                plane_normal=(0, 0, 1),
            )

            # Tier 3 — non-matching shell-to-solid interface
            g.constraints.tie(
                "shell", "solid",
                master_entities=[(2, 17)],
                slave_entities=[(2, 41)],
            )

            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data(dim=3)

            for master, slaves in fem.nodes.constraints.rigid_link_groups():
                for slave in slaves:
                    ops.rigidLink("beam", master, slave)
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.constraint_defs: list[ConstraintDef] = []
        self.constraint_records: list[ConstraintRecord] = []

    def _add_def(self, defn: _ConstraintT) -> _ConstraintT:
        """Validate labels and store a constraint definition.
        Label validation is skipped for NodeToSurfaceDef (bare tags)."""
        if not isinstance(defn, (NodeToSurfaceDef, EmbeddedDef)):
            parts = getattr(self._parent, "parts", None)
            if parts is not None and hasattr(parts, "_instances"):
                for lbl in (defn.master_label, defn.slave_label):
                    if lbl not in parts._instances:
                        raise KeyError(
                            f"Part label \'{lbl}\' not found in g.parts.  "
                            f"Available: {list(parts._instances)}"
                        )
        self.constraint_defs.append(defn)
        return defn

    # ── Tier 1 — Node-to-Node ────────────────────────────────────────
    def equal_dof(self, master_label, slave_label, *, master_entities=None,
                  slave_entities=None, dofs=None, tolerance=1e-6,
                  name=None) -> EqualDOFDef:
        """Tie matching DOFs between **co-located** node pairs.

        At resolution time the resolver finds every master node
        whose coordinates match a slave node within ``tolerance``
        and emits one
        :class:`~apeGmsh.solvers.Constraints.NodePairRecord` per
        match. Each pair becomes ``ops.equalDOF(master, slave, *dofs)``
        downstream — i.e. ``u_slave[i] = u_master[i]`` for every
        ``i`` in ``dofs``.

        Use this for **conformal** interfaces only — meshes that share
        nodes at the boundary. For non-matching meshes use :meth:`tie`.

        Parameters
        ----------
        master_label : str
            Part label whose nodes drive the constraint.
        slave_label : str
            Part label whose matching nodes are slaved.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict the node search to specific Gmsh entities of
            each side. Useful when only one face of a multi-face
            part is the interface.
        dofs : list[int], optional
            1-based DOF indices to constrain (``1=ux, 2=uy, 3=uz,
            4=rx, 5=ry, 6=rz``). ``None`` (default) means *all DOFs
            available* — the actual count depends on the model's
            ``ndf``.
        tolerance : float, default 1e-6
            Maximum distance (in model units) between two nodes for
            them to be treated as co-located. **Unit-sensitive**:
            ``1e-3`` for millimetre models, ``1e-6`` for metre
            models.
        name : str, optional
            Friendly name shown in :meth:`summary` and the viewer.

        Returns
        -------
        EqualDOFDef
            The stored definition; the same object is appended to
            ``self.constraint_defs``.

        Raises
        ------
        KeyError
            If ``master_label`` or ``slave_label`` is not in
            ``g.parts``.

        See Also
        --------
        tie : Non-matching mesh equivalent (shape-function projection).
        rigid_link : Add a kinematic offset on top of co-location.

        Examples
        --------
        Translational continuity between a column and a beam at a
        joint::

            g.constraints.equal_dof(
                "column", "beam",
                dofs=[1, 2, 3],
                tolerance=1e-3,        # mm model
            )
        """
        return self._add_def(EqualDOFDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def rigid_link(self, master_label, slave_label, *, link_type="beam",
                   master_point=None, slave_entities=None,
                   tolerance=1e-6, name=None) -> RigidLinkDef:
        """Rigid bar between a master node and one or more slave nodes.

        Each slave node is constrained to follow the master through
        a rigid offset arm ``r = x_slave − x_master``::

            link_type="beam":     u_s = u_m + θ_m × r,   θ_s = θ_m
            link_type="rod":      u_s = u_m + θ_m × r,   θ_s free

        Use ``"beam"`` for fully rigid kinematic offsets (eccentric
        connections, lumped-mass arms, fictitious rigid extensions).
        Use ``"rod"`` when you want to transmit translation but leave
        the slave free to rotate — e.g. pinned eccentric supports.

        Parameters
        ----------
        master_label : str
            Part label that owns the master node. The master is
            identified inside this part either by ``master_point``
            (proximity match) or by being the unique node when the
            part collapses to a single point.
        slave_label : str
            Part label whose nodes become slaves.
        link_type : ``"beam"`` or ``"rod"``, default ``"beam"``
            ``"beam"`` couples 6 DOFs with rotational offset;
            ``"rod"`` couples translations only.
        master_point : (x, y, z), optional
            Explicit master coordinates. If ``None``, the resolver
            picks the master node by proximity within ``tolerance``.
        slave_entities : list of (dim, tag), optional
            Restrict the slave node search to specific entities.
        tolerance : float, default 1e-6
            Proximity tolerance for master-node detection.
        name : str, optional
            Friendly name.

        Returns
        -------
        RigidLinkDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        kinematic_coupling : Same idea, but lets you pick which DOFs
            to couple instead of the fixed beam/rod sets.
        node_to_surface : When the slave side has only translational
            DOFs (3-DOF solid nodes).

        Examples
        --------
        Lumped-mass arm at the top of a tower::

            g.constraints.rigid_link(
                "tower_top", "lumped_mass",
                link_type="beam",
                master_point=(0, 0, 30.0),
            )
        """
        return self._add_def(RigidLinkDef(
            master_label=master_label, slave_label=slave_label,
            link_type=link_type, master_point=master_point,
            slave_entities=slave_entities, tolerance=tolerance, name=name))

    def penalty(self, master_label, slave_label, *, stiffness=1e10,
                dofs=None, tolerance=1e-6, name=None) -> PenaltyDef:
        """Soft-spring (penalty) coupling between co-located node pairs.

        Numerically approximates :meth:`equal_dof` as
        ``stiffness → ∞``. The resolver still requires master and
        slave nodes to be co-located within ``tolerance``, but
        downstream the constraint is enforced by inserting a stiff
        spring element between each pair instead of a hard MPC.

        Use this when:

        * The hard ``equal_dof`` constraint causes the
          constraint-handler to ill-condition the reduced stiffness
          matrix (typical with mismatched DOF spaces).
        * You want a tunable interface compliance — e.g. a soft
          contact at a bearing pad.

        Parameters
        ----------
        master_label : str
            Part label of the master side.
        slave_label : str
            Part label of the slave side.
        stiffness : float, default 1e10
            Penalty spring stiffness in force/length units. Pick
            ~3–6 orders of magnitude above the stiffest neighbouring
            element diagonal — overshoot causes ill-conditioning,
            undershoot leaks displacement.
        dofs : list[int], optional
            1-based DOFs to penalise. ``None`` = all available.
        tolerance : float, default 1e-6
            Spatial co-location tolerance.
        name : str, optional
            Friendly name.

        Returns
        -------
        PenaltyDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        equal_dof : Hard MPC equivalent (no tunable stiffness).
        """
        return self._add_def(PenaltyDef(
            master_label=master_label, slave_label=slave_label,
            stiffness=stiffness, dofs=dofs, tolerance=tolerance, name=name))

    # ── Tier 2 — Node-to-Group ───────────────────────────────────────
    def rigid_diaphragm(self, master_label, slave_label, *,
                        master_point=(0., 0., 0.),
                        plane_normal=(0., 0., 1.),
                        constrained_dofs=None, plane_tolerance=1.0,
                        name=None) -> RigidDiaphragmDef:
        """In-plane rigid floor — slaves follow master in the
        diaphragm plane.

        Classic use: each floor of a multi-storey building. All
        slab nodes within ``plane_tolerance`` of the diaphragm
        plane share in-plane translation and rotation about the
        out-of-plane axis with the master node, while remaining
        free in the out-of-plane direction.

        Resolution emits a single
        :class:`~apeGmsh.solvers.Constraints.NodeGroupRecord`
        with one master and many slaves. Downstream this becomes
        ``ops.rigidDiaphragm(perpDirn, master, *slaves)``.

        Parameters
        ----------
        master_label : str
            Part label that contains (or whose proximity will
            select) the master node — typically a centre-of-mass
            point.
        slave_label : str
            Part label whose nodes are gathered into the diaphragm.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the master node. Used to disambiguate
            when the master part has more than one node.
        plane_normal : (nx, ny, nz), default (0, 0, 1)
            Unit normal to the diaphragm plane. ``(0, 0, 1)`` is a
            horizontal floor; ``(0, 1, 0)`` is a vertical wall, etc.
        constrained_dofs : list[int], optional
            DOFs slaved to the master. Default for a horizontal
            floor (Z up) is ``[1, 2, 6]`` — ux, uy, rz. For a
            vertical wall use ``[1, 3, 5]``.
        plane_tolerance : float, default 1.0
            Perpendicular distance (in model units) from the
            diaphragm plane within which a slave node is
            collected. **Unit-sensitive** — set this to a fraction
            of slab thickness.
        name : str, optional
            Friendly name.

        Returns
        -------
        RigidDiaphragmDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        kinematic_coupling : When you need a different DOF subset
            than ``[1, 2, 6]`` and don't need plane filtering.
        rigid_body : When all 6 DOFs must follow the master.

        Examples
        --------
        A horizontal slab at z = 3.0 m::

            g.constraints.rigid_diaphragm(
                "slab", "slab_master",
                master_point=(2.5, 2.5, 3.0),
                plane_normal=(0, 0, 1),
                constrained_dofs=[1, 2, 6],
                plane_tolerance=0.05,
            )
        """
        return self._add_def(RigidDiaphragmDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, plane_normal=plane_normal,
            constrained_dofs=constrained_dofs or [1, 2, 6],
            plane_tolerance=plane_tolerance, name=name))

    def rigid_body(self, master_label, slave_label, *,
                   master_point=(0., 0., 0.), name=None) -> RigidBodyDef:
        """Fully rigid cluster — every slave DOF follows the master.

        All six DOFs (``ux, uy, uz, rx, ry, rz``) of every node in
        the slave part follow the master node through a rigid
        transformation::

            u_s = u_m + θ_m × (x_s − x_m)
            θ_s = θ_m

        Use this for genuinely rigid pieces (bearing blocks, lumped
        rigid masses) where the slave region must not deform.

        Parameters
        ----------
        master_label : str
            Part label that contains (or whose proximity selects)
            the master node.
        slave_label : str
            Part label whose nodes are gathered into the rigid
            body.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the master node.
        name : str, optional
            Friendly name.

        Returns
        -------
        RigidBodyDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        kinematic_coupling : Same topology but with a user-selectable
            DOF subset.
        rigid_diaphragm : In-plane variant with plane filtering.
        """
        return self._add_def(RigidBodyDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, name=name))

    def kinematic_coupling(self, master_label, slave_label, *,
                           master_point=(0., 0., 0.), dofs=None,
                           name=None) -> KinematicCouplingDef:
        """Generalised one-master-many-slaves coupling on a chosen
        DOF subset.

        The "parent" of :meth:`rigid_diaphragm` and :meth:`rigid_body`
        — they are special cases with pre-set DOF lists. Use this
        directly when you need a non-standard combination, e.g.::

            * vertical-only follower: dofs=[3]
            * 2-D in-plane rigid:     dofs=[1, 2, 6]
            * symmetry plane:         dofs=[1, 4, 5]

        Resolution emits a single
        :class:`~apeGmsh.solvers.Constraints.NodeGroupRecord`.
        Downstream this is typically expanded to one
        ``ops.equalDOF`` per slave.

        Parameters
        ----------
        master_label : str
            Part label that owns the master node.
        slave_label : str
            Part label whose nodes are slaved.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the master node.
        dofs : list[int], optional
            1-based DOFs to couple. Default ``[1, 2, 3, 4, 5, 6]``
            (full 6-DOF, equivalent to :meth:`rigid_body`).
        name : str, optional
            Friendly name.

        Returns
        -------
        KinematicCouplingDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.
        """
        return self._add_def(KinematicCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, dofs=dofs or [1, 2, 3, 4, 5, 6],
            name=name))

    # ── Tier 3 — Node-to-Surface ─────────────────────────────────────
    def tie(self, master_label, slave_label, *, master_entities=None,
            slave_entities=None, dofs=None, tolerance=1.0,
            name=None) -> TieDef:
        """Non-matching mesh tie via shape-function interpolation.

        For each slave node, the resolver finds the closest master
        element face, projects the node onto it, and constrains its
        DOFs to the master corner DOFs through that face's shape
        functions::

            u_slave = Σ N_i(ξ, η) · u_master_i

        where ``(ξ, η)`` are the projected parametric coordinates
        and ``N_i`` are the master face's shape functions (tri3,
        quad4, tri6, quad8 supported). This is what Abaqus
        ``*TIE`` does — it preserves displacement continuity across
        non-matching meshes.

        Resolution emits one
        :class:`~apeGmsh.solvers.Constraints.InterpolationRecord`
        per successfully projected slave node. apeGmsh emits these
        downstream as ``ASDEmbeddedNodeElement`` penalty elements
        (default K = 1e18; tunable via
        ``g.opensees.ingest.constraints(fem, tie_penalty=…)``).

        Parameters
        ----------
        master_label : str
            Part label of the master surface (the side whose mesh
            will provide the shape functions).
        slave_label : str
            Part label of the slave surface (whose nodes are
            projected).
        master_entities : list of (dim, tag), optional
            Restrict the master surface to specific Gmsh
            entities. **Strongly recommended** when the master
            part has more than one face.
        slave_entities : list of (dim, tag), optional
            Restrict the slave surface to specific entities.
        dofs : list[int], optional
            DOFs to tie. ``None`` (default) ties all translational
            DOFs available — typically ``[1, 2, 3]``.
        tolerance : float, default 1.0
            Maximum allowed projection distance from a slave node
            to the master surface. Slave nodes farther than this
            are silently skipped — set generously if the two
            meshes have a small geometric gap, but not so large
            that the wrong face is selected. **Unit-sensitive.**
        name : str, optional
            Friendly name.

        Returns
        -------
        TieDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        equal_dof : Conformal-mesh equivalent (no interpolation).
        tied_contact : Bidirectional surface-to-surface tie.
        mortar : Higher-accuracy variant via Lagrange multipliers.

        Notes
        -----
        Master/slave choice matters for accuracy. As a rule:

        * The master should have the **finer** mesh (more shape
          functions to project onto).
        * The slave should have the **coarser** mesh (fewer
          projection operations).

        Examples
        --------
        Shell-to-solid tie at a column-top interface::

            g.constraints.tie(
                "shell_floor", "solid_column",
                master_entities=[(2, 17)],     # column top face
                slave_entities=[(2, 41)],      # shell bottom face
                tolerance=5.0,                 # mm gap
            )
        """
        return self._add_def(TieDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def distributing_coupling(self, master_label, slave_label, *,
                              master_point=(0., 0., 0.), dofs=None,
                              weighting="uniform",
                              name=None) -> DistributingCouplingDef:
        """Distribute a master-point load over a slave surface
        without rigidising it.

        Unlike :meth:`rigid_body` or :meth:`tie`, the slave surface
        is **not** kinematically constrained — it remains free to
        deform. Forces and moments applied at the master are
        distributed to the slave nodes as consistent nodal forces,
        with weights chosen so that overall force and moment
        equilibrium are preserved.

        This is the right primitive for:

        * Applying a single concentrated load to a face without
          introducing artificial stiffness.
        * Connecting a beam reference node to a solid face when
          the connection is statically equivalent but should not
          enforce rigid kinematics.

        Parameters
        ----------
        master_label : str
            Part label that owns the reference node.
        slave_label : str
            Part label of the slave surface.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the reference (master) node.
        dofs : list[int], optional
            DOFs distributed from master to slave. ``None`` = all
            translational DOFs.
        weighting : ``"uniform"`` or ``"area"``, default ``"uniform"``
            How nodal weights are computed:

            * ``"uniform"`` — equal weights at every slave node.
            * ``"area"`` — tributary-area weights (more physical
              for non-uniform meshes).
        name : str, optional
            Friendly name.

        Returns
        -------
        DistributingCouplingDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        rigid_body : Kinematically rigid alternative (no
            compliance, but no compatibility either).
        node_to_surface : When the slave side is a 3-DOF solid and
            you also want the master's rotational DOFs to drive
            translations through a rigid arm.
        """
        return self._add_def(DistributingCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, dofs=dofs, weighting=weighting,
            name=name))

    def embedded(self, host_label, embedded_label, *, tolerance=1.0,
                 name=None) -> EmbeddedDef:
        """Embed lower-dimensional elements inside a host volume or
        surface.

        Each node of the embedded part is constrained to the
        displacement field of the host element it falls inside via
        host shape functions. Used for **rebar in concrete**,
        stiffeners in shells, fibres in composite hosts, etc.

        Currently supports:

        * **3-D host:** tet4 (Gmsh element type 4) volumes.
        * **2-D host:** tri3 (Gmsh element type 2) surfaces.

        Higher-order or hex/quad hosts are not yet supported and
        will be silently skipped — fall back to :meth:`tie` if you
        need that.

        The resolver automatically drops embedded nodes that
        coincide with host element corners, since those are
        already rigidly attached through shared connectivity.

        Parameters
        ----------
        host_label : str
            Part label whose tet4/tri3 elements form the host
            field. Stored internally as ``master_label``.
        embedded_label : str
            Part label whose nodes are embedded. Stored as
            ``slave_label``. (Label validation is bypassed for
            ``EmbeddedDef`` — these labels may also be physical
            group names if no part registry is in use.)
        tolerance : float, default 1.0
            Search tolerance for locating each embedded node
            inside a host element. **Unit-sensitive.**
        name : str, optional
            Friendly name.

        Returns
        -------
        EmbeddedDef

        Notes
        -----
        Emitted downstream as ``ASDEmbeddedNodeElement``. The
        ``host_label`` / ``embedded_label`` argument names mirror
        Abaqus's ``*EMBEDDED ELEMENT`` vocabulary; internally the
        composite still stores them as ``master``/``slave`` for
        consistency with the rest of the constraint records.

        Examples
        --------
        Rebar curve embedded inside a concrete tet mesh::

            g.constraints.embedded(
                host_label="concrete_block",
                embedded_label="rebar_curve",
                tolerance=2.0,        # mm
            )
        """
        return self._add_def(EmbeddedDef(
            master_label=host_label, slave_label=embedded_label,
            tolerance=tolerance, name=name))

    # Level 2b
    def node_to_surface(self, master, slave, *,
                        dofs=None, tolerance=1e-6,
                        name=None):
        """6-DOF node to 3-DOF surface coupling via phantom nodes.

        Creates a single constraint that aggregates all surface
        entities in *slave*.  Shared-edge mesh nodes are deduplicated
        so each original slave node gets exactly one phantom.

        Parameters
        ----------
        master : int, str, or (dim, tag)
            The 6-DOF reference node.
        slave : int, str, or (dim, tag)
            The surface(s) to couple.  If it resolves to multiple
            surface entities, they are combined into a single
            constraint and slave nodes are deduplicated.

        Returns
        -------
        NodeToSurfaceDef
            A single def covering all resolved surface entities.
        """
        from ._helpers import resolve_to_tags
        m_tags = resolve_to_tags(master, dim=0, session=self._parent)
        s_tags = resolve_to_tags(slave,  dim=2, session=self._parent)
        master_tag = m_tags[0]

        if name is None:
            m_name = str(master) if isinstance(master, str) else str(master_tag)
            s_name = str(slave) if isinstance(slave, str) else "surface"
            display_name = f"{m_name} → {s_name}"
        else:
            display_name = name

        # Store ALL surface tags as a comma-separated string so the
        # resolver can union their slave nodes and deduplicate.
        slave_label = ",".join(str(int(t)) for t in s_tags)

        return self._add_def(NodeToSurfaceDef(
            master_label=str(master_tag),
            slave_label=slave_label,
            dofs=dofs, tolerance=tolerance,
            name=display_name))

    def node_to_surface_spring(self, master, slave, *,
                               dofs=None, tolerance=1e-6,
                               name=None):
        """Spring-based variant of :meth:`node_to_surface`.

        Identical topology and call signature, but the master → phantom
        links are tagged for downstream emission as stiff
        ``elasticBeamColumn`` elements instead of kinematic
        ``rigidLink('beam', ...)`` constraints. Use this variant when
        the master carries **free rotational DOFs** (fork support on a
        solid end face) that receive direct moment loading — the
        constraint-based variant of ``node_to_surface`` can produce an
        ill-conditioned reduced stiffness matrix in that case because
        the master rotation DOFs get stiffness only through the
        kinematic constraint back-propagation, with nothing attaching
        directly to them.

        See :class:`~apeGmsh.solvers.Constraints.NodeToSurfaceSpringDef`
        for the full rationale.

        Emission in OpenSees::

            # Each master → phantom link becomes a stiff beam element
            next_eid = max_tet_eid + 1
            for master, slaves in fem.nodes.constraints.stiff_beam_groups():
                for phantom in slaves:
                    ops.element(
                        'elasticBeamColumn', next_eid,
                        master, phantom,
                        A_big, E, I_big, I_big, J_big, transf_tag,
                    )
                    next_eid += 1

            # equalDOFs are unchanged from the normal variant
            for pair in fem.nodes.constraints.equal_dofs():
                ops.equalDOF(
                    pair.master_node, pair.slave_node, *pair.dofs)

        Parameters
        ----------
        Same as :meth:`node_to_surface`.

        Returns
        -------
        NodeToSurfaceSpringDef
        """
        from ._helpers import resolve_to_tags
        m_tags = resolve_to_tags(master, dim=0, session=self._parent)
        s_tags = resolve_to_tags(slave,  dim=2, session=self._parent)
        master_tag = m_tags[0]

        if name is None:
            m_name = str(master) if isinstance(master, str) else str(master_tag)
            s_name = str(slave) if isinstance(slave, str) else "surface"
            display_name = f"{m_name} \u2192 {s_name} (spring)"
        else:
            display_name = name

        slave_label = ",".join(str(int(t)) for t in s_tags)

        return self._add_def(NodeToSurfaceSpringDef(
            master_label=str(master_tag),
            slave_label=slave_label,
            dofs=dofs, tolerance=tolerance,
            name=display_name))

    # ── Tier 4 — Surface-to-Surface ──────────────────────────────────
    def tied_contact(self, master_label, slave_label, *,
                     master_entities=None, slave_entities=None,
                     dofs=None, tolerance=1.0,
                     name=None) -> TiedContactDef:
        """Bidirectional surface-to-surface tie.

        Conceptually a :meth:`tie` applied in **both directions** —
        slave nodes are projected onto the master surface, and
        master nodes are also projected onto the slave surface, so
        every node on either side is interpolated against the
        opposite mesh. Useful when neither side can be picked as
        clearly finer than the other and you want a symmetric
        treatment.

        Resolution emits
        :class:`~apeGmsh.solvers.Constraints.SurfaceCouplingRecord`
        objects on ``fem.elements.constraints``.

        Parameters
        ----------
        master_label : str
            Part label of the first surface.
        slave_label : str
            Part label of the second surface.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities.
        dofs : list[int], optional
            DOFs to tie. ``None`` = all translational.
        tolerance : float, default 1.0
            Maximum projection distance. **Unit-sensitive.**
        name : str, optional
            Friendly name.

        Returns
        -------
        TiedContactDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        tie : One-directional tie (slave-projected only).
        mortar : Mathematically rigorous Lagrange-multiplier
            coupling.
        """
        return self._add_def(TiedContactDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def mortar(self, master_label, slave_label, *,
               master_entities=None, slave_entities=None,
               dofs=None, integration_order=2,
               name=None) -> MortarDef:
        """Mortar surface coupling — Lagrange multipliers on the
        interface.

        The most rigorous of the surface-coupling primitives.
        Mortar methods introduce a Lagrange-multiplier space
        ``ψ_i`` on the slave side and integrate the coupling
        operator over the **overlapping** surface segments::

            B_ij = ∫_Γ ψ_i · N_j dΓ

        where ``N_j`` are master shape functions. This satisfies
        the inf-sup (LBB) condition and produces an optimally
        accurate non-matching coupling — preferable to
        :meth:`tie` / :meth:`tied_contact` when accuracy at the
        interface is important (mixed-dimension models, contact
        mechanics, optimisation).

        Parameters
        ----------
        master_label : str
            Part label of the master surface.
        slave_label : str
            Part label of the slave surface (Lagrange multipliers
            live here).
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities.
        dofs : list[int], optional
            DOFs to couple.
        integration_order : int, default 2
            Gauss quadrature order for the coupling integral.
            Increase for curved interfaces or higher-order
            elements.
        name : str, optional
            Friendly name.

        Returns
        -------
        MortarDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.

        See Also
        --------
        tied_contact : Cheaper non-matching alternative.
        """
        return self._add_def(MortarDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, integration_order=integration_order, name=name))

    def validate_pre_mesh(self) -> None:
        """No-op: constraints validate targets eagerly at ``_add_def``.

        Present so :meth:`Mesh.generate` can invoke ``validate_pre_mesh``
        on all three composites uniformly.
        """
        return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(self, node_tags, node_coords, elem_tags=None,
                connectivity=None, *, node_map=None, face_map=None) -> ConstraintSet:
        has_face_constraints = any(
            isinstance(d, _FACE_TYPES) for d in self.constraint_defs)
        if has_face_constraints and face_map is None:
            raise TypeError(
                "Surface constraints are defined but face_map=None. "
                "Call resolve(..., face_map=parts.build_face_map(node_map)) "
                "or use Mesh.get_fem_data() which builds face_map automatically."
            )

        resolver = ConstraintResolver(
            node_tags=node_tags, node_coords=node_coords,
            elem_tags=elem_tags, connectivity=connectivity)
        all_nodes = set(int(t) for t in node_tags)
        records: list[ConstraintRecord] = []

        for defn in self.constraint_defs:
            dispatch = _DISPATCH.get(type(defn))
            if dispatch is None:
                warnings.warn(
                    f"No dispatch for {type(defn).__name__}, skipping.",
                    stacklevel=2)
                continue
            result = getattr(self, dispatch)(
                resolver, defn, node_map or {}, face_map or {}, all_nodes)
            if isinstance(result, list):
                records.extend(result)
            elif result is not None:
                records.append(result)

        self.constraint_records = records
        return ConstraintSet(records)

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------
    def _resolve_nodes(self, label, role, defn, node_map, all_nodes):
        import gmsh
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            tags: set[int] = set()
            for dim, tag in selected:
                try:
                    nt, _, _ = gmsh.model.mesh.getNodes(
                        dim=int(dim), tag=int(tag),
                        includeBoundary=True, returnParametricCoord=False)
                    tags.update(int(t) for t in nt)
                except Exception:
                    pass
            if tags:
                return tags
        return node_map.get(label, all_nodes)

    def _resolve_faces(self, label, role, defn, face_map):
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            parts = getattr(self._parent, "parts", None)
            if parts is not None:
                return parts._collect_surface_faces(selected)
        return face_map.get(label, np.empty((0, 0), dtype=int))

    def _resolve_node_pair(self, resolver, defn, node_map, face_map, all_nodes):
        m = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        method = getattr(resolver, _RESOLVER_METHOD[type(defn)])
        return method(defn, m, s)

    def _resolve_diaphragm(self, resolver, defn, node_map, face_map, all_nodes):
        m = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        return resolver.resolve_rigid_diaphragm(defn, m | s)

    def _resolve_kinematic(self, resolver, defn, node_map, face_map, all_nodes):
        m = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        method = getattr(resolver, _RESOLVER_METHOD[type(defn)])
        return method(defn, m, s)

    def _resolve_face_slave(self, resolver, defn, node_map, face_map, all_nodes):
        m_faces = self._resolve_faces(defn.master_label, "master", defn, face_map)
        s_nodes = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        if m_faces.size == 0:
            return []
        return resolver.resolve_tie(defn, m_faces, s_nodes)

    def _resolve_face_both(self, resolver, defn, node_map, face_map, all_nodes):
        m_faces = self._resolve_faces(defn.master_label, "master", defn, face_map)
        s_faces = self._resolve_faces(defn.slave_label, "slave", defn, face_map)
        m_nodes = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s_nodes = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        if m_faces.size == 0 or s_faces.size == 0:
            return []
        method = getattr(resolver, _RESOLVER_METHOD[type(defn)])
        return method(defn, m_faces, s_faces, m_nodes, s_nodes)

    def _resolve_node_to_surface(self, resolver, defn, node_map, face_map, all_nodes):
        import gmsh
        # master_label stores a geometry entity tag (dim=0), NOT a mesh
        # node tag.  Query Gmsh for the mesh node on that entity.
        master_entity_tag = int(defn.master_label)
        try:
            nt, _, _ = gmsh.model.mesh.getNodes(
                dim=0, tag=master_entity_tag,
                includeBoundary=False, returnParametricCoord=False)
            if len(nt) == 0:
                raise ValueError(
                    f"node_to_surface: geometry point entity "
                    f"(dim=0, tag={master_entity_tag}) has no mesh node.")
            master_node = int(nt[0])
        except Exception as exc:
            raise ValueError(
                f"node_to_surface: cannot get mesh node from point "
                f"entity (dim=0, tag={master_entity_tag}): {exc}") from exc

        if master_node not in all_nodes:
            raise ValueError(
                f"node_to_surface: master mesh node {master_node} "
                f"(from entity {master_entity_tag}) not found in "
                f"the mesh node set.")

        # slave_label is a comma-separated list of surface entity tags.
        # Union the slave nodes across all surfaces — shared-edge
        # nodes are naturally deduplicated by the set.
        slave_entity_tags = [
            int(s) for s in defn.slave_label.split(",") if s]
        slave_nodes: set[int] = set()
        for s_tag in slave_entity_tags:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=2, tag=s_tag,
                    includeBoundary=True, returnParametricCoord=False)
                slave_nodes.update(int(t) for t in nt)
            except Exception as exc:
                raise ValueError(
                    f"node_to_surface: cannot get nodes from surface "
                    f"entity (dim=2, tag={s_tag}): {exc}") from exc
        if not slave_nodes:
            raise ValueError(
                f"node_to_surface: surface entities "
                f"{slave_entity_tags} have no nodes.")
        # Dispatch to the constraint- or spring-variant resolver via
        # the _RESOLVER_METHOD table so both def subclasses share this
        # lookup code.
        resolver_fn = getattr(
            resolver, _RESOLVER_METHOD[type(defn)])
        return resolver_fn(defn, master_node, slave_nodes)

    def _resolve_embedded(self, resolver, defn, node_map, face_map, all_nodes):
        """Resolve an embedded constraint.

        Host elements are the tet4 (dim=3) or tri3 (dim=2) elements
        belonging to ``defn.master_label``. Embedded nodes come from
        the ``defn.slave_label`` part (all nodes of that part's
        entities, typically the mesh nodes along a rebar curve).
        """
        import gmsh

        host_entities = (
            defn.host_entities if defn.host_entities
            else self._entities_for_label(defn.master_label)
        )
        embedded_entities = (
            defn.embedded_entities if defn.embedded_entities
            else self._entities_for_label(defn.slave_label)
        )

        host_elems = self._collect_host_elems(host_entities)
        if host_elems.size == 0:
            import warnings
            warnings.warn(
                f"embedded: host label '{defn.master_label}' has no "
                f"tet4 or tri3 elements; constraint skipped.",
                stacklevel=2,
            )
            return []

        embedded_nodes: set[int] = set()
        for dim, tag in embedded_entities:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(dim), tag=int(tag),
                    includeBoundary=True, returnParametricCoord=False)
                embedded_nodes.update(int(t) for t in nt)
            except Exception:
                pass
        if not embedded_nodes:
            import warnings
            warnings.warn(
                f"embedded: embedded label '{defn.slave_label}' has no "
                f"mesh nodes; constraint skipped.",
                stacklevel=2,
            )
            return []

        # Don't embed a node that coincides with a host corner — it's
        # already rigidly attached via shared connectivity.
        host_corner_nodes = set(int(t) for t in np.unique(host_elems))
        embedded_nodes = embedded_nodes - host_corner_nodes

        return resolver.resolve_embedded(defn, host_elems, embedded_nodes)

    def _entities_for_label(self, label: str) -> list[tuple[int, int]]:
        """Look up geometric entities for *label*.

        Tries, in order: a part instance in ``g.parts``, then a
        physical group by name. Returns ``[]`` if neither matches.
        """
        import gmsh
        parts = getattr(self._parent, "parts", None)
        if parts is not None and label in getattr(parts, "_instances", {}):
            inst = parts._instances[label]
            return [
                (int(dim), int(tag))
                for dim, tags in inst.entities.items()
                for tag in tags
            ]
        # Physical-group fallback.
        ents: list[tuple[int, int]] = []
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(int(d), int(pg_tag))
            except Exception:
                continue
            if name != label:
                continue
            for ent in gmsh.model.getEntitiesForPhysicalGroup(
                    int(d), int(pg_tag)):
                ents.append((int(d), int(ent)))
        return ents

    @staticmethod
    def _collect_host_elems(
        entities: list[tuple[int, int]],
    ) -> np.ndarray:
        """Gather tet4 / tri3 connectivity rows from *entities*.

        Gmsh element type codes: 2 = tri3, 4 = tet4. Other host types
        (hex, quad, higher-order) are not supported by
        ``ASDEmbeddedNodeElement`` and are silently ignored.
        """
        import gmsh
        tri_rows: list[np.ndarray] = []
        tet_rows: list[np.ndarray] = []
        for dim, tag in entities:
            try:
                etypes, _, enodes = gmsh.model.mesh.getElements(
                    dim=int(dim), tag=int(tag))
            except Exception:
                continue
            for etype, nodes in zip(etypes, enodes):
                if len(nodes) == 0:
                    continue
                if int(etype) == 2:
                    tri_rows.append(
                        np.asarray(nodes, dtype=int).reshape(-1, 3))
                elif int(etype) == 4:
                    tet_rows.append(
                        np.asarray(nodes, dtype=int).reshape(-1, 4))
        # Prefer 3D host if present; otherwise fall back to 2D tris.
        if tet_rows:
            return np.vstack(tet_rows)
        if tri_rows:
            return np.vstack(tri_rows)
        return np.empty((0, 0), dtype=int)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_defs(self) -> list[dict]:
        return [
            {"kind": d.kind, "master": d.master_label,
             "slave": d.slave_label, "name": d.name}
            for d in self.constraint_defs]

    def summary(self):
        """DataFrame of the declared constraint intent — one row per def.

        Columns: ``kind, name, master, slave, params``.  ``params`` is a
        short stringified view of the kind-specific fields (``dofs``,
        ``tolerance``, etc.).
        """
        import pandas as pd
        from dataclasses import fields

        _COMMON = {"kind", "name", "master_label", "slave_label"}

        rows: list[dict] = []
        for d in self.constraint_defs:
            params = {
                f.name: getattr(d, f.name)
                for f in fields(d)
                if f.name not in _COMMON
            }
            params = {k: v for k, v in params.items() if v is not None}
            rows.append({
                "kind"  : d.kind,
                "name"  : d.name or "",
                "master": d.master_label,
                "slave" : d.slave_label,
                "params": ", ".join(f"{k}={v}" for k, v in params.items()),
            })

        cols = ["kind", "name", "master", "slave", "params"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    def list_records(self) -> list[dict]:
        out = []
        for r in self.constraint_records:
            d: dict[str, Any] = {"kind": r.kind, "name": r.name}
            if hasattr(r, "master_node"):
                d["master_node"] = r.master_node
            if hasattr(r, "slave_node"):
                d["slave_node"] = r.slave_node
            if hasattr(r, "slave_nodes"):
                d["n_slaves"] = len(r.slave_nodes)
            out.append(d)
        return out

    def clear(self) -> None:
        self.constraint_defs.clear()
        self.constraint_records.clear()

    def __repr__(self) -> str:
        return (
            f"<ConstraintsComposite {len(self.constraint_defs)} defs, "
            f"{len(self.constraint_records)} records>")
