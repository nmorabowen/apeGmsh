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

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.core.constraints.defs import (
    BCDef,
    ConstraintDef,
    DistributingCouplingDef,
    EmbeddedDef,
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
from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver
from apeGmsh._kernel.record_sets import NodeConstraintSet as ConstraintSet
from apeGmsh._kernel.records._constraints import ConstraintRecord

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

# Decomposition of a hex8 into 6 right-handed (positive-volume) Kuhn
# tets that share the main diagonal (vertex 0 → vertex 6).  Indices
# refer to Gmsh hex8 vertex ordering (bottom face [0,1,2,3] CCW from
# +z, top face [4,5,6,7] aligned above).  Used by
# ``_collect_host_subelements`` to virtualise hex8 / hex20 hosts into
# tet sub-elements that ``ASDEmbeddedNodeElement`` accepts; the table
# does NOT correspond to elements in the gmsh mesh — it is a coupling
# decomposition only.  See ``test_embedded_decomposition.py`` for the
# positive-volume + Gmsh-ordering invariant guards.
HEX8_TO_6_TETS = np.array([
    [0, 1, 2, 6],
    [0, 2, 3, 6],
    [0, 3, 7, 6],
    [0, 7, 4, 6],
    [0, 4, 5, 6],
    [0, 5, 1, 6],
], dtype=int)

# Decomposition of a prism6 (3-node triangle bottom + top, with
# vertical edges) into 3 right-handed Kuhn-style tets sharing the
# diagonal (vertex 0 → vertex 5).  Indices refer to Gmsh prism6
# vertex ordering (bottom triangle [0, 1, 2], top triangle [3, 4, 5]
# aligned above).  Each tet is positive-volume in the canonical Gmsh
# reference coordinates; see ``test_embedded_decomposition.py``.
PRISM6_TO_3_TETS = np.array([
    [0, 1, 2, 5],
    [0, 1, 5, 4],
    [0, 4, 5, 3],
], dtype=int)

# Decomposition of a pyramid5 (square base + apex) into 2 tets
# sharing the (0, 2) diagonal of the base, both sharing the apex
# (vertex 4).  Indices refer to Gmsh pyramid5 vertex ordering
# (base [0, 1, 2, 3] CCW from +z, apex node 4).  Each tet is
# positive-volume in the canonical Gmsh reference coordinates.
PYRAMID5_TO_2_TETS = np.array([
    [0, 1, 2, 4],
    [0, 2, 3, 4],
], dtype=int)

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
        # Single-point constraints (BCDef) are kept apart from
        # constraint_defs: they carry no master/slave, are not in
        # _DISPATCH, and resolve to fem.nodes.sp (not
        # fem.nodes.constraints).  Mixing them into constraint_defs
        # would break _add_def validation and the resolve() dispatch.
        self._bc_defs: list[BCDef] = []

    def _add_def(self, defn: _ConstraintT) -> _ConstraintT:
        """Validate dispatch + labels and store a constraint definition.

        A def whose type has no ``_DISPATCH`` entry cannot be resolved
        and would be silently dropped at :meth:`resolve` — reject it at
        declaration instead, mirroring
        ``LoadsComposite``/``MassesComposite._add_def``.  Label
        validation is skipped for NodeToSurfaceDef / EmbeddedDef
        (bare tags)."""
        if type(defn) not in _DISPATCH:
            raise TypeError(
                f"{type(defn).__name__} has no _DISPATCH entry — it "
                f"cannot be resolved and would be silently dropped. "
                f"Register it in _DISPATCH (and _RESOLVER_METHOD) "
                f"before use."
            )
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

    # ── Tier 0 — Single-point (fix to ground) ────────────────────────
    def bc(self, target=None, *, pg=None, label=None, tag=None,
           dofs=None, name=None) -> BCDef:
        """Homogeneous single-point constraint — fix a pattern to ground.

        The natural (essential / Dirichlet) boundary condition: every
        mesh node in the resolved pattern gets ``ops.fix(node, *mask)``
        downstream. There is **no master and no slave** — unlike every
        other method on this composite, this is a constraint *to
        ground*, not between two parts. It resolves into
        ``fem.nodes.sp`` (homogeneous :class:`SPRecord`\\ s) — the same
        broker channel as :meth:`g.loads.face_sp` — **not**
        ``fem.nodes.constraints``.

        Because it is a *permanent* constraint (not a pattern-scoped
        quantity), it lives here on ``g.constraints`` rather than on
        ``g.loads``: there is no load-pattern context to accidentally
        scope it into, and the downstream emitter places it in the
        ``model → bcs → patterns`` deck order via ``ops.fix``.

        Parameters
        ----------
        target : str or list[(dim, tag)]
            Pattern to fix. Resolved label → physical group → raw
            tags (or a mesh selection) — the same flexible target
            model as :meth:`g.loads.face_sp`. Pass ``pg=`` / ``label=``
            / ``tag=`` instead to force a specific resolution path.
        dofs : list[int], optional
            Restraint **mask** (``1`` = constrained, ``0`` = free), in
            DOF order ``[ux, uy, uz, rx, ry, rz]``. Default
            ``[1, 1, 1]`` (pin all translations). This is the
            OpenSees ``ops.fix`` / ``face_sp`` convention — **not**
            the index-list convention used by
            :meth:`equal_dof` (``dofs=[1,2,3]``).
        name : str, optional
            Friendly name shown in summaries / the viewer.

        Returns
        -------
        BCDef
            The stored definition; the same object is appended to
            ``self._bc_defs``.

        Warnings
        --------
        Resolution is dimension-agnostic — a point, edge, surface, or
        volume pattern all just contribute their mesh nodes. Pointing
        a BC at a **volume** physical group therefore fixes *every
        interior node of the solid*, which is almost never intended;
        target a boundary surface/edge instead.

        Examples
        --------
        ::

            g.constraints.bc("base_face")                  # pin x,y,z
            g.constraints.bc(pg="Supports", dofs=[1, 1, 0])
            g.constraints.bc(label="col.base",
                             dofs=[1, 1, 1, 1, 1, 1])       # full fixity
        """
        t, src = self._parent.loads._coalesce_target(
            target, pg=pg, label=label, tag=tag)
        defn = BCDef(target=t, target_source=src,
                     dofs=list(dofs) if dofs is not None else [1, 1, 1],
                     name=name)
        self._bc_defs.append(defn)
        return defn

    def resolve_bcs(self, node_tags, *, node_map=None) -> list:
        """Resolve every :meth:`bc` def to homogeneous ``SPRecord``\\ s.

        Mirrors the load/SP resolution path: each ``BCDef`` target is
        run through the loads composite's dimension-agnostic
        ``_target_nodes`` (label → PG → tag → mesh-selection, any
        dim), then one ``SPRecord(value=0.0, is_homogeneous=True)`` is
        emitted per restrained DOF per node.

        Fails loud — consistent with :meth:`_resolve_nodes` and the
        resolver contract — if a pattern resolves to zero mesh nodes:
        a BC that silently binds nothing is worse than one that errors.
        """
        from apeGmsh._kernel.records._loads import SPRecord

        if not self._bc_defs:
            return []

        loads = self._parent.loads
        all_nodes = {int(t) for t in node_tags}
        out: list = []
        for defn in self._bc_defs:
            nodes = loads._target_nodes(
                defn.target, node_map or {}, all_nodes,
                source=defn.target_source, expected_dim=None,
            )
            if not nodes:
                raise ValueError(
                    f"bc: target {defn.target!r} (source="
                    f"{defn.target_source!r}) resolved to zero mesh "
                    f"nodes — check it is meshed and names a real "
                    f"label / physical group / entity. Refusing to "
                    f"emit an empty boundary condition.")
            for nid in sorted(nodes):
                for d_idx, mask in enumerate(defn.dofs):
                    if mask != 1:
                        continue
                    out.append(SPRecord(
                        name=defn.name,
                        node_id=int(nid),
                        dof=d_idx + 1,
                        value=0.0,
                        is_homogeneous=True,
                    ))
        return out

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
            stiffness=1.0e18, stiffness_p=None,
            rotational=False, pressure=False,
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
        :class:`~apeGmsh.mesh.records.InterpolationRecord` per
        successfully projected slave node. Downstream the apeGmsh
        OpenSees bridge emits these as ``ASDEmbeddedNodeElement``
        penalty elements (default K = 1e18; tunable on the bridge
        ingest API).

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
            dofs=dofs, tolerance=tolerance, name=name,
            stiffness=stiffness, stiffness_p=stiffness_p,
            rotational=rotational, pressure=pressure))

    def distributing_coupling(self, master_label, slave_label, *,
                              master_point=(0., 0., 0.), dofs=None,
                              weighting="uniform",
                              name=None) -> DistributingCouplingDef:
        """**Not implemented — raises ``NotImplementedError``.**

        A true distributing coupling (RBE3) distributes a master
        force/moment to the slave nodes so that ``Σ Fᵢ = F`` **and**
        ``Σ rᵢ × Fᵢ = M`` while leaving the surface free to deform —
        a *force* relation, not a kinematic one.

        The previous implementation did **not** do this: it emitted a
        *kinematic* mean constraint (``u_ref = Σ wᵢ u_surfᵢ``) and its
        ``weighting="area"`` option was inverse-distance-from-centroid
        (physically meaningless — the opposite of tributary area),
        with no moment-equilibrium term.  It silently produced a
        mechanically wrong model under a correct-looking API, so it is
        refused rather than shipped.

        Use instead, depending on intent:

        * :meth:`kinematic_coupling` — a DOF-selective rigid coupling
          of the surface to a reference node.
        * :meth:`tie` — shape-function interpolation onto a master
          face (compatible, non-rigid).
        * a distributed nodal load (``g.loads``) — to introduce a
          statically-equivalent load without any kinematic tie.

        Raises
        ------
        NotImplementedError
            Always.  Parameters are accepted only so the message is
            actionable.
        """
        raise NotImplementedError(
            "distributing_coupling (RBE3 force distribution) is not "
            "implemented.  The prior implementation silently emitted a "
            "kinematic mean with a physically-meaningless 'area' "
            "weighting and no moment equilibrium.  Use "
            "kinematic_coupling (DOF-selective rigid coupling), tie "
            "(shape-function interpolation), or a distributed nodal "
            "load instead."
        )

    def embedded(self, host_label, embedded_label, *, tolerance=1.0,
                 host_entities=None, embedded_entities=None,
                 stiffness=1.0e18, stiffness_p=None,
                 rotational=False, pressure=False,
                 host_coupling="linear",
                 name=None) -> EmbeddedDef:
        """Embed lower-dimensional elements inside a host volume or
        surface.

        Each node of the embedded part is constrained to the
        displacement field of the host element it falls inside via
        host shape functions. Used for **rebar in concrete**,
        stiffeners in shells, fibres in composite hosts, etc.

        Supported host element types:

        * **3-D host:** tet4 (etype 4), tet10 (11), hex8 (5),
          hex20 (17), prism6 (6), prism15 (18), pyramid5 (7),
          pyramid13 (14).
        * **2-D host:** tri3 / CST (etype 2), tri6 / LST (9),
          quad4 (3), quad8 (16), quad9 (10).

        Non-simplex and higher-order hosts are decomposed to
        linear sub-tris / sub-tets using corner nodes only (hex8
        → 6 Kuhn tets; prism6 → 3 tets; pyramid5 → 2 tets;
        quad4 → 2 tris on the (0,2) diagonal; tri6 / tet10 /
        hex20 / quad8 / quad9 / prism15 / pyramid13 →
        corner-only).  The embedded coupling is therefore
        **linear** regardless of the host's native interpolation
        order — see ``host_coupling`` and :class:`EmbeddedDef`
        for the full contract.  A ``UserWarning`` fires once per
        (host type, entity) the first time a midside-bearing host
        is decomposed.

        The resolver automatically drops embedded nodes that
        coincide with host element corners, since those are
        already rigidly attached through shared connectivity.

        Parameters
        ----------
        host_label : str
            Part label whose host elements form the embedding
            field.  Stored internally as ``master_label``.
        embedded_label : str
            Part label whose nodes are embedded. Stored as
            ``slave_label``. (Label validation is bypassed for
            ``EmbeddedDef`` — these labels may also be physical
            group names if no part registry is in use.)
        tolerance : float, default 1.0
            Maximum dimensionless barycentric excess allowed when
            locating an embedded node inside a host sub-element.
            ``0.0`` means strictly inside; the default ``1.0``
            preserves pre-Phase-2 permissive behaviour.  See
            :class:`EmbeddedDef` for the fail-loud gate.
        host_entities, embedded_entities : list of (dim, tag), optional
            Restrict the host / embedded sides to specific Gmsh
            entities.  When omitted the whole label is used.
        host_coupling : {"linear"}, default ``"linear"``
            Reserved keyword pinning the coupling kinematics.  Only
            ``"linear"`` is currently accepted (coupling to 3 or 4
            corner nodes via barycentric shape functions, matching
            ``ASDEmbeddedNodeElement``).  Reserved so that future
            higher-order options (``"trilinear"``, ``"biquadratic"``)
            can be added without breaking old models.
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

        Same rebar embedded into a hex8 mesh (each rebar node is
        located inside one of the 6 Kuhn sub-tets of the
        enclosing hex and coupled to that sub-tet's 4 corners)::

            g.constraints.embedded(
                host_label="concrete_block_hex",
                embedded_label="rebar_curve",
            )
        """
        return self._add_def(EmbeddedDef(
            master_label=host_label, slave_label=embedded_label,
            tolerance=tolerance, host_entities=host_entities,
            embedded_entities=embedded_entities, name=name,
            stiffness=stiffness, stiffness_p=stiffness_p,
            rotational=rotational, pressure=pressure,
            host_coupling=host_coupling))

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
        if len(m_tags) != 1:
            raise ValueError(
                f"node_to_surface master {master!r} resolved to "
                f"{len(m_tags)} dim-0 entities {m_tags} — the master "
                f"must identify exactly one reference point.")
        if not s_tags:
            raise ValueError(
                f"node_to_surface slave {slave!r} resolved to no "
                f"dim-2 surface entities.")
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
        if len(m_tags) != 1:
            raise ValueError(
                f"node_to_surface_spring master {master!r} resolved "
                f"to {len(m_tags)} dim-0 entities {m_tags} — the "
                f"master must identify exactly one reference point.")
        if not s_tags:
            raise ValueError(
                f"node_to_surface_spring slave {slave!r} resolved to "
                f"no dim-2 surface entities.")
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
                     stiffness=1.0e18, stiffness_p=None,
                     rotational=False, pressure=False,
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
            dofs=dofs, tolerance=tolerance, name=name,
            stiffness=stiffness, stiffness_p=stiffness_p,
            rotational=rotational, pressure=pressure))

    def mortar(self, master_label, slave_label, *,
               master_entities=None, slave_entities=None,
               dofs=None, integration_order=2,
               name=None) -> MortarDef:
        """**Not implemented — raises ``NotImplementedError``.**

        A true mortar method introduces a Lagrange-multiplier space
        ``ψᵢ`` on the slave side and integrates the coupling operator
        ``Bᵢⱼ = ∫_Γ ψᵢ·Nⱼ dΓ`` over the overlapping surface segments,
        satisfying the inf-sup (LBB) condition.

        The previous implementation did **none** of that: no segment
        intersection, no surface integral, no dual basis — it scattered
        ``tied_contact`` collocation weights onto a block-diagonal
        ``B`` with a hardcoded ``tolerance=10.0`` (model-unit
        dependent → mis-pairs on millimetre models), yet returned a
        record labelled ``MORTAR`` that downstream could not
        distinguish from a real one.  It is refused rather than
        shipped as a plausible-but-wrong operator.

        Use :meth:`tied_contact` for a collocation-based non-matching
        tie (the honest version of what the old code actually did).

        Raises
        ------
        NotImplementedError
            Always.  Parameters are accepted only so the message is
            actionable.
        """
        raise NotImplementedError(
            "mortar (∫ ψ·N dΓ Lagrange-multiplier coupling) is not "
            "implemented.  The prior implementation was a collocation "
            "tie with a unit-dependent hardcoded tolerance mislabelled "
            "as MORTAR.  Use tied_contact for a non-matching "
            "collocation tie."
        )

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
                raise TypeError(
                    f"No _DISPATCH entry for constraint type "
                    f"{type(defn).__name__}; refusing to silently drop "
                    f"the constraint (defs are normally gated by "
                    f"_add_def)."
                )
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
        """Resolve a constraint role to its mesh-node set — fail loud.

        Explicit ``{role}_entities`` are resolved strictly: a Gmsh
        failure or a zero-node result raises (a stale/wrong-dim tag
        must not be swallowed).  Otherwise the part ``label``'s node
        set is taken from ``node_map``; a missing or empty entry
        raises rather than silently binding every node in the model —
        a constraint must bind a *known* node set.
        """
        import gmsh
        kind = type(defn).__name__
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            tags: set[int] = set()
            for dim, tag in selected:
                try:
                    nt, _, _ = gmsh.model.mesh.getNodes(
                        dim=int(dim), tag=int(tag),
                        includeBoundary=True, returnParametricCoord=False)
                except Exception as exc:
                    raise ValueError(
                        f"{kind} {role}: cannot get mesh nodes for "
                        f"entity (dim={dim}, tag={tag}) from "
                        f"{role}_entities={selected!r}: {exc}") from exc
                tags.update(int(t) for t in nt)
            if not tags:
                raise ValueError(
                    f"{kind} {role}: {role}_entities={selected!r} "
                    f"resolved to zero mesh nodes — check the entities "
                    f"are meshed and of the intended dimension.")
            return tags
        nodes = node_map.get(label)
        if not nodes:
            raise ValueError(
                f"{kind} {role}: part label {label!r} contributed no "
                f"nodes to the constraint node map (is it meshed and "
                f"registered in g.parts?). Refusing to fall back to "
                f"all model nodes — a constraint must bind a known "
                f"node set; pass {role}_entities= to scope explicitly.")
        return nodes

    def _resolve_faces(self, label, role, defn, face_map):
        """Resolve a face-constraint role to its surface connectivity
        — fail loud.

        A face constraint requires *surfaces*: ``{role}_entities`` may
        be dim=2 (surfaces directly) or dim=3 (a volume — its boundary
        surfaces are used).  dim 0/1 is a wrong-dimension reference and
        raises.  An empty/missing result raises rather than letting the
        caller silently drop the constraint with ``return []``.
        """
        kind = type(defn).__name__
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            bad = [(int(d), int(t)) for d, t in selected
                   if int(d) not in (2, 3)]
            if bad:
                raise ValueError(
                    f"{kind} {role}: {role}_entities {selected!r} "
                    f"contains non-surface entities {bad} — a face "
                    f"constraint requires dim=2 surfaces (or dim=3 "
                    f"volumes, whose boundary surfaces are used).")
            parts = getattr(self._parent, "parts", None)
            if parts is None:
                raise ValueError(
                    f"{kind} {role}: {role}_entities was given but "
                    f"g.parts is unavailable to resolve surface faces.")
            faces = parts._collect_surface_faces(selected)
            if faces.size == 0:
                raise ValueError(
                    f"{kind} {role}: {role}_entities={selected!r} "
                    f"produced no surface mesh faces — are the "
                    f"entities meshed?")
            return faces
        faces = face_map.get(label)
        if faces is None or faces.size == 0:
            raise ValueError(
                f"{kind} {role}: part label {label!r} has no surface "
                f"faces in the face map (is its interface meshed and "
                f"registered in g.parts?). Refusing to silently drop "
                f"the constraint; pass {role}_entities= to scope it.")
        return faces

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
        return resolver.resolve_tie(defn, m_faces, s_nodes)

    def _resolve_face_both(self, resolver, defn, node_map, face_map, all_nodes):
        m_faces = self._resolve_faces(defn.master_label, "master", defn, face_map)
        s_faces = self._resolve_faces(defn.slave_label, "slave", defn, face_map)
        m_nodes = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s_nodes = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
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

        host_elems = self._collect_host_subelements(host_entities)
        if host_elems.size == 0:
            raise ValueError(
                f"embedded: host label {defn.master_label!r} resolved "
                f"to entities but none carry embeddable host elements. "
                f"Supported host types: tri3 / tri6 / quad4 / quad8 / "
                f"quad9 (2D); tet4 / tet10 / hex8 / hex20 / prism6 / "
                f"prism15 / pyramid5 / pyramid13 (3D).  Non-simplex "
                f"and higher-order hosts are decomposed to linear "
                f"sub-tris / sub-tets using corner nodes only — the "
                f"coupling is linear regardless of the host's native "
                f"interpolation order.  The constraint cannot be "
                f"built; fix the host mesh rather than skipping it.")

        embedded_nodes: set[int] = set()
        for dim, tag in embedded_entities:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(dim), tag=int(tag),
                    includeBoundary=True, returnParametricCoord=False)
            except Exception as exc:
                raise ValueError(
                    f"embedded: cannot get mesh nodes for embedded "
                    f"entity (dim={dim}, tag={tag}) of label "
                    f"{defn.slave_label!r}: {exc}") from exc
            embedded_nodes.update(int(t) for t in nt)
        if not embedded_nodes:
            raise ValueError(
                f"embedded: embedded label {defn.slave_label!r} "
                f"resolved to entities but they carry no mesh nodes "
                f"(is the embedded geometry meshed?). The constraint "
                f"cannot be built; fix the mesh rather than skipping.")

        # Don't embed a node that coincides with a host corner — it's
        # already rigidly attached via shared connectivity.
        host_corner_nodes = set(int(t) for t in np.unique(host_elems))
        embedded_nodes = embedded_nodes - host_corner_nodes

        return resolver.resolve_embedded(defn, host_elems, embedded_nodes)

    def _entities_for_label(self, label: str) -> list[tuple[int, int]]:
        """Look up geometric entities for *label* — fail loud.

        Tries, in order: a part instance in ``g.parts`` (spans dims,
        as a part legitimately does), then a physical group by name.
        A physical group maps to a single dimension; a name carried
        at several dims raises (multi-dimensional PGs are not
        supported).  Raises ``KeyError`` if the name resolves to
        neither — never returns ``[]`` (a silent empty would surface
        downstream as a misleading "no host elements" skip).
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
        # Physical-group fallback (single-dim by construction).
        ents: list[tuple[int, int]] = []
        pg_dims: set[int] = set()
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(int(d), int(pg_tag))
            except Exception:
                continue
            if name != label:
                continue
            pg_dims.add(int(d))
            for ent in gmsh.model.getEntitiesForPhysicalGroup(
                    int(d), int(pg_tag)):
                ents.append((int(d), int(ent)))
        if len(pg_dims) > 1:
            raise ValueError(
                f"Physical group {label!r} exists at multiple "
                f"dimensions {sorted(pg_dims)}. Multi-dimensional "
                f"physical groups are not supported; assign one "
                f"dimension per group name.")
        if not ents:
            raise KeyError(
                f"Constraint label {label!r} resolved to neither a "
                f"g.parts instance nor a physical group. Register the "
                f"part or create the physical group before resolving.")
        return ents

    @staticmethod
    def _collect_host_subelements(
        entities: list[tuple[int, int]],
    ) -> np.ndarray:  # noqa: C901 — single dispatch, kept flat for clarity
        """Gather virtual tri3 / tet4 sub-element rows from *entities*.

        Returns connectivity rows that ``ASDEmbeddedNodeElement``
        can consume directly: each row is 3 node tags (treated as
        a tri3 host) or 4 node tags (treated as a tet4 host).
        Higher-order and non-simplex host elements are decomposed
        on the fly into sub-tris / sub-tets sharing the host's
        corner nodes:

        ============== ====== =====================================
        Gmsh etype     Code   Decomposition (corner nodes only)
        ============== ====== =====================================
        tri3 (CST)     2      identity (1 tri per host)
        tet4           4      identity (1 tet per host)
        quad4          3      2 tris via (0,2) diagonal split
        hex8           5      6 right-handed Kuhn tets (see
                              :data:`HEX8_TO_6_TETS`)
        prism6         6      3 tets (see :data:`PRISM6_TO_3_TETS`)
        pyramid5       7      2 tets (see
                              :data:`PYRAMID5_TO_2_TETS`)
        tri6 (LST)     9      corners only → 1 tri (midsides
                              discarded)
        tet10          11     corners only → 1 tet
        pyramid13      14     corners only → 2 tets
        quad8 / quad9  16/10  corners only → 2 tris
        hex20          17     corners only → 6 Kuhn tets
        prism15        18     corners only → 3 tets
        ============== ====== =====================================

        Returned rows are **not** real gmsh elements — they are a
        coupling-layer fabrication so that the linear-shape-function
        coupling of ``ASDEmbeddedNodeElement`` works against any
        supported host topology.  The embedded coupling is
        consequently linear regardless of the host's native
        interpolation order; see :class:`EmbeddedDef` for the
        ``host_coupling="linear"`` contract.

        Any other element type (prism6, prism15, pyramid5, line,
        point, …) raises ``ValueError`` naming the offending etype
        and entity — silent drop would let an embedded node in the
        dropped region project onto a distant sub-element
        (extrapolation, wrong physics, see the off-host case in
        ``resolve_embedded``).

        When both tris and tets are produced from the same host
        (mixed shell+solid PG), the 3D side wins and the 2D rows
        are dropped — embedded nodes outside the 3D hull will then
        fail-loud at the resolver's barycentric gate.  Split the
        host PG if you need both.
        """
        import gmsh
        import warnings as _warnings
        # Human-readable Gmsh element type names (subset relevant to
        # embedded hosts) — keyed by the integer etype code.
        _ETYPE_NAMES = {
            1: "line2", 2: "tri3", 3: "quad4", 4: "tet4",
            5: "hex8", 6: "prism6", 7: "pyramid5", 8: "line3",
            9: "tri6", 10: "quad9", 11: "tet10", 15: "point1",
            16: "quad8", 17: "hex20", 18: "prism15",
        }
        # Higher-order etypes whose decomposition discards the host's
        # native interpolation richness (midside / center nodes) and
        # falls back to linear-over-corners coupling.  Tracked so we
        # can emit one ``UserWarning`` per (host type, entity) instead
        # of one per element — keeps the warning informative without
        # spamming users with thousands of identical messages.
        _HIGHER_ORDER_CODES = {9, 10, 11, 14, 16, 17, 18}
        warned_higher_order: set[tuple[int, int, int]] = set()
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
                code = int(etype)
                # One warning per (etype, entity) — fires the first
                # time a higher-order host hits the decomposition
                # path so the user knows the embed coupling is
                # linear regardless of the host's native order.
                if (code in _HIGHER_ORDER_CODES
                        and (code, int(dim), int(tag))
                        not in warned_higher_order):
                    warned_higher_order.add(
                        (code, int(dim), int(tag)))
                    _warnings.warn(
                        f"embedded: host entity (dim={dim}, tag={tag}) "
                        f"carries {_ETYPE_NAMES.get(code, code)} "
                        f"elements — decomposing to corner-node-only "
                        f"linear sub-elements.  The embedded coupling "
                        f"will be linear regardless of the host's "
                        f"native interpolation order; quadratic / "
                        f"bilinear / trilinear host kinematics will "
                        f"NOT be felt by the embedded node.  Set "
                        f"`host_coupling=` explicitly on the "
                        f"`embedded(...)` call to acknowledge.",
                        UserWarning, stacklevel=4,
                    )
                if code == 2:                              # tri3 (CST)
                    tri_rows.append(
                        np.asarray(nodes, dtype=int).reshape(-1, 3))
                elif code == 4:                            # tet4
                    tet_rows.append(
                        np.asarray(nodes, dtype=int).reshape(-1, 4))
                elif code == 3:                            # quad4
                    q = np.asarray(nodes, dtype=int).reshape(-1, 4)
                    tri_rows.append(q[:, [0, 1, 2]])
                    tri_rows.append(q[:, [0, 2, 3]])
                elif code == 5:                            # hex8
                    h = np.asarray(nodes, dtype=int).reshape(-1, 8)
                    for tet_idx in HEX8_TO_6_TETS:
                        tet_rows.append(h[:, tet_idx])
                elif code == 9:                            # tri6 (LST)
                    tri_rows.append(
                        np.asarray(nodes, dtype=int)
                        .reshape(-1, 6)[:, :3])
                elif code == 11:                           # tet10
                    tet_rows.append(
                        np.asarray(nodes, dtype=int)
                        .reshape(-1, 10)[:, :4])
                elif code in (16, 10):                     # quad8 / quad9
                    n_per_elem = {16: 8, 10: 9}[code]
                    q = (np.asarray(nodes, dtype=int)
                         .reshape(-1, n_per_elem)[:, :4])
                    tri_rows.append(q[:, [0, 1, 2]])
                    tri_rows.append(q[:, [0, 2, 3]])
                elif code == 17:                           # hex20
                    h = (np.asarray(nodes, dtype=int)
                         .reshape(-1, 20)[:, :8])
                    for tet_idx in HEX8_TO_6_TETS:
                        tet_rows.append(h[:, tet_idx])
                elif code == 6:                            # prism6
                    p = np.asarray(nodes, dtype=int).reshape(-1, 6)
                    for tet_idx in PRISM6_TO_3_TETS:
                        tet_rows.append(p[:, tet_idx])
                elif code == 18:                           # prism15
                    p = (np.asarray(nodes, dtype=int)
                         .reshape(-1, 15)[:, :6])
                    for tet_idx in PRISM6_TO_3_TETS:
                        tet_rows.append(p[:, tet_idx])
                elif code == 7:                            # pyramid5
                    p = np.asarray(nodes, dtype=int).reshape(-1, 5)
                    for tet_idx in PYRAMID5_TO_2_TETS:
                        tet_rows.append(p[:, tet_idx])
                elif code == 14:                           # pyramid13
                    p = (np.asarray(nodes, dtype=int)
                         .reshape(-1, 13)[:, :5])
                    for tet_idx in PYRAMID5_TO_2_TETS:
                        tet_rows.append(p[:, tet_idx])
                else:
                    name = _ETYPE_NAMES.get(code, f"etype={code}")
                    raise ValueError(
                        f"embedded: host entity (dim={dim}, tag={tag}) "
                        f"carries {name} elements.  ASDEmbeddedNodeElement "
                        f"hosts supported: tri3 / tri6 / quad4 / quad8 / "
                        f"quad9 (2D); tet4 / tet10 / hex8 / hex20 / "
                        f"prism6 / prism15 / pyramid5 / pyramid13 (3D).  "
                        f"Higher-order and non-tet hosts are decomposed "
                        f"to linear sub-elements using corner nodes only "
                        f"— the embedded coupling is linear regardless "
                        f"of the host's native interpolation order.  "
                        f"Got an etype not in the supported set; remesh "
                        f"the host or open an issue."
                    )
        # Mixed-dim host (shell PG + brick PG combined) is fail-loud:
        # an embedded node near the shell/brick interface would couple
        # to either side's corners depending on opaque kNN search
        # outcome — different physics, silently chosen.  Split the
        # host PG into two ``embedded`` constraints if you really
        # need both.
        if tet_rows and tri_rows:
            raise ValueError(
                "embedded: host entities produced BOTH 2D sub-tris "
                "and 3D sub-tets — the linear coupling cannot pick "
                "between them deterministically (kNN would dispatch "
                "an embedded node to either side based on centroid "
                "proximity, which is opaque physics).  Split the "
                "host into separate `g.constraints.embedded(...)` "
                "calls — one for the 2D part, one for the 3D part — "
                "or restrict the host PG to a single dimensionality."
            )
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
