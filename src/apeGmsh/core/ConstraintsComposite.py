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
    EqualDOFMixedDef,
    KinematicCouplingDef,
    NodeToSurfaceDef,
    NodeToSurfaceSpringDef,
    PenaltyDef,
    RigidBodyDef,
    RigidDiaphragmDef,
    RigidLinkDef,
    TieDef,
    TiedContactDef,
)
from apeGmsh._kernel.defs.constraints import (
    ContactDef, ContactPlaneDef, InterfaceDef,
)
from apeGmsh._kernel._coupling_control import CouplingControl
from apeGmsh._kernel.resolvers._constraint_resolver import ConstraintResolver
from apeGmsh._kernel.record_sets import NodeConstraintSet as ConstraintSet
from apeGmsh._kernel.records._constraints import (
    ConstraintRecord, ContactRecord, ContactPlaneRecord, InterfaceRecord,
)

_DISPATCH: dict[type, str] = {
    EqualDOFDef:             "_resolve_node_pair",
    EqualDOFMixedDef:        "_resolve_node_pair",
    RigidLinkDef:            "_resolve_node_pair",
    PenaltyDef:              "_resolve_node_pair",
    RigidDiaphragmDef:       "_resolve_diaphragm",
    RigidBodyDef:            "_resolve_kinematic",
    KinematicCouplingDef:    "_resolve_kinematic",
    TieDef:                  "_resolve_face_slave",
    DistributingCouplingDef: "_resolve_distributing",
    EmbeddedDef:             "_resolve_embedded",
    # Both the constraint- and spring-based variants go through the
    # same composite-level lookup; the final call to the resolver is
    # dispatched off ``_RESOLVER_METHOD`` at runtime so the lookup
    # code is not duplicated.
    NodeToSurfaceDef:        "_resolve_node_to_surface",
    NodeToSurfaceSpringDef:  "_resolve_node_to_surface",
    TiedContactDef:          "_resolve_face_both",
}

_RESOLVER_METHOD: dict[type, str] = {
    EqualDOFDef:             "resolve_equal_dof",
    EqualDOFMixedDef:        "resolve_equal_dof_mixed",
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
}

_FACE_TYPES = (TieDef, TiedContactDef)

# ADR 0049 OQ2 slice — only RBE2 / RBE3 accept a g.decouple_node label as a
# constraint role. Other kinds (equal_dof, rigid_link, rigid_diaphragm, …)
# need their own semantics and refuse the role at declare.
_DECOUPLED_ROLE_TYPES = (KinematicCouplingDef, DistributingCouplingDef)


def _as_role_label(ref, role: str) -> str:
    """Normalize a ``g.decouple_node`` handle to its label string.

    Constraint defs store roles as label strings (not session handles).
    A handle without ``label=`` fails loud — there is nothing to look up.
    """
    from apeGmsh._kernel.defs.decoupled import DecoupledNodeDef

    if isinstance(ref, DecoupledNodeDef):
        if not ref.label:
            raise ValueError(
                f"{role}: this g.decouple_node handle has no label= — a "
                f"constraint role is stored on the def as a label string. "
                f"Declare it as g.decouple_node(coords=..., label='work_pt')."
            )
        return ref.label
    return ref


# Surface-facet full node count → corner-node count. The fork contact
# subsystem (contactSurface -master/-slave-segments) only understands
# 3-node (tri) / 4-node (quad) facets; a higher-order surface mesh is
# dropped to its corner facet (gmsh orders corner nodes first, so the
# corner facet is the leading columns of the flat connectivity). Mirrors
# the g.embed / g.reinforce host corner-drop (_GMSH_HOST_KIND).
_SURFACE_CORNER_NPS: dict[int, int] = {3: 3, 6: 3, 4: 4, 8: 4, 9: 4}


def _drop_to_corner_facets(faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Reduce a (n_facets, full_npe) surface-connectivity array to its
    corner facets, returning ``(corner_faces, corner_nps)``.

    Raises on an unsupported facet width (e.g. mixed tri/quad collapsed by
    :meth:`PartsRegistry._collect_surface_faces`, or an exotic element) so a
    misread surface fails loud instead of emitting a wrong ``nps`` stride.
    """
    full = int(faces.shape[1])
    nps = _SURFACE_CORNER_NPS.get(full)
    if nps is None:
        raise ValueError(
            f"contact: surface mesh has {full} nodes per facet, which is not a "
            f"supported tri (3/6) or quad (4/8/9) facet. The fork contact "
            f"subsystem only handles 3-node / 4-node facets; mixed tri/quad "
            f"surfaces are not yet supported.")
    return faces[:, :nps], nps


# Line-segment full node count → corner-node count, for the fork's 2D
# contact lane (`-master 2` / `-slave-segments 2`, a 2-node segment being
# the 2D analogue of the 3D tri/quad facet). A deliberate SIBLING of
# _SURFACE_CORNER_NPS, never a merged table: the two key spaces overlap
# (a width of `3` means tri6 → tri3 there and line3 → line2 here), so one
# shared table would silently remap every 3D tri3 master — the worst
# possible failure, tri3 being the commonest 3D facet. Dispatch is by
# model dimension, which keeps the spaces disjoint by construction rather
# than by a parameter someone can pass wrong.
_LINE_CORNER_NPS: dict[int, int] = {2: 2, 3: 2}


def _drop_to_corner_segments(segs: np.ndarray) -> tuple[np.ndarray, int]:
    """Reduce a (n_segments, full_npe) line-connectivity array to its
    corner segments, returning ``(corner_segments, 2)``.

    ``line3 → line2`` is the exact analogue of the existing ``tri6 →
    tri3`` drop (gmsh orders corner nodes first, so the corner segment is
    the leading columns); refusing it instead would make quadratic 2D
    meshes undeclarable.
    """
    full = int(segs.shape[1])
    nps = _LINE_CORNER_NPS.get(full)
    if nps is None:
        raise ValueError(
            f"contact: the 2D surface is meshed with {full}-node line "
            f"elements, which is not a supported line2 (2) or line3 (3) "
            f"segment. The fork 2D contact lane takes 2-node segments; a "
            f"line3 master drops to its two corner nodes.")
    return segs[:, :nps], nps


def _stack_line_blocks(blocks, label, role) -> np.ndarray:
    """Stack per-entity line-element connectivity into one rectangular
    array, refusing a mixed ``line2``/``line3`` width.

    Mirrors ``PartsRegistry._collect_surface_faces``'s "Mixed surface
    element types not supported" — a mixed width has no single stride, so
    the flat pair list would be misread.
    """
    npe: int | None = None
    rows: list[np.ndarray] = []
    for width, conn in blocks:
        if npe is None:
            npe = int(width)
        elif npe != int(width):
            raise ValueError(
                f"contact: {role} label {label!r} mixes {npe}-node and "
                f"{int(width)}-node line elements across its entities — "
                f"mixed line element types are not supported in 2D contact "
                f"segment extraction. Mesh the whole curve at one element "
                f"order.")
        rows.append(np.asarray(conn, dtype=int).reshape(-1, npe))
    if not rows:
        return np.empty((0, 0), dtype=int)
    return np.vstack(rows)


def _normalise_outward(outward):
    """``None`` / the ``"winding"`` sentinel / a 2-or-3-vector as a tuple.

    The sentinel must survive ``tuple()`` — ``tuple("winding")`` is seven
    one-character strings, a silent corruption rather than an error.
    """
    if outward is None or isinstance(outward, str):
        return outward
    return tuple(outward)


#: The fork's own flush gate, restated: it aborts when the centroid datum
#: is shorter than ``1e-6 * Lref`` with ``Lref`` the master's MIN initial
#: segment length (``LadrunoContactHandler.cpp:482``, ``:1165-1178``).
#: apeGmsh refuses on exactly that condition — a looser threshold would
#: refuse decks the fork accepts, which is a worse failure than the one
#: being prevented.
_FLUSH_CENTROID_FLOOR = 1.0e-6


def _contact_label(defn) -> str:
    """The refusal prefix for one contact def — its name if it has one,
    else its master label (the ``interface{label}`` convention of
    :func:`edge_frames`)."""
    return f" {defn.name!r}" if defn.name else f" {defn.master_label!r}"


def _refuse_flush_without_orientation(
    defn, master_faces, slave_nodes, xyz, *, lref_segments=None,
) -> None:
    """Refuse a FLUSH 2D interface that declares no side.

    The fork's 2D NTS **and mortar** lanes both orient from one
    interface-level centroid vote (slave-surface centroid − master-surface
    centroid) — ``ladruno2DOrientationVote`` is shared by the two call
    sites. On a flush interface — the masonry joint, the footing seated on
    soil, every zero-gap deck this lane exists for — those centroids
    coincide, the vote's magnitude gate fails, and ``handle()`` aborts by
    name rather than guessing. That abort is loud but late: it costs a
    full build and solve launch, and it names a Tcl surface tag rather
    than the two labels the user wrote.

    So refuse here, on the same condition, naming both surfaces and the
    call to add. apeGmsh still derives NO direction — the point of the
    ADR 0073 "never auto-derive a global outward" policy — it only
    refuses earlier and more legibly.

``slave_nodes`` is the tag set the centroid datum is computed from;
    ``lref_segments`` is a SECOND segment block folded into ``Lref``, and
    it is named for that job rather than for the surface it comes from
    because that is all it does here. The mortar lane passes its
    ``-slave-segments`` chain: the fork's 2D mortar site takes the min
    initial segment length over **both** surfaces, since its interval clip
    projects the slave endpoints too, so a slave meshed finer than the
    master lowers the floor. Reproducing that matters in one direction — a
    master-only ``Lref`` would be too LARGE and would refuse decks the fork
    accepts. ``None`` (the NTS lane, whose slave is a bare node set with no
    segments at all) leaves ``Lref`` master-only.

    The remedy the message offers is lane-dependent: ``outward="winding"``
    is the fork's NTS-lane mode, so a mortar interface is offered the
    explicit vector alone (see :class:`ContactDef`).
    """
    rows = np.asarray(master_faces, dtype=np.int64).reshape(-1, 2)
    m_pts = np.asarray(
        [xyz[int(t)][:2] for t in sorted({int(t) for t in rows.ravel()})],
        dtype=float)
    s_pts = np.asarray(
        [xyz[int(t)][:2] for t in sorted({int(t) for t in slave_nodes})],
        dtype=float)
    if m_pts.size == 0 or s_pts.size == 0:
        return
    lref_blocks = [rows]
    if lref_segments is not None:
        lref_blocks.append(
            np.asarray(lref_segments, dtype=np.int64).reshape(-1, 2))
    seg_len = np.linalg.norm(
        np.asarray([xyz[int(b)][:2] - xyz[int(a)][:2]
                    for block in lref_blocks for a, b in block],
                   dtype=float), axis=1)
    l_ref = float(seg_len.min())
    datum = float(np.linalg.norm(s_pts.mean(axis=0) - m_pts.mean(axis=0)))
    if datum > _FLUSH_CENTROID_FLOOR * l_ref:
        return

    which = ("the shortest master segment" if lref_segments is None
             else "the shortest segment of EITHER surface — the 2D mortar "
                  "clip projects the slave endpoints too")
    remedy = (
        f"    g.constraints.contact({defn.master_label!r}, "
        f"{defn.slave_label!r}, ..., outward=(ox, oy))\n"
        f"        the in-plane direction from the master TOWARD the slave")
    if defn.formulation == "nts":
        remedy += (
            f"; or\n"
            f"    g.constraints.contact({defn.master_label!r}, "
            f"{defn.slave_label!r}, ..., outward='winding')\n"
            f"        which declares the side through the master chain's own "
            f"winding (exact per segment, so it also orients curved and "
            f"closed masters) — needs a fork build carrying "
            f"`-outward winding`. ")
    else:
        remedy += (
            f". outward='winding' is NOT available on this lane: the fork "
            f"shipped declared winding on the NTS lane only, because its 2D "
            f"mortar lane has no chain-integrity scan to rest winding's "
            f"one-connected-chain invariant on — so a flush MORTAR "
            f"interface always needs the explicit vector. ")
    raise ValueError(
        f"contact{_contact_label(defn)}: master {defn.master_label!r} and "
        f"slave {defn.slave_label!r} are FLUSH — their reference-configuration "
        f"centroids are separated by {datum:.3g}, below the fork's own "
        f"magnitude floor {_FLUSH_CENTROID_FLOOR:g} x Lref "
        f"({_FLUSH_CENTROID_FLOOR * l_ref:.3g}, Lref = {which}). "
        f"The fork's 2D lane orients from that centroid datum, so "
        f"the outward side is genuinely ambiguous and handle() ABORTS "
        f"instead of guessing — apeGmsh never auto-derives it either. "
        f"Declare the side:\n"
        f"{remedy}"
        f"A flush interface is the normal 2D case, not an edge case.")


def _refuse_contact_entity_dim(
    entities, label, *, model_dim, role, verb="contact", formulation="nts",
) -> None:
    """The contact dimension gate — the single branch point of the 2D lane.

    In a 2D model a contact master is the meshed interface **curve** (a
    dim-1 PG); in a 3D model it is a dim-2 face PG (or a dim-3 volume,
    whose boundary faces are used). Without this gate a 2D model that
    names its dim-2 plane PG — the natural mistake — hands the continuum's
    tri3/quad4 elements to ``_collect_surface_faces``, which returns them
    as "facets"; ``_drop_to_corner_facets`` reports nps=3/4 and a
    structurally valid ``ContactRecord`` gets built out of SOLID elements,
    uncaught all the way to emit.

    The 3D **slave** is deliberately NOT gated: ``_collect_node_set`` is
    dimension-agnostic there today and retrofitting it would move the 3D
    battery.

    ``verb`` names the caller in the refusal (the ``edge_frames`` idiom).
    ``g.constraints.contact_plane`` reuses the gate with ``role="slave"``
    only — a rigid plane has no master mesh — so its 2D slave gets the same
    dim ≤ 1 rule and its 3D behaviour is untouched.

    ``formulation`` narrows the 2D **slave** rule, which is the one place
    the two lanes genuinely differ: an NTS slave is a node set, so a dim-0
    point group is legitimate, while a mortar slave is
    ``-slave-segments 2`` and needs SEGMENTS to chain. It is threaded here
    rather than checked separately at the call site so the 2D dimension
    rules stay in one function with one message family.
    """
    dims = sorted({int(d) for d, _ in entities})
    if model_dim == 2:
        allowed = ((1,) if role == "master" or formulation == "mortar"
                   else (0, 1))
        bad = [d for d in dims if d not in allowed]
        if not bad:
            return
        if role == "master":
            raise ValueError(
                f"{verb}: the model is 2D and master label {label!r} "
                f"resolves to entities of dimension {bad} — a 2D contact "
                f"master must be the meshed interface CURVE (a dim-1 "
                f"physical group), not the plane it bounds. Naming the "
                f"dim-2 plane collects the continuum's own tri/quad "
                f"elements as contact facets, which builds a contact out "
                f"of SOLID elements. Declare the physical group on the "
                f"boundary curve.")
        shape = ("the meshed interface CURVE (dim-1) — a 2D MORTAR slave "
                 "is `-slave-segments 2`, a chained segment surface, and a "
                 "point set has no segments to chain (dim-0 is the NTS "
                 "lane's slave shape, not this one)"
                 if formulation == "mortar" else
                 "the meshed interface CURVE (dim-1) or a point set (dim-0)")
        raise ValueError(
            f"{verb}: the model is 2D and slave label {label!r} resolves "
            f"to entities of dimension {bad} — a 2D contact slave must be "
            f"{shape}. "
            f"A dim-2 slave physical group collects every INTERIOR node of "
            f"the body, not the interface.")
    if role != "master":
        return
    bad = [d for d in dims if d not in (2, 3)]
    if bad:
        raise ValueError(
            f"{verb}: the model is {model_dim}D and master label "
            f"{label!r} resolves to entities of dimension {bad} — a 3D "
            f"contact master must be a dim-2 face physical group (or a "
            f"dim-3 volume, whose boundary faces are used). A dim-1 curve "
            f"has no surface facets at all; the 2D segment lane is "
            f"reachable only in a 2D model.")


def _refuse_contact_plane_out_of_plane(defn) -> None:
    """Refuse a rigid plane whose ``normal``/``point`` leaves the 2D plane.

    apeGmsh emits the zero-padded 9-argument ``contactPlane`` in both
    dimensions, and on a 2D slave surface the fork *refuses* a non-zero
    nz/pz rather than dropping it — there is no out-of-plane coordinate for
    them to act on. So this is not a silent-wrong, it is a loud abort at
    parse; refusing here only moves it earlier and names the declaration and
    the component instead of a Tcl surface tag.
    """
    for nm, v in (("normal", defn.normal), ("point", defn.point)):
        z = float(tuple(v)[2])
        if z == 0.0:
            continue
        raise ValueError(
            f"contact_plane: rigid plane "
            f"{defn.name or defn.slave_label!r} declares {nm}={tuple(v)!r} "
            f"with a non-zero z, but the model is 2D — a 2D plane lies in "
            f"the modelling plane. The fork refuses a non-zero nz/pz on a 2D "
            f"slave surface (there is no third coordinate for it to act on), "
            f"so z has nowhere to go. Pass {nm}=({tuple(v)[0]}, "
            f"{tuple(v)[1]}).")


# Kuhn-decomposition tables — re-exported aliases for backward compat.
#
# The canonical definitions live in
# :mod:`apeGmsh._kernel.geometry._host_decomposition` per ADR 0041
# §"Decision 1 — lift Kuhn decomposition into a geometric utility
# module".  Tests and external consumers that previously imported
# these names from ``apeGmsh.core.ConstraintsComposite`` continue to
# work unchanged via these aliases (ADR 0041 §"Decision 7 — build-
# phase rename ripple": audit found ``tests/test_embedded_decomposition.py``
# imports the names; preserve the public surface).
from apeGmsh._kernel.geometry._host_decomposition import (
    HEX8_TO_6_TETS,
    PRISM6_TO_3_TETS,
    PYRAMID5_TO_2_TETS,
    decompose_hosts_to_subelements as _decompose_hosts_to_subelements,
)

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
    4 — Contact   :meth:`tied_contact`                                  ``SurfaceCouplingRecord``
    5 — Fork      :meth:`contact`, :meth:`mortar` (deprecated alias)    ``ContactRecord``
    ============= ===================================================== =================================

    All constraints ultimately express the linear MPC equation
    ``u_slave = C · u_master``. Tiers differ in **how** ``C`` is
    built — by node co-location (Tier 1), kinematic transformation
    around a master point (Tier 2), shape-function interpolation
    (Tier 3), or numerical integration on the interface (Tier 4).

    Target identification
    ---------------------
    Most methods identify their master and slave sides by name — a
    **part label** (a key of ``g.parts._instances``), a **physical
    group** (``g.physical`` / ``.to_physical``), or a **label**
    (``g.labels``).  :meth:`_add_def` validates both names and raises
    ``KeyError`` on a typo::

        g.constraints.tie(master_label="column",
                          slave_label="slab",
                          master_entities=[(2, 13)],   # optional scope
                          slave_entities=[(2, 17)])

    A physical-group model therefore constrains without building
    Parts (``tie("A_top", "B_bot")`` just works), matching how
    ``g.loads`` / ``g.masses`` already resolve names.  **Precedence:**
    a Part registered under the name wins (the part node/face map is
    consulted first); otherwise the name resolves through the shared
    label→PG→part geometry resolver.  A name that is simultaneously a
    Part and a physical group binds the Part's node set.

    Optional ``master_entities`` / ``slave_entities`` (list of
    ``(dim, tag)``) narrow the search to a subset of the target's
    entities — useful when a target has many surfaces and only one is
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
        # Contact interactions (ContactDef) are kept apart from
        # constraint_defs: they are NOT in the interpolation _DISPATCH
        # (they emit contactSurface/contact commands + the LadrunoContact
        # handler, not embeddedNode/element records) and resolve to an
        # additive fem.elements.contacts list — mirroring g.reinforce /
        # g.embed rather than the MP-constraint dispatch.
        self.contact_defs: list[ContactDef] = []
        self.contact_records: list[ContactRecord] = []
        self.contact_plane_defs: list[ContactPlaneDef] = []
        self.contact_plane_records: list[ContactPlaneRecord] = []
        # Oriented coincident-pair zeroLength interfaces (ADR 0093).
        # Another additive side-list, like the contacts above: they
        # resolve to fem.elements.interfaces, not the MP dispatch.
        self.interface_defs: list[InterfaceDef] = []
        self.interface_records: list[InterfaceRecord] = []
        # Shared phantom-node-tag high-water mark. The MP lane's
        # ConstraintResolver mints node_to_surface phantoms from
        # max(node_tags)+1 (``_next_phantom_tag``); the interface
        # resolver mints its own from a different code path. A model
        # carrying both would collide unless one counter crosses
        # between them — resolve() publishes the MP lane's final value
        # here and resolve_interfaces() starts above it.
        self._phantom_tag_high_water: int | None = None

    def contact(
        self, master, slave, *,
        formulation="nts",
        kn=None, kt=None, mu=None,
        eps_n=None, eps_t=None,
        cohesion=None, tau_max=None,
        aug_tol=None, max_aug=None, ngp=None,
        tie=False, thickness=None, outward=None,
        soft=None, visc=None, consistent_tan=False, geom_tan=False,
        cell=None,
        edge_edge=False, edge_kn=None, edge_band=None,
        edge_mu=None, edge_kt=None, edge_cohesion=None, edge_tau_max=None,
        edge_consistent_tan=False, edge_soft=None, edge_alm=False,
        edge_aug_tol=None,
        master_entities=None, slave_entities=None,
        name=None,
    ) -> ContactDef:
        """Declare a face-to-face contact between two meshed surfaces
        (fork `contactSurface` + `contact` + `LadrunoContact` handler).

        Parameters
        ----------
        master, slave : str
            The two surface PG / part labels in contact. The master is
            faceted (`-master`); the slave is a node set (NTS, `-slave`) or
            faceted (mortar, `-slave-segments`).
        formulation : {"nts", "mortar"}
            ``"nts"`` = node-to-segment penalty; ``"mortar"`` =
            segment-to-segment ALM (the non-matching-mesh accuracy lane).
        kn, kt, mu : float, optional
            NTS normal/tangential penalty + Coulomb friction (``kn`` may be
            ``"auto"``). Rejected for mortar.
        eps_n, eps_t : float | "auto", optional
            Mortar ALM normal/tangential penalty. Rejected for NTS.
        cohesion, tau_max : float, optional
            Mortar friction-cone adhesion + Tresca cap.
        aug_tol, max_aug, ngp : optional
            Mortar Uzawa tolerance / max augmentations / slave-facet Gauss order.
        tie : bool
            Permanent mesh-tie bond (mortar only; excludes friction).
        thickness : float, optional
            **2D mortar only** — the plane-model out-of-plane thickness ``h``
            (``-thickness``; fork default 1.0). The mortar lane's interval
            integrals produce force per unit thickness, so the fork applies
            ``h`` once, at its 2D injection site, to ``eps_n``/``eps_t``/
            ``visc``/``cohesion``/``tau_max`` and the tie stiffness. Keep the
            three thickness conventions apart: the ELEMENT thickness
            (``ops.element.FourNodeQuad(thickness=…)``) is baked into element
            stiffness and contact never re-reads it; this ``h`` scales the
            EXPLICIT penalties above; and ``eps_n="auto"`` is deliberately NOT
            h-scaled (it already absorbs the element thickness through
            ``getInitialStiff()``, so re-scaling would be an h² error). An
            ``eps_t="auto"`` (or an eps_t defaulted from eps_n under
            friction) inherits ``eps_n``'s provenance, so it h-scales only
            when ``eps_n`` is explicit. The NTS lane has no ``-thickness``
            at all, and a 3D model is refused by name here.
        soft : float | bool, optional
            Explicit-only Courant-stable SOFT penalty (``-soft``): ``True`` ⇒
            the fork default SOFSCL (0.10); a float ⇒ an explicit SOFSCL. Needs
            a base penalty (``kn``/``eps_n``); excludes ``tie``. NTS=SOFT=1,
            mortar=SOFT=2. See :class:`ContactDef`.
        visc : float, optional
            Viscous normal-stabilisation coefficient μ_c (``-visc``); excludes
            ``tie``.
        consistent_tan : bool
            Non-symmetric consistent friction tangent (``-consistanttan``) —
            needs an unsymmetric solver (FullGeneral / UmfPack / BandGeneral).
        geom_tan : bool
            NTS ∂n/∂u geometric normal tangent (``-geomtan``) for curved /
            large-sliding interfaces. NTS-only.
        cell : float, optional
            Broad-phase cell-size scale (``-cell``): the spatial-hash bucket size
            as a fraction of the median segment diagonal (must be > 0). A
            performance knob — omit for the fork default. Both formulations.
        edge_edge : bool
            Enable the perpendicular edge-edge contact fallback (``-edgeedge``,
            ADR-57 E2). **Mortar-only.** See :class:`ContactDef`.
        edge_kn : float | "auto", optional
            Edge-edge normal penalty (``-edgeKn``); ``None`` ⇒ the mortar penalty.
        edge_band : float, optional
            Edge-edge gap activation band (``-edgeBand``).
        edge_mu, edge_kt, edge_cohesion, edge_tau_max : float, optional
            Edge-edge Coulomb/Tresca friction (``-edgeMu``/``-edgeKt``/
            ``-edgeCohesion``/``-edgeTauMax``).
        edge_consistent_tan : bool
            Edge-edge non-symmetric Csl friction tangent (``-edgeConsistentTan``).
        edge_soft : float | bool, optional
            Edge-edge explicit Courant-stable SOFT penalty (``-edgeSoft``).
        edge_alm : bool
            Edge-edge commit-cycle augmented Lagrangian (``-edgeAlm``).
        edge_aug_tol : float, optional
            Edge-edge ALM tolerance (``-edgeAugTol``).
        outward : (float, float, float) | (float, float) | "winding", optional
            ``None`` (default) → no ``-outward`` is emitted; the fork derives a
            correct per-facet normal (right for separated bodies and curved /
            closed / solid masters). Set an explicit direction ONLY for an
            initially-coincident (zero-gap) FLAT contact, where the fork's
            per-pair sign reference is in-plane and ambiguous. A single global
            outward is wrong on a non-flat master. See :class:`ContactDef`.

            **In a 2D model** this is a 2-vector ``(ox, oy)``, and a flush
            interface REQUIRES one (or ``"winding"``) — the fork's 2D lanes
            orient from an interface-level centroid vote that is ambiguous
            there and aborts. ``outward="winding"`` (2D NTS only) declares the
            side through the master chain's own winding instead of a
            direction, so it also orients curved and closed masters; it needs
            a fork build carrying ``-outward winding``.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities.
        name : str, optional
            Friendly name (round-trips into the emitted deck comment).

        Returns
        -------
        ContactDef
        """
        # Silent-failures slice 2 (PR #889) supersedes the ADR 0086 D4
        # draft guard: gate only from_h5/compose sessions — a LIVE
        # session legitimately re-resolves contact defs at the next
        # extraction. For a permanent bond on a composed assembly use
        # g.constraints.tie(method='mortar', enforce='equation').
        from ._compose_errors import raise_if_from_h5_session
        raise_if_from_h5_session(self._parent, "g.constraints.contact()")
        defn = ContactDef(
            master_label=master, slave_label=slave,
            master_entities=master_entities, slave_entities=slave_entities,
            formulation=formulation,
            kn=kn, kt=kt, mu=mu,
            eps_n=eps_n, eps_t=eps_t,
            cohesion=cohesion, tau_max=tau_max,
            aug_tol=aug_tol, max_aug=max_aug, ngp=ngp,
            tie=tie, thickness=thickness,
            outward=_normalise_outward(outward),
            soft=soft, visc=visc,
            consistent_tan=consistent_tan, geom_tan=geom_tan,
            cell=cell,
            edge_edge=edge_edge, edge_kn=edge_kn, edge_band=edge_band,
            edge_mu=edge_mu, edge_kt=edge_kt, edge_cohesion=edge_cohesion,
            edge_tau_max=edge_tau_max, edge_consistent_tan=edge_consistent_tan,
            edge_soft=edge_soft, edge_alm=edge_alm, edge_aug_tol=edge_aug_tol,
            name=name,
        )
        self.contact_defs.append(defn)
        return defn

    def resolve_contacts(self, node_tags, node_coords) -> list[ContactRecord]:
        """Resolve every :meth:`contact` def to a :class:`ContactRecord`.

        Pulls the master faceted surface (+ slave node set / faceted surface)
        from the live Gmsh session, dropping higher-order facets to corners,
        mirroring the additive g.reinforce / g.embed resolve. The outward
        normal is carried through only when the user set it explicitly — the
        fork kernel derives a correct per-facet normal otherwise (see the
        outward note below). ``node_tags`` / ``node_coords`` are accepted for
        signature parity with the sibling resolvers. Serial-only (the fork
        contact subsystem is not parallel).

        **The model dimension is read once, here, and threaded** — it is the
        single branch point of the 2D lane, mirroring
        :meth:`resolve_interfaces`. In a 2D model the master is a dim-1
        curve, collected as line segments and CHAINED head-to-tail into the
        fork's stride-2 pair list; in a 3D model nothing below changes.
        """
        records: list[ContactRecord] = []
        if not self.contact_defs:
            self.contact_records = records
            return records

        import gmsh
        model_dim = int(gmsh.model.getDimension())

        parts = getattr(self._parent, "parts", None)
        if parts is None:
            raise RuntimeError(
                "contact: g.parts is unavailable to collect surface faces.")

        # The whole-domain scratch every 2D chain walk needs. Built LAZILY
        # on the first surface (so its refusals still carry that
        # interaction's label, byte-identical to the per-call build) and
        # then shared: without it each surface pays its own Python-level
        # pass over every domain element, and the mortar lane walks two
        # surfaces per contact.
        frames = None
        if model_dim == 2:
            domain_tags, domain_conn = self._collect_domain_elements(
                verb="contact")
            xyz = {
                int(t): np.asarray(c, dtype=float)
                for t, c in zip(np.asarray(node_tags, dtype=int).ravel(),
                                np.asarray(node_coords, dtype=float))
            }

        for defn in self.contact_defs:
            m_ents = (defn.master_entities
                      or self._entities_for_label(defn.master_label))
            s_ents = (defn.slave_entities
                      or self._entities_for_label(defn.slave_label))
            _refuse_contact_entity_dim(
                m_ents, defn.master_label, model_dim=model_dim, role="master")

            if model_dim != 2 and defn.thickness is not None:
                raise ValueError(
                    f"contact: interaction "
                    f"{defn.name or defn.master_label!r} declares "
                    f"thickness={defn.thickness!r}, but the model is "
                    f"{model_dim}D. `-thickness` is the 2D plane-model "
                    f"out-of-plane thickness; a 3D mortar deck's thickness "
                    f"lives in its ELEMENTS, and the fork FATALs on a 3D "
                    f"pair carrying it. Drop thickness=.")

            if model_dim == 2:
                _refuse_contact_entity_dim(
                    s_ents, defn.slave_label,
                    model_dim=model_dim, role="slave",
                    formulation=defn.formulation)
                if frames is None:
                    from apeGmsh._kernel.geometry._boundary_chain import (
                        domain_frames,
                    )
                    frames = domain_frames(
                        domain_tags, domain_conn, xyz,
                        f" {defn.master_label!r}", verb="contact",
                        role="master")
                master_faces, master_nps = self._chain_2d_segments(
                    m_ents, defn.master_label, xyz, domain_tags, domain_conn,
                    role="master", frames=frames)
            else:
                master_faces = parts._collect_surface_faces(m_ents)
                if master_faces.size == 0:
                    raise ValueError(
                        f"contact: master label {defn.master_label!r} resolved to "
                        f"entities but carries no surface mesh faces (is it meshed?).")
                master_faces, master_nps = _drop_to_corner_facets(master_faces)

            if defn.formulation == "nts":
                slave_nodes = self._collect_node_set(s_ents, defn.slave_label)
                slave_faces, slave_nps = None, 0
            elif model_dim == 2:
                # 2D mortar — the slave is `-slave-segments 2`, the SAME
                # chained stride-2 pair list as the master and with the same
                # hole hazard, so it goes through the same walk (an unchained
                # slave listing is silently legal fork-side too). Its
                # direction is not load-bearing — the fork's orientation vote
                # reads sigma off the MASTER segments and uses the slave tags
                # only for the centroid datum — but winding it against its own
                # material costs nothing and keeps one code path.
                slave_faces, slave_nps = self._chain_2d_segments(
                    s_ents, defn.slave_label, xyz, domain_tags, domain_conn,
                    role="slave", frames=frames)
                slave_nodes = None
            else:  # mortar — faceted slave
                slave_faces = parts._collect_surface_faces(s_ents)
                if slave_faces.size == 0:
                    raise ValueError(
                        f"contact: mortar slave label {defn.slave_label!r} "
                        f"carries no surface mesh faces (is it meshed?).")
                slave_faces, slave_nps = _drop_to_corner_facets(slave_faces)
                slave_nodes = None

            # Outward normal: pass it ONLY when the user explicitly set it.
            # The fork contact kernel computes a correct PER-FACET normal from
            # each facet's connectivity and uses the supplied -outward purely as
            # a single global SIGN reference (LadrunoContactProjection.h
            # normalOriented). On a curved/closed/solid-part master a single
            # outward silently skips facets ~perpendicular to it and inverts
            # (→ inward, wrong contact) facets opposed to it; omitting it lets
            # the kernel use its correct per-pair (slave − segment-centroid)
            # sign reference. So never auto-derive a global outward here.
            #
            # 2D CARVE-OUT. The paragraph above settles 3D and does not reach
            # 2D, where the fork orients from ONE interface-level centroid
            # vote and aborts when it is ambiguous — flush interfaces (the
            # workhorse 2D case) and strongly curved / closed masters. The 2D
            # answer is `outward="winding"`: apeGmsh still derives no VECTOR
            # (the reasoning above is not overturned, it is out of scope) —
            # the side is declared through the master chain's ORIENTATION,
            # which the chained walk fixed against each segment's own
            # material. That is exact per segment, so it carries the cases a
            # single direction structurally cannot. The price is that the
            # fork's centroid vote is then bypassed, which is why the
            # wrong-side guard below exists.
            outward = _normalise_outward(defn.outward)
            if isinstance(outward, str) and model_dim != 2:
                raise ValueError(
                    f"contact: interaction "
                    f"{defn.name or defn.master_label!r} declares "
                    f"outward='winding', but the model is {model_dim}D. "
                    f"Declared-winding orientation is the fork's 2D NTS lane "
                    f"only — a 3D master is a facet SET with no head-to-tail "
                    f"chain to wind, and the 3D kernel already derives a "
                    f"correct per-facet normal from connectivity. Drop "
                    f"outward= in 3D, or pass an explicit (ox, oy, oz) for a "
                    f"coincident flat interface.")
            if outward is not None and not isinstance(outward, str):
                outward = tuple(float(x) for x in outward)
                if model_dim == 2 and len(outward) == 3 \
                        and outward[2] != 0.0:
                    raise ValueError(
                        f"contact: interaction "
                        f"{defn.name or defn.master_label!r} declares "
                        f"outward={outward!r} with a non-zero oz, but the "
                        f"model is 2D — a 2D outward lies in the plane. The "
                        f"fork takes only the 2-component form on a 2D "
                        f"surface (the 3-component 3D form is rejected "
                        f"there), so oz has nowhere to go. Pass "
                        f"outward=(ox, oy).")

            if model_dim == 2:
                # The two orientation guards apeGmsh owes a 2D deck, on
                # BOTH lanes: refuse the flush interface the fork cannot
                # orient, and refuse the master that faces away from the
                # slave — which nothing fork-side refuses (its vote only
                # picks a SIGN, and declared winding bypasses the vote
                # outright).
                #
                # Both take the slave as NODE TAGS, which the NTS lane has
                # directly and the mortar lane carries inside its own chain
                # (deduplicated — the fork's own vote dedups the
                # `-slave-segments` list for exactly this reason: every
                # interior chain vertex appears twice, and a
                # duplicate-weighted centroid can cross the magnitude floor).
                if slave_nodes is not None:
                    slave_tags = slave_nodes
                else:
                    slave_tags = sorted(
                        {int(t) for t in
                         np.asarray(slave_faces, dtype=np.int64).ravel()})
                if outward is None:
                    _refuse_flush_without_orientation(
                        defn, master_faces, slave_tags, xyz,
                        lref_segments=slave_faces)
                from apeGmsh._kernel.geometry._boundary_chain import (
                    refuse_wrong_side_master,
                )
                refuse_wrong_side_master(
                    master_faces, xyz, slave_tags,
                    _contact_label(defn), verb="contact")

            records.append(ContactRecord(
                kind="contact", name=defn.name,
                formulation=defn.formulation,
                master_faces=master_faces, master_nps=master_nps,
                slave_nodes=slave_nodes,
                slave_faces=slave_faces, slave_nps=slave_nps,
                outward=outward,
                kn=defn.kn, kt=defn.kt, mu=defn.mu,
                eps_n=defn.eps_n, eps_t=defn.eps_t,
                cohesion=defn.cohesion, tau_max=defn.tau_max,
                aug_tol=defn.aug_tol, max_aug=defn.max_aug, ngp=defn.ngp,
                tie=defn.tie, thickness=defn.thickness,
                soft=defn.soft, visc=defn.visc,
                consistent_tan=defn.consistent_tan, geom_tan=defn.geom_tan,
                cell=defn.cell,
                edge_edge=defn.edge_edge, edge_kn=defn.edge_kn,
                edge_band=defn.edge_band, edge_mu=defn.edge_mu,
                edge_kt=defn.edge_kt, edge_cohesion=defn.edge_cohesion,
                edge_tau_max=defn.edge_tau_max,
                edge_consistent_tan=defn.edge_consistent_tan,
                edge_soft=defn.edge_soft, edge_alm=defn.edge_alm,
                edge_aug_tol=defn.edge_aug_tol,
            ))

        self.contact_records = records
        return records

    def contact_plane(
        self, slave, *, normal, point, kn, visc=None, soft=None,
        slave_entities=None, name=None,
    ) -> ContactPlaneDef:
        """Declare a rigid analytical-plane contact (fork ``contactPlane``).

        The meshed ``slave`` surface contacts a fixed infinite **rigid plane**
        (``normal`` + ``point``) with normal penalty ``kn`` — frictionless, no
        master mesh. Use it for a rigid floor / wall / foundation where the
        counter-body needn't be meshed. Optional ``visc`` (viscous normal
        stabilisation) and ``soft`` (explicit Courant-stable SOFT penalty;
        ``True`` ⇒ the fork default SOFSCL 0.10, or a float SOFSCL). Fork-only
        at run time.

        Parameters
        ----------
        slave : str
            The meshed surface PG / part label whose nodes contact the plane.
            **In a 2D model** it is the meshed boundary CURVE (a dim-1 PG) or
            a point set — naming the dim-2 plane collects every interior node
            of the body and is refused by name.
        normal : (float, float, float) | (float, float)
            The plane's outward unit normal (toward the slave / open side).
            **In a 2D model** it is the 2-vector ``(nx, ny)``; it is z-padded
            internally and emitted as the fork's permanently-valid zero-padded
            9-argument form, so there is only ever one grammar to read.
        point : (float, float, float) | (float, float)
            Any point on the plane; ``(px, py)`` in a 2D model.
        kn : float
            Normal penalty stiffness (**required** — there is no ``"auto"`` on
            ``contactPlane``).
        visc : float, optional
            Viscous normal-stabilisation coefficient μ_c (``-visc``).
        soft : float | bool, optional
            Explicit-only Courant-stable SOFT penalty (``-soft``).
        slave_entities : list of (dim, tag), optional
            Restrict the slave to specific Gmsh entities.
        name : str, optional
            Friendly name (round-trips into the emitted deck comment).

        Returns
        -------
        ContactPlaneDef
        """
        from ._compose_errors import raise_if_from_h5_session
        raise_if_from_h5_session(
            self._parent, "g.constraints.contact_plane()")
        defn = ContactPlaneDef(
            slave_label=slave, slave_entities=slave_entities,
            normal=tuple(normal), point=tuple(point),
            kn=kn, visc=visc, soft=soft, name=name,
        )
        self.contact_plane_defs.append(defn)
        return defn

    def resolve_contact_planes(
        self, node_tags, node_coords,
    ) -> list[ContactPlaneRecord]:
        """Resolve every :meth:`contact_plane` def to a
        :class:`ContactPlaneRecord` — the slave node set is pulled from the live
        Gmsh session (mirroring the NTS slave of :meth:`resolve_contacts`).
        ``node_tags`` / ``node_coords`` are accepted for signature parity with
        the sibling resolvers. Serial-only (the fork contact subsystem is not
        parallel).

        **The model dimension is read once, here, and threaded** — the
        :meth:`resolve_contacts` idiom. This lane has no master mesh and no
        facets, so the whole 2D difference is the slave gate below plus the
        refusal of an out-of-plane normal / point: the rigid-plane lane keeps
        the fork's ``ndf >= ndm`` (its adapter couples the first ``ndm`` DOFs
        by construction, which is what lets a 3D ndf-6 shell sit on a plane),
        so unlike the NTS/mortar lanes there is nothing else to branch on.
        """
        records: list[ContactPlaneRecord] = []
        if not self.contact_plane_defs:
            self.contact_plane_records = records
            return records

        import gmsh
        model_dim = int(gmsh.model.getDimension())

        for defn in self.contact_plane_defs:
            s_ents = (defn.slave_entities
                      or self._entities_for_label(defn.slave_label))
            _refuse_contact_entity_dim(
                s_ents, defn.slave_label, model_dim=model_dim,
                role="slave", verb="contact_plane")
            if model_dim == 2:
                _refuse_contact_plane_out_of_plane(defn)
            slave_nodes = self._collect_node_set(s_ents, defn.slave_label)
            if not slave_nodes:
                raise ValueError(
                    f"contact_plane: slave label {defn.slave_label!r} resolved "
                    f"to no surface nodes (is it meshed?).")
            records.append(ContactPlaneRecord(
                kind="contact_plane", name=defn.name,
                slave_nodes=slave_nodes,
                normal=tuple(float(x) for x in defn.normal),
                point=tuple(float(x) for x in defn.point),
                kn=defn.kn, visc=defn.visc, soft=defn.soft,
            ))

        self.contact_plane_records = records
        return records

    def interface(
        self, master, slave, *,
        normal, tangential, thickness,
        tolerance=1e-6, slave_ndf=None,
        master_entities=None, slave_entities=None,
        name=None,
    ) -> InterfaceDef:
        """Declare an oriented coincident-pair ``zeroLength`` interface
        (ADR 0093).

        One ``zeroLength`` spring per coincident (master, slave) node
        pair, with the local axes taken **per pair** from the master
        face geometry — so a curved master's normal follows the face
        from wall to crown instead of collapsing to one average frame —
        and the normal / tangential laws scaled by each pair's
        tributary area. The point of the verb is a *unilateral*
        (compression-only, separation allowed) and *strength-capped*
        interface: with a bilateral bond a converging ground drives the
        liner's demand without bound.

        2D line masters only in v1; a 3D model or a surface master
        raises :class:`NotImplementedError` (ADR 0093 D2).

        Parameters
        ----------
        master, slave : str
            The master curve PG / part label — a **free boundary** of
            the meshed 2D continuum — and the node-for-node coincident
            slave label. The two node sets must be disjoint.
        normal : NormalLaw
            Per-area normal law: ``NormalLaw(kind="ent"|"epp_gap"|
            "elastic", k_per_area=..., ...)``. Declarative kernel data,
            translated to a typed uniaxial material (scaled by
            ``A_trib``) only at emit.
        tangential : TangentialLaw
            Per-area tangential law: ``TangentialLaw(kind="epp"|
            "elastic", k_per_area=..., tau_b=...)``.
        thickness : float
            Out-of-plane thickness (**required**, ``> 0``) —
            ``A_trib = ell_trib * thickness``.
        tolerance : float
            Coincidence radius for the node pairing. A slave with no
            master inside it is an error, never a silent skip.
        slave_ndf : {None, 2, 3}
            The ndf the slave wire will be declared with. ``None`` /
            ``2`` ⇒ the slave matches the 2D continuum and the pair
            connects directly; ``3`` ⇒ a beam slave, so each pair gets
            the phantom bridge of ADR 0093 D4 (the fork refuses a
            mixed-ndf ``zeroLength``). Explicit by design — see
            :class:`~apeGmsh._kernel.defs.constraints.InterfaceDef`.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities.
        name : str, optional
            Friendly name (carried onto every resolved record).

        Returns
        -------
        InterfaceDef
        """
        # Same gate as contact(): a from_h5 / composed session has no
        # live geometry to re-derive the per-pair frames from (ADR 0086
        # precedent, silent-failures slice 2).
        from ._compose_errors import raise_if_from_h5_session
        raise_if_from_h5_session(self._parent, "g.constraints.interface()")
        self._refuse_3d_interface()
        defn = InterfaceDef(
            master_label=master, slave_label=slave,
            master_entities=master_entities, slave_entities=slave_entities,
            normal=normal, tangential=tangential,
            thickness=thickness, tolerance=tolerance,
            slave_ndf=slave_ndf, name=name,
        )
        self.interface_defs.append(defn)
        return defn

    @staticmethod
    def _refuse_3d_interface() -> None:
        """Refuse ``interface()`` on a 3D model as early as we can see it.

        The authoritative gate is in :meth:`resolve_interfaces` (the
        master entity must be a curve, and its edges must be in-plane);
        this one just fires at declaration time when the model already
        carries 3D geometry, so the user learns before meshing. A model
        with no geometry yet reports dimension 0 and falls through to
        the resolve-time gate.
        """
        import gmsh
        try:
            dim = int(gmsh.model.getDimension())
        except Exception:
            return  # no live model — the resolve-time gate still applies
        if dim == 3:
            raise NotImplementedError(
                "g.constraints.interface(): the model is 3D, and only 2D "
                "line masters are implemented. A 3D surface master needs "
                "per-facet frames and a surface tributary model — "
                "deferred, ADR 0093 D2. Use g.constraints.contact() / "
                "tie() for a 3D interface.")

    def resolve_interfaces(self, node_tags, node_coords) -> list:
        """Resolve every :meth:`interface` def to :class:`InterfaceRecord`\\ s.

        Gathers the live-Gmsh inputs — both node sets, the master's
        boundary line elements, and the model's 2D domain elements —
        and hands the geometry math to
        :func:`~apeGmsh._kernel.resolvers._interface_resolver.resolve_interface_records`
        (pure kernel, no Gmsh), mirroring how :meth:`resolve_contacts`
        gathers and delegates.

        Must run **after** :meth:`resolve` so the interface phantom tags
        start above the MP lane's phantom high-water mark; the factory
        orders them that way.
        """
        from apeGmsh._kernel.resolvers._interface_resolver import (
            resolve_interface_records,
        )

        records: list = []
        if not self.interface_defs:
            self.interface_records = records
            return records

        import gmsh
        model_dim = int(gmsh.model.getDimension())
        if model_dim != 2:
            raise NotImplementedError(
                f"interface: the model is {model_dim}D and only 2D line "
                f"masters are implemented (ADR 0093 D2).")

        domain_tags, domain_conn = self._collect_domain_elements()
        # Start above BOTH the model's own node tags and any phantom the
        # MP lane already minted this resolve (see
        # ``_phantom_tag_high_water``).
        next_phantom = int(np.max(np.asarray(node_tags, dtype=int))) + 1
        if self._phantom_tag_high_water is not None:
            next_phantom = max(next_phantom, int(self._phantom_tag_high_water))

        for defn in self.interface_defs:
            m_ents = (defn.master_entities
                      or self._entities_for_label(defn.master_label))
            s_ents = (defn.slave_entities
                      or self._entities_for_label(defn.slave_label))
            bad_dims = sorted({int(d) for d, _ in m_ents} - {1})
            if bad_dims:
                raise NotImplementedError(
                    f"interface: master label {defn.master_label!r} "
                    f"resolves to entities of dimension {bad_dims}, but "
                    f"only 2D line (dim-1) masters are implemented — a "
                    f"surface master is deferred (ADR 0093 D2).")

            master_nodes = self._collect_node_set(
                m_ents, defn.master_label, kind="interface", role="master")
            slave_nodes = self._collect_node_set(
                s_ents, defn.slave_label, kind="interface", role="slave")
            edges = self._collect_master_edges(m_ents, defn.master_label)

            recs, next_phantom = resolve_interface_records(
                node_tags, node_coords,
                master_nodes=master_nodes,
                slave_nodes=slave_nodes,
                master_edges=edges,
                domain_elem_tags=domain_tags,
                domain_elem_nodes=domain_conn,
                normal_law=defn.normal,
                tangential_law=defn.tangential,
                thickness=defn.thickness,
                tolerance=defn.tolerance,
                slave_ndf=defn.slave_ndf,
                ndm=model_dim,
                phantom_tag_start=next_phantom,
                name=defn.name,
            )
            records.extend(recs)

        # Deliberately NOT written back to ``_phantom_tag_high_water``:
        # that field carries the MP lane's mark *for this extraction*,
        # and ``resolve()`` refreshes it every time. Folding this run's
        # advance into it would make a second ``get_fem_data()`` on the
        # same session mint different phantom tags — a tag-determinism
        # break (ADR 0027). Nothing downstream of here mints phantoms.
        self.interface_records = records
        return records

    @staticmethod
    def _collect_master_edges(entities, label) -> np.ndarray:
        """The master curve's 2-node boundary segments, as ``(n, 2)`` tags."""
        import gmsh
        rows: list[list[int]] = []
        for dim, tag in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(int(dim), int(tag))
            for etype, conn in zip(etypes, enodes):
                _, _, _, npe, *_ = gmsh.model.mesh.getElementProperties(
                    int(etype))
                if int(npe) == 2:
                    rows.extend(
                        np.asarray(conn, dtype=int).reshape(-1, 2).tolist())
                elif int(npe) == 3:
                    # gmsh line3: (end0, end1, mid) → two half-segments.
                    # Same expansion as ssi face_polyline / RevA FaceWalls
                    # rebuild.  Dropping to corners would leave midsides
                    # unsprung and break INV-3 against a coincident quadratic
                    # slave.  ADR 0093 D2/D3: the mid is a real polyline
                    # station with its own tributary + normal.
                    for e0, e1, mid in np.asarray(conn, dtype=int).reshape(-1, 3):
                        rows.append([int(e0), int(mid)])
                        rows.append([int(mid), int(e1)])
                else:
                    raise NotImplementedError(
                        f"interface: master label {label!r} is meshed with "
                        f"{int(npe)}-node line elements; only line2 and "
                        f"line3 (expanded to two half-segments) are "
                        f"implemented (ADR 0093 D2/D3).")
        if not rows:
            raise ValueError(
                f"interface: master label {label!r} resolved to entities "
                f"but carries no line elements (is the curve meshed?).")
        return np.asarray(rows, dtype=int)

    @staticmethod
    def _collect_master_segments(entities, label, *, role="master") -> np.ndarray:
        """The 2D contact surface's line elements, as ``(n, npe)`` tags.

        The 2D sibling of ``PartsRegistry._collect_surface_faces``: a
        rectangular connectivity block per entity, refusing a mixed
        ``line2``/``line3`` width. ``line3`` is dropped to its corners by
        :func:`_drop_to_corner_segments`, not here.
        """
        import gmsh
        blocks: list[tuple[int, Any]] = []
        for dim, tag in entities:
            etypes, _, enodes = gmsh.model.mesh.getElements(int(dim), int(tag))
            for etype, conn in zip(etypes, enodes):
                if len(conn) == 0:
                    continue
                _, _, _, npe, *_ = gmsh.model.mesh.getElementProperties(
                    int(etype))
                blocks.append((int(npe), conn))
        segs = _stack_line_blocks(blocks, label, role)
        if segs.size == 0:
            raise ValueError(
                f"contact: {role} label {label!r} resolved to entities but "
                f"carries no line elements (is the interface curve meshed?).")
        return segs

    @staticmethod
    def _chain_2d_segments(
        entities, label, xyz, domain_tags, domain_conn, *, role,
        frames=None,
    ) -> tuple[np.ndarray, int]:
        """A 2D contact surface as the fork's CHAINED stride-2 pair list.

        Gathers the curve's line elements, drops ``line3`` to its corners,
        fixes each segment's outward normal against its owning continuum
        element (the same :func:`edge_frames` the interface lane uses), and
        walks the directed segments into one head-to-tail chain. Returns
        ``(chained (n, 2) connectivity, 2)``.

        ``frames`` is the shared :class:`DomainFrames` scratch. Every call
        over one mesh derives the same per-element centroids and node
        adjacency, and the mortar lane makes two calls per contact (master
        chain + slave chain) on top of one per contact already — so the
        caller builds it once and hands it down rather than paying a
        whole-domain pass per surface.
        """
        from apeGmsh._kernel.geometry._boundary_chain import (
            chain_edges, edge_frames,
        )
        segs = ConstraintsComposite._collect_master_segments(
            entities, label, role=role)
        segs, nps = _drop_to_corner_segments(segs)
        m_set = {int(t) for t in segs.ravel()}
        missing = sorted(m_set - xyz.keys())
        if missing:
            raise ValueError(
                f"contact: {role} label {label!r} references node(s) "
                f"{missing[:20]} that are not in the model node pool.")
        text = f" {label!r}"
        data = edge_frames(
            segs, m_set, xyz, domain_tags, domain_conn, text,
            verb="contact", role=role, frames=frames)
        return (chain_edges(data, xyz, text, verb="contact", role=role),
                nps)

    @staticmethod
    def _collect_domain_elements(
        *, verb: str = "interface",
    ) -> tuple[list[int], list[np.ndarray]]:
        """Every 2D (top-dimension) element in the model: tags + connectivity.

        Deliberately the *domain* elements only — never the boundary
        curve elements. ADR 0093 INV-5: top-dimension elements are the
        ones ``_fem_extract`` never replicates across a partition cut,
        so a backing element picked from this set identifies exactly one
        owner rank.
        """
        import gmsh
        tags: list[int] = []
        conn: list[np.ndarray] = []
        etypes, etags, enodes = gmsh.model.mesh.getElements(2)
        for etype, et, en in zip(etypes, etags, enodes):
            _, _, _, npe, *_ = gmsh.model.mesh.getElementProperties(int(etype))
            rows = np.asarray(en, dtype=int).reshape(-1, int(npe))
            tags.extend(int(t) for t in et)
            conn.extend(rows)
        if not tags:
            raise ValueError(
                f"{verb}: the model carries no 2D elements — the "
                "outward normal (ADR 0093 D2) and the backing element "
                "(INV-5) are both derived from the continuum elements "
                "adjacent to the master face.")
        return tags, conn

    def _collect_node_set(self, entities, label, *, kind="contact",
                          role="slave") -> list[int]:
        """The mesh-node tags of *entities* (unique, first-seen order).

        A discrete curve may ``addElements`` that cite nodes classified
        on another entity (so a FaceWalls sub-chain can be named as
        ``master_entities`` without ``addNodes`` duplicating those tags
        in the OpenSees deck). ``getNodes`` is empty on that host;
        the stretch is the unique connectivity of its line elements.
        Only used when ``getNodes`` returned nothing — a classified
        slave (Arch with feet/fuse omitted from ``addNodes``) must
        NOT expand to element connectivity, or the omitted stations
        would re-enter the slave set.
        """
        import gmsh
        seen: dict[int, None] = {}
        for dim, tag in entities:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    int(dim), int(tag),
                    includeBoundary=True, returnParametricCoord=False)
            except Exception as exc:
                raise ValueError(
                    f"{kind}: cannot get mesh nodes for {role} entity "
                    f"(dim={dim}, tag={tag}) of label {label!r}: {exc}") from exc
            for t in nt:
                seen.setdefault(int(t), None)
            if len(nt) > 0:
                continue
            etypes, _, enodes = gmsh.model.mesh.getElements(
                int(dim), int(tag))
            for conn in enodes:
                for t in np.asarray(conn, dtype=int).ravel():
                    if int(t) > 0:
                        seen.setdefault(int(t), None)
        if not seen:
            raise ValueError(
                f"{kind}: {role} label {label!r} resolved to entities but "
                f"carries no mesh nodes (is it meshed?).")
        return list(seen.keys())

    def _add_def(self, defn: _ConstraintT) -> _ConstraintT:
        """Validate dispatch + labels and store a constraint definition.

        A def whose type has no ``_DISPATCH`` entry cannot be resolved
        and would be silently dropped at :meth:`resolve` — reject it at
        declaration instead, mirroring
        ``LoadsComposite``/``MassesComposite._add_def``.  Label
        validation is skipped for NodeToSurfaceDef / EmbeddedDef
        (bare tags).

        Chain phase
        -----------
        When the session is in chain phase (``g._fem is not None``)
        ``g.parts._instances`` is typically empty — the user came in
        via ``apeGmsh.from_h5(...)`` or ``g.compose(...)`` rather than
        building parts up from gmsh.  Validate labels against the
        FEMData broker's labels / physical groups via
        :class:`FEMDataSource.has_target` instead, so cross-session
        interface bridging (Compose v1.1-A) keeps the same fail-loud
        contract without requiring a parts registry.
        """
        if type(defn) not in _DISPATCH:
            raise TypeError(
                f"{type(defn).__name__} has no _DISPATCH entry — it "
                f"cannot be resolved and would be silently dropped. "
                f"Register it in _DISPATCH (and _RESOLVER_METHOD) "
                f"before use."
            )
        if not isinstance(defn, (NodeToSurfaceDef, EmbeddedDef)):
            # Decoupled labels (ADR 0049 OQ2) are known to the session
            # composite, not to Parts / PGs / FEMDataSource — pre-approve
            # them for RBE2/RBE3 before either phase gate runs.  Live
            # sessions often have ``g._fem`` set after the first
            # ``get_fem_data``, so both branches must see the same set.
            decoupled_ok: set[str] = set()
            for lbl in (defn.master_label, defn.slave_label):
                if not isinstance(lbl, str) or not lbl:
                    continue
                if self._decoupled_def_for(lbl) is not None:
                    decoupled_ok.add(lbl)

            in_chain_phase = getattr(self._parent, "_fem", None) is not None
            if in_chain_phase:
                # Chain phase: validate via the FEMData broker.
                from apeGmsh._kernel.resolvers._source import FEMDataSource

                src = FEMDataSource(self._parent._fem)
                for lbl in (defn.master_label, defn.slave_label):
                    is_decoupled = lbl in decoupled_ok
                    ok = src.has_target(lbl) if isinstance(lbl, str) else False
                    if is_decoupled and ok:
                        raise ValueError(
                            f"Constraint label {lbl!r} names both a "
                            f"decoupled node and a label / physical group "
                            f"in the FEMData chain head — rename one so "
                            f"the constraint role is unambiguous."
                        )
                    if is_decoupled and type(defn) not in _DECOUPLED_ROLE_TYPES:
                        raise ValueError(
                            f"Constraint label {lbl!r} is a decoupled node, "
                            f"but {type(defn).__name__} does not accept "
                            f"decoupled roles. Use kinematic_coupling "
                            f"(RBE2) or distributing_coupling (RBE3), or "
                            f"pass master_entities=/slave_entities=."
                        )
                    if not is_decoupled and not ok:
                        raise KeyError(
                            f"Constraint label {lbl!r} resolves to "
                            f"neither a label nor a physical group in "
                            f"the current FEMData chain head — pass a "
                            f"label/PG name that exists in the loaded "
                            f"model."
                        )
            else:
                parts = getattr(self._parent, "parts", None)
                if parts is not None and hasattr(parts, "_instances"):
                    part_names = parts._instances
                    for lbl in (defn.master_label, defn.slave_label):
                        is_decoupled = lbl in decoupled_ok
                        ok = self._label_resolvable(lbl, part_names)
                        if is_decoupled and ok:
                            raise ValueError(
                                f"Constraint label {lbl!r} names both a "
                                f"decoupled node and a part / physical "
                                f"group / label — rename one so the "
                                f"constraint role is unambiguous."
                            )
                        if is_decoupled and type(defn) not in _DECOUPLED_ROLE_TYPES:
                            raise ValueError(
                                f"Constraint label {lbl!r} is a decoupled "
                                f"node, but {type(defn).__name__} does not "
                                f"accept decoupled roles. Use "
                                f"kinematic_coupling (RBE2) or "
                                f"distributing_coupling (RBE3), or pass "
                                f"master_entities=/slave_entities=."
                            )
                        if not is_decoupled and not ok:
                            raise KeyError(
                                f"Constraint label {lbl!r} is not a part, "
                                f"physical group, or label.  Build it as a "
                                f"Part (g.parts), tag it as a physical group "
                                f"(g.physical / .to_physical({lbl!r})), or "
                                f"pass explicit master_entities=/"
                                f"slave_entities=.  Available parts: "
                                f"{sorted(part_names)}."
                            )
        self.constraint_defs.append(defn)
        # Phase 3B.2d / ADR 0038 — chain-phase routing.  Constraint
        # defs (equalDOF / rigidLink / rigidDiaphragm / embedded /
        # tied_contact) need element-side resolution that the
        # minimum-viable router does not yet cover; the call falls
        # through to the bump-counter pattern with a documented gap
        # (the def is stored on ``self.constraint_defs`` but not
        # applied to ``_fem`` until a build-phase
        # ``get_fem_data()`` re-extraction).
        from apeGmsh._kernel.resolvers._chain_phase_router import (
            try_chain_phase_route,
        )
        try_chain_phase_route(self._parent, defn)
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()
        return defn

    def _label_resolvable(self, lbl, part_names) -> bool:
        """True if ``lbl`` names a part, physical group, or label.

        Bare-tag / empty labels are accepted here and validated at
        resolve time.  Part labels short-circuit; other strings are run
        through the shared label→PG→part geometry resolver (the same one
        loads/masses use) so a physical-group model can constrain without
        building Parts.  A multi-dim PG raises ``ValueError`` from the
        resolver (a real config error) rather than being reported here as
        "unknown".
        """
        if not isinstance(lbl, str) or not lbl:
            return True
        if lbl in part_names:
            return True
        from ._helpers import resolve_to_dimtags
        try:
            resolve_to_dimtags(lbl, default_dim=3, session=self._parent)
        except KeyError:
            return False
        return True

    def _decoupled_def_for(self, label):
        """The :class:`DecoupledNodeDef` named ``label``, or ``None``.

        Raises ``ValueError`` when two defs share the label — an ambiguous
        constraint reference must not silently pick one.  Uniqueness is
        enforced only at *reference* time so ``g.decouple_node`` itself
        stays unchanged.
        """
        if not isinstance(label, str) or not label:
            return None
        comp = getattr(self._parent, "decoupled_nodes", None)
        defs = getattr(comp, "node_defs", None) if comp is not None else None
        if not defs:
            return None
        matches = [d for d in defs if getattr(d, "label", None) == label]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                f"decoupled label {label!r} matches {len(matches)} "
                f"g.decouple_node declarations — rename so the constraint "
                f"role is unambiguous."
            )
        return matches[0]

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
        broker channel as ``g.displacements.surface`` — **not**
        ``fem.nodes.constraints``.

        Because it is a *permanent* constraint (not a pattern-scoped
        quantity), it lives here on ``g.constraints`` rather than on
        ``g.displacements``: there is no load-pattern context to accidentally
        scope it into, and the downstream emitter places it in the
        ``model → bcs → patterns`` deck order via ``ops.fix``.

        Parameters
        ----------
        target : str or list[(dim, tag)]
            Pattern to fix. Resolved label → physical group → raw
            tags (or a mesh selection) — the same flexible target
            model as ``g.displacements.surface``. Pass ``pg=`` / ``label=``
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
        # Phase 3B.2d / ADR 0038 — chain-phase routing.  See
        # ``MassesComposite._add_def`` for the contract.
        from apeGmsh._kernel.resolvers._chain_phase_router import (
            try_chain_phase_route,
        )
        try_chain_phase_route(self._parent, defn)
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()
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
            Part, physical-group, or label name whose nodes drive the
            constraint.
        slave_label : str
            Part, physical-group, or label name whose matching nodes
            are slaved.
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

    def equal_dof_mixed(self, master_label, slave_label, *, dof_pairs,
                        master_entities=None, slave_entities=None,
                        tolerance=1e-6, name=None) -> EqualDOFMixedDef:
        """Tie *differently-numbered* DOFs between **co-located** node pairs.

        The mixed analog of :meth:`equal_dof`: rather than tying DOF
        ``i`` to DOF ``i``, each ``(retained_dof, constrained_dof)``
        couple in ``dof_pairs`` is tied explicitly — so master ``ux``
        can drive slave ``rz``, etc. Resolves like ``equal_dof`` (one
        record per co-located pair) and emits
        ``ops.equalDOF_Mixed(R, C, numDOF, RDOF1, CDOF1, ...)`` per pair.

        Use this for **conformal** interfaces where the two sides expose
        the coupled quantity under different DOF indices (e.g. tying a
        solid's translation to a shell's drilling rotation). For matching
        DOFs use :meth:`equal_dof`; for non-matching meshes use :meth:`tie`.

        Parameters
        ----------
        master_label : str
            Part label whose nodes are **retained** (the ``R`` node).
        slave_label : str
            Part label whose matching nodes are **constrained** (``C``).
        dof_pairs : list of (int, int)
            ``(retained_dof, constrained_dof)`` couples, 1-based
            (``1=ux, 2=uy, 3=uz, 4=rx, 5=ry, 6=rz``). Required and
            non-empty; the two members of a couple may differ.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict the node search to specific Gmsh entities of each side.
        tolerance : float, default 1e-6
            Co-location distance (model units). **Unit-sensitive** — see
            :meth:`equal_dof`.
        name : str, optional
            Friendly name shown in :meth:`summary` and the viewer.

        Returns
        -------
        EqualDOFMixedDef
            The stored definition; also appended to ``self.constraint_defs``.

        See Also
        --------
        equal_dof : Same-DOF co-located tie (the common case).

        Examples
        --------
        Tie a solid face's z-translation to a shell edge's drilling DOF::

            g.constraints.equal_dof_mixed(
                "solid", "shell",
                dof_pairs=[(3, 6)],    # master uz → slave rz
                tolerance=1e-3,        # mm model
            )
        """
        return self._add_def(EqualDOFMixedDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dof_pairs=[tuple(p) for p in dof_pairs],
            tolerance=tolerance, name=name))

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
            Part, physical-group, or label name that owns the master
            node. The master is
            identified inside this part either by ``master_point``
            (proximity match) or by being the unique node when the
            part collapses to a single point.
        slave_label : str
            Part, physical-group, or label name whose nodes become
            slaves.
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
            Part, physical-group, or label name that contains (or whose
            proximity will
            select) the master node — typically a centre-of-mass
            point.
        slave_label : str
            Part, physical-group, or label name whose nodes are
            gathered into the diaphragm.
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
                   master_point=(0., 0., 0.), as_element=False, mass=None,
                   omega=None, name=None) -> RigidBodyDef:
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
            Part, physical-group, or label name that contains (or whose
            proximity selects)
            the master node.
        slave_label : str
            Part, physical-group, or label name whose nodes are
            gathered into the rigid
            body.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the master node.
        as_element : bool, default False
            Emit the fork ``element LadrunoRigidBody`` over the whole node
            set ``{master, *slaves}`` (class tag 33015, **3D only**)
            instead of the default ``rigidLink`` chain. The element gives
            a private centre-of-mass node, condensed body mass, and
            explicit-dynamics support that the rigidLink chain cannot.
            Fork-only: deck emission works on any build; running needs the
            Ladruno fork.
        mass : float or None
            Total body mass for ``as_element`` (``-mass``); ``None``
            condenses it from the slaves' nodal mass. Only valid with
            ``as_element=True``.
        omega : (wx, wy, wz) or None
            Initial body-frame angular velocity for ``as_element``
            (``-omega``) — an explicit-dynamics initial condition (the body
            spins from t=0). Only valid with ``as_element=True``.
        name : str, optional
            Friendly name.

        Returns
        -------
        RigidBodyDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts``.
        ValueError
            If ``mass``/``omega`` is set without ``as_element=True``, or
            ``mass < 0``.

        See Also
        --------
        kinematic_coupling : Same topology but with a user-selectable
            DOF subset.
        rigid_diaphragm : In-plane variant with plane filtering.
        """
        return self._add_def(RigidBodyDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, as_element=as_element, mass=mass,
            omega=(None if omega is None else tuple(omega)), name=name))

    def kinematic_coupling(self, master_label, slave_label, *,
                           master_point=(0., 0., 0.), dofs=None,
                           k=None, k_alpha=None, host=None,
                           kr=None, enforce="penalty",
                           bipenalty_dtcr=None, bipenalty_wcap=None,
                           absolute=False,
                           name=None) -> KinematicCouplingDef:
        """RBE2 / kinematic coupling — a reference node rigidly drives a
        node set.

        Emits the Ladruno-fork ``element LadrunoKinematicCoupling`` (class
        tag 33012): a penalty rigid-body driver with the correct moment-arm
        transport ``u_i = u_R + θ_R × d_i``, so an *offset* reference is
        coupled rigidly. This replaces the previous ``equalDOF``-per-slave
        expansion, which ignored the lever arm (correct only for coincident
        nodes). **Fork-only:** the deck emits on any build, but running it
        needs the Ladruno fork — stock OpenSees fails loud at the element
        line (it does not know class tag 33012).

        Reach for this when the region must move as a **rigid body** (a
        loading platen, a rigid offset / connection block, a rigid
        diaphragm over an arbitrary node set). To *introduce a load at a
        point while the region stays flexible*, use a distributing coupling
        (RBE3) instead.

        Parameters
        ----------
        master_label : str | DecoupledNodeDef
            Part, physical-group, or label name that owns the reference
            (master) node — **or** a ``g.decouple_node`` handle / its
            ``label=`` (ADR 0049 OQ2). The reference node must carry the
            rotational DOFs (ndf 6 in 3D / 3 in 2D); the fork refuses a
            too-small reference at ``setDomain``.
        slave_label : str | DecoupledNodeDef
            Part, physical-group, or label name whose nodes are slaved
            (may mix 3- and 6-DOF nodes). A ``g.decouple_node`` handle
            is accepted the same way as ``master_label``.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the reference node when the master role
            resolves to a multi-node set (nearest-in-set). **Ignored**
            when the role is a single decoupled node — that node's own
            coordinates are used.
        dofs : list[int], optional
            1-based dependent components tied on each slave (``-dof``).
            ``None`` (default) ties *every DOF the slave has* — the right
            choice for a mixed 3/6-DOF slave set; pass an explicit list to
            restrict, e.g. ``[1, 2, 3]`` for translations only or
            ``[3]`` for a vertical-only follower.
        k : float | ``"auto"``, optional
            Translational penalty stiffness (``-k``). ``None`` ⇒ the fork
            default (``1e12``). ``"auto"`` scales it off a representative
            host element's stiffness diagonal
            (``K_t = k_alpha · max|K_host(i,i)|``) — requires ``host``.
        k_alpha : float, optional
            Multiplier for ``k="auto"`` (``-kAlpha``; fork default ``1e3``).
            Only valid together with ``k="auto"``.
        host : int, optional
            Representative host element for ``k="auto"`` / ``bipenalty_wcap``
            (``-host``) as a **FEM element id** — the bridge translates it to
            the emitted OpenSees tag at emit time. Pick a typical element of
            the coupled part (e.g. one touching the slave surface).
        kr : float, optional
            Rotational penalty stiffness (``-kr``). ``None`` ⇒ fork-derived
            ``K_t·ℓ²`` (keeps the translation/rotation conditioning matched).
        enforce : ``"penalty"`` | ``"al"``, default ``"penalty"``
            ``"al"`` = augmented Lagrangian (near-exact rigidity at moderate
            ``k``; **implicit only** — cannot combine with the bipenalty
            knobs).
        bipenalty_dtcr : float, optional
            Explicit-dynamics critical-time-step target (``-bipenalty
            -dtcr``); lumps a penalty mass on any massless tied DOF so the
            stiff tie doesn't collapse the explicit step. ``None`` ⇒ off
            (the master is usually a massed node).
        bipenalty_wcap : float, optional
            Bipenalty via the host frequency (``-bipenalty -wcap``):
            ``m_p = K_t/(β·ω_host)²`` with ``β`` = this value — sets the
            penalty-mode frequency at ``β·ω_host`` instead of a hard ``dt``
            budget. Requires ``host``; mutually exclusive with
            ``bipenalty_dtcr``.
        absolute : bool, default False
            Keep the **absolute** tie (``-absolute``) — skip the default
            ``g0`` stress-free birth (a coupling added to a deformed model
            is otherwise born force-free).
        name : str, optional
            Friendly name (also the stage-claim key for ``s.kinematic_coupling``).

        Returns
        -------
        KinematicCouplingDef

        Raises
        ------
        KeyError
            If either label is not in ``g.parts`` / PGs / labels and is not
            a labelled ``g.decouple_node``.
        ValueError
            On an invalid knob (``enforce`` not in {penalty, al};
            non-positive ``k``/``kr``/``bipenalty_dtcr``/``bipenalty_wcap``;
            ``al`` + a bipenalty knob; ``k="auto"`` or ``bipenalty_wcap``
            without ``host``; ``k_alpha`` without ``k="auto"``; a dangling
            ``host`` no knob consumes; ``bipenalty_dtcr`` + ``bipenalty_wcap``);
            a ``g.decouple_node`` handle without ``label=``; a label that
            names both a decoupled node and a Part/PG; an ambiguous
            duplicate decoupled label.
        """
        master_label = _as_role_label(master_label, "master_label")
        slave_label = _as_role_label(slave_label, "slave_label")
        return self._add_def(KinematicCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, dofs=dofs,
            control=CouplingControl(
                k=k, k_alpha=k_alpha, host=host, kr=kr, enforce=enforce,
                bipenalty_dtcr=bipenalty_dtcr, bipenalty_wcap=bipenalty_wcap,
                absolute=absolute,
            ),
            name=name))

    # ── Tier 3 — Node-to-Surface ─────────────────────────────────────
    def tie(self, master_label, slave_label, *, master_entities=None,
            slave_entities=None, dofs=None, tolerance=1.0,
            stiffness="auto", stiffness_p=None,
            rotational=False, pressure=False,
            enforce="penalty", control=None,
            method="collocation", outward=None,
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
        successfully projected slave node. The emitted OpenSees coupling
        depends on ``enforce=`` (see below): a penalty
        ``ASDEmbeddedNodeElement`` (default), the fork
        ``LadrunoEmbeddedNode``, or an exact ``equationConstraint``
        (EQ_Constraint).

        Parameters
        ----------
        master_label : str
            Part, physical-group, or label name of the master surface
            (the side whose mesh
            will provide the shape functions).
        slave_label : str
            Part, physical-group, or label name of the slave surface
            (whose nodes are
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
            are skipped with a warning (an all-skipped tie raises) —
            set generously if the two meshes have a small geometric
            gap, but not so large that the wrong face is selected.
            **Unit-sensitive.**
        stiffness : float or ``"auto"``, default ``"auto"``
            Penalty stiffness ``K`` of the emitted
            ``ASDEmbeddedNodeElement`` (penalty routes only; ignored by
            ``"equation"``). ``"auto"`` resolves at emit from the host
            material: ``K = α·E_host·L_char`` (α = 1e3, E from the
            element's material, L from the master-face size) — a few
            orders above the host element stiffness, which is all the
            penalty needs. **A numeric value is unit-dependent and must
            be calibrated against a known solution**: the pre-slice-B
            default ``1e18`` (the OpenSees C++ default) destroys the
            conditioning in N/mm/MPa (E ≈ 2e5) and Newton stalls, while
            ``1e10``–``1e12`` converge with sub-percent stiffness
            error; emit still warns when a record carries 1e18.
        stiffness_p : float, optional
            Separate rotational/pressure penalty (``-KP``); ``None`` ⇒
            the element falls back to ``K``. Same unit caveat.
        rotational, pressure : bool, default False
            Extend the coupling to rotational / pressure DOFs
            (``-rot`` / ``-p``); penalty routes only.
        enforce : {"penalty", "penalty_al", "equation"}, default "penalty"
            Coupling route (ADR 0068). ``"penalty"`` →
            ``ASDEmbeddedNodeElement`` penalty element (tunable ``K``,
            handler-independent). ``"equation"`` → exact
            ``equationConstraint`` (EQ_Constraint), translations only,
            enforced by the ``Lagrange`` (implicit) / ``LadrunoProjection``
            (explicit, Δt-neutral) handler — auto-selected at emit, and
            penalty-only knobs (``rotational``/``pressure``/``stiffness_p``)
            are rejected. ``"penalty_al"`` → fork ``LadrunoEmbeddedNode``
            (penalty + augmented-Lagrange + bipenalty, translations only),
            configured via ``control=`` (see below).
        control : CouplingControl, optional
            LadrunoEmbeddedNode penalty/AL/bipenalty knobs — only valid with
            ``enforce="penalty_al"`` (reuses the RBE2/RBE3
            :class:`CouplingControl`: ``-k``/``-kAlpha``/``-host``/
            ``-enforce al``/``-bipenalty``/``-absolute``). ``None`` ⇒ the
            fork element's own defaults.
        method : {"collocation", "mortar"}, default "collocation"
            Weight-computation method (ADR 0086). ``"collocation"`` is
            the classic node-to-face projection above. ``"mortar"``
            integrates the interface over the slave/master facet
            overlaps with a dual (biorthogonal) slave basis, so
            **neither side's interpolation order is imposed on the
            other** — the fix for order-mismatched interfaces (e.g.
            hex20 faces tied onto hex8 faces, where collocation
            over-constrains the quadratic side). Requires
            ``enforce="equation"`` (v1), works on composed assemblies
            (chain phase), and is fail-loud end to end: a flat,
            coincident, convex interface is required and every
            degenerate case raises ``MortarTieError`` — a mortar tie
            never silently resolves to nothing. ``tolerance`` becomes
            the out-of-plane coincidence tolerance. Interface edges must
            be straight (every midside node at its edge midpoint) and no
            master facet may overlap another — both are hard errors,
            because the kernel integrates on the corner polygon and its
            coverage check counts multiplicity. tri6 SLAVE facets are
            refused (dual-basis degeneracy) — swap the sides or use
            collocation.
        outward : (ox, oy, oz), optional
            ``method="mortar"`` only: interface-plane normal override.
            Normally derived from master facet winding; needed only
            when the winding sum cancels (the kernel raises naming
            this knob — there is no silent zero-force path).
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
        mortar : Deprecated alias for a fork mortar mesh-tie
            (``contact(formulation="mortar", tie=True)``).

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
            rotational=rotational, pressure=pressure, enforce=enforce,
            control=control, method=method, outward=outward))

    def distributing_coupling(self, master_label, slave_label, *,
                              master_point=(0., 0., 0.),
                              weighting="uniform",
                              k=None, k_alpha=None, host=None,
                              kr=None, enforce="penalty",
                              bipenalty_dtcr=None, bipenalty_wcap=None,
                              absolute=False,
                              name=None) -> DistributingCouplingDef:
        """RBE3 / distributing coupling — distribute a load at a reference
        point over a node set while the set stays flexible.

        Emits the Ladruno-fork ``element LadrunoDistributingCoupling``
        (class tag 33011): the reference (dependent) node R is the
        weighted-average rigid-body fit of the independent set, and a
        force/moment at R is distributed to the set as a
        statically-equivalent pattern (``Σ Fᵢ = F``, ``Σ rᵢ × Fᵢ = M``)
        **adding no stiffness** to the independents. This is the proper
        RBE3 — it replaces the prior `NotImplementedError` stub (whose
        predecessor emitted a mechanically-wrong kinematic mean). It is
        the flexible counterpart of :meth:`kinematic_coupling` (RBE2,
        which holds the set rigid). **Fork-only:** the deck emits on any
        build, but running it needs the Ladruno fork — stock OpenSees
        fails loud at the element line (it does not know class tag 33011).

        Reach for this to **introduce/transmit a load or BC at a point
        while the region stays flexible** (a column base on a footing, an
        actuator head on a face, a beam/shell moment into a solid face);
        use :meth:`kinematic_coupling` (RBE2) when the region must move as
        a rigid body, or :meth:`tie` for a compatible face interpolation.

        Parameters
        ----------
        master_label : str | DecoupledNodeDef
            Part, physical-group, or label name owning the reference
            (dependent) node R — **or** a ``g.decouple_node`` handle / its
            ``label=`` (ADR 0049 OQ2). R must carry the rotational DOFs
            (ndf 6 in 3D / 3 in 2D) to transmit a moment; the fork refuses
            a too-small reference at ``setDomain``.
        slave_label : str | DecoupledNodeDef
            Part, physical-group, or label name whose nodes form the
            **independent** set (translations-only is fine — no rotational
            stiffness is injected). A ``g.decouple_node`` handle is
            accepted the same way as ``master_label``.
        master_point : (x, y, z), default (0, 0, 0)
            Coordinates of the reference node R when the master role
            resolves to a multi-node set (nearest-in-set). **Ignored**
            when the role is a single decoupled node — that node's own
            coordinates are used.
        weighting : ``"uniform"`` | ``"area"``, default ``"uniform"``
            ``"uniform"`` ⇒ equal weights (``-w`` omitted, the fork
            element's default). ``"area"`` ⇒ apeGmsh computes each
            independent node's **tributary area** over the slave
            surface (each face's area split equally among its nodes —
            the same lumping model as ``g.loads`` surface-tributary
            resolution) and emits ``-w w1..wN``, so a force at R
            distributes like a uniform traction on the surface.
            Requires the slave label (or ``slave_entities``) to resolve
            to meshed surface faces; an independent node on no slave
            face fails loud.
        k : float | ``"auto"``, optional
            Translational penalty stiffness (``-k``). ``None`` ⇒ the fork
            default (``1e12``). ``"auto"`` scales it off a representative
            host element's stiffness diagonal
            (``K_t = k_alpha · max|K_host(i,i)|``) — requires ``host``.
            Note the **force distribution is exact for any penalty** (the
            RBE3 property); ``k`` only relaxes the *kinematic* fit of the
            reference node.
        k_alpha : float, optional
            Multiplier for ``k="auto"`` (``-kAlpha``; fork default ``1e3``).
            Only valid together with ``k="auto"``.
        host : int, optional
            Representative host element for ``k="auto"`` / ``bipenalty_wcap``
            (``-host``) as a **FEM element id** — the bridge translates it to
            the emitted OpenSees tag at emit time. RBE3 has no single host
            by construction; name ONE typical element among the independents'
            parents — it is read ONLY to scale the penalties.
        kr : float, optional
            Rotational penalty stiffness (``-kr``). ``None`` ⇒ fork-derived.
        enforce : ``"penalty"`` | ``"al"``, default ``"penalty"``
            ``"al"`` = augmented Lagrangian — recovers a near-exact weighted
            fit of R at moderate ``k`` (**implicit only**; cannot combine
            with the bipenalty knobs).
        bipenalty_dtcr : float, optional
            Explicit-dynamics critical-time-step target (``-bipenalty
            -dtcr``). The reference node is **massless** by construction, so
            an explicit run needs this (or it has a zero stable step).
            ``None`` ⇒ off.
        bipenalty_wcap : float, optional
            Bipenalty via the host frequency (``-bipenalty -wcap``):
            ``m_p = K_t/(β·ω_host)²`` with ``β`` = this value. Requires
            ``host``; mutually exclusive with ``bipenalty_dtcr``.
        absolute : bool, default False
            Keep the **absolute** tie (``-absolute``) — skip the default
            ``g0`` stress-free birth.
        name : str, optional
            Friendly name (also the stage-claim key for `s.distributing`).

        Returns
        -------
        DistributingCouplingDef

        Raises
        ------
        ValueError
            On an invalid knob (``enforce`` not in {penalty, al};
            non-positive ``k``/``kr``/``bipenalty_dtcr``/``bipenalty_wcap``;
            ``al`` + a bipenalty knob; ``k="auto"`` or ``bipenalty_wcap``
            without ``host``; ``k_alpha`` without ``k="auto"``; a dangling
            ``host`` no knob consumes; ``bipenalty_dtcr`` + ``bipenalty_wcap``;
            ``weighting`` not in {uniform, area}); a ``g.decouple_node``
            handle without ``label=``; a label that names both a
            decoupled node and a Part/PG; an ambiguous duplicate
            decoupled label.
        """
        if weighting not in ("uniform", "area"):
            raise ValueError(
                "distributing_coupling: weighting must be 'uniform' (equal "
                "weights) or 'area' (tributary-area -w weights computed over "
                f"the slave surface), got {weighting!r}."
            )
        master_label = _as_role_label(master_label, "master_label")
        slave_label = _as_role_label(slave_label, "slave_label")
        return self._add_def(DistributingCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, weighting=weighting,
            control=CouplingControl(
                k=k, k_alpha=k_alpha, host=host, kr=kr, enforce=enforce,
                bipenalty_dtcr=bipenalty_dtcr, bipenalty_wcap=bipenalty_wcap,
                absolute=absolute,
            ),
            name=name))

    def embedded(self, host_label, embedded_label, *, tolerance=1.0,
                 host_entities=None, embedded_entities=None,
                 stiffness="auto", stiffness_p=None,
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
        stiffness : float or ``"auto"``, default ``"auto"``
            Penalty stiffness ``K`` of the emitted
            ``ASDEmbeddedNodeElement``. ``"auto"`` resolves at emit
            from the host material (``K = α·E_host·L_char``, α = 1e3).
            **A numeric value is unit-dependent and must be calibrated
            against a known solution** — the pre-slice-B default
            ``1e18`` (the OpenSees C++ default) stalls Newton in
            N/mm/MPa models while ``1e10``–``1e12`` converge; emit
            still warns when a record carries 1e18. (Unlike
            :meth:`tie`, ``embedded`` has no ``enforce="equation"``
            escape hatch — the fork ``g.embed`` is the conditioned
            alternative.)
        stiffness_p : float, optional
            Separate rotational/pressure penalty (``-KP``); ``None`` ⇒
            falls back to ``K``. Same unit caveat.
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
                     stiffness="auto", stiffness_p=None,
                     rotational=False, pressure=False,
                     enforce="penalty", control=None,
                     name=None) -> TiedContactDef:
        """Full surface-to-surface tie (slave conforms to master).

        Every slave-surface node is tied to the master surface via
        shape-function interpolation. **One-directional**: an earlier
        bidirectional variant (also projecting master nodes onto slave
        faces) was removed because it produced cyclic / over-determined
        MPCs the constraint handler cannot satisfy. Pick the **finer**
        mesh as the master.

        ``enforce=`` selects the coupling route exactly as in :meth:`tie`
        (``"penalty"`` default → ``ASDEmbeddedNodeElement``; ``"equation"``
        → exact ``equationConstraint``; ``"penalty_al"`` →
        ``LadrunoEmbeddedNode``).

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
        stiffness : float or ``"auto"``, default ``"auto"``
            Penalty stiffness ``K`` of each emitted
            ``ASDEmbeddedNodeElement`` (penalty routes only).
            ``"auto"`` resolves at emit from the host material — see
            :meth:`tie` for the formula and the numeric-value caveat
            (a fixed number is unit-dependent; the old ``1e18`` default
            stalls Newton in N/mm/MPa and still warns at emit).
        stiffness_p : float, optional
            Separate rotational/pressure penalty (``-KP``); ``None`` ⇒
            falls back to ``K``.
        enforce : {"penalty", "penalty_al", "equation"}, default "penalty"
            Coupling route (ADR 0068) — same semantics as :meth:`tie`;
            ``"equation"`` emits exact ``equationConstraint`` rows and
            rejects the penalty-only knobs.
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
        mortar : Deprecated alias for a fork mortar mesh-tie
            (``contact(formulation="mortar", tie=True)``).
        """
        return self._add_def(TiedContactDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name,
            stiffness=stiffness, stiffness_p=stiffness_p,
            rotational=rotational, pressure=pressure, enforce=enforce,
            control=control))

    def mortar(self, master_label, slave_label, *,
               eps_n="auto", outward,
               master_entities=None, slave_entities=None,
               name=None) -> ContactDef:
        """Deprecated alias for a fork segment-to-segment mortar mesh-tie.

        .. deprecated::
           ``mortar()`` is a thin convenience alias for
           :meth:`contact` with ``formulation="mortar", tie=True``; call
           that directly. It emits a :class:`DeprecationWarning`.

        Delegates to the fork's ALM-penalty mortar **mesh-tie** (ADR 0073):
        a permanent segment-to-segment bond (the zero-gap limit — the full
        3-vector residual driven to zero, no friction). Returns a
        :class:`ContactDef` resolving to ``fem.elements.contacts`` (the fork
        ``contactSurface`` + ``contact -mortar -tie`` pair + the
        ``LadrunoContact`` handler), **not** the old ``MortarDef`` /
        Lagrange-multiplier path. Fork-only at run time; deck emission works on
        any build.

        This is a **breaking** change from the prior stub (which raised
        ``NotImplementedError``): the return type is now ``ContactDef``, the
        semantics are an ALM penalty contact-tie (not a Lagrange-multiplier
        operator), and the never-functional ``dofs`` / ``integration_order``
        parameters are removed (a permanent penalty tie bonds the full
        3-vector with a single penalty and has no DOF-subset or quadrature-order
        knob — passing them now raises ``TypeError``).

        Parameters
        ----------
        master_label, slave_label : str
            The two surface PG / part labels to bond. Both resolve to faceted
            surfaces (master ``-master``, slave ``-slave-segments``); pick the
            finer mesh as whichever side you trust more — the fork integrates
            the overlap either way.
        eps_n : float | "auto", default "auto"
            ALM normal penalty for the tie (``"auto"`` sizes it from the solid).
        outward : (float, float, float)
            **Required.** The master surface normal toward the slave. A tie
            interface is coincident-flat, so without an explicit sign the fork's
            per-pair reference is in-plane and gate H2 silently drops every pair
            to zero force (the tie would bond nothing). See :class:`ContactDef`.
        master_entities, slave_entities : list of (dim, tag), optional
            Restrict each side to specific Gmsh entities.
        name : str, optional
            Friendly name (round-trips into the emitted deck comment).

        Returns
        -------
        ContactDef

        See Also
        --------
        contact : The canonical fork contact / mortar-tie generator.
        tied_contact : Collocation-based non-matching tie (no fork required).
        """
        import warnings
        warnings.warn(
            "g.constraints.mortar() is a deprecated alias for "
            "g.constraints.contact(formulation='mortar', tie=True, ...); "
            "call contact() directly. mortar() now delegates to the fork "
            "ALM-penalty mortar mesh-tie (returns a ContactDef, not the old "
            "MortarDef Lagrange path) — see ADR 0073.",
            DeprecationWarning, stacklevel=2,
        )
        return self.contact(
            master_label, slave_label,
            formulation="mortar", tie=True,
            eps_n=eps_n, outward=outward,
            master_entities=master_entities, slave_entities=slave_entities,
            name=name,
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
        # weighting="area" distributing couplings consume slave faces too.
        has_face_constraints = has_face_constraints or any(
            isinstance(d, DistributingCouplingDef)
            and getattr(d, "weighting", "uniform") == "area"
            for d in self.constraint_defs)
        if (has_face_constraints and face_map is None
                and not self._session_can_resolve_names()):
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

        # Publish the MP lane's phantom-tag high-water mark so the
        # interface resolver (a different code path, same tag space —
        # ADR 0093 D4) mints above it instead of colliding with the
        # node_to_surface phantoms minted just now.
        self._phantom_tag_high_water = int(resolver._next_phantom_tag)

        self.constraint_records = records
        return ConstraintSet(records)

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------
    def _session_can_resolve_names(self) -> bool:
        """True when the session can resolve label / PG names to entities.

        A physical-group model has no ``face_map`` (no Parts) but can
        still resolve surface labels via ``g.physical`` / ``g.labels`` in
        :meth:`_resolve_faces`.  A bare stub session (no name composites)
        cannot, so the ``face_map=None`` guard in :meth:`resolve` still
        fires for it.
        """
        p = getattr(self, "_parent", None)
        return p is not None and (
            getattr(p, "labels", None) is not None
            or getattr(p, "physical", None) is not None
        )

    def _nodes_from_dimtags(self, selected, kind, role, *, source):
        """Union the mesh nodes of the given ``(dim, tag)`` entities.

        Shared by the ``{role}_entities`` path and the physical-group /
        label fallback in :meth:`_resolve_nodes` — a Gmsh failure or a
        zero-node result raises (a stale/wrong-dim reference must not be
        swallowed).
        """
        import gmsh
        tags: set[int] = set()
        for dim, tag in selected:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(dim), tag=int(tag),
                    includeBoundary=True, returnParametricCoord=False)
            except Exception as exc:
                raise ValueError(
                    f"{kind} {role}: cannot get mesh nodes for "
                    f"entity (dim={dim}, tag={tag}) from {source}: "
                    f"{exc}") from exc
            tags.update(int(t) for t in nt)
        if not tags:
            raise ValueError(
                f"{kind} {role}: {source} resolved to zero mesh nodes "
                f"— check the entities are meshed and of the intended "
                f"dimension.")
        return tags

    def _resolve_nodes(self, label, role, defn, node_map, all_nodes):
        """Resolve a constraint role to its mesh-node set — fail loud.

        Explicit ``{role}_entities`` win.  A labelled ``g.decouple_node``
        (ADR 0049 OQ2) resolves to the singleton ``{tag}`` *before* the
        Part ``node_map`` lookup so a Part bbox that swallows the knot
        cannot shadow the explicit name.  Otherwise the part ``label``'s
        node set is taken from ``node_map``; a *known* Part with an empty
        entry raises (a meshing error).  A label that is not a Part falls
        back to the shared label→PG resolver so physical-group models can
        constrain without Parts.  A missing/empty result always raises
        rather than silently binding every node in the model.
        """
        kind = type(defn).__name__
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            return self._nodes_from_dimtags(
                selected, kind, role, source=f"{role}_entities={selected!r}")
        dn = self._decoupled_def_for(label)
        if dn is not None:
            if dn.tag is None:
                raise ValueError(
                    f"{kind} {role}: decoupled node {label!r} has no "
                    f"resolved tag — internal ordering bug (factory must "
                    f"assign tags before constraint resolve)."
                )
            tag = int(dn.tag)
            # ``all_nodes`` may be a set, list, or ndarray of tags.
            pool = {int(t) for t in all_nodes}
            if tag not in pool:
                raise ValueError(
                    f"{kind} {role}: decoupled node {label!r} resolved "
                    f"to tag {tag}, which is absent from the extracted "
                    f"node pool — was get_fem_data called after "
                    f"g.decouple_node?"
                )
            return {tag}
        nodes = node_map.get(label)
        if nodes:
            return nodes
        parts = getattr(self._parent, "parts", None)
        if parts is not None and label in getattr(parts, "_instances", {}):
            raise ValueError(
                f"{kind} {role}: part label {label!r} contributed no "
                f"nodes to the constraint node map (is it meshed and "
                f"registered in g.parts?). Refusing to fall back to "
                f"all model nodes — a constraint must bind a known "
                f"node set; pass {role}_entities= to scope explicitly.")
        from ._helpers import resolve_to_dimtags
        try:
            dimtags = resolve_to_dimtags(
                label, default_dim=3, session=self._parent)
        except KeyError:
            raise ValueError(
                f"{kind} {role}: {label!r} is not a part, physical "
                f"group, or label — cannot resolve a node set. Tag it "
                f"(g.physical / .to_physical) or pass {role}_entities=."
            ) from None
        return self._nodes_from_dimtags(
            dimtags, kind, role, source=f"label {label!r}")

    def _faces_from_dimtags(self, selected, kind, role, *, source):
        """Collect surface faces for the given ``(dim, tag)`` entities.

        Shared by the ``{role}_entities`` path and the physical-group /
        label fallback in :meth:`_resolve_faces`.  dim=2 surfaces are
        used directly, dim=3 volumes contribute their boundary surfaces;
        dim 0/1 is a wrong-dimension reference and raises, as does an
        empty result.
        """
        bad = [(int(d), int(t)) for d, t in selected
               if int(d) not in (2, 3)]
        if bad:
            raise ValueError(
                f"{kind} {role}: {source} contains non-surface "
                f"entities {bad} — a face constraint requires dim=2 "
                f"surfaces (or dim=3 volumes, whose boundary surfaces "
                f"are used).")
        parts = getattr(self._parent, "parts", None)
        if parts is None or not hasattr(parts, "_collect_surface_faces"):
            raise ValueError(
                f"{kind} {role}: cannot resolve surface faces from "
                f"{source} — g.parts is unavailable.")
        faces = parts._collect_surface_faces(selected)
        if faces.size == 0:
            raise ValueError(
                f"{kind} {role}: {source} produced no surface mesh "
                f"faces — are the entities meshed?")
        return faces

    def _resolve_faces(self, label, role, defn, face_map):
        """Resolve a face-constraint role to its surface connectivity
        — fail loud.

        ``{role}_entities`` win.  Otherwise the part ``face_map`` is
        consulted; a *known* Part with no faces raises.  A label that is
        not a Part falls back to the shared label→PG resolver so
        physical-group models can tie without Parts.  An empty/missing
        result raises rather than silently dropping the constraint.
        """
        kind = type(defn).__name__
        selected = getattr(defn, f"{role}_entities", None)
        if selected:
            return self._faces_from_dimtags(
                selected, kind, role, source=f"{role}_entities={selected!r}")
        faces = face_map.get(label)
        if faces is not None and faces.size:
            return faces
        parts = getattr(self._parent, "parts", None)
        if parts is not None and label in getattr(parts, "_instances", {}):
            raise ValueError(
                f"{kind} {role}: part label {label!r} has no surface "
                f"faces in the face map (is its interface meshed and "
                f"registered in g.parts?). Refusing to silently drop "
                f"the constraint; pass {role}_entities= to scope it.")
        from ._helpers import resolve_to_dimtags
        try:
            dimtags = resolve_to_dimtags(
                label, default_dim=2, session=self._parent)
        except KeyError:
            raise ValueError(
                f"{kind} {role}: {label!r} is not a part, physical "
                f"group, or label — cannot resolve a surface. Tag the "
                f"surface (g.physical / .to_physical) or pass "
                f"{role}_entities=.") from None
        return self._faces_from_dimtags(
            dimtags, kind, role, source=f"label {label!r}")

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

    def _resolve_distributing(self, resolver, defn, node_map, face_map, all_nodes):
        m = self._resolve_nodes(defn.master_label, "master", defn, node_map, all_nodes)
        s = self._resolve_nodes(defn.slave_label, "slave", defn, node_map, all_nodes)
        # weighting="area" additionally needs the slave surface's face
        # connectivity for the tributary-area accumulation; "uniform"
        # stays node-set-only (no face_map requirement).
        s_faces = None
        if getattr(defn, "weighting", "uniform") == "area":
            s_faces = self._resolve_faces(defn.slave_label, "slave", defn, face_map)
        return resolver.resolve_distributing(defn, m, s, slave_face_conn=s_faces)

    def _resolve_face_slave(self, resolver, defn, node_map, face_map, all_nodes):
        m_faces = self._resolve_faces(defn.master_label, "master", defn, face_map)
        # ADR 0086 — method="mortar" integrates over BOTH facet sets
        # (segment-to-segment) instead of projecting slave nodes.
        if getattr(defn, "method", "collocation") == "mortar":
            s_faces = self._resolve_faces(defn.slave_label, "slave", defn, face_map)
            return resolver.resolve_tie_mortar(defn, m_faces, s_faces)
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
    ) -> np.ndarray:
        """Gather virtual tri3 / tet4 sub-element rows from *entities*.

        Build-phase thin adapter over
        :func:`apeGmsh._kernel.geometry._host_decomposition.decompose_hosts_to_subelements`
        (ADR 0041 §"Decision 1").  Pulls ``(etype, conn)`` pairs from
        the live gmsh session for each ``(dim, tag)`` entity, then
        delegates to the pure-numpy decomposition function shared with
        the chain-phase router.

        Returns connectivity rows that ``ASDEmbeddedNodeElement`` can
        consume directly: each row is 3 node tags (tri3 host) or 4
        node tags (tet4 host).  Per-etype dispatch table and
        invariants are documented on
        :func:`decompose_hosts_to_subelements`.

        Higher-order warning channel
        ----------------------------
        Fires one ``UserWarning`` per ``(etype, entity)`` the first
        time a higher-order host (tri6 / tet10 / quad8 / quad9 / hex20
        / prism15 / pyramid13) is decomposed — caller knows the
        embedded coupling is linear regardless of the host's native
        interpolation order.  Chain-phase variant phrases the message
        per ``(etype, target-label)`` since entity tags don't exist in
        chain phase (ADR 0041 §"Decision 8").
        """
        import gmsh
        import warnings as _warnings

        # Dispatch one decomposition call per entity so the
        # higher-order warning fires per ``(etype, entity)`` — the
        # ``decompose_hosts_to_subelements`` per-call dedup is then
        # naturally scoped to one entity, matching the pre-refactor
        # build-phase cadence.  Chain-phase callers (see
        # :class:`FEMDataSource.host_subelements_for`) pass one call
        # per target instead, getting per-(etype, target) cadence per
        # ADR 0041 §"Decision 8".
        warned_entity_codes: set[tuple[int, int, int]] = set()
        sub_arrays: list[np.ndarray] = []
        had_tri = False
        had_tet = False
        for dim, tag in entities:
            try:
                etypes, _, enodes = gmsh.model.mesh.getElements(
                    dim=int(dim), tag=int(tag))
            except Exception:
                continue
            d, t = int(dim), int(tag)
            entity_groups: list[tuple[int, np.ndarray]] = []
            for etype, nodes in zip(etypes, enodes):
                if len(nodes) == 0:
                    continue
                entity_groups.append(
                    (int(etype), np.asarray(nodes, dtype=int))
                )
            if not entity_groups:
                continue

            def _warn_for_entity(
                code: int, name: str, *, _d: int = d, _t: int = t,
            ) -> None:
                key = (code, _d, _t)
                if key in warned_entity_codes:
                    return
                warned_entity_codes.add(key)
                _warnings.warn(
                    f"embedded: host entity (dim={_d}, tag={_t}) "
                    f"carries {name} elements — decomposing to "
                    f"corner-node-only linear sub-elements.  The "
                    f"embedded coupling will be linear regardless of "
                    f"the host's native interpolation order; "
                    f"quadratic / bilinear / trilinear host kinematics "
                    f"will NOT be felt by the embedded node.  Set "
                    f"`host_coupling=` explicitly on the "
                    f"`embedded(...)` call to acknowledge.",
                    UserWarning, stacklevel=5,
                )

            sub = _decompose_hosts_to_subelements(
                entity_groups, warn_higher_order=_warn_for_entity,
            )
            if sub.size == 0:
                continue
            sub_arrays.append(sub)
            if sub.shape[1] == 3:
                had_tri = True
            elif sub.shape[1] == 4:
                had_tet = True

        # Mixed-dim host (shell PG + brick PG combined) is fail-loud:
        # an embedded node near the shell/brick interface would couple
        # to either side's corners depending on opaque kNN search
        # outcome — different physics, silently chosen.  Split the
        # host PG into two ``embedded`` constraints if you really
        # need both.
        if had_tet and had_tri:
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
        if not sub_arrays:
            return np.empty((0, 0), dtype=int)
        # When only one dim produced rows, vstack directly; mixed-dim
        # was already fail-loud above.
        return np.vstack(sub_arrays)

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
