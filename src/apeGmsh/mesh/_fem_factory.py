"""
_fem_factory — Factory functions for FEMData construction.
===========================================================

Implements ``FEMData.from_gmsh()`` and ``FEMData.from_msh()`` by
orchestrating the raw Gmsh extraction helpers from ``_fem_extract.py``
and splitting resolved records into node-side vs element-side
sub-composites.
"""

from __future__ import annotations

import logging

import numpy as np

from ._element_types import ElementGroup, make_type_info
from ._fem_extract import (
    extract_raw, extract_physical_groups, extract_labels,
    extract_partitions, gmsh_model_identity,
)
from ._group_set import PhysicalGroupSet, LabelSet

_log = logging.getLogger(__name__)


# =====================================================================
# Constraint splitting
# =====================================================================

def _split_constraints(records: list) -> tuple[list, list]:
    """Split resolved constraint records into node-level and surface-level."""
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord, InterfaceRecord,
    )

    node_recs = []
    surface_recs = []

    for rec in records:
        if isinstance(rec, (NodePairRecord, NodeGroupRecord,
                            NodeToSurfaceRecord)):
            node_recs.append(rec)
        elif isinstance(rec, (InterpolationRecord,
                              SurfaceCouplingRecord)):
            surface_recs.append(rec)
        elif isinstance(rec, InterfaceRecord):
            # ADR 0093 "Alternatives rejected" §"The _DISPATCH
            # MP-constraint lane": InterfaceRecord is an additive
            # side-list record (like ContactRecord) — it belongs on
            # ``fem.elements.interfaces``, never on
            # ``fem.nodes.constraints`` / ``fem.elements.constraints``.
            # The warn-and-fallback path below would silently place it
            # in the node bucket, where it writes a
            # ``/constraints/interface`` group that round-trips back
            # as 0 records (no dtype/decoder is registered for that
            # kind) — exactly the silent-drop class ADR 0093 S3 exists
            # to refuse. Fail loud instead: a caller that reaches this
            # function with an InterfaceRecord has a routing bug, not
            # a record this splitter can place.
            raise TypeError(
                f"_split_constraints: got an InterfaceRecord "
                f"(kind={rec.kind!r}) — interface records are a "
                f"side-list family (ADR 0093) and never route through "
                f"the _DISPATCH MP-constraint pipeline; they belong on "
                f"fem.elements.interfaces, populated by the ADR 0093 "
                f"S4 resolver, not by constraint resolution."
            )
        else:
            _log.warning(
                "Unknown constraint record type %s (kind=%r) — "
                "placed in node-level set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            node_recs.append(rec)

    return node_recs, surface_recs


# =====================================================================
# Load splitting
# =====================================================================

def _split_loads(records: list) -> tuple[list, list, list]:
    """Split resolved load records into nodal, element, and SP."""
    from apeGmsh._kernel.records._loads import NodalLoadRecord, ElementLoadRecord, SPRecord

    nodal = []
    element = []
    sp = []

    for rec in records:
        if isinstance(rec, NodalLoadRecord):
            nodal.append(rec)
        elif isinstance(rec, ElementLoadRecord):
            element.append(rec)
        elif isinstance(rec, SPRecord):
            sp.append(rec)
        else:
            _log.warning(
                "Unknown load record type %s (kind=%r) — "
                "placed in nodal set as fallback.",
                type(rec).__name__, getattr(rec, 'kind', '?'))
            nodal.append(rec)

    return nodal, element, sp


# =====================================================================
# Constraint-connected node collection
# =====================================================================

def _collect_constraint_nodes(
    node_constraints: list,
    surface_constraints: list,
    nodal_loads: list,
    sp_records: list,
    mass_records: list,
) -> set[int]:
    """Collect every mesh node ID referenced by resolved BCs."""
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord, NodeGroupRecord, NodeToSurfaceRecord,
        InterpolationRecord, SurfaceCouplingRecord,
    )

    ids: set[int] = set()

    for rec in node_constraints:
        if isinstance(rec, NodePairRecord):
            ids.add(rec.master_node)
            ids.add(rec.slave_node)
        elif isinstance(rec, NodeGroupRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)
        elif isinstance(rec, NodeToSurfaceRecord):
            ids.add(rec.master_node)
            ids.update(rec.slave_nodes)

    for rec in surface_constraints:
        if isinstance(rec, InterpolationRecord):
            ids.add(rec.slave_node)
            ids.update(rec.master_nodes)
        elif isinstance(rec, SurfaceCouplingRecord):
            ids.update(rec.master_nodes)
            ids.update(rec.slave_nodes)

    for rec in nodal_loads:
        ids.add(rec.node_id)
    for rec in sp_records:
        ids.add(rec.node_id)
    for rec in mass_records:
        ids.add(rec.node_id)

    return ids


# =====================================================================
# Build ElementGroup dict from raw groups
# =====================================================================

def _build_element_groups(raw_groups: dict[int, dict]) -> dict[int, ElementGroup]:
    """Convert raw extraction groups to ElementGroup objects."""
    result: dict[int, ElementGroup] = {}
    for etype_code, info in raw_groups.items():
        type_info = make_type_info(
            code=etype_code,
            gmsh_name=info['gmsh_name'],
            dim=info['dim'],
            order=info['order'],
            npe=info['npe'],
            count=len(info['ids']),
        )
        result[etype_code] = ElementGroup(
            element_type=type_info,
            ids=info['ids'],
            connectivity=info['conn'],
        )
    return result


def _flat_connectivity(groups: dict[int, ElementGroup]) -> np.ndarray:
    """Temporary flat connectivity for resolver kwargs.

    Resolvers receive this but never read it.  If types have
    different npe, pad shorter rows with -1.
    """
    if not groups:
        return np.empty((0, 0), dtype=np.int64)

    blocks = [g.connectivity for g in groups.values() if len(g) > 0]
    if not blocks:
        return np.empty((0, 0), dtype=np.int64)

    max_npe = max(b.shape[1] for b in blocks)
    padded = []
    for b in blocks:
        if b.shape[1] < max_npe:
            pad = np.full(
                (b.shape[0], max_npe - b.shape[1]), -1, dtype=np.int64)
            padded.append(np.hstack([b, pad]))
        else:
            padded.append(b)
    return np.vstack(padded)


def _flat_elem_tags(groups: dict[int, ElementGroup]) -> np.ndarray:
    """Concatenated element tags from all groups."""
    if not groups:
        return np.array([], dtype=np.int64)
    return np.concatenate([g.ids for g in groups.values()])


# =====================================================================
# Shared extraction core
# =====================================================================

def _extract_mesh_core(dim: int | None):
    """Shared extraction: raw arrays → element groups + PGs + labels.

    Returns
    -------
    tuple
        (node_tags, node_coords, elem_tags, groups,
         used_tags, physical, labels, partitions)
    """
    raw = extract_raw(dim=dim)

    node_tags   = raw['node_tags']
    node_coords = raw['node_coords']
    used_tags   = raw['used_tags']

    groups = _build_element_groups(raw['groups'])
    elem_tags = _flat_elem_tags(groups)

    physical = PhysicalGroupSet(extract_physical_groups())
    labels   = LabelSet(extract_labels())
    partitions = extract_partitions(dim)

    return (node_tags, node_coords, elem_tags, groups,
            used_tags, physical, labels, partitions)


# =====================================================================
# from_gmsh
# =====================================================================

def _from_gmsh(
    cls,
    *,
    dim: int | None,
    session=None,
    ndf: int = 6,
    remove_orphans: bool = False,
):
    """Build a FEMData from the live Gmsh session.

    Parameters
    ----------
    cls : type
        The FEMData class.
    dim : int or None
        Element dimension to extract.  None = all dims.
    session : apeGmsh session, optional
        Provides constraints, loads, masses composites.
    ndf : int
        DOFs per node for load/mass padding.
    remove_orphans : bool
        If True, remove orphan nodes.  Default False.
    """
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )

    # ── 1. Extract ────────────────────────────────────────────
    (node_tags, node_coords, elem_tags, groups,
     used_tags, physical, labels, partitions) = _extract_mesh_core(dim)
    # The model just read: a raw (dim, tag) selection on the snapshot
    # asks Gmsh only while this model is still the current one.
    gmsh_source = gmsh_model_identity()

    node_ids = np.asarray(node_tags, dtype=int)
    node_coords_all = np.asarray(node_coords, dtype=float)

    # ── 1b. Inject decoupled nodes (ADR 0049) ─────────────────
    # Auxiliary nodes declared via ``g.decouple_node(...)`` are NOT
    # Gmsh vertices.  Append them to the node pool *here* — before
    # constraint resolution — so the constraint resolver's phantom-tag
    # base (``max(node_ids) + 1``) lands strictly above every decoupled
    # tag, making the two synthetic ranges disjoint by construction.
    # Their tags come from ``getMaxNodeTag() + i`` (rank-invariant, so
    # MP-deterministic) and they fold into the snapshot_id via _ids +
    # _coords for free.
    decoupled_tags, decoupled_coords = _resolve_decoupled_nodes(session)
    # Decoupled node *sets* (ADR 0118 D1) continue the same tag range,
    # resolved by name against the mesh nodes just extracted.
    set_tags, set_coords, set_ties = _resolve_decoupled_node_sets(
        session, node_ids, node_coords_all, physical, labels,
        first_tag=(max(decoupled_tags) + 1 if decoupled_tags else None),
    )
    decoupled_tags = list(decoupled_tags) + set_tags
    decoupled_coords = list(decoupled_coords) + set_coords
    decoupled_set: set[int] = set(int(t) for t in decoupled_tags)
    if decoupled_tags:
        dt = np.asarray(decoupled_tags, dtype=int)
        dc = np.asarray(decoupled_coords, dtype=float).reshape(-1, 3)
        node_ids = np.concatenate([node_ids, dt])
        node_coords_all = (
            np.vstack([node_coords_all, dc])
            if node_coords_all.size else dc
        )
        # Keep the raw orphan-filter pool in lockstep with the broker
        # pool (orphan filtering rebuilds node_ids from node_tags).
        node_tags = node_ids
        node_coords = node_coords_all

    # ── 2. Resolve BCs ────────────────────────────────────────
    node_constraints: list = []
    surface_constraints: list = []
    nodal_loads: list = []
    element_loads: list = []
    sp_records: list = []
    mass_records: list = []
    reinforce_ties: list = []
    embed_ties: list = []
    contacts: list = []
    contact_planes: list = []
    rebar_elements: list = []
    # g.constraints.interface() (ADR 0093 S4) — resolved below, after
    # the MP-constraint pass (see the phantom-tag note at that call).
    interfaces: list = []

    if session is not None:
        parts_comp = getattr(session, "parts", None)
        node_map = None
        face_map = None
        if (parts_comp is not None
                and getattr(parts_comp, "_instances", None)):
            # No broad swallow: a node/face-map build failure must
            # surface with its real cause rather than degrade to
            # None and resurface later as a vaguer constraint error.
            node_map = parts_comp.build_node_map(
                node_ids, node_coords_all)
            face_map = parts_comp.build_face_map(node_map)

        # Build temp flat connectivity for resolver kwargs
        flat_conn = _flat_connectivity(groups)
        resolve_kw = dict(
            elem_tags=elem_tags,
            connectivity=flat_conn,
            node_map=node_map,
            face_map=face_map,
        )

        # Constraints / loads / masses.
        #
        # These resolve() calls are deliberately NOT wrapped in a
        # broad ``except Exception: log.warning`` swallow.  The
        # resolvers raise precise, actionable ValueError/KeyError when
        # a reference is wrong-dimension, multi-dim, unresolved, or
        # would otherwise silently bind the wrong node/face set.  A
        # structural model that silently drops a tie / load / mass is
        # worse than one that errors — get_fem_data() must fail loud
        # so the user fixes the model, not discover it post-analysis.
        constraints_comp = getattr(session, "constraints", None)
        if (constraints_comp is not None
                and getattr(constraints_comp, "constraint_defs", None)):
            all_constraints = constraints_comp.resolve(
                node_ids, node_coords_all, **resolve_kw)
            node_constraints, surface_constraints = \
                _split_constraints(all_constraints)
        # The equal_dof ties of g.decouple_node_set(tie_dofs=...)
        # (ADR 0118 D1): plain broker MP records, emitted like any other.
        if set_ties:
            node_constraints = list(node_constraints) + set_ties

        # Embedded reinforcement (g.reinforce, ADR 20 / R2b). Pure
        # geometry like embedded(): each rebar-PG node is inverse-mapped
        # into its non-matching solid host, producing one
        # ReinforceTieRecord per node. Fail-loud (a stray rebar node, an
        # empty host, an unsupported host kind all raise) — same policy as
        # the constraint resolve above.
        reinforce_comp = getattr(session, "reinforce", None)
        if (reinforce_comp is not None
                and getattr(reinforce_comp, "reinforce_defs", None)):
            reinforce_ties = reinforce_comp.resolve(
                node_ids, node_coords_all)

        # General node-to-host embedment (g.embed). Isotropic sibling of
        # reinforcement: each node of the constrained set is inverse-mapped
        # into its non-matching solid host, producing one EmbedTieRecord per
        # node. Same fail-loud policy.
        embed_comp = getattr(session, "embed", None)
        if (embed_comp is not None
                and getattr(embed_comp, "embed_defs", None)):
            embed_ties = embed_comp.resolve(
                node_ids, node_coords_all)

        # Face-to-face contact (g.constraints.contact). The contact defs live
        # on the constraints composite but resolve to an additive list (like
        # reinforce/embed), not the MP-constraint dispatch.
        if (constraints_comp is not None
                and getattr(constraints_comp, "contact_defs", None)):
            contacts = constraints_comp.resolve_contacts(
                node_ids, node_coords_all)
        # Rigid analytical-plane contact (g.constraints.contact_plane) —
        # additive sibling of the face-to-face contact above.
        if (constraints_comp is not None
                and getattr(constraints_comp, "contact_plane_defs", None)):
            contact_planes = constraints_comp.resolve_contact_planes(
                node_ids, node_coords_all)
        # Oriented coincident-pair zeroLength interfaces
        # (g.constraints.interface, ADR 0093 S4) — additive side-list
        # like the two contact resolves above.
        #
        # Ordering is load-bearing, not incidental: this call MUST come
        # after ``constraints_comp.resolve()`` (the MP pass, above).
        # Both lanes mint phantom nodes from the same integer space —
        # the MP lane's ConstraintResolver starts at max(node_tags)+1
        # (``_next_phantom_tag``) and the interface resolver mints its
        # D4 mixed-ndf bridges from a different code path — so the MP
        # pass publishes its final high-water mark onto the composite
        # and resolve_interfaces() starts strictly above it. Reordering
        # these two calls re-opens the collision.
        if (constraints_comp is not None
                and getattr(constraints_comp, "interface_defs", None)):
            interfaces = constraints_comp.resolve_interfaces(
                node_ids, node_coords_all)
        # Structural rebar elements (ADR 0067 P5.2 / B1): the cage's
        # auto-emit bars from g.rebar.place(emit_elements=True). resolve()
        # extracts each bar's line-cell connectivity from the live mesh (the
        # dim-1 cells are dropped from a dim-3 FEMData) → one
        # RebarElementRecord per bar. Fail-loud (an unmeshed bar raises).
        rebar_comp = getattr(session, "rebar", None)
        if (rebar_comp is not None
                and getattr(rebar_comp, "_emit_members", None)):
            rebar_elements = rebar_comp.resolve()

        loads_comp = getattr(session, "loads", None)
        if (loads_comp is not None
                and getattr(loads_comp, "load_defs", None)):
            all_loads = loads_comp.resolve(
                node_ids, node_coords_all, **resolve_kw)
            nodal_loads, element_loads, sp_records = _split_loads(all_loads)

        # Single-point constraints declared via g.constraints.bc().
        # Kept separate from constraints_comp.resolve() above: BCDefs
        # have no master/slave and resolve to homogeneous SPRecords in
        # fem.nodes.sp, not to fem.nodes.constraints.  Independent of
        # the constraint_defs guard so a BC-only model still resolves.
        if (constraints_comp is not None
                and getattr(constraints_comp, "_bc_defs", None)):
            sp_records.extend(
                constraints_comp.resolve_bcs(
                    node_ids, node_map=node_map))

        # Prescribed displacements declared via g.displacements (ADR 0050).
        # Like g.constraints.bc, these resolve to SPRecords on
        # fem.nodes.sp — but carry nonzero/pattern-bound values.
        disp_comp = getattr(session, "displacements", None)
        if (disp_comp is not None
                and getattr(disp_comp, "disp_defs", None)):
            sp_records.extend(
                disp_comp.resolve(node_ids, node_coords_all, **resolve_kw))

        masses_comp = getattr(session, "masses", None)
        if (masses_comp is not None
                and getattr(masses_comp, "mass_defs", None)):
            mass_records = masses_comp.resolve(
                node_ids, node_coords_all, **resolve_kw)

    # ── 3. Orphan filtering ───────────────────────────────────
    if remove_orphans:
        protected = _collect_constraint_nodes(
            node_constraints, surface_constraints,
            nodal_loads, sp_records, mass_records,
        )
        # Decoupled nodes are intentional and never element-attached;
        # protect them so orphan filtering can't drop them (ADR 0049).
        protected.update(decoupled_set)
        node_ids, node_coords_all = _filter_orphans(
            node_tags, node_coords, used_tags, protected)

    # ── 4. Build MeshInfo ─────────────────────────────────────
    type_list = [g.element_type for g in groups.values()]
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=int(sum(len(g) for g in groups.values())),
        bandwidth=_compute_bandwidth(groups),
        types=type_list,
    )

    # ── 5. Build composites ───────────────────────────────────
    # If the session carries a parts registry, snapshot its
    # label -> {mesh-node-ids} and label -> {mesh-element-ids} maps
    # now so fem.nodes.select(target=part_label) and
    # fem.elements.select(target=part_label) can resolve without
    # needing a live Gmsh session later.
    part_node_map: dict[str, set[int]] = {}
    part_elem_map: dict[str, set[int]] = {}
    if session is not None:
        parts = getattr(session, "parts", None)
        if parts is not None and getattr(parts, "_instances", None):
            import gmsh  # local import — gmsh is alive during factory
            # Fail loud, mirroring the constraint-path policy above
            # (the same build_node_map call there is deliberately not
            # swallowed): a part-map build failure must surface with
            # its real cause, not degrade to an empty map that
            # resurfaces later as a misleading "part not found".
            part_node_map = parts.build_node_map(
                node_ids, node_coords_all,
            ) or {}
            # Element map: iterate each part instance's DimTags and
            # ask Gmsh for the elements on each entity (the registry
            # has no element-map builder today).
            #
            # inst.entities can hold tags that fragment_all() / boolean
            # ops have retagged out of existence (skill pitfall 7.6 —
            # OCC renumbers entities; the coordinate-based node map is
            # the robust contract).  Skip absent entities *explicitly*
            # by pre-filtering against the live model — not via a
            # blanket ``except`` — so a genuine getElements failure on
            # an entity the model DOES have still fails loud.
            present = set(gmsh.model.getEntities())
            for label, inst in parts._instances.items():
                e_ids: set[int] = set()
                for d in sorted(inst.entities.keys(), reverse=True):
                    for t in inst.entities[d]:
                        if (int(d), int(t)) not in present:
                            continue
                        _, etags_list, _ = gmsh.model.mesh.getElements(
                            int(d), int(t))
                        for arr in etags_list:
                            e_ids.update(int(x) for x in arr)
                if e_ids:
                    part_elem_map[label] = e_ids

    # Per-node provenance (ADR 0049) — computed from the *final*
    # node_ids (post orphan-filter, which may have reordered/dropped
    # rows) by membership test, so it's robust to any reshuffle.
    # ``None`` when there are no decoupled nodes keeps the snapshot_id
    # + H5 bytes identical to a model without the feature.
    node_provenance = _build_provenance(node_ids, decoupled_set)

    nodes = NodeComposite(
        node_ids=node_ids,
        node_coords=node_coords_all,
        physical=physical,
        labels=labels,
        constraints=node_constraints or None,
        loads=nodal_loads or None,
        sp=sp_records or None,
        masses=mass_records or None,
        partitions=partitions or None,
        part_node_map=part_node_map or None,
        provenance=node_provenance,
        gmsh_source=gmsh_source,
    )
    elements = ElementComposite(
        groups=groups,
        physical=physical,
        labels=labels,
        constraints=surface_constraints or None,
        loads=element_loads or None,
        partitions=partitions or None,
        part_elem_map=part_elem_map or None,
        reinforce_ties=reinforce_ties or None,
        embed_ties=embed_ties or None,
        contacts=contacts or None,
        contact_planes=contact_planes or None,
        rebar_elements=rebar_elements or None,
        interfaces=interfaces or None,
        gmsh_source=gmsh_source,
    )

    # ── 6. Snapshot mesh selections ───────────────────────────
    ms_store = None
    if session is not None:
        ms_comp = getattr(session, "mesh_selection", None)
        if ms_comp is not None and len(ms_comp) > 0:
            # Fail loud (see the note above): a snapshot failure must
            # not silently drop every mesh-selection set and resurface
            # later as a misleading "selection not found".
            ms_store = ms_comp._snapshot()

    return cls(
        nodes=nodes,
        elements=elements,
        info=info,
        mesh_selection=ms_store,
    )


# =====================================================================
# Orphan filtering
# =====================================================================

def _filter_orphans(
    node_tags: np.ndarray,
    node_coords: np.ndarray,
    used_tags: set[int],
    protected: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove orphan nodes, optionally protecting some."""
    keep = np.isin(node_tags, list(used_tags))

    if protected:
        also_keep = np.isin(node_tags, list(protected))
        keep = keep | also_keep

    orphan_mask = ~keep
    n_orphans = int(orphan_mask.sum())
    if n_orphans > 0:
        orphan_tags = node_tags[orphan_mask]
        orphan_coords = node_coords[orphan_mask]
        detail = ", ".join(
            f"{int(t)} ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"
            for t, c in zip(orphan_tags[:20], orphan_coords[:20])
        )
        _log.warning(
            "%d orphan node(s) removed (not connected to any element). "
            "First: [%s]%s",
            n_orphans,
            detail,
            f" ... (+{n_orphans - 20} more)" if n_orphans > 20 else "")

    node_ids = np.asarray(node_tags[keep], dtype=int)
    node_coords_filtered = node_coords[keep]
    return node_ids, node_coords_filtered


# =====================================================================
# Decoupled nodes (ADR 0049)
# =====================================================================

def _resolve_decoupled_nodes(session) -> tuple[list[int], list[tuple]]:
    """Resolve the session's decoupled-node defs to (tags, coords).

    Assigns each def a deterministic tag ``getMaxNodeTag() + i + 1``
    (rank-invariant → MP-deterministic) and snapshots any ``point=``
    label to its coordinates.  Writes the resolved tag back onto each
    def so the handle returned by ``g.decouple_node(...)`` exposes it.

    Returns ``([], [])`` when the session has no decoupled nodes.
    """
    if session is None:
        return [], []
    comp = getattr(session, "decoupled_nodes", None)
    defs = getattr(comp, "node_defs", None) if comp is not None else None
    if not defs:
        return [], []

    import gmsh
    base = int(gmsh.model.mesh.getMaxNodeTag())

    tags: list[int] = []
    coords: list[tuple] = []
    for i, defn in enumerate(defs):
        xyz = defn.coords
        if xyz is None:
            xyz = _snapshot_point_coords(session, defn.point)
        tag = base + i + 1
        defn.tag = tag
        tags.append(tag)
        coords.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
    return tags, coords


def _resolve_decoupled_node_sets(
    session, node_ids, node_coords, physical, labels, *,
    first_tag: "int | None",
) -> "tuple[list[int], list[tuple], list]":
    """Resolve the session's decoupled node *sets* (ADR 0118 D1).

    Each :class:`~apeGmsh._kernel.defs.decoupled.DecoupledNodeSetDef`
    names a label or physical group; its mesh nodes (ascending tags)
    are the source nodes, and one new node is placed at each source
    node plus the def's offset.  Tags continue from ``first_tag`` (the
    tag after the last single decoupled node) or ``getMaxNodeTag() + 1``,
    so the range stays rank-invariant.  Writes ``source_ids`` / ``tags``
    back onto each def and returns ``(tags, coords, equal_dof records)``.
    """
    if session is None:
        return [], [], []
    comp = getattr(session, "decoupled_nodes", None)
    defs = getattr(comp, "node_set_defs", None) if comp is not None else None
    if not defs:
        return [], [], []

    from apeGmsh._kernel.records._constraints import NodePairRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind

    if first_tag is None:
        import gmsh
        first_tag = int(gmsh.model.mesh.getMaxNodeTag()) + 1
    row_of = {int(t): i for i, t in enumerate(np.asarray(node_ids, dtype=int))}
    xyz_all = np.asarray(node_coords, dtype=float).reshape(-1, 3)

    tags: list[int] = []
    coords: list[tuple] = []
    ties: list = []
    next_tag = int(first_tag)
    for defn in defs:
        src = _decoupled_set_source(defn, physical, labels)
        missing = [t for t in src if t not in row_of]
        if missing:
            raise ValueError(
                f"g.decouple_node_set({defn.source!r}): source nodes "
                f"{missing[:5]} are not in the extracted node pool."
            )
        xyz = xyz_all[[row_of[t] for t in src]]
        if callable(defn.offset):
            off = np.asarray(defn.offset(xyz.copy()), dtype=float)
            if off.shape != xyz.shape or not np.all(np.isfinite(off)):
                raise ValueError(
                    f"g.decouple_node_set({defn.source!r}): the offset "
                    f"callable must return finite offsets of shape "
                    f"{xyz.shape}; got shape {off.shape}."
                )
        else:
            off = np.broadcast_to(
                np.asarray(defn.offset, dtype=float), xyz.shape)
        new_xyz = xyz + off
        new_tags = list(range(next_tag, next_tag + len(src)))
        next_tag += len(src)
        defn.source_ids = tuple(int(t) for t in src)
        defn.tags = tuple(new_tags)
        tags.extend(new_tags)
        coords.extend(tuple(float(v) for v in row) for row in new_xyz)
        if defn.tie_dofs:
            for s_tag, n_tag in zip(defn.source_ids, defn.tags):
                ties.append(NodePairRecord(
                    kind=ConstraintKind.EQUAL_DOF,
                    name=defn.label,
                    master_node=s_tag,
                    slave_node=n_tag,
                    dofs=list(defn.tie_dofs),
                ))
    return tags, coords, ties


def _decoupled_set_source(defn, physical, labels) -> list[int]:
    """The ascending mesh-node tags of a set's ``source`` name: a label
    first, then a physical group (the session's resolution order)."""
    name = defn.source
    for groups in (labels, physical):
        if groups is not None and name in groups:
            ids = sorted({int(t) for t in groups.node_ids(name)})
            if ids:
                return ids
    raise ValueError(
        f"g.decouple_node_set: source {name!r} names no label or physical "
        f"group with mesh nodes in this extraction."
    )


def _snapshot_point_coords(session, point_label: str) -> tuple:
    """Resolve a geometry point label to its (x, y, z) coordinates.

    Snapshotted at mesh-extraction time (not tracked through later
    transforms).  Fails loud if the label doesn't resolve to exactly
    one dim-0 entity.
    """
    from apeGmsh.core._resolution import resolve_target
    import gmsh

    dts = resolve_target(
        session, point_label, "auto",
        expected_dim=0,
        not_found_prefix="Decoupled-node point",
        noun="decoupled node",
    )
    pts = [(int(d), int(t)) for d, t in dts if int(d) == 0]
    if len(pts) != 1:
        raise ValueError(
            f"g.decouple_node(point={point_label!r}) must resolve to "
            f"exactly one geometry point; resolved to {len(pts)} dim-0 "
            f"entit{'y' if len(pts) == 1 else 'ies'} ({pts})."
        )
    return tuple(gmsh.model.getValue(0, pts[0][1], []))


def _build_provenance(node_ids: np.ndarray, decoupled_set: set[int]):
    """Build the int8 provenance array, or ``None`` if no decoupled nodes.

    ``None`` (the no-decoupled-nodes case) keeps the snapshot_id hash
    and the persisted H5 bytes identical to a model without the feature.
    """
    if not decoupled_set:
        return None
    from .FEMData import PROVENANCE_DECOUPLED, PROVENANCE_MESH
    prov = np.where(
        np.isin(np.asarray(node_ids, dtype=int), list(decoupled_set)),
        np.int8(PROVENANCE_DECOUPLED), np.int8(PROVENANCE_MESH),
    ).astype(np.int8)
    return prov


# =====================================================================
# from_msh
# =====================================================================

def _from_msh(
    cls,
    *,
    path: str,
    dim: int | None = 2,
    remove_orphans: bool = False,
):
    """Build a FEMData from an external ``.msh`` file."""
    import gmsh
    from .FEMData import (
        NodeComposite, ElementComposite, MeshInfo, _compute_bandwidth,
    )
    from .._session import _gmsh_acquire, _gmsh_release

    _gmsh_acquire()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(path))

        (node_tags, node_coords, elem_tags, groups,
         used_tags, physical, labels, partitions) = _extract_mesh_core(dim)
        gmsh_source = gmsh_model_identity()

        if remove_orphans:
            node_ids, node_coords = _filter_orphans(
                node_tags, node_coords, used_tags)
        else:
            node_ids = np.asarray(node_tags, dtype=int)

        type_list = [g.element_type for g in groups.values()]
        info = MeshInfo(
            n_nodes=len(node_ids),
            n_elems=int(sum(len(g) for g in groups.values())),
            bandwidth=_compute_bandwidth(groups),
            types=type_list,
        )

        # Per-node ndf — leave at ``None``, as ``from_gmsh`` does: per-node
        # ndf is inferred by the bridge from the declared elements
        # (ADR 0048), so the broker carries no ndf channel and the
        # ``_femdata_hash`` fold skips it on both paths.
        nodes = NodeComposite(
            node_ids=node_ids, node_coords=node_coords,
            physical=physical, labels=labels,
            partitions=partitions or None,
            gmsh_source=gmsh_source,
        )
        elements = ElementComposite(
            groups=groups,
            physical=physical, labels=labels,
            partitions=partitions or None,
            gmsh_source=gmsh_source,
        )
    finally:
        _gmsh_release()

    return cls(nodes=nodes, elements=elements, info=info)
