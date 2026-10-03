"""STKO mesh -> apeGmsh session, mesh-faithful (ADR 0111 D1).

:func:`build_mesh` puts STKO's own mesh into the session through gmsh's
discrete-entity API, with STKO's node and element tags: one discrete
entity per referenced STKO sub-shape (split by local axis, rule M6),
physical groups as unions of those entities, and point "carrier"
entities that own each rigid-diaphragm master's and slaves' nodes.
Nothing is meshed or renumbered; the caller goes straight to
``g.mesh.queries.get_fem_data(dim=None)``.

Rules M1-M6 and work package M of ``internal_docs/stko_translator_rules.md``.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from .model import ELEMENT_TYPES, KINDS, ScdModel
from .translate_conditions import diaphragm_groups
from .translate_types import (
    KIND_DIM,
    ElementGroup,
    MeshMap,
    SubShape,
    Unsupported,
    UnsupportedSTKOTypes,
    Vec3,
    local_axis,
    quat_matrix,
)

__all__ = ["check_supported", "build_mesh", "session_nodes", "shell_node_order"]

#: STKO mesh element type -> gmsh element type (line2, quad4).
GMSH_TYPE: dict[int, int] = {102: 1, 303: 3}
#: gmsh type of the point elements on vertices and diaphragm carriers.
_POINT = 15
#: STKO's interaction-generated link element.
_LINK = 600
#: Mesh element types that belong to tiers this version does not translate.
TIER_ONLY_TYPES: frozenset[int] = frozenset({401, 500})

_RIGID_DIAPHRAGM = "Constraints.mp.rigidDiaphragm"
_TIER_ONLY = "tier-only: not in this version"
_UNKNOWN = "unknown STKO type"


# ── rule M3 ───────────────────────────────────────────────────────────

def shell_node_order(scd: ScdModel, eid: int) -> tuple[int, ...]:
    """Rule M3 (STKO ``shell_utils.getNodeString``): the node order STKO
    writes for a 4-node shell.

    With local x ``ux`` (column 0 of the element quaternion) and
    ``vx = normalize(p1 + p2 - p3 - p0)``, keep ``(n0, n1, n2, n3)`` when
    ``|ux . vx| > 0.5``, else write ``(n1, n2, n3, n0)``. Any other
    element is returned as stored.
    """
    nodes = scd.mesh.elements[eid].nodes
    if len(nodes) != 4:
        return nodes
    p = np.array([scd.mesh.node_xyz(n) for n in nodes], dtype=float)
    vx = p[1] + p[2] - p[3] - p[0]
    vx = vx / np.linalg.norm(vx)
    ux = quat_matrix(scd.mesh.orientation[eid])[:, 0]
    if abs(float(ux @ vx)) > 0.5:
        return nodes
    return (nodes[1], nodes[2], nodes[3], nodes[0])


# ── what the mesh references ──────────────────────────────────────────

def _expand(scd: ScdModel, items: dict[int, Any], whole: Iterable[int]) -> list[SubShape]:
    """Sub-shapes of a ``{geometry: SubShapes}`` assignment, ``whole``
    geometries expanded to every sub-shape they have."""
    out: list[SubShape] = []
    for gid, sub in items.items():
        if gid in whole:
            continue
        for kind in KINDS:
            out.extend((gid, kind, int(i)) for i in sub.of(kind))
    for gid in whole:
        counts = scd.geometries[gid].counts
        for kind in KINDS:
            out.extend((gid, kind, i) for i in range(counts.get(kind, 0)))
    return out


def _set_subshapes(scd: ScdModel) -> dict[str, list[SubShape]]:
    return {
        name: _expand(scd, dict(s.items), s.whole)
        for name, s in scd.selection_sets.items()
    }


def _referenced(scd: ScdModel, analysis: dict[int, Any],
                sets: dict[str, list[SubShape]]) -> list[SubShape]:
    """Step 2: analysis-element sub-shapes, condition sub-shapes and
    selection-set items, in node-ownership order (vertices, edges,
    faces, solids)."""
    ref: set[SubShape] = set()
    for a in analysis.values():
        if a.geometry is not None:
            ref.add((a.geometry, a.kind, a.index))
    for c in scd.conditions.values():
        ref.update(_expand(scd, c.geometry, ()))
    for subs in sets.values():
        ref.update(subs)
    return sorted(ref, key=lambda s: (KIND_DIM[s[1]], s[0], s[2]))


def _domain(scd: ScdModel, sub: SubShape) -> tuple[int, ...]:
    gid, kind, idx = sub
    return tuple(int(e) for e in scd.mesh.domains.get((gid, kind), {}).get(idx, ()))


def _diaphragm_interactions(scd: ScdModel) -> dict[int, list[int]]:
    """Interaction id -> the condition ids that use it."""
    users: dict[int, list[int]] = defaultdict(list)
    for c in scd.conditions.values():
        for iid in c.interactions:
            users[iid].append(c.id)
    return users


# ── the registry slice ────────────────────────────────────────────────

def check_supported(scd: ScdModel) -> list[Unsupported]:
    """The mesh slice of the registry (rules doc section 4).

    - every interaction: an ``NN`` interaction used only by
      ``Constraints.mp.rigidDiaphragm`` conditions and carrying no element
      property is supported; other ``NN`` / ``NE`` are tier-only; any other
      interaction type is unknown;
    - the mesh element types of the analysis elements on sub-shapes
      (``line2``, ``quad4``; ``hex8`` is tier-only; anything else, a link
      element on a sub-shape included, is unknown), listed with the
      element properties that carry them;
    - the mesh element types of the non-analysis elements on referenced
      sub-shapes (the carriers), listed with their geometries;
    - the node-subset guard (:func:`_node_guard`): the session's nodes must be
      the analysis elements' nodes plus the referenced diaphragms' nodes, and
      those must be every node of the mesh (STKO writes them all).
    """
    out: list[Unsupported] = []
    users = _diaphragm_interactions(scd)
    for inter in sorted(scd.interactions.values(), key=lambda i: i.id):
        cids = users.get(inter.id, [])
        ok = (
            inter.type == "NN" and bool(cids) and inter.element_property == 0
            and all(scd.conditions[c].type == _RIGID_DIAPHRAGM for c in cids)
        )
        if not ok:
            out.append(Unsupported(
                category="interaction", xobj_meta=inter.type, ids=(inter.id,),
                names=(inter.name,),
                reason=_TIER_ONLY if inter.type in ("NN", "NE") else _UNKNOWN,
            ))

    analysis = scd.analysis_elements()
    by_type: dict[int, set[int]] = defaultdict(set)
    for a in analysis.values():
        if a.interaction is None and a.element.type not in GMSH_TYPE:
            by_type[a.element.type].add(a.element_property)
    for code in sorted(by_type):
        eps = sorted(by_type[code])
        out.append(Unsupported(
            category="mesh_element", xobj_meta=_type_meta(code), ids=tuple(eps),
            names=tuple(_name(scd.element_properties, p) for p in eps),
            reason=_TIER_ONLY if code in TIER_ONLY_TYPES else _UNKNOWN,
        ))

    carriers: dict[int, set[int]] = defaultdict(set)
    for sub in _referenced(scd, analysis, _set_subshapes(scd)):
        if sub[1] == "vertices":
            continue
        for e in _domain(scd, sub):
            code = scd.mesh.elements[e].type
            if e not in analysis and code not in GMSH_TYPE:
                carriers[code].add(sub[0])
    for code in sorted(carriers):
        gids = sorted(carriers[code])
        out.append(Unsupported(
            category="mesh_element", xobj_meta=_type_meta(code) + " carrier",
            ids=tuple(gids), names=tuple(scd.geometries[g].name for g in gids),
            reason=_TIER_ONLY if code in TIER_ONLY_TYPES else _UNKNOWN,
        ))
    out.extend(_node_guard(scd, analysis))
    return out


def session_nodes(scd: ScdModel) -> frozenset[int]:
    """The nodes :func:`build_mesh` puts in the session (pure, no gmsh): the
    diaphragm groups' masters and slaves, the nodes of every vertex referenced,
    and every node of every mesh element on a referenced edge, face or solid
    (analysis element or not)."""
    analysis = scd.analysis_elements()
    nodes: set[int] = set()
    for d in diaphragm_groups(scd):
        nodes.add(d.master_node)
        nodes.update(d.slave_nodes)
    for gid, kind, idx in _referenced(scd, analysis, _set_subshapes(scd)):
        if kind == "vertices":
            vn = scd.mesh.vertex_nodes.get(gid)
            if vn is not None and idx < len(vn):
                nodes.add(int(vn[idx]))
            continue
        for e in _domain(scd, (gid, kind, idx)):
            nodes.update(int(n) for n in scd.mesh.elements[e].nodes)
    return frozenset(nodes)


def _node_guard(scd: ScdModel, analysis: dict[int, Any]) -> list[Unsupported]:
    """Every session node becomes an OpenSees node, and STKO writes every mesh
    node (``write_node.write_node_not_assigned`` writes the ones no element uses,
    with ``ndf 3``). So the session must hold exactly the nodes of the analysis
    elements (interaction links of the referenced diaphragms included) and those
    must be all the mesh's nodes:

    - a session node outside that set (a condition or selection set on a
      sub-shape whose elements carry no element property) would be a free node
      in the translated deck that STKO's deck does not have as such;
    - a mesh node outside the session (an unassigned meshed geometry, or the
      links of an interaction no constraint pattern references) is a node
      STKO writes that the translated deck would not have.
    """
    out: list[Unsupported] = []
    model: set[int] = set()
    for a in analysis.values():
        if a.interaction is None:
            model.update(int(n) for n in a.element.nodes)
    for d in diaphragm_groups(scd):
        model.add(d.master_node)
        model.update(d.slave_nodes)
    session = session_nodes(scd)
    extra = sorted(session - model)
    if extra:
        out.append(Unsupported(
            category="option", xobj_meta="mesh:session nodes outside the analysis model",
            ids=tuple(extra[:10]), names=(f"{len(extra)} nodes",),
            reason=(f"{len(extra)} node(s) of referenced sub-shapes (conditions, selection "
                    "sets) belong to no analysis element and no referenced rigid diaphragm: "
                    "they would be free nodes in the translated deck"),
        ))
    omitted = sorted(set(int(n) for n in scd.mesh.node_ids) - session)
    if omitted:
        out.append(Unsupported(
            category="option", xobj_meta="mesh:mesh nodes outside the session",
            ids=tuple(omitted[:10]), names=(f"{len(omitted)} nodes",),
            reason=(f"STKO writes {len(omitted)} mesh node(s) that no analysis element or "
                    "referenced rigid diaphragm uses (unassigned meshed geometry, links of an "
                    "unreferenced interaction); the translated deck would not have them"),
        ))
    return out


def _type_meta(code: int) -> str:
    return f"{ELEMENT_TYPES.get(code, 'type')}({code})"


# ── build ─────────────────────────────────────────────────────────────

class _Builder:
    """Adds discrete entities, nodes and elements; tracks node ownership
    and the synthetic (carrier) element ids."""

    def __init__(self, scd: ScdModel) -> None:
        import gmsh
        self.gmsh = gmsh
        self.scd = scd
        self.owned: set[int] = set()
        self.first_synthetic = max(scd.mesh.elements, default=0) + 1
        self.next_eid = self.first_synthetic

    def xyz(self, nodes: list[int]) -> np.ndarray:
        mesh = self.scd.mesh
        ids = np.asarray(nodes, dtype=np.int64)
        rows = np.searchsorted(mesh.node_ids, ids)
        bad = (rows >= len(mesh.node_ids)) | (mesh.node_ids[np.minimum(rows, len(mesh.node_ids) - 1)] != ids)
        if bad.any():
            raise KeyError(f"nodes {ids[bad].tolist()[:10]} are not in the STKO mesh")
        return mesh.coordinates[rows]

    def entity(self, dim: int, nodes: Iterable[int]) -> int:
        """A new discrete entity owning those of ``nodes`` not owned yet."""
        tag = self.gmsh.model.addDiscreteEntity(dim)
        new = [n for n in dict.fromkeys(nodes) if n not in self.owned]
        if new:
            self.gmsh.model.mesh.addNodes(dim, tag, new, self.xyz(new).ravel().tolist())
            self.owned.update(new)
        return tag

    def points(self, tag: int, nodes: list[int]) -> None:
        """One synthetic point element per node."""
        ids = list(range(self.next_eid, self.next_eid + len(nodes)))
        self.next_eid += len(nodes)
        self.gmsh.model.mesh.addElementsByType(tag, _POINT, ids, nodes)

    def elements(self, tag: int, eids: list[int], analysis: dict[int, Any]) -> None:
        """STKO elements with their STKO ids, by type; analysis shells in
        rule M3's order."""
        by_type: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
        for e in eids:
            el = self.scd.mesh.elements[e]
            nodes = shell_node_order(self.scd, e) if (el.type == 303 and e in analysis) else el.nodes
            ids, conn = by_type[GMSH_TYPE[el.type]]
            ids.append(e)
            conn.extend(nodes)
        for gtype, (ids, conn) in by_type.items():
            self.gmsh.model.mesh.addElementsByType(tag, gtype, ids, conn)


def build_mesh(g: Any, scd: ScdModel) -> MeshMap:
    """Put STKO's mesh into the (empty) session ``g`` and name its groups.

    Steps (rules doc section 6): diaphragm carriers first, then one
    discrete entity per referenced sub-shape (vertices, edges, faces,
    solids: the node-ownership order), split by local axis; then the
    physical groups of element groups, selection sets, conditions and
    diaphragms. Raises :class:`UnsupportedSTKOTypes` for anything the mesh
    slice does not support, before touching the session. Never meshes,
    renumbers or removes anything.
    """
    unsupported = check_supported(scd)
    if unsupported:
        raise UnsupportedSTKOTypes(tuple(unsupported))
    b = _Builder(scd)
    gmsh = b.gmsh
    if gmsh.model.getEntities():
        raise ValueError(
            "build_mesh needs an empty session: STKO's node and element tags "
            "would collide with the entities already in it"
        )
    analysis = scd.analysis_elements()
    sets = _set_subshapes(scd)
    pgs = _PGs(g)

    # 1. diaphragm carriers (they own their nodes before anything else): the
    # groups of the rigidDiaphragm conditions a constraint pattern references,
    # the same groups the conditions plan pairs come from (rule C7)
    diaphragms = list(diaphragm_groups(scd))
    for d in diaphragms:
        taken = [n for n in (d.master_node, *d.slave_nodes) if n in b.owned]
        if taken:
            raise ValueError(
                f"rigid diaphragm condition {d.condition}, master {d.master_node}: "
                f"nodes {taken[:10]} already belong to another diaphragm"
            )
        slave_tag = b.entity(0, list(d.slave_nodes))
        b.points(slave_tag, list(d.slave_nodes))
        master_tag = b.entity(0, [d.master_node])
        b.points(master_tag, [d.master_node])
        pgs.add(0, [master_tag], d.master_pg)
        pgs.add(0, [slave_tag], d.slave_pg)

    # 2-3. one entity per referenced sub-shape, split by local axis
    subshape_entities: dict[SubShape, tuple[int, ...]] = {}
    group_entities: dict[tuple[int, int, Vec3], list[int]] = defaultdict(list)
    group_elements: dict[tuple[int, int, Vec3], list[int]] = defaultdict(list)
    group_dim: dict[tuple[int, int, Vec3], int] = {}
    for sub in _referenced(scd, analysis, sets):
        gid, kind, idx = sub
        dim = KIND_DIM[kind]
        if kind == "vertices":
            try:
                node = int(scd.mesh.vertex_nodes[gid][idx])
            except (KeyError, IndexError):
                raise ValueError(
                    f"geometry {gid} ({scd.geometries[gid].name}) vertex {idx} is "
                    "referenced but has no mesh node"
                ) from None
            tag = b.entity(0, [node])
            b.points(tag, [node])
            subshape_entities[sub] = (tag,)
            continue
        eids = _domain(scd, sub)
        split: dict[Vec3 | None, list[int]] = {}
        for e in eids:
            a = analysis.get(e)
            key = None
            if a is not None:
                if e not in scd.mesh.orientation:
                    raise ValueError(f"analysis element {e} has no orientation quaternion")
                key = local_axis(scd, e)
            split.setdefault(key, []).append(e)
        if not split:
            split[None] = []
        tags = []
        for key in sorted(split, key=lambda k: (k is not None, k or ())):
            part = split[key]
            conn = [n for e in part for n in scd.mesh.elements[e].nodes]
            tag = b.entity(dim, conn)
            if part:
                b.elements(tag, part, analysis)
            tags.append(tag)
            if key is not None:
                a = analysis[part[0]]
                gkey = (a.element_property, a.physical_property, key)
                if group_dim.setdefault(gkey, dim) != dim:
                    raise ValueError(
                        f"element property {a.element_property} / physical property "
                        f"{a.physical_property} is assigned to sub-shapes of two dimensions"
                    )
                group_entities[gkey].append(tag)
                group_elements[gkey].extend(part)
        subshape_entities[sub] = tuple(tags)

    # 4. physical groups: element groups
    element_groups: list[ElementGroup] = []
    for gkey, name in _group_names(scd, list(group_entities)).items():
        ep, pp, axis = gkey
        pgs.add(group_dim[gkey], group_entities[gkey], name)
        element_groups.append(ElementGroup(
            pg=name, dim=group_dim[gkey], element_property=ep, physical_property=pp,
            local_axis=axis, element_ids=tuple(sorted(group_elements[gkey])),
        ))

    # selection sets
    selection_set_pgs: dict[str, dict[str, str]] = {}
    for name, subs in sets.items():
        selection_set_pgs[name] = _kind_pgs(pgs, name, ".", subs, subshape_entities)

    # conditions
    condition_pgs: dict[int, dict[str, str]] = {}
    for cid in sorted(scd.conditions):
        c = scd.conditions[cid]
        condition_pgs[cid] = _kind_pgs(
            pgs, f"cond:{cid}:{c.name}", ".", _expand(scd, c.geometry, ()),
            subshape_entities,
        )

    session_nodes = sorted(b.owned)
    return MeshMap(
        node_ids=tuple(session_nodes),
        element_groups=tuple(element_groups),
        subshape_entities=subshape_entities,
        selection_set_pgs=selection_set_pgs,
        condition_pgs=condition_pgs,
        diaphragms=tuple(diaphragms),
        carrier_ids=range(b.first_synthetic, b.next_eid),
    )


# ── physical-group naming ─────────────────────────────────────────────

class _PGs:
    """``g.physical.add`` with a name registry: ``g.physical.add`` merges
    into an existing PG of the same name, so a clash would silently merge
    two groups. Refuse it instead."""

    def __init__(self, g: Any) -> None:
        self.g = g
        self.names: dict[str, int] = {}

    def add(self, dim: int, tags: list[int], name: str) -> None:
        if name in self.names:
            raise ValueError(f"two STKO groups map to the physical-group name {name!r}")
        self.names[name] = dim
        self.g.physical.add(dim, list(tags), name=name)


def _kind_pgs(pgs: _PGs, name: str, sep: str, subs: list[SubShape],
              subshape_entities: dict[SubShape, tuple[int, ...]]) -> dict[str, str]:
    """One PG per kind: ``name`` alone when there is one kind, else
    ``name + sep + kind``."""
    per_kind: dict[str, list[int]] = {}
    for sub in subs:
        per_kind.setdefault(sub[1], []).extend(subshape_entities[sub])
    out: dict[str, str] = {}
    for kind in KINDS:
        if kind not in per_kind:
            continue
        pg = name if len(per_kind) == 1 else f"{name}{sep}{kind}"
        pgs.add(KIND_DIM[kind], sorted(set(per_kind[kind])), pg)
        out[kind] = pg
    return out


def _group_names(scd: ScdModel, keys: list[tuple[int, int, Vec3]]) -> dict[tuple[int, int, Vec3], str]:
    """Rule M6 names, in ``(element property, physical property, axis)``
    order: ``"{ep}|{pp}"``; ``"{ep}#{id}|{pp}#{id}"`` when two pairs share
    that; ``"|L{j}"`` appended (j from 1, by sorted axis) when one pair
    has several axes."""
    keys = sorted(keys)
    pairs = sorted({(ep, pp) for ep, pp, _ in keys})

    def label(ep: int, pp: int) -> str:
        return f"{_name(scd.element_properties, ep)}|{_name(scd.physical_properties, pp)}"

    counts: dict[str, int] = defaultdict(int)
    for ep, pp in pairs:
        counts[label(ep, pp)] += 1
    base = {
        (ep, pp): label(ep, pp) if counts[label(ep, pp)] == 1 else
        f"{_name(scd.element_properties, ep)}#{ep}|{_name(scd.physical_properties, pp)}#{pp}"
        for ep, pp in pairs
    }
    out: dict[tuple[int, int, Vec3], str] = {}
    for ep, pp in pairs:
        axes = sorted(axis for e, p, axis in keys if (e, p) == (ep, pp))
        for j, axis in enumerate(axes, start=1):
            out[(ep, pp, axis)] = base[(ep, pp)] if len(axes) == 1 else f"{base[(ep, pp)]}|L{j}"
    return out


def _name(objects: dict[int, Any], oid: int) -> str:
    """An object's name; ``""`` for id 0 or an id the document lacks."""
    x = objects.get(oid)
    return x.name if x is not None else ""
