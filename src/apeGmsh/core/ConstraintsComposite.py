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
    """Constraint composite — define + resolve kinematic interactions.

    Target model
    ------------
    Constraints identify their master and slave sides by **part label**
    (a key of ``g.parts._instances``), not by the multi-tier scheme
    used by :class:`LoadsComposite`. :meth:`_add_def` validates both
    labels against the registry and raises ``KeyError`` on a typo::

        g.constraints.tie(master_label="column",
                          slave_label="slab",
                          master_entities=[(2, 13)],   # optional scope
                          slave_entities=[(2, 17)])

    Optional ``master_entities`` / ``slave_entities`` (list of
    ``(dim, tag)``) narrow the search to a subset of the part's
    entities — useful when a part has many surfaces and only one is
    the interface.

    Exceptions
    ~~~~~~~~~~

    * :class:`NodeToSurfaceDef` (and its spring variant) bypass label
      validation because their ``master`` is a bare node tag and their
      ``slave`` is a raw ``(dim=2, tag)`` surface. :meth:`node_to_surface`
      accepts ``int``, ``str``, or ``(dim, tag)`` for both arguments.
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.constraint_defs: list[ConstraintDef] = []
        self.constraint_records: list[ConstraintRecord] = []

    def _add_def(self, defn: _ConstraintT) -> _ConstraintT:
        """Validate labels and store a constraint definition.
        Label validation is skipped for NodeToSurfaceDef (bare tags)."""
        if not isinstance(defn, NodeToSurfaceDef):
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

    # Level 1
    def equal_dof(self, master_label, slave_label, *, master_entities=None,
                  slave_entities=None, dofs=None, tolerance=1e-6,
                  name=None) -> EqualDOFDef:
        return self._add_def(EqualDOFDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def rigid_link(self, master_label, slave_label, *, link_type="beam",
                   master_point=None, slave_entities=None,
                   tolerance=1e-6, name=None) -> RigidLinkDef:
        return self._add_def(RigidLinkDef(
            master_label=master_label, slave_label=slave_label,
            link_type=link_type, master_point=master_point,
            slave_entities=slave_entities, tolerance=tolerance, name=name))

    def penalty(self, master_label, slave_label, *, stiffness=1e10,
                dofs=None, tolerance=1e-6, name=None) -> PenaltyDef:
        return self._add_def(PenaltyDef(
            master_label=master_label, slave_label=slave_label,
            stiffness=stiffness, dofs=dofs, tolerance=tolerance, name=name))

    # Level 2
    def rigid_diaphragm(self, master_label, slave_label, *,
                        master_point=(0., 0., 0.),
                        plane_normal=(0., 0., 1.),
                        constrained_dofs=None, plane_tolerance=1.0,
                        name=None) -> RigidDiaphragmDef:
        return self._add_def(RigidDiaphragmDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, plane_normal=plane_normal,
            constrained_dofs=constrained_dofs or [1, 2, 6],
            plane_tolerance=plane_tolerance, name=name))

    def rigid_body(self, master_label, slave_label, *,
                   master_point=(0., 0., 0.), name=None) -> RigidBodyDef:
        return self._add_def(RigidBodyDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, name=name))

    def kinematic_coupling(self, master_label, slave_label, *,
                           master_point=(0., 0., 0.), dofs=None,
                           name=None) -> KinematicCouplingDef:
        return self._add_def(KinematicCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, dofs=dofs or [1, 2, 3, 4, 5, 6],
            name=name))

    # Level 3
    def tie(self, master_label, slave_label, *, master_entities=None,
            slave_entities=None, dofs=None, tolerance=1.0,
            name=None) -> TieDef:
        return self._add_def(TieDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def distributing_coupling(self, master_label, slave_label, *,
                              master_point=(0., 0., 0.), dofs=None,
                              weighting="uniform",
                              name=None) -> DistributingCouplingDef:
        return self._add_def(DistributingCouplingDef(
            master_label=master_label, slave_label=slave_label,
            master_point=master_point, dofs=dofs, weighting=weighting,
            name=name))

    def embedded(self, host_label, embedded_label, *, tolerance=1.0,
                 name=None) -> EmbeddedDef:
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

    # Level 4
    def tied_contact(self, master_label, slave_label, *,
                     master_entities=None, slave_entities=None,
                     dofs=None, tolerance=1.0,
                     name=None) -> TiedContactDef:
        return self._add_def(TiedContactDef(
            master_label=master_label, slave_label=slave_label,
            master_entities=master_entities, slave_entities=slave_entities,
            dofs=dofs, tolerance=tolerance, name=name))

    def mortar(self, master_label, slave_label, *,
               master_entities=None, slave_entities=None,
               dofs=None, integration_order=2,
               name=None) -> MortarDef:
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
        raise NotImplementedError("Embedded constraint resolution is not implemented yet.")

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
