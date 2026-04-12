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
    nm  = g.parts.build_node_map(fem.node_ids, fem.node_coords)
    fm  = g.parts.build_face_map(nm)
    recs = g.constraints.resolve(
        fem.node_ids, fem.node_coords, node_map=nm, face_map=fm,
    )
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.mesh.FEMData import ConstraintSet
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
    NodeToSurfaceDef:        "_resolve_node_to_surface",
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
    TiedContactDef:          "resolve_tied_contact",
    MortarDef:               "resolve_mortar",
}

_FACE_TYPES = (TieDef, TiedContactDef, MortarDef)

_ConstraintT = TypeVar("_ConstraintT", bound=ConstraintDef)


class ConstraintsComposite:
    """Constraint composite -- define + resolve kinematic interactions."""

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

        Parameters
        ----------
        master : int, str, or (dim, tag)
            The 6-DOF reference node.  Accepts a raw tag, a label
            name, or a dimtag.  If a label/PG resolves to multiple
            tags, only the first is used.
        slave : int, str, or (dim, tag)
            The surface entity to couple.  Accepts a raw tag, a
            label, or a PG name.  If it resolves to multiple
            surfaces, one constraint is created per surface.

        Returns
        -------
        NodeToSurfaceDef or list[NodeToSurfaceDef]
            A single def when slave resolves to one surface,
            a list when it resolves to multiple.
        """
        from ._helpers import resolve_to_tags
        m_tags = resolve_to_tags(master, dim=0, session=self._parent)
        s_tags = resolve_to_tags(slave,  dim=2, session=self._parent)
        master_tag = m_tags[0]

        defs = []
        for s_tag in s_tags:
            d = self._add_def(NodeToSurfaceDef(
                master_label=str(master_tag), slave_label=str(s_tag),
                dofs=dofs, tolerance=tolerance, name=name))
            defs.append(d)
        return defs[0] if len(defs) == 1 else defs

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

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(self, node_tags, node_coords, elem_tags=None,
                connectivity=None, *, node_map=None, face_map=None) -> ConstraintSet:
        has_face_constraints = any(
            isinstance(d, _FACE_TYPES) for d in self.constraint_defs)
        if has_face_constraints and face_map is None:
            warnings.warn(
                "Surface constraints defined but face_map=None.", stacklevel=2)

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
        master_tag = int(defn.master_label)
        if master_tag not in all_nodes:
            raise ValueError(
                f"node_to_surface: master node tag {master_tag} not found in the mesh.")
        slave_entity_tag = int(defn.slave_label)
        try:
            nt, _, _ = gmsh.model.mesh.getNodes(
                dim=2, tag=slave_entity_tag,
                includeBoundary=True, returnParametricCoord=False)
            slave_nodes = {int(t) for t in nt}
        except Exception as exc:
            raise ValueError(
                f"node_to_surface: cannot get nodes from surface "
                f"entity (dim=2, tag={slave_entity_tag}): {exc}") from exc
        if not slave_nodes:
            raise ValueError(
                f"node_to_surface: surface entity (dim=2, "
                f"tag={slave_entity_tag}) has no nodes.")
        return resolver.resolve_node_to_surface(defn, master_tag, slave_nodes)

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
