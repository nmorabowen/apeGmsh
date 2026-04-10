"""
LoadsComposite -- Define and resolve loads.

Two-stage pipeline mirroring :class:`ConstraintsComposite`:

1. **Define** (pre-mesh): factory methods (``point``, ``line``,
   ``surface``, ``gravity``, ``body``) store :class:`LoadDef` objects.
   ``with g.loads.pattern(name):`` groups loads under a named pattern.
2. **Resolve** (post-mesh): :meth:`resolve` delegates to
   :class:`LoadResolver` (in ``solvers/Loads.py``) with caller-provided
   node/face maps.  Auto-called by ``Mesh.get_fem_data()``.

Targets accept any of:
    * a list of ``(dim, tag)`` tuples
    * a part label (``g.parts.instances[label]``)
    * a physical group name (``g.physical``)
    * a mesh selection name (``g.mesh_selection``)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._session import _SessionBase

from apeGmsh.mesh.FEMData import LoadSet
from apeGmsh.solvers.Loads import (
    LoadDef,
    LoadRecord,
    LoadResolver,
    PointLoadDef,
    LineLoadDef,
    SurfaceLoadDef,
    GravityLoadDef,
    BodyLoadDef,
)


# (LoadDefType, reduction, target_form) → method name on LoadsComposite
_DISPATCH: dict[type, dict[tuple[str, str], str]] = {
    PointLoadDef: {
        ("tributary",  "nodal"):   "_resolve_point",
        ("consistent", "nodal"):   "_resolve_point",
    },
    LineLoadDef: {
        ("tributary",  "nodal"):   "_resolve_line_tributary",
        ("consistent", "nodal"):   "_resolve_line_consistent",
        ("tributary",  "element"): "_resolve_line_element",
        ("consistent", "element"): "_resolve_line_element",
    },
    SurfaceLoadDef: {
        ("tributary",  "nodal"):   "_resolve_surface_tributary",
        ("consistent", "nodal"):   "_resolve_surface_consistent",
        ("tributary",  "element"): "_resolve_surface_element",
        ("consistent", "element"): "_resolve_surface_element",
    },
    GravityLoadDef: {
        ("tributary",  "nodal"):   "_resolve_gravity_tributary",
        ("consistent", "nodal"):   "_resolve_gravity_consistent",
        ("tributary",  "element"): "_resolve_gravity_element",
        ("consistent", "element"): "_resolve_gravity_element",
    },
    BodyLoadDef: {
        ("tributary",  "nodal"):   "_resolve_body_tributary",
        ("consistent", "nodal"):   "_resolve_body_tributary",
        ("tributary",  "element"): "_resolve_body_element",
        ("consistent", "element"): "_resolve_body_element",
    },
}


class LoadsComposite:
    """Loads composite — define + resolve loads."""

    def __init__(self, parent: "_SessionBase") -> None:
        self._parent = parent
        self.load_defs: list[LoadDef] = []
        self.load_records: list[LoadRecord] = []
        self._active_pattern: str = "default"

    # ------------------------------------------------------------------
    # Pattern grouping
    # ------------------------------------------------------------------

    @contextmanager
    def pattern(self, name: str) -> Iterator[None]:
        """Group subsequent load definitions under a named pattern.

        Example
        -------
        ::

            with g.loads.pattern("dead"):
                g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
                g.loads.line("beams", magnitude=-2e3, direction="z")

            with g.loads.pattern("live"):
                g.loads.surface("slabs", magnitude=-3e3)
        """
        prev = self._active_pattern
        self._active_pattern = name
        try:
            yield
        finally:
            self._active_pattern = prev

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def point(self, target, *, force_xyz=None, moment_xyz=None,
              name=None) -> PointLoadDef:
        """Concentrated force/moment at the node(s) of *target*."""
        return self._add_def(PointLoadDef(
            target=target, pattern=self._active_pattern, name=name,
            force_xyz=force_xyz, moment_xyz=moment_xyz,
        ))

    def line(self, target, *, magnitude=None, direction=(0., 0., -1.),
             q_xyz=None, reduction="tributary", target_form="nodal",
             name=None) -> LineLoadDef:
        """Distributed line load along curve(s) of *target*."""
        if magnitude is None and q_xyz is None:
            raise ValueError("line() requires either magnitude or q_xyz.")
        return self._add_def(LineLoadDef(
            target=target, pattern=self._active_pattern, name=name,
            magnitude=magnitude or 0.0, direction=direction, q_xyz=q_xyz,
            reduction=reduction, target_form=target_form,
        ))

    def surface(self, target, *, magnitude=0.0, normal=True,
                direction=(0., 0., -1.), reduction="tributary",
                target_form="nodal", name=None) -> SurfaceLoadDef:
        """Pressure or traction on surface(s) of *target*."""
        return self._add_def(SurfaceLoadDef(
            target=target, pattern=self._active_pattern, name=name,
            magnitude=magnitude, normal=normal, direction=direction,
            reduction=reduction, target_form=target_form,
        ))

    def gravity(self, target, *, g=(0., 0., -9.81), density=None,
                reduction="tributary", target_form="nodal",
                name=None) -> GravityLoadDef:
        """Body weight from gravity over volume(s) of *target*."""
        return self._add_def(GravityLoadDef(
            target=target, pattern=self._active_pattern, name=name,
            g=g, density=density,
            reduction=reduction, target_form=target_form,
        ))

    def body(self, target, *, force_per_volume=(0., 0., 0.),
             reduction="tributary", target_form="nodal",
             name=None) -> BodyLoadDef:
        """Generic per-volume body force on volume(s) of *target*."""
        return self._add_def(BodyLoadDef(
            target=target, pattern=self._active_pattern, name=name,
            force_per_volume=force_per_volume,
            reduction=reduction, target_form=target_form,
        ))

    # ------------------------------------------------------------------
    # Internal: store + validate
    # ------------------------------------------------------------------

    def _add_def(self, defn: LoadDef) -> LoadDef:
        # Light validation: ensure the dispatch supports this combo
        cfg = _DISPATCH.get(type(defn), {})
        key = (defn.reduction, defn.target_form)
        if key not in cfg:
            raise ValueError(
                f"{type(defn).__name__} does not support "
                f"reduction={defn.reduction!r}, target_form={defn.target_form!r}. "
                f"Supported: {list(cfg.keys())}"
            )
        self.load_defs.append(defn)
        return defn

    # ------------------------------------------------------------------
    # Target resolution: convert flexible target → DimTag list
    # ------------------------------------------------------------------

    def _resolve_target(self, target) -> list[tuple[int, int]]:
        """Resolve a target identifier to a list of ``(dim, tag)`` pairs.

        Lookup order:
            1. ``list[tuple[int, int]]``  → as-is
            2. mesh selection name        → entities from g.mesh_selection
            3. physical group name        → entities from g.physical
            4. part label                 → entities from g.parts.instances
        """
        # 1. Raw DimTag list
        if isinstance(target, (list, tuple)) and len(target) > 0 \
                and isinstance(target[0], (list, tuple)):
            return [(int(d), int(t)) for d, t in target]

        if not isinstance(target, str):
            raise TypeError(
                f"target must be a string label or list of (dim, tag), "
                f"got {type(target).__name__}"
            )

        # 2. Mesh selection name
        ms = getattr(self._parent, "mesh_selection", None)
        if ms is not None and hasattr(ms, "_sets"):
            for (dim, tag), info in ms._sets.items():
                if info.get("name") == target:
                    # Mesh selections live at node/element level — return
                    # as a sentinel (handled in _resolve_target_nodes/elements)
                    return [("__ms__", dim, tag)]

        # 3. Physical group name
        physical = getattr(self._parent, "physical", None)
        if physical is not None:
            try:
                import gmsh
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    try:
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == target:
                            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
                            return [(pg_dim, int(t)) for t in ents]
                    except Exception:
                        pass
            except ImportError:
                pass

        # 4. Part label
        parts = getattr(self._parent, "parts", None)
        if parts is not None and hasattr(parts, "_instances"):
            inst = parts._instances.get(target)
            if inst is not None:
                out = []
                for d, ts in inst.entities.items():
                    out.extend((int(d), int(t)) for t in ts)
                return out

        raise KeyError(
            f"Target {target!r} not found as part label, "
            f"physical group name, or mesh selection name."
        )

    def _target_nodes(self, target, node_map, all_nodes) -> set[int]:
        """Resolve target to a set of mesh node IDs."""
        dts = self._resolve_target(target)

        # Mesh selection sentinel
        if dts and dts[0][0] == "__ms__":
            _, dim, tag = dts[0]
            ms = self._parent.mesh_selection
            info = ms._sets.get((dim, tag))
            if info is None:
                return set()
            return set(int(n) for n in info.get("node_ids", []))

        # Part label fast path: use the precomputed node map
        parts = getattr(self._parent, "parts", None)
        if isinstance(target, str) and parts is not None:
            if target in getattr(parts, "_instances", {}):
                if node_map is not None and target in node_map:
                    return set(node_map[target])

        # General path: query gmsh for nodes of each entity
        import gmsh
        nodes: set[int] = set()
        for d, t in dts:
            try:
                nt, _, _ = gmsh.model.mesh.getNodes(
                    dim=int(d), tag=int(t),
                    includeBoundary=True, returnParametricCoord=False,
                )
                nodes.update(int(n) for n in nt)
            except Exception:
                pass
        return nodes

    def _target_edges(self, target) -> list[tuple[int, int]]:
        """Resolve target to a list of (n1, n2) line edges."""
        dts = self._resolve_target(target)
        if dts and dts[0][0] == "__ms__":
            return []  # mesh selections don't expose edge connectivity
        import gmsh
        edges: list[tuple[int, int]] = []
        for d, t in dts:
            if d != 1:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                # gmsh element type 1 = 2-node line
                # type 8 = 3-node line (treat as 2-node end-to-end for now)
                npe = 2 if int(etype) == 1 else 3
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    edges.append((int(row[0]), int(row[-1])))
        return edges

    def _target_faces(self, target) -> list[list[int]]:
        """Resolve target to a list of node-id lists (one per face element)."""
        dts = self._resolve_target(target)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        faces: list[list[int]] = []
        for d, t in dts:
            if d != 2:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype = int(etype)
                # 2 = tri3, 3 = quad4, 9 = tri6, 16 = quad8
                npe = {2: 3, 3: 4, 9: 6, 16: 8}.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                # Use only corner nodes for face area / normal
                corners_per = {3: 3, 4: 4, 6: 3, 8: 4}[npe]
                for row in arr:
                    faces.append([int(n) for n in row[:corners_per]])
        return faces

    def _target_elements(self, target):
        """Resolve target to (element_ids, connectivity_rows) for volume elements."""
        dts = self._resolve_target(target)
        if dts and dts[0][0] == "__ms__":
            return [], []
        import gmsh
        eids: list[int] = []
        conns: list[np.ndarray] = []
        for d, t in dts:
            if d != 3:
                continue
            try:
                etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, etags, enodes in zip(etypes, etags_list, enodes_list):
                etype = int(etype)
                # 4 = tet4, 5 = hex8, 6 = prism6, 11 = tet10, 17 = hex20
                npe_map = {4: 4, 5: 8, 6: 6, 11: 10, 17: 20}
                npe = npe_map.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for tag, row in zip(etags, arr):
                    eids.append(int(tag))
                    conns.append(row)
        return eids, conns

    # ------------------------------------------------------------------
    # resolve()
    # ------------------------------------------------------------------

    def resolve(
        self,
        node_tags,
        node_coords,
        elem_tags=None,
        connectivity=None,
        *,
        node_map=None,
        face_map=None,
        ndf: int = 6,
    ) -> LoadSet:
        """Resolve all stored LoadDefs into a :class:`LoadSet`."""
        resolver = LoadResolver(
            node_tags, node_coords, elem_tags, connectivity, ndf=ndf,
        )
        all_nodes = set(int(t) for t in node_tags)
        records: list = []
        for defn in self.load_defs:
            cfg = _DISPATCH[type(defn)]
            key = (defn.reduction, defn.target_form)
            method_name = cfg.get(key)
            if method_name is None:
                raise ValueError(
                    f"{type(defn).__name__} does not support "
                    f"reduction={defn.reduction!r}, target_form={defn.target_form!r}"
                )
            method = getattr(self, method_name)
            result = method(resolver, defn, node_map, all_nodes)
            records.extend(result)
        self.load_records = records
        return LoadSet(records)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    def _resolve_point(self, resolver, defn, node_map, all_nodes):
        nodes = self._target_nodes(defn.target, node_map, all_nodes)
        return resolver.resolve_point(defn, nodes)

    def _resolve_line_tributary(self, resolver, defn, node_map, all_nodes):
        edges = self._target_edges(defn.target)
        return resolver.resolve_line_tributary(defn, edges)

    def _resolve_line_consistent(self, resolver, defn, node_map, all_nodes):
        edges = self._target_edges(defn.target)
        return resolver.resolve_line_consistent(defn, edges)

    def _resolve_line_element(self, resolver, defn, node_map, all_nodes):
        # For element-form output we need element IDs of the target curves
        dts = self._resolve_target(defn.target)
        import gmsh
        eids: list[int] = []
        for d, t in dts:
            if d != 1:
                continue
            try:
                _, etags_list, _ = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etags in etags_list:
                eids.extend(int(e) for e in etags)
        return resolver.resolve_line_element(defn, eids)

    def _resolve_surface_tributary(self, resolver, defn, node_map, all_nodes):
        faces = self._target_faces(defn.target)
        return resolver.resolve_surface_tributary(defn, faces)

    def _resolve_surface_consistent(self, resolver, defn, node_map, all_nodes):
        faces = self._target_faces(defn.target)
        return resolver.resolve_surface_consistent(defn, faces)

    def _resolve_surface_element(self, resolver, defn, node_map, all_nodes):
        dts = self._resolve_target(defn.target)
        import gmsh
        eids: list[int] = []
        for d, t in dts:
            if d != 2:
                continue
            try:
                _, etags_list, _ = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etags in etags_list:
                eids.extend(int(e) for e in etags)
        return resolver.resolve_surface_element(defn, eids)

    def _resolve_gravity_tributary(self, resolver, defn, node_map, all_nodes):
        _, conns = self._target_elements(defn.target)
        return resolver.resolve_gravity_tributary(defn, conns)

    def _resolve_gravity_consistent(self, resolver, defn, node_map, all_nodes):
        _, conns = self._target_elements(defn.target)
        return resolver.resolve_gravity_consistent(defn, conns)

    def _resolve_gravity_element(self, resolver, defn, node_map, all_nodes):
        eids, _ = self._target_elements(defn.target)
        return resolver.resolve_gravity_element(defn, eids)

    def _resolve_body_tributary(self, resolver, defn, node_map, all_nodes):
        _, conns = self._target_elements(defn.target)
        return resolver.resolve_body_tributary(defn, conns)

    def _resolve_body_element(self, resolver, defn, node_map, all_nodes):
        eids, _ = self._target_elements(defn.target)
        return resolver.resolve_body_element(defn, eids)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_pattern(self, name: str) -> list[LoadDef]:
        return [d for d in self.load_defs if d.pattern == name]

    def patterns(self) -> list[str]:
        seen: list[str] = []
        for d in self.load_defs:
            if d.pattern not in seen:
                seen.append(d.pattern)
        return seen

    def __len__(self) -> int:
        return len(self.load_defs)

    def __repr__(self) -> str:
        if not self.load_defs:
            return "LoadsComposite(empty)"
        return (
            f"LoadsComposite({len(self.load_defs)} defs, "
            f"{len(self.patterns())} pattern(s))"
        )
