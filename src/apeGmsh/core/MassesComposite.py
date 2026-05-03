"""
MassesComposite -- Define and resolve nodal masses.

Two-stage pipeline mirroring :class:`LoadsComposite` but simpler:

1. **Define** (pre-mesh): factory methods (``point``, ``line``,
   ``surface``, ``volume``) store :class:`MassDef` objects.
2. **Resolve** (post-mesh): :meth:`resolve` delegates to
   :class:`MassResolver` and accumulates per-node mass into a
   :class:`MassSet`.  Auto-called by ``Mesh.get_fem_data()``.

There is **no pattern grouping** — mass is intrinsic to the model.
Multiple mass definitions targeting the same nodes accumulate.

Targets accept any of:
    * a list of ``(dim, tag)`` tuples
    * a part label (``g.parts.instances[label]``)
    * a physical group name (``g.physical``)
    * a mesh selection name (``g.mesh_selection``)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.mesh._record_set import MassSet
from apeGmsh.solvers.Masses import (
    MassDef,
    MassRecord,
    MassResolver,
    PointMassDef,
    LineMassDef,
    SurfaceMassDef,
    VolumeMassDef,
)


# (MassDefType, reduction) -> method name on MassesComposite
_DISPATCH: dict[type, dict[str, str]] = {
    PointMassDef: {
        "lumped":     "_resolve_point",
        "consistent": "_resolve_point",
    },
    LineMassDef: {
        "lumped":     "_resolve_line_lumped",
        "consistent": "_resolve_line_consistent",
    },
    SurfaceMassDef: {
        "lumped":     "_resolve_surface_lumped",
        "consistent": "_resolve_surface_consistent",
    },
    VolumeMassDef: {
        "lumped":     "_resolve_volume_lumped",
        "consistent": "_resolve_volume_consistent",
    },
}

_MassT = TypeVar("_MassT", bound=MassDef)


class MassesComposite:
    """Solver-agnostic nodal-mass composite — declare on geometry,
    accumulate per-node mass after meshing.

    Two-stage pipeline
    ------------------
    1. **Declare** (pre-mesh): the factory methods on this composite
       (:meth:`point`, :meth:`line`, :meth:`surface`, :meth:`volume`)
       store :class:`~apeGmsh.solvers.Masses.MassDef` dataclasses
       describing *intent* on geometric targets — concentrated
       lumps, line densities, areal densities, or material density
       on volumes.
    2. **Resolve** (post-mesh): :meth:`resolve` (called automatically
       by :meth:`Mesh.queries.get_fem_data`) walks the def list,
       hands each one to
       :class:`~apeGmsh.solvers.Masses.MassResolver`, and
       **accumulates** contributions across overlapping targets so
       each node ends up with at most one
       :class:`~apeGmsh.solvers.Masses.MassRecord`.

    Resolved records land on ``fem.nodes.masses`` as a
    :class:`~apeGmsh.mesh._record_set.MassSet`. Each record carries
    a length-6 mass vector ``(mx, my, mz, Ixx, Iyy, Izz)``;
    downstream solver bridges slice it to the model's ``ndf`` (the
    rotational components are dropped for ``ndf < 4``).

    No patterns
    -----------
    Unlike :class:`LoadsComposite`, masses are **not** grouped by
    pattern. Mass is intrinsic to the model — there is one nodal
    mass per node regardless of which load pattern is active.

    Reduction modes
    ---------------
    Each factory accepts ``reduction="lumped"`` (default) or
    ``reduction="consistent"``:

    * **lumped** — each element's total mass is split equally
      among its corner nodes. Diagonal mass matrix; cheap and
      stable for explicit dynamics.
    * **consistent** — line elements use the proper
      ``ρ_l L / 6 · [[2,1],[1,2]]`` consistent matrix; surface
      and volume paths currently fall through to lumped because,
      for tri3 / quad4 / tet4 / hex8 with constant density, the
      consistent diagonal sum equals the lumped per-node share.
      The separate paths are kept so higher-order types (tri6,
      quad8, tet10, hex20) can be wired in without changing the
      public API.

    Avoiding double-counting
    ------------------------
    apeGmsh always emits explicit ``ops.mass(node, mx, my, mz, …)``
    commands. If your OpenSees material or section also carries a
    non-zero ``rho``, those contributions add to whatever this
    composite emits. Either:

    * keep ``rho=0`` on the material and let this composite carry
      all inertia, **or**
    * skip the matching :meth:`volume` / :meth:`surface` call and
      let the material handle it.

    Target identification
    ---------------------
    Targets follow the same flexible scheme as :class:`LoadsComposite`:

    * a list of ``(dim, tag)`` tuples
    * a part label (``g.parts.instances[label]``)
    * a physical group name (``g.physical``)
    * a mesh-selection name (``g.mesh_selection``)
    * a Tier 1 internal label

    Pass ``pg=`` / ``label=`` / ``tag=`` to bypass auto-resolution
    and pin a specific source.

    Examples
    --------
    Declare three sources of mass and resolve to nodes::

        with apeGmsh(model_name="frame") as g:
            # ... geometry + Parts ...

            # Lumped mass at the top of a tower
            g.masses.point("Antenna", mass=350.0)

            # Cladding mass spread along an exterior edge
            g.masses.line("Cladding", linear_density=120.0)

            # Self-mass of all column volumes
            g.masses.volume("Columns", density=7850.0)

            g.mesh.generation.generate(dim=3)
            fem = g.mesh.queries.get_fem_data(dim=3)

            for m in fem.nodes.masses:
                ops.mass(m.node_id, *m.mass[:3])    # ndm=3 only
            print("Total mass:", fem.nodes.masses.total_mass())
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.mass_defs: list[MassDef] = []
        self.mass_records: list[MassRecord] = []

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def point(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        mass: float,
        rotational: tuple | None = None,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> PointMassDef:
        """Concentrated mass at every node of *target*.

        The same scalar ``mass`` (and optional rotational inertia
        triple) is applied to every targeted node. Useful for
        equipment, lumped fixtures, or any localised inertial
        contribution that doesn't come from a material density.

        Resolution emits one
        :class:`~apeGmsh.solvers.Masses.MassRecord` per targeted
        node, accumulated with any other mass contributions on the
        same node.

        Parameters
        ----------
        target : str or list of (dim, tag), optional
            Target node(s). See class docstring for the lookup
            order.
        pg, label, tag :
            Explicit-source overrides.
        mass : float
            Translational mass (per node) in model mass units.
        rotational : (Ixx, Iyy, Izz), optional
            Rotational inertia triple. Required for ``ndf >= 4``
            models that carry rotational DOFs. Default ``None``
            stores zero rotational inertia.
        reduction : ``"lumped"`` or ``"consistent"``, default
            ``"lumped"``
            Has no effect for point masses (already lumped) — the
            argument is accepted for API symmetry with the
            distributed factories.
        name : str, optional
            Friendly name.

        Returns
        -------
        PointMassDef

        Raises
        ------
        KeyError
            If ``target`` doesn't resolve.

        Examples
        --------
        Equipment mass on a slab corner::

            g.masses.point(
                "Equipment", mass=2500.0,
                rotational=(800.0, 800.0, 1200.0),
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(PointMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            mass=mass, rotational=rotational,
        ))

    def line(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        linear_density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> LineMassDef:
        """Distributed line mass on the curve(s) of *target*.

        ``linear_density`` is in mass per unit length (e.g. kg/m).
        Each curve element contributes ``linear_density × edge_length``
        of mass, distributed to its end nodes:

        * ``"lumped"``: split equally between the two end nodes
          (``½ × ρ_l × L`` each).
        * ``"consistent"``: shape-function-integrated 2-node line
          mass matrix ``ρ_l L / 6 · [[2, 1], [1, 2]]``. The
          diagonal-summed nodal share equals ``ρ_l L / 2`` — same
          as lumped at the diagonal level — so the visible
          difference today is only in the coupling term emitted
          for solvers that consume it.

        Use cases: cladding mass per unit length on shells, ducts
        or piping along a beam, façade panels along an edge.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Curve(s) carrying the line mass.
        pg, label, tag :
            Explicit-source overrides.
        linear_density : float
            Mass per unit length.
        reduction : ``"lumped"`` or ``"consistent"``, default
            ``"lumped"``
            Lumping scheme.
        name : str, optional
            Friendly name.

        Returns
        -------
        LineMassDef

        Raises
        ------
        KeyError
            If ``target`` doesn't resolve.

        Examples
        --------
        Curtain-wall cladding along a building edge::

            g.masses.line("PerimeterEdge", linear_density=85.0)
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(LineMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            linear_density=linear_density,
        ))

    def surface(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        areal_density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> SurfaceMassDef:
        """Distributed surface mass on the face(s) of *target*.

        ``areal_density`` is in mass per unit area (e.g. kg/m² or
        ``ρ × t`` for a shell of thickness ``t``). Each face element
        contributes ``areal_density × face_area`` of mass,
        distributed to its corner nodes:

        * ``"lumped"``: equal split among corner nodes
          (``ρ_a × A / n_corners`` each).
        * ``"consistent"``: for tri3 / quad4 with constant areal
          density the consistent diagonal sum equals lumped, so
          the resolver currently falls through to the lumped
          implementation. The separate path is reserved for
          future higher-order shells.

        Use cases: slab self-mass when modelled as a 2-D shell,
        cladding panels, water mass on a deck, pavement.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Surface(s) carrying the areal mass.
        pg, label, tag :
            Explicit-source overrides.
        areal_density : float
            Mass per unit area. For a shell of constant thickness
            ``t`` and material density ``ρ``, pass ``ρ * t``.
        reduction : ``"lumped"`` or ``"consistent"``, default
            ``"lumped"``
            Lumping scheme.
        name : str, optional
            Friendly name.

        Returns
        -------
        SurfaceMassDef

        Examples
        --------
        Concrete slab self-mass via shell areal density::

            g.masses.surface(
                "Slab",
                areal_density=2400.0 * 0.20,   # ρ·t (kg/m²)
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(SurfaceMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            areal_density=areal_density,
        ))

    def volume(
        self,
        target=None,
        *,
        pg=None, label=None, tag=None,
        density: float,
        reduction: str = "lumped",
        name: str | None = None,
    ) -> VolumeMassDef:
        """Distributed mass from material density over the volume(s)
        of *target*.

        ``density`` is in mass per unit volume (e.g. kg/m³). Each
        volume element contributes ``density × element_volume`` of
        mass, distributed to its corner nodes:

        * ``"lumped"``: equal split among the element's corner
          nodes (``ρ × V / n_corners`` each).
        * ``"consistent"``: for tet4 / hex8 with constant density
          the consistent diagonal sum equals lumped, so the
          resolver currently falls through to the lumped
          implementation. The separate path is reserved for
          future higher-order solid types.

        Pair this with a matching :meth:`loads.gravity` call for
        gravitational body weight. **Don't double-count** — see the
        class docstring on avoiding double-counting when the
        OpenSees material also carries ``rho``.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Volume(s) carrying the mass.
        pg, label, tag :
            Explicit-source overrides.
        density : float
            Material density (mass per unit volume).
        reduction : ``"lumped"`` or ``"consistent"``, default
            ``"lumped"``
            Lumping scheme.
        name : str, optional
            Friendly name.

        Returns
        -------
        VolumeMassDef

        See Also
        --------
        loads.gravity : Apply matching gravitational body weight.

        Examples
        --------
        Steel column self-mass::

            g.masses.volume("Columns", density=7850.0)
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(VolumeMassDef(
            target=t, target_source=src, name=name, reduction=reduction,
            density=density,
        ))

    @staticmethod
    def _coalesce_target(target, *, pg=None, label=None, tag=None):
        """Resolve explicit pg=/label=/tag= into (target, source) pair."""
        if tag is not None:
            return tag, "tag"
        if pg is not None:
            return pg, "pg"
        if label is not None:
            return label, "label"
        if target is not None:
            return target, "auto"
        raise ValueError(
            "One of target, pg=, label=, or tag= is required.")

    # ------------------------------------------------------------------
    # Internal: store + validate
    # ------------------------------------------------------------------

    def _add_def(self, defn: _MassT) -> _MassT:
        cfg = _DISPATCH.get(type(defn), {})
        if defn.reduction not in cfg:
            raise ValueError(
                f"{type(defn).__name__} does not support "
                f"reduction={defn.reduction!r}.  Supported: {list(cfg.keys())}"
            )
        self.mass_defs.append(defn)
        return defn

    def validate_pre_mesh(self) -> None:
        """Validate every registered mass's target can be resolved.

        Called by :meth:`Mesh.generate` before meshing so typos fail
        fast.  Raw ``(dim, tag)`` lists are skipped.
        """
        for defn in self.mass_defs:
            target = defn.target
            if not isinstance(target, str):
                continue
            self._resolve_target(target, source=defn.target_source)

    # ------------------------------------------------------------------
    # Target resolution (same lookup order as LoadsComposite)
    # ------------------------------------------------------------------

    def _resolve_target(self, target, source: str = "auto") -> list[tuple]:
        """Resolve target -> list of ``(dim, tag)`` or mesh-selection sentinel.

        Lookup order (for ``source="auto"``):
            1. Raw DimTag list -> as-is
            2. Mesh selection name
            3. Label name (Tier 1, ``_label:`` prefixed PG)
            4. Physical group name (Tier 2)
            5. Part label
        """
        import gmsh

        if isinstance(target, (list, tuple)) and len(target) > 0 \
                and isinstance(target[0], (list, tuple)):
            return [(int(d), int(t)) for d, t in target]

        if not isinstance(target, str):
            raise TypeError(
                f"target must be a string label or list of (dim, tag), "
                f"got {type(target).__name__}"
            )

        # Mesh selection name (auto only)
        if source == "auto":
            ms = getattr(self._parent, "mesh_selection", None)
            if ms is not None and hasattr(ms, "_sets"):
                for (dim, tag), info in ms._sets.items():
                    if info.get("name") == target:
                        return [("__ms__", dim, tag)]

        # Label name (Tier 1)
        if source in ("auto", "label"):
            try:
                from apeGmsh.core.Labels import add_prefix
                prefixed = add_prefix(target)
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    try:
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == prefixed:
                            ents = gmsh.model.getEntitiesForPhysicalGroup(
                                pg_dim, pg_tag)
                            return [(pg_dim, int(t)) for t in ents]
                    except Exception:
                        pass
            except Exception:
                pass

        # Physical group name (Tier 2)
        if source in ("auto", "pg"):
            try:
                for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
                    try:
                        if gmsh.model.getPhysicalName(pg_dim, pg_tag) == target:
                            ents = gmsh.model.getEntitiesForPhysicalGroup(
                                pg_dim, pg_tag)
                            return [(pg_dim, int(t)) for t in ents]
                    except Exception:
                        pass
            except Exception:
                pass

        # Part label
        if source == "auto":
            parts = getattr(self._parent, "parts", None)
            if parts is not None and hasattr(parts, "_instances"):
                inst = parts._instances.get(target)
                if inst is not None:
                    out: list = []
                    for d, ts in inst.entities.items():
                        out.extend((int(d), int(t)) for t in ts)
                    return out

        raise KeyError(
            f"Mass target {target!r} not found as label, physical group, "
            f"part label, or mesh selection."
        )

    def _target_nodes(self, target, node_map, all_nodes,
                      source: str = "auto") -> set[int]:
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            _, dim, tag = dts[0]
            ms = self._parent.mesh_selection
            info = ms._sets.get((dim, tag))
            if info is None:
                return set()
            return set(int(n) for n in info.get("node_ids", []))

        parts = getattr(self._parent, "parts", None)
        if isinstance(target, str) and parts is not None:
            if target in getattr(parts, "_instances", {}):
                if node_map is not None and target in node_map:
                    return set(node_map[target])

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

    def _target_edges(self, target, source: str = "auto") -> list[tuple[int, int]]:
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
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
                npe = 2 if int(etype) == 1 else 3
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    edges.append((int(row[0]), int(row[-1])))
        return edges

    def _target_faces(self, target, source: str = "auto") -> list[list[int]]:
        dts = self._resolve_target(target, source=source)
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
                npe = {2: 3, 3: 4, 9: 6, 16: 8}.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                corners_per = {3: 3, 4: 4, 6: 3, 8: 4}[npe]
                for row in arr:
                    faces.append([int(n) for n in row[:corners_per]])
        return faces

    def _target_elements(self, target, source: str = "auto"):
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        conns: list[np.ndarray] = []
        for d, t in dts:
            if d != 3:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype = int(etype)
                npe_map = {4: 4, 5: 8, 6: 6, 11: 10, 17: 20}
                npe = npe_map.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    conns.append(row)
        return conns

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
    ) -> MassSet:
        """Resolve all stored MassDefs into a :class:`MassSet`.

        Multiple definitions targeting overlapping nodes are
        accumulated — each node ends up with at most one
        :class:`MassRecord` whose vector is the sum of contributions.
        """
        resolver = MassResolver(
            node_tags, node_coords, elem_tags, connectivity, ndf=ndf,
        )
        all_nodes = set(int(t) for t in node_tags)

        # Per-node accumulator across all defs
        accum: dict[int, np.ndarray] = {}

        for defn in self.mass_defs:
            cfg = _DISPATCH[type(defn)]
            method_name = cfg.get(defn.reduction)
            if method_name is None:
                raise ValueError(
                    f"{type(defn).__name__} does not support "
                    f"reduction={defn.reduction!r}"
                )
            method = getattr(self, method_name)
            raw_records = method(resolver, defn, node_map, all_nodes)
            for r in raw_records:
                vec = np.asarray(r.mass, dtype=float)
                if r.node_id in accum:
                    accum[r.node_id] += vec
                else:
                    accum[r.node_id] = vec.copy()

        # Build final flattened MassRecord list (one per node)
        records: list[MassRecord] = [
            MassRecord(
                node_id=int(nid),
                mass=tuple(float(v) for v in vec),
            )
            for nid, vec in sorted(accum.items())
        ]
        self.mass_records = records
        return MassSet(records)

    # ------------------------------------------------------------------
    # Private dispatch methods
    # ------------------------------------------------------------------

    def _resolve_point(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes, source=src)
        return resolver.resolve_point_lumped(defn, nodes)

    def _resolve_line_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_lumped(defn, edges)

    def _resolve_line_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_consistent(defn, edges)

    def _resolve_surface_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces(defn.target, source=src)
        return resolver.resolve_surface_lumped(defn, faces)

    def _resolve_surface_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces(defn.target, source=src)
        return resolver.resolve_surface_consistent(defn, faces)

    def _resolve_volume_lumped(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        elements = self._target_elements(defn.target, source=src)
        return resolver.resolve_volume_lumped(defn, elements)

    def _resolve_volume_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        elements = self._target_elements(defn.target, source=src)
        return resolver.resolve_volume_consistent(defn, elements)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.mass_defs)

    def __repr__(self) -> str:
        if not self.mass_defs:
            return "MassesComposite(empty)"
        return f"MassesComposite({len(self.mass_defs)} defs)"
