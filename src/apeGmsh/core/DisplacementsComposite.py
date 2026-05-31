"""apeGmsh.core.DisplacementsComposite — prescribed-displacement defs.

Pattern-bound single-point constraints (non-homogeneous ``sp`` under a
load pattern) — the force-free sibling of :class:`LoadsComposite`
(ADR 0050). Prescribed *motion* lives here; permanent *homogeneous*
fixes live on ``g.constraints.bc``; *forces* live on ``g.loads``.

Defs resolve to :class:`SPRecord` rows on ``fem.nodes.sp`` — the same
sink ``g.constraints.bc`` and the former ``g.loads.face_sp`` used.

v1 reuses the existing :class:`FaceSPDef` (kind ``face_sp``) for the
``surface`` verb and the new :class:`PointSPDef` (kind ``point_sp``) for
the ``point`` verb. Target resolution is **delegated** to the sibling
``g.loads`` composite, which owns the canonical gmsh target machinery
(lifting those helpers to a shared mixin is the natural future
refactor).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh._kernel.defs.loads import FaceSPDef, PointSPDef
from apeGmsh._kernel.records._loads import SPRecord
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver


class DisplacementsComposite:
    """Prescribed-displacement composite — define + resolve ``sp`` motion.

    **Ownership rule (ADR 0050).** ``g.constraints.bc`` owns permanent
    homogeneous fixes; ``g.displacements`` owns prescribed motion (a
    nonzero or time-varying value). A zero authored here is allowed —
    a pattern-bound hold — but it is your explicit choice, not a silent
    alias for ``bc``.

    Verbs
    -----
    * :meth:`surface` — rigid-body motion at a face centroid mapped to
      its nodes (was ``g.loads.face_sp``).
    * :meth:`point` — prescribed value applied verbatim at every node of
      the target.
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.disp_defs: list = []
        self.disp_records: list = []
        self._active_case: str = "default"

    # ------------------------------------------------------------------
    # Load-case grouping
    # ------------------------------------------------------------------

    @contextmanager
    def case(self, name: str) -> Iterator[None]:
        """Group subsequent prescribed-displacement defs under a load case.

        A *case* is a grouping label only (no time series, no stage); the
        OpenSees ``pattern`` is chosen on the apeSees bridge (ADR 0051).
        """
        prev = self._active_case
        self._active_case = name
        try:
            yield
        finally:
            self._active_case = prev

    # ------------------------------------------------------------------
    # Factory verbs
    # ------------------------------------------------------------------

    def surface(self, target=None, *, pg=None, label=None, tag=None,
                dofs=None, disp_xyz=None, rot_xyz=None,
                magnitude=0.0, normal=False, direction=None,
                name=None) -> FaceSPDef:
        """Prescribed rigid-body motion at a face centroid, mapped to nodes.

        Each face node receives ``u_i = disp_xyz + rot_xyz x r_i`` (plus a
        ``magnitude``-along-normal/direction contribution). When
        ``disp_xyz``, ``rot_xyz`` and ``magnitude`` are all zero / ``None``
        the result is a homogeneous fix. (Was ``g.loads.face_sp``.)

        Parameters
        ----------
        dofs : list[int], optional
            Restraint mask (``1`` = constrained). Defaults to ``[1, 1, 1]``.
        disp_xyz, rot_xyz : tuple, optional
            Prescribed translation / rotation at the face centroid.
        magnitude : float, default 0.0
            Scalar centroid translation routed by ``normal``/``direction``.
        normal : bool, default False
            Use the area-weighted face normal as the direction.
        direction : (dx, dy, dz), optional
            Explicit unit direction; mutually exclusive with ``normal=True``.
        """
        if disp_xyz is not None and magnitude != 0.0:
            raise ValueError(
                "displacements.surface(): pass either disp_xyz or magnitude, "
                "not both.")
        if normal and direction is not None:
            raise ValueError(
                "displacements.surface(): pass either normal=True or "
                "direction=, not both.")
        if magnitude != 0.0 and not normal and direction is None:
            raise ValueError(
                "displacements.surface(magnitude=...) requires normal=True or "
                "direction=(dx, dy, dz).")
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(FaceSPDef(
            target=t, target_source=src,
            pattern=self._active_case, name=name,
            dofs=dofs or [1, 1, 1],
            disp_xyz=disp_xyz, rot_xyz=rot_xyz,
            magnitude=magnitude, normal=normal, direction=direction,
        ))

    def point(self, target=None, *, pg=None, label=None, tag=None,
              dofs=None, values=None, name=None) -> PointSPDef:
        """Prescribed displacement/rotation applied directly at node(s).

        Every targeted node receives the **same** prescribed value for
        each constrained DOF — no centroid / rigid-body mapping.

        Parameters
        ----------
        dofs : list[int], optional
            Restraint mask (``1`` = constrained). Defaults to ``[1, 1, 1]``.
        values : sequence of float, optional
            Prescribed value per DOF index (aligned with ``dofs``).
            ``None`` = homogeneous (all zero) — a pattern-bound hold.
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(PointSPDef(
            target=t, target_source=src,
            pattern=self._active_case, name=name,
            dofs=dofs or [1, 1, 1],
            values=tuple(values) if values is not None else None,
        ))

    # ------------------------------------------------------------------
    # Internal: store + validate (target resolution delegated to g.loads)
    # ------------------------------------------------------------------

    def _coalesce_target(self, target, *, pg=None, label=None, tag=None):
        return self._parent.loads._coalesce_target(
            target, pg=pg, label=label, tag=tag)

    def _add_def(self, defn):
        self.disp_defs.append(defn)
        # Mirror LoadsComposite._add_def: chain-phase route (no-op for SP
        # defs today — the router returns None for unrouted kinds) + bump
        # the FEMData cache counter so a fresh get_fem_data() re-resolves.
        from apeGmsh._kernel.resolvers._chain_phase_router import (
            try_chain_phase_route,
        )
        try_chain_phase_route(self._parent, defn)
        bump = getattr(self._parent, "_bump_fem_counter", None)
        if bump is not None:
            bump()
        return defn

    def validate_pre_mesh(self) -> None:
        """Validate every registered displacement's target resolves.

        Called by ``Mesh.generate`` before meshing so typos fail fast.
        Raw ``(dim, tag)`` lists are skipped — only string targets.
        """
        loads = self._parent.loads
        for defn in self.disp_defs:
            target = defn.target
            if not isinstance(target, str):
                continue
            loads._resolve_target(target, source=defn.target_source)

    # ------------------------------------------------------------------
    # resolve()
    # ------------------------------------------------------------------

    def resolve(self, node_tags, node_coords, elem_tags=None,
                connectivity=None, *, node_map=None,
                face_map=None) -> list[SPRecord]:
        """Resolve all stored prescribed-displacement defs into SPRecords."""
        resolver = LoadResolver(node_tags, node_coords, elem_tags, connectivity)
        all_nodes = set(int(t) for t in node_tags)
        loads = self._parent.loads
        records: list[SPRecord] = []
        for defn in self.disp_defs:
            src = getattr(defn, "target_source", "auto")
            if defn.kind == "face_sp":
                nodes = loads._target_nodes(
                    defn.target, node_map, all_nodes,
                    source=src, expected_dim=2)
                faces = outwards = None
                if defn.magnitude != 0.0 and defn.normal:
                    faces = loads._target_faces(defn.target, source=src)
                    outwards = loads._face_outward_normals(faces)
                records.extend(resolver.resolve_face_sp(
                    defn, sorted(nodes), faces=faces, outwards=outwards))
            elif defn.kind == "point_sp":
                nodes = loads._target_nodes(
                    defn.target, node_map, all_nodes, source=src)
                records.extend(resolver.resolve_point_sp(defn, sorted(nodes)))
            else:
                raise ValueError(
                    f"DisplacementsComposite: unknown def kind {defn.kind!r}")
        self.disp_records = records
        return records

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_case(self, name: str) -> list:
        return [d for d in self.disp_defs if d.pattern == name]

    def cases(self) -> list[str]:
        seen: list[str] = []
        for d in self.disp_defs:
            if d.pattern not in seen:
                seen.append(d.pattern)
        return seen

    def __len__(self) -> int:
        return len(self.disp_defs)

    def __repr__(self) -> str:
        if not self.disp_defs:
            return "DisplacementsComposite(empty)"
        return (f"DisplacementsComposite({len(self.disp_defs)} defs, "
                f"{len(self.cases())} case(s))")
