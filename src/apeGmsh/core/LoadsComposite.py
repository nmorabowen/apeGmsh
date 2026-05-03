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
from typing import TYPE_CHECKING, Iterator, TypeVar

import numpy as np

if TYPE_CHECKING:
    from apeGmsh._core import apeGmsh as _ApeGmshSession

from apeGmsh.mesh._record_set import NodalLoadSet as LoadSet
from apeGmsh.solvers.Loads import (
    LoadDef,
    LoadRecord,
    LoadResolver,
    PointLoadDef,
    PointClosestLoadDef,
    LineLoadDef,
    SurfaceLoadDef,
    GravityLoadDef,
    BodyLoadDef,
    FaceLoadDef,
    FaceSPDef,
)


# (LoadDefType, reduction, target_form) -> method name on LoadsComposite
_DISPATCH: dict[type, dict[tuple[str, str], str]] = {
    PointLoadDef: {
        ("tributary",  "nodal"):   "_resolve_point",
        ("consistent", "nodal"):   "_resolve_point",
    },
    PointClosestLoadDef: {
        ("tributary",  "nodal"):   "_resolve_point_closest",
        ("consistent", "nodal"):   "_resolve_point_closest",
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
    FaceLoadDef: {
        ("tributary",  "nodal"):   "_resolve_face_load",
    },
    FaceSPDef: {
        ("tributary",  "nodal"):   "_resolve_face_sp",
    },
}

_LoadT = TypeVar("_LoadT", bound=LoadDef)


class LoadsComposite:
    """Loads composite — define + resolve loads.

    Target resolution
    -----------------
    All factory methods (``point``, ``line``, ``surface``, ``gravity``,
    ``body``, ``face_load``, ``face_sp``) accept a flexible positional
    ``target`` argument plus three explicit keyword overrides::

        g.loads.point("my_pt",     force_xyz=(0, 0, -1))   # auto
        g.loads.point(pg="my_pg",  force_xyz=(0, 0, -1))   # force PG
        g.loads.point(label="top", force_xyz=(0, 0, -1))   # force label
        g.loads.point(tag=[(0, 7)], force_xyz=(0, 0, -1))  # raw DimTag

    When the caller passes ``target=...`` (the auto path),
    :meth:`_resolve_target` tries each of these in order until one
    matches:

    ===  ========================  =============================
    #    Source                    Provided by
    ===  ========================  =============================
    1    raw ``list[(dim, tag)]``  the caller
    2    mesh selection name       ``g.mesh_selection``
    3    label (Tier 1, prefixed)  ``_label:`` physical groups
    4    physical group (Tier 2)   user-authored PGs
    5    part label                ``g.parts._instances``
    ===  ========================  =============================

    The first match wins. If two namespaces share a name (e.g. a label
    and a PG both called ``"top"``), label wins because it is checked
    first. To bypass auto resolution and pin a specific source use the
    keyword form: ``pg=`` skips straight to step 4, ``label=`` to step
    3, ``tag=`` to step 1.

    A ``KeyError`` is raised if auto resolution exhausts all five
    sources without finding the name.

    Patterns
    --------
    All load definitions inherit the ``pattern`` of the active
    :meth:`pattern` context (default ``"default"``). Group loads into
    named patterns so downstream solvers can emit one ``timeSeries`` /
    ``pattern`` block per group.
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
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

    def point(self, target=None, *, pg=None, label=None, tag=None,
              force_xyz=None, moment_xyz=None,
              name=None) -> PointLoadDef:
        """Concentrated force and/or moment applied at every node of
        *target*.

        Each node in the resolved target receives the **same** force
        and moment vectors. Use this when the load point lives on a
        named entity (a physical group, label, part, mesh selection,
        or raw `(dim, tag)` list); use :meth:`point_closest` instead
        when you only have world coordinates.

        Resolution emits one
        :class:`~apeGmsh.solvers.Loads.NodalLoadRecord` per
        targeted node onto ``fem.nodes.loads``. Both ``force_xyz``
        and ``moment_xyz`` may be supplied (or either alone), but
        at least one of the two must be non-``None`` for the load
        to do anything useful.

        Parameters
        ----------
        target : str or list of (dim, tag), optional
            Auto-resolved positional target — see the
            :class:`LoadsComposite` docstring for the lookup order.
            Pass ``pg=``, ``label=``, or ``tag=`` to bypass
            auto-resolution.
        pg, label, tag :
            Explicit-source overrides. See the class docstring.
        force_xyz : (Fx, Fy, Fz), optional
            Concentrated force vector applied at each targeted
            node, in model force units.
        moment_xyz : (Mx, My, Mz), optional
            Concentrated moment vector. For 2-D models pass a
            length-1 tuple ``(Mz,)`` — the resolver will accept it.
        name : str, optional
            Friendly name for :meth:`summary` and the viewer.

        Returns
        -------
        PointLoadDef
            The stored definition (also appended to
            ``self.load_defs``).

        Raises
        ------
        KeyError
            If ``target`` is a string that doesn't resolve to any
            of label, physical group, part, or mesh selection.
        ValueError
            If neither ``target`` nor an explicit-source kwarg is
            given.

        See Also
        --------
        point_closest : Coordinate-driven variant — snap to the
            nearest mesh node.
        face_load : Apply a centroidal force/moment to a whole
            face without rigidising it.

        Examples
        --------
        >>> with g.loads.pattern("Lateral"):
        ...     g.loads.point(
        ...         "ColTop",
        ...         force_xyz=(120e3, 0.0, 0.0),
        ...     )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(PointLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            force_xyz=force_xyz, moment_xyz=moment_xyz,
        ))

    def point_closest(self, xyz, *, within=None,
                      pg=None, label=None, tag=None,
                      force_xyz=None, moment_xyz=None,
                      tol=None, name=None) -> PointClosestLoadDef:
        """Concentrated load at the mesh node closest to ``xyz``.

        Coordinate-driven targeting — useful when the load point doesn't
        live on a named PG/label. The snap happens at :meth:`resolve`,
        and the snap distance is recorded back on the def.

        Parameters
        ----------
        xyz : (x, y, z)
            World-coordinate target.
        within : str | list, optional
            Restrict the snap pool to nodes inside this PG/label/part/
            DimTag list. ``pg=``/``label=``/``tag=`` force the source.
            Default = global (search every mesh node).
        tol : float, optional
            If given, every node within ``tol`` of ``xyz`` receives the
            load. Default ``None`` = single nearest node.
        """
        if force_xyz is None and moment_xyz is None:
            raise ValueError("point_closest() requires force_xyz or moment_xyz.")
        w_t, w_src = (None, "auto")
        if any(v is not None for v in (within, pg, label, tag)):
            w_t, w_src = self._coalesce_target(within, pg=pg, label=label, tag=tag)
        xyz_t = tuple(float(c) for c in xyz)
        return self._add_def(PointClosestLoadDef(
            target=xyz_t, target_source="closest_xyz",
            pattern=self._active_pattern, name=name,
            force_xyz=force_xyz, moment_xyz=moment_xyz,
            xyz_request=xyz_t, within=w_t, within_source=w_src, tol=tol,
        ))

    def line(self, target=None, *, pg=None, label=None, tag=None,
             magnitude=None, direction=(0., 0., -1.),
             q_xyz=None, normal=False, away_from=None,
             reduction="tributary", target_form="nodal",
             name=None) -> LineLoadDef:
        """Distributed load (force per unit length) along the curve(s)
        of *target*.

        Three ways to specify the load vector:

        * ``magnitude`` + ``direction``: scalar magnitude along a
          fixed unit vector (or axis name ``"x"``/``"y"``/``"z"``).
        * ``q_xyz``: explicit ``(qx, qy, qz)`` force-per-length
          vector.
        * ``normal=True`` + ``away_from``: edge-by-edge in-plane
          pressure. The in-plane normal is sign-flipped per edge so
          it points away from ``away_from`` (a reference point that
          represents the *source* of the load — e.g. the centre of
          an arched cavity loaded by internal pressure). Positive
          ``magnitude`` then pushes into the structure.

        For ``normal=True`` without ``away_from``, apeGmsh consults
        the parent surface's Gmsh boundary orientation to decide
        which side is "into the structure". If the curve has no
        adjacent surface, or bounds more than one, the resolver
        raises ``ValueError`` — disambiguate by passing
        ``away_from``, or fall back to ``direction``/``q_xyz``.

        Reduction and emission form
        ---------------------------
        * ``reduction="tributary"`` (default): split each edge's
          length-weighted load equally between its two end nodes.
          Emits :class:`NodalLoadRecord` on ``fem.nodes.loads``.
        * ``reduction="consistent"``: shape-function integration
          (line2 / line3) — equivalent to the FEM consistent load
          vector. Required for higher-order elements where simple
          tributary lumping is wrong.
        * ``target_form="element"``: skip nodal lumping entirely
          and emit one ``ElementLoadRecord`` per beam element with
          ``load_type="beamUniform"`` — the solver's element
          formulation handles the integration.

        Parameters
        ----------
        target : str or list of (dim, tag), optional
            Curve(s) to load.
        pg, label, tag :
            Explicit-source overrides. See class docstring.
        magnitude : float, optional
            Scalar force per unit length. Required if ``q_xyz`` is
            ``None``. Required when ``normal=True``.
        direction : tuple or {"x", "y", "z"}, default ``(0, 0, -1)``
            Unit direction for ``magnitude``. Ignored when
            ``q_xyz`` or ``normal=True`` is given.
        q_xyz : (qx, qy, qz), optional
            Explicit force-per-length vector — overrides
            ``magnitude`` × ``direction``.
        normal : bool, default False
            If ``True``, treat the load as a 2-D pressure normal
            to each edge in the xy-plane.
        away_from : (x, y, z), optional
            Reference point for ``normal=True`` direction
            disambiguation.
        reduction : ``"tributary"`` or ``"consistent"``, default
            ``"tributary"``
            How distributed loads are reduced to nodal records.
        target_form : ``"nodal"`` or ``"element"``, default
            ``"nodal"``
            Output record type. ``"element"`` skips nodal lumping
            and emits ``eleLoad``-style records.
        name : str, optional
            Friendly name.

        Returns
        -------
        LineLoadDef

        Raises
        ------
        ValueError
            If neither ``magnitude`` nor ``q_xyz`` is supplied, or
            ``normal=True`` is set without ``magnitude``.
        KeyError
            If ``target`` doesn't resolve.

        Examples
        --------
        Uniform vertical line load on a beam edge::

            g.loads.line(
                "BeamEdge",
                magnitude=-15e3,
                direction=(0, 0, -1),
            )

        Internal pressure on a curved 2-D arch::

            g.loads.line(
                "InnerArc",
                magnitude=p_int,
                normal=True,
                away_from=(0.0, 0.0, 0.0),
            )

        Element-form output for a beam carrying its own ``eleLoad``
        per element::

            g.loads.line(
                "Girder",
                magnitude=-25e3,
                direction=(0, 0, -1),
                target_form="element",
            )
        """
        if magnitude is None and q_xyz is None:
            raise ValueError("line() requires either magnitude or q_xyz.")
        if normal and magnitude is None:
            raise ValueError("line(normal=True) requires magnitude=.")
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(LineLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            magnitude=magnitude or 0.0, direction=direction, q_xyz=q_xyz,
            normal=normal, away_from=away_from,
            reduction=reduction, target_form=target_form,
        ))

    def surface(self, target=None, *, pg=None, label=None, tag=None,
                magnitude=0.0, normal=True,
                direction=(0., 0., -1.), reduction="tributary",
                target_form="nodal", name=None) -> SurfaceLoadDef:
        """Pressure or traction on the surface(s) of *target*.

        Two regimes selected by ``normal``:

        * ``normal=True`` (default): scalar pressure normal to each
          face. The face normal is computed at resolution time from
          the mesh; positive ``magnitude`` *pushes into* the face
          (i.e. acts opposite to the outward normal).
        * ``normal=False``: vector traction along ``direction``,
          independent of face orientation.

        Reduction and emission form
        ---------------------------
        * ``reduction="tributary"`` (default): split each face's
          area-weighted load equally among its corner nodes
          (tri3 / quad4 corner mass).
        * ``reduction="consistent"``: shape-function integration
          via Gauss quadrature on the curved face — required for
          tri6, quad8, quad9. For ``normal=True``, the curved
          normal at each Gauss point is used.
        * ``target_form="element"``: emit one
          ``ElementLoadRecord`` per face with
          ``load_type="surfacePressure"`` and let the solver's
          element handle integration.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Surface(s) to load.
        pg, label, tag :
            Explicit-source overrides.
        magnitude : float, default 0.0
            Pressure (force per unit area) when ``normal=True``,
            or traction magnitude along ``direction`` otherwise.
        normal : bool, default True
            ``True`` → normal pressure; ``False`` → vector
            traction.
        direction : (dx, dy, dz), default ``(0, 0, -1)``
            Unit traction direction. Ignored when ``normal=True``.
        reduction : ``"tributary"`` or ``"consistent"``, default
            ``"tributary"``
            Lumping scheme.
        target_form : ``"nodal"`` or ``"element"``, default
            ``"nodal"``
            Output record type.
        name : str, optional
            Friendly name.

        Returns
        -------
        SurfaceLoadDef

        Raises
        ------
        KeyError
            If ``target`` doesn't resolve.

        Examples
        --------
        Wind pressure on a vertical façade (positive into the
        face)::

            g.loads.surface(
                "Facade",
                magnitude=1.2e3,
                normal=True,
            )

        Vertical live load on a slab (vector traction, not
        pressure)::

            g.loads.surface(
                "Slab",
                magnitude=2.5e3,
                normal=False,
                direction=(0, 0, -1),
            )

        Higher-order pressure with consistent reduction on a
        quad8 mesh::

            g.loads.surface(
                "CurvedShell",
                magnitude=p,
                normal=True,
                reduction="consistent",
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(SurfaceLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            magnitude=magnitude, normal=normal, direction=direction,
            reduction=reduction, target_form=target_form,
        ))

    def gravity(self, target=None, *, pg=None, label=None, tag=None,
                g=(0., 0., -9.81), density=None,
                reduction="tributary", target_form="nodal",
                name=None) -> GravityLoadDef:
        """Body weight (``ρ · g``) over the volume(s) of *target*.

        Convenience wrapper over :meth:`body` for the common case of
        gravity loading. The total per-element load is
        ``density × element_volume × g_vec``, distributed to the
        element's nodes.

        Reduction and emission form
        ---------------------------
        * ``reduction="tributary"`` (default): split each element's
          weight equally among its corner nodes. Requires
          ``density``.
        * ``reduction="consistent"``: for tet4 / hex8 with constant
          density, reduces to the same per-node share as tributary
          (so behaviourally equivalent today, but the path is kept
          separate for higher-order extensions).
        * ``target_form="element"``: emit one
          ``ElementLoadRecord`` per volume element with
          ``load_type="bodyForce"`` carrying ``g`` and ``density``;
          the solver's element formulation handles integration.
          ``density=None`` is allowed in this form — the solver
          reads it from the assigned material.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Volume(s) carrying body weight.
        pg, label, tag :
            Explicit-source overrides.
        g : (gx, gy, gz), default ``(0, 0, -9.81)``
            Gravitational acceleration vector. **Unit-sensitive** —
            use ``(0, 0, -9810)`` for mm models with kg-mm-s units,
            etc.
        density : float, optional
            Material density (mass per unit volume). Required when
            ``target_form="nodal"``; optional in element form.
        reduction : ``"tributary"`` or ``"consistent"``, default
            ``"tributary"``
            Lumping scheme.
        target_form : ``"nodal"`` or ``"element"``, default
            ``"nodal"``
            Output record type.
        name : str, optional
            Friendly name.

        Returns
        -------
        GravityLoadDef

        Raises
        ------
        ValueError
            If ``density`` is missing for ``target_form="nodal"``.
        KeyError
            If ``target`` doesn't resolve.

        See Also
        --------
        body : Generic per-volume body force vector.
        masses.volume : Add the same density as nodal mass for
            inertial response (don't double-count if the OpenSees
            material already carries ``rho``).

        Examples
        --------
        Self-weight of a concrete slab (kg-m-s, ρ = 2400 kg/m³)::

            with g.loads.pattern("Dead"):
                g.loads.gravity("Slab", density=2400)

        Element-form gravity reading density from the material::

            g.loads.gravity(
                "ConcreteBlock",
                target_form="element",
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(GravityLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            g=g, density=density,
            reduction=reduction, target_form=target_form,
        ))

    def body(self, target=None, *, pg=None, label=None, tag=None,
             force_per_volume=(0., 0., 0.),
             reduction="tributary", target_form="nodal",
             name=None) -> BodyLoadDef:
        """Generic per-volume body force on the volume(s) of *target*.

        General sibling of :meth:`gravity` — accepts an arbitrary
        force-per-volume vector. The total per-element load is
        ``force_per_volume × element_volume``, distributed to the
        element's nodes.

        Use cases beyond gravity:

        * Centrifugal / rotational body force.
        * Magnetic body force in coupled-physics models.
        * Thermal expansion modelled as an equivalent body force.
        * Any prescribed loading proportional to volume.

        Parameters
        ----------
        target : str or list of (dim, tag)
            Volume(s) to load.
        pg, label, tag :
            Explicit-source overrides.
        force_per_volume : (bx, by, bz), default ``(0, 0, 0)``
            Body force vector in force per unit volume.
        reduction : ``"tributary"`` or ``"consistent"``, default
            ``"tributary"``
            Lumping scheme. ``"consistent"`` falls back to
            tributary for tet4/hex8 (same per-node share for
            constant body force).
        target_form : ``"nodal"`` or ``"element"``, default
            ``"nodal"``
            Output record type. ``"element"`` emits one
            ``ElementLoadRecord`` per volume element with
            ``load_type="bodyForce"`` and ``params={"bf": ...}``.
        name : str, optional
            Friendly name.

        Returns
        -------
        BodyLoadDef

        See Also
        --------
        gravity : Convenience wrapper for ``ρ · g`` body force.

        Examples
        --------
        Centrifugal body force ``ρ · ω² · r`` evaluated as a
        constant approximation::

            g.loads.body(
                "Rotor",
                force_per_volume=(omega**2 * rho * r_cg, 0, 0),
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(BodyLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            force_per_volume=force_per_volume,
            reduction=reduction, target_form=target_form,
        ))

    def face_load(self, target=None, *, pg=None, label=None, tag=None,
                  force_xyz=None, moment_xyz=None,
                  name=None) -> FaceLoadDef:
        """Concentrated force/moment at face centroid, distributed to nodes.

        ``force_xyz`` is split equally among all face nodes.
        ``moment_xyz`` is converted to statically equivalent nodal
        forces via least-norm distribution.

        Use this instead of a reference node when you only need to
        apply a load to a face without structural coupling.

        Parameters
        ----------
        target : str or list[(dim, tag)]
            Surface to load (PG name, label, part, or raw DimTag list).
        force_xyz : tuple, optional
            Concentrated force ``(Fx, Fy, Fz)`` at the face centroid.
        moment_xyz : tuple, optional
            Concentrated moment ``(Mx, My, Mz)`` about the face centroid.
        """
        if force_xyz is None and moment_xyz is None:
            raise ValueError("face_load() requires force_xyz or moment_xyz.")
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(FaceLoadDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            force_xyz=force_xyz, moment_xyz=moment_xyz,
        ))

    def face_sp(self, target=None, *, pg=None, label=None, tag=None,
                dofs=None, disp_xyz=None, rot_xyz=None,
                name=None) -> FaceSPDef:
        """Prescribed displacement/rotation at face centroid, mapped to nodes.

        Each face node receives ``u_i = disp_xyz + rot_xyz x r_i``
        where ``r_i`` is the arm from the face centroid to node *i*.

        When ``disp_xyz`` and ``rot_xyz`` are both ``None``, the result
        is a homogeneous fix (equivalent to ``fix()``).

        Parameters
        ----------
        target : str or list[(dim, tag)]
            Surface to constrain.
        dofs : list[int], optional
            Restraint mask (``1`` = constrained, ``0`` = free).
            Defaults to ``[1, 1, 1]``.
        disp_xyz : tuple, optional
            Prescribed translation ``(ux, uy, uz)`` at centroid.
        rot_xyz : tuple, optional
            Prescribed rotation ``(θx, θy, θz)`` about centroid.
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(FaceSPDef(
            target=t, target_source=src,
            pattern=self._active_pattern, name=name,
            dofs=dofs or [1, 1, 1],
            disp_xyz=disp_xyz, rot_xyz=rot_xyz,
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

    def _add_def(self, defn: _LoadT) -> _LoadT:
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

    def validate_pre_mesh(self) -> None:
        """Validate every registered load's target can be resolved.

        Called by :meth:`Mesh.generate` before meshing so typos fail
        fast instead of after minutes of meshing.  Raw ``(dim, tag)``
        lists are skipped — only string targets are looked up.
        """
        for defn in self.load_defs:
            target = defn.target
            if not isinstance(target, str):
                continue
            self._resolve_target(target, source=defn.target_source)

    # ------------------------------------------------------------------
    # Target resolution: convert flexible target -> DimTag list
    # ------------------------------------------------------------------

    def _resolve_target(self, target, source: str = "auto") -> list:
        """Resolve a target identifier to a list of ``(dim, tag)`` pairs.

        Lookup order (for ``source="auto"``):
            1. ``list[tuple[int, int]]``  -> as-is
            2. mesh selection name        -> entities from g.mesh_selection
            3. label name (Tier 1)        -> ``_label:``-prefixed PG
            4. physical group name (Tier 2)-> user PG
            5. part label                 -> entities from g.parts.instances

        When ``source="pg"`` only step 4 is tried.
        When ``source="label"`` only step 3 is tried.
        """
        import gmsh

        # 1. Raw DimTag list
        if isinstance(target, (list, tuple)) and len(target) > 0 \
                and isinstance(target[0], (list, tuple)):
            return [(int(d), int(t)) for d, t in target]

        if not isinstance(target, str):
            raise TypeError(
                f"target must be a string label or list of (dim, tag), "
                f"got {type(target).__name__}"
            )

        # 2. Mesh selection name (only in auto mode)
        if source == "auto":
            ms = getattr(self._parent, "mesh_selection", None)
            if ms is not None and hasattr(ms, "_sets"):
                for (dim, tag), info in ms._sets.items():
                    if info.get("name") == target:
                        return [("__ms__", dim, tag)]

        # 3. Label name (Tier 1 — _label: prefixed PG)
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

        # 4. Physical group name (Tier 2 — user PGs)
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

        # 5. Part label
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
            f"Target {target!r} not found as label, physical group, "
            f"part label, or mesh selection."
        )

    def _target_nodes(self, target, node_map, all_nodes,
                      source: str = "auto") -> set[int]:
        """Resolve target to a set of mesh node IDs."""
        dts = self._resolve_target(target, source=source)

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

    def _target_edges(self, target, source: str = "auto") -> list[tuple[int, int]]:
        """Resolve target to a list of (n1, n2) line edges."""
        dts = self._resolve_target(target, source=source)
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

    def _target_faces(self, target, source: str = "auto") -> list[list[int]]:
        """Resolve target to a list of node-id lists (one per face element)."""
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

    def _target_edges_full(self, target, source: str = "auto") -> list[list[int]]:
        """Like :meth:`_target_edges` but preserves higher-order connectivity.

        Returns a list of node-id sequences; each sequence has length 2
        (line2) or 3 (line3) depending on the mesh order.  Used for the
        consistent-reduction path where midside nodes participate.
        """
        dts = self._resolve_target(target, source=source)
        if dts and dts[0][0] == "__ms__":
            return []
        import gmsh
        edges: list[list[int]] = []
        for d, t in dts:
            if d != 1:
                continue
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                # gmsh elem types: 1 = line2, 8 = line3
                npe = {1: 2, 8: 3}.get(int(etype))
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    edges.append([int(n) for n in row])
        return edges

    def _target_faces_full(self, target, source: str = "auto") -> list[list[int]]:
        """Like :meth:`_target_faces` but preserves higher-order connectivity.

        Returns a list of node-id sequences; each has length 3/4/6/8/9
        for tri3/quad4/tri6/quad8/quad9.  Used for the
        consistent-reduction path.
        """
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
                # 2 = tri3, 3 = quad4, 9 = tri6, 16 = quad8, 10 = quad9
                npe = {2: 3, 3: 4, 9: 6, 16: 8, 10: 9}.get(int(etype))
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    faces.append([int(n) for n in row])
        return faces

    def _target_elements(self, target, source: str = "auto"):
        """Resolve target to (element_ids, connectivity_rows) for volume elements."""
        dts = self._resolve_target(target, source=source)
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
    ) -> LoadSet:
        """Resolve all stored LoadDefs into a :class:`LoadSet`."""
        resolver = LoadResolver(
            node_tags, node_coords, elem_tags, connectivity,
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
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes, source=src)
        return resolver.resolve_point(defn, nodes)

    def _snap_node_xyz(self, xyz, within, within_source, tol,
                       resolver, node_map):
        """Return ``(node_ids, min_snap_distance)`` for one xyz target.

        Restricts the candidate pool to ``within`` (resolved via the
        usual target machinery) when given. ``tol=None`` returns the
        single nearest node; ``tol > 0`` returns every node inside the
        radius.
        """
        node_tags = resolver.node_tags
        node_coords = resolver.node_coords
        if within is not None:
            all_n = set(int(t) for t in node_tags)
            wnodes = self._target_nodes(within, node_map, all_n,
                                        source=within_source)
            if not wnodes:
                raise ValueError(
                    "point_closest: 'within' resolved to 0 nodes")
            idx = np.fromiter(
                (resolver._node_to_idx[n] for n in wnodes
                 if n in resolver._node_to_idx),
                dtype=np.int64,
            )
            if idx.size == 0:
                raise ValueError(
                    "point_closest: 'within' nodes not present in resolver")
        else:
            idx = np.arange(len(node_tags), dtype=np.int64)

        target = np.asarray(xyz, dtype=np.float64)
        d2 = np.sum((node_coords[idx] - target) ** 2, axis=1)
        if tol is None:
            i = int(np.argmin(d2))
            return [int(node_tags[idx[i]])], float(np.sqrt(d2[i]))
        mask = d2 <= float(tol) * float(tol)
        if not mask.any():
            raise ValueError(
                f"point_closest: no nodes within tol={tol} of {tuple(target)}")
        sel_idx = idx[mask]
        return ([int(n) for n in node_tags[sel_idx]],
                float(np.sqrt(d2[mask].min())))

    def _resolve_point_closest(self, resolver, defn, node_map, all_nodes):
        nids, snap = self._snap_node_xyz(
            defn.xyz_request, defn.within, defn.within_source, defn.tol,
            resolver, node_map,
        )
        defn.snap_distance = snap
        if defn.tol is None and snap > 0.0:
            import warnings
            warnings.warn(
                f"point_closest({defn.xyz_request}) snapped to node "
                f"{nids[0]} at distance {snap:.6g} (no exact mesh node "
                f"at requested xyz).",
                stacklevel=4,
            )
        return resolver.resolve_point(defn, set(nids))

    def _resolve_line_tributary(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        if defn.normal:
            items = self._collect_line_normal_items(defn, src, resolver, full=False)
            return resolver.resolve_line_per_edge_tributary(defn, items)
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_tributary(defn, edges)

    def _resolve_line_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        if defn.normal:
            items = self._collect_line_normal_items(defn, src, resolver, full=True)
            return resolver.resolve_line_per_edge_consistent(defn, items)
        edges = self._target_edges_full(defn.target, source=src)
        return resolver.resolve_line_consistent(defn, edges)

    # ------------------------------------------------------------------
    # Per-edge normal-pressure helpers
    # ------------------------------------------------------------------

    def _curve_inplane_sign(self, curve_tag: int) -> int:
        """Return ±1 from Gmsh's surface boundary orientation.

        With sign ``+1`` (curve runs forward in the bounding loop), the
        in-plane "into-structure" direction at an edge with chord
        tangent ``T = (Tx, Ty, 0)`` is ``(-Ty, Tx, 0)``.  Sign ``-1``
        flips it.

        Raises ``ValueError`` when the curve has no adjacent 2-D
        surface or bounds more than one — in those cases the user
        should pass ``away_from=`` or use ``direction=``/``q_xyz=``.
        """
        import gmsh
        upward, _ = gmsh.model.getAdjacencies(1, int(curve_tag))
        surfaces = [int(s) for s in upward]
        if len(surfaces) == 0:
            raise ValueError(
                f"line(normal=True): curve {curve_tag} has no adjacent "
                f"surface. Provide away_from= or use direction=/q_xyz=."
            )
        if len(surfaces) > 1:
            raise ValueError(
                f"line(normal=True): curve {curve_tag} bounds "
                f"{len(surfaces)} surfaces — outward direction is "
                f"ambiguous. Provide away_from= or split the load."
            )
        boundary = gmsh.model.getBoundary(
            [(2, surfaces[0])], oriented=True, recursive=False)
        for _, t in boundary:
            if int(abs(t)) == int(curve_tag):
                return 1 if int(t) > 0 else -1
        return 1

    @staticmethod
    def _edge_normal_q(magnitude: float,
                       p1: np.ndarray, p2: np.ndarray,
                       sign: int | None,
                       away_from: np.ndarray | None) -> np.ndarray | None:
        """Force-per-length for normal pressure on one edge.

        Convention: ``magnitude > 0`` pushes into the structure.

        * ``away_from`` given → in-plane normal flipped to point away
          from that reference point (which is treated as the load
          source, on the *outside* of the structure).
        * Otherwise → uses Gmsh boundary ``sign`` and chord tangent.
        """
        t = p2 - p1
        L = float(np.linalg.norm(t))
        if L < 1e-30:
            return None
        Tn = t / L
        if away_from is not None:
            n = np.array([Tn[1], -Tn[0], 0.0])
            if float(np.linalg.norm(n)) < 1e-30:
                return None
            mid = 0.5 * (p1 + p2)
            if float(np.dot(mid - away_from, n)) < 0.0:
                n = -n
            return float(magnitude) * n
        # Gmsh-orientation path: into-structure = sign * (-Ty, Tx, 0)
        s = float(sign if sign is not None else 1)
        n = np.array([-s * Tn[1], s * Tn[0], 0.0])
        if float(np.linalg.norm(n)) < 1e-30:
            return None
        return float(magnitude) * n

    def _collect_line_normal_items(self, defn, src, resolver, *, full: bool):
        """Build per-edge ``(..., q_xyz)`` items for ``normal=True`` line loads.

        ``full=False`` returns ``(n1, n2, q)`` tuples (tributary path).
        ``full=True``  returns ``(node_seq, q)`` tuples (consistent path).
        """
        import gmsh
        away = (np.asarray(defn.away_from, dtype=float)
                if defn.away_from is not None else None)
        dts = self._resolve_target(defn.target, source=src)
        if dts and dts[0] and dts[0][0] == "__ms__":
            return []
        items: list = []
        for d, t in dts:
            if d != 1:
                continue
            sign = None if away is not None else self._curve_inplane_sign(int(t))
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype = int(etype)
                npe = {1: 2, 8: 3}.get(etype)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    n_first, n_last = int(row[0]), int(row[-1])
                    p1 = resolver.coords_of(n_first)
                    p2 = resolver.coords_of(n_last)
                    q = self._edge_normal_q(
                        defn.magnitude, p1, p2, sign, away)
                    if q is None:
                        continue
                    if full:
                        items.append(([int(n) for n in row], q))
                    else:
                        items.append((n_first, n_last, q))
        return items

    def _resolve_line_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        dts = self._resolve_target(defn.target, source=src)
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
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces(defn.target, source=src)
        return resolver.resolve_surface_tributary(defn, faces)

    def _resolve_surface_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces_full(defn.target, source=src)
        return resolver.resolve_surface_consistent(defn, faces)

    def _resolve_surface_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        dts = self._resolve_target(defn.target, source=src)
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
        src = getattr(defn, 'target_source', 'auto')
        _, conns = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_tributary(defn, conns)

    def _resolve_gravity_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        _, conns = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_consistent(defn, conns)

    def _resolve_gravity_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        eids, _ = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_element(defn, eids)

    def _resolve_body_tributary(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        _, conns = self._target_elements(defn.target, source=src)
        return resolver.resolve_body_tributary(defn, conns)

    def _resolve_body_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        eids, _ = self._target_elements(defn.target, source=src)
        return resolver.resolve_body_element(defn, eids)

    def _resolve_face_load(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes, source=src)
        return resolver.resolve_face_load(defn, sorted(nodes))

    def _resolve_face_sp(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes, source=src)
        return resolver.resolve_face_sp(defn, sorted(nodes))

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

    def summary(self):
        """DataFrame of the declared load intent — one row per def.

        Columns: ``kind, name, pattern, target, source, reduction,
        target_form, params``. ``params`` is a short stringified view
        of the kind-specific fields (force, magnitude, direction, ...).
        """
        import pandas as pd
        from dataclasses import fields

        _COMMON = {
            "kind", "name", "pattern", "target", "target_source",
            "reduction", "target_form",
        }

        def _fmt_target(t) -> str:
            if isinstance(t, str):
                return t
            if isinstance(t, (list, tuple)):
                return "[" + ", ".join(str(x) for x in t) + "]"
            return repr(t)

        rows: list[dict] = []
        for d in self.load_defs:
            params = {
                f.name: getattr(d, f.name)
                for f in fields(d)
                if f.name not in _COMMON
            }
            params = {k: v for k, v in params.items() if v is not None}
            rows.append({
                "kind"       : d.kind,
                "name"       : d.name or "",
                "pattern"    : d.pattern,
                "target"     : _fmt_target(d.target),
                "source"     : d.target_source,
                "reduction"  : d.reduction,
                "target_form": d.target_form,
                "params"     : ", ".join(f"{k}={v}" for k, v in params.items()),
            })

        cols = ["kind", "name", "pattern", "target", "source",
                "reduction", "target_form", "params"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    def __len__(self) -> int:
        return len(self.load_defs)

    def __repr__(self) -> str:
        if not self.load_defs:
            return "LoadsComposite(empty)"
        return (
            f"LoadsComposite({len(self.load_defs)} defs, "
            f"{len(self.patterns())} pattern(s))"
        )
