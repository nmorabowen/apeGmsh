"""
LoadsComposite -- Define and resolve loads.

Two-stage pipeline mirroring :class:`ConstraintsComposite`:

1. **Define** (pre-mesh): factory methods (``point``, ``line``,
   ``surface``, ``gravity``, ``body``) store :class:`LoadDef` objects.
   ``with g.loads.case(name):`` groups loads under a named load case.
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

from apeGmsh.core.loads.defs import (
    BodyLoadDef,
    FaceLoadDef,
    GravityLoadDef,
    LineLoadDef,
    LoadDef,
    PointClosestLoadDef,
    PointLoadDef,
    SurfaceLoadDef,
)
from apeGmsh._kernel.resolvers._load_resolver import LoadResolver
from apeGmsh._kernel.record_sets import NodalLoadSet as LoadSet
from apeGmsh._kernel.records._loads import LoadRecord


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
}

_LoadT = TypeVar("_LoadT", bound=LoadDef)


class LoadsComposite:
    """Loads composite — define + resolve loads.

    Surface (dimension-indexed, ADR 0050)
    -------------------------------------
    Load verbs are grouped by the dimension of their target. Single-verb
    dimensions are plain callables; multi-verb dimensions are namespaces::

        g.loads.point.force / .moment / .force_closest / .moment_closest
        g.loads.line(...)                                  # distributed q
        g.loads.surface.pressure / .traction / .force_resultant_center_mass
        g.loads.volume(...)                               # body force
        g.loads.gravity(...)                              # self-weight

    Target resolution
    -----------------
    Every load verb accepts a flexible positional ``target`` argument
    plus three explicit keyword overrides::

        g.loads.point.force("my_pt",     force=(0, 0, -1))   # auto
        g.loads.point.force(pg="my_pg",  force=(0, 0, -1))   # force PG
        g.loads.point.force(label="top", force=(0, 0, -1))   # force label
        g.loads.point.force(tag=[(0, 7)], force=(0, 0, -1))  # raw DimTag

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

    Load cases
    ----------
    All load definitions inherit the ``pattern`` field of the active
    :meth:`case` context (default ``"default"``). A *case* is a grouping
    label only — it carries no time series and no stage. The OpenSees
    ``timeSeries`` / ``pattern`` is chosen later, on the apeSees bridge
    (ADR 0051: case on the geometry, pattern on the bridge).
    """

    def __init__(self, parent: "_ApeGmshSession") -> None:
        self._parent = parent
        self.load_defs: list[LoadDef] = []
        self.load_records: list[LoadRecord] = []
        self._active_case: str = "default"
        # Dimension-indexed verb namespaces (ADR 0050).
        self.point = _PointLoads(self)
        self.surface = _SurfaceLoads(self)

    # ------------------------------------------------------------------
    # Load-case grouping
    # ------------------------------------------------------------------

    @contextmanager
    def case(self, name: str) -> Iterator[None]:
        """Group subsequent load definitions under a named load case.

        A *case* is a grouping label only (no time series, no stage). The
        OpenSees ``timeSeries`` / ``pattern`` is chosen on the apeSees
        bridge when the case is imported (ADR 0051).

        Example
        -------
        ::

            with g.loads.case("dead"):
                g.loads.gravity("concrete", g=(0, 0, -9.81), density=2400)
                g.loads.line("beams", magnitude=-2e3, direction="z")

            with g.loads.case("live"):
                g.loads.surface.pressure("slabs", magnitude=-3e3)
        """
        prev = self._active_case
        self._active_case = name
        try:
            yield
        finally:
            self._active_case = prev

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def _point_closest_def(self, xyz, *, force_xyz=None, moment_xyz=None,
                           within=None, pg=None, label=None, tag=None,
                           tol=None, name=None) -> PointClosestLoadDef:
        """Build a coordinate-snapped concentrated-load def.

        Shared by ``g.loads.point.force_closest`` and
        ``.moment_closest``.  The snap happens at :meth:`resolve`, and
        the snap distance is recorded back on the def.
        """
        if force_xyz is None and moment_xyz is None:
            raise ValueError(
                "point.force_closest()/moment_closest() requires a "
                "force or moment vector.")
        w_t, w_src = (None, "auto")
        if any(v is not None for v in (within, pg, label, tag)):
            w_t, w_src = self._coalesce_target(within, pg=pg, label=label, tag=tag)
        xyz_t = tuple(float(c) for c in xyz)
        return self._add_def(PointClosestLoadDef(
            target=xyz_t, target_source="closest_xyz",
            pattern=self._active_case, name=name,
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
          pressure. The pressure direction is the edge normal lying
          in the plane of the loaded curves (any plane — XY, XZ, YZ,
          or arbitrary; the plane is fitted from the curve geometry),
          sign-flipped per edge so it points away from ``away_from``
          (a reference point representing the *source* of the load —
          e.g. the centre of an arched cavity loaded by internal
          pressure). Positive ``magnitude`` then pushes into the
          structure.

        For ``normal=True`` without ``away_from``, apeGmsh consults
        the parent surface's Gmsh-oriented normal + boundary loop to
        decide which side is "into the structure" (also plane-
        general). If the curve has no adjacent surface, or bounds
        more than one, the resolver raises ``ValueError`` —
        disambiguate by passing ``away_from``, or fall back to
        ``direction``/``q_xyz``. If the loaded curves are collinear
        or non-planar the in-plane normal is undefined and the
        resolver raises ``ValueError``.

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
        magnitude : float or callable, optional
            Scalar force per unit length. Required if ``q_xyz`` is
            ``None``. Required when ``normal=True``.

            May also be a **callable** ``q(xyz) -> float`` that
            receives the ``(x, y, z)`` coordinate as a length-3
            array and returns the local force-per-length — for a
            spatially varying load such as a depth-dependent ground
            / convergence pressure, e.g.
            ``magnitude=lambda p: gamma * (z_top - p[2])``. Works
            with ``normal=True`` and with ``direction=``, in every
            ``target_form``; mutually exclusive with ``q_xyz``.

            Accuracy of the varying field depends on ``reduction``:

            * ``reduction="consistent"`` — the field is integrated
              against the shape functions at the element **Gauss
              points** (:func:`integrate_edge_scaled`), i.e. the
              exact consistent load vector to quadrature order. No
              over/undershoot of the resultant; mesh-converged even
              on a coarse mesh.
            * ``reduction="tributary"`` (default) — sampled once at
              each edge **midpoint** and lumped (the tributary model
              is itself a lumping approximation). This is the
              midpoint rule: ``O(h^2)`` and exact for a linear field
              on straight edges, but it can over/undershoot the true
              ∫q on curved edges or steep gradients — pass
              ``reduction="consistent"`` or refine the mesh.
        direction : tuple or {"x", "y", "z"}, default ``(0, 0, -1)``
            Unit direction for ``magnitude``. Ignored when
            ``q_xyz`` or ``normal=True`` is given.
        q_xyz : (qx, qy, qz), optional
            Explicit force-per-length vector — overrides
            ``magnitude`` × ``direction``.
        normal : bool, default False
            If ``True``, treat the load as a pressure normal to each
            edge, acting in the plane of the loaded curves (fitted
            from the curve geometry — any plane, not just XY).
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
        if callable(magnitude) and q_xyz is not None:
            raise ValueError(
                "line(): a callable magnitude and q_xyz= are mutually "
                "exclusive — q_xyz is a fixed vector. Use the callable "
                "magnitude with normal=True or direction=."
            )
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(LineLoadDef(
            target=t, target_source=src,
            pattern=self._active_case, name=name,
            magnitude=magnitude or 0.0, direction=direction, q_xyz=q_xyz,
            normal=normal, away_from=away_from,
            reduction=reduction, target_form=target_form,
        ))

    def gravity(self, target=None, *, pg=None, label=None, tag=None,
                g=(0., 0., -9.81), density=None,
                reduction="tributary", target_form="nodal",
                name=None) -> GravityLoadDef:
        """Body weight (``ρ · g``) over the continuum region(s) of *target*.

        Convenience wrapper over :meth:`volume` for the common case of
        gravity loading. The total per-element load is
        ``density × element_measure × g_vec``, distributed to the
        element's nodes. *element_measure* is the element **volume**
        for a 3D (``dim=3``) target and its **area** for a 2D
        (``dim=2``) target — so ``density`` is mass-per-volume in a 3D
        model and mass-per-area in a 2D model. The target's dimension
        is detected from the mesh, so the same call works for both.

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
        volume : Generic per-volume body force vector.
        masses.volume : Add the same density as nodal mass for
            inertial response (don't double-count if the OpenSees
            material already carries ``rho``).

        Examples
        --------
        Self-weight of a concrete slab (kg-m-s, ρ = 2400 kg/m³)::

            with g.loads.case("Dead"):
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
            pattern=self._active_case, name=name,
            g=g, density=density,
            reduction=reduction, target_form=target_form,
        ))

    def volume(self, target=None, *, pg=None, label=None, tag=None,
               force_per_volume=(0., 0., 0.),
               reduction="tributary", target_form="nodal",
               name=None) -> BodyLoadDef:
        """Generic body force on the continuum region(s) of *target*.

        General sibling of :meth:`gravity` — accepts an arbitrary
        intensity vector. The total per-element load is
        ``force_per_volume × element_measure``, distributed to the
        element's nodes. *element_measure* is volume for a 3D target
        and area for a 2D target — so the supplied intensity is
        force-per-volume in a 3D model and force-per-area in a 2D one.

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

            g.loads.volume(
                "Rotor",
                force_per_volume=(omega**2 * rho * r_cg, 0, 0),
            )
        """
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(BodyLoadDef(
            target=t, target_source=src,
            pattern=self._active_case, name=name,
            force_per_volume=force_per_volume,
            reduction=reduction, target_form=target_form,
        ))

    def _face_load_def(self, target=None, *, pg=None, label=None, tag=None,
                       force_xyz=None, moment_xyz=None,
                       magnitude=0.0, normal=False, direction=None,
                       name=None) -> FaceLoadDef:
        """Concentrated force/moment at face centroid, distributed to nodes.

        Backs ``g.loads.surface.force_resultant_center_mass``.

        ``force_xyz`` is split equally among all face nodes.
        ``moment_xyz`` is converted to statically equivalent nodal
        forces via least-norm distribution.

        ``magnitude`` (a scalar in **Newtons**, not pressure) combined
        with ``normal=True`` or an explicit ``direction`` produces the
        equivalent total force without manually computing the face
        normal.  Sign convention: total = ``magnitude * unit_direction``,
        where ``unit_direction`` is ``+n_avg`` for ``normal=True`` or
        the normalised ``direction`` vector otherwise.  Pass
        ``magnitude=-F`` for an "into-face" (pressure-like) load.
        Combining ``magnitude`` with ``force_xyz`` is an error;
        combining with ``moment_xyz`` is fine.

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
        magnitude : float, default 0.0
            Total scalar force in Newtons.  Routed by ``normal``/
            ``direction`` to produce the equivalent ``force_xyz``.
        normal : bool, default False
            When ``True``, the area-weighted average face normal
            supplies the direction; positive ``magnitude`` pushes into
            the face.
        direction : (dx, dy, dz), optional
            Explicit unit-direction override (auto-normalised) for the
            ``magnitude`` path; mutually exclusive with ``normal=True``.

        Examples
        --------
        Symmetric pull on the two faces of an embedded crack —
        ``normal=True`` resolves a per-face physical outward via
        adjacent-tet centroids, so the **same** negative magnitude on
        both coincident entities pulls each face away from its own
        bonded body (opening the crack)::

            with m.loads.case("Open"):
                m.loads.surface.force_resultant_center_mass(
                    "Crack_normal",   magnitude=-1e3, normal=True)
                m.loads.surface.force_resultant_center_mass(
                    "Crack_inverted", magnitude=-1e3, normal=True)
        """
        _v = "surface.force_resultant_center_mass"
        nothing_set = (
            force_xyz is None and moment_xyz is None and magnitude == 0.0
        )
        if nothing_set:
            raise ValueError(
                f"{_v}() requires force, moment, or magnitude."
            )
        if force_xyz is not None and magnitude != 0.0:
            raise ValueError(
                f"{_v}(): pass either force or magnitude, not both."
            )
        if normal and direction is not None:
            raise ValueError(
                f"{_v}(): pass either normal=True or direction=, not both."
            )
        if magnitude != 0.0 and not normal and direction is None:
            raise ValueError(
                f"{_v}(magnitude=...) requires normal=True or "
                "direction=(dx, dy, dz)."
            )
        t, src = self._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._add_def(FaceLoadDef(
            target=t, target_source=src,
            pattern=self._active_case, name=name,
            force_xyz=force_xyz, moment_xyz=moment_xyz,
            magnitude=magnitude, normal=normal, direction=direction,
        ))

    @staticmethod
    def _coalesce_target(target, *, pg=None, label=None, tag=None):
        """Resolve explicit pg=/label=/tag= into (target, source) pair.

        Exactly one of ``target``/``pg=``/``label=``/``tag=`` may be
        given. Passing more than one is a hard error rather than a
        silent precedence pick: an ignored target specifier otherwise
        surfaces much later as an opaque resolution ``KeyError``.
        """
        specified = [n for n, v in (("target", target), ("pg", pg),
                                     ("label", label), ("tag", tag))
                     if v is not None]
        if len(specified) > 1:
            hint = (" (Did you mean name= for a friendly label?)"
                    if "label" in specified else "")
            raise ValueError(
                f"Conflicting target specifiers: {', '.join(specified)}. "
                f"Pass exactly one of target=, pg=, label=, or tag=. "
                f"target= auto-resolves (label -> PG -> tag); "
                f"pg=/label=/tag= force a single resolution path." + hint)
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

    def _resolve_target(self, target, source: str = "auto", *,
                        expected_dim: int | None = None) -> list:
        """Resolve a target identifier to a list of ``(dim, tag)`` pairs.

        Thin facade over :func:`apeGmsh.core._resolution.resolve_target`
        (the one shared tier engine — also used by
        :meth:`MassesComposite._resolve_target`).  Lookup order
        (``source="auto"``): raw DimTag list -> mesh selection name
        (returned as the ``[("__ms__", dim, tag)]`` sentinel) -> label
        (Tier 1, ``_label:`` PG) -> physical group (Tier 2) -> part
        label; ``source="pg"``/``"label"`` restrict to that tier.  A
        label/part is unioned across every matching dimension;
        ``expected_dim`` scopes a name-resolved target to one dim and
        fails loud if none match.  Raw DimTag lists and mesh
        selections bypass ``expected_dim`` scoping.
        """
        from apeGmsh.core._resolution import resolve_target

        return resolve_target(
            self._parent, target, source,
            expected_dim=expected_dim,
            not_found_prefix="Target",
            noun="load",
        )

    def _target_nodes(self, target, node_map, all_nodes,
                      source: str = "auto", *,
                      expected_dim: int | None = None) -> set[int]:
        """Resolve target to a set of mesh node IDs.

        ``expected_dim`` scopes a name-resolved target to one
        dimension (e.g. 2 for a face load); ``None`` (point loads)
        gathers nodes from whatever dims the name covers.
        """
        dts = self._resolve_target(target, source=source,
                                   expected_dim=expected_dim)

        # Mesh selection sentinel
        if dts and dts[0][0] == "__ms__":
            _, dim, tag = dts[0]
            ms = self._parent.mesh_selection
            info = ms._sets.get((dim, tag))
            if info is None:
                # S5: silent return set() -> fail-loud raise
                raise KeyError(
                    f"load target {target!r} resolved to mesh-selection "
                    f"sentinel ('__ms__', {dim}, {tag}), but that set is "
                    f"absent from g.mesh_selection._sets — the named "
                    f"selection is gone or the store is inconsistent. "
                    f"Refusing to silently bind this load to zero nodes "
                    f"(fail loud)."
                )
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
        dts = self._resolve_target(target, source=source, expected_dim=1)
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
        dts = self._resolve_target(target, source=source, expected_dim=2)
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
        dts = self._resolve_target(target, source=source, expected_dim=1)
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
        dts = self._resolve_target(target, source=source, expected_dim=2)
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

    def _face_outward_normals(
        self,
        faces: list[list[int]],
    ) -> list[np.ndarray] | None:
        """Per-face physical outward unit normals.

        For each face element (a list of node IDs), find an adjacent
        3-D element — one sharing at least 3 of the face's nodes — and
        return the unit vector pointing **from the volume centroid
        toward the face centroid**, i.e. out of the bonded body.

        This is the "physical outward" used by
        :meth:`apeGmsh.mesh._mesh_editing._Editing._classify_face_side`
        but evaluated per face element rather than per entity.  It
        gives the right answer on:

        * regular volume-boundary faces (matches connectivity normal),
        * tilted rectangles (where connectivity orientation isn't
          predictable),
        * embedded crack faces produced by
          :meth:`apeGmsh.mesh.editing.crack` (where the two crack
          entities share the same connectivity normal but lie on
          opposite sides of their bonded volumes — connectivity
          disagrees with physical outward on one of them).

        Returns
        -------
        list[ndarray] | None
            Length-``len(faces)`` list of length-3 unit vectors, or
            ``None`` when there are no 3-D elements in the model
            (caller should fall back to the connectivity normal).
            For any face whose adjacent volume can't be located the
            connectivity normal is returned in that slot.
        """
        import gmsh

        vol_ents = list(gmsh.model.getEntities(3))
        if not vol_ents:
            return None

        # gmsh element-type → nodes per element for the 3-D types we
        # know how to walk.  Higher-order (tet10/hex20) carry their
        # corner nodes first; using the full list still produces the
        # right centroid for adjacency, but we only need the first
        # ``corners`` nodes for the "≥ 3 shared" test.
        _NPE: dict[int, tuple[int, int]] = {
            # gmsh_type: (npe, corner_count)
            4:  (4, 4),    # tet4
            5:  (8, 8),    # hex8
            6:  (6, 6),    # prism6
            11: (10, 4),   # tet10
            17: (20, 8),   # hex20
        }

        node_coord_cache: dict[int, np.ndarray] = {}

        def _coord(nid: int) -> np.ndarray:
            v = node_coord_cache.get(nid)
            if v is None:
                v = np.asarray(
                    gmsh.model.mesh.getNode(int(nid))[0],
                    dtype=float,
                )
                node_coord_cache[nid] = v
            return v

        # node_id -> list of (centroid, corner_node_set)
        adjacency: dict[int, list[tuple[np.ndarray, frozenset[int]]]] = {}
        warned_types: set[int] = set()
        for vd, vt in vol_ents:
            try:
                etypes, _, enodes_list = gmsh.model.mesh.getElements(vd, vt)
            except Exception:
                continue
            for etype, enodes in zip(etypes, enodes_list):
                etype_i = int(etype)
                spec = _NPE.get(etype_i)
                if spec is None:
                    if etype_i not in warned_types:
                        import warnings
                        warnings.warn(
                            f"_face_outward_normals: unsupported 3-D "
                            f"element type {etype_i}; skipping for "
                            f"adjacency.",
                            stacklevel=3,
                        )
                        warned_types.add(etype_i)
                    continue
                npe, corners = spec
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for row in arr:
                    corner_ids = [int(n) for n in row[:corners]]
                    pts = np.array([_coord(nid) for nid in corner_ids])
                    centroid = pts.mean(axis=0)
                    cset = frozenset(corner_ids)
                    entry = (centroid, cset)
                    for nid in corner_ids:
                        adjacency.setdefault(nid, []).append(entry)

        outwards: list[np.ndarray] = []
        for face in faces:
            face_nodes = [int(n) for n in face]
            face_set = set(face_nodes)
            face_pts = np.array([_coord(n) for n in face_nodes])
            face_centroid = face_pts.mean(axis=0)
            # Connectivity normal — fallback when adjacency is missing
            # and reference for the sign-flip check below.
            p0, p1, p2 = face_pts[0], face_pts[1], face_pts[2]
            n_conn = np.cross(p1 - p0, p2 - p0)
            nn = float(np.linalg.norm(n_conn))
            n_conn = n_conn / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])

            outward = n_conn  # default
            # Probe candidate volumes via any node of the face
            seen: set[frozenset[int]] = set()
            for nid in face_nodes:
                for centroid, cset in adjacency.get(nid, []):
                    if cset in seen:
                        continue
                    seen.add(cset)
                    if len(cset & face_set) >= 3:
                        d = face_centroid - centroid
                        d_norm = float(np.linalg.norm(d))
                        if d_norm < 1e-30:
                            continue
                        outward = (
                            n_conn
                            if float(np.dot(n_conn, d)) >= 0.0
                            else -n_conn
                        )
                        break
                else:
                    continue
                break
            outwards.append(outward)

        return outwards

    def _target_elements(self, target, source: str = "auto"):
        """Resolve *target* to ``(element_ids, connectivity_rows, dim)``.

        Continuum elements only — volume (``dim=3``) or surface
        (``dim=2``). The dimension is the highest continuum dim the
        target covers (3 if any volume is present, else 2). Gravity /
        body loads use *dim* to read the supplied intensity as
        per-volume (3D) or per-area (2D); a 3D target therefore keeps
        the volume path unchanged.
        """
        dts = self._resolve_target(target, source=source, expected_dim=None)
        if dts and dts[0][0] == "__ms__":
            return [], [], None
        cont = sorted({d for d, _ in dts if d in (2, 3)})
        if not cont:
            found = sorted({d for d, _ in dts})
            raise ValueError(
                f"Target {target!r} resolved to dimension(s) {found}, "
                f"but gravity/body loads require a continuum target "
                f"(a surface dim=2 or volume dim=3 region)."
            )
        dim = cont[-1]
        # gmsh element types — 2D: 2=tri3, 3=quad4, 9=tri6, 16=quad8,
        # 10=quad9; 3D: 4=tet4, 5=hex8, 6=prism6, 11=tet10, 17=hex20
        npe_map = (
            {2: 3, 3: 4, 9: 6, 16: 8, 10: 9}
            if dim == 2 else
            {4: 4, 5: 8, 6: 6, 11: 10, 17: 20}
        )
        import gmsh
        eids: list[int] = []
        conns: list[np.ndarray] = []
        for d, t in dts:
            if d != dim:
                continue
            try:
                etypes, etags_list, enodes_list = gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, etags, enodes in zip(etypes, etags_list, enodes_list):
                etype = int(etype)
                npe = npe_map.get(etype, None)
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                for tag, row in zip(etags, arr):
                    eids.append(int(tag))
                    conns.append(row)
        return eids, conns, dim

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
        if callable(defn.magnitude) and defn.q_xyz is None:
            items = self._collect_line_vector_items(defn, src, resolver, full=False)
            return resolver.resolve_line_per_edge_tributary(defn, items)
        edges = self._target_edges(defn.target, source=src)
        return resolver.resolve_line_tributary(defn, edges)

    def _resolve_line_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        if callable(defn.magnitude) and defn.q_xyz is None:
            # Gauss-accurate: integrate the varying field at the
            # element Gauss points (no midpoint over/undershoot).
            if defn.normal:
                items = self._collect_line_normal_items(
                    defn, src, resolver, full=True, varying=True)
            else:
                items = self._collect_line_vector_items(
                    defn, src, resolver, full=True, varying=True)
            return resolver.resolve_line_per_edge_consistent_varying(
                defn, items)
        if defn.normal:
            items = self._collect_line_normal_items(defn, src, resolver, full=True)
            return resolver.resolve_line_per_edge_consistent(defn, items)
        edges = self._target_edges_full(defn.target, source=src)
        return resolver.resolve_line_consistent(defn, edges)

    # ------------------------------------------------------------------
    # Per-edge normal-pressure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
        """Unit normal of the plane the loaded curves lie in.

        Least-squares (SVD) plane fit over the supplied points.  The
        **sign** of the returned normal is arbitrary — callers that need
        a definite orientation must disambiguate it themselves (via
        ``away_from`` or an adjacent surface).

        Raises ``ValueError`` when the points are collinear (no unique
        plane) or not coplanar (a genuinely 3-D space curve) — in those
        cases an "in-plane normal" is undefined and the caller should
        use ``direction=`` or ``q_xyz=``.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        _, s, vt = np.linalg.svd(pts - pts.mean(axis=0))
        scale = float(s[0]) if s[0] > 0.0 else 1.0
        if float(s[1]) < 1e-7 * scale:
            raise ValueError(
                "line(normal=True): the loaded curve(s) are collinear, "
                "so the in-plane pressure direction is undefined. Pass "
                "an away_from= point off the line, or use "
                "direction=/q_xyz=."
            )
        if float(s[2]) > 1e-4 * scale:
            raise ValueError(
                "line(normal=True): the loaded curve(s) are not planar "
                "(a 3-D space curve); a single in-plane normal is "
                "undefined. Use q_xyz=, or split into planar groups."
            )
        n = vt[2]
        return n / float(np.linalg.norm(n))

    @staticmethod
    def _surface_unit_normal(surf_tag: int) -> np.ndarray:
        """Gmsh-oriented unit normal of a (planar) surface.

        Sampled at the parametric-domain midpoint.  For a planar face
        the normal is constant so the sample point is irrelevant; the
        orientation is Gmsh's, which is what the boundary-loop ``sign``
        in :meth:`_curve_inplane_frame` is relative to.
        """
        import gmsh
        (umin, vmin), (umax, vmax) = gmsh.model.getParametrizationBounds(
            2, int(surf_tag))
        nrm = gmsh.model.getNormal(
            int(surf_tag),
            [0.5 * (umin + umax), 0.5 * (vmin + vmax)],
        )
        n = np.asarray(nrm[:3], dtype=float)
        L = float(np.linalg.norm(n))
        return n / L if L > 1e-30 else np.array([0.0, 0.0, 1.0])

    def _curve_inplane_frame(
        self, curve_tag: int,
    ) -> tuple[int, np.ndarray]:
        """``(sign, surface_normal)`` from the curve's adjacent surface.

        ``sign`` is ``+1`` when the curve runs forward in the bounding
        loop, ``-1`` otherwise.  The in-plane "into-structure" direction
        at an edge with chord tangent ``T`` is ``sign * (P x T)`` where
        ``P`` is the Gmsh-oriented face normal — the plane-general form
        of the historical XY-only ``sign * (-Ty, Tx, 0)``.

        Raises ``ValueError`` when the curve has no adjacent 2-D surface
        or bounds more than one — the user should pass ``away_from=`` or
        use ``direction=``/``q_xyz=``.
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
        p_surf = self._surface_unit_normal(surfaces[0])
        sign = 1
        boundary = gmsh.model.getBoundary(
            [(2, surfaces[0])], oriented=True, recursive=False)
        for _, t in boundary:
            if int(abs(t)) == int(curve_tag):
                sign = 1 if int(t) > 0 else -1
                break
        return sign, p_surf

    @staticmethod
    def _edge_normal_q(magnitude: float,
                       p1: np.ndarray, p2: np.ndarray,
                       plane_normal: np.ndarray,
                       sign: int | None,
                       away_from: np.ndarray | None) -> np.ndarray | None:
        """Force-per-length for normal pressure on one edge.

        Plane-agnostic: the in-plane normal is ``T x P`` where ``T`` is
        the chord tangent and ``P`` (``plane_normal``) is the plane the
        loaded curves lie in.  For a model in the XY plane (``P = +z``)
        this reduces **exactly** to the historical ``(Ty, -Tx, 0)``
        formula.

        Convention: ``magnitude > 0`` pushes into the structure.

        * ``away_from`` given → normal flipped to point away from that
          reference point (the load source, outside the structure);
          ``plane_normal``'s sign is irrelevant.
        * Otherwise → ``plane_normal`` is the adjacent surface's
          Gmsh-oriented normal and the into-structure direction is
          ``sign * (P x T)``.
        """
        t = p2 - p1
        L = float(np.linalg.norm(t))
        if L < 1e-30:
            return None
        Tn = t / L
        P = np.asarray(plane_normal, dtype=float)
        if away_from is not None:
            n = np.cross(Tn, P)
            ln = float(np.linalg.norm(n))
            if ln < 1e-12:
                return None
            n /= ln
            mid = 0.5 * (p1 + p2)
            if float(np.dot(mid - away_from, n)) < 0.0:
                n = -n
            return float(magnitude) * n
        # Surface path: into-structure = sign * (P x T)
        s = float(sign if sign is not None else 1)
        n = s * np.cross(P, Tn)
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            return None
        return float(magnitude) * (n / ln)

    @staticmethod
    def _eval_magnitude(magnitude, xyz) -> float:
        """Scalar force-per-length at *xyz*.

        ``magnitude`` is a constant float, or a callable receiving the
        ``(x, y, z)`` coordinate as a length-3 NumPy array and returning
        a scalar — used for spatially varying line loads (e.g. a
        depth-dependent ground / convergence pressure). The callable is
        sampled once per edge, at the edge midpoint.
        """
        if not callable(magnitude):
            return float(magnitude)
        p = np.asarray(xyz, dtype=float)
        try:
            val = float(magnitude(p))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"line(): the magnitude callable must accept a length-3 "
                f"coordinate and return a scalar; it failed at point "
                f"{tuple(p)} ({exc})."
            ) from exc
        if not np.isfinite(val):
            raise ValueError(
                f"line(): the magnitude callable returned a non-finite "
                f"value ({val}) at point {tuple(p)}."
            )
        return val

    def _iter_curve_edges(self, defn, src, resolver):
        """Yield ``(etag, node_row, n_first, n_last, p1, p2, curve_tag)``
        for every 1-D mesh element on the load's resolved target curves.

        Shared by the per-edge line-load collectors (normal pressure,
        spatially varying magnitude, element form) so the Gmsh element
        walk lives in one place.
        """
        import gmsh
        dts = self._resolve_target(defn.target, source=src, expected_dim=1)
        if dts and dts[0] and dts[0][0] == "__ms__":
            return
        for d, t in dts:
            if d != 1:
                continue
            try:
                etypes, etags_list, enodes_list = \
                    gmsh.model.mesh.getElements(d, t)
            except Exception:
                continue
            for etype, etags, enodes in zip(
                    etypes, etags_list, enodes_list):
                npe = {1: 2, 8: 3}.get(int(etype))
                if npe is None:
                    continue
                arr = np.asarray(enodes, dtype=np.int64).reshape(-1, npe)
                tags = np.asarray(etags, dtype=np.int64)
                for eid, row in zip(tags, arr):
                    n_first, n_last = int(row[0]), int(row[-1])
                    yield (int(eid), row, n_first, n_last,
                           resolver.coords_of(n_first),
                           resolver.coords_of(n_last), int(t))

    def _collect_line_normal_items(self, defn, src, resolver, *,
                                   full: bool, varying: bool = False):
        """Build per-edge items for ``normal=True`` line loads.

        * ``full=False`` → ``(n1, n2, q)``     tributary path.
        * ``full=True``  → ``(node_seq, q)``   consistent path.
        * ``varying=True`` (consistent only) → ``(node_seq, n_hat,
          scalar_fn)``: the per-edge **unit** in-plane normal plus the
          scalar magnitude callable, so the resolver can Gauss-sample
          the field instead of using one midpoint value.

        For the non-varying paths a callable ``defn.magnitude`` is
        sampled once at the edge midpoint (constant per edge).
        """
        away = (np.asarray(defn.away_from, dtype=float)
                if defn.away_from is not None else None)
        edges = list(self._iter_curve_edges(defn, src, resolver))
        if not edges:
            return []
        mag_fn = (lambda p: self._eval_magnitude(defn.magnitude, p))

        # Plane normal.  away path → one fit over all loaded points
        # (falling back to include away_from when the curves are a
        # single straight segment); surface path → per-curve from the
        # adjacent face's Gmsh-oriented normal.
        plane_global = None
        surf_cache: dict[int, tuple[int, np.ndarray]] = {}
        if away is not None:
            pts = [p for _e, _r, _nf, _nl, p1, p2, _c in edges
                   for p in (p1, p2)]
            fit_pts = np.asarray(pts, dtype=float)
            try:
                plane_global = self._fit_plane_normal(fit_pts)
            except ValueError:
                plane_global = self._fit_plane_normal(
                    np.vstack([fit_pts, away.reshape(1, 3)]))

        items: list = []
        for _eid, row, n_first, n_last, p1, p2, ctag in edges:
            if away is not None:
                P, sign = plane_global, None
            else:
                if ctag not in surf_cache:
                    surf_cache[ctag] = self._curve_inplane_frame(ctag)
                sign, P = surf_cache[ctag]
            if varying:
                n_hat = self._edge_normal_q(1.0, p1, p2, P, sign, away)
                if n_hat is None:
                    continue
                items.append(([int(n) for n in row], n_hat, mag_fn))
                continue
            mag = self._eval_magnitude(
                defn.magnitude, 0.5 * (p1 + p2))
            q = self._edge_normal_q(mag, p1, p2, P, sign, away)
            if q is None:
                continue
            if full:
                items.append(([int(n) for n in row], q))
            else:
                items.append((n_first, n_last, q))
        return items

    def _collect_line_vector_items(self, defn, src, resolver, *,
                                   full: bool, varying: bool = False):
        """Per-edge items for a non-``normal`` line load whose
        ``magnitude`` is a callable.

        ``q = magnitude(edge_midpoint) * direction`` — the direction is
        used verbatim (axis name or vector), matching the uniform
        ``resolve_line_*`` path so a constant callable reproduces it
        exactly.

        ``varying=True`` (consistent path only) instead yields
        ``(node_seq, dvec, scalar_fn)`` so the resolver can Gauss-sample
        the field rather than use a single midpoint value.
        """
        from .._kernel.resolvers._load_resolver import _direction_vec
        dvec = _direction_vec(defn.direction)
        mag_fn = (lambda p: self._eval_magnitude(defn.magnitude, p))
        items: list = []
        for _eid, row, n_first, n_last, p1, p2, _ctag in \
                self._iter_curve_edges(defn, src, resolver):
            if varying:
                items.append(([int(n) for n in row], dvec, mag_fn))
                continue
            mag = self._eval_magnitude(defn.magnitude, 0.5 * (p1 + p2))
            q = mag * dvec
            if full:
                items.append(([int(n) for n in row], q))
            else:
                items.append((n_first, n_last, q))
        return items

    def _resolve_line_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        if callable(defn.magnitude) and defn.q_xyz is None:
            from .._kernel.resolvers._load_resolver import _direction_vec
            dvec = _direction_vec(defn.direction)
            items: list = []
            for eid, _row, _nf, _nl, p1, p2, _ct in \
                    self._iter_curve_edges(defn, src, resolver):
                mag = self._eval_magnitude(
                    defn.magnitude, 0.5 * (p1 + p2))
                q = mag * dvec
                items.append(
                    (eid, (float(q[0]), float(q[1]), float(q[2]))))
            return resolver.resolve_line_element_varying(defn, items)
        dts = self._resolve_target(defn.target, source=src, expected_dim=1)
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
        # Only normal pressure needs the per-face physical outward; shear
        # projects against the (sign-independent) connectivity normal in
        # the resolver, and traction is orientation-free.
        outwards = (
            self._face_outward_normals(faces)
            if defn.mode == "pressure" else None
        )
        return resolver.resolve_surface_tributary(
            defn, faces, outwards=outwards,
        )

    def _resolve_surface_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        faces = self._target_faces_full(defn.target, source=src)
        return resolver.resolve_surface_consistent(defn, faces)

    def _resolve_surface_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        dts = self._resolve_target(defn.target, source=src, expected_dim=2)
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
        _, conns, dim = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_tributary(defn, conns, dim or 3)

    def _resolve_gravity_consistent(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        _, conns, dim = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_consistent(defn, conns, dim or 3)

    def _resolve_gravity_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        eids, _, _ = self._target_elements(defn.target, source=src)
        return resolver.resolve_gravity_element(defn, eids)

    def _resolve_body_tributary(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        _, conns, dim = self._target_elements(defn.target, source=src)
        return resolver.resolve_body_tributary(defn, conns, dim or 3)

    def _resolve_body_element(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        eids, _, _ = self._target_elements(defn.target, source=src)
        return resolver.resolve_body_element(defn, eids)

    def _resolve_face_load(self, resolver, defn, node_map, all_nodes):
        src = getattr(defn, 'target_source', 'auto')
        nodes = self._target_nodes(defn.target, node_map, all_nodes,
                                   source=src, expected_dim=2)
        faces: list[list[int]] | None = None
        outwards: list[np.ndarray] | None = None
        if defn.magnitude != 0.0 and defn.normal:
            # Only the normal-magnitude path needs per-element area
            # and connectivity-derived normals; force_xyz / direction /
            # moment_xyz are geometry-free.
            faces = self._target_faces(defn.target, source=src)
            outwards = self._face_outward_normals(faces)
        return resolver.resolve_face_load(
            defn, sorted(nodes), faces=faces, outwards=outwards,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def by_case(self, name: str) -> list[LoadDef]:
        return [d for d in self.load_defs if d.pattern == name]

    def cases(self) -> list[str]:
        seen: list[str] = []
        for d in self.load_defs:
            if d.pattern not in seen:
                seen.append(d.pattern)
        return seen

    def summary(self):
        """DataFrame of the declared load intent — one row per def.

        Columns: ``kind, name, case, target, source, reduction,
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
                "case"       : d.pattern,
                "target"     : _fmt_target(d.target),
                "source"     : d.target_source,
                "reduction"  : d.reduction,
                "target_form": d.target_form,
                "params"     : ", ".join(f"{k}={v}" for k, v in params.items()),
            })

        cols = ["kind", "name", "case", "target", "source",
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
            f"{len(self.cases())} case(s))"
        )


# ----------------------------------------------------------------------
# Dimension-indexed verb namespaces (ADR 0050)
# ----------------------------------------------------------------------

class _PointLoads:
    """Concentrated nodal loads — the ``g.loads.point`` namespace.

    Force and moment, each on a named target or snapped to the nearest
    mesh node (``*_closest``).  All four verbs accept the flexible
    ``target`` plus ``pg=``/``label=``/``tag=`` overrides documented on
    :class:`LoadsComposite`.
    """
    __slots__ = ("_c",)

    def __init__(self, composite: "LoadsComposite") -> None:
        self._c = composite

    def force(self, target=None, force=None, *, pg=None, label=None,
              tag=None, name=None) -> PointLoadDef:
        """Concentrated force applied at every node of *target*."""
        t, src = self._c._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._c._add_def(PointLoadDef(
            target=t, target_source=src,
            pattern=self._c._active_case, name=name,
            force_xyz=force,
        ))

    def moment(self, target=None, moment=None, *, pg=None, label=None,
               tag=None, name=None) -> PointLoadDef:
        """Concentrated moment applied at every node of *target*.

        For 2-D models pass a length-1 tuple ``(Mz,)``.
        """
        t, src = self._c._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._c._add_def(PointLoadDef(
            target=t, target_source=src,
            pattern=self._c._active_case, name=name,
            moment_xyz=moment,
        ))

    def force_closest(self, xyz, force=None, *, within=None, pg=None,
                      label=None, tag=None, tol=None,
                      name=None) -> PointClosestLoadDef:
        """Concentrated force at the mesh node(s) closest to ``xyz``.

        ``within`` (+ ``pg=``/``label=``/``tag=``) restricts the snap
        pool; ``tol`` (if given) loads every node within that radius.
        """
        return self._c._point_closest_def(
            xyz, force_xyz=force, within=within,
            pg=pg, label=label, tag=tag, tol=tol, name=name)

    def moment_closest(self, xyz, moment=None, *, within=None, pg=None,
                       label=None, tag=None, tol=None,
                       name=None) -> PointClosestLoadDef:
        """Concentrated moment at the mesh node(s) closest to ``xyz``."""
        return self._c._point_closest_def(
            xyz, moment_xyz=moment, within=within,
            pg=pg, label=label, tag=tag, tol=tol, name=name)


class _SurfaceLoads:
    """Surface loads — the ``g.loads.surface`` namespace.

    Per-area **fields** (:meth:`pressure`, :meth:`traction`) carry the
    ``reduction`` / ``target_form`` knobs; the lumped
    :meth:`force_resultant_center_mass` does not.  (``shear`` — strict
    in-plane traction — lands in ADR 0050 P3.)
    """
    __slots__ = ("_c",)

    def __init__(self, composite: "LoadsComposite") -> None:
        self._c = composite

    def pressure(self, target=None, magnitude=0.0, *, pg=None, label=None,
                 tag=None, reduction="tributary", target_form="nodal",
                 name=None) -> SurfaceLoadDef:
        """Scalar pressure normal to each face.

        Positive ``magnitude`` pushes *into* the face (opposite the
        outward normal).  The normal is computed from the mesh at
        resolution time.
        """
        t, src = self._c._coalesce_target(target, pg=pg, label=label, tag=tag)
        return self._c._add_def(SurfaceLoadDef(
            target=t, target_source=src,
            pattern=self._c._active_case, name=name,
            magnitude=magnitude, mode="pressure",
            reduction=reduction, target_form=target_form,
        ))

    def traction(self, target=None, vector=(0., 0., -1.), *, pg=None,
                 label=None, tag=None, reduction="tributary",
                 target_form="nodal", name=None) -> SurfaceLoadDef:
        """Vector traction per unit area, in **global** coordinates.

        Independent of face orientation — e.g. a slab live load that is
        always ``(0, 0, -w)`` regardless of how the face is tilted.
        """
        t, src = self._c._coalesce_target(target, pg=pg, label=label, tag=tag)
        v = tuple(float(x) for x in vector)
        mag = float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
        return self._c._add_def(SurfaceLoadDef(
            target=t, target_source=src,
            pattern=self._c._active_case, name=name,
            magnitude=mag, mode="traction", direction=v,
            reduction=reduction, target_form=target_form,
        ))

    def shear(self, target=None, vector=(1., 0., 0.), *, pg=None,
              label=None, tag=None, reduction="tributary",
              name=None) -> SurfaceLoadDef:
        """Strict **in-plane** (tangential) traction per unit area.

        ``vector`` is a global reference traction; on each face only its
        **in-plane** component is applied — the normal component is
        projected out against the face's tangent plane at resolve time,
        so one call works across a faceted/curved surface. A face where
        the projection vanishes (the vector is normal there) is
        fail-loud.

        Has no ``target_form="element"`` — OpenSees ``surfacePressure``
        is normal-only, so an in-plane shear has no element-load form;
        use the default nodal reduction.
        """
        t, src = self._c._coalesce_target(target, pg=pg, label=label, tag=tag)
        v = tuple(float(x) for x in vector)
        mag = float(np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
        return self._c._add_def(SurfaceLoadDef(
            target=t, target_source=src,
            pattern=self._c._active_case, name=name,
            magnitude=mag, mode="shear", direction=v,
            reduction=reduction, target_form="nodal",
        ))

    def force_resultant_center_mass(
            self, target=None, *, force=None, moment=None,
            magnitude=0.0, normal=False, direction=None,
            pg=None, label=None, tag=None, name=None) -> FaceLoadDef:
        """Total force/moment at the face area-centroid, lumped to nodes.

        A **resultant**, not a per-area field: ``force`` is split equally
        among the face nodes and ``moment`` becomes statically equivalent
        nodal forces.  A scalar ``magnitude`` (Newtons) routed by
        ``normal=True`` or ``direction=`` produces the equivalent total
        without computing the face normal by hand.
        """
        return self._c._face_load_def(
            target, pg=pg, label=label, tag=tag,
            force_xyz=force, moment_xyz=moment,
            magnitude=magnitude, normal=normal, direction=direction,
            name=name)
