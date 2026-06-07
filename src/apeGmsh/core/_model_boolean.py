from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from ._geometry_topology import sweep_dangling
from ._helpers import Tag, resolve_to_dimtags
from .Labels import pg_preserved
from apeGmsh._types import DimTag, EntityRefs

if TYPE_CHECKING:
    from .Model import Model
    from ._parts_registry import Instance, PartsRegistry


def _collect_topology_rebuild_targets(
    parts: "PartsRegistry | None",
    input_dimtags: list[DimTag],
) -> tuple[list[tuple["Instance", callable]], set[str]]:
    """Find Part instances whose topology PGs need rebuild after a boolean.

    A Part instance qualifies when:
    (a) it owns metadata identifying it as a known topology-PG host
        (currently: ``inst.properties["drm_box"]["line_pgs"]``), and
    (b) at least one of its volume tags appears in ``input_dimtags``.

    Returns
    -------
    targets
        ``[(inst, rebuild_fn), ...]`` — call each ``rebuild_fn(parent,
        inst)`` post-op to recreate the PGs from the stored
        predicate.
    skip_pg_names
        Union of all PG names the rebuild will own.  Pass to
        ``pg.skip(...)`` so the standard remap doesn't emit
        "Cannot remap" / "is now empty" warnings for these.
    """
    targets: list[tuple["Instance", callable]] = []
    skip_pg_names: set[str] = set()

    if parts is None:
        return targets, skip_pg_names

    input_vol_tags = {int(t) for d, t in input_dimtags if int(d) == 3}
    if not input_vol_tags:
        return targets, skip_pg_names

    for inst in parts._instances.values():
        drm_meta = inst.properties.get("drm_box")
        if not drm_meta:
            continue
        line_pgs = drm_meta.get("line_pgs")
        if not line_pgs:
            continue
        inst_vol_tags = set(int(t) for t in inst.entities.get(3, []))
        if not (inst_vol_tags & input_vol_tags):
            continue

        # Lazy import to avoid cycle: drm_box -> Part -> ... -> core
        from apeGmsh.parts.drm_box import rebuild_drm_box_line_pgs
        targets.append((inst, rebuild_drm_box_line_pgs))
        skip_pg_names.update(line_pgs.values())

    return targets, skip_pg_names


class _Boolean:
    """Boolean-operation sub-composite extracted from Model."""

    def __init__(self, model: "Model") -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------
    # Every boolean op returns a list[Tag] of the surviving volumes.
    # ``objects`` and ``tools`` accept any :data:`EntityRefs` form —
    # raw tags, dimtags, label names, physical-group names, or lists
    # mixing all of the above.  Resolution order for strings matches
    # :func:`resolve_to_tags`: label (Tier 1) first, then user PG
    # (Tier 2).  Identical shape to what ``g.physical.add`` accepts.

    def _bool_op(
        self,
        fn_name   : str,
        objects   : EntityRefs,
        tools     : EntityRefs,
        default_dim   : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
        label         : str | None = None,
        tolerance     : float | None = None,
    ) -> list[Tag]:
        parent = self._model._parent
        # Phase 3B.2d / ADR 0038 — every boolean operation
        # (fuse / cut / intersect / fragment) routes through ``_bool_op``;
        # one guard covers the whole composite.
        from ._compose_errors import chain_phase_guard
        chain_phase_guard(parent, f"g.model.boolean.{fn_name}")
        obj_dt  = resolve_to_dimtags(
            objects, default_dim=default_dim, session=parent,
        )
        tool_dt = resolve_to_dimtags(
            tools, default_dim=default_dim, session=parent,
        )
        fn      = getattr(gmsh.model.occ, fn_name)

        # Snapshot label names attached to inputs before the op.  Used
        # below when ``label=`` is set so we can drop the old labels
        # off the result and replace them with the new one.
        labels_comp = getattr(parent, 'labels', None)
        input_label_names: set[str] = set()
        if label is not None and labels_comp is not None:
            for d, t in obj_dt + tool_dt:
                input_label_names.update(labels_comp.labels_for_entity(d, t))

        absorbed = fn_name in ('fuse', 'intersect')

        # ── Pre-op: identify Part instances with topology-rebuild
        # hooks whose volumes are touched by this boolean.  Collect
        # their owned PG names so the post-op remap skips them
        # (the rebuild will recreate them from a stored predicate
        # against post-op geometry — OCC doesn't expose edge-level
        # parent→child lineage for the standard remap to use).
        parts = getattr(parent, 'parts', None)
        rebuild_targets, skip_pg_names = _collect_topology_rebuild_targets(
            parts, obj_dt + tool_dt,
        )

        # Opt-in tolerance override (default None = current global
        # ``Geometry.ToleranceBoolean``).  Reuses the same context
        # manager the dedup / make_conformal paths use, so solid
        # booleans get the near-coincidence lever the curve booleans
        # already had.  Function-local import mirrors the
        # ``chain_phase_guard`` idiom above and sidesteps any core
        # intra-package import-order concern.
        from ._model_queries import _temporary_tolerance
        with pg_preserved() as pg, _temporary_tolerance(
            tolerance, keys=("Geometry.ToleranceBoolean",),
        ):
            result, result_map = fn(
                obj_dt, tool_dt,
                removeObject=remove_object,
                removeTool=remove_tool,
            )
            if sync:
                gmsh.model.occ.synchronize()
            pg.set_result(
                obj_dt + tool_dt, result_map,
                result=result,
                absorbed_into_result=absorbed,
            )
            if skip_pg_names:
                pg.skip(skip_pg_names)
            if parts is not None:
                parts._remap_from_result(
                    obj_dt + tool_dt, result_map,
                    result=result,
                    absorbed_into_result=absorbed,
                )

            # Rebuild topology-derived PGs (e.g. DRM box line PGs)
            # *inside* the pg_preserved block so the rebuilt PGs are
            # created after the standard remap removed the stale
            # snapshot entries and before the user observes any
            # state.  Runs after ``_remap_from_result`` because the
            # rebuild reads ``inst.entities[3]`` to scope its
            # boundary query.
            for inst, rebuild_fn in rebuild_targets:
                rebuild_fn(parent, inst)

        # Clean up registry: remove consumed objects/tools
        result_set = set(result)
        if remove_object:
            for dt in obj_dt:
                if dt not in result_set:
                    self._model._metadata.pop(dt, None)
        if remove_tool:
            for dt in tool_dt:
                if dt not in result_set:
                    self._model._metadata.pop(dt, None)

        tags = [t for _, t in result]
        # Register only the target-dim outputs as user-intentional —
        # ``fragment`` returns ALL surviving entities of every dim
        # (split surfaces, edges, etc.) as byproducts, and stashing
        # those in ``_metadata`` would confuse :func:`sweep_dangling`
        # into preserving structural orphans (the overhang half of a
        # fragmented cutting plane).  ``cut_by_surface`` re-registers
        # the cut interface explicitly when ``keep_surface=True`` —
        # that path is the right channel for "I want this surface
        # tracked", not a side effect of the boolean.
        for d, t in result:
            if int(d) != int(default_dim):
                continue
            self._model._register(d, t, None, fn_name)

        # Apply the label override.  Strip input-side labels off the
        # result entities (per-dim, so a 3-D result does not affect a
        # surface label of the same name), then attach the new label.
        # Labels with no remaining entities after the strip are
        # removed entirely.  User-defined PGs are untouched.
        if label is not None and labels_comp is not None and result:
            result_by_dim: dict[int, list[int]] = {}
            for d, t in result:
                result_by_dim.setdefault(int(d), []).append(int(t))
            for nm in input_label_names:
                for d, res_tags in result_by_dim.items():
                    try:
                        existing = labels_comp.entities(nm, dim=d)
                    except (KeyError, ValueError):
                        continue
                    res_set = set(res_tags)
                    kept = [t for t in existing if t not in res_set]
                    if len(kept) == len(existing):
                        continue
                    labels_comp.remove(nm, dim=d)
                    if kept:
                        labels_comp.add(d, kept, name=nm)
            for d, res_tags in result_by_dim.items():
                labels_comp.add(d, res_tags, name=label)

        self._model._log(f"{fn_name}(obj={obj_dt}, tool={tool_dt}) -> tags {tags}")
        return tags

    def fuse(
        self,
        objects : EntityRefs,
        tools   : EntityRefs,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
        label         : str | None = None,
        tolerance     : float | None = None,
    ) -> list[Tag]:
        """
        Boolean union (A \u222a B).  Returns surviving volume tags.

        When ``label=`` is supplied, the labels carried by the inputs
        are dropped from the result and the new ``label`` is attached
        instead.  Without ``label=``, all input labels survive on the
        merged volume.

        ``tolerance`` optionally overrides ``Geometry.ToleranceBoolean``
        for the duration of this op (restored after) \u2014 bump it (e.g.
        ``tolerance=1e-3`` for mm-scale models) when near-coincident
        faces defeat OCC's default coincidence detection. ``None``
        (default) leaves the current global tolerance unchanged.

        Example
        -------
        ``result = g.model.boolean.fuse(box, sphere, label='merged')``
        """
        return self._bool_op(
            'fuse', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
            tolerance=tolerance,
        )

    def cut(
        self,
        objects : EntityRefs,
        tools   : EntityRefs,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
        label         : str | None = None,
        tolerance     : float | None = None,
    ) -> list[Tag]:
        """
        Boolean difference (A \u2212 B).  Returns surviving volume tags.

        When ``label=`` is supplied, the object's label is dropped
        from the result and the new ``label`` is attached instead.

        ``tolerance`` optionally overrides ``Geometry.ToleranceBoolean``
        for the duration of this op (see :meth:`fuse`); ``None``
        (default) leaves the current global tolerance unchanged.

        Example
        -------
        ``result = g.model.boolean.cut(box, cylinder, label='holey')``
        """
        return self._bool_op(
            'cut', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
            tolerance=tolerance,
        )

    def intersect(
        self,
        objects : EntityRefs,
        tools   : EntityRefs,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        sync          : bool = True,
        label         : str | None = None,
        tolerance     : float | None = None,
    ) -> list[Tag]:
        """Boolean intersection (A \u2229 B).  Returns surviving volume tags.

        When ``label=`` is supplied, the input labels are dropped from
        the intersection and the new ``label`` is attached instead.

        ``tolerance`` optionally overrides ``Geometry.ToleranceBoolean``
        for the duration of this op (see :meth:`fuse`); ``None``
        (default) leaves the current global tolerance unchanged.
        """
        return self._bool_op(
            'intersect', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
            tolerance=tolerance,
        )

    def fragment(
        self,
        objects : EntityRefs,
        tools   : EntityRefs,
        *,
        dim           : int  = 3,
        remove_object : bool = True,
        remove_tool   : bool = True,
        cleanup_free  : bool = True,
        sync          : bool = True,
        tolerance     : float | None = None,
    ) -> list[Tag]:
        """
        Boolean fragment \u2014 splits all shapes at their intersections and
        preserves all sub-volumes (useful for conformal meshing).

        Parameters
        ----------
        objects : tag(s) of the entities to fragment.
        tools : tag(s) of the cutting entities (e.g. rectangles).
            Dimensions are auto-resolved from the registry, so bare
            integer tags work even when tools have a different dimension
            than *dim*.
        dim : target dimension for bare integer tags in *objects*
            (default 3).
        remove_object, remove_tool : passed to OCC (default True).
        cleanup_free : bool, default True
            When True, run :func:`sweep_dangling` after the fragment to
            reap free-floating dim<=2 entities that bound no surviving
            volume AND are not user-intentional (not in
            ``model._metadata``, not carrying a label).  The previous
            centroid-in-bbox heuristic over-collected shell-on-solid
            geometry whose centroid happened to fall outside a volume
            bbox; the topology-driven sweep preserves any standalone
            shell the user explicitly created (``add_rectangle``,
            ``add_plane_surface``, etc.) because those entities live
            in ``_metadata``.  Default flipped to ``True`` once the
            safer sweep landed; pass ``cleanup_free=False`` only when
            you need OCC's raw output (no orphan removal, no stale-
            metadata reap) for downstream inspection.
        sync : synchronise the OCC kernel (default True).
        tolerance : float | None
            Optional override for ``Geometry.ToleranceBoolean`` during
            the fragment (see :meth:`fuse`) — raise it when shapes touch
            at near-coincident faces the default tolerance misses.
            ``None`` (default) leaves the current global tolerance
            unchanged.

        Returns
        -------
        list[Tag]
            Tags of all surviving entities at the target dimension.
        """
        result = self._bool_op(
            'fragment', objects, tools, dim,
            remove_object, remove_tool, sync,
            tolerance=tolerance,
        )

        # In a 2D-only model every surface is "free" (there ARE no
        # volumes to bound), so the sweep would destroy every surface
        # the user created.  ``sweep_dangling`` protects metadata-
        # registered surfaces by definition, but skipping the sweep
        # in the 2D-only case is cheaper and avoids any debate about
        # what "orphan" means without volumes.
        if cleanup_free and gmsh.model.getEntities(3):
            sweep_dangling(self._model)

        return result

    def conformal(
        self,
        *,
        dims     : list[int] | None = None,
        tolerance: float | None     = None,
        sync     : bool             = True,
    ):
        """
        Fragment all entities against each other to produce a conformal model.

        Convenience alias living next to :meth:`fragment` — delegates to
        :meth:`Model.queries.make_conformal`.  Use this when you want to weld
        an entire imported/arc-built model into a single connected topology
        (the whole-model "fragment everything against everything" case),
        rather than fragmenting one explicit object/tool pair.

        See :meth:`_Queries.make_conformal` for full parameter docs and the
        Part-instance renumbering caveat.
        """
        return self._model.queries.make_conformal(
            dims=dims, tolerance=tolerance, sync=sync,
        )
