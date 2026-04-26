from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from ._helpers import Tag, resolve_to_dimtags
from .Labels import pg_preserved, cleanup_label_pgs
from apeGmsh._types import EntityRefs

if TYPE_CHECKING:
    from .Model import Model


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
    ) -> list[Tag]:
        parent = self._model._parent
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
        with pg_preserved() as pg:
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
            parts = getattr(parent, 'parts', None)
            if parts is not None:
                parts._remap_from_result(
                    obj_dt + tool_dt, result_map,
                    result=result,
                    absorbed_into_result=absorbed,
                )

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
        for d, t in result:
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
    ) -> list[Tag]:
        """
        Boolean union (A \u222a B).  Returns surviving volume tags.

        When ``label=`` is supplied, the labels carried by the inputs
        are dropped from the result and the new ``label`` is attached
        instead.  Without ``label=``, all input labels survive on the
        merged volume.

        Example
        -------
        ``result = g.model.boolean.fuse(box, sphere, label='merged')``
        """
        return self._bool_op(
            'fuse', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
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
    ) -> list[Tag]:
        """
        Boolean difference (A \u2212 B).  Returns surviving volume tags.

        When ``label=`` is supplied, the object's label is dropped
        from the result and the new ``label`` is attached instead.

        Example
        -------
        ``result = g.model.boolean.cut(box, cylinder, label='holey')``
        """
        return self._bool_op(
            'cut', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
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
    ) -> list[Tag]:
        """Boolean intersection (A \u2229 B).  Returns surviving volume tags.

        When ``label=`` is supplied, the input labels are dropped from
        the intersection and the new ``label`` is attached instead.
        """
        return self._bool_op(
            'intersect', objects, tools, dim,
            remove_object, remove_tool, sync, label=label,
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
        cleanup_free : bool
            When True (default), remove any "free" surfaces that do not
            bound a volume after the fragment operation.  This cleans up
            exterior remnants of cutting planes that fall outside the
            solid.  Set to False to keep all surface fragments.
        sync : synchronise the OCC kernel (default True).

        Returns
        -------
        list[Tag]
            Tags of all surviving entities at the target dimension.
        """
        result = self._bool_op(
            'fragment', objects, tools, dim,
            remove_object, remove_tool, sync,
        )

        # ``cleanup_free`` removes dim=2 surfaces that have no upward
        # adjacency to a volume — useful after a 3D fragment to drop
        # stray cutting-plane remnants. In a 2D-only model every
        # surface has no volume neighbour (there ARE no volumes), so
        # the sweep would destroy every surface in the model. Skip
        # the cleanup when no 3D entities exist.
        if cleanup_free and gmsh.model.getEntities(3):
            free: list[tuple[int, int]] = []
            for _, tag_s in gmsh.model.getEntities(2):
                up, _ = gmsh.model.getAdjacencies(2, tag_s)
                if len(up) == 0:
                    free.append((2, tag_s))
            if free:
                gmsh.model.occ.remove(free, recursive=True)
                if sync:
                    gmsh.model.occ.synchronize()
                for dt in free:
                    self._model._metadata.pop(dt, None)
                cleanup_label_pgs(free)
                self._model._log(
                    f"fragment cleanup: removed {len(free)} free surface(s)"
                )

        return result
