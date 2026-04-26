"""Fragmentation mixin for PartsRegistry — fragment_all, fragment_pair, fuse_group."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import gmsh

from .Labels import pg_preserved
from apeGmsh._types import DimTag

if TYPE_CHECKING:
    from ._parts_registry import Instance


class _PartsFragmentationMixin:
    """Mixin providing fragment/fuse operations on tracked instances.

    Expects ``self._instances``, ``self._parent``, and
    ``self._compute_bbox`` from the host class.
    """

    # Host-provided attribute contract (supplied by PartsRegistry).
    # These are declared here without assignment so mypy knows they
    # exist on ``self`` inside this mixin's methods.
    if TYPE_CHECKING:
        _instances: dict[str, "Instance"]
        _parent: Any

        @staticmethod
        def _compute_bbox(
            dimtags: list[DimTag],
        ) -> tuple[float, float, float, float, float, float] | None: ...

    def _remap_from_result(
        self,
        input_ents: list[DimTag],
        result_map: list[list[DimTag]],
        *,
        result: list[DimTag] | None = None,
        absorbed_into_result: bool = False,
    ) -> None:
        """Rewrite ``Instance.entities`` using an OCC boolean's result_map.

        ``input_ents[i]`` corresponds to ``result_map[i]`` — the old
        dimtag and the list of new dimtags that replaced it.  We build
        a per-dim ``old_tag -> [new_tag, ...]`` table (matching each
        input dimtag to outputs at the *same* dim so surface
        entries do not absorb new volumes and vice versa) and apply it
        to every tracked Instance in place.  Tags that are not present
        in any Instance are silently ignored.

        For ``fuse``/``intersect`` (``absorbed_into_result=True``) OCC
        often returns ``result_map=[[], []]`` even though a merged
        entity exists in ``result``.  In that case inputs whose map
        is empty are remapped to all surviving same-dim entities in
        ``result`` so Part registries retain ownership of the merged
        volume rather than getting silently emptied.

        Safe no-op for empty inputs.  Called by every low-level boolean
        (``_bool_op``) and by the Parts-level fragment/fuse methods so
        remap logic lives in a single place.
        """
        if not input_ents:
            return

        result_by_dim: dict[int, list[int]] = {}
        if absorbed_into_result and result is not None:
            for d, t in result:
                result_by_dim.setdefault(int(d), []).append(int(t))
            for d in result_by_dim:
                result_by_dim[d] = sorted(set(result_by_dim[d]))

        per_dim: dict[int, dict[int, list[int]]] = {}
        for old_dt, new_dts in zip(input_ents, result_map):
            old_dim, old_tag = old_dt
            same_dim_news = [t for d, t in new_dts if d == old_dim]
            if not same_dim_news and absorbed_into_result:
                same_dim_news = list(result_by_dim.get(old_dim, []))
            per_dim.setdefault(old_dim, {})[old_tag] = same_dim_news

        for inst in self._instances.values():
            for d, old_to_new in per_dim.items():
                old_tags = inst.entities.get(d, [])
                if not old_tags:
                    continue
                new_tags: list[int] = []
                for ot in old_tags:
                    new_tags.extend(old_to_new.get(ot, [ot]))
                inst.entities[d] = new_tags

    def fragment_all(self, *, dim: int | None = None) -> list[int]:
        """Fragment all entities so interfaces become conformal.

        Updates each Instance.entities in-place with post-fragment tags.

        Parameters
        ----------
        dim : int, optional
            Target dimension.  Auto-detects highest present if None.

        Returns
        -------
        list[int]
            Tags of all surviving entities at the target dimension.
        """
        if dim is None:
            for d in (3, 2, 1):
                if gmsh.model.getEntities(d):
                    dim = d
                    break
            else:
                raise RuntimeError("No entities found.")

        all_ents = gmsh.model.getEntities(dim)

        # Warn about untracked entities
        tracked = set()
        for inst in self._instances.values():
            for t in inst.entities.get(dim, []):
                tracked.add(t)
        all_tags = set(t for _, t in all_ents)
        orphans = all_tags - tracked
        if orphans:
            warnings.warn(
                f"{len(orphans)} entities at dim={dim} are not tracked "
                f"by any part (tags: {sorted(orphans)}).  They will "
                f"participate in fragmentation but won't be remapped.  "
                f"Use g.parts.register() or g.parts.from_model() to "
                f"adopt them.",
                stacklevel=2,
            )

        if len(all_ents) < 2:
            return [t for _, t in all_ents]

        obj = [all_ents[0]]
        tool = list(all_ents[1:])
        input_ents = obj + tool

        with pg_preserved() as pg:
            result, result_map = gmsh.model.occ.fragment(
                obj, tool, removeObject=True, removeTool=True,
            )
            gmsh.model.occ.synchronize()
            pg.set_result(input_ents, result_map)
            self._remap_from_result(input_ents, result_map)

        return [t for _, t in result]

    def fragment_pair(
        self,
        label_a: str,
        label_b: str,
        *,
        dim: int | None = None,
    ) -> list[int]:
        """Fragment only two instances against each other.

        Returns
        -------
        list[int]
            Surviving entity tags at the target dimension.
        """
        inst_a = self._instances[label_a]
        inst_b = self._instances[label_b]

        if dim is None:
            for d in (3, 2, 1):
                if d in inst_a.entities and d in inst_b.entities:
                    dim = d
                    break
            else:
                raise RuntimeError(
                    f"No common dimension between '{label_a}' and '{label_b}'."
                )

        obj = [(dim, t) for t in inst_a.entities.get(dim, [])]
        tool = [(dim, t) for t in inst_b.entities.get(dim, [])]
        input_ents = obj + tool

        with pg_preserved() as pg:
            result, result_map = gmsh.model.occ.fragment(
                obj, tool, removeObject=True, removeTool=True,
            )
            gmsh.model.occ.synchronize()
            pg.set_result(input_ents, result_map)
            self._remap_from_result(input_ents, result_map)

        return [t for _, t in result]

    def fuse_group(
        self,
        labels: list[str],
        *,
        label: str | None = None,
        dim: int | None = None,
        properties: dict | None = None,
    ) -> "Instance":
        """Fuse multiple instances into a single new instance.

        Calls ``gmsh.model.occ.fuse()`` on the entities of the listed
        instances at the target dimension.  Internal interfaces vanish,
        the surviving entities are stored under a new instance, and the
        old instances are removed from the registry.

        Parameters
        ----------
        labels : list of str
            Existing instance labels to fuse (minimum 2).
        label : str, optional
            Name for the resulting instance.  Defaults to the first
            label in the list (the "survivor").
        dim : int, optional
            Target dimension.  Auto-detects highest common dimension
            across all listed instances if None.
        properties : dict, optional
            Metadata for the new instance.  Inherits from the first
            label if None.

        Returns
        -------
        Instance
            The new fused instance.
        """
        from ._parts_registry import Instance

        # ── Validate input ──────────────────────────────────────────
        if len(labels) < 2:
            raise ValueError(
                f"fuse_group requires at least 2 labels, got {len(labels)}."
            )
        if len(set(labels)) != len(labels):
            raise ValueError(f"fuse_group: duplicate labels in {labels}.")
        for lbl in labels:
            if lbl not in self._instances:
                raise ValueError(f"No part '{lbl}'.")

        new_label = label if label is not None else labels[0]
        if new_label in self._instances and new_label not in labels:
            raise ValueError(
                f"Part label '{new_label}' already exists "
                f"and is not in the fuse list."
            )

        instances = [self._instances[lbl] for lbl in labels]

        # ── Auto-detect common dimension ────────────────────────────
        if dim is None:
            for d in (3, 2, 1):
                if all(d in inst.entities and inst.entities[d]
                       for inst in instances):
                    dim = d
                    break
            else:
                raise RuntimeError(
                    f"No common dimension across instances {labels}."
                )

        # ── Collect entities ────────────────────────────────────────
        obj_inst = instances[0]
        tool_insts = instances[1:]

        obj = [(dim, t) for t in obj_inst.entities.get(dim, [])]
        tool_dt: list[tuple[int, int]] = []
        for tool_inst in tool_insts:
            tool_dt.extend((dim, t) for t in tool_inst.entities.get(dim, []))

        if not obj or not tool_dt:
            raise RuntimeError(
                f"fuse_group: no entities at dim={dim} in one of {labels}."
            )

        # ── OCC fuse ────────────────────────────────────────────────
        input_ents = obj + tool_dt

        with pg_preserved() as pg:
            result, result_map = gmsh.model.occ.fuse(
                obj, tool_dt, removeObject=True, removeTool=True,
            )
            gmsh.model.occ.synchronize()
            pg.set_result(input_ents, result_map, absorbed_into_result=True)
            self._remap_from_result(input_ents, result_map)

        # ── Drop old instances from registry ────────────────────────
        for lbl in labels:
            del self._instances[lbl]

        # ── Build new instance ──────────────────────────────────────
        new_entities: dict[int, list[int]] = {}
        for d, t in result:
            new_entities.setdefault(d, []).append(t)

        new_props = (
            dict(properties) if properties is not None
            else dict(obj_inst.properties)
        )

        inst = Instance(
            label=new_label,
            part_name=new_label,
            entities=new_entities,
            properties=new_props,
            bbox=self._compute_bbox(result),
        )
        self._instances[new_label] = inst
        return inst
