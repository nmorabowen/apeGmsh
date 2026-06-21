"""
ColorModeController — Drives mesh-viewer color modes.

Two coloring strategies coexist:

* **Idle-fn strategy** (Default / Element Type / Physical Group)
  Swaps ``ColorManager``'s idle-color function. Hover/pick/hidden state
  layers correctly on top, so mode switches preserve interaction.

* **Scalar-mapper strategy** (Quality)
  Mutates each per-dim actor's VTK mapper to display a per-cell scalar
  with a lookup table + colorbar. Hover/pick writes still go to
  ``cell_data["colors"]`` but are visually masked while Quality is
  active; switching back to an idle-fn mode triggers a full repaint
  so no stale state leaks.

Supported modes
---------------
- ``Default``        — per-dim theme colors (ColorManager's built-in default)
- ``Element Type``   — color per entity based on dominant element type
- ``Physical Group`` — color per entity based on physical group membership
- ``Quality``        — per-cell scalar from gmsh.model.mesh.getElementQualities
"""
from __future__ import annotations

import zlib
from collections import Counter
from typing import TYPE_CHECKING, Any

import gmsh
import numpy as np

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..data import ViewerData
    from ..scene.mesh_scene import MeshSceneData
    from .color_manager import ColorManager
    from .entity_registry import EntityRegistry
    from .selection import SelectionState
    from .visibility import VisibilityManager


_FALLBACK_RGB = np.array([136, 136, 136], dtype=np.uint8)


# Element-type category -> RGB. Matches ELEM_TYPE_COLORS in scene/mesh_scene.py.
_ELEM_TYPE_RGB: dict[str, np.ndarray] = {
    "Triangle":      np.array([67, 99, 216], dtype=np.uint8),
    "Quadrilateral": np.array([60, 180, 75], dtype=np.uint8),
    "Quad":          np.array([60, 180, 75], dtype=np.uint8),
    "Tetrahedron":   np.array([230, 25, 75], dtype=np.uint8),
    "Hexahedron":    np.array([245, 130, 49], dtype=np.uint8),
    "Prism":         np.array([145, 30, 180], dtype=np.uint8),
    "Pyramid":       np.array([66, 212, 244], dtype=np.uint8),
    "Line":          np.array([170, 170, 170], dtype=np.uint8),
    "Point":         np.array([255, 255, 255], dtype=np.uint8),
}


_GROUP_PALETTE_HEX = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9",
)


def _hex_to_rgb(s: str) -> np.ndarray:
    s = s.lstrip("#")
    return np.array(
        [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)],
        dtype=np.uint8,
    )


_GROUP_PALETTE_RGB: list[np.ndarray] = [_hex_to_rgb(h) for h in _GROUP_PALETTE_HEX]


_DEFAULT_QUALITY_METRIC = "minSICN"
_QUALITY_CMAP = "viridis"


def _split_joined_module_label(label: str) -> tuple[str, ...]:
    """Inverse of the compose-side ``_join_module_label`` rule —
    inlined locally so the viewer layer doesn't reach into
    ``apeGmsh.mesh`` (forbidden by the layering invariant test
    ``test_viewers_pure_h5_consumer.py``).

    Joined-label format: components separated by `.` at odd depths
    and `/` at even depths, with the LEFTMOST separator at depth N
    (the outermost join) and the RIGHTMOST at depth 2. A depth-1
    label has no separator.

    Returns the component tuple in outer-to-inner order. Raises
    ``ValueError`` if a separator at any position violates the
    alternation rule — fail-loud, since a malformed label in the
    broker indicates upstream corruption (``ComposeLabelError``
    should have prevented it at write time).

    Canonical implementation lives at
    ``apeGmsh.mesh._compose._split_joined_label``. Keep this copy in
    lock-step; the round-trip property is tested on both sides.
    """
    if not label:
        return ()
    # Find separator positions
    seps = [i for i, ch in enumerate(label) if ch in ("/", ".")]
    if not seps:
        return (label,)
    depth = len(seps) + 1
    # Validate alternation: leftmost sep is depth N, next is N-1,
    # ..., rightmost is depth 2. The depth-k separator must be `.`
    # for odd k and `/` for even k.
    for i, pos in enumerate(seps):
        k = depth - i  # depth of this separator (decreasing left to right)
        expected = "." if (k % 2 == 1) else "/"
        if label[pos] != expected:
            raise ValueError(
                f"malformed joined module label {label!r}: "
                f"separator at position {pos} is {label[pos]!r}, "
                f"alternation rule requires {expected!r} at depth {k}"
            )
    # Split at separator positions; reject any empty component
    parts = []
    start = 0
    for pos in seps:
        comp = label[start:pos]
        if not comp:
            raise ValueError(
                f"malformed joined module label {label!r}: empty component"
            )
        parts.append(comp)
        start = pos + 1
    tail = label[start:]
    if not tail:
        raise ValueError(
            f"malformed joined module label {label!r}: empty component"
        )
    parts.append(tail)
    return tuple(parts)


class ColorModeController:
    """Switch the mesh viewer between named color modes."""

    __slots__ = (
        "_color_mgr", "_registry", "_scene", "_sel", "_vis_mgr",
        "_plotter", "_view", "_mode",
        "_quality_metric", "_quality_mapper_state", "_quality_bar_title",
    )

    def __init__(
        self,
        *,
        color_mgr: "ColorManager",
        registry: "EntityRegistry",
        scene: "MeshSceneData",
        sel: "SelectionState",
        vis_mgr: "VisibilityManager",
        plotter,
        view: "ViewerData | None" = None,
    ) -> None:
        self._color_mgr = color_mgr
        self._registry = registry
        self._scene = scene
        self._sel = sel
        self._vis_mgr = vis_mgr
        self._plotter = plotter
        # ViewerData snapshot — supplies per-element bridge enrichment
        # the gmsh-only scene lacks (partition rank per FEM eid). None
        # for from_fem-only viewers without an h5 source; the Partition
        # mode degrades to a uniform fallback color in that case.
        self._view: "ViewerData | None" = view
        self._mode = "Default"
        self._quality_metric = _DEFAULT_QUALITY_METRIC
        # Per-dim VTK mapper state captured on entry to Quality mode,
        # restored on exit. Keys: dim -> {color_mode, scalar_mode, ...}.
        self._quality_mapper_state: dict[int, dict[str, Any]] = {}
        self._quality_bar_title: str | None = None
        # Re-enter Quality mode after a visibility rebuild swaps the actors —
        # the new actor starts in rgb mode and needs the quality mapper
        # reinstalled. Uses the existing on_changed hook (no new dispatch).
        on_changed = getattr(vis_mgr, "on_changed", None)
        if on_changed is not None:
            on_changed.append(self._on_vis_rebuild)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        # If leaving Quality, revert mapper state first so the next
        # _repaint sees rgb-mode mappers.
        if self._mode == "Quality" and mode != "Quality":
            self._exit_quality_mode()

        if mode == "Default":
            self._color_mgr.reset_idle_fn()
        elif mode == "Element Type":
            self._color_mgr.set_idle_fn(self._elem_type_idle)
        elif mode == "Physical Group":
            self._color_mgr.set_idle_fn(self._phys_group_idle)
        elif mode == "Partition":
            self._color_mgr.set_idle_fn(self._partition_idle)
        elif mode == "Module":
            self._color_mgr.set_idle_fn(self._module_idle)
        elif mode == "Module: Root":
            self._color_mgr.set_idle_fn(self._module_idle_by_root)
        elif mode == "Module: Leaf":
            self._color_mgr.set_idle_fn(self._module_idle_by_leaf)
        elif mode == "Quality":
            self._enter_quality_mode()
            self._mode = mode
            return  # _enter_quality_mode renders; skip _repaint
        else:
            return

        self._mode = mode
        self._repaint()

    def refresh(self) -> None:
        """Re-apply the current mode (e.g. after theme change)."""
        if self._mode == "Quality":
            return  # quality colors are theme-independent
        self._repaint()

    # ------------------------------------------------------------------
    # Idle-fn strategy (Default / Element Type / Physical Group)
    # ------------------------------------------------------------------

    def _elem_type_idle(self, dt: "DimTag") -> np.ndarray:
        cat = self._scene.brep_dominant_type.get(dt, "Line")
        return _ELEM_TYPE_RGB.get(cat, _FALLBACK_RGB)

    def _phys_group_idle(self, dt: "DimTag") -> np.ndarray:
        name = self._scene.brep_to_group.get(dt)
        if name is None:
            return _FALLBACK_RGB
        # zlib.crc32 instead of Python's hash() — hash() is randomized
        # across processes via PYTHONHASHSEED so the same model would
        # get different group colors in different sessions (and unlucky
        # CI seeds would land collisions on the 19-color palette).
        # Mirrors the determinism fix applied to _module_idle in #374.
        idx = zlib.crc32(name.encode("utf-8")) % len(_GROUP_PALETTE_RGB)
        return _GROUP_PALETTE_RGB[idx]

    def _partition_idle(self, dt: "DimTag") -> np.ndarray:
        """Color one BRep entity by its dominant OpenSeesMP rank.

        Schema 2.10.0 (ADR 0027). Per-entity granularity (not per-cell)
        — typical METIS-style partitioners produce contiguous rank
        assignments that align with CAD entity boundaries; the
        ``most common rank`` reduction is a faithful summary for the
        common case and a graceful approximation for split-entity
        cases (a single-rank uniform color rather than a mixed-rank
        per-cell painting).

        Degrades to ``_FALLBACK_RGB`` when:

        * No ViewerData is bound (``from_fem``-only viewers with no
          OpenSees enrichment).
        * The view carries no partition labelling
          (``has_partitions == False`` — single-partition models or
          pre-2.10.0 archives).
        * The DimTag has no elements in the scene, or every owning
          element has ``partition_for(eid) is None``.

        The full per-cell fidelity (a scalar-mapper strategy mirroring
        Quality mode) is a follow-up; PR1 ships the per-entity dispatch
        because (a) it composes cleanly with the existing idle-fn
        infrastructure and (b) it covers the common case
        (METIS-contiguous partitions) at zero extra scaffolding cost.
        """
        view = self._view
        if view is None or not view.elements.has_partitions:
            return _FALLBACK_RGB
        eids = self._scene.brep_to_elems.get(dt)
        if not eids:
            return _FALLBACK_RGB
        # Collect non-None ranks; np.bincount picks the most common.
        ranks: list[int] = []
        for eid in eids:
            r = view.elements.partition_for(int(eid))
            if r is not None:
                ranks.append(int(r))
        if not ranks:
            return _FALLBACK_RGB
        dominant = int(np.bincount(np.asarray(ranks, dtype=np.int64)).argmax())
        return _GROUP_PALETTE_RGB[dominant % len(_GROUP_PALETTE_RGB)]

    def _module_idle(self, dt: "DimTag") -> np.ndarray:
        """Color one BRep entity by its dominant compose-module label.

        Schema 2.9.0 / ADR 0038. Mirror of :meth:`_partition_idle` for
        the compose-provenance dimension: each FEM element carries a
        joined module label (e.g. ``"bayP/frameA"`` for a nested
        compose) and the BRep entity is colored by the most-common
        label across its owning elements. The full joined label is
        used as-is — distinct nesting paths get distinct colors.

        Reuses ``_GROUP_PALETTE_RGB`` (same palette as ``Physical
        Group`` and ``Partition``); the active mode label disambiguates
        contextually and there's no maintenance cost to a second
        palette.

        Degrades to ``_FALLBACK_RGB`` when:

        * No ViewerData is bound (``from_fem``-only viewers with no
          OpenSees enrichment).
        * The view carries no module labelling (``has_modules ==
          False`` — uncomposed FEMData, pre-2.9.0 archives).
        * The DimTag has no elements in the scene, or every owning
          element has ``module_for(eid) is None`` (host-owned, or
          unlabelled in the source).
        """
        view = self._view
        if view is None or not view.elements.has_modules:
            return _FALLBACK_RGB
        eids = self._scene.brep_to_elems.get(dt)
        if not eids:
            return _FALLBACK_RGB
        labels: list[str] = []
        for eid in eids:
            label = view.elements.module_for(int(eid))
            if label is not None:
                labels.append(label)
        if not labels:
            return _FALLBACK_RGB
        dominant = Counter(labels).most_common(1)[0][0]
        # zlib.crc32 instead of Python's hash() — hash() is randomized
        # across processes (PYTHONHASHSEED), so the same module would
        # get different colors in different sessions. crc32 is stable.
        # `_phys_group_idle` above uses the same crc32 pattern.
        idx = zlib.crc32(dominant.encode("utf-8")) % len(_GROUP_PALETTE_RGB)
        return _GROUP_PALETTE_RGB[idx]

    def _module_idle_by_root(self, dt: "DimTag") -> np.ndarray:
        """Color by **root** module of a nested compose label.

        Phase 3F.2d. Projects each owning element's joined label
        (e.g. ``"bayP/frameA"``) onto its root component (``"bayP"``)
        before the dominant-label / palette lookup. All modules sharing
        the same depth-1 ancestor color identically — useful for "show
        me the top-level subsystems" inspection on deeply-nested
        composes.

        Single-level (depth-1) labels project to themselves, so this
        mode behaves identically to ``"Module"`` on flat composes.
        Uncomposed sources degrade to ``_FALLBACK_RGB`` (same as
        :meth:`_module_idle`).
        """
        return self._module_idle_projected(
            dt, lambda label: _split_joined_module_label(label)[0]
        )

    def _module_idle_by_leaf(self, dt: "DimTag") -> np.ndarray:
        """Color by **leaf** module of a nested compose label.

        Phase 3F.2d. Projects each owning element's joined label
        (e.g. ``"bayP/frameA"``) onto its leaf component (``"frameA"``)
        before the dominant-label / palette lookup. All modules
        sharing the same leaf name across sub-trees color identically
        — useful for "where do all the ``frameA`` instances live?"
        cross-cuts.

        Single-level (depth-1) labels project to themselves. Uncomposed
        sources degrade to ``_FALLBACK_RGB``.
        """
        return self._module_idle_projected(
            dt, lambda label: _split_joined_module_label(label)[-1]
        )

    def _module_idle_projected(self, dt: "DimTag", projector) -> np.ndarray:
        """Shared body for ``_module_idle_by_root`` / ``_by_leaf``.

        Mirrors :meth:`_module_idle`'s degraded-fallback contract:
        same short-circuits on missing ViewerData / ``has_modules`` /
        no owning elements / all-host-owned. The only difference is
        that each label is run through ``projector(label) -> str``
        before being collected into the Counter for dominant-label
        resolution.

        ``projector`` is called only on non-``None`` labels (host-
        owned rows are filtered before projection); a malformed label
        (one that doesn't match the alternation rule) propagates
        ``ComposeError`` upward — fail-loud, not silent.
        """
        view = self._view
        if view is None or not view.elements.has_modules:
            return _FALLBACK_RGB
        eids = self._scene.brep_to_elems.get(dt)
        if not eids:
            return _FALLBACK_RGB
        labels: list[str] = []
        for eid in eids:
            label = view.elements.module_for(int(eid))
            if label is not None:
                labels.append(projector(label))
        if not labels:
            return _FALLBACK_RGB
        dominant = Counter(labels).most_common(1)[0][0]
        idx = zlib.crc32(dominant.encode("utf-8")) % len(_GROUP_PALETTE_RGB)
        return _GROUP_PALETTE_RGB[idx]

    def close(self) -> None:
        """Remove the on_changed subscription to avoid a reference cycle."""
        on_changed = getattr(self._vis_mgr, "on_changed", None)
        if on_changed is not None:
            try:
                on_changed.remove(self._on_vis_rebuild)
            except ValueError:
                pass

    def _on_vis_rebuild(self) -> None:
        """Re-install the Quality scalar mapper after a hide/reveal rebuild.

        A rebuild replaces dim_actors with new actors (rgb mode by default).
        For Quality mode we reinstall the scalar mapper on the new actor so
        quality colors are not lost after the user hides an entity.
        """
        if self._mode == "Quality":
            self._enter_quality_mode()

    def _repaint(self) -> None:
        if self._mode == "Partition":
            self._repaint_partition()
            return
        # Single batched recolor — one VTK rebind per dim, not per entity.
        # On large meshes this is ~100x faster than the per-entity loop;
        # per-entity idle (Element-Type / Physical-Group modes) is still
        # honored because recolor_all evaluates _idle_fn(dt) per entity.
        self._color_mgr.recolor_all(
            picks=set(self._sel.picks),
            hidden=self._vis_mgr.hidden,
        )
        self._plotter.render()

    def _repaint_partition(self) -> None:
        """Color partitioned meshes per rendered cell/element."""
        view = self._view
        if view is None or not view.elements.has_partitions:
            self._color_mgr.recolor_all(
                picks=set(self._sel.picks),
                hidden=self._vis_mgr.hidden,
            )
            self._plotter.render()
            return

        hidden = set(self._vis_mgr.hidden)
        picks = set(self._sel.picks)
        reg = self._registry
        for dim in reg.dims:
            mesh = reg.dim_meshes.get(dim)
            if mesh is None:
                continue
            colors = mesh.cell_data.get("colors")
            if colors is None:
                continue
            elem_tags = self._scene.batch_cell_to_elem.get(dim)
            if elem_tags is None or len(elem_tags) != mesh.n_cells:
                for dt in reg.all_entities(dim=dim):
                    cells = reg.cells_for_entity(dt)
                    if cells:
                        colors[cells] = self._partition_idle(dt)
            else:
                colors[:] = _FALLBACK_RGB
                for ci, eid in enumerate(elem_tags):
                    rank = view.elements.partition_for(int(eid))
                    if rank is not None:
                        colors[ci] = _GROUP_PALETTE_RGB[
                            int(rank) % len(_GROUP_PALETTE_RGB)
                        ]

            for dt in hidden:
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if cells:
                    colors[cells] = self._color_mgr._hidden_rgb
            for dt in picks:
                if dt[0] != dim:
                    continue
                cells = reg.cells_for_entity(dt)
                if cells:
                    colors[cells] = self._color_mgr._pick_rgb
            mesh.cell_data["colors"] = colors
        self._plotter.render()

    # ------------------------------------------------------------------
    # Quality strategy (scalar mapper)
    # ------------------------------------------------------------------

    def _enter_quality_mode(self) -> None:
        import pyvista as pv

        metric = self._quality_metric

        # Compute per-dim quality and the global value range. Lines
        # (dim=1) are skipped — gmsh's quality metrics return 0 for
        # straight 2-node line elements, which would flag every edge as
        # "worst" and compress the meaningful colorbar range.
        per_dim: dict[int, tuple[np.ndarray, Any]] = {}
        for dim in self._registry.dims:
            if dim < 2:
                continue
            grid = self._registry.dim_meshes.get(dim)
            actor = self._registry.dim_actors.get(dim)
            if grid is None or actor is None:
                continue
            qarr = self._get_quality(metric, dim)
            if qarr is None or len(qarr) != grid.n_cells:
                continue
            per_dim[dim] = (qarr, grid)

        if not per_dim:
            return

        all_q = np.concatenate([q for q, _ in per_dim.values()])
        qmin = float(all_q.min())
        qmax = float(all_q.max())
        if qmax - qmin < 1e-12:
            qmax = qmin + 1e-6

        lut = pv.LookupTable(cmap=_QUALITY_CMAP)
        lut.scalar_range = (qmin, qmax)

        scalar_name = f"_quality_{metric}"

        first_mapper = None
        for dim, (qarr, grid) in per_dim.items():
            actor = self._registry.dim_actors[dim]
            grid.cell_data[scalar_name] = qarr
            mapper = actor.GetMapper()
            self._quality_mapper_state.setdefault(dim, {
                "color_mode":         mapper.GetColorMode(),
                "scalar_mode":        mapper.GetScalarMode(),
                "array_name":         mapper.GetArrayName() or "colors",
                "scalar_visibility":  bool(mapper.GetScalarVisibility()),
                "lut":                mapper.GetLookupTable(),
            })
            mapper.SelectColorArray(scalar_name)
            mapper.SetScalarModeToUseCellData()
            mapper.SetColorModeToMapScalars()
            mapper.SetLookupTable(lut)
            mapper.SetUseLookupTableScalarRange(True)
            mapper.ScalarVisibilityOn()
            if first_mapper is None:
                first_mapper = mapper

        title = f"Quality ({metric})"
        self._quality_bar_title = title
        try:
            self._plotter.add_scalar_bar(title=title, mapper=first_mapper)
        except Exception:
            pass

        self._plotter.render()

    def _exit_quality_mode(self) -> None:
        for dim, state in self._quality_mapper_state.items():
            actor = self._registry.dim_actors.get(dim)
            if actor is None:
                continue
            mapper = actor.GetMapper()
            mapper.SelectColorArray(state["array_name"])
            mapper.SetScalarMode(state["scalar_mode"])
            mapper.SetColorMode(state["color_mode"])
            if state["lut"] is not None:
                mapper.SetLookupTable(state["lut"])
            mapper.SetScalarVisibility(int(state["scalar_visibility"]))
        self._quality_mapper_state.clear()

        if self._quality_bar_title is not None:
            try:
                self._plotter.remove_scalar_bar(self._quality_bar_title)
            except Exception:
                pass
            self._quality_bar_title = None

    def _get_quality(self, metric: str, dim: int) -> np.ndarray | None:
        """Return per-cell quality for *dim*, computing & caching on first call."""
        cache = self._scene.quality.setdefault(metric, {})
        if dim in cache:
            return cache[dim]

        cell_map = self._scene.batch_cell_to_elem.get(dim)
        if cell_map is None or len(cell_map) == 0:
            return None

        elem_tags = cell_map.tolist()
        try:
            qlist = gmsh.model.mesh.getElementQualities(
                elem_tags, qualityName=metric,
            )
            qarr = np.asarray(qlist, dtype=np.float64)
        except Exception:
            return None

        cache[dim] = qarr
        return qarr
