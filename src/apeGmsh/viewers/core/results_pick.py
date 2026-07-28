"""Plain-LMB pick controller for ResultsViewer.

Routes plain (no-modifier) left-mouse events — installed via the shared
:class:`~apeGmsh.viewers.backends._pyvista_pick.PyVistaPickBackend`
(ADR 0047 R-D) — into FEM picks, dispatching per the controller's current
``mode``:

* **Click** (no drag) — fires :func:`on_pick` with a :class:`PickResult`
  resolving to ``"node"`` (world coords for snap-to-nearest in the
  consumer), ``"element"`` (FEM element id via ``scene.cell_to_element_id``),
  or ``"gp"`` (resolved through the scene's :class:`PickInventory`).
* **Drag** — the backend draws a rubber-band; on release fires
  :func:`on_box_pick` with a :class:`BoxPickResult` of the FEM nodes /
  elements / GPs inside the rectangle.

This module owns **no VTK**: the ``vtkCellPicker``, the press/move/release
gesture machine, the rubber-band overlay, and the screen↔world projection
all live in the backend. Here we only interpret the geometric hit
(``PickHit`` / ``BoxGesture``) into a FEM result (ADR 0047 INV-3). Shift+LMB
stays owned by ``install_navigation`` (priority 11); Ctrl falls through to
the trackball.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from numpy import ndarray
    from ..scene.fem_scene import FEMSceneData


# Allowed values for the controller's ``mode``.
MODE_NODE = "node"
MODE_ELEMENT = "element"
MODE_GP = "gp"
_VALID_MODES = (MODE_NODE, MODE_ELEMENT, MODE_GP)


# ``gp_candidates`` returns ``(centers, element_ids, gp_indices)`` — all
# GP centers across the active GaussPointDiagrams, with the matching FEM
# element IDs and per-diagram center indices. Empty arrays (or ``None``)
# signal "no GP markers on screen → nothing to box-pick".
GpCandidates = Callable[[], Optional[tuple]]


@dataclass(frozen=True)
class PickResult:
    """The outcome of a single click pick.

    Attributes
    ----------
    kind
        One of ``"node"``, ``"element"``, or ``"gp"``.
    world
        World-space point hit by the cell picker (or the GP center for
        the ``"gp"`` mode).
    element_id
        FEM element ID — set for ``"element"`` and ``"gp"``.
    cell_id
        Substrate cell index used by the highlight overlay — set for
        ``"element"`` (``None`` when the element was reached via a GP
        marker, in which case the highlight uses the element id).
    gp_index
        GP row index within the diagram's slab — set for ``"gp"``.
    geometry_id
        ADR 0058 S2c (additive, ADR 0047 widening precedent): id of
        the :class:`Geometry` whose substrate actor was hit — set when
        the viewer-injected ``scene_resolver`` recognised the hit
        actor; ``None`` for single-scene installs, overlay (GP-marker)
        hits, and unregistered actors.
    """
    kind: str
    world: tuple
    element_id: Optional[int] = None
    cell_id: Optional[int] = None
    gp_index: Optional[int] = None
    geometry_id: Optional[str] = None


@dataclass(frozen=True)
class BoxPickResult:
    """The outcome of a rubber-band drag-pick.

    Attributes
    ----------
    kind
        One of ``"node"``, ``"element"``, or ``"gp"``.
    ids
        FEM IDs inside the box. For ``"node"`` these are node IDs; for
        ``"element"`` and ``"gp"`` these are element IDs.
    cell_ids
        Substrate-grid cell indices — set for ``"element"``, empty
        otherwise.
    gp_indices
        Per-diagram GP center indices — set for ``"gp"``, empty otherwise.
        Aligned with ``ids`` row-for-row.
    box
        ``(x0, y0, x1, y1)`` in display pixels.
    crossing
        ``True`` when the drag went right→left (``x1 < x0``).
    geometry_id
        ADR 0058 S2c (additive): id of the geometry whose scene the
        box candidates were sourced from — the ACTIVE geometry under
        concurrent rendering (multi-geometry box picking is deferred
        to S3). ``None`` for single-scene installs.
    """
    kind: str
    ids: "ndarray"
    cell_ids: "ndarray"
    gp_indices: "ndarray"
    box: tuple
    crossing: bool
    geometry_id: Optional[str] = None


class ResultsPickController:
    """Public controller returned by :func:`install_results_pick`.

    The host (typically :class:`ResultsViewer`) holds a reference and
    flips :attr:`mode` from keyboard shortcuts (e.g. ``N`` / ``E`` / ``G``).
    The pick adapter reads :attr:`mode` at release time.
    """

    def __init__(self) -> None:
        self.mode: str = MODE_NODE
        # ADR 0045 S4b — the dimensional pick filter (0/1/2/3/4). ``None``
        # = no filter (every dim pickable); a frozenset gates ELEMENT
        # click/box resolution to cells of those dims. Set by the results
        # viewer's FilterController.
        self.active_dims: "Optional[frozenset]" = None
        # The PickBackend driving the gesture machine (set by install).
        self._backend: Any = None

    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"ResultsPickController.mode must be one of "
                f"{_VALID_MODES}; got {mode!r}."
            )
        self.mode = mode

    def uninstall(self) -> None:
        """Remove the backend's observers + overlay. Idempotent."""
        if self._backend is not None:
            self._backend.uninstall()


def _accept_cell_dim(cell_dim, cell_id: int, active_dims) -> bool:
    """Whether a picked cell's element dimension is active (ADR 0045 S4b).

    ``active_dims is None`` (no filter) or absent ``cell_dim`` accepts
    everything — the dim-gate only restricts when a filter is set and the
    scene carries per-cell dims. An out-of-range ``cell_id`` is rejected.
    """
    if active_dims is None:
        return True
    if cell_dim is None or getattr(cell_dim, "size", 0) == 0:
        return True
    if not (0 <= cell_id < cell_dim.size):
        return False
    return int(cell_dim[cell_id]) in active_dims


def _inside_box(
    xy: "ndarray", x0: float, y0: float, x1: float, y1: float,
) -> "ndarray":
    """Boolean mask for points whose display coords fall in the box."""
    bx0, bx1 = (x0, x1) if x0 <= x1 else (x1, x0)
    by0, by1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return (
        (xy[:, 0] >= bx0) & (xy[:, 0] <= bx1)
        & (xy[:, 1] >= by0) & (xy[:, 1] <= by1)
    )


# ======================================================================
# Public install
# ======================================================================


def install_results_pick(
    plotter: pv.Plotter,
    on_pick: Callable[[PickResult], None],
    *,
    scene: "FEMSceneData",
    on_box_pick: Optional[Callable[[BoxPickResult], None]] = None,
    gp_candidates: Optional[GpCandidates] = None,
    drag_threshold_px: int = 4,
    pick_backend: Any = None,
    scene_resolver: Optional[Callable[[Optional[int]], Optional[tuple]]] = None,
    clip_planes: Any = None,
) -> ResultsPickController:
    """Install plain-LMB click + drag picking on *plotter* via a PickBackend.

    Parameters
    ----------
    plotter
        The PyVista plotter (or QtInteractor).
    on_pick
        Invoked with a :class:`PickResult` on a no-drag plain-LMB release
        that hit an actor.
    scene
        :class:`FEMSceneData` whose ``cell_to_element_id`` / ``node_ids``
        resolve VTK indices to FEM IDs, whose ``grid`` projects points for
        box-pick, and whose ``pick_engine`` (a :class:`PickInventory`)
        resolves GP-marker hits. With a ``scene_resolver`` this is the
        FALLBACK scene; without one every pick resolves against it.
    on_box_pick
        Invoked with a :class:`BoxPickResult` on every drag release whose
        rectangle has positive area. ``None`` disables the box-pick path.
    gp_candidates
        Callable returning ``(centers, element_ids, gp_indices)`` for the
        GP box-pick. ``None`` (or empty) → no GP box result.
    drag_threshold_px
        Pixel distance during the press required to be treated as a drag.
    pick_backend
        Injectable :class:`PickBackend` (for headless tests). Defaults to a
        :class:`PyVistaPickBackend` over ``plotter``.
    scene_resolver
        ADR 0058 S2c — ``resolver(prop_id) -> (geometry_id, scene) | None``.
        Maps a hit's ``prop_id`` (``id(actor)``) to the geometry whose
        substrate actor was hit and THAT geometry's scene, so concurrent
        geometries resolve cell→element / coordinates against the hit
        grid (its deformed points), not the install-time scene. Called
        with ``None`` at box-gesture time (no single hit actor) —
        implementations return the ACTIVE geometry's scene there. A
        ``None`` return (or ``None`` resolver) falls back to ``scene``.
    clip_planes
        ADR 0083 Part 5 — the viewport's ``ClipPlaneSetController``.
        A click whose world point lies on the discarded side of any
        active section plane is reported as a **miss**: the cell picker
        is a geometric ray-cast and cannot see mapper clip planes, so
        without this a click on apparently-empty space picks the cell
        the cut is hiding. ``None`` disables the filter. Picking
        *through* the hole onto visible geometry behind it is
        deliberately not attempted in slice 1.

    Returns
    -------
    ResultsPickController
        Live controller — flip ``ctrl.mode`` to switch dispatch between
        node, element, and GP picks; call ``ctrl.uninstall()`` to tear down.
    """
    controller = ResultsPickController()
    inventory = getattr(scene, "pick_engine", None)
    _empty_dim = np.array([], dtype=np.int8)

    def _scene_for_hit(prop_id: Optional[int]) -> tuple:
        """``(geometry_id, scene)`` a pick resolves against (ADR 0058
        S2c). The viewer-injected resolver wins when it recognises the
        prop; otherwise the install-time scene (single-geometry
        behaviour, ``geometry_id=None``)."""
        if scene_resolver is not None:
            try:
                resolved = scene_resolver(prop_id)
                if resolved is not None and resolved[1] is not None:
                    return (resolved[0], resolved[1])
            except Exception:
                pass
        return (None, scene)

    # ------------------------------------------------------------------
    # Click resolution (geometric hit -> FEM result, routed by mode)
    # ------------------------------------------------------------------

    def _build_result(hit) -> Optional[PickResult]:
        cell_id = hit.cell_id
        if cell_id is None or cell_id < 0:
            return None
        world = hit.world
        prop_id = hit.prop_id
        mode = controller.mode
        # ADR 0083 Part 5 — a hit the cut has hidden is a miss, in
        # every mode. Checked before any FEM resolution: the point is
        # geometrically behind a plane regardless of what it resolves to.
        if clip_planes is not None:
            try:
                if clip_planes.is_clipped(world):
                    return None
            except Exception:
                pass
        # Hit-scene resolution (ADR 0058 S2c): index arrays + grid come
        # from the geometry whose actor was hit, read at resolve time.
        geom_id, hit_scene = _scene_for_hit(prop_id)

        if mode == MODE_NODE:
            return PickResult(
                kind=MODE_NODE, world=world, geometry_id=geom_id,
            )

        if mode == MODE_ELEMENT:
            # GP markers are always pickable (ADR 0047 R-D.2b), so an
            # element-mode click may land on a GP glyph. A registered
            # overlay hit resolves to its owning element; a substrate hit
            # (prop_id not in the inventory) takes the cell→element path.
            gp = (
                inventory.resolve(prop_id, cell_id)
                if inventory is not None else None
            )
            if gp is not None:
                element_id = int(gp[0])
                # GP glyphs are overlay actors, not substrate — the hit
                # scene fell back, so ``geometry_id`` stays None (the
                # inventory carries no geometry).
                hl_cell = (
                    getattr(hit_scene, "element_id_to_cell", {}) or {}
                ).get(element_id)
                return PickResult(
                    kind=MODE_ELEMENT,
                    world=world,
                    element_id=element_id,
                    cell_id=int(hl_cell) if hl_cell is not None else None,
                )
            # Per-cell element dims for the dim-pick gate (ADR 0045
            # S4b). Empty when the scene carries no cell_dim (older
            # builds) → gate is inert.
            cell_dim = np.asarray(
                getattr(hit_scene, "cell_dim", _empty_dim),
            )
            if not _accept_cell_dim(cell_dim, cell_id, controller.active_dims):
                return None
            cell_to_element_id = hit_scene.cell_to_element_id
            if not (0 <= cell_id < cell_to_element_id.size):
                return None
            return PickResult(
                kind=MODE_ELEMENT,
                world=world,
                element_id=int(cell_to_element_id[cell_id]),
                cell_id=int(cell_id),
                geometry_id=geom_id,
            )

        if mode == MODE_GP:
            if inventory is None:
                return None
            gp = inventory.resolve(prop_id, cell_id)
            if gp is None:
                return None
            element_id, gp_index, gp_world = gp
            try:
                gp_world_t = tuple(float(c) for c in gp_world)
            except Exception:
                gp_world_t = tuple(world)
            return PickResult(
                kind=MODE_GP,
                world=gp_world_t,
                element_id=int(element_id),
                gp_index=int(gp_index),
            )
        return None

    # ------------------------------------------------------------------
    # Box-pick resolution (domain candidates over backend projection)
    # ------------------------------------------------------------------

    def _build_box_result(box) -> Optional[BoxPickResult]:
        x0, y0, x1, y1 = box
        if x0 == x1 or y0 == y1:
            return None    # Degenerate rectangle — nothing to pick.
        crossing = x1 < x0
        mode = controller.mode
        # A box has no single hit actor — candidates come from the
        # ACTIVE geometry's scene, resolved at gesture time (ADR 0058
        # S2c; multi-geometry box picking is deferred to S3).
        geom_id, box_scene = _scene_for_hit(None)
        grid = box_scene.grid

        if mode == MODE_NODE:
            try:
                pts = np.asarray(grid.points, dtype=np.float64)
            except Exception:
                return None
            display = pick_backend.project_points(pts)
            mask = _inside_box(display, x0, y0, x1, y1)
            node_ids_arr = np.asarray(box_scene.node_ids, dtype=np.int64)
            return BoxPickResult(
                kind=MODE_NODE,
                ids=node_ids_arr[mask],
                cell_ids=np.zeros(0, dtype=np.int64),
                gp_indices=np.zeros(0, dtype=np.int64),
                box=(x0, y0, x1, y1),
                crossing=crossing,
                geometry_id=geom_id,
            )

        if mode == MODE_ELEMENT:
            try:
                centroids = np.asarray(
                    grid.cell_centers().points, dtype=np.float64,
                )
            except Exception:
                return None
            display = pick_backend.project_points(centroids)
            mask = _inside_box(display, x0, y0, x1, y1)
            # Exclude cells hidden via ElementVisibility — ``vtkGhostType``
            # bit 0x01 (HIDDENCELL). cell_centers() drops the ghost array,
            # so read it back from the source grid by cell index.
            try:
                ghosts = np.asarray(grid.cell_data["vtkGhostType"])
                if ghosts.size == mask.size:
                    mask = mask & ~(ghosts & 0x01).astype(bool)
            except (KeyError, IndexError):
                pass
            # Dim-pick gate: keep only cells whose dim is active (S4b).
            cell_dim = np.asarray(
                getattr(box_scene, "cell_dim", _empty_dim),
            )
            if (
                controller.active_dims is not None
                and cell_dim.size == mask.size
            ):
                mask = mask & np.isin(cell_dim, list(controller.active_dims))
            cell_idx = np.nonzero(mask)[0].astype(np.int64)
            element_ids = (
                box_scene.cell_to_element_id[cell_idx]
                if cell_idx.size else np.zeros(0, dtype=np.int64)
            )
            return BoxPickResult(
                kind=MODE_ELEMENT,
                ids=np.asarray(element_ids, dtype=np.int64),
                cell_ids=cell_idx,
                gp_indices=np.zeros(0, dtype=np.int64),
                box=(x0, y0, x1, y1),
                crossing=crossing,
                geometry_id=geom_id,
            )

        if mode == MODE_GP:
            if gp_candidates is None:
                return None
            try:
                cand = gp_candidates()
            except Exception:
                cand = None
            if cand is None:
                return None
            try:
                centers, gp_eids, gp_idxs = cand
                centers = np.asarray(centers, dtype=np.float64)
                gp_eids = np.asarray(gp_eids, dtype=np.int64)
                gp_idxs = np.asarray(gp_idxs, dtype=np.int64)
            except Exception:
                return None
            if (
                centers.ndim != 2 or centers.shape[1] != 3
                or centers.shape[0] != gp_eids.size
                or centers.shape[0] != gp_idxs.size
            ):
                return None
            if centers.shape[0] == 0:
                return BoxPickResult(
                    kind=MODE_GP,
                    ids=np.zeros(0, dtype=np.int64),
                    cell_ids=np.zeros(0, dtype=np.int64),
                    gp_indices=np.zeros(0, dtype=np.int64),
                    box=(x0, y0, x1, y1),
                    crossing=crossing,
                )
            display = pick_backend.project_points(centers)
            mask = _inside_box(display, x0, y0, x1, y1)
            return BoxPickResult(
                kind=MODE_GP,
                ids=gp_eids[mask],
                cell_ids=np.zeros(0, dtype=np.int64),
                gp_indices=gp_idxs[mask],
                box=(x0, y0, x1, y1),
                crossing=crossing,
            )
        return None

    # ------------------------------------------------------------------
    # Geometric-callback adapters (backend -> FEM resolution)
    # ------------------------------------------------------------------

    def _on_geom_pick(hit, _mods) -> None:
        if hit is None:
            return
        result = _build_result(hit)
        if result is None:
            return
        try:
            on_pick(result)
        except Exception as exc:
            import sys
            print(f"[results_pick] on_pick raised: {exc}", file=sys.stderr)

    def _on_geom_box(gesture) -> None:
        if on_box_pick is None:
            return
        box_result = _build_box_result(gesture.box)
        if box_result is None:
            return
        try:
            on_box_pick(box_result)
        except Exception as exc:
            import sys
            print(f"[results_pick] on_box_pick raised: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Install the gesture machine via the shared PickBackend.
    # ------------------------------------------------------------------

    if pick_backend is None:
        from ..backends._pyvista_pick import PyVistaPickBackend

        assert plotter.iren is not None, (
            "plotter.iren is None; call plotter.show() before installing "
            "the results pick observer."
        )
        pick_backend = PyVistaPickBackend(plotter, drag_threshold=drag_threshold_px)

    controller._backend = pick_backend
    pick_backend.install(on_pick=_on_geom_pick, on_box=_on_geom_box)
    return controller
