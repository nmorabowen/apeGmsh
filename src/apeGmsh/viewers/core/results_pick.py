"""Plain-LMB pick controller for ResultsViewer (Phases 2a / 2b / 2c).

Installs VTK observers that intercept plain (no-modifier) left-mouse
events on a PyVista plotter and dispatches per the controller's
current ``mode``:

* **Click** (no drag) — fires :func:`on_pick` with a
  :class:`PickResult` resolving either to ``"node"`` (world coords
  for snap-to-nearest in the consumer) or ``"element"`` (FEM element
  id resolved via ``scene.cell_to_element_id``).
* **Drag** — draws a yellow rectangle while LMB is held; on release
  fires :func:`on_box_pick` with a :class:`BoxPickResult` whose
  ``ids`` are the FEM nodes (or elements) inside the rectangle.

Drag is recognised and absorbed (priority-10 abort) so the trackball
camera does **not** rotate on plain LMB. Shift+LMB stays owned by
:func:`install_navigation` at priority 11. Ctrl+LMB falls through to
the trackball's spin gesture untouched.
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


# ``gp_resolver`` returns ``(element_id, gp_index, world_xyz)`` if the
# picked actor + cell index belong to a GaussPointDiagram, else None.
GpResolver = Callable[[Any, int], Optional[tuple]]

# ``gp_candidates`` returns ``(centers, element_ids, gp_indices)`` —
# all GP centers across the active GaussPointDiagrams, with the
# matching FEM element IDs and per-diagram center indices. Empty
# arrays (or ``None``) signal "no GP markers on screen → nothing to
# box-pick" and the box-pick path silently returns no IDs.
GpCandidates = Callable[
    [], Optional[tuple],
]


@dataclass(frozen=True)
class PickResult:
    """The outcome of a single click pick.

    Attributes
    ----------
    kind
        One of ``"node"``, ``"element"``, or ``"gp"``.
    world
        World-space point hit by the cell picker (or the GP center
        for the ``"gp"`` mode).
    element_id
        FEM element ID — set for ``"element"`` and ``"gp"``.
    cell_id
        VTK cell index — set for ``"element"`` (substrate cell index
        used by the highlight overlay).
    gp_index
        GP row index within the diagram's slab — set for ``"gp"``.
    """
    kind: str
    world: tuple
    element_id: Optional[int] = None
    cell_id: Optional[int] = None
    gp_index: Optional[int] = None


@dataclass(frozen=True)
class BoxPickResult:
    """The outcome of a rubber-band drag-pick.

    Attributes
    ----------
    kind
        One of ``"node"``, ``"element"``, or ``"gp"``.
    ids
        FEM IDs inside the box. For ``"node"`` these are node IDs;
        for ``"element"`` and ``"gp"`` these are element IDs.
    cell_ids
        Substrate-grid cell indices — set for ``"element"`` (used by
        the highlight overlay), empty otherwise.
    gp_indices
        Per-diagram GP center indices — set for ``"gp"``, empty
        otherwise. Aligned with ``ids`` row-for-row.
    box
        ``(x0, y0, x1, y1)`` in display pixels.
    crossing
        ``True`` when the drag went right→left (``x1 < x0``).
    """
    kind: str
    ids: "ndarray"
    cell_ids: "ndarray"
    gp_indices: "ndarray"
    box: tuple
    crossing: bool


class ResultsPickController:
    """Public controller returned by :func:`install_results_pick`.

    The host (typically :class:`ResultsViewer`) holds a reference and
    flips :attr:`mode` from keyboard shortcuts (e.g. ``N`` / ``E``).
    The pick observer reads :attr:`mode` at release time.
    """

    def __init__(self) -> None:
        self.mode: str = MODE_NODE

    def set_mode(self, mode: str) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"ResultsPickController.mode must be one of "
                f"{_VALID_MODES}; got {mode!r}."
            )
        self.mode = mode


# ======================================================================
# Display-space projection + box-pick helpers
# ======================================================================


def _project_points_to_display(
    points: "ndarray", renderer: Any,
) -> "ndarray":
    """Project ``(N, 3)`` world points to ``(N, 2)`` display pixels.

    Per-point ``WorldToDisplay`` is a few thousand calls per drag-
    release — fine on typical viewer-scale meshes (≤ ~50k nodes).
    Hot-path callers can swap this for a cached camera-projection-
    matrix multiply if profiling shows it matters.
    """
    out = np.empty((points.shape[0], 2), dtype=np.float64)
    for i in range(points.shape[0]):
        renderer.SetWorldPoint(
            float(points[i, 0]),
            float(points[i, 1]),
            float(points[i, 2]),
            1.0,
        )
        renderer.WorldToDisplay()
        d = renderer.GetDisplayPoint()
        out[i, 0] = d[0]
        out[i, 1] = d[1]
    return out


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
    gp_resolver: Optional[GpResolver] = None,
    gp_candidates: Optional[GpCandidates] = None,
    drag_threshold_px: int = 4,
) -> ResultsPickController:
    """Install plain-LMB click + drag-pick observers on *plotter*.

    Parameters
    ----------
    plotter
        The PyVista plotter (or QtInteractor).
    on_pick
        Invoked with a :class:`PickResult` on a no-drag plain-LMB
        release that hit an actor.
    scene
        :class:`FEMSceneData` whose ``cell_to_element_id`` and
        ``node_ids`` resolve VTK indices to FEM IDs. The substrate
        ``grid`` is also used to project point coords for box-pick.
    on_box_pick
        Invoked with a :class:`BoxPickResult` on every drag release
        whose rectangle has positive area. ``None`` disables the
        box-pick path entirely.
    gp_resolver
        Callable ``(picked_actor, cell_id) -> (element_id, gp_index,
        world_xyz) | None``. Required for the ``"gp"`` pick mode;
        consulted only when ``controller.mode == "gp"``. The
        ResultsViewer typically passes a closure that walks active
        GaussPointDiagrams and calls
        :meth:`GaussPointDiagram.resolve_picked_cell`.
    drag_threshold_px
        Pixel distance during the press required to be treated as a
        drag rather than a click.

    Returns
    -------
    ResultsPickController
        Live controller — flip ``ctrl.mode`` to switch dispatch
        between node, element, and GP picks.
    """
    import vtk

    iren_wrap = plotter.iren
    assert iren_wrap is not None, (
        "plotter.iren is None; call plotter.show() before installing "
        "the results pick observer."
    )
    iren = iren_wrap.interactor
    renderer = plotter.renderer

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    controller = ResultsPickController()
    cell_to_element_id = scene.cell_to_element_id
    node_ids_arr = np.asarray(scene.node_ids, dtype=np.int64)
    grid = scene.grid

    _press_pos: list[tuple | None] = [None]
    _dragging: list[bool] = [False]
    _tags: dict[str, int] = {}

    # Lazy-init rubber-band rectangle overlay (mirrors pick_engine.py).
    _rubberband_pts: list[Any] = [None]
    _rubberband_actor: list[Any] = [None]

    def _abort(caller: Any, tag: int) -> None:
        cmd = caller.GetCommand(tag)
        if cmd is not None:
            cmd.SetAbortFlag(1)

    # ------------------------------------------------------------------
    # Rubber-band overlay
    # ------------------------------------------------------------------

    def _ensure_rubberband() -> None:
        if _rubberband_pts[0] is not None:
            return
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(4)
        for i in range(4):
            pts.SetPoint(i, 0.0, 0.0, 0.0)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(5)
        for i in (0, 1, 2, 3, 0):
            lines.InsertCellPoint(i)
        poly = vtk.vtkPolyData()
        poly.SetPoints(pts)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 0.0)
        actor.GetProperty().SetLineWidth(1.5)
        actor.VisibilityOff()
        renderer.AddActor2D(actor)
        _rubberband_pts[0] = pts
        _rubberband_actor[0] = actor

    def _update_rubberband(x0: int, y0: int, x1: int, y1: int) -> None:
        _ensure_rubberband()
        pts = _rubberband_pts[0]
        pts.SetPoint(0, x0, y0, 0.0)
        pts.SetPoint(1, x1, y0, 0.0)
        pts.SetPoint(2, x1, y1, 0.0)
        pts.SetPoint(3, x0, y1, 0.0)
        pts.Modified()
        # Stippled line for crossing (R→L); solid for window (L→R).
        prop = _rubberband_actor[0].GetProperty()
        prop.SetLineStipplePattern(0xAAAA if x1 < x0 else 0xFFFF)
        _rubberband_actor[0].VisibilityOn()
        plotter.render()

    def _hide_rubberband() -> None:
        if _rubberband_actor[0] is not None:
            _rubberband_actor[0].VisibilityOff()
            plotter.render()

    # ------------------------------------------------------------------
    # Click resolution
    # ------------------------------------------------------------------

    def _build_result(x: int, y: int) -> Optional[PickResult]:
        try:
            picker.Pick(int(x), int(y), 0, renderer)
            cell_id = picker.GetCellId()
            if cell_id < 0:
                return None
            world = tuple(picker.GetPickPosition())
        except Exception:
            return None

        mode = controller.mode
        if mode == MODE_NODE:
            return PickResult(kind=MODE_NODE, world=world)
        if mode == MODE_ELEMENT:
            try:
                if 0 <= cell_id < cell_to_element_id.size:
                    element_id = int(cell_to_element_id[cell_id])
                else:
                    return None
            except Exception:
                return None
            return PickResult(
                kind=MODE_ELEMENT,
                world=world,
                element_id=element_id,
                cell_id=int(cell_id),
            )
        if mode == MODE_GP:
            if gp_resolver is None:
                return None
            try:
                actor = picker.GetActor()
            except Exception:
                actor = None
            try:
                hit = gp_resolver(actor, int(cell_id))
            except Exception:
                hit = None
            if hit is None:
                return None
            element_id, gp_index, gp_world = hit
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
    # Box-pick resolution
    # ------------------------------------------------------------------

    def _build_box_result(
        x0: int, y0: int, x1: int, y1: int,
    ) -> Optional[BoxPickResult]:
        if x0 == x1 or y0 == y1:
            return None    # Degenerate rectangle — nothing to pick.
        crossing = x1 < x0
        mode = controller.mode
        if mode == MODE_NODE:
            try:
                pts = np.asarray(grid.points, dtype=np.float64)
            except Exception:
                return None
            display = _project_points_to_display(pts, renderer)
            mask = _inside_box(display, x0, y0, x1, y1)
            picked = node_ids_arr[mask]
            return BoxPickResult(
                kind=MODE_NODE,
                ids=picked,
                cell_ids=np.zeros(0, dtype=np.int64),
                gp_indices=np.zeros(0, dtype=np.int64),
                box=(x0, y0, x1, y1),
                crossing=crossing,
            )
        if mode == MODE_ELEMENT:
            try:
                centroids = np.asarray(
                    grid.cell_centers().points, dtype=np.float64,
                )
            except Exception:
                return None
            display = _project_points_to_display(centroids, renderer)
            mask = _inside_box(display, x0, y0, x1, y1)
            cell_idx = np.nonzero(mask)[0].astype(np.int64)
            element_ids = (
                cell_to_element_id[cell_idx]
                if cell_idx.size else np.zeros(0, dtype=np.int64)
            )
            return BoxPickResult(
                kind=MODE_ELEMENT,
                ids=np.asarray(element_ids, dtype=np.int64),
                cell_ids=cell_idx,
                gp_indices=np.zeros(0, dtype=np.int64),
                box=(x0, y0, x1, y1),
                crossing=crossing,
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
            display = _project_points_to_display(centers, renderer)
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
    # Observer callbacks
    # ------------------------------------------------------------------

    def on_lmb_press(caller: Any, _event: str) -> None:
        try:
            if caller.GetShiftKey():
                return
        except Exception:
            pass
        _press_pos[0] = caller.GetEventPosition()
        _dragging[0] = False
        _abort(caller, _tags["lmb_press"])

    def on_mouse_move(caller: Any, _event: str) -> None:
        if _press_pos[0] is None:
            return
        px, py = caller.GetEventPosition()
        sx, sy = _press_pos[0]
        if not _dragging[0]:
            if (px - sx) ** 2 + (py - sy) ** 2 > drag_threshold_px ** 2:
                _dragging[0] = True
        if _dragging[0] and on_box_pick is not None:
            _update_rubberband(sx, sy, px, py)
        _abort(caller, _tags["move"])

    def on_lmb_release(caller: Any, _event: str) -> None:
        if _press_pos[0] is None:
            return
        was_drag = _dragging[0]
        sx, sy = _press_pos[0]
        x, y = caller.GetEventPosition()
        _press_pos[0] = None
        _dragging[0] = False
        _abort(caller, _tags["lmb_release"])
        _hide_rubberband()
        if was_drag:
            if on_box_pick is None:
                return
            box_result = _build_box_result(sx, sy, x, y)
            if box_result is None:
                return
            try:
                on_box_pick(box_result)
            except Exception as exc:
                import sys
                print(
                    f"[results_pick] on_box_pick raised: {exc}",
                    file=sys.stderr,
                )
            return
        result = _build_result(x, y)
        if result is None:
            return
        try:
            on_pick(result)
        except Exception as exc:
            import sys
            print(
                f"[results_pick] on_pick raised: {exc}",
                file=sys.stderr,
            )

    _tags["lmb_press"]   = iren.AddObserver(
        "LeftButtonPressEvent",   on_lmb_press,   10.0,
    )
    _tags["move"]        = iren.AddObserver(
        "MouseMoveEvent",         on_mouse_move,   9.0,
    )
    _tags["lmb_release"] = iren.AddObserver(
        "LeftButtonReleaseEvent", on_lmb_release, 10.0,
    )

    return controller
