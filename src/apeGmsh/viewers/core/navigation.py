"""
Navigation — Camera control via VTK observers.

Installs mouse/keyboard bindings on a PyVista plotter:

    Shift + LMB drag     : rotate (yaw/pitch via trackball Rotate())
    Shift + MMB drag     : quaternion orbit around pivot (or scene centre)
    MMB drag             : pan
    RMB drag             : pan (secondary)
    Scroll wheel         : zoom toward mouse cursor

LMB without a modifier is NOT intercepted — the :mod:`pick_engine`
handles it (click = pick, drag = rubber-band). Shift+LMB IS
intercepted at priority 11 (above pick_engine's 10) so the rotate
gesture shadows the rubber-band gesture cleanly.

Usage::

    from apeGmsh.viewers.core.navigation import install_navigation
    install_navigation(plotter, get_orbit_pivot=lambda: (0, 0, 0))
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pyvista as pv


# ======================================================================
# Quaternion helpers (pure math, no VTK)
# ======================================================================

def _quat(axis: tuple | list, angle: float) -> tuple:
    """Quaternion (w, x, y, z) from axis-angle."""
    ha = angle * 0.5
    s = math.sin(ha)
    return (math.cos(ha), axis[0] * s, axis[1] * s, axis[2] * s)


def _qmul(a: tuple, b: tuple) -> tuple:
    return (
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    )


def _qconj(q: tuple) -> tuple:
    return (q[0], -q[1], -q[2], -q[3])


def _qrot(q: tuple, v: tuple) -> tuple:
    """Rotate vector *v* by quaternion *q*."""
    vq = (0.0, v[0], v[1], v[2])
    r = _qmul(_qmul(q, vq), _qconj(q))
    return (r[1], r[2], r[3])


def _orbit_around(renderer, pivot: tuple, dx_px: int, dy_px: int) -> None:
    """Orbit camera around *pivot* using quaternion rotation."""
    cam = renderer.GetActiveCamera()
    pos = cam.GetPosition()
    fp  = cam.GetFocalPoint()
    up  = cam.GetViewUp()

    az = -dx_px * 0.005
    el =  dy_px * 0.005

    vd = [fp[i] - pos[i] for i in range(3)]
    d = math.sqrt(sum(v * v for v in vd))
    if d < 1e-12:
        return
    vd = [v / d for v in vd]

    # Right = viewdir × up
    right = [
        vd[1] * up[2] - vd[2] * up[1],
        vd[2] * up[0] - vd[0] * up[2],
        vd[0] * up[1] - vd[1] * up[0],
    ]
    rn = math.sqrt(sum(v * v for v in right))
    if rn < 1e-12:
        return
    right = [v / rn for v in right]

    un = math.sqrt(sum(v * v for v in up))
    up_n = [v / un for v in up] if un > 1e-12 else [0, 0, 1]

    q = _qmul(_quat(right, el), _quat(up_n, az))

    p_rel  = tuple(pos[i] - pivot[i] for i in range(3))
    fp_rel = tuple(fp[i] - pivot[i] for i in range(3))

    rp  = _qrot(q, p_rel)
    rfp = _qrot(q, fp_rel)

    cam.SetPosition(*(pivot[i] + rp[i] for i in range(3)))
    cam.SetFocalPoint(*(pivot[i] + rfp[i] for i in range(3)))
    cam.OrthogonalizeViewUp()
    renderer.ResetCameraClippingRange()


def _scene_center(renderer) -> tuple[float, float, float]:
    """Centre of the visible scene bounding box."""
    try:
        b = renderer.ComputeVisiblePropBounds()
        if b[0] <= b[1]:
            return (
                (b[0] + b[1]) * 0.5,
                (b[2] + b[3]) * 0.5,
                (b[4] + b[5]) * 0.5,
            )
    except Exception:
        pass
    return (0.0, 0.0, 0.0)


# ======================================================================
# Public API
# ======================================================================

def install_navigation(
    plotter: pv.Plotter,
    *,
    get_orbit_pivot: Callable[[], tuple[float, float, float] | None] | None = None,
    on_shift_click: Callable[["tuple[float, float, float]"], None] | None = None,
    drag_threshold_px: int = 4,
) -> None:
    """Install camera-control observers on *plotter*.

    Parameters
    ----------
    plotter : pv.Plotter
        The PyVista plotter (or QtInteractor).
    get_orbit_pivot : callable, optional
        Returns ``(x, y, z)`` for orbit centre (e.g. selection centroid).
        When ``None`` or when the callable returns ``None``, orbit
        defaults to the visible scene centre.
    on_shift_click : callable, optional
        Invoked with the world-space ``(x, y, z)`` of a Shift+LMB click
        when no drag occurred (drag triggers a rotate gesture instead).
        Use this to wire a "shift-click adds a probe / time-history"
        consumer without registering a separate VTK observer.
    drag_threshold_px : int
        Pixel distance the mouse must travel after Shift+LMB press
        before the gesture is treated as a drag (rotate). Below this
        threshold, ``on_shift_click`` fires on release instead.
    """
    import vtk

    iren_wrap = plotter.iren
    assert iren_wrap is not None, (
        "plotter.iren is None; call plotter.show() or pass an active "
        "plotter with a realized render window before attaching the "
        "orbit-camera style."
    )
    iren = iren_wrap.interactor
    renderer = plotter.renderer

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)

    # ── orbit state ─────────────────────────────────────────────────
    _orbit_pivot: list[tuple | None] = [None]
    _orbit_last:  list[tuple | None] = [None]

    # ── Shift+LMB drag-rotate state ─────────────────────────────────
    # Set on Shift+LMB press; cleared on release. While set, the
    # mouse-move observer watches for drag-threshold crossing and
    # promotes the gesture to a trackball rotate.
    _shift_lmb_press_pos: list[tuple | None] = [None]
    _shift_lmb_did_rotate: list[bool] = [False]

    # Cached lazy picker for the on_shift_click callback's world coord.
    _shift_click_picker: list[Any] = [None]

    # ── tag storage for observer removal (if needed) ────────────────
    _tags: dict[str, int] = {}

    def _abort(caller, tag: int) -> None:
        cmd = caller.GetCommand(tag)
        if cmd is not None:
            cmd.SetAbortFlag(1)

    # ── mouse move: orbit if active ─────────────────────────────────
    def on_mouse_move(caller, _event):
        if _orbit_pivot[0] is not None and _orbit_last[0] is not None:
            px, py = caller.GetEventPosition()
            lx, ly = _orbit_last[0]
            _orbit_last[0] = (px, py)
            _orbit_around(renderer, _orbit_pivot[0], px - lx, py - ly)
            plotter.render()
            _abort(caller, _tags["move"])
            return
        # Shift+LMB drag-detect: promote to rotate once the mouse
        # crosses the drag threshold. Below the threshold, do nothing
        # so a quick click still falls through to ``on_shift_click``.
        if (
            _shift_lmb_press_pos[0] is not None
            and not _shift_lmb_did_rotate[0]
        ):
            px, py = caller.GetEventPosition()
            sx, sy = _shift_lmb_press_pos[0]
            if (px - sx) ** 2 + (py - sy) ** 2 > drag_threshold_px ** 2:
                style.StartRotate()
                _shift_lmb_did_rotate[0] = True
            # Don't abort — the trackball at the default priority
            # consumes the move and runs Rotate() now that state is
            # VTKIS_ROTATE.
        # Don't abort — let trackball handle pan, let pick_engine see moves

    # ── Shift+Scroll: orbit ─────────────────────────────────────────
    def on_mmb_press(caller, _event):
        if caller.GetShiftKey():
            # Shift+MMB -> orbit
            pivot = None
            if get_orbit_pivot is not None:
                pivot = get_orbit_pivot()
            if pivot is None:
                pivot = _scene_center(renderer)
            _orbit_pivot[0] = pivot
            _orbit_last[0] = caller.GetEventPosition()
        else:
            # MMB -> pan
            _orbit_pivot[0] = None
            style.StartPan()
        _abort(caller, _tags["mmb_press"])

    def on_mmb_release(caller, _event):
        if _orbit_pivot[0] is not None:
            _orbit_pivot[0] = None
            _orbit_last[0] = None
            _abort(caller, _tags["mmb_release"])
            return
        state = style.GetState()
        if state == 2:  # VTKIS_PAN
            style.EndPan()
        _abort(caller, _tags["mmb_release"])

    # ── Scroll: zoom toward cursor ─────────────────────────────────
    def _zoom_to_cursor(caller, direction: int):
        """Zoom toward the world point under the mouse cursor."""
        mx, my = caller.GetEventPosition()
        cam = renderer.GetActiveCamera()

        # World point under cursor (at the front clipping plane)
        renderer.SetDisplayPoint(mx, my, 0.0)
        renderer.DisplayToWorld()
        wp = renderer.GetWorldPoint()
        if abs(wp[3]) < 1e-12:
            return
        world_pt = (wp[0] / wp[3], wp[1] / wp[3], wp[2] / wp[3])

        factor = 1.1 if direction > 0 else 1.0 / 1.1

        pos = cam.GetPosition()
        fp = cam.GetFocalPoint()

        # Scale distance from cursor point
        new_pos = tuple(world_pt[i] + (pos[i] - world_pt[i]) / factor for i in range(3))
        new_fp = tuple(world_pt[i] + (fp[i] - world_pt[i]) / factor for i in range(3))

        cam.SetPosition(*new_pos)
        cam.SetFocalPoint(*new_fp)

        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / factor)

        renderer.ResetCameraClippingRange()
        plotter.render()

    def on_scroll_fwd(caller, _event):
        _zoom_to_cursor(caller, +1)
        _abort(caller, _tags["scroll_fwd"])

    def on_scroll_bwd(caller, _event):
        _zoom_to_cursor(caller, -1)
        _abort(caller, _tags["scroll_bwd"])

    # ── RMB: pan ────────────────────────────────────────────────────
    def on_rmb_press(caller, _event):
        style.StartPan()
        _abort(caller, _tags["rmb_press"])

    def on_rmb_release(caller, _event):
        state = style.GetState()
        if state == 2:
            style.EndPan()
        _abort(caller, _tags["rmb_release"])

    # ── Shift+LMB: drag-detect → rotate; click → on_shift_click ─────
    # Plain LMB is reserved for pick_engine (click = pick, drag =
    # rubber-band). Shift+LMB drags engage the trackball's Rotate
    # state mid-gesture; Shift+LMB clicks (no drag) fire the optional
    # ``on_shift_click`` callback so consumers can wire a "shift-click
    # adds a probe" behavior without their own observer. The press
    # observer aborts so pick_engine (priority 10) and any default
    # trackball Shift+LMB handling (which would Pan) never see the
    # event.
    def on_lmb_press(caller, _event):
        if not caller.GetShiftKey():
            return    # Plain LMB falls through to pick_engine.
        _shift_lmb_press_pos[0] = caller.GetEventPosition()
        _shift_lmb_did_rotate[0] = False
        _abort(caller, _tags["lmb_press"])

    def _world_pos_at_screen(x: int, y: int):
        if _shift_click_picker[0] is None:
            import vtk
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            _shift_click_picker[0] = picker
        picker = _shift_click_picker[0]
        try:
            picker.Pick(int(x), int(y), 0, renderer)
            if picker.GetCellId() < 0:
                return None
            return tuple(picker.GetPickPosition())
        except Exception:
            return None

    def on_lmb_release(caller, _event):
        if _shift_lmb_press_pos[0] is None:
            return
        did_rotate = _shift_lmb_did_rotate[0]
        x, y = caller.GetEventPosition()
        _shift_lmb_press_pos[0] = None
        _shift_lmb_did_rotate[0] = False
        if did_rotate:
            style.EndRotate()
            _abort(caller, _tags["lmb_release"])
            return
        # No drag — fire the click callback if one is wired.
        if on_shift_click is not None:
            world = _world_pos_at_screen(x, y)
            if world is not None:
                try:
                    on_shift_click(world)
                except Exception as exc:
                    import sys
                    print(
                        f"[navigation] on_shift_click raised: {exc}",
                        file=sys.stderr,
                    )
        _abort(caller, _tags["lmb_release"])

    # ── register ────────────────────────────────────────────────────
    _tags["move"]        = iren.AddObserver("MouseMoveEvent",            on_mouse_move,   10.0)
    _tags["mmb_press"]   = iren.AddObserver("MiddleButtonPressEvent",    on_mmb_press,    10.0)
    _tags["mmb_release"] = iren.AddObserver("MiddleButtonReleaseEvent",  on_mmb_release,  10.0)
    _tags["scroll_fwd"]  = iren.AddObserver("MouseWheelForwardEvent",    on_scroll_fwd,   10.0)
    _tags["scroll_bwd"]  = iren.AddObserver("MouseWheelBackwardEvent",   on_scroll_bwd,   10.0)
    _tags["rmb_press"]   = iren.AddObserver("RightButtonPressEvent",     on_rmb_press,    10.0)
    _tags["rmb_release"] = iren.AddObserver("RightButtonReleaseEvent",   on_rmb_release,  10.0)
    _tags["lmb_press"]   = iren.AddObserver("LeftButtonPressEvent",      on_lmb_press,    11.0)
    _tags["lmb_release"] = iren.AddObserver("LeftButtonReleaseEvent",    on_lmb_release,  11.0)
