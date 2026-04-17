"""
Navigation — Camera control via VTK observers.

Installs mouse/keyboard bindings on a PyVista plotter:

    Shift + Scroll wheel : quaternion orbit (around pivot or scene centre)
    Scroll click (MMB)   : pan camera
    Scroll wheel         : zoom toward mouse cursor
    RMB drag             : pan camera (secondary)

LMB is NOT intercepted — the :mod:`pick_engine` handles it.

Usage::

    from apeGmsh.viewers.core.navigation import install_navigation
    install_navigation(plotter, get_orbit_pivot=lambda: (0, 0, 0))
"""
from __future__ import annotations

import math
from typing import Callable

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

    # ── register ────────────────────────────────────────────────────
    _tags["move"]        = iren.AddObserver("MouseMoveEvent",            on_mouse_move,   10.0)
    _tags["mmb_press"]   = iren.AddObserver("MiddleButtonPressEvent",    on_mmb_press,    10.0)
    _tags["mmb_release"] = iren.AddObserver("MiddleButtonReleaseEvent",  on_mmb_release,  10.0)
    _tags["scroll_fwd"]  = iren.AddObserver("MouseWheelForwardEvent",    on_scroll_fwd,   10.0)
    _tags["scroll_bwd"]  = iren.AddObserver("MouseWheelBackwardEvent",   on_scroll_bwd,   10.0)
    _tags["rmb_press"]   = iren.AddObserver("RightButtonPressEvent",     on_rmb_press,    10.0)
    _tags["rmb_release"] = iren.AddObserver("RightButtonReleaseEvent",   on_rmb_release,  10.0)
