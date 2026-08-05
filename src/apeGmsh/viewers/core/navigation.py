"""
Navigation — Camera control via VTK observers.

Installs mouse/keyboard bindings on a PyVista plotter:

    Shift + LMB drag     : turntable orbit (yaw + pitch, no roll)
    Shift + MMB drag     : turntable orbit (same; legacy chord)
    MMB drag             : pan
    RMB drag             : pan (secondary)
    Scroll wheel         : zoom (focal point fixed → stable rotate pivot)

LMB without a modifier is NOT intercepted — the :mod:`pick_engine`
handles it (click = pick, drag = rubber-band). Shift+LMB IS
intercepted at priority 11 (above pick_engine's 10) so the orbit
gesture shadows the rubber-band gesture cleanly.

Both orbit gestures route through :func:`_orbit_around` — a
TURNTABLE: azimuth spins around the WORLD +Z axis (structural models
are Z-up, and a column must rotate about its own vertical no matter
how the camera is pitched), elevation pitches around the horizontal
right vector and is clamped short of the poles so the model can
never flip. After every tick the view-up is re-derived as the
projection of world +Z onto the view plane, so the horizon is level
by construction — no roll can accumulate. (The previous scheme yawed
around ``cam.GetViewUp()``: once pitched, yaw precessed the model
around a tilted axis and long combined drags drifted the horizon.)

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


# The turntable's DEFAULT spin axis. Structural models are Z-up; a
# building must rotate about its own vertical regardless of camera
# pitch, and the horizon must stay level. The axis is selectable per
# viewer (``NavigationHandle.set_spin_axis`` — e.g. a bridge deck
# spun about its longitudinal X) but world +Z is the main one.
_WORLD_UP = (0.0, 0.0, 1.0)

# Elevation clamp — the view direction may come this close to the spin
# axis and no closer (radians from the horizontal plane). Stops the
# pole-flip: pitching past straight-down used to invert the up vector,
# after which yaw reversed direction.
_ELEVATION_LIMIT = math.radians(89.0)


def _orbit_around(
    renderer, pivot: tuple, dx_px: int, dy_px: int,
    clip_bounds: tuple | None = None,
    spin_axis: tuple = _WORLD_UP,
) -> None:
    """Turntable-orbit the camera around *pivot*.

    Azimuth rotates around *spin_axis* (default :data:`_WORLD_UP`);
    elevation rotates around the right vector perpendicular to it,
    clamped to ±:data:`_ELEVATION_LIMIT` measured from the plane
    normal to the axis. Because the yaw axis is world-fixed, azimuth
    never changes the view direction's angle to the axis — the two
    axes decouple exactly, and the clamp can be computed on elevation
    alone. The view-up is then re-derived as the spin axis projected
    onto the view plane: a level "horizon" (axis-up on screen) by
    construction, not by incremental orthogonalisation.

    *clip_bounds* — when given, a static ``(xmin,xmax,ymin,ymax,
    zmin,zmax)`` passed to ``ResetCameraClippingRange(bounds)``. The
    no-arg form calls ``ComputeVisiblePropBounds`` every tick, which
    queries the camera-dependent silhouette actor and forces a full
    silhouette recompute (~107 ms/frame on a 600 k mesh). Geometry
    bounds don't change while orbiting (only the camera moves), so a
    cached superset is correct and skips that cascade entirely.
    """
    cam = renderer.GetActiveCamera()
    pos = cam.GetPosition()
    fp  = cam.GetFocalPoint()

    az = -dx_px * 0.005
    el =  dy_px * 0.005

    vd = [fp[i] - pos[i] for i in range(3)]
    d = math.sqrt(sum(v * v for v in vd))
    if d < 1e-12:
        return
    vd = [v / d for v in vd]

    an = math.sqrt(sum(v * v for v in spin_axis))
    if an < 1e-12:
        return
    axis = [v / an for v in spin_axis]

    # Elevation of the view direction out of the plane normal to the
    # spin axis. Pitching about the right vector changes it by exactly
    # ``el``; clamp the increment so it never reaches the poles.
    dot_va = sum(vd[i] * axis[i] for i in range(3))
    sin_e = max(-1.0, min(1.0, dot_va))
    e_now = math.asin(sin_e)
    e_new = max(-_ELEVATION_LIMIT, min(_ELEVATION_LIMIT, e_now + el))
    el = e_new - e_now

    # Right = viewdir × spin-axis — the pitch axis, perpendicular to
    # both. Near a pole (view direction parallel to the axis — only
    # reachable if the camera ARRIVED there via a preset view; the
    # clamp keeps orbits away) the cross degenerates: fall back to the
    # camera's own right so a pitch can still walk the view off the
    # pole.
    right = [
        vd[1] * axis[2] - vd[2] * axis[1],
        vd[2] * axis[0] - vd[0] * axis[2],
        vd[0] * axis[1] - vd[1] * axis[0],
    ]
    rn = math.sqrt(sum(v * v for v in right))
    if rn < 1e-9:
        up = cam.GetViewUp()
        right = [
            vd[1] * up[2] - vd[2] * up[1],
            vd[2] * up[0] - vd[0] * up[2],
            vd[0] * up[1] - vd[1] * up[0],
        ]
        rn = math.sqrt(sum(v * v for v in right))
        if rn < 1e-12:
            return
    right = [v / rn for v in right]

    q = _qmul(_quat(right, el), _quat(axis, az))

    p_rel  = tuple(pos[i] - pivot[i] for i in range(3))
    fp_rel = tuple(fp[i] - pivot[i] for i in range(3))

    rp  = _qrot(q, p_rel)
    rfp = _qrot(q, fp_rel)

    new_pos = tuple(pivot[i] + rp[i] for i in range(3))
    new_fp  = tuple(pivot[i] + rfp[i] for i in range(3))
    cam.SetPosition(*new_pos)
    cam.SetFocalPoint(*new_fp)

    # Level horizon by construction: view-up = the spin axis projected
    # onto the plane perpendicular to the new view direction. The
    # clamp bounds |vd'·axis| ≤ sin(89°), so the projection never
    # degenerates.
    vd2 = [new_fp[i] - new_pos[i] for i in range(3)]
    d2 = math.sqrt(sum(v * v for v in vd2))
    if d2 > 1e-12:
        vd2 = [v / d2 for v in vd2]
        dot_av = sum(axis[i] * vd2[i] for i in range(3))
        proj = [axis[i] - dot_av * vd2[i] for i in range(3)]
        pn = math.sqrt(sum(v * v for v in proj))
        if pn > 1e-9:
            cam.SetViewUp(*(v / pn for v in proj))

    if clip_bounds is not None and clip_bounds[0] <= clip_bounds[1]:
        renderer.ResetCameraClippingRange(*clip_bounds)
    else:
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
    on_shift_click: (
        Callable[["tuple[float, float, float]", Any], None] | None
    ) = None,
    drag_threshold_px: int = 4,
) -> "NavigationHandle":
    """Install camera-control observers on *plotter*.

    Returns a :class:`NavigationHandle`; call its
    ``invalidate_bounds()`` whenever the scene's world extent changes
    (deform scale, geometry offsets, added geometries) so the cached
    clipping-range superset is recomputed on the next gesture instead
    of clipping the grown scene.

    Parameters
    ----------
    plotter : pv.Plotter
        The PyVista plotter (or QtInteractor).
    get_orbit_pivot : callable, optional
        Returns ``(x, y, z)`` for orbit centre (e.g. selection centroid).
        When ``None`` or when the callable returns ``None``, orbit
        defaults to the camera's focal point.
    on_shift_click : callable, optional
        Invoked with the world-space ``(x, y, z)`` of a Shift+LMB click
        when no drag occurred (drag triggers a rotate gesture instead),
        plus the picked VTK prop (``None`` when the picker has no hit
        actor) so consumers can attribute the click to a geometry
        (ADR 0058 S3b — pin-aware time-history).
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
    # Armed on Shift+LMB drag-threshold cross or Shift+MMB press;
    # cleared on release. Both chords drive the same turntable orbit
    # (yaw about the spin axis + clamped pitch — see ``_orbit_around``).
    _orbit_pivot: list[tuple | None] = [None]
    _orbit_last:  list[tuple | None] = [None]
    # The turntable's spin axis — world +Z (structural Z-up) unless
    # the handle's ``set_spin_axis`` selects another (e.g. a bridge
    # deck spun about its longitudinal axis).
    _spin_axis: list[tuple] = [_WORLD_UP]

    # ── Shift+LMB drag-rotate state ─────────────────────────────────
    # Set on Shift+LMB press; cleared on release. While set, the
    # mouse-move observer watches for drag-threshold crossing and
    # promotes the gesture to a trackball rotate.
    _shift_lmb_press_pos: list[tuple | None] = [None]
    _shift_lmb_did_rotate: list[bool] = [False]

    # Cached lazy picker for the on_shift_click callback's world coord.
    _shift_click_picker: list[Any] = [None]

    # ── cached static clipping bounds ───────────────────────────────
    # Computed once, lazily, on the first camera gesture. The initial
    # scene has every actor visible, so its bounds are an encompassing
    # superset for the whole session — hiding entities only shrinks
    # geometry, and a too-generous clipping range is harmless (minor
    # depth precision) whereas recomputing per frame stalls on the
    # silhouette filter. Reused by orbit + zoom.
    _clip_bounds: list[tuple | None] = [None]

    def _ensure_clip_bounds() -> tuple | None:
        if _clip_bounds[0] is None:
            try:
                b = renderer.ComputeVisiblePropBounds()
                if b is not None and b[0] <= b[1]:
                    _clip_bounds[0] = tuple(b)
            except Exception:
                pass
        return _clip_bounds[0]

    def _pivot_center() -> tuple[float, float, float]:
        """Orbit-pivot fallback — the camera's current focal point.

        The focal point is what the user is actually looking at: zoom
        deliberately anchors it and pan carries it along, so orbiting
        about it keeps the framed detail on screen. (The previous
        fallback — scene-bounds centre cached at the first gesture —
        swung a zoomed-in view around the whole-model centre, and went
        stale the moment deform / offsets grew the scene.)
        """
        try:
            fp = renderer.GetActiveCamera().GetFocalPoint()
            return (float(fp[0]), float(fp[1]), float(fp[2]))
        except Exception:
            return _scene_center(renderer)

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
            dx = px - lx
            dy = py - ly
            _orbit_around(
                renderer, _orbit_pivot[0], dx, dy, _ensure_clip_bounds(),
                spin_axis=_spin_axis[0],
            )
            plotter.render()
            _abort(caller, _tags["move"])
            return
        # Shift+LMB drag-detect: promote to turntable orbit once the
        # mouse crosses the drag threshold. Below the threshold, do
        # nothing so a quick click still falls through to
        # ``on_shift_click``. We arm the same orbit state Shift+MMB
        # uses (``_orbit_pivot`` + ``_orbit_last``) so the next move
        # event drives ``_orbit_around`` via the branch above.
        if (
            _shift_lmb_press_pos[0] is not None
            and not _shift_lmb_did_rotate[0]
        ):
            px, py = caller.GetEventPosition()
            sx, sy = _shift_lmb_press_pos[0]
            if (px - sx) ** 2 + (py - sy) ** 2 > drag_threshold_px ** 2:
                pivot = None
                if get_orbit_pivot is not None:
                    pivot = get_orbit_pivot()
                if pivot is None:
                    pivot = _pivot_center()
                _orbit_pivot[0] = pivot
                _orbit_last[0] = (px, py)
                _shift_lmb_did_rotate[0] = True
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

    # ── Scroll: zoom around the focal point ─────────────────────────
    # Cursor-anchored zoom (the previous behaviour) drifted the focal
    # point on every scroll, which made the trackball Rotate() pivot
    # on whatever world point was last under the cursor — disorienting
    # once you've scrolled a few times. We trade that for a stable
    # rotate pivot: parallel projection scales ``parallel_scale``,
    # perspective dollies ``position`` along the view ray, and the
    # focal point never moves.
    def _zoom(direction: int):
        cam = renderer.GetActiveCamera()
        factor = 1.1 if direction > 0 else 1.0 / 1.1

        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / factor)
        else:
            pos = cam.GetPosition()
            fp = cam.GetFocalPoint()
            new_pos = tuple(fp[i] + (pos[i] - fp[i]) / factor for i in range(3))
            cam.SetPosition(*new_pos)

        cb = _ensure_clip_bounds()
        if cb is not None and cb[0] <= cb[1]:
            renderer.ResetCameraClippingRange(*cb)
        else:
            renderer.ResetCameraClippingRange()
        plotter.render()

    def on_scroll_fwd(caller, _event):
        _zoom(+1)
        _abort(caller, _tags["scroll_fwd"])

    def on_scroll_bwd(caller, _event):
        _zoom(-1)
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

    # ── Shift+LMB: drag-detect → orbit; click → on_shift_click ──────
    # Plain LMB is reserved for pick_engine (click = pick, drag =
    # rubber-band). Shift+LMB drags engage the quaternion turntable
    # orbit (no-roll); Shift+LMB clicks (no drag) fire the optional
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
        """``(world, prop)`` under the cursor, or ``None`` on a miss.

        ``prop`` is the picked actor (``GetProp3D``) so the consumer
        can resolve which geometry was hit (ADR 0058 S3b)."""
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
            return tuple(picker.GetPickPosition()), picker.GetProp3D()
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
            _orbit_pivot[0] = None
            _orbit_last[0] = None
            _abort(caller, _tags["lmb_release"])
            return
        # No drag — fire the click callback if one is wired.
        if on_shift_click is not None:
            hit = _world_pos_at_screen(x, y)
            if hit is not None:
                world, prop = hit
                try:
                    on_shift_click(world, prop)
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

    def _invalidate_bounds() -> None:
        _clip_bounds[0] = None

    def _set_spin_axis(axis: "tuple[float, float, float]") -> None:
        n = math.sqrt(sum(float(v) * float(v) for v in axis))
        if n < 1e-12:
            raise ValueError(f"spin axis must be non-zero, got {axis!r}")
        _spin_axis[0] = tuple(float(v) / n for v in axis)

    def _get_spin_axis() -> "tuple[float, float, float]":
        return _spin_axis[0]

    return NavigationHandle(
        invalidate_bounds=_invalidate_bounds,
        set_spin_axis=_set_spin_axis,
        get_spin_axis=_get_spin_axis,
    )


class NavigationHandle:
    """Handle returned by :func:`install_navigation`.

    ``invalidate_bounds()`` drops the cached clipping-range superset
    so the next gesture recomputes it. The cache exists because
    ``ComputeVisiblePropBounds`` per orbit tick stalls on the
    silhouette filter; recomputing once per gesture after the scene
    grows (deform, offsets, new geometries) keeps both properties.

    ``set_spin_axis((x, y, z))`` re-aims the turntable: yaw spins
    about the given world axis (normalised; zero raises) and the
    "horizon" levels against it. ``get_spin_axis()`` reads it back.
    The default is world +Z.
    """

    def __init__(
        self,
        *,
        invalidate_bounds: Callable[[], None],
        set_spin_axis: "Callable[[tuple[float, float, float]], None]",
        get_spin_axis: "Callable[[], tuple[float, float, float]]",
    ) -> None:
        self.invalidate_bounds = invalidate_bounds
        self.set_spin_axis = set_spin_axis
        self.get_spin_axis = get_spin_axis
