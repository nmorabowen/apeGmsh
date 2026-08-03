"""Scope-box drag gizmo — the box you can grab (catalog).

Same discipline as ``test_clip_gizmo.py``: a scriptable fake interactor
drives the priority-12 observers, and a deterministic pixel-to-ray
camera lets a test aim at world points instead of hard-coding pixels.

* **S-HIT** — each of the six faces resolves to its own handle; the
  NEAREST face wins when two are on the ray; an edge-on face is not a
  target; ties break by declared handle order.
* **S-MISS** — a press away from the box does not abort, so camera and
  picking outside the gizmo are bit-for-bit what they were.
* **S-DRAG** — a face slides along its own world axis in the right
  direction; the gain stays bounded near face-on viewing (ADR 0083 S2
  finding C1); a LeaveEvent mid-drag ends it.
* **S-CLAMP** — a face dragged past its opposite STOPS there and the
  resulting ``BBox`` is valid; a degenerate box is still grabbable.
* **S-OWNER** — one ``set_scope`` + one announcement per move, and
  zero actor rebuilds.
* **S-RENDER** — layers exist for the ACTIVE geometry only, carry the
  clip exemption, and disappear with ``show_gizmo``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from apeGmsh.viewers.core._scope_gizmo import (
    DRAG_MIN_SEPARATION,
    FACE_HANDLES,
    SCOPE_GIZMO_LAYER_PREFIX,
    ScopeGizmoRenderer,
    axis_param,
    box_with_bound,
    clamp_bound,
    ray_hit,
    scope_gizmo_geometry,
)
from apeGmsh.viewers.core._scope_gizmo_interactor import (
    install_scope_gizmo_interactor,
)
from apeGmsh.viewers.core.scope_controller import ScopeController
from apeGmsh.viewers.scene_ir import BBox

_BBOX = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)
_GID = "geom-1"


def _box(lo=(-1.0, -1.0, -1.0), hi=(1.0, 1.0, 1.0)) -> BBox:
    return BBox(lo, hi)


# =====================================================================
# Fakes — a scriptable interactor + a ray-serving recording backend
# =====================================================================

class _Command:
    def __init__(self) -> None:
        self.aborted = False

    def SetAbortFlag(self, flag) -> None:
        self.aborted = bool(flag)


class _Iren:
    """Records observers and lets a test synthesize events."""

    def __init__(self) -> None:
        self.observers: dict[int, tuple[str, object, float]] = {}
        self.commands: dict[int, _Command] = {}
        self._next = 1
        self.pos = (0, 0)
        self.shift = False

    def AddObserver(self, event, cb, priority=0.0):
        tag = self._next
        self._next += 1
        self.observers[tag] = (event, cb, priority)
        self.commands[tag] = _Command()
        return tag

    def RemoveObserver(self, tag):
        self.observers.pop(tag, None)

    def GetCommand(self, tag):
        return self.commands.get(tag)

    def GetEventPosition(self):
        return self.pos

    def GetShiftKey(self):
        return self.shift

    def send(self, event, pos=None, *, shift=False):
        """Fire ``event`` at ``pos``; returns True if someone aborted."""
        if pos is not None:
            self.pos = pos
        self.shift = shift
        for tag, (name, cb, _prio) in sorted(
            self.observers.items(), key=lambda kv: -kv[1][2]
        ):
            if name != event:
                continue
            self.commands[tag].aborted = False
            cb(self, event)
            if self.commands[tag].aborted:
                return True
        return False


class _Window:
    def __init__(self) -> None:
        self.cursor = None

    def SetCurrentCursor(self, value):
        self.cursor = value


class _Handle:
    def __init__(self, layer_id: str) -> None:
        self.layer_id = layer_id


def _screen_basis(direction) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    ref = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(d[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    u = np.cross(d, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(d, u)
    return d, u, v


class _ScopeBackend:
    """Recording backend + an exact parallel-projection camera.

    Every pixel maps to a ray origin on a plane far behind the model,
    with one shared direction. ``pixel_for`` inverts it exactly (the
    basis is orthonormal), so tests aim at world points.
    """

    _WPP = 0.01     # world units per pixel
    _BACK = 50.0    # how far behind the origin the image plane sits

    def __init__(self, direction=(-0.4, -0.7, -1.0), viewport=(1000, 800)):
        self.viewport = viewport
        self._d, self._u, self._v = _screen_basis(direction)
        self._centre = -self._d * self._BACK
        self.layers: dict[str, object] = {}
        self.calls: list[tuple] = []
        self.iren = _Iren()
        self.render_window = _Window()
        outer = self
        self.plotter = type(
            "P", (), {
                "iren": type("W", (), {"interactor": outer.iren})(),
                "render_window": outer.render_window,
            },
        )()

    # -- rays --
    def display_to_world_ray(self, x, y):
        self.calls.append(("ray", float(x), float(y)))
        vw, vh = self.viewport
        origin = (
            self._centre
            + (float(x) - vw / 2.0) * self._WPP * self._u
            + (float(y) - vh / 2.0) * self._WPP * self._v
        )
        return tuple(origin), tuple(self._d)

    def pixel_for(self, world):
        """The pixel whose ray passes through ``world``."""
        rel = np.asarray(world, dtype=float) - self._centre
        vw, vh = self.viewport
        return (
            float(rel @ self._u) / self._WPP + vw / 2.0,
            float(rel @ self._v) / self._WPP + vh / 2.0,
        )

    # -- RenderBackend subset the renderer touches --
    def add_layer(self, layer):
        self.calls.append(("add", layer.layer_id))
        self.layers[layer.layer_id] = layer
        return _Handle(layer.layer_id)

    def update_layer(self, handle, layer):
        self.calls.append(("update", handle.layer_id))
        self.layers[handle.layer_id] = layer

    def remove_layer(self, handle):
        self.calls.append(("remove", handle.layer_id))
        self.layers.pop(handle.layer_id, None)


class _CountingScopes:
    """The real controller, with every ``set_scope`` recorded."""

    def __init__(self) -> None:
        self.inner = ScopeController()
        self.set_calls: list[tuple] = []

    def set_scope(self, geometry_id, box):
        self.set_calls.append((geometry_id, box))
        self.inner.set_scope(geometry_id, box)

    def box_for(self, geometry_id):
        return self.inner.box_for(geometry_id)

    @property
    def show_gizmo(self) -> bool:
        return self.inner.show_gizmo


def _rig(direction=(-0.4, -0.7, -1.0), box=None, active=_GID):
    """Backend + controller (one scoped geometry) + renderer + interactor."""
    backend = _ScopeBackend(direction)
    scopes = _CountingScopes()
    scopes.set_scope(_GID, box if box is not None else _box())
    scopes.set_calls.clear()
    renderer = ScopeGizmoRenderer(
        backend, scopes, bbox=_BBOX, active_id=lambda: active,
    )
    renderer.refresh()
    fired: list[str] = []
    interactor = install_scope_gizmo_interactor(
        backend, scopes, renderer, on_changed=fired.append,
    )
    assert interactor is not None
    return backend, scopes, renderer, interactor, fired


def _geom(renderer):
    return renderer.geometries()[_GID]


# =====================================================================
# S-HIT — six faces, one deterministic answer
# =====================================================================

@pytest.mark.parametrize("handle", FACE_HANDLES)
def test_each_face_resolves_to_its_own_handle(handle):
    """Look straight down a face's outward normal and you grab THAT
    face — all six of them, near one and far one alike."""
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    centre = np.asarray(geom.face_centre(handle))
    outward = np.zeros(3)
    from apeGmsh.viewers.core._scope_gizmo import face_axis
    axis = face_axis(handle)
    outward[axis] = 1.0 if handle.endswith("max") else -1.0

    backend = _ScopeBackend(-outward)
    origin, direction = backend.display_to_world_ray(
        *backend.pixel_for(centre)
    )
    result = ray_hit(geom, origin, direction)
    assert result is not None
    assert result[0] == handle


def test_the_nearest_face_wins_when_two_are_on_the_ray():
    """A ray down -Z passes through zmax then zmin; the one you can see
    is the one you get."""
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    origin, direction = (0.0, 0.0, 10.0), (0.0, 0.0, -1.0)

    # Both faces really are on this ray — otherwise the test proves
    # nothing about ordering.
    for handle, expect_t in (("zmax", 9.0), ("zmin", 11.0)):
        axis_hit = (geom.bound(handle) - origin[2]) / direction[2]
        assert axis_hit == pytest.approx(expect_t)

    assert ray_hit(geom, origin, direction) == ("zmax", pytest.approx(9.0))


def test_an_edge_on_face_is_not_a_target():
    """A face parallel to the ray is a zero-pixel target; admitting it
    would make the winner depend on noise in the denominator."""
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    # Straight down -Z: the four side faces are exactly edge-on.
    handle, _t = ray_hit(geom, (0.0, 0.0, 10.0), (0.0, 0.0, -1.0))
    assert handle == "zmax"
    handle, _t = ray_hit(geom, (10.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert handle == "xmax"


def test_a_tie_breaks_by_declared_handle_order():
    """A ray straight at a corner hits three faces at the same distance.
    Whichever one wins, it must be the same one every time — and it is
    the earliest in ``FACE_HANDLES``."""
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    origin = (10.0, 10.0, 10.0)
    direction = tuple(-np.ones(3) / math.sqrt(3.0))
    first = ray_hit(geom, origin, direction)
    assert first is not None
    assert first[0] == "xmax"                 # before ymax, before zmax
    for _ in range(5):
        assert ray_hit(geom, origin, direction) == first


def test_a_degenerate_box_is_still_grabbable():
    """Collapsed on two axes there is no face with any area left. The
    grab floor is what keeps that from being a dead end the user can
    only escape through the panel."""
    flat = _box(lo=(0.0, 0.0, -1.0), hi=(0.0, 0.0, 1.0))   # a Z segment
    geom = scope_gizmo_geometry(_GID, flat, _BBOX)
    assert geom.grab_floor > 0.0
    result = ray_hit(geom, (10.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    # The X faces are coincident, so both are at the same distance and
    # the declared order decides — deterministically, every time.
    assert result is not None and result[0] == "xmin"


# =====================================================================
# S-MISS / install — the priority-12 abort discipline
# =====================================================================

def test_a_miss_does_not_abort():
    backend, _s, _r, _it, _f = _rig()
    assert backend.iren.send("LeftButtonPressEvent", (5.0, 5.0)) is False


def test_a_press_on_a_face_aborts_and_starts_a_drag():
    backend, _s, renderer, interactor, _f = _rig()
    pixel = backend.pixel_for(_geom(renderer).face_centre("zmax"))
    assert interactor.hit(*pixel)[1] == "zmax"
    assert backend.iren.send("LeftButtonPressEvent", pixel) is True
    assert interactor.is_dragging


def test_shift_lmb_over_a_face_is_left_to_navigation():
    backend, _s, renderer, interactor, _f = _rig()
    pixel = backend.pixel_for(_geom(renderer).face_centre("zmax"))
    assert backend.iren.send(
        "LeftButtonPressEvent", pixel, shift=True,
    ) is False
    assert not interactor.is_dragging


def test_installed_above_navigation_and_picking():
    backend, _s, _r, _it, _f = _rig()
    assert {p for _e, _cb, p in backend.iren.observers.values()} == {12.0}
    assert {e for e, _cb, _p in backend.iren.observers.values()} == {
        "LeftButtonPressEvent", "MouseMoveEvent",
        "LeftButtonReleaseEvent", "LeaveEvent",
    }


def test_uninstall_removes_every_observer():
    backend, _s, _r, interactor, _f = _rig()
    interactor.uninstall()
    assert backend.iren.observers == {}


def test_a_hidden_gizmo_is_not_hit():
    backend, scopes, renderer, interactor, _f = _rig()
    pixel = backend.pixel_for(_geom(renderer).face_centre("zmax"))
    scopes.inner.set_show_gizmo(False)
    renderer.refresh()
    assert interactor.hit(*pixel) == (None, None)
    assert backend.iren.send("LeftButtonPressEvent", pixel) is False


# =====================================================================
# S-DRAG — the face follows the cursor along its own axis
# =====================================================================

def test_dragging_a_face_moves_that_one_bound():
    backend, scopes, renderer, _it, _f = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.4])),
    )
    backend.iren.send("LeftButtonReleaseEvent")

    moved = scopes.box_for(_GID)
    assert float(moved.max[2]) == pytest.approx(1.4)
    # Nothing else budged.
    assert list(moved.min) == pytest.approx([-1.0, -1.0, -1.0])
    assert list(moved.max[:2]) == pytest.approx([1.0, 1.0])


def test_dragging_a_min_face_moves_it_the_same_way_the_cursor_goes():
    """The drag axis is the POSITIVE world axis for both faces of a
    pair — a min face that ran backwards would be unusable."""
    # Look UP at the box, so zmin is the face nearest the eye.
    backend, scopes, renderer, _it, _f = _rig(direction=(-0.4, -0.7, 1.0))
    start = np.asarray(_geom(renderer).face_centre("zmin"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.3])),
    )
    assert float(scopes.box_for(_GID).min[2]) == pytest.approx(-0.7)


def test_leaving_the_window_ends_a_drag():
    """A release delivered outside the window never reaches us, and a
    drag with no end follows the cursor forever afterwards."""
    backend, scopes, renderer, interactor, _f = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.2])),
    )
    assert interactor.is_dragging
    backend.iren.send("LeaveEvent")
    assert not interactor.is_dragging

    settled = float(scopes.box_for(_GID).max[2])
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, 0.9])),
    )
    assert float(scopes.box_for(_GID).max[2]) == pytest.approx(settled)


# =====================================================================
# S-CLAMP — a drag must never construct an invalid BBox
# =====================================================================

def test_a_face_dragged_past_its_opposite_stops_there():
    """``BBox`` RAISES on min > max, so an unclamped drag throws out of
    a Qt handler the moment a face crosses. It must stop instead — and
    the box it stops at is valid and zero-extent on that axis."""
    backend, scopes, renderer, _it, _f = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    # Shove it a long way below zmin (-1.0).
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, -6.0])),
    )

    box = scopes.box_for(_GID)
    assert isinstance(box, BBox)                    # constructed at all
    assert bool(np.all(box.max >= box.min))         # …and valid
    assert float(box.max[2]) == pytest.approx(-1.0)  # stopped, not crossed
    assert float(box.min[2]) == pytest.approx(-1.0)  # no swap


def test_the_clamp_is_a_stop_not_a_swap():
    geom = scope_gizmo_geometry(_GID, _box(), _BBOX)
    assert clamp_bound(geom, "zmax", -9.0) == pytest.approx(-1.0)
    assert clamp_bound(geom, "zmin", +9.0) == pytest.approx(+1.0)
    # Inside the box the value passes through untouched.
    assert clamp_bound(geom, "zmax", 0.5) == pytest.approx(0.5)
    squashed = box_with_bound(geom, "xmax", -50.0)
    assert float(squashed.max[0]) == pytest.approx(-1.0)


@pytest.mark.parametrize("reopen", (+0.5, -0.5))
def test_a_squashed_face_can_be_dragged_back_out(reopen):
    """The zero-extent state is honest (an empty scope hides
    everything) but it must not be a trap.

    Both directions, on purpose: at coincidence the two faces are the
    same pixels, so the hit test's tie-break names one of them
    arbitrarily. If the drag honoured that name the box would only
    reopen one way — grab it, push the other way, and the clamp pins it
    (see ``resolve_side``).
    """
    backend, scopes, renderer, _it, _f = _rig()
    start = np.asarray(_geom(renderer).face_centre("zmax"))
    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(start + np.array([0.0, 0.0, -6.0])),
    )
    backend.iren.send("LeftButtonReleaseEvent")
    renderer.refresh()
    assert float(scopes.box_for(_GID).max[2]) == pytest.approx(-1.0)

    # The collapsed box is still on the hit-test surface, and grabbing
    # the coincident face reopens it — whichever way you pull.
    geom = _geom(renderer)
    face = np.asarray(geom.face_centre("zmax"))
    assert backend.iren.send(
        "LeftButtonPressEvent", backend.pixel_for(face),
    ) is True
    backend.iren.send(
        "MouseMoveEvent",
        backend.pixel_for(face + np.array([0.0, 0.0, reopen])),
    )
    box = scopes.box_for(_GID)
    assert float(box.max[2]) - float(box.min[2]) == pytest.approx(
        abs(reopen), abs=1e-6,
    )


# =====================================================================
# C1 — the drag's gain near face-on viewing
# =====================================================================

def test_the_drag_path_passes_the_separation_floor():
    """The bare projection divides by ``1 - (n.d)^2``; the floor is the
    only thing standing between a face-on camera and a gesture that
    throws the bound off the model."""
    centre = np.array([0.0, 0.0, 1.0])
    axis = np.array([0.0, 0.0, 1.0])
    pixel = 0.01

    def delta_per_pixel(deg: float, floored: bool) -> float:
        a = math.radians(deg)
        eye = np.array([math.sin(a), 0.0, math.cos(a)]) * 30.0
        d = -eye / np.linalg.norm(eye)
        right = np.array([math.cos(a), 0.0, -math.sin(a)])
        kw = {"min_separation": DRAG_MIN_SEPARATION} if floored else {}
        s0 = axis_param(centre, axis, eye, d, **kw)
        s1 = axis_param(centre, axis, eye + right * pixel, d, **kw)
        return abs(s1 - s0)

    for deg in (90.0, 30.0, 5.0, 1.0, 0.1):
        assert delta_per_pixel(deg, True) < 0.05, f"{deg} deg runs away"
    # Unfloored, the same one-pixel move at 0.1 degrees is off the map.
    assert delta_per_pixel(0.1, False) > 1.0


def test_a_face_on_drag_does_not_fling_the_bound():
    """C1 end to end, through the real interactor — this is the one
    that fails if the drag path stops passing the floor. Looking ~1
    degree off a face's own normal is one click of an axis-view snap."""
    tilt = math.sin(math.radians(1.0))
    backend, scopes, renderer, interactor, _f = _rig(
        direction=(tilt, 0.0, -1.0),
    )
    geom = _geom(renderer)
    start = backend.pixel_for(geom.face_centre("zmax"))
    assert interactor.hit(*start) == (_GID, "zmax"), (
        "this test is only meaningful while the press grabs zmax"
    )
    assert backend.iren.send("LeftButtonPressEvent", start) is True, (
        "a press on a face must be claimed even face-on — bailing here "
        "hands the click to the picker"
    )
    backend.iren.send("MouseMoveEvent", (start[0] + 1.0, start[1]))
    moved = abs(float(scopes.box_for(_GID).max[2]) - 1.0)
    # The model spans 2 units.
    assert moved < 0.1, f"one pixel moved the bound {moved:.3f} units"


# =====================================================================
# S-OWNER — one write per move, and no actor churn
# =====================================================================

def test_a_drag_issues_one_set_scope_per_move_and_no_rebuilds():
    backend, scopes, renderer, _it, fired = _rig()
    start = np.asarray(_geom(renderer).face_centre("xmax"))

    backend.iren.send("LeftButtonPressEvent", backend.pixel_for(start))
    assert scopes.set_calls == [], "a press alone mutates nothing"

    backend.calls.clear()
    for i in range(1, 21):
        backend.iren.send(
            "MouseMoveEvent",
            backend.pixel_for(start + np.array([i * 0.02, 0.0, 0.0])),
        )
        renderer.refresh()          # what the RENDER-lane subscriber does
    backend.iren.send("LeftButtonReleaseEvent")

    assert len(scopes.set_calls) == 20
    assert fired == [_GID] * 20
    # The interactor's only backend traffic is the pick ray; the layers
    # are reused in place, so a drag adds and removes nothing.
    kinds = [c[0] for c in backend.calls]
    assert set(kinds) == {"ray", "update"}
    assert "add" not in kinds and "remove" not in kinds
    assert kinds.count("update") == 40          # faces + edges, 20 moves


# =====================================================================
# S-RENDER — the layers, and who they belong to
# =====================================================================

def _layer_ids(backend):
    return {
        lid for lid in backend.layers
        if lid.startswith(SCOPE_GIZMO_LAYER_PREFIX)
    }


def test_layers_appear_for_the_active_geometry_only():
    backend = _ScopeBackend()
    scopes = _CountingScopes()
    scopes.set_scope("g1", _box())
    scopes.set_scope("g2", _box(lo=(0.0, 0.0, 0.0), hi=(2.0, 2.0, 2.0)))
    active = ["g1"]
    renderer = ScopeGizmoRenderer(
        backend, scopes, bbox=_BBOX, active_id=lambda: active[0],
    )
    renderer.refresh()
    assert _layer_ids(backend) == {
        f"{SCOPE_GIZMO_LAYER_PREFIX}g1:faces",
        f"{SCOPE_GIZMO_LAYER_PREFIX}g1:edges",
    }
    assert set(renderer.geometries()) == {"g1"}

    active[0] = "g2"
    renderer.refresh()
    assert _layer_ids(backend) == {
        f"{SCOPE_GIZMO_LAYER_PREFIX}g2:faces",
        f"{SCOPE_GIZMO_LAYER_PREFIX}g2:edges",
    }


def test_no_scope_and_no_gizmo_mean_no_layers():
    backend, scopes, renderer, _it, _f = _rig()
    assert _layer_ids(backend)

    scopes.inner.set_show_gizmo(False)
    renderer.refresh()
    assert _layer_ids(backend) == set()
    assert renderer.geometries() == {}

    scopes.inner.set_show_gizmo(True)
    renderer.refresh()
    assert _layer_ids(backend)

    scopes.inner.clear_scope(_GID)
    renderer.refresh()
    assert _layer_ids(backend) == set()


def test_gizmo_layers_declare_the_clip_exemption_and_are_unpickable():
    backend, _s, _r, _it, _f = _rig()
    layers = [backend.layers[lid] for lid in _layer_ids(backend)]
    assert len(layers) == 2
    for layer in layers:
        assert layer.clip_exempt is True
        assert layer.pickable is False


def test_the_drawn_box_is_the_box():
    """The corners the renderer paints are the controller's own bounds
    — the gizmo must not draw a box that filters differently."""
    box = _box(lo=(-0.5, 0.0, 0.25), hi=(1.5, 2.0, 3.0))
    backend, _s, _r, _it, _f = _rig(box=box)
    faces = backend.layers[f"{SCOPE_GIZMO_LAYER_PREFIX}{_GID}:faces"]
    pts = np.asarray(faces.points.coords)
    assert pts.shape == (8, 3)
    assert pts.min(axis=0) == pytest.approx(list(box.min))
    assert pts.max(axis=0) == pytest.approx(list(box.max))


def test_clear_takes_every_layer_down():
    backend, _s, renderer, _it, _f = _rig()
    renderer.clear()
    assert _layer_ids(backend) == set()
    renderer.clear()                                   # idempotent
    assert renderer.geometries() == {}


# =====================================================================
# C2 — the window cursor now has THREE would-be owners
# =====================================================================

def test_a_scope_hover_routes_through_the_arbiter_and_yields():
    """ADR 0083 S2 finding C2, with one more hoverer.

    Three priority-12 interactors run on every move and none of them
    aborts on a miss — that is the design — so a direct
    ``SetCurrentCursor`` here would erase whatever the legend or the
    clip gizmo had just asked for. The scope gizmo requests instead,
    and it is declared LAST, matching the order that decides the press.
    """
    import vtk

    from apeGmsh.viewers.core._cursor_arbiter import request_cursor
    from apeGmsh.viewers.core._scope_gizmo_interactor import (
        ScopeGizmoInteractor,
    )

    backend = _ScopeBackend()
    window = backend.render_window
    scope = ScopeGizmoInteractor(backend, None, None)

    # Hovering a face asks the ARBITER — not the window — for a cursor.
    scope._set_cursor("zmax")
    assert window.cursor == vtk.VTK_CURSOR_SIZEALL

    # Slide onto a legend: the legend asks (it runs first), then the
    # scope gizmo misses and withdraws (it runs last). The legend keeps
    # what it asked for.
    request_cursor(window, "legend", vtk.VTK_CURSOR_SIZENE)
    scope._set_cursor(None)
    assert window.cursor == vtk.VTK_CURSOR_SIZENE, (
        "the scope gizmo's miss erased the legend's hover cursor"
    )

    # And it never outranks a live clip-gizmo request either, whichever
    # order the two arrive in.
    request_cursor(window, "legend", None)
    scope._set_cursor("xmin")
    request_cursor(window, "clip_gizmo", vtk.VTK_CURSOR_HAND)
    assert window.cursor == vtk.VTK_CURSOR_HAND

    request_cursor(window, "clip_gizmo", None)
    scope._set_cursor(None)
    assert window.cursor == vtk.VTK_CURSOR_DEFAULT


# =====================================================================
# Real window (qt lane — run per file in a fresh process)
# =====================================================================
#
# The headless block above proves the math and the observer discipline
# against a fake camera. What it cannot prove is the WIRING: that a real
# VTK press lands on our observer, that the box moves the way the pixels
# say it should through the real renderer's projection, that the panel's
# six spin boxes follow — and that they are still typable afterwards
# (ADR 0083 B3's measured class: a UI-lane refresh eating keystrokes).

# The qt-lane fixture, imported rather than re-typed: a meshed cube
# written to real HDF5.
from tests.viewers.test_threshold import cube_results  # noqa: E402,F401


def _need_qt():
    pytest.importorskip("pytestqt", reason="needs pytest-qt")
    pytest.importorskip("pyvistaqt")
    qtw = pytest.importorskip("qtpy.QtWidgets")
    qtw.QApplication.instance() or qtw.QApplication([])


def _to_pixel(plotter, point) -> "tuple[float, float]":
    """Where ``point`` lands on screen, per the RENDERER's own camera.

    Ground truth for "the user put the cursor there" — and pointedly
    not the interactor's own unprojection, so the assertion below is
    not the implementation checking itself.
    """
    renderer = plotter.renderer
    renderer.SetWorldPoint(
        float(point[0]), float(point[1]), float(point[2]), 1.0,
    )
    renderer.WorldToDisplay()
    x, y, _z = renderer.GetDisplayPoint()
    return float(x), float(y)


@pytest.mark.qt
def test_a_real_drag_moves_the_box_and_the_panel_follows(cube_results):
    """Press, drag, release — in a real window, on a real interactor.

    The drag target is a world point a quarter of the box up the +Z
    axis from the zmax face; projecting it with the renderer's camera
    gives the pixel the user would have to move to, so the bound must
    land on that point. Then the panel's Max-Z spin box must agree.
    """
    _need_qt()
    from qtpy import QtCore, QtWidgets

    from apeGmsh.viewers.diagrams._dispatch import GEOMETRY_SCOPE_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-gizmo-drag",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geom = director.geometries.active
            scene = director.scene_for(geom)
            plotter = viewer._plotter
            pts = np.asarray(scene.reference_points)
            box = BBox(pts.min(axis=0), pts.max(axis=0))
            director.scopes.set_scope(geom.id, box)
            director.dispatcher.fire(
                GEOMETRY_SCOPE_CHANGED, payload=geom.id,
            )
            viewer._geometry_panel.show_geometry(geom.id)
            QtWidgets.QApplication.processEvents()

            # An oblique camera: face-on to the handle would saturate
            # the gesture by design (DRAG_MIN_SEPARATION) and prove
            # nothing about direction.
            span = float(box.max[2] - box.min[2])
            centre = box.center
            plotter.camera_position = [
                tuple(centre + np.array([6.0, 5.0, 4.0]) * span),
                tuple(centre),
                (0.0, 0.0, 1.0),
            ]
            plotter.render()

            interactor = viewer._scope_gizmo_interactor
            seen["installed"] = interactor is not None
            gizmo = viewer._scope_gizmos.geometries()[geom.id]
            face = np.asarray(gizmo.face_centre("zmax"))
            delta = 0.25 * span
            px0 = _to_pixel(plotter, face)
            px1 = _to_pixel(plotter, face + np.array([0.0, 0.0, delta]))
            seen["hit"] = interactor.hit(*px0)
            seen["expected"] = float(box.max[2]) + delta
            seen["span"] = span

            iren = plotter.iren.interactor
            for event, pixel in (
                ("LeftButtonPressEvent", px0),
                ("MouseMoveEvent", px1),
                ("LeftButtonReleaseEvent", px1),
            ):
                iren.SetEventPosition(
                    int(round(pixel[0])), int(round(pixel[1])),
                )
                iren.InvokeEvent(event)

            moved = director.scopes.box_for(geom.id)
            seen["moved_max"] = float(moved.max[2])
            seen["moved_min"] = list(moved.min)
            seen["moved_max_xy"] = list(moved.max[:2])
            seen["start_min"] = list(box.min)
            seen["start_max_xy"] = list(box.max[:2])
            seen["start_max_z"] = float(box.max[2])
            seen["dragging_after_release"] = interactor.is_dragging

            # The panel is a subscriber on the UI lane; let it run.
            QtWidgets.QApplication.processEvents()
            seen["panel_max_z"] = float(
                viewer._geometry_panel._sb_scope_max[2].value()
            )
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(600, _drive_then_close)
    viewer.show()

    assert seen.get("installed") is True
    assert seen.get("hit", (None, None))[1] == "zmax", (
        "the test only means something while the press grabs zmax"
    )
    # A pixel of rounding at each end; 8 % of the box is ample slack.
    tol = 0.08 * seen["span"]
    assert seen["moved_max"] == pytest.approx(seen["expected"], abs=tol)
    assert seen["moved_max"] > seen["start_max_z"], (
        "the cursor went towards +Z; the bound must too"
    )
    # One bound moved. The other five did not.
    assert seen["moved_min"] == pytest.approx(seen["start_min"])
    assert seen["moved_max_xy"] == pytest.approx(seen["start_max_xy"])
    assert seen["dragging_after_release"] is False
    # …and the panel says what the controller says.
    assert seen["panel_max_z"] == pytest.approx(seen["moved_max"], abs=1e-6)


@pytest.mark.qt
def test_typing_into_a_scope_field_survives_a_scope_refresh(cube_results):
    """Hazard: the drag gives the six bounds a SECOND editor.

    ADR 0083 B3 measured this exact shape on the clip panel — a UI-lane
    refresh landing between keystrokes rewrote the field being typed
    into, so ``2.5`` committed as ``2`` and a custom value could not be
    entered at all. Here the refresh arrives on every drag tick.

    The value typed is deliberately one the owner will REFUSE (it
    inverts the box), so the controller keeps its old number and a
    refresh that ignored focus would visibly stomp what was typed.
    """
    _need_qt()
    from qtpy import QtCore, QtWidgets
    from qtpy.QtTest import QTest

    from apeGmsh.viewers.diagrams._dispatch import GEOMETRY_SCOPE_CHANGED
    from apeGmsh.viewers.results_viewer import ResultsViewer

    viewer = ResultsViewer(
        cube_results, title="scope-gizmo-typing",
        restore_session=False, save_session=False,
    )
    seen: dict = {}

    def _drive_then_close():
        try:
            director = viewer._director
            geom = director.geometries.active
            scene = director.scene_for(geom)
            pts = np.asarray(scene.reference_points)
            box = BBox(pts.min(axis=0), pts.max(axis=0))
            director.scopes.set_scope(geom.id, box)
            director.dispatcher.fire(
                GEOMETRY_SCOPE_CHANGED, payload=geom.id,
            )
            panel = viewer._geometry_panel
            panel.show_geometry(geom.id)
            QtWidgets.QApplication.processEvents()
            seen["before"] = float(panel._sb_scope_max[2].value())

            sb = panel._sb_scope_min[2]
            sb.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            QtWidgets.QApplication.processEvents()
            seen["focused"] = bool(sb.hasFocus())

            # Type a Min Z above Max Z: the box inverts, so
            # ``_push_scope`` refuses it and the OWNER keeps its bounds.
            sb.selectAll()
            QTest.keyClicks(sb, "99")
            QtWidgets.QApplication.processEvents()
            seen["owner_min_z"] = float(
                director.scopes.box_for(geom.id).min[2]
            )

            # Now the drag's announcement arrives, twice, exactly as a
            # moving mouse would deliver it.
            for _ in range(2):
                director.dispatcher.fire(
                    GEOMETRY_SCOPE_CHANGED, payload=geom.id,
                )
                QtWidgets.QApplication.processEvents()
            seen["typed_still_there"] = float(sb.value())
            seen["still_focused"] = bool(sb.hasFocus())

            # An UNfocused field, by contrast, must follow the owner —
            # that is the whole point of the subscription.
            other = panel._sb_scope_max[2]
            other.blockSignals(True)
            other.setValue(-12345.0)
            other.blockSignals(False)
            director.dispatcher.fire(
                GEOMETRY_SCOPE_CHANGED, payload=geom.id,
            )
            QtWidgets.QApplication.processEvents()
            seen["unfocused_followed"] = float(other.value())
        finally:
            viewer._win.window.close()

    QtCore.QTimer.singleShot(600, _drive_then_close)
    viewer.show()

    assert seen.get("focused") is True, (
        "the guard is only meaningful while the editor really has focus"
    )
    # The owner refused the inverted box — so a refresh HAS something
    # different to say, and a stomp would be visible.
    assert seen["owner_min_z"] != pytest.approx(99.0)
    assert seen["typed_still_there"] == pytest.approx(99.0), (
        "the refresh ate the keystrokes (ADR 0083 B3)"
    )
    assert seen["still_focused"] is True
    assert seen["unfocused_followed"] == pytest.approx(
        seen["before"], abs=1e-6,
    )
